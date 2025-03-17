// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Package huggingface implements a client for the HuggingFace serverless
// inference API.
//
// It is described at https://huggingface.co/docs/api-inference/
package huggingface

import (
	"bufio"
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/invopop/jsonschema"
	"github.com/maruel/genai/genaiapi"
	"github.com/maruel/genai/internal"
	"github.com/maruel/httpjson"
)

// https://huggingface.co/docs/api-inference/tasks/chat-completion#api-specification
type CompletionRequest struct {
	// Model            string    `json:"model"` It's already in the URL.
	Stream           bool      `json:"stream"`
	Messages         []Message `json:"messages"`
	FrequencyPenalty float64   `json:"frequency_penalty,omitzero"` // [-2.0, 2.0]
	Logprobs         bool      `json:"logprobs,omitzero"`
	MaxTokens        int64     `json:"max_tokens,omitzero"`
	PresencePenalty  float64   `json:"presence_penalty,omitzero"` // [-2.0, 2.0]
	ResponseFormat   struct {
		Type string `json:"type"` // "json", "regex"
		// Type == "regexp": a regex string.
		// Type == "json": a JSONSchema.
		Value *jsonschema.Schema `json:"value"`
	} `json:"response_format,omitzero"`
	Seed          int64    `json:"seed,omitzero"`
	Stop          []string `json:"stop,omitzero"`
	StreamOptions struct {
		IncludeUsage bool `json:"include_usage,omitzero"`
	} `json:"stream_options,omitzero"`
	Temperature float64 `json:"temperature,omitzero"` // [0, 2.0]
	// Alternative when forcing a specific function. This can probably be achieved
	// by providing a single tool and ToolChoice == "required".
	// ToolChoice struct {
	// 	Type     string `json:"type,omitzero"` // "function"
	// 	Function struct {
	// 		Name string `json:"name,omitzero"`
	// 	} `json:"function,omitzero"`
	// } `json:"tool_choice,omitzero"`
	ToolChoice  string  `json:"tool_choice,omitzero"` // "auto", "none", "required"
	ToolPrompt  string  `json:"tool_prompt,omitzero"`
	Tools       []Tool  `json:"tools,omitzero"`
	TopLogprobs int64   `json:"top_logprobs,omitzero"`
	TopP        float64 `json:"top_p,omitzero"` // [0, 1]
}

func (c *CompletionRequest) Init(msgs []genaiapi.Message, opts genaiapi.Validatable) error {
	var errs []error
	sp := ""
	if opts != nil {
		if err := opts.Validate(); err != nil {
			errs = append(errs, err)
		} else {
			switch v := opts.(type) {
			case *genaiapi.CompletionOptions:
				c.MaxTokens = v.MaxTokens
				c.Temperature = v.Temperature
				c.TopP = v.TopP
				sp = v.SystemPrompt
				c.Seed = v.Seed
				if v.TopK != 0 {
					errs = append(errs, errors.New("huggingface does not support TopK"))
				}
				c.Stop = v.Stop
				if v.ReplyAsJSON || v.DecodeAs != nil {
					errs = append(errs, errors.New("hugginface client doesn't support JSON yet; to be implemented"))
				}
				if v.ReplyAsJSON {
					c.ResponseFormat.Type = "json"
				}
				if v.DecodeAs != nil {
					c.ResponseFormat.Type = "json"
					c.ResponseFormat.Value = jsonschema.Reflect(v.DecodeAs)
				}
				if len(v.Tools) != 0 {
					// Let's assume if the user provides tools, they want to use them.
					c.ToolChoice = "required"
					c.Tools = make([]Tool, len(v.Tools))
					for i, t := range v.Tools {
						c.Tools[i].Type = "function"
						c.Tools[i].Function.Name = t.Name
						c.Tools[i].Function.Description = t.Description
						if t.InputsAs != nil {
							c.Tools[i].Function.Arguments = jsonschema.Reflect(t.InputsAs)
						}
					}
				}
			default:
				errs = append(errs, fmt.Errorf("unsupported options type %T", opts))
			}
		}
	}

	if err := genaiapi.ValidateMessages(msgs); err != nil {
		errs = append(errs, err)
	} else {
		offset := 0
		if sp != "" {
			offset = 1
		}
		c.Messages = make([]Message, len(msgs)+offset)
		if sp != "" {
			c.Messages[0].Role = "system"
			c.Messages[0].Content = []Content{{Type: "text", Text: sp}}
		}
		for i := range msgs {
			if err := c.Messages[i+offset].From(&msgs[i]); err != nil {
				errs = append(errs, fmt.Errorf("message %d: %w", i, err))
			}
		}
	}
	return errors.Join(errs...)
}

type Message struct {
	Role      string    `json:"role"`
	Content   []Content `json:"content,omitzero"`
	ToolCalls []struct {
		ID       string `json:"id,omitzero"`
		Type     string `json:"type,omitzero"` // "function"
		Function struct {
			Name      string `json:"name,omitzero"`
			Arguments string `json:"arguments,omitzero"`
		} `json:"function,omitzero"`
	} `json:"tool_calls,omitzero"`
}

func (msg *Message) From(m *genaiapi.Message) error {
	// We don't filter the role here.
	msg.Role = string(m.Role)
	msg.Content = []Content{{}}
	switch m.Type {
	case genaiapi.Text:
		msg.Content[0].Type = "text"
		msg.Content[0].Text = m.Text
	case genaiapi.Document:
		mimeType, data, err := internal.ParseDocument(m, 10*1024*1024)
		if err != nil {
			return err
		}
		switch {
		case (m.URL != "" && mimeType == "") || strings.HasPrefix(mimeType, "image/"):
			msg.Content[0].Type = "image_url"
			if m.URL == "" {
				msg.Content[0].ImageURL.URL = fmt.Sprintf("data:%s;base64,%s", mimeType, base64.StdEncoding.EncodeToString(data))
			} else {
				msg.Content[0].ImageURL.URL = m.URL
			}
		default:
			return fmt.Errorf("unsupported mime type %s", mimeType)
		}
	default:
		return fmt.Errorf("unsupported content type %s", m.Type)
	}
	return nil
}

type Content struct {
	Type     string `json:"type"` // text,image_url
	Text     string `json:"text,omitzero"`
	ImageURL struct {
		URL string `json:"url,omitzero"`
	} `json:"image_url,omitzero"`
}

type Tool struct {
	Type     string `json:"type,omitzero"` // "function"
	Function struct {
		Name        string             `json:"name,omitzero"`
		Description string             `json:"description,omitzero"`
		Arguments   *jsonschema.Schema `json:"arguments,omitzero"`
	} `json:"function,omitzero"`
}

type CompletionResponse struct {
	Object            string `json:"object"`
	ID                string `json:"id"`
	Created           Time   `json:"created"`
	Model             string `json:"model"`
	SystemFingerprint string `json:"system_fingerprint"`

	Choices []struct {
		FinishReason string `json:"finish_reason"`
		Index        int64  `json:"index"`
		Message      struct {
			Role       string `json:"role"`
			Content    string `json:"content"`
			ToolCallID string `json:"tool_call_id"`
			ToolCalls  []struct {
				ID       string `json:"id"`
				Type     string `json:"type"` // function
				Function struct {
					Name        string   `json:"name"`
					Description struct{} `json:"description"` // Passed in as null
					Arguments   any      `json:"arguments"`
				} `json:"function"`
			} `json:"tool_calls"`
		} `json:"message"`
		Logprobs struct {
			Content []struct {
				Logprob     float64 `json:"logprob"`
				Token       string  `json:"token"`
				TopLogprobs []struct {
					Token   string  `json:"token"`
					Logprob float64 `json:"logprob"`
				} `json:"top_logprobs"`
			} `json:"content"`
		} `json:"logprobs"`
	} `json:"choices"`
	Usage struct {
		PromptTokens     int64 `json:"prompt_tokens"`
		CompletionTokens int64 `json:"completion_tokens"`
		TotalTokens      int64 `json:"total_tokens"`
	} `json:"usage"`
}

func (c *CompletionResponse) ToResult() (genaiapi.CompletionResult, error) {
	out := genaiapi.CompletionResult{}
	out.InputTokens = c.Usage.PromptTokens
	out.OutputTokens = c.Usage.CompletionTokens
	if len(c.Choices) != 1 {
		return out, fmt.Errorf("server returned an unexpected number of choices, expected 1, got %d", len(c.Choices))
	}
	if len(c.Choices[0].Message.ToolCalls) != 0 {
		out.Type = genaiapi.ToolCalls
		out.ToolCalls = make([]genaiapi.ToolCall, len(c.Choices[0].Message.ToolCalls))
		for i, t := range c.Choices[0].Message.ToolCalls {
			out.ToolCalls[i].ID = t.ID
			out.ToolCalls[i].Name = t.Function.Name
			b, err := json.Marshal(t.Function.Arguments)
			if err != nil {
				return out, fmt.Errorf("failed to marshal arguments: %w", err)
			}
			out.ToolCalls[i].Arguments = string(b)
		}
	} else {
		out.Type = genaiapi.Text
		out.Text = c.Choices[0].Message.Content
	}
	switch role := c.Choices[0].Message.Role; role {
	case "system", "assistant", "user":
		out.Role = genaiapi.Role(role)
	default:
		return out, fmt.Errorf("unsupported role %q", role)
	}
	return out, nil
}

type CompletionStreamChunkResponse struct {
	Object            string `json:"object"`
	Created           Time   `json:"created"`
	ID                string `json:"id"`
	Model             string `json:"model"`
	SystemFingerprint string `json:"system_fingerprint"`
	Choices           []struct {
		Index        int64  `json:"index"`
		FinishReason string `json:"finish_reason"` // stop
		Delta        struct {
			Role       genaiapi.Role `json:"role"`
			Content    string        `json:"content"`
			ToolCallID string        `json:"tool_call_id"`
			ToolCalls  []struct {
				ID       string `json:"id"`
				Type     string `json:"type"` // function
				Function struct {
					Name        string   `json:"name"`
					Description struct{} `json:"description"`
					Arguments   any      `json:"arguments"`
				} `json:"function"`
			} `json:"tool_calls"`
		} `json:"delta"`
		Logprobs struct {
			Content []struct {
				Logprob     float64 `json:"logprob"`
				Token       string  `json:"token"`
				TopLogprobs []struct {
					Token   string  `json:"token"`
					Logprob float64 `json:"logprob"`
				} `json:"top_logprobs"`
			} `json:"content"`
		} `json:"logprobs"`
	} `json:"choices"`
	Usage struct {
		PromptTokens     int64 `json:"prompt_tokens"`
		CompletionTokens int64 `json:"completion_tokens"`
		TotalTokens      int64 `json:"total_tokens"`
	} `json:"usage"`
}

type errorResponse struct {
	Error string `json:"error"`
}

// Client implements the REST JSON based API.
type Client struct {
	apiKey string
	model  string
}

// TODO: Investigate https://huggingface.co/blog/inference-providers and https://huggingface.co/docs/inference-endpoints/

// New creates a new client to talk to the HuggingFace serverless inference API.
//
// If apiKey is not provided, it tries to load it from the HUGGINGFACE_API_KEY environment variable.
// Otherwise, it tries to load it from the huggingface python client's cache.
// If none is found, it returns an error.
// Get your API key at https://huggingface.co/settings/tokens
// If no model is provided, only functions that do not require a model, like ListModels, will work.
// To use multiple models, create multiple clients.
// Use one of the tens of thousands of models to chose from at https://huggingface.co/models?inference=warm&sort=trending
func New(apiKey, model string) (*Client, error) {
	if apiKey == "" {
		if apiKey = os.Getenv("HUGGINGFACE_API_KEY"); apiKey == "" {
			// Fallback to loading from the python client's cache.
			h, err := os.UserHomeDir()
			if err != nil {
				return nil, fmt.Errorf("can't find home directory; failed to load hugginface key; get one at %s: %w", apiKeyURL, err)
			}
			// TODO: Windows.
			b, err := os.ReadFile(filepath.Join(h, ".cache", "huggingface", "token"))
			if err != nil {
				return nil, fmt.Errorf("no cached token file; failed to load hugginface key; get one at %s: %w", apiKeyURL, err)
			}
			if apiKey = strings.TrimSpace(string(b)); apiKey == "" {
				return nil, errors.New("token file exist but is empty; huggingface API key is required; get one at " + apiKeyURL)
			}
		}
	}
	return &Client{apiKey: apiKey, model: model}, nil
}

func (c *Client) Completion(ctx context.Context, msgs []genaiapi.Message, opts genaiapi.Validatable) (genaiapi.CompletionResult, error) {
	// https://huggingface.co/docs/api-inference/tasks/chat-completion#api-specification
	rpcin := CompletionRequest{}
	if err := rpcin.Init(msgs, opts); err != nil {
		return genaiapi.CompletionResult{}, err
	}
	rpcout := CompletionResponse{}
	if err := c.CompletionRaw(ctx, &rpcin, &rpcout); err != nil {
		return genaiapi.CompletionResult{}, fmt.Errorf("failed to get chat response: %w", err)
	}
	return rpcout.ToResult()
}

func (c *Client) CompletionRaw(ctx context.Context, in *CompletionRequest, out *CompletionResponse) error {
	if err := c.validate(); err != nil {
		return err
	}
	in.Stream = false
	url := "https://router.huggingface.co/hf-inference/models/" + c.model + "/v1/chat/completions"
	return c.post(ctx, url, in, out)
}

func (c *Client) CompletionStream(ctx context.Context, msgs []genaiapi.Message, opts genaiapi.Validatable, chunks chan<- genaiapi.MessageChunk) error {
	in := CompletionRequest{}
	if err := in.Init(msgs, opts); err != nil {
		return err
	}
	ch := make(chan CompletionStreamChunkResponse)
	end := make(chan error)
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()
	go func() {
		var lastRole genaiapi.Role
		for pkt := range ch {
			if len(pkt.Choices) != 1 {
				continue
			}
			switch role := pkt.Choices[0].Delta.Role; role {
			case "system", "assistant", "user":
				lastRole = genaiapi.Role(role)
			case "":
			default:
				cancel()
				// We need to empty the channel to avoid blocking the goroutine.
				for range ch {
				}
				end <- fmt.Errorf("unexpected role %q", role)
				return
			}
			if word := pkt.Choices[0].Delta.Content; word != "" {
				chunks <- genaiapi.MessageChunk{Role: lastRole, Type: genaiapi.Text, Text: word}
			}
		}
		end <- nil
	}()
	err := c.CompletionStreamRaw(ctx, &in, ch)
	close(ch)
	if err2 := <-end; err2 != nil {
		err = err2
	}
	return err
}

func (c *Client) CompletionStreamRaw(ctx context.Context, in *CompletionRequest, out chan<- CompletionStreamChunkResponse) error {
	if err := c.validate(); err != nil {
		return err
	}
	in.Stream = true
	h := make(http.Header)
	h.Add("Authorization", "Bearer "+c.apiKey)
	// HuggingFace support all three of gzip, br and zstd!
	p := httpjson.DefaultClient
	// p.PostCompress = "zstd"
	url := "https://router.huggingface.co/hf-inference/models/" + c.model + "/v1/chat/completions"
	resp, err := p.PostRequest(ctx, url, h, in)
	if err != nil {
		return fmt.Errorf("failed to get server response: %w", err)
	}
	defer resp.Body.Close()
	r := bufio.NewReader(resp.Body)
	for first := true; ; first = false {
		line, err := r.ReadBytes('\n')
		if line = bytes.TrimSpace(line); err == io.EOF {
			if len(line) == 0 {
				return nil
			}
		} else if err != nil {
			return fmt.Errorf("failed to get server response: %w", err)
		}
		if len(line) != 0 {
			// HuggingFace has the bad habit of returning errors as HTML pages.
			if first && bytes.HasPrefix(line, []byte("<!DOCTYPE html>")) {
				// Often has a 503 in there as a <div>.
				rest, _ := io.ReadAll(r)
				return fmt.Errorf("unexpected error: %s\n%s", line, rest)
			}
			if err := parseStreamLine(line, out); err != nil {
				return err
			}
		}
	}
}

func parseStreamLine(line []byte, out chan<- CompletionStreamChunkResponse) error {
	const prefix = "data: "
	if !bytes.HasPrefix(line, []byte(prefix)) {
		return fmt.Errorf("unexpected line. expected \"data: \", got %q", line)
	}
	suffix := string(line[len(prefix):])
	if suffix == "[DONE]" {
		return nil
	}
	d := json.NewDecoder(strings.NewReader(suffix))
	d.DisallowUnknownFields()
	d.UseNumber()
	msg := CompletionStreamChunkResponse{}
	if err := d.Decode(&msg); err != nil {
		return fmt.Errorf("failed to decode server response %q: %w", string(line), err)
	}
	out <- msg
	return nil
}

type Time int64

func (t *Time) AsTime() time.Time {
	return time.Unix(int64(*t), 0)
}

type Model struct {
	ID            string    `json:"id"`
	ID2           string    `json:"_id"`
	Likes         int64     `json:"likes"`
	TrendingScore float64   `json:"trendingScore"`
	Private       bool      `json:"private"`
	Downloads     int64     `json:"downloads"`
	Tags          []string  `json:"tags"` // Tags can be a single word or key:value, like base_model, doi, license, region, arxiv.
	PipelineTag   string    `json:"pipeline_tag"`
	LibraryName   string    `json:"library_name"`
	CreatedAt     time.Time `json:"createdAt"`
	ModelId       string    `json:"modelId"`

	// When full=true is specified:
	Author       string    `json:"author"`
	Gated        bool      `json:"gated"`
	LastModified time.Time `json:"lastModified"`
	SHA          string    `json:"sha"`
	Siblings     []struct {
		RFilename string `json:"r_filename"`
	} `json:"siblings"`
}

func (m *Model) GetID() string {
	return m.ID
}

func (m *Model) String() string {
	return fmt.Sprintf("%s (%s) %s Trending: %.1f", m.ID, m.CreatedAt.Format("2006-01-02"), m.PipelineTag, m.TrendingScore)
}

func (m *Model) Context() int64 {
	return 0
}

func (c *Client) ListModels(ctx context.Context) ([]genaiapi.Model, error) {
	// https://huggingface.co/docs/hub/api

	// return nil, errors.New("not implemented; there's just too many, tens of thousands to chose from at https://huggingface.co/models?inference=warm&sort=trending")
	h := make(http.Header)
	h.Add("Authorization", "Bearer "+c.apiKey)
	var out []Model
	// There's 20k models warm as of March 2025. There's no way to sort by
	// trending. Sorting by download is not useful.
	err := httpjson.DefaultClient.Get(ctx, "https://huggingface.co/api/models?inference=warm", h, &out)
	if err != nil {
		return nil, err
	}
	models := make([]genaiapi.Model, len(out))
	for i := range out {
		models[i] = &out[i]
	}
	return models, err
}

func (c *Client) validate() error {
	if c.model == "" {
		return errors.New("a model is required")
	}
	return nil
}

func (c *Client) post(ctx context.Context, url string, in, out any) error {
	h := make(http.Header)
	h.Add("Authorization", "Bearer "+c.apiKey)
	// HuggingFace support all three of gzip, br and zstd!
	p := httpjson.DefaultClient
	p.PostCompress = "zstd"
	resp, err := p.PostRequest(ctx, url, h, in)
	if err != nil {
		return err
	}
	er := errorResponse{}
	switch i, err := httpjson.DecodeResponse(resp, out, &er); i {
	case 0:
		return nil
	case 1:
		var herr *httpjson.Error
		if errors.As(err, &herr) {
			if herr.StatusCode >= 400 {
				return fmt.Errorf("huggingface: %s; %s", http.StatusText(herr.StatusCode), er.Error)
			}
		}
		return fmt.Errorf("huggingface: %s", er.Error)
	default:
		// HuggingFace rarely return a structured error.
		var herr *httpjson.Error
		if errors.As(err, &herr) {
			// Only include the body if it's not a whole HTML page.
			suffix := ""
			if bytes.HasPrefix(herr.ResponseBody, []byte("{")) {
				suffix = "; " + string(herr.ResponseBody)
			}
			if herr.StatusCode >= 400 {
				return fmt.Errorf("huggingface: %s%s", http.StatusText(herr.StatusCode), suffix)
			}
			// slog.WarnContext(ctx, "huggingface", "url", url, "err", err, "response", string(herr.ResponseBody), "status", herr.StatusCode)
			// Hugginface returns raw unstructured text if it fails decoding. Just
			// return the content. Sometimes it's a web page because why not?
			// return fmt.Errorf("%s: %w", string(herr.ResponseBody), err)
			return fmt.Errorf("%w%s", err, suffix)
		}
		return err
	}
}

const apiKeyURL = "https://huggingface.co/settings/tokens"

var (
	_ genaiapi.CompletionProvider = &Client{}
	_ genaiapi.ModelProvider      = &Client{}
)
