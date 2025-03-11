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
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/maruel/genai/genaiapi"
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
		Value JSONSchema `json:"value"`
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

func (c *CompletionRequest) fromOpts(opts any) error {
	switch v := opts.(type) {
	case *genaiapi.CompletionOptions:
		c.MaxTokens = v.MaxTokens
		c.Seed = v.Seed
		c.Temperature = v.Temperature
	default:
		return fmt.Errorf("unsupported options type %T", opts)
	}
	return nil
}

func (c *CompletionRequest) fromMsgs(msgs []genaiapi.Message) error {
	c.Messages = make([]Message, len(msgs))
	for i, m := range msgs {
		// We don't filter the role here.
		switch m.Type {
		case genaiapi.Text:
			if m.Content == "" {
				return fmt.Errorf("message %d: missing text content", i)
			}
		default:
			return fmt.Errorf("message %d: unsupported content type %s", i, m.Type)
		}
		c.Messages[i].Role = string(m.Role)
		c.Messages[i].Content = []Content{{Type: "text", Text: m.Content}}
	}
	return nil
}

type Message struct {
	Role      string    `json:"role"`
	Content   []Content `json:"content,omitzero"`
	ToolCalls []struct {
		ID       string   `json:"id,omitzero"`
		Type     string   `json:"type,omitzero"` // "function"
		Function Function `json:"function,omitzero"`
	} `json:"tool_calls,omitzero"`
}

type Content struct {
	Type     string `json:"type"` // text,image_url
	Text     string `json:"text,omitzero"`
	ImageURL struct {
		URL string `json:"url,omitzero"`
	} `json:"image_url,omitzero"`
}

type Tool struct {
	Type     string   `json:"type,omitzero"` // "function"
	Function Function `json:"function,omitzero"`
}

type Function struct {
	Name        string   `json:"name,omitzero"`
	Description string   `json:"description,omitzero"`
	Arguments   []string `json:"arguments,omitzero"`
}

type JSONSchema any

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
			Role       genaiapi.Role `json:"role"`
			Content    string        `json:"content"`
			ToolCallID string        `json:"tool_call_id"`
			ToolCalls  []struct {
				ID       string `json:"id"`
				Type     string `json:"type"` // function
				Function struct {
					Name        string   `json:"name"`
					Description string   `json:"description"`
					Arguments   []string `json:"arguments"`
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
					Description string   `json:"description"`
					Arguments   []string `json:"arguments"`
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

type Client struct {
	// ApiKey can be retrieved from https://huggingface.co/settings/tokens. It
	// will be loaded from the huggingface python library cache by default.
	ApiKey string
	// Model to use, tens of thousands to chose from at https://huggingface.co/models?inference=warm&sort=trending
	Model string
}

// TODO: Investigate https://huggingface.co/blog/inference-providers and https://huggingface.co/docs/inference-endpoints/

func (c *Client) Completion(ctx context.Context, msgs []genaiapi.Message, opts any) (string, error) {
	// https://huggingface.co/docs/api-inference/tasks/chat-completion#api-specification
	in := CompletionRequest{}
	if err := in.fromOpts(opts); err != nil {
		return "", err
	}
	if err := in.fromMsgs(msgs); err != nil {
		return "", err
	}
	out := CompletionResponse{}
	if err := c.CompletionRaw(ctx, &in, &out); err != nil {
		return "", fmt.Errorf("failed to get chat response: %w", err)
	}
	if len(out.Choices) != 1 {
		return "", fmt.Errorf("server returned an unexpected number of choices, expected 1, got %d", len(out.Choices))
	}
	return out.Choices[0].Message.Content, nil
}

func (c *Client) CompletionRaw(ctx context.Context, in *CompletionRequest, out *CompletionResponse) error {
	if err := c.validate(true); err != nil {
		return err
	}
	url := "https://router.huggingface.co/hf-inference/models/" + c.Model + "/v1/chat/completions"
	return c.post(ctx, url, in, out)
}

func (c *Client) CompletionStream(ctx context.Context, msgs []genaiapi.Message, opts any, words chan<- string) error {
	in := CompletionRequest{Stream: true}
	if err := in.fromOpts(opts); err != nil {
		return err
	}
	if err := in.fromMsgs(msgs); err != nil {
		return err
	}
	ch := make(chan CompletionStreamChunkResponse)
	end := make(chan struct{})
	go func() {
		for msg := range ch {
			if len(msg.Choices) != 1 {
				continue
			}
			word := msg.Choices[0].Delta.Content
			if word != "" {
				words <- word
			}
		}
		end <- struct{}{}
	}()
	err := c.CompletionStreamRaw(ctx, &in, ch)
	close(ch)
	<-end
	return err
}

func (c *Client) CompletionStreamRaw(ctx context.Context, in *CompletionRequest, out chan<- CompletionStreamChunkResponse) error {
	if err := c.validate(true); err != nil {
		return err
	}
	h := make(http.Header)
	h.Add("Authorization", "Bearer "+c.ApiKey)
	// HuggingFace support all three of gzip, br and zstd!
	p := httpjson.DefaultClient
	// p.PostCompress = "zstd"
	url := "https://router.huggingface.co/hf-inference/models/" + c.Model + "/v1/chat/completions"
	resp, err := p.PostRequest(ctx, url, h, in)
	if err != nil {
		return fmt.Errorf("failed to get server response: %w", err)
	}
	defer resp.Body.Close()
	r := bufio.NewReader(resp.Body)
	for first := true; ; first = false {
		line, err := r.ReadBytes('\n')
		line = bytes.TrimSpace(line)
		if err == io.EOF {
			err = nil
			if len(line) == 0 {
				return nil
			}
		}
		if err != nil {
			return fmt.Errorf("failed to get server response: %w", err)
		}
		if len(line) == 0 {
			continue
		}
		const prefix = "data: "
		if !bytes.HasPrefix(line, []byte(prefix)) {
			// HuggingFace has the bad habit of returning errors as HTML pages.
			if first {
				// Often has a 503 in there as a <div>.
				rest, _ := io.ReadAll(r)
				return fmt.Errorf("unexpected error: %s\n%s", line, rest)
			}
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
		if err = d.Decode(&msg); err != nil {
			return fmt.Errorf("failed to decode server response %q: %w", string(line), err)
		}
		out <- msg
	}
}

func (c *Client) CompletionContent(ctx context.Context, msgs []genaiapi.Message, opts any, mime string, content []byte) (string, error) {
	return "", errors.New("not implemented")
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
	if err := c.validate(false); err != nil {
		return nil, err
	}
	h := make(http.Header)
	h.Add("Authorization", "Bearer "+c.ApiKey)
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

func (c *Client) validate(needModel bool) error {
	if c.ApiKey == "" {
		// Fallback to loading from the python client's cache.
		h, err := os.UserHomeDir()
		if err != nil {
			return fmt.Errorf("failed to load hugginface key, get one at %s: %w", apiKeyURL, err)
		}
		// TODO: Windows.
		b, err := os.ReadFile(filepath.Join(h, ".cache", "huggingface", "token"))
		if err != nil {
			return fmt.Errorf("failed to load hugginface key, get one at %s: %w", apiKeyURL, err)
		}
		c.ApiKey = strings.TrimSpace(string(b))
		if c.ApiKey == "" {
			return fmt.Errorf("loaded an empty api key, get one at %s", apiKeyURL)
		}
	}
	if needModel && c.Model == "" {
		return errors.New("a Model is required")
	}
	return nil
}

func (c *Client) post(ctx context.Context, url string, in, out any) error {
	h := make(http.Header)
	h.Add("Authorization", "Bearer "+c.ApiKey)
	// HuggingFace support all three of gzip, br and zstd!
	p := httpjson.DefaultClient
	p.PostCompress = "zstd"
	resp, err := p.PostRequest(ctx, url, h, in)
	if err != nil {
		return err
	}
	switch i, err := httpjson.DecodeResponse(resp, out); i {
	case 0:
		return nil
	default:
		// HuggingFace never return a structured error?
		var herr *httpjson.Error
		if errors.As(err, &herr) {
			slog.WarnContext(ctx, "huggingface", "url", url, "err", err, "response", string(herr.ResponseBody), "status", herr.StatusCode)
			// Hugginface returns raw unstructured text if it fails decoding. Just
			// return the content. Sometimes it's a web page because why not?
			return errors.New(string(herr.ResponseBody))
		}
		slog.WarnContext(ctx, "huggingface", "url", url, "err", err)
		return err
	}
}

const apiKeyURL = "https://huggingface.co/settings/tokens"

var _ genaiapi.CompletionProvider = &Client{}
