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
	"github.com/maruel/genai"
	"github.com/maruel/genai/internal"
	"github.com/maruel/httpjson"
	"github.com/maruel/roundtrippers"
	"golang.org/x/sync/errgroup"
)

// https://huggingface.co/docs/api-inference/tasks/chat-completion#api-specification
type ChatRequest struct {
	Model            string    `json:"model,omitempty"` // It's already in the URL.
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

// Init initializes the provider specific completion request with the generic completion request.
func (c *ChatRequest) Init(msgs genai.Messages, opts genai.Validatable, model string) error {
	var errs []error
	var unsupported []string
	sp := ""
	if opts != nil {
		if err := opts.Validate(); err != nil {
			errs = append(errs, err)
		} else {
			switch v := opts.(type) {
			case *genai.ChatOptions:
				c.MaxTokens = v.MaxTokens
				c.Temperature = v.Temperature
				c.TopP = v.TopP
				sp = v.SystemPrompt
				c.Seed = v.Seed
				if v.TopK != 0 {
					unsupported = append(unsupported, "TopK")
				}
				c.Stop = v.Stop
				if v.ReplyAsJSON {
					c.ResponseFormat.Type = "json"
				}
				if v.DecodeAs != nil {
					c.ResponseFormat.Type = "json"
					c.ResponseFormat.Value = jsonschema.Reflect(v.DecodeAs)
				}
				if len(v.Tools) != 0 {
					if v.ToolCallRequired {
						c.ToolChoice = "required"
					} else {
						c.ToolChoice = "auto"
					}
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

	if err := msgs.Validate(); err != nil {
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
	if len(unsupported) > 0 {
		// If we have unsupported features but no other errors, return a continuable error
		if len(errs) == 0 {
			return &genai.UnsupportedContinuableError{Unsupported: unsupported}
		}
		// Otherwise, add the unsupported features to the error list
		errs = append(errs, &genai.UnsupportedContinuableError{Unsupported: unsupported})
	}
	return errors.Join(errs...)
}

// https://huggingface.co/docs/api-inference/tasks/chat-completion#api-specification
type Message struct {
	Role      string     `json:"role"` // "system", "assistant", "user"
	Content   []Content  `json:"content,omitzero"`
	ToolCalls []ToolCall `json:"tool_calls,omitzero"`
}

func (m *Message) From(in *genai.Message) error {
	switch in.Role {
	case genai.User, genai.Assistant:
		m.Role = string(in.Role)
	default:
		return fmt.Errorf("unsupported role %q", in.Role)
	}
	if len(in.Contents) != 0 {
		m.Content = make([]Content, len(in.Contents))
		for i := range in.Contents {
			if err := m.Content[i].From(&in.Contents[i]); err != nil {
				return fmt.Errorf("block %d: %w", i, err)
			}
		}
	}
	if len(in.ToolCalls) != 0 {
		for j := range in.ToolCalls {
			if err := m.ToolCalls[j].From(&in.ToolCalls[j]); err != nil {
				return fmt.Errorf("tool call %d: %w", j, err)
			}
		}
	}
	return nil
}

type Content struct {
	Type     string `json:"type"` // "text", "image_url"
	Text     string `json:"text,omitzero"`
	ImageURL struct {
		URL string `json:"url,omitzero"`
	} `json:"image_url,omitzero"`
}

func (c *Content) From(in *genai.Content) error {
	if in.Text != "" {
		c.Type = "text"
		c.Text = in.Text
		return nil
	}
	mimeType, data, err := in.ReadDocument(10 * 1024 * 1024)
	if err != nil {
		return err
	}
	switch {
	case (in.URL != "" && mimeType == "") || strings.HasPrefix(mimeType, "image/"):
		c.Type = "image_url"
		if in.URL == "" {
			c.ImageURL.URL = fmt.Sprintf("data:%s;base64,%s", mimeType, base64.StdEncoding.EncodeToString(data))
		} else {
			c.ImageURL.URL = in.URL
		}
	default:
		return fmt.Errorf("unsupported mime type %s", mimeType)
	}
	return nil
}

type ToolCall struct {
	ID       string `json:"id,omitzero"`
	Type     string `json:"type,omitzero"` // "function"
	Function struct {
		Name        string   `json:"name,omitzero"`
		Description struct{} `json:"description,omitzero"` // Passed in as null in response
		Arguments   string   `json:"arguments,omitzero"`
	} `json:"function,omitzero"`
}

func (t *ToolCall) From(in *genai.ToolCall) error {
	t.ID = in.ID
	t.Type = "function"
	t.Function.Name = in.Name
	// The API seems to flip-flop between JSON and string.
	// return json.Unmarshal([]byte(in.Arguments), &t.Function.Arguments)
	t.Function.Arguments = in.Arguments
	return nil
}

func (t *ToolCall) To(out *genai.ToolCall) error {
	out.ID = t.ID
	out.Name = t.Function.Name
	// b, err := json.Marshal(t.Function.Arguments)
	// if err != nil {
	//	return fmt.Errorf("failed to marshal arguments: %w", err)
	// }
	// out.Arguments = string(b)
	out.Arguments = t.Function.Arguments
	return nil
}

type Tool struct {
	Type     string `json:"type,omitzero"` // "function"
	Function struct {
		Name        string             `json:"name,omitzero"`
		Description string             `json:"description,omitzero"`
		Arguments   *jsonschema.Schema `json:"arguments,omitzero"`
	} `json:"function,omitzero"`
}

type ChatResponse struct {
	Object            string `json:"object"`
	ID                string `json:"id"`
	Created           Time   `json:"created"`
	Model             string `json:"model"`
	SystemFingerprint string `json:"system_fingerprint"`

	Choices []struct {
		FinishReason string          `json:"finish_reason"`
		Index        int64           `json:"index"`
		Message      MessageResponse `json:"message"`
		Logprobs     struct {
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
	Usage Usage `json:"usage"`
}

type Usage struct {
	PromptTokens     int64 `json:"prompt_tokens"`
	CompletionTokens int64 `json:"completion_tokens"`
	TotalTokens      int64 `json:"total_tokens"`
}

// The structure is different than the request. :(
type MessageResponse struct {
	Role       string     `json:"role"`
	Content    string     `json:"content"`
	ToolCallID string     `json:"tool_call_id"`
	ToolCalls  []ToolCall `json:"tool_calls"`
}

func (m *MessageResponse) To(out *genai.Message) error {
	switch role := m.Role; role {
	case "assistant", "user":
		out.Role = genai.Role(role)
	default:
		return fmt.Errorf("unsupported role %q", role)
	}
	if len(m.ToolCalls) != 0 {
		out.ToolCalls = make([]genai.ToolCall, len(m.ToolCalls))
		for i := range m.ToolCalls {
			if err := m.ToolCalls[i].To(&out.ToolCalls[i]); err != nil {
				return fmt.Errorf("tool call %d: %w", i, err)
			}
		}
	}

	if m.Content != "" {
		out.Contents = []genai.Content{{Text: m.Content}}
	}
	return nil
}

func (c *ChatResponse) ToResult() (genai.ChatResult, error) {
	out := genai.ChatResult{
		// At the moment, Huggingface doesn't support caching.
		Usage: genai.Usage{
			InputTokens:  c.Usage.PromptTokens,
			OutputTokens: c.Usage.CompletionTokens,
		},
		FinishReason: c.Choices[0].FinishReason,
	}
	if len(c.Choices) != 1 {
		return out, fmt.Errorf("server returned an unexpected number of choices, expected 1, got %d", len(c.Choices))
	}
	err := c.Choices[0].Message.To(&out.Message)
	return out, err
}

type ChatStreamChunkResponse struct {
	Object            string `json:"object"`
	Created           Time   `json:"created"`
	ID                string `json:"id"`
	Model             string `json:"model"`
	SystemFingerprint string `json:"system_fingerprint"`
	Choices           []struct {
		Index        int64  `json:"index"`
		FinishReason string `json:"finish_reason"` // stop
		Delta        struct {
			Role       genai.Role `json:"role"`
			Content    string     `json:"content"`
			ToolCallID string     `json:"tool_call_id"`
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
	Usage Usage `json:"usage"`
}

type errorResponse1 struct {
	Error string `json:"error"`
}

type errorResponse2 struct {
	Error struct {
		Message        string `json:"message"`
		HTTPStatusCode int64  `json:"http_status_code"`
	} `json:"error"`
}

// Client implements the REST JSON based API.
type Client struct {
	// Client is exported for testing replay purposes.
	Client httpjson.Client

	model string
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
	return &Client{
		model: model,
		Client: httpjson.Client{
			Client: &http.Client{Transport: &roundtrippers.Header{
				Header: http.Header{"Authorization": {"Bearer " + apiKey}},
				Transport: &roundtrippers.Retry{
					Transport: &roundtrippers.RequestID{
						Transport: http.DefaultTransport,
					},
				},
			}},
			Lenient: internal.BeLenient,
		},
	}, nil
}

func (c *Client) Chat(ctx context.Context, msgs genai.Messages, opts genai.Validatable) (genai.ChatResult, error) {
	// https://huggingface.co/docs/api-inference/tasks/chat-completion#api-specification
	for i, msg := range msgs {
		if len(msg.Opaque) != 0 {
			return genai.ChatResult{}, fmt.Errorf("message #%d: field Opaque not supported", i)
		}
	}
	rpcin := ChatRequest{}
	var continuableErr error
	if err := rpcin.Init(msgs, opts, c.model); err != nil {
		// If it's an UnsupportedContinuableError, we can continue
		if uce, ok := err.(*genai.UnsupportedContinuableError); ok {
			// Store the error to return later if no other error occurs
			continuableErr = uce
			// Otherwise log the error but continue
		} else {
			return genai.ChatResult{}, err
		}
	}
	rpcout := ChatResponse{}
	if err := c.ChatRaw(ctx, &rpcin, &rpcout); err != nil {
		return genai.ChatResult{}, fmt.Errorf("failed to get chat response: %w", err)
	}
	result, err := rpcout.ToResult()
	if err != nil {
		return result, err
	}
	// Return the continuable error if no other error occurred
	if continuableErr != nil {
		return result, continuableErr
	}
	return result, nil
}

func (c *Client) ChatRaw(ctx context.Context, in *ChatRequest, out *ChatResponse) error {
	if err := c.validate(); err != nil {
		return err
	}
	in.Stream = false
	url := "https://router.huggingface.co/hf-inference/models/" + c.model + "/v1/chat/completions"
	return c.post(ctx, url, in, out)
}

func (c *Client) ChatStream(ctx context.Context, msgs genai.Messages, opts genai.Validatable, chunks chan<- genai.MessageFragment) (genai.Usage, error) {
	// Check for non-empty Opaque field
	for _, msg := range msgs {
		if len(msg.Opaque) != 0 {
			return genai.Usage{}, fmt.Errorf("Opaque field not supported")
		}
	}

	in := ChatRequest{}
	usage := genai.Usage{}
	var continuableErr error
	if err := in.Init(msgs, opts, c.model); err != nil {
		// If it's an UnsupportedContinuableError, we can continue
		if uce, ok := err.(*genai.UnsupportedContinuableError); ok {
			// Store the error to return later if no other error occurs
			continuableErr = uce
			// Otherwise log the error but continue
		} else {
			return usage, err
		}
	}
	ch := make(chan ChatStreamChunkResponse)
	eg, ctx := errgroup.WithContext(ctx)
	finalUsage := Usage{}
	eg.Go(func() error {
		return processStreamPackets(ch, chunks, &finalUsage)
	})
	err := c.ChatStreamRaw(ctx, &in, ch)
	close(ch)
	if err2 := eg.Wait(); err2 != nil {
		err = err2
	}
	usage.InputTokens = finalUsage.PromptTokens
	usage.OutputTokens = finalUsage.CompletionTokens
	// Return the continuable error if no other error occurred
	if err == nil && continuableErr != nil {
		return usage, continuableErr
	}
	return usage, err
}

func processStreamPackets(ch <-chan ChatStreamChunkResponse, chunks chan<- genai.MessageFragment, finalUsage *Usage) error {
	defer func() {
		// We need to empty the channel to avoid blocking the goroutine.
		for range ch {
		}
	}()
	for pkt := range ch {
		if len(pkt.Choices) != 1 {
			continue
		}
		if pkt.Usage.TotalTokens != 0 {
			*finalUsage = pkt.Usage
		}
		switch role := pkt.Choices[0].Delta.Role; role {
		case "assistant", "":
		default:
			return fmt.Errorf("unexpected role %q", role)
		}
		if word := pkt.Choices[0].Delta.Content; word != "" {
			fragment := genai.MessageFragment{TextFragment: word}
			// Include FinishReason if available
			if pkt.Choices[0].FinishReason != "" {
				fragment.FinishReason = pkt.Choices[0].FinishReason
			}
			chunks <- fragment
		}
	}
	return nil
}

func (c *Client) ChatStreamRaw(ctx context.Context, in *ChatRequest, out chan<- ChatStreamChunkResponse) error {
	if err := c.validate(); err != nil {
		return err
	}
	in.Stream = true
	url := "https://router.huggingface.co/hf-inference/models/" + c.model + "/v1/chat/completions"
	resp, err := c.Client.PostRequest(ctx, url, nil, in)
	if err != nil {
		return fmt.Errorf("failed to get server response: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode > 400 {
		// HuggingFace has the bad habit of returning errors as HTML pages.
		_, _ = io.Copy(io.Discard, resp.Body)
		return fmt.Errorf("http %d: %s", resp.StatusCode, resp.Status)
	}
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

func parseStreamLine(line []byte, out chan<- ChatStreamChunkResponse) error {
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
	msg := ChatStreamChunkResponse{}
	if err := d.Decode(&msg); err != nil {
		d = json.NewDecoder(strings.NewReader(suffix))
		d.DisallowUnknownFields()
		d.UseNumber()
		er2 := errorResponse2{}
		if err := d.Decode(&er2); err != nil {
			return fmt.Errorf("failed to decode server response %q: %w", string(line), err)
		}
		return fmt.Errorf("got error from server: http %d: %s", er2.Error.HTTPStatusCode, er2.Error.Message)
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

func (c *Client) ListModels(ctx context.Context) ([]genai.Model, error) {
	// https://huggingface.co/docs/hub/api

	// return nil, errors.New("not implemented; there's just too many, tens of thousands to chose from at https://huggingface.co/models?inference=warm&sort=trending")
	var out []Model
	// There's 20k models warm as of March 2025. There's no way to sort by
	// trending. Sorting by download is not useful.
	if err := c.Client.Get(ctx, "https://huggingface.co/api/models?inference=warm", nil, &out); err != nil {
		return nil, err
	}
	models := make([]genai.Model, len(out))
	for i := range out {
		models[i] = &out[i]
	}
	return models, nil
}

func (c *Client) validate() error {
	if c.model == "" {
		return errors.New("a model is required")
	}
	return nil
}

func (c *Client) post(ctx context.Context, url string, in, out any) error {
	resp, err := c.Client.PostRequest(ctx, url, nil, in)
	if err != nil {
		return err
	}
	er1 := errorResponse1{}
	switch i, err := httpjson.DecodeResponse(resp, out, &er1); i {
	case 0:
		return nil
	case 1:
		var herr *httpjson.Error
		if errors.As(err, &herr) {
			if herr.StatusCode >= 400 {
				return fmt.Errorf("huggingface: %s; %s", http.StatusText(herr.StatusCode), er1.Error)
			}
		}
		return fmt.Errorf("huggingface: %s", er1.Error)
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
	_ genai.ChatProvider  = &Client{}
	_ genai.ModelProvider = &Client{}
)
