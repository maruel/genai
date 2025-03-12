// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Package mistral implements a client for the Mistral API.
//
// It is described at https://docs.mistral.ai/api/
package mistral

import (
	"bufio"
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"strings"
	"time"

	"github.com/maruel/genai/genaiapi"
	"github.com/maruel/genai/internal"
	"github.com/maruel/httpjson"
)

// https://docs.mistral.ai/api/#tag/chat/operation/chat_completion_v1_chat_completions_post
type CompletionRequest struct {
	Model          string    `json:"model"`
	Temperature    float64   `json:"temperature,omitzero"` // [0, 2]
	TopP           float64   `json:"top_p,omitzero"`       // [0, 1]
	MaxTokens      int64     `json:"max_tokens,omitzero"`
	Stream         bool      `json:"stream"`
	Stop           []string  `json:"stop,omitzero"` // keywords to stop completion
	RandomSeed     int64     `json:"random_seed,omitzero"`
	Messages       []Message `json:"messages"`
	ResponseFormat struct {
		Type       string `json:"type,omitzero"` // "text", "json_object", "json_schema"
		JSONSchema struct {
			Name        string `json:"name,omitzero"`
			Description string `json:"description,omitzero"`
			Strict      bool   `json:"strict,omitzero"`
			Schema      struct {
				// JSONSchema.
				Property string `json:"property,omitzero"`
			} `json:"schema,omitzero"`
		} `json:"json_schema,omitzero"`
	} `json:"response_format,omitzero"`
	Tools []Tool `json:"tools,omitzero"`
	// Alternative when forcing a specific function. This can probably be achieved
	// by providing a single tool and ToolChoice == "required".
	// ToolChoice struct {
	// 	Type     string `json:"type,omitzero"` // "function"
	// 	Function struct {
	// 		Name string `json:"name,omitzero"`
	// 	} `json:"function,omitzero"`
	// } `json:"tool_choice,omitzero"`
	ToolChoice       string  `json:"tool_choice,omitzero"`       // "auto", "none", "any", "required"
	PresencePenalty  float64 `json:"presence_penalty,omitzero"`  // [-2.0, 2.0]
	FrequencyPenalty float64 `json:"frequency_penalty,omitzero"` // [-2.0, 2.0]
	N                int64   `json:"n,omitzero"`                 // Number of choices
	Prediction       struct {
		// Enable users to specify expected results, optimizing response times by
		// leveraging known or predictable content. This approach is especially
		// effective for updating text documents or code files with minimal
		// changes, reducing latency while maintaining high-quality results.
		Type    string `json:"type,omitzero"` // "content"
		Content string `json:"content,omitzero"`
	} `json:"prediction,omitzero"`
	SafePrompt bool `json:"safe_prompt,omitzero"`
}

func (c *CompletionRequest) fromOpts(opts any) error {
	switch v := opts.(type) {
	case *genaiapi.CompletionOptions:
		c.MaxTokens = v.MaxTokens
		c.RandomSeed = v.Seed
		c.Temperature = v.Temperature
	default:
		return fmt.Errorf("unsupported options type %T", opts)
	}
	return nil
}

func (c *CompletionRequest) fromMsgs(msgs []genaiapi.Message) error {
	c.Messages = make([]Message, len(msgs))
	for i, m := range msgs {
		if err := m.Validate(); err != nil {
			return fmt.Errorf("message %d: %w", i, err)
		}
		switch m.Role {
		case genaiapi.System:
			if i != 0 {
				return fmt.Errorf("message %d: system message must be first message", i)
			}
		case genaiapi.User, genaiapi.Assistant, genaiapi.Tool:
		default:
			return fmt.Errorf("message %d: unsupported role %q", i, m.Role)
		}
		c.Messages[i].Role = string(m.Role)
		c.Messages[i].Content = []Content{{}}
		switch m.Type {
		case genaiapi.Text:
			c.Messages[i].Content[0].Type = "text"
			c.Messages[i].Content[0].Text = m.Text
		case genaiapi.Document:
			mimeType, data, err := internal.ParseDocument(&m, 10*1024*1024)
			if err != nil {
				return fmt.Errorf("message %d: %w", i, err)
			}
			switch {
			case (m.URL != "" && mimeType == "") || strings.HasPrefix(mimeType, "image/"):
				c.Messages[i].Content[0].Type = "image_url"
				if m.URL == "" {
					c.Messages[i].Content[0].ImageURL.URL = fmt.Sprintf("data:%s;base64,%s", mimeType, base64.StdEncoding.EncodeToString(data))
				} else {
					c.Messages[i].Content[0].ImageURL.URL = m.URL
				}
			default:
				return fmt.Errorf("message %d: unsupported mime type %s", i, mimeType)
			}
		default:
			return fmt.Errorf("message %d: unsupported content type %s", i, m.Type)
		}
	}
	return nil
}

type Message struct {
	Role    string    `json:"role"`
	Content []Content `json:"content"`
}

type Content struct {
	Type string `json:"type"` // "text", "reference", "document_url", "image_url"

	// Type == "text"
	Text string `json:"text,omitzero"`

	// Type == "reference"
	ReferenceIDs []int64 `json:"reference_ids,omitzero"`

	// Type == "document_url"
	DocumentURL  string `json:"document_url,omitzero"`
	DocumentName string `json:"document_name,omitzero"`

	// Type == "image_url"
	ImageURL struct {
		URL    string `json:"url,omitzero"`
		Detail string `json:"detail,omitzero"` // undocumented, likely "auto" like OpenAI
	} `json:"image_url,omitzero"`
}

type Tool struct {
	Type     string `json:"type,omitzero"` // "function"
	Function struct {
		Name        string         `json:"name,omitzero"`
		Description string         `json:"description,omitzero"`
		String      bool           `json:"string,omitzero"`
		Parameters  map[string]any `json:"parameters,omitzero"`
	} `json:"function,omitzero"`
}

type CompletionResponse struct {
	ID      string `json:"id"`
	Object  string `json:"object"` // chat.completion
	Model   string `json:"model"`
	Created Time   `json:"created"` // Unix timestamp
	Choices []struct {
		// FinishReason is one of "stop", "length", "content_filter" or "tool_calls".
		FinishReason string `json:"finish_reason"`
		Index        int64  `json:"index"`
		Message      struct {
			Role      genaiapi.Role `json:"role"`
			Content   string        `json:"content"`
			Prefix    bool          `json:"prefix"`
			ToolCalls []struct {
				ID       string `json:"id"`
				Type     string `json:"type"`
				Function struct {
					Name      string `json:"name"`
					Arguments []any  `json:"arguments"`
				} `json:"function"`
				Index int64 `json:"index"`
			} `json:"tool_calls"`
		} `json:"message"`
	} `json:"choices"`
	Usage struct {
		// QueueTime        float64 `json:"queue_time"`
		PromptTokens int64 `json:"prompt_tokens"`
		// PromptTime       float64 `json:"prompt_time"`
		CompletionTokens int64 `json:"completion_tokens"`
		// CompletionTime   float64 `json:"completion_time"`
		TotalTokens int64 `json:"total_tokens"`
		// TotalTime        float64 `json:"total_time"`
	} `json:"usage"`
}

type CompletionStreamChunkResponse struct {
	ID      string `json:"id"`
	Object  string `json:"object"` // chat.completion.chunk
	Created Time   `json:"created"`
	Model   string `json:"model"`
	Choices []struct {
		Index int64 `json:"index"`
		Delta struct {
			Role    genaiapi.Role `json:"role"`
			Content string        `json:"content"`
		} `json:"delta"`
		FinishReason string `json:"finish_reason"`
	} `json:"choices"`
	Usage struct {
		PromptTokens     int64 `json:"prompt_tokens"`
		CompletionTokens int64 `json:"completion_tokens"`
		TotalTokens      int64 `json:"total_tokens"`
	} `json:"usage"`
}

//

// errorResponseAuth is used when simple issue like auth failure.
type errorResponseAuth struct {
	Message   string `json:"message"`
	RequestID string `json:"request_id"`
}

type errorResponseAPI1 struct {
	Object  string `json:"object"` // error
	Message string `json:"message"`
	Type    string `json:"type"`
	Param   string `json:"param"`
	Code    int64  `json:"code"`
}

type errorResponseAPI2 struct {
	Detail []struct {
		Type string `json:"type"` // "string_type", "missing"
		Msg  string `json:"msg"`
		Loc  []any  `json:"loc"` // to be joined, a mix of string and nuber
		// Input is either a list or an instance of struct { Type string `json:"type"` }.
		Input any    `json:"input"`
		Ctx   any    `json:"ctx"`
		URL   string `json:"url"`
	} `json:"detail"`
}

type errorResponseAPI3 struct {
	Object  string            `json:"object"` // error
	Message errorResponseAPI2 `json:"message"`
	Type    string            `json:"type"`
	Param   string            `json:"param"`
	Code    int64             `json:"code"`
}

type Client struct {
	// ApiKey can be retrieved from https://console.mistral.ai/api-keys or https://console.mistral.ai/codestral
	ApiKey string
	// Model to use, see https://docs.mistral.ai/getting-started/models/models_overview/
	Model string
}

// https://codestral.mistral.ai/v1/fim/completions
// https://codestral.mistral.ai/v1/chat/completions

func (c *Client) Completion(ctx context.Context, msgs []genaiapi.Message, opts any) (string, error) {
	in := CompletionRequest{Model: c.Model}
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
	// https://docs.mistral.ai/api/#tag/chat/operation/chat_completion_v1_chat_completions_post
	if err := c.validate(true); err != nil {
		return err
	}
	return c.post(ctx, "https://api.mistral.ai/v1/chat/completions", in, out)
}

func (c *Client) CompletionStream(ctx context.Context, msgs []genaiapi.Message, opts any, words chan<- string) error {
	in := CompletionRequest{Model: c.Model, Stream: true}
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
	// Mistral doesn't HTTP POST support compression.
	resp, err := httpjson.DefaultClient.PostRequest(ctx, "https://api.mistral.ai/v1/chat/completions", h, in)
	if err != nil {
		return fmt.Errorf("failed to get server response: %w", err)
	}
	defer resp.Body.Close()
	r := bufio.NewReader(resp.Body)
	for {
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
			d := json.NewDecoder(bytes.NewReader(line))
			d.DisallowUnknownFields()
			d.UseNumber()
			erAuth := errorResponseAuth{}
			if err = d.Decode(&erAuth); err != nil {
				return fmt.Errorf("unexpected line. expected \"data: \", got %q", line)
			}
			// Happens with Requests rate limit exceeded.
			// TODO: Wrap it so it can be retried like a 429.
			return errors.New(erAuth.Message)
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
	if err := c.validate(true); err != nil {
		return "", err
	}
	return "", errors.New("not implemented")
}

type Time int64

func (t *Time) AsTime() time.Time {
	return time.Unix(int64(*t), 0)
}

// https://docs.mistral.ai/api/#tag/models/operation/retrieve_model_v1_models__model_id__get
type Model struct {
	ID           string `json:"id"`
	Object       string `json:"object"`
	Created      Time   `json:"created"`
	OwnedBy      string `json:"owned_by"`
	Capabilities struct {
		CompletionChat  bool `json:"completion_chat"`
		CompletionFim   bool `json:"completion_fim"`
		FunctionCalling bool `json:"function_calling"`
		FineTuning      bool `json:"fine_tuning"`
		Vision          bool `json:"vision"`
	} `json:"capabilities"`
	Name                    string   `json:"name"`
	Description             string   `json:"description"`
	MaxContextLength        int64    `json:"max_context_length"`
	Aliases                 []string `json:"aliases"`
	Deprecation             string   `json:"deprecation"`
	DefaultModelTemperature float64  `json:"default_model_temperature"`
	Type                    string   `json:"type"`
}

func (m *Model) GetID() string {
	return m.ID
}

func (m *Model) String() string {
	var caps []string
	if m.Capabilities.CompletionChat {
		caps = append(caps, "chat")
	}
	if m.Capabilities.CompletionFim {
		caps = append(caps, "fim")
	}
	if m.Capabilities.FunctionCalling {
		caps = append(caps, "function")
	}
	if m.Capabilities.FineTuning {
		caps = append(caps, "fine-tuning")
	}
	if m.Capabilities.Vision {
		caps = append(caps, "vision")
	}
	suffix := ""
	if m.Deprecation != "" {
		suffix += " (deprecated)"
	}
	prefix := m.ID
	if m.ID != m.Name {
		prefix += " (" + m.Name + ")"
	}
	// Not including Created and Description because Created is not set and Description is not useful.
	return fmt.Sprintf("%s: %s Context: %d%s", prefix, strings.Join(caps, "/"), m.MaxContextLength, suffix)
}

func (m *Model) Context() int64 {
	return m.MaxContextLength
}

func (c *Client) ListModels(ctx context.Context) ([]genaiapi.Model, error) {
	// https://docs.mistral.ai/api/#tag/models
	if err := c.validate(false); err != nil {
		return nil, err
	}
	h := make(http.Header)
	h.Add("Authorization", "Bearer "+c.ApiKey)
	var out struct {
		Object string  `json:"object"` // list
		Data   []Model `json:"data"`
	}
	err := httpjson.DefaultClient.Get(ctx, "https://api.mistral.ai/v1/models", h, &out)
	if err != nil {
		return nil, err
	}
	models := make([]genaiapi.Model, len(out.Data))
	for i := range out.Data {
		models[i] = &out.Data[i]
	}
	return models, err
}

func (c *Client) validate(needModel bool) error {
	if c.ApiKey == "" {
		return errors.New("mistral ApiKey is required; get one at " + apiKeyURL)
	}
	if needModel && c.Model == "" {
		return errors.New("a Model is required")
	}
	return nil
}

func (c *Client) post(ctx context.Context, url string, in, out any) error {
	if c.ApiKey == "" {
		return errors.New("mistral ApiKey is required; get one at " + apiKeyURL)
	}
	h := make(http.Header)
	h.Add("Authorization", "Bearer "+c.ApiKey)
	// Mistral doesn't HTTP POST support compression.
	resp, err := httpjson.DefaultClient.PostRequest(ctx, url, h, in)
	if err != nil {
		return err
	}
	// This is so cute.
	erAuth := errorResponseAuth{}
	erAPI1 := errorResponseAPI1{}
	erAPI2 := errorResponseAPI2{}
	erAPI3 := errorResponseAPI3{}
	switch i, err := httpjson.DecodeResponse(resp, out, &erAuth, &erAPI1, &erAPI2, &erAPI3); i {
	case 0:
		return nil
	case 1:
		var herr *httpjson.Error
		if errors.As(err, &herr) {
			if herr.StatusCode == http.StatusUnauthorized {
				return fmt.Errorf("%w: %s. You can get a new API key at %s", herr, erAuth.Message, apiKeyURL)
			}
			return fmt.Errorf("%w: %s", herr, erAuth.Message)
		}
		return errors.New(erAuth.Message)
	case 2:
		var herr *httpjson.Error
		if errors.As(err, &herr) {
			return fmt.Errorf("%w: error %s: %s", herr, erAPI1.Type, erAPI1.Message)
		}
		return fmt.Errorf("error %s: %s", erAPI1.Type, erAPI1.Message)
	case 3:
		var herr *httpjson.Error
		if errors.As(err, &herr) {
			return fmt.Errorf("%w: error %s: %s at %s", herr, erAPI2.Detail[0].Type, erAPI2.Detail[0].Msg, erAPI2.Detail[0].Loc)
		}
		return fmt.Errorf("error %s: %s at %s", erAPI2.Detail[0].Type, erAPI2.Detail[0].Msg, erAPI2.Detail[0].Loc)
	case 4:
		var herr *httpjson.Error
		if errors.As(err, &herr) {
			return fmt.Errorf("%w: error %s/%s: %s at %s", herr, erAPI3.Type, erAPI3.Message.Detail[0].Type, erAPI3.Message.Detail[0].Msg, erAPI3.Message.Detail[0].Loc)
		}
		return fmt.Errorf("error %s/%s: %s at %s", erAPI3.Type, erAPI3.Message.Detail[0].Type, erAPI3.Message.Detail[0].Msg, erAPI3.Message.Detail[0].Loc)
	default:
		var herr *httpjson.Error
		if errors.As(err, &herr) {
			slog.WarnContext(ctx, "mistral", "url", url, "err", err, "response", string(herr.ResponseBody), "status", herr.StatusCode)
		} else {
			slog.WarnContext(ctx, "mistral", "url", url, "err", err)
		}
		return err
	}
}

const apiKeyURL = "https://console.mistral.ai/api-keys"

var _ genaiapi.CompletionProvider = &Client{}
