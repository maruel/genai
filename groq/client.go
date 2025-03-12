// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Package groq implements a client for the Groq API.
//
// It is described at https://console.groq.com/docs/api-reference
package groq

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

// https://console.groq.com/docs/api-reference#chat-create
type CompletionRequest struct {
	FrequencyPenalty    float64   `json:"frequency_penalty,omitzero"` // [-2.0, 2.0]
	MaxCompletionTokens int64     `json:"max_completion_tokens,omitzero"`
	Messages            []Message `json:"messages"`
	Model               string    `json:"model"`
	ParallelToolCalls   bool      `json:"parallel_tool_calls,omitzero"`
	PresencePenalty     float64   `json:"presence_penalty,omitzero"` // [-2.0, 2.0]
	ReasoningFormat     string    `json:"reasoning_format,omitzero"`
	ResponseFormat      struct {
		Type string `json:"type,omitzero"` // "json_object"
	} `json:"response_format,omitzero"`
	Seed          int64    `json:"seed,omitzero"`
	ServiceTier   string   `json:"service_tier,omitzero"` // "on_demand", "auto", "flex"
	Stop          []string `json:"stop,omitzero"`         // keywords to stop completion
	Stream        bool     `json:"stream"`
	StreamOptions struct {
		IncludeUsage bool `json:"include_usage,omitzero"`
	} `json:"stream_options,omitzero"`
	Temperature float64 `json:"temperature,omitzero"` // [0, 2]
	Tools       []Tool  `json:"tools,omitzero"`
	// Alternative when forcing a specific function. This can probably be achieved
	// by providing a single tool and ToolChoice == "required".
	// ToolChoice struct {
	// 	Type     string `json:"type,omitzero"` // "function"
	// 	Function struct {
	// 		Name string `json:"name,omitzero"`
	// 	} `json:"function,omitzero"`
	// } `json:"tool_choice,omitzero"`
	ToolChoice string  `json:"tool_choice,omitzero"` // "none", "auto", "required"
	TopP       float64 `json:"top_p,omitzero"`       // [0, 1]
	User       string  `json:"user,omitzero"`

	// Explicitly Unsupported:
	// LogitBias           map[string]float64 `json:"logit_bias,omitzero"`
	// Logprobs            bool               `json:"logprobs,omitzero"`
	// TopLogprobs         int64                `json:"top_logprobs,omitzero"`     // [0, 20]
	// N                   int64                `json:"n,omitzero"`                // Number of choices
}

func (c *CompletionRequest) fromOpts(opts any) error {
	if opts != nil {
		switch v := opts.(type) {
		case *genaiapi.CompletionOptions:
			c.MaxCompletionTokens = v.MaxTokens
			c.Seed = v.Seed
			c.Temperature = v.Temperature
		default:
			return fmt.Errorf("unsupported options type %T", opts)
		}
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
	Role       string     `json:"role"`
	Content    []Content  `json:"content,omitzero"`
	Name       string     `json:"name,omitzero"`
	ToolCalls  []ToolCall `json:"tool_calls,omitzero"`
	ToolCallID string     `json:"tool_call_id,omitzero"`
}

type Content struct {
	Type string `json:"type,omitzero"` // "text", "image_url"

	// Type == "text"
	Text string `json:"text,omitzero"`

	// Type == "image_url"
	ImageURL struct {
		Detail string `json:"detail,omitzero"` // "auto", "low", "high"
		URL    string `json:"url,omitzero"`    // URL or base64 encoded image
	} `json:"image_url,omitzero"`
}

type Tool struct {
	Type     string `json:"type,omitzero"` // "function"
	Function struct {
		Name        string         `json:"name,omitzero"`
		Description string         `json:"description,omitzero"`
		Parameters  map[string]any `json:"parameters,omitzero"`
	} `json:"function,omitzero"`
}

type ToolCall struct {
	Type     string `json:"type,omitzero"` // "function"
	ID       string `json:"id,omitzero"`
	Function struct {
		Name      string `json:"name,omitzero"`
		Arguments string `json:"arguments,omitzero"`
	} `json:"function,omitzero"`
}

type CompletionResponse struct {
	Choices []struct {
		// FinishReason is one of "stop", "length", "content_filter" or "tool_calls".
		FinishReason string `json:"finish_reason"`
		Index        int64  `json:"index"`
		Message      struct {
			Role    genaiapi.Role `json:"role"`
			Content string        `json:"content"`
		} `json:"message"`
		Logprobs struct{} `json:"logprobs"`
	} `json:"choices"`
	Created Time   `json:"created"`
	ID      string `json:"id"`
	Model   string `json:"model"`
	Object  string `json:"object"` // chat.completion
	Usage   struct {
		QueueTime        float64 `json:"queue_time"`
		PromptTokens     int64   `json:"prompt_tokens"`
		PromptTime       float64 `json:"prompt_time"`
		CompletionTokens int64   `json:"completion_tokens"`
		CompletionTime   float64 `json:"completion_time"`
		TotalTokens      int64   `json:"total_tokens"`
		TotalTime        float64 `json:"total_time"`
	} `json:"usage"`
	SystemFingerprint string `json:"system_fingerprint"`
	Xgroq             struct {
		ID string `json:"id"`
	} `json:"x_groq"`
}

type CompletionStreamChunkResponse struct {
	ID                string `json:"id"`
	Object            string `json:"object"`
	Created           Time   `json:"created"`
	Model             string `json:"model"`
	SystemFingerprint string `json:"system_fingerprint"`
	Choices           []struct {
		Index int64 `json:"index"`
		Delta struct {
			Role    genaiapi.Role `json:"role"`
			Content string        `json:"content"`
		} `json:"delta"`
		Logprobs     struct{} `json:"logprobs"`
		FinishReason string   `json:"finish_reason"` // stop
	} `json:"choices"`
	Xgroq struct {
		ID    string `json:"id"`
		Usage struct {
			QueueTime        float64 `json:"queue_time"`
			PromptTokens     int64   `json:"prompt_tokens"`
			PromptTime       float64 `json:"prompt_time"`
			CompletionTokens int64   `json:"completion_tokens"`
			CompletionTime   float64 `json:"completion_time"`
			TotalTokens      int64   `json:"total_tokens"`
			TotalTime        float64 `json:"total_time"`
		} `json:"usage"`
	} `json:"x_groq"`
}

//

type errorResponse struct {
	Error struct {
		Message string `json:"message"`
		Type    string `json:"type"`
		Code    string `json:"code"`
	} `json:"error"`
}

type Client struct {
	// ApiKey can be retrieved from https://console.groq.com/keys
	ApiKey string
	// Model to use, from https://console.groq.com/dashboard/limits or https://console.groq.com/docs/models
	Model string
}

func (c *Client) Completion(ctx context.Context, msgs []genaiapi.Message, opts any) (string, error) {
	// https://console.groq.com/docs/api-reference#chat-create
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
	if err := c.validate(true); err != nil {
		return err
	}
	return c.post(ctx, "https://api.groq.com/openai/v1/chat/completions", in, out)
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
	// Groq doesn't HTTP POST support compression.
	resp, err := httpjson.DefaultClient.PostRequest(ctx, "https://api.groq.com/openai/v1/chat/completions", h, in)
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
			er := errorResponse{}
			if err = d.Decode(&er); err != nil {
				return fmt.Errorf("unexpected line. expected \"data: \", got %q", line)
			}
			return fmt.Errorf("server error %s (%s): %s", er.Error.Code, er.Error.Type, er.Error.Message)
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

type Time int64

func (t *Time) AsTime() time.Time {
	return time.Unix(int64(*t), 0)
}

type Model struct {
	ID            string   `json:"id"`
	Object        string   `json:"object"`
	Created       Time     `json:"created"`
	OwnedBy       string   `json:"owned_by"`
	Active        bool     `json:"active"`
	ContextWindow int64    `json:"context_window"`
	PublicApps    []string `json:"public_apps"`
}

func (m *Model) GetID() string {
	return m.ID
}

func (m *Model) String() string {
	suffix := ""
	if !m.Active {
		suffix = " (inactive)"
	}
	return fmt.Sprintf("%s (%s) Context: %d%s", m.ID, m.Created.AsTime().Format("2006-01-02"), m.ContextWindow, suffix)
}

func (m *Model) Context() int64 {
	return m.ContextWindow
}

func (c *Client) ListModels(ctx context.Context) ([]genaiapi.Model, error) {
	// https://console.groq.com/docs/api-reference#models-list
	if err := c.validate(false); err != nil {
		return nil, err
	}
	h := make(http.Header)
	h.Add("Authorization", "Bearer "+c.ApiKey)
	var out struct {
		Object string  `json:"object"` // list
		Data   []Model `json:"data"`
	}
	err := httpjson.DefaultClient.Get(ctx, "https://api.groq.com/openai/v1/models", h, &out)
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
		return errors.New("groq ApiKey is required; get one at " + apiKeyURL)
	}
	if needModel && c.Model == "" {
		return errors.New("a Model is required")
	}
	return nil
}

func (c *Client) post(ctx context.Context, url string, in, out any) error {
	h := make(http.Header)
	h.Add("Authorization", "Bearer "+c.ApiKey)
	// Groq doesn't HTTP POST support compression.
	resp, err := httpjson.DefaultClient.PostRequest(ctx, url, h, in)
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
			if herr.StatusCode == http.StatusUnauthorized {
				return fmt.Errorf("%w: error %s (%s): %s. You can get a new API key at %s", herr, er.Error.Code, er.Error.Type, er.Error.Message, apiKeyURL)
			}
			return fmt.Errorf("%w: error %s (%s): %s", herr, er.Error.Code, er.Error.Type, er.Error.Message)
		}
		return fmt.Errorf("error %s (%s): %s", er.Error.Code, er.Error.Type, er.Error.Message)
	default:
		var herr *httpjson.Error
		if errors.As(err, &herr) {
			slog.WarnContext(ctx, "groq", "url", url, "err", err, "response", string(herr.ResponseBody), "status", herr.StatusCode)
		} else {
			slog.WarnContext(ctx, "groq", "url", url, "err", err)
		}
		return err
	}
}

const apiKeyURL = "https://console.groq.com/keys"

var _ genaiapi.CompletionProvider = &Client{}
