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
	"os"
	"strings"
	"time"

	"github.com/invopop/jsonschema"
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

func (c *CompletionRequest) Init(msgs []genaiapi.Message, opts genaiapi.Validatable) error {
	var errs []error
	if opts != nil {
		if err := opts.Validate(); err != nil {
			errs = append(errs, err)
		} else {
			switch v := opts.(type) {
			case *genaiapi.CompletionOptions:
				c.MaxCompletionTokens = v.MaxTokens
				c.Seed = v.Seed
				c.Temperature = v.Temperature
				c.TopP = v.TopP
				if v.TopK != 0 {
					errs = append(errs, errors.New("groq doesn't support TopK"))
				}
				c.Stop = v.Stop
				if v.ReplyAsJSON {
					c.ResponseFormat.Type = "json_object"
				}
				if v.JSONSchema != nil {
					errs = append(errs, errors.New("groq doesn't support JSONSchema"))
				}
				if len(v.Tools) != 0 {
					// Documentation states max is 128 tools.
					// Let's assume if the user provides tools, they want to use them.
					c.ToolChoice = "required"
					c.Tools = make([]Tool, len(v.Tools))
					for i, t := range v.Tools {
						c.Tools[i].Type = "function"
						c.Tools[i].Function.Name = t.Name
						c.Tools[i].Function.Description = t.Description
						c.Tools[i].Function.Parameters = t.Parameters
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
		c.Messages = make([]Message, len(msgs))
		for i, m := range msgs {
			if err := c.Messages[i].From(m); err != nil {
				errs = append(errs, fmt.Errorf("message %d: %w", i, err))
			}
		}
	}
	return errors.Join(errs...)
}

type Message struct {
	Role       string     `json:"role"`
	Content    []Content  `json:"content,omitzero"`
	Name       string     `json:"name,omitzero"`
	ToolCalls  []ToolCall `json:"tool_calls,omitzero"`
	ToolCallID string     `json:"tool_call_id,omitzero"`
}

func (msg *Message) From(m genaiapi.Message) error {
	switch m.Role {
	case genaiapi.System, genaiapi.User, genaiapi.Assistant:
		msg.Role = string(m.Role)
	default:
		return fmt.Errorf("unsupported role %q", m.Role)
	}
	msg.Content = []Content{{}}
	switch m.Type {
	case genaiapi.Text:
		msg.Content[0].Type = "text"
		msg.Content[0].Text = m.Text
	case genaiapi.Document:
		mimeType, data, err := internal.ParseDocument(&m, 10*1024*1024)
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
		Name        string             `json:"name,omitzero"`
		Description string             `json:"description,omitzero"`
		Parameters  *jsonschema.Schema `json:"parameters,omitzero"`
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
			Role      genaiapi.Role `json:"role"`
			Content   string        `json:"content"`
			ToolCalls []struct {
				ID       string `json:"id"`
				Type     string `json:"type"` // "function"
				Function struct {
					Name      string `json:"name"`
					Arguments string `json:"arguments"` // JSON
				} `json:"function"`
			} `json:"tool_calls"`
		} `json:"message"`
		Logprobs struct{} `json:"logprobs"`
	} `json:"choices"`
	Created Time   `json:"created"`
	ID      string `json:"id"`
	Model   string `json:"model"`
	Object  string `json:"object"` // "chat.completion"
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

func (c *CompletionResponse) ToResult() (genaiapi.CompletionResult, error) {
	out := genaiapi.CompletionResult{}
	out.InputTokens = c.Usage.PromptTokens
	out.OutputTokens = c.Usage.CompletionTokens
	if len(c.Choices) != 1 {
		return out, fmt.Errorf("server returned an unexpected number of choices, expected 1, got %d", len(c.Choices))
	}
	switch c.Choices[0].FinishReason {
	case "tool_calls":
		out.Type = genaiapi.ToolCalls
		out.ToolCalls = make([]genaiapi.ToolCall, len(c.Choices[0].Message.ToolCalls))
		for i, t := range c.Choices[0].Message.ToolCalls {
			out.ToolCalls[i].ID = t.ID
			out.ToolCalls[i].Name = t.Function.Name
			out.ToolCalls[i].Arguments = t.Function.Arguments
		}
	default:
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
	ID                string `json:"id"`
	Object            string `json:"object"`
	Created           Time   `json:"created"`
	Model             string `json:"model"`
	SystemFingerprint string `json:"system_fingerprint"`
	Choices           []struct {
		Index int64 `json:"index"`
		Delta struct {
			Role    string `json:"role"`
			Content string `json:"content"`
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
		Message          string `json:"message"`
		Type             string `json:"type"`
		Code             string `json:"code"`
		FailedGeneration string `json:"failed_generation"`
	} `json:"error"`
}

// Client implements the REST JSON based API.
type Client struct {
	apiKey string
	model  string
}

// New creates a new client to talk to the Groq platform API.
//
// If apiKey is not provided, it tries to load it from the GROQ_API_KEY environment variable.
// If none is found, it returns an error.
// Get your API key at https://console.groq.com/keys
// If no model is provided, only functions that do not require a model, like ListModels, will work.
// To use multiple models, create multiple clients.
// Use one of the model from https://console.groq.com/dashboard/limits or https://console.groq.com/docs/models
func New(apiKey, model string) (*Client, error) {
	if apiKey == "" {
		if apiKey = os.Getenv("GROQ_API_KEY"); apiKey == "" {
			return nil, errors.New("groq API key is required; get one at " + apiKeyURL)
		}
	}
	return &Client{apiKey: apiKey, model: model}, nil
}

func (c *Client) Completion(ctx context.Context, msgs []genaiapi.Message, opts genaiapi.Validatable) (genaiapi.CompletionResult, error) {
	// https://console.groq.com/docs/api-reference#chat-create
	in := CompletionRequest{Model: c.model}
	if err := in.Init(msgs, opts); err != nil {
		return genaiapi.CompletionResult{}, err
	}
	rpcout := CompletionResponse{}
	if err := c.CompletionRaw(ctx, &in, &rpcout); err != nil {
		return genaiapi.CompletionResult{}, fmt.Errorf("failed to get chat response: %w", err)
	}
	return rpcout.ToResult()
}

func (c *Client) CompletionRaw(ctx context.Context, in *CompletionRequest, out *CompletionResponse) error {
	if err := c.validate(); err != nil {
		return err
	}
	in.Stream = false
	return c.post(ctx, "https://api.groq.com/openai/v1/chat/completions", in, out)
}

func (c *Client) CompletionStream(ctx context.Context, msgs []genaiapi.Message, opts genaiapi.Validatable, chunks chan<- genaiapi.MessageChunk) error {
	in := CompletionRequest{Model: c.model}
	if err := in.Init(msgs, opts); err != nil {
		return err
	}
	ch := make(chan CompletionStreamChunkResponse)
	end := make(chan error)
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()
	go func() {
		lastRole := genaiapi.System
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
	// Groq doesn't HTTP POST support compression.
	resp, err := httpjson.DefaultClient.PostRequest(ctx, "https://api.groq.com/openai/v1/chat/completions", h, in)
	if err != nil {
		return fmt.Errorf("failed to get server response: %w", err)
	}
	defer resp.Body.Close()
	for r := bufio.NewReader(resp.Body); ; {
		line, err := r.ReadBytes('\n')
		if line = bytes.TrimSpace(line); err == io.EOF {
			if len(line) == 0 {
				return nil
			}
		} else if err != nil {
			return fmt.Errorf("failed to get server response: %w", err)
		}
		if len(line) != 0 {
			if err := parseStreamLine(line, out); err != nil {
				return err
			}
		}
	}
}

func parseStreamLine(line []byte, out chan<- CompletionStreamChunkResponse) error {
	const prefix = "data: "
	if !bytes.HasPrefix(line, []byte(prefix)) {
		d := json.NewDecoder(bytes.NewReader(line))
		d.DisallowUnknownFields()
		d.UseNumber()
		er := errorResponse{}
		if err := d.Decode(&er); err != nil {
			return fmt.Errorf("unexpected line. expected \"data: \", got %q", line)
		}
		suffix := ""
		if er.Error.FailedGeneration != "" {
			suffix = fmt.Sprintf(" Failed generation: %q", er.Error.FailedGeneration)
		}
		return fmt.Errorf("server error %s (%s): %s%s", er.Error.Code, er.Error.Type, er.Error.Message, suffix)
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
	h := make(http.Header)
	h.Add("Authorization", "Bearer "+c.apiKey)
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

func (c *Client) validate() error {
	if c.model == "" {
		return errors.New("a model is required")
	}
	return nil
}

func (c *Client) post(ctx context.Context, url string, in, out any) error {
	h := make(http.Header)
	h.Add("Authorization", "Bearer "+c.apiKey)
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
		suffix := ""
		if er.Error.FailedGeneration != "" {
			suffix = fmt.Sprintf(" Failed generation: %q", er.Error.FailedGeneration)
		}
		var herr *httpjson.Error
		if errors.As(err, &herr) {
			if herr.StatusCode == http.StatusUnauthorized {
				return fmt.Errorf("%w: error %s (%s): %s%s. You can get a new API key at %s", herr, er.Error.Code, er.Error.Type, er.Error.Message, suffix, apiKeyURL)
			}
			return fmt.Errorf("%w: error %s (%s): %s%s", herr, er.Error.Code, er.Error.Type, er.Error.Message, suffix)
		}
		return fmt.Errorf("error %s (%s): %s%s", er.Error.Code, er.Error.Type, er.Error.Message, suffix)
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

var (
	_ genaiapi.CompletionProvider = &Client{}
	_ genaiapi.ModelProvider      = &Client{}
)
