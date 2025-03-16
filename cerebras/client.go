// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Package cerebras implements a client for the Cerebras API.
//
// It is described at https://inference-docs.cerebras.ai/api-reference/
package cerebras

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
	"strings"
	"time"

	"github.com/invopop/jsonschema"
	"github.com/maruel/genai/genaiapi"
	"github.com/maruel/httpjson"
)

// https://inference-docs.cerebras.ai/api-reference/chat-completions
type CompletionRequest struct {
	Model               string    `json:"model"`
	Messages            []Message `json:"messages"`
	MaxCompletionTokens int64     `json:"max_completion_tokens,omitzero"`
	ResponseFormat      struct {
		// https://inference-docs.cerebras.ai/capabilities/structured-outputs
		Type       string `json:"type"` // "json_object", "json_schema"
		JSONSchema struct {
			Name   string             `json:"name"`
			Schema *jsonschema.Schema `json:"schema"`
			Strict bool               `json:"strict"`
		} `json:"json_schema,omitzero"`
	} `json:"response_format,omitzero"`
	Seed        int64    `json:"seed,omitzero"`
	Stop        []string `json:"stop,omitzero"`
	Stream      bool     `json:"stream,omitzero"`
	Temperature float64  `json:"temperature,omitzero"`
	TopP        float64  `json:"top_p,omitzero"`       // [0, 1.0]
	ToolChoice  string   `json:"tool_choice,omitzero"` // "none", "auto", "required" or a struct.
	Tools       []Tool   `json:"tools,omitzero"`
	User        string   `json:"user,omitzero"`
	Logprobs    bool     `json:"logprobs,omitzero"`
	TopLogprobs int64    `json:"top_logprobs,omitzero"` // [0, 20]
}

func (c *CompletionRequest) Init(msgs []genaiapi.Message, opts any) error {
	var errs []error
	if opts != nil {
		switch v := opts.(type) {
		case *genaiapi.CompletionOptions:
			if err := v.Validate(); err != nil {
				errs = append(errs, err)
			} else {
				c.MaxCompletionTokens = v.MaxTokens
				c.Temperature = v.Temperature
				c.Seed = v.Seed
				c.TopP = v.TopP
				if v.TopK != 0 {
					errs = append(errs, errors.New("cerebras does not support TopK"))
				}
				c.Stop = v.Stop
				if v.ReplyAsJSON {
					c.ResponseFormat.Type = "json_object"
				}
				if v.JSONSchema != nil {
					c.ResponseFormat.Type = "json_schema"
					// Cerebras will fail if additonalProperties is present, even if false.
					c.ResponseFormat.JSONSchema.Schema = v.JSONSchema
					c.ResponseFormat.JSONSchema.Strict = true
				}
				if len(v.Tools) != 0 {
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
			}
		default:
			errs = append(errs, fmt.Errorf("unsupported options type %T", opts))
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
	Role      string `json:"role,omitzero"`
	Content   string `json:"content,omitzero"`
	ToolCalls []struct {
		Type     string `json:"type,omitzero"` // "function"
		ID       string `json:"id,omitzero"`
		Function struct {
			Name      string `json:"name,omitzero"`
			Arguments string `json:"arguments,omitzero"`
		} `json:"function,omitzero"`
	} `json:"tool_calls,omitzero"`
}

func (msg *Message) From(m genaiapi.Message) error {
	switch m.Role {
	case genaiapi.System, genaiapi.User, genaiapi.Assistant:
		msg.Role = string(m.Role)
	default:
		return fmt.Errorf("unsupported role %q", m.Role)
	}
	switch m.Type {
	case genaiapi.Text:
		msg.Content = m.Text
	default:
		return fmt.Errorf("unsupported content type %s", m.Type)
	}
	return nil
}

type Tool struct {
	Type     string `json:"type"` // function
	Function struct {
		Name        string             `json:"name"`
		Description string             `json:"description"`
		Parameters  *jsonschema.Schema `json:"parameters"`
	} `json:"function"`
}

type CompletionResponse struct {
	ID                string `json:"id"`
	Model             string `json:"model"`
	Object            string `json:"object"` // "chat.completion"
	SystemFingerprint string `json:"system_fingerprint"`
	Created           Time   `json:"created"`
	Choices           []struct {
		Index        int64   `json:"index"`
		FinishReason string  `json:"finish_reason"` // "tool_calls"
		Message      Message `json:"message"`
	} `json:"choices"`
	Usage struct {
		PromptTokens     int64 `json:"prompt_tokens"`
		CompletionTokens int64 `json:"completion_tokens"`
		TotalTokens      int64 `json:"total_tokens"`
	} `json:"usage"`
	TimeInfo struct {
		QueueTime      float64 `json:"queue_time"`
		PromptTime     float64 `json:"prompt_time"`
		CompletionTime float64 `json:"completion_time"`
		TotalTime      float64 `json:"total_time"`
		Created        Time    `json:"created"`
	} `json:"time_info"`
}

func (c *CompletionResponse) ToResult() (genaiapi.CompletionResult, error) {
	out := genaiapi.CompletionResult{}
	out.InputTokens = c.Usage.PromptTokens
	out.OutputTokens = c.Usage.CompletionTokens
	if len(c.Choices) != 1 {
		return out, fmt.Errorf("expected 1 choice, got %#v", c.Choices)
	}
	if len(c.Choices[0].Message.ToolCalls) != 0 {
		out.Role = genaiapi.Assistant
		out.Type = genaiapi.ToolCalls
		out.ToolCalls = make([]genaiapi.ToolCall, len(c.Choices[0].Message.ToolCalls))
		for i, t := range c.Choices[0].Message.ToolCalls {
			out.ToolCalls[i].ID = t.ID
			out.ToolCalls[i].Name = t.Function.Name
			out.ToolCalls[i].Arguments = t.Function.Arguments
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
	ID                string `json:"id"`
	Model             string `json:"model"`
	Object            string `json:"object"`
	SystemFingerprint string `json:"system_fingerprint"`
	Created           Time   `json:"created"`
	Choices           []struct {
		Delta struct {
			Role    string `json:"role"`
			Content string `json:"content"`
		} `json:"delta"`
		Index        int64  `json:"index"`
		FinishReason string `json:"finish_reason"`
	} `json:"choices"`
	Usage struct {
		PromptTokens     int64 `json:"prompt_tokens"`
		CompletionTokens int64 `json:"completion_tokens"`
		TotalTokens      int64 `json:"total_tokens"`
	} `json:"usage"`
	TimeInfo struct {
		QueueTime      float64 `json:"queue_time"`
		PromptTime     float64 `json:"prompt_time"`
		CompletionTime float64 `json:"completion_time"`
		TotalTime      float64 `json:"total_time"`
		Created        Time    `json:"created"`
	} `json:"time_info"`
}

type Time int64

func (t *Time) AsTime() time.Time {
	return time.Unix(int64(*t), 0)
}

//

type errorResponse1 struct {
	Detail string `json:"detail"`
}

type errorResponse2 struct {
	Message string `json:"message"`
	Type    string `json:"type"`
	Param   string `json:"param"`
	Code    string `json:"code"`
}

// Client implements the REST JSON based API.
type Client struct {
	apiKey string
	model  string
}

// New creates a new client to talk to the Cerebras platform API.
//
// If apiKey is not provided, it tries to load it from the CEREBRAS_API_KEY environment variable.
// If none is found, it returns an error.
// Get an API key at http://cloud.cerebras.ai/
// If no model is provided, only functions that do not require a model, like ListModels, will work.
// To use multiple models, create multiple clients.
// Use one of the model from https://cerebras.ai/inference
func New(apiKey, model string) (*Client, error) {
	if apiKey == "" {
		if apiKey = os.Getenv("CEREBRAS_API_KEY"); apiKey == "" {
			return nil, errors.New("cerebras API key is required; get one at " + apiKeyURL)
		}
	}
	return &Client{apiKey: apiKey, model: model}, nil
}

func (c *Client) Completion(ctx context.Context, msgs []genaiapi.Message, opts any) (genaiapi.CompletionResult, error) {
	// https://inference-docs.cerebras.ai/api-reference/chat-completions
	rpcin := CompletionRequest{Model: c.model}
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
	return c.post(ctx, "https://api.cerebras.ai/v1/chat/completions", in, out)
}

func (c *Client) CompletionStream(ctx context.Context, msgs []genaiapi.Message, opts any, chunks chan<- genaiapi.MessageChunk) error {
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
	resp, err := httpjson.DefaultClient.PostRequest(ctx, "https://api.cerebras.ai/v1/chat/completions", h, in)
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
	const dataPrefix = "data: "
	if !bytes.HasPrefix(line, []byte(dataPrefix)) {
		d := json.NewDecoder(bytes.NewReader(line))
		d.DisallowUnknownFields()
		d.UseNumber()
		er1 := errorResponse1{}
		if err := d.Decode(&er1); err != nil {
			return fmt.Errorf("unexpected line. expected \"data: \", got %q", line)
		}
		return fmt.Errorf("server error: %s", er1.Detail)
	}
	suffix := string(line[len(dataPrefix):])
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

func (c *Client) validate() error {
	if c.model == "" {
		return errors.New("a model is required")
	}
	return nil
}

type Model struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Created Time   `json:"created"`
	OwnedBy string `json:"owned_by"`
}

func (m *Model) GetID() string {
	return m.ID
}

func (m *Model) String() string {
	return fmt.Sprintf("%s (%s)", m.ID, m.Created.AsTime().Format("2006-01-02"))
}

func (m *Model) Context() int64 {
	return 0
}

func (c *Client) ListModels(ctx context.Context) ([]genaiapi.Model, error) {
	// https://inference-docs.cerebras.ai/api-reference/models
	h := make(http.Header)
	h.Add("Authorization", "Bearer "+c.apiKey)
	var out struct {
		Object string  `json:"object"`
		Data   []Model `json:"data"`
	}
	err := httpjson.DefaultClient.Get(ctx, "https://api.cerebras.ai/v1/models", h, &out)
	if err != nil {
		return nil, err
	}
	models := make([]genaiapi.Model, len(out.Data))
	for i := range out.Data {
		models[i] = &out.Data[i]
	}
	return models, err
}

func (c *Client) post(ctx context.Context, url string, in, out any) error {
	h := make(http.Header)
	h.Add("Authorization", "Bearer "+c.apiKey)
	resp, err := httpjson.DefaultClient.PostRequest(ctx, url, h, in)
	if err != nil {
		return err
	}
	er1 := errorResponse1{}
	er2 := errorResponse2{}
	switch i, err := httpjson.DecodeResponse(resp, out, &er1, &er2); i {
	case 0:
		return nil
	case 1:
		var herr *httpjson.Error
		if errors.As(err, &herr) {
			if herr.StatusCode == http.StatusUnauthorized {
				return fmt.Errorf("%w: error: %s. You can get a new API key at %s", herr, er1.Detail, apiKeyURL)
			}
			return fmt.Errorf("%w: error: %s", herr, er1.Detail)
		}
		return fmt.Errorf("error: %s", er1.Detail)
	case 2:
		var herr *httpjson.Error
		if errors.As(err, &herr) {
			if herr.StatusCode == http.StatusUnauthorized {
				return fmt.Errorf("%w: error: %s/%s/%s: %s. You can get a new API key at %s", herr, er2.Type, er2.Param, er2.Code, er2.Message, apiKeyURL)
			}
			return fmt.Errorf("%w: error: %s/%s/%s: %s", herr, er2.Type, er2.Param, er2.Code, er2.Message)
		}
		return fmt.Errorf("error: %s/%s/%s: %s", er2.Type, er2.Param, er2.Code, er2.Message)
	default:
		var herr *httpjson.Error
		if errors.As(err, &herr) {
			slog.WarnContext(ctx, "cerebras", "url", url, "err", err, "response", string(herr.ResponseBody), "status", herr.StatusCode)
		} else {
			slog.WarnContext(ctx, "cerebras", "url", url, "err", err)
		}
		return err
	}
}

const apiKeyURL = "https://cloud.cerebras.ai/platform/"

var (
	_ genaiapi.CompletionProvider = &Client{}
	_ genaiapi.ModelProvider      = &Client{}
)
