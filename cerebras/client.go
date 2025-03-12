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
	"strings"
	"time"

	"github.com/maruel/genai/genaiapi"
	"github.com/maruel/httpjson"
)

// https://inference-docs.cerebras.ai/api-reference/chat-completions
type CompletionRequest struct {
	Model               string         `json:"model"`
	Messages            []Message      `json:"messages"`
	MaxCompletionTokens int64          `json:"max_completion_tokens,omitzero"`
	ResponseFormat      ResponseFormat `json:"response_format,omitzero"`
	Seed                int64          `json:"seed,omitzero"`
	Stop                []string       `json:"stop,omitzero"`
	Stream              bool           `json:"stream,omitzero"`
	Temperature         float64        `json:"temperature,omitzero"`
	TopP                float64        `json:"top_p,omitzero"` // [0, 1.0]
	ToolChoice          string         `json:"tool_choice,omitzero"`
	Tools               Tools          `json:"tools,omitzero"`
	User                string         `json:"user,omitzero"`
	Logprobs            bool           `json:"logprobs,omitzero"`
	TopLogprobs         int64          `json:"top_logprobs,omitzero"` // [0, 20]
}

func (c *CompletionRequest) fromOpts(opts any) error {
	if opts != nil {
		switch v := opts.(type) {
		case *genaiapi.CompletionOptions:
			c.MaxCompletionTokens = v.MaxTokens
			c.Temperature = v.Temperature
			if v.Seed != 0 {
				return errors.New("cerebras doesn't support seed")
			}
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
		case genaiapi.User, genaiapi.Assistant:
		default:
			return fmt.Errorf("message %d: unsupported role %q", i, m.Role)
		}
		switch m.Type {
		case genaiapi.Text:
		default:
			return fmt.Errorf("message %d: unsupported content type %s", i, m.Type)
		}
		c.Messages[i].Role = string(m.Role)
		c.Messages[i].Content = m.Text
	}
	return nil
}

type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// https://inference-docs.cerebras.ai/capabilities/structured-outputs
type ResponseFormat struct {
	JSONSchema struct {
		Name   string     `json:"name,omitzero"`
		Strict bool       `json:"strict,omitzero"`
		Schema JSONSchema `json:"schema,omitzero"`
	} `json:"json_schema,omitzero"`
}

type Tools struct {
	Function struct {
		Description string     `json:"description"`
		Name        string     `json:"name"`
		Parameters  JSONSchema `json:"parameters"`
		Type        string     `json:"type"` // function
	} `json:"function"`
}

// TODO
type JSONSchema any

type CompletionResponse struct {
	ID                string `json:"id"`
	Model             string `json:"model"`
	Object            string `json:"object"`
	SystemFingerprint string `json:"system_fingerprint"`
	Created           Time   `json:"created"`
	Choices           []struct {
		Index        int64   `json:"index"`
		FinishReason string  `json:"finish_reason"`
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

type errorResponse struct {
	Detail string `json:"detail"`
}

type Client struct {
	// ApiKey can be retrieved from https://cloud.cerebras.ai/platform/
	ApiKey string
	// Model to use.
	Model string
}

func (c *Client) Completion(ctx context.Context, msgs []genaiapi.Message, opts any) (string, error) {
	// https://inference-docs.cerebras.ai/api-reference/chat-completions
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
		return "", errors.New("expected 1 choice")
	}
	return out.Choices[0].Message.Content, nil
}

func (c *Client) CompletionRaw(ctx context.Context, in *CompletionRequest, out *CompletionResponse) error {
	if err := c.validate(true); err != nil {
		return err
	}
	return c.post(ctx, "https://api.cerebras.ai/v1/chat/completions", in, out)
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
	resp, err := httpjson.DefaultClient.PostRequest(ctx, "https://api.cerebras.ai/v1/chat/completions", h, in)
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
		const dataPrefix = "data: "
		if !bytes.HasPrefix(line, []byte(dataPrefix)) {
			d := json.NewDecoder(bytes.NewReader(line))
			d.DisallowUnknownFields()
			d.UseNumber()
			er := errorResponse{}
			if err = d.Decode(&er); err != nil {
				return fmt.Errorf("unexpected line. expected \"data: \", got %q", line)
			}
			return fmt.Errorf("server error: %s", er.Detail)
		}
		suffix := string(line[len(dataPrefix):])
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

func (c *Client) validate(needModel bool) error {
	if c.ApiKey == "" {
		return errors.New("cerebras ApiKey is required; get one at " + apiKeyURL)
	}
	if needModel && c.Model == "" {
		return errors.New("a Model is required")
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
	if err := c.validate(false); err != nil {
		return nil, err
	}
	h := make(http.Header)
	h.Add("Authorization", "Bearer "+c.ApiKey)
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
	h.Add("Authorization", "Bearer "+c.ApiKey)
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
				return fmt.Errorf("%w: error: %s. You can get a new API key at %s", herr, er.Detail, apiKeyURL)
			}
			return fmt.Errorf("%w: error: %s", herr, er.Detail)
		}
		return fmt.Errorf("error: %s", er.Detail)
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

var _ genaiapi.CompletionProvider = &Client{}
