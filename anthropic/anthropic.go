// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Package groq implements a client for the Anthropic API, to use Claude.
//
// It is described at
// https://docs.anthropic.com/en/api/messages
package anthropic

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

type Message struct {
	Role    genaiapi.Role `json:"role"`
	Content string        `json:"content"`
}

// https://docs.anthropic.com/en/api/messages
type CompletionRequest struct {
	Model         string    `json:"model,omitempty"`
	MaxToks       int64     `json:"max_tokens"`
	Messages      []Message `json:"messages"`
	Metadata      any       `json:"metadata,omitempty"`
	StopSequences []string  `json:"stop_sequences,omitempty"`
	Stream        bool      `json:"stream,omitempty"`
	System        string    `json:"system,omitzero"`
	Temperature   float64   `json:"temperature,omitzero"` // [0, 1]
	Thinking      any       `json:"thinking,omitempty"`
	ToolChoice    any       `json:"tool_choice,omitempty"`
	Tools         any       `json:"tools,omitempty"`
	TopK          int64     `json:"top_k,omitzero"` // [1, ]
	TopP          float64   `json:"top_p,omitzero"` // [0, 1]
}

func (c *CompletionRequest) fromOpts(opts any) error {
	switch v := opts.(type) {
	case *genaiapi.CompletionOptions:
		c.MaxToks = v.MaxTokens
		c.Temperature = v.Temperature
		if v.Seed != 0 {
			return errors.New("seed is not supported")
		}
	default:
		return fmt.Errorf("unsupported options type %T", opts)
	}
	if c.MaxToks == 0 {
		// TODO: Query the model. Anthropic requires a value! Use the lowest common denominator for now.
		c.MaxToks = 4096
	}
	return nil
}

func (c *CompletionRequest) fromMsgs(msgs []genaiapi.Message) error {
	for _, m := range msgs {
		switch m.Role {
		case genaiapi.System:
			if c.System != "" {
				return errors.New("only one system message is supported")
			}
			c.System = m.Content
		case genaiapi.User, genaiapi.Assistant:
			c.Messages = append(c.Messages, Message{Role: m.Role, Content: m.Content})
		default:
			return fmt.Errorf("unsupported role %v", m.Role)
		}
	}
	return nil
}

type CompletionResponse struct {
	Content []struct {
		Type      string `json:"type"`
		Text      string `json:"text"`
		Citations any    `json:"citations,omitempty"`
		// TODO Add rest.
	} `json:"content"`
	ID           string `json:"id"`
	Model        string `json:"model"`
	Role         string `json:"role"` // Always "assistant"
	StopReason   string `json:"stop_reason"`
	StopSequence string `json:"stop_sequence"`
	Type         string `json:"type"`
	Usage        struct {
		InputTokens              int64 `json:"input_tokens"`
		OutputTokens             int64 `json:"output_tokens"`
		CacheCreationInputTokens int64 `json:"cache_creation_input_tokens"`
		CacheReadInputTokens     int64 `json:"cache_read_input_tokens"`
	} `json:"usage"`
}

type CompletionStreamChunkResponse struct {
	// "message_start", "content_block_start", "content_block_delta"
	Type string `json:"type"`
	// "message_start"
	Message struct {
		ID           string   `json:"id"`
		Type         string   `json:"type"`
		Role         string   `json:"role"`
		Model        string   `json:"model"`
		Content      []string `json:"content"`
		StopReason   string   `json:"stop_reason"`
		StopSequence string   `json:"stop_sequence"`
		Usage        struct {
			InputTokens              int64 `json:"input_tokens"`
			OutputTokens             int64 `json:"output_tokens"`
			CacheCreationInputTokens int64 `json:"cache_creation_input_tokens"`
			CacheReadInputTokens     int64 `json:"cache_read_input_tokens"`
		} `json:"usage"`
	} `json:"message"`

	Index int64 `json:"index"`
	// "content_block_start"
	ContentBlock struct {
		Type string `json:"type"` // text
		Text string `json:"text"`
	} `json:"content_block"`
	// "content_block_delta"
	Delta struct {
		Type         string `json:"type"` // text_delta
		Text         string `json:"text"`
		StopReason   string `json:"stop_reason"` // end_turn
		StopSequence string `json:"stop_sequence"`
	} `json:"delta"`
	Usage struct {
		OutputTokens int64 `json:"output_tokens"`
	} `json:"usage"`
}

//

type errorResponse struct {
	Type  string `json:"type"`
	Error struct {
		Type    string `json:"type"`
		Message string `json:"message"`
	} `json:"error"`
}

type Client struct {
	// ApiKey retrieved from https://console.anthropic.com/settings/keys
	ApiKey string
	// One of the model from https://docs.anthropic.com/en/docs/about-claude/models/all-models
	Model string
}

func (c *Client) Completion(ctx context.Context, msgs []genaiapi.Message, opts any) (string, error) {
	// https://docs.anthropic.com/en/api/messages
	in := CompletionRequest{Model: c.Model, Messages: make([]Message, 0, len(msgs))}
	if err := in.fromOpts(opts); err != nil {
		return "", err
	}
	if err := in.fromMsgs(msgs); err != nil {
		return "", err
	}
	out := CompletionResponse{}
	if err := c.CompletionRaw(ctx, &in, &out); err != nil {
		return "", err
	}
	return out.Content[0].Text, nil
}

func (c *Client) CompletionRaw(ctx context.Context, in *CompletionRequest, out *CompletionResponse) error {
	if err := c.validate(true); err != nil {
		return err
	}
	return c.post(ctx, "https://api.anthropic.com/v1/messages", in, out)
}

func (c *Client) CompletionStream(ctx context.Context, msgs []genaiapi.Message, opts any, words chan<- string) error {
	in := CompletionRequest{Model: c.Model, Messages: make([]Message, 0, len(msgs)), Stream: true}
	if err := in.fromOpts(opts); err != nil {
		return err
	}
	if err := in.fromMsgs(msgs); err != nil {
		return err
	}
	ch := make(chan CompletionStreamChunkResponse)
	end := make(chan struct{})
	start := time.Now()
	go func() {
		for msg := range ch {
			word := ""
			switch msg.Type {
			case "message_start":
			case "content_block_start":
				word = msg.ContentBlock.Text
			case "content_block_delta":
				word = msg.Delta.Text
			}
			slog.DebugContext(ctx, "anthropic", "word", word, "duration", time.Since(start).Round(time.Millisecond))
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
	h.Set("x-api-key", c.ApiKey)
	h.Set("anthropic-version", "2023-06-01")
	// Anthropic doesn't HTTP POST support compression.
	resp, err := httpjson.DefaultClient.PostRequest(ctx, "https://api.anthropic.com/v1/messages", h, in)
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
		// https://docs.anthropic.com/en/api/messages-streaming
		// and
		// https://developer.mozilla.org/en-US/docs/Web/API/Server-sent%5Fevents/Using%5Fserver-sent%5Fevents
		const dataPrefix = "data: "
		switch {
		case bytes.HasPrefix(line, []byte(dataPrefix)):
			suffix := string(line[len(dataPrefix):])
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
		case bytes.HasPrefix(line, []byte("event:")):
			// Ignore for now.
		default:
			d := json.NewDecoder(bytes.NewReader(line))
			d.DisallowUnknownFields()
			d.UseNumber()
			er := errorResponse{}
			if err = d.Decode(&er); err == nil {
				return fmt.Errorf("error %s: %s", er.Error.Type, er.Error.Message)
			}
			return fmt.Errorf("unexpected line. expected \"data: \", got %q", line)
		}
	}
}

func (c *Client) CompletionContent(ctx context.Context, msgs []genaiapi.Message, opts any, mime string, context []byte) (string, error) {
	return "", errors.New("not implemented")
}

type Model struct {
	CreatedAt   time.Time `json:"created_at"`
	DisplayName string    `json:"display_name"`
	ID          string    `json:"id"`
	Type        string    `json:"type"`
}

func (m *Model) GetID() string {
	return m.ID
}

func (m *Model) String() string {
	return fmt.Sprintf("%s: %s (%s)", m.ID, m.DisplayName, m.CreatedAt.Format("2006-01-02"))
}

func (m *Model) Context() int64 {
	return 0
}

func (c *Client) ListModels(ctx context.Context) ([]genaiapi.Model, error) {
	if err := c.validate(false); err != nil {
		return nil, err
	}
	// https://docs.anthropic.com/en/api/models-list
	h := make(http.Header)
	h.Set("x-api-key", c.ApiKey)
	h.Set("anthropic-version", "2023-06-01")
	var out struct {
		Data    []Model `json:"data"`
		FirstID string  `json:"first_id"`
		HasMore bool    `json:"has_more"`
		LastID  string  `json:"last_id"`
	}
	err := httpjson.DefaultClient.Get(ctx, "https://api.anthropic.com/v1/models?limit=1000", h, &out)
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
		return errors.New("anthropic ApiKey is required; get one at " + apiKeyURL)
	}
	if needModel && c.Model == "" {
		return errors.New("a Model is required")
	}
	return nil
}

func (c *Client) post(ctx context.Context, url string, in, out any) error {
	h := make(http.Header)
	h.Set("x-api-key", c.ApiKey)
	h.Set("anthropic-version", "2023-06-01")
	// Anthropic doesn't HTTP POST support compression.
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
				return fmt.Errorf("%w: error %s: %s. You can get a new API key at %s", herr, er.Error.Type, er.Error.Message, apiKeyURL)
			}
			return fmt.Errorf("%w: error %s: %s", herr, er.Error.Type, er.Error.Message)
		}
		return fmt.Errorf("error %s: %s", er.Error.Type, er.Error.Message)
	default:
		var herr *httpjson.Error
		if errors.As(err, &herr) {
			slog.WarnContext(ctx, "anthropic", "url", url, "err", err, "response", string(herr.ResponseBody), "status", herr.StatusCode)
		} else {
			slog.WarnContext(ctx, "anthropic", "url", url, "err", err)
		}
		return err
	}
}

const apiKeyURL = "https://console.anthropic.com/settings/keys"

var _ genaiapi.CompletionProvider = &Client{}
