// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package anthropic

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"net/http"

	"github.com/maruel/genai/genaiapi"
	"github.com/maruel/httpjson"
)

type message struct {
	Role    genaiapi.Role `json:"role"`
	Content string        `json:"content"`
}

// https://docs.anthropic.com/en/api/messages
type messagesRequest struct {
	Model         string    `json:"model,omitempty"`
	MaxToks       int       `json:"max_tokens"`
	Messages      []message `json:"messages"`
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

type messageResponse struct {
	Type      string `json:"type"`
	Text      string `json:"text"`
	Citations any    `json:"citations,omitempty"`
	// TODO Add rest.
}

type messagesResponse struct {
	Content      []messageResponse `json:"content"`
	ID           string            `json:"id"`
	Model        string            `json:"model"`
	Role         string            `json:"role"` // Always "assistant"
	StopReason   string            `json:"stop_reason"`
	StopSequence string            `json:"stop_sequence"`
	Type         string            `json:"type"`
	Usage        struct {
		InputTokens              int `json:"input_tokens"`
		OutputTokens             int `json:"output_tokens"`
		CacheCreationInputTokens int `json:"cache_creation_input_tokens"`
		CacheReadInputTokens     int `json:"cache_read_input_tokens"`
	} `json:"usage"`
}

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

func (c *Client) Completion(ctx context.Context, msgs []genaiapi.Message, maxtoks, seed int, temperature float64) (string, error) {
	// https://docs.anthropic.com/en/api/messages
	in := messagesRequest{
		Model:       c.Model,
		Messages:    make([]message, 0, len(msgs)),
		MaxToks:     maxtoks,
		Temperature: temperature,
	}
	if in.MaxToks == 0 {
		// TODO: Query the model?
		in.MaxToks = 8192
	}
	for _, m := range msgs {
		switch m.Role {
		case genaiapi.System:
			if in.System != "" {
				return "", errors.New("only one system message is supported")
			}
			in.System = m.Content
		case genaiapi.User, genaiapi.Assistant:
			in.Messages = append(in.Messages, message{Role: m.Role, Content: m.Content})
		default:
			return "", fmt.Errorf("unsupported role %v", m.Role)
		}
	}
	out := messagesResponse{}
	if err := c.post(ctx, "https://api.anthropic.com/v1/messages", &in, &out); err != nil {
		return "", err
	}
	return out.Content[0].Text, nil
}

func (c *Client) CompletionStream(ctx context.Context, msgs []genaiapi.Message, maxtoks, seed int, temperature float64, words chan<- string) (string, error) {
	return "", errors.New("not implemented")
}

func (c *Client) CompletionContent(ctx context.Context, msgs []genaiapi.Message, maxtoks, seed int, temperature float64, mime string, context []byte) (string, error) {
	return "", errors.New("not implemented")
}

func (c *Client) post(ctx context.Context, url string, in, out any) error {
	p := httpjson.DefaultClient
	// Anthropic doesn't support compression. lol.
	p.Compress = ""
	h := make(http.Header)
	h.Set("x-api-key", c.ApiKey)
	h.Set("anthropic-version", "2023-06-01")
	resp, err := p.PostRequest(ctx, url, h, in)
	if err != nil {
		return err
	}
	er := errorResponse{}
	switch i, err := httpjson.DecodeResponse(resp, out, &er); i {
	case 0:
		return nil
	case 1:
		return fmt.Errorf("error %s: %s", er.Error.Type, er.Error.Message)
	default:
		var herr *httpjson.Error
		if errors.As(err, &herr) {
			slog.WarnContext(ctx, "anthropic", "url", url, "err", err, "response", string(herr.ResponseBody))
		} else {
			slog.WarnContext(ctx, "anthropic", "url", url, "err", err)
		}
		return err
	}
}

var _ genaiapi.ChatProvider = &Client{}
