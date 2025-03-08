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

// https://docs.anthropic.com/en/api/messages

type message struct {
	Role    genaiapi.Role `json:"role"`
	Content string        `json:"content"`
}

type messagesRequest struct {
	Model         string    `json:"model,omitempty"`
	MaxToks       int64     `json:"max_tokens"`
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
		InputTokens              int64 `json:"input_tokens"`
		OutputTokens             int64 `json:"output_tokens"`
		CacheCreationInputTokens int64 `json:"cache_creation_input_tokens"`
		CacheReadInputTokens     int64 `json:"cache_read_input_tokens"`
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
	in := messagesRequest{Model: c.Model, Messages: make([]message, 0, len(msgs))}
	switch v := opts.(type) {
	case *genaiapi.CompletionOptions:
		in.MaxToks = v.MaxTokens
		in.Temperature = v.Temperature
		if v.Seed != 0 {
			return "", errors.New("seed is not supported")
		}
	default:
		return "", fmt.Errorf("unsupported options type %T", opts)
	}
	if in.MaxToks == 0 {
		// TODO: Query the model. Anthropic requires a value!
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

func (c *Client) CompletionStream(ctx context.Context, msgs []genaiapi.Message, opts any, words chan<- string) (string, error) {
	return "", errors.New("not implemented")
}

func (c *Client) CompletionContent(ctx context.Context, msgs []genaiapi.Message, opts any, mime string, context []byte) (string, error) {
	return "", errors.New("not implemented")
}

func (c *Client) post(ctx context.Context, url string, in, out any) error {
	if c.ApiKey == "" {
		return errors.New("anthropic ApiKey is required; get one at " + apiKeyURL)
	}
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
		var herr *httpjson.Error
		if errors.As(err, &herr) {
			if herr.StatusCode == http.StatusUnauthorized {
				return fmt.Errorf("http %d: error %s: %s. You can get a new API key at %s", herr.StatusCode, er.Error.Type, er.Error.Message, apiKeyURL)
			}
			return fmt.Errorf("http %d: error %s: %s", herr.StatusCode, er.Error.Type, er.Error.Message)
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
