// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package mistral

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"net/http"

	"github.com/maruel/genai/genaiapi"
	"github.com/maruel/httpjson"
)

// Messages.
type chatCompletionRequest struct {
	Model            string             `json:"model"`
	Temperature      float64            `json:"temperature,omitzero"` // [0, 2]
	TopP             float64            `json:"top_p,omitzero"`       // [0, 1]
	MaxTokens        int64              `json:"max_tokens,omitzero"`
	Stream           bool               `json:"stream"`
	Stop             []string           `json:"stop,omitempty"` // keywords to stop completion
	RandomSeed       int64              `json:"random_seed,omitzero"`
	Messages         []genaiapi.Message `json:"messages"`
	ResponseFormat   any                `json:"response_format,omitempty"`   // TODO e.g. json_object with json_schema
	Tools            []any              `json:"tools,omitempty"`             // TODO
	ToolChoices      []any              `json:"tool_choices,omitempty"`      // TODO
	PresencePenalty  float64            `json:"presence_penalty,omitzero"`   // [-2.0, 2.0]
	FrequencyPenalty float64            `json:"frequency_penalty,omitempty"` // [-2.0, 2.0]
	N                int64              `json:"n,omitzero"`                  // Number of choices
	Prediction       any                `json:"prediction,omitempty"`        // TODO
	SafePrompt       bool               `json:"safe_prompt,omitzero"`
}

type chatCompletionsResponse struct {
	ID      string    `json:"id"`
	Object  string    `json:"object"` // chat.completion
	Model   string    `json:"model"`
	Created int64     `json:"created"` // Unix timestamp
	Choices []choices `json:"choices"`
	Usage   struct {
		// QueueTime        float64 `json:"queue_time"`
		PromptTokens int64 `json:"prompt_tokens"`
		// PromptTime       float64 `json:"prompt_time"`
		CompletionTokens int64 `json:"completion_tokens"`
		// CompletionTime   float64 `json:"completion_time"`
		TotalTokens int64 `json:"total_tokens"`
		// TotalTime        float64 `json:"total_time"`
	} `json:"usage"`
}

type choice struct {
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
}

type choices struct {
	// FinishReason is one of "stop", "length", "content_filter" or "tool_calls".
	FinishReason string `json:"finish_reason"`
	Index        int64  `json:"index"`
	Message      choice `json:"message"`
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
	Object  string `json:"object"` // error
	Message struct {
		Detail []struct {
			Msg   string   `json:"msg"`
			Type  string   `json:"type"`
			Loc   []string `json:"loc"`
			Input any      `json:"input"`
			Ctx   any      `json:"ctx"`
			URL   string   `json:"url"`
		} `json:"detail"`
	} `json:"message"`
	Type  string `json:"type"`
	Param string `json:"param"`
	Code  int64  `json:"code"`
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
	if err := c.validate(); err != nil {
		return "", err
	}
	in := chatCompletionRequest{Model: c.Model, Messages: msgs}
	switch v := opts.(type) {
	case *genaiapi.CompletionOptions:
		in.MaxTokens = v.MaxTokens
		in.RandomSeed = v.Seed
		in.Temperature = v.Temperature
	default:
		return "", fmt.Errorf("unsupported options type %T", opts)
	}
	out := chatCompletionsResponse{}
	if err := c.post(ctx, "https://api.mistral.ai/v1/chat/completions", in, &out); err != nil {
		return "", fmt.Errorf("failed to get chat response: %w", err)
	}
	if len(out.Choices) != 1 {
		return "", fmt.Errorf("server returned an unexpected number of choices, expected 1, got %d", len(out.Choices))
	}
	return out.Choices[0].Message.Content, nil
}

func (c *Client) CompletionStream(ctx context.Context, msgs []genaiapi.Message, opts any, words chan<- string) (string, error) {
	if err := c.validate(); err != nil {
		return "", err
	}
	return "", errors.New("not implemented")
}

func (c *Client) CompletionContent(ctx context.Context, msgs []genaiapi.Message, opts any, mime string, content []byte) (string, error) {
	if err := c.validate(); err != nil {
		return "", err
	}
	return "", errors.New("not implemented")
}

func (c *Client) post(ctx context.Context, url string, in, out any) error {
	if c.ApiKey == "" {
		return errors.New("mistral ApiKey is required; get one at " + apiKeyURL)
	}
	h := make(http.Header)
	h.Add("Authorization", "Bearer "+c.ApiKey)
	p := httpjson.DefaultClient
	// Mistral doesn't support any compression. lol.
	p.Compress = ""
	resp, err := p.PostRequest(ctx, url, h, in)
	if err != nil {
		return err
	}
	// This is so cute.
	erAuth := errorResponseAuth{}
	erAPI1 := errorResponseAPI1{}
	erAPI2 := errorResponseAPI2{}
	switch i, err := httpjson.DecodeResponse(resp, out, &erAuth, &erAPI1, &erAPI2); i {
	case 0:
		return nil
	case 1:
		var herr *httpjson.Error
		if errors.As(err, &herr) {
			if herr.StatusCode == http.StatusUnauthorized {
				return fmt.Errorf("http %d: %s. You can get a new API key at %s", herr.StatusCode, erAuth.Message, apiKeyURL)
			}
			return fmt.Errorf("http %d: %s", herr.StatusCode, erAuth.Message)
		}
		return errors.New(erAuth.Message)
	case 2:
		return fmt.Errorf("error %s: %s", erAPI1.Type, erAPI1.Message)
	case 3:
		return fmt.Errorf("error %s/%s: %s at %s", erAPI2.Type, erAPI2.Message.Detail[0].Type, erAPI2.Message.Detail[0].Msg, erAPI2.Message.Detail[0].Loc)
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

func (c *Client) validate() error {
	if c.ApiKey == "" {
		return errors.New("missing API key")
	}
	if c.Model == "" {
		return errors.New("missing model")
	}
	return nil
}

const apiKeyURL = "https://console.mistral.ai/api-keys"

var _ genaiapi.CompletionProvider = &Client{}
