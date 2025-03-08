// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package groq

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"net/http"

	"github.com/maruel/genai/genaiapi"
	"github.com/maruel/httpjson"
)

// https://console.groq.com/docs/api-reference#chat-create

type chatCompletionRequest struct {
	FrequencyPenalty    float64            `json:"frequency_penalty,omitempty"` // [-2.0, 2.0]
	MaxCompletionTokens int64              `json:"max_completion_tokens,omitzero"`
	Messages            []genaiapi.Message `json:"messages"`
	Model               string             `json:"model"`
	ParallelToolCalls   bool               `json:"parallel_tool_calls,omitzero"`
	PresencePenalty     float64            `json:"presence_penalty,omitzero"` // [-2.0, 2.0]
	ReasoningFormat     string             `json:"reasoning_format,omitempty"`
	ResponseFormat      any                `json:"response_format,omitempty"` // TODO e.g. json_object with json_schema
	Seed                int64              `json:"seed,omitzero"`
	ServiceTier         string             `json:"service_tier,omitzero"` // "on_demand", "auto", "flex"
	Stop                []string           `json:"stop,omitempty"`        // keywords to stop completion
	Stream              bool               `json:"stream"`
	StreamOptions       any                `json:"stream_options,omitempty"` // TODO
	Temperature         float64            `json:"temperature,omitzero"`     // [0, 2]
	Tools               []any              `json:"tools,omitempty"`          // TODO
	ToolChoices         []any              `json:"tool_choices,omitempty"`   // TODO
	TopP                float64            `json:"top_p,omitzero"`           // [0, 1]
	User                string             `json:"user,omitzero"`

	// Explicitly Unsupported:
	// LogitBias           map[string]float64 `json:"logit_bias,omitempty"`
	// LogProbs            bool               `json:"logprobs,omitzero"`
	// TopLogProbs         int64                `json:"top_logprobs,omitzero"`     // [0, 20]
	// N                   int64                `json:"n,omitzero"`                // Number of choices
}

type chatCompletionsResponse struct {
	Choices []choices `json:"choices"`
	Created int64     `json:"created"` // Unix timestamp
	ID      string    `json:"id"`
	Model   string    `json:"model"`
	Object  string    `json:"object"` // chat.completion
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

type choice struct {
	Role    genaiapi.Role `json:"role"`
	Content string        `json:"content"`
}

type choices struct {
	// FinishReason is one of "stop", "length", "content_filter" or "tool_calls".
	FinishReason string `json:"finish_reason"`
	Index        int64  `json:"index"`
	Message      choice `json:"message"`
	Logprobs     any    `json:"logprobs"`
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
	in := chatCompletionRequest{Model: c.Model, Messages: msgs}
	switch v := opts.(type) {
	case *genaiapi.CompletionOptions:
		in.MaxCompletionTokens = v.MaxTokens
		in.Seed = v.Seed
		in.Temperature = v.Temperature
	default:
		return "", fmt.Errorf("unsupported options type %T", opts)
	}
	out := chatCompletionsResponse{}
	if err := c.post(ctx, "https://api.groq.com/openai/v1/chat/completions", in, &out); err != nil {
		return "", fmt.Errorf("failed to get chat response: %w", err)
	}
	if len(out.Choices) != 1 {
		return "", fmt.Errorf("server returned an unexpected number of choices, expected 1, got %d", len(out.Choices))
	}
	return out.Choices[0].Message.Content, nil
}

func (c *Client) CompletionStream(ctx context.Context, msgs []genaiapi.Message, opts any, words chan<- string) (string, error) {
	return "", errors.New("not implemented")
}

func (c *Client) CompletionContent(ctx context.Context, msgs []genaiapi.Message, opts any, mime string, content []byte) (string, error) {
	return "", errors.New("not implemented")
}

func (c *Client) post(ctx context.Context, url string, in, out any) error {
	h := make(http.Header)
	h.Add("Authorization", "Bearer "+c.ApiKey)
	p := httpjson.DefaultClient
	// OpenAI doesn't support any compression. lol.
	p.Compress = ""
	if err := p.Post(ctx, url, h, in, out); err != nil {
		if err2, ok := err.(*httpjson.Error); ok {
			er := errorResponse{}
			d := json.NewDecoder(bytes.NewReader(err2.ResponseBody))
			d.DisallowUnknownFields()
			if err3 := d.Decode(&er); err3 == nil {
				return fmt.Errorf("error %s (%s): %s", er.Error.Code, er.Error.Type, er.Error.Message)
			}
			slog.WarnContext(ctx, "qroq", "url", url, "err", err2, "response", string(err2.ResponseBody))
		}
		return err
	}
	return nil
}

var _ genaiapi.CompletionProvider = &Client{}
