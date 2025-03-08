// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package cohere

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"net/http"

	"github.com/maruel/genai/genaiapi"
	"github.com/maruel/httpjson"
)

// https://docs.cohere.com/reference/chat

type message struct {
	Role genaiapi.Role `json:"role"`
	// Assistant, System or User.
	Content struct {
		Type     string `json:"type"` // "text", "image_url" or "document"
		Text     string `json:"text,omitempty"`
		ImageURL struct {
			URL string `json:"url,omitempty"`
		} `json:"image_url,omitzero"`
		Document struct {
			Data map[string]any `json:"data,omitempty"` // TODO
			ID   string         `json:"id,omitempty"`   // TODO
		} `json:"document,omitempty"`
	} `json:"content"`
	// Assistant
	Citations any `json:"citations,omitempty"` // TODO
	// Assistant
	ToolCalls []any `json:"tool_calls,omitempty"` // TODO
	// Tool
	ToolCallID string `json:"tool_call_id,omitempty"`
}

type chatCompletionRequest struct {
	Stream           bool      `json:"stream"`
	Model            string    `json:"model"`
	Messages         []message `json:"messages"`
	Tools            []any     `json:"tools,omitempty"`            // TODO
	Documents        []any     `json:"documents,omitempty"`        // TODO
	CitationOptions  any       `json:"citation_options,omitempty"` // TODO
	ResponseFormat   any       `json:"response_format,omitempty"`  // TODO e.g. json_object with json_schema
	SafetyMode       string    `json:"safety_mode,omitempty"`      // "CONTEXTUAL", "STRICT", "OFF"
	MaxTokens        int64     `json:"max_tokens,omitzero"`
	StopSequences    []string  `json:"stop_sequences,omitempty"` // keywords to stop completion
	Temperature      float64   `json:"temperature,omitzero"`
	Seed             int64     `json:"seed,omitzero"`
	FrequencyPenalty float64   `json:"frequency_penalty,omitempty"` // [0, 1.0]
	PresencePenalty  float64   `json:"presence_penalty,omitzero"`   // [0, 1.0]
	K                float64   `json:"k,omitzero"`                  // [0, 500.0]
	P                float64   `json:"p,omitzero"`                  // [0.01, 0.99]
	LogProbs         bool      `json:"logprobs,omitzero"`
	ToolChoices      []any     `json:"tool_choices,omitempty"` // TODO
	StrictTools      bool      `json:"strict_tools,omitzero"`
}

type chatCompletionsResponse struct {
	ID           string `json:"id"`
	FinishReason string `json:"finish_reason"` // COMPLETE, STOP_SEQUENCe, MAX_TOKENS, TOOL_CALL, ERROR
	Message      struct {
		Role      genaiapi.Role `json:"role"`
		ToolCalls []struct {
			ID       string `json:"id"`
			Type     string `json:"type"` // function
			Function struct {
				Name      string `json:"name"`
				Arguments string `json:"arguments"`
			} `json:"function"`
		} `json:"tool_calls"`
		ToolPlan string `json:"tool_plan"`
		Content  []struct {
			Type string `json:"type"` // text
			Text string `json:"text"`
		} `json:"content"`
		Citations []struct {
			Start   int64  `json:"start"`
			End     int64  `json:"end"`
			Text    string `json:"text"`
			Sources []any  `json:"sources"`
			Type    string `json:"type"` // TEXT_CONTENT, PLAN
		} `json:"citations"`
	} `json:"message"`
	Usage struct {
		BilledUnits struct {
			InputTokens     int64 `json:"input_tokens"`
			OutputTokens    int64 `json:"output_tokens"`
			SearchUnits     int64 `json:"search_units"`
			Classifications int64 `json:"classifications"`
		} `json:"billed_units"`
		Tokens struct {
			InputTokens  int64 `json:"input_tokens"`
			OutputTokens int64 `json:"output_tokens"`
		} `json:"tokens"`
	} `json:"usage"`
	LogProbs struct {
		TokenIDs []int64   `json:"token_ids"`
		Text     string    `json:"text"`
		LogProbs []float64 `json:"logprobs"`
	} `json:"logprobs"`
}

//

type errorResponse struct {
	Message   string `json:"message"`
	RequestID string `json:"request_id"`
}

type Client struct {
	// ApiKey can be retrieved from https://dashboard.cohere.com/api-keys
	ApiKey string
	// Model to use, see https://cohere.com/pricing and https://docs.cohere.com/v2/docs/models
	Model string
}

func (c *Client) Completion(ctx context.Context, msgs []genaiapi.Message, opts any) (string, error) {
	// https://docs.cohere.com/reference/chat
	in := chatCompletionRequest{Model: c.Model, Messages: make([]message, len(msgs))}
	for i, m := range msgs {
		in.Messages[i].Role = m.Role
		in.Messages[i].Content.Type = "text"
		in.Messages[i].Content.Text = m.Content
	}
	switch v := opts.(type) {
	case *genaiapi.CompletionOptions:
		in.MaxTokens = v.MaxTokens
		in.Seed = v.Seed
		in.Temperature = v.Temperature
	default:
		return "", fmt.Errorf("unsupported options type %T", opts)
	}
	out := chatCompletionsResponse{}
	if err := c.post(ctx, "https://api.cohere.com/v2/chat", in, &out); err != nil {
		return "", fmt.Errorf("failed to get chat response: %w", err)
	}
	return out.Message.Content[0].Text, nil
}

func (c *Client) CompletionStream(ctx context.Context, msgs []genaiapi.Message, opts any, words chan<- string) (string, error) {
	return "", errors.New("not implemented")
}

func (c *Client) CompletionContent(ctx context.Context, msgs []genaiapi.Message, opts any, mime string, content []byte) (string, error) {
	return "", errors.New("not implemented")
}

func (c *Client) post(ctx context.Context, url string, in, out any) error {
	if c.ApiKey == "" {
		return errors.New("cohere ApiKey is required; get one at " + apiKeyURL)
	}
	h := make(http.Header)
	h.Add("Authorization", "Bearer "+c.ApiKey)
	p := httpjson.DefaultClient
	// Cohere doesn't support any compression. lol.
	p.Compress = ""
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
				return fmt.Errorf("http %d: error: %s. You can get a new API key at %s", herr.StatusCode, er.Message, apiKeyURL)
			}
			return fmt.Errorf("http %d: error: %s", herr.StatusCode, er.Message)
		}
		return fmt.Errorf("error: %s", er.Message)
	default:
		var herr *httpjson.Error
		if errors.As(err, &herr) {
			slog.WarnContext(ctx, "cohere", "url", url, "err", err, "response", string(herr.ResponseBody), "status", herr.StatusCode)
		} else {
			slog.WarnContext(ctx, "cohere", "url", url, "err", err)
		}
		return err
	}
}

const apiKeyURL = "https://dashboard.cohere.com/api-keys"

var _ genaiapi.CompletionProvider = &Client{}
