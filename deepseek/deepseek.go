// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package deepseek

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"net/http"

	"github.com/maruel/genai/genaiapi"
	"github.com/maruel/httpjson"
)

// https://api-docs.deepseek.com/api/create-chat-completion

type Message struct {
	Role             genaiapi.Role `json:"role"`
	Content          string        `json:"content"`
	Name             string        `json:"name,omitzero"`
	Prefix           bool          `json:"prefix,omitzero"`
	ReasoningContent string        `json:"reasoning_content,omitzero"`
	ToolCallID       string        `json:"tool_call_id,omitzero"`
}

type CompletionRequest struct {
	Model            string    `json:"model,omitempty"`
	Messages         []Message `json:"messages"`
	Stream           bool      `json:"stream"`
	Temperature      float64   `json:"temperature,omitzero"`       // [0, 2]
	FrequencyPenalty float64   `json:"frequency_penalty,omitzero"` // [-2, 2]
	MaxToks          int64     `json:"max_tokens,omitzero"`        // [1, 8192]
	PresencePenalty  float64   `json:"presence_penalty,omitzero"`  // [-2, 2]
	ResponseFormat   struct {
		Type string `json:"type,omitzero"` // text, json_object
	} `json:"response_format,omitzero"`
	Stop          []string `json:"stop,omitempty"`
	StreamOptions any      `json:"stream_options,omitempty"`
	TopP          float64  `json:"top_p,omitzero"` // [0, 1]
	ToolChoice    any      `json:"tool_choice,omitempty"`
	Tools         any      `json:"tools,omitempty"`
	Logprobs      bool     `json:"logprobs,omitzero"`
	TopLogprob    int64    `json:"top_logprobs,omitzero"`
}

type CompletionResponse struct {
	ID      string `json:"id"`
	Choices []struct {
		FinishReason string  `json:"finish_reason"`
		Index        int64   `json:"index"`
		Message      Message `json:"message"`
		Logprobs     struct {
			Content []struct {
				Token       string  `json:"token"`
				Logprob     float64 `json:"logprob"`
				Bytes       []int64 `json:"bytes"`
				TopLogprobs []struct {
					Token   string  `json:"token"`
					Logprob float64 `json:"logprob"`
					Bytes   []int64 `json:"bytes"`
				} `json:"top_logprobs"`
			} `json:"content"`
		} `json:"logprobs"`
	} `json:"choices"`
	Created           int64  `json:"created"` // Unix timestamp
	Model             string `json:"model"`
	SystemFingerPrint string `json:"system_fingerprint"`
	Object            string `json:"object"` // chat.completion
	Usage             struct {
		CompletionTokens      int64 `json:"completion_tokens"`
		PromptTokens          int64 `json:"prompt_tokens"`
		PromptCacheHitTokens  int64 `json:"prompt_cache_hit_tokens"`
		PromptCacheMissTokens int64 `json:"prompt_cache_miss_tokens"`
		TotalTokens           int64 `json:"total_tokens"`
		PromptTokensDetails   struct {
			CachedTokens int64 `json:"cached_tokens"`
		} `json:"prompt_tokens_details"`
		CompletionTokensDetails struct {
			ReasoningTokens int64 `json:"reasoning_tokens"`
		} `json:"completion_tokens_details"`
	} `json:"usage"`
}

//

type errorResponse struct {
	// Type  string `json:"type"`
	Error struct {
		Message string `json:"message"`
		Type    string `json:"type"`
		Param   any    `json:"param"`
		Code    string `json:"code"`
	} `json:"error"`
}

type Client struct {
	// ApiKey retrieved from https://platform.deepseek.com/api_keys
	ApiKey string
	// One of the model from https://api-docs.deepseek.com/quick_start/pricing
	Model string
}

func (c *Client) Completion(ctx context.Context, msgs []genaiapi.Message, opts any) (string, error) {
	// https://api-docs.deepseek.com/api/create-chat-completion
	in := CompletionRequest{Model: c.Model, Messages: make([]Message, 0, len(msgs))}
	for _, m := range msgs {
		switch m.Role {
		case genaiapi.System, genaiapi.User, genaiapi.Assistant:
			in.Messages = append(in.Messages, Message{Role: m.Role, Content: m.Content})
		default:
			return "", fmt.Errorf("unsupported role %v", m.Role)
		}
	}
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
	out := CompletionResponse{}
	if err := c.CompletionRaw(ctx, &in, &out); err != nil {
		return "", err
	}
	return out.Choices[0].Message.Content, nil
}

func (c *Client) CompletionRaw(ctx context.Context, in *CompletionRequest, out *CompletionResponse) error {
	return c.post(ctx, "https://api.deepseek.com/chat/completions", in, out)
}

func (c *Client) CompletionStream(ctx context.Context, msgs []genaiapi.Message, opts any, words chan<- string) error {
	return errors.New("not implemented")
}

func (c *Client) CompletionContent(ctx context.Context, msgs []genaiapi.Message, opts any, mime string, context []byte) (string, error) {
	return "", errors.New("not implemented")
}

func (c *Client) post(ctx context.Context, url string, in, out any) error {
	if c.ApiKey == "" {
		return errors.New("deepseek ApiKey is required; get one at " + apiKeyURL)
	}
	p := httpjson.DefaultClient
	// DeepSeek doesn't support any compression. lol.
	p.Compress = ""
	h := make(http.Header)
	h.Set("Authorization", "Bearer "+c.ApiKey)
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
				return fmt.Errorf("%w: error %s: %s. You can get a new API key at %s", herr, er.Error.Type, er.Error.Message, apiKeyURL)
			}
			return fmt.Errorf("%w: error %s: %s", herr, er.Error.Type, er.Error.Message)
		}
		return fmt.Errorf("error %s: %s", er.Error.Type, er.Error.Message)
	default:
		var herr *httpjson.Error
		if errors.As(err, &herr) {
			slog.WarnContext(ctx, "deepseek", "url", url, "err", err, "response", string(herr.ResponseBody), "status", herr.StatusCode)
		} else {
			slog.WarnContext(ctx, "deepseek", "url", url, "err", err)
		}
		return err
	}
}

const apiKeyURL = "https://platform.deepseek.com/api_keys"

var _ genaiapi.CompletionProvider = &Client{}
