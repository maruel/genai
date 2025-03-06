// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package deepseek

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"net/http"

	"github.com/maruel/genai"
	"github.com/maruel/httpjson"
)

type message struct {
	Role             genai.Role `json:"role"`
	Content          string     `json:"content"`
	Name             string     `json:"name,omitzero"`
	Prefix           bool       `json:"prefix,omitzero"`
	ReasoningContent string     `json:"reasoning_content,omitzero"`
	ToolCallID       string     `json:"tool_call_id,omitzero"`
}

// https://api-docs.deepseek.com/api/create-chat-completion
type messagesRequest struct {
	Model            string    `json:"model,omitempty"`
	Messages         []message `json:"messages"`
	Stream           bool      `json:"stream"`
	Temperature      float64   `json:"temperature,omitzero"`       // [0, 2]
	FrequencyPenalty float64   `json:"frequency_penalty,omitzero"` // [-2, 2]
	MaxToks          int       `json:"max_tokens,omitzero"`        // [1, 8192]
	PresencePenalty  float64   `json:"presence_penalty,omitzero"`  // [-2, 2]
	ResponseFormat   struct {
		Type string `json:"type,omitzero"` // text, json_object
	} `json:"response_format,omitzero"`
	Stop          []string `json:"stop,omitempty"`
	StreamOptions any      `json:"stream_options,omitempty"`
	TopP          float64  `json:"top_p,omitzero"` // [0, 1]
	ToolChoice    any      `json:"tool_choice,omitempty"`
	Tools         any      `json:"tools,omitempty"`
	LogProbs      bool     `json:"logprobs,omitzero"`
	TopLogProb    int      `json:"top_logprobs,omitzero"`
}

type messageResponse struct {
	Content          string `json:"content"`
	ReasoningContent string `json:"reasoning_content"`
	ToolCalls        []struct {
		ID       string `json:"id"`
		Type     string `json:"type"`
		Function struct {
			Name      string `json:"name"`
			Arguments string `json:"arguments"`
		} `json:"function"`
	} `json:"tool_calls"`
}

type choice struct {
	FinishReason string  `json:"finish_reason"`
	Index        int     `json:"index"`
	Message      message `json:"message"`
	LogProbs     struct {
		Content []struct {
			Token       string  `json:"token"`
			LogProb     float64 `json:"logprob"`
			Bytes       []int   `json:"bytes"`
			TopLogProbs []struct {
				Token   string  `json:"token"`
				LogProb float64 `json:"logprob"`
				Bytes   []int   `json:"bytes"`
			} `json:"top_logprobs"`
		} `json:"content"`
	} `json:"logprobs"`
}

type messagesResponse struct {
	ID                string   `json:"id"`
	Choices           []choice `json:"choices"`
	Created           int      `json:"created"` // Unix timestamp
	Model             string   `json:"model"`
	SystemFingerPrint string   `json:"system_fingerprint"`
	Object            string   `json:"object"` // chat.completion
	Usage             struct {
		CompletionTokens      int `json:"completion_tokens"`
		PromptTokens          int `json:"prompt_tokens"`
		PromptCacheHitTokens  int `json:"prompt_cache_hit_tokens"`
		PromptCacheMissTokens int `json:"prompt_cache_miss_tokens"`
		TotalTokens           int `json:"total_tokens"`
		PromptTokensDetails   struct {
			CachedTokens int `json:"cached_tokens"`
		} `json:"prompt_tokens_details"`
		CompletionTokensDetails struct {
			ReasoningTokens int `json:"reasoning_tokens"`
		} `json:"completion_tokens_details"`
	} `json:"usage"`
}

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

func (c *Client) Completion(ctx context.Context, msgs []genai.Message, maxtoks, seed int, temperature float64) (string, error) {
	in := messagesRequest{
		Model:       c.Model,
		Messages:    make([]message, 0, len(msgs)),
		MaxToks:     maxtoks,
		Temperature: temperature,
	}
	for _, m := range msgs {
		switch m.Role {
		case genai.System, genai.User, genai.Assistant:
			in.Messages = append(in.Messages, message{Role: m.Role, Content: m.Content})
		default:
			return "", fmt.Errorf("unsupported role %v", m.Role)
		}
	}
	out := messagesResponse{}
	if err := c.post(ctx, "https://api.deepseek.com/chat/completions", &in, &out); err != nil {
		return "", err
	}
	return out.Choices[0].Message.Content, nil
}

func (c *Client) CompletionStream(ctx context.Context, msgs []genai.Message, maxtoks, seed int, temperature float64, words chan<- string) (string, error) {
	return "", errors.New("not implemented")
}

func (c *Client) CompletionContent(ctx context.Context, msgs []genai.Message, maxtoks, seed int, temperature float64, mime string, context []byte) (string, error) {
	return "", errors.New("not implemented")
}

func (c *Client) post(ctx context.Context, url string, in, out any) error {
	p := httpjson.Default
	p.Compress = ""
	h := make(http.Header)
	h.Set("Authorization", "Bearer "+c.ApiKey)
	if err := p.Post(ctx, url, h, in, out); err != nil {
		if err2, ok := err.(*httpjson.Error); ok {
			er := errorResponse{}
			d := json.NewDecoder(bytes.NewReader(err2.ResponseBody))
			d.DisallowUnknownFields()
			if err3 := d.Decode(&er); err3 == nil {
				return fmt.Errorf("error %s: %s", er.Error.Type, er.Error.Message)
			}
			slog.WarnContext(ctx, "deepseek", "url", url, "err", err2, "response", string(err2.ResponseBody))
		}
		return err
	}
	return nil
}

var _ genai.Backend = &Client{}
