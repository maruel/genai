// Copyright 2024 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package openai

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

	"github.com/maruel/genai"
	"github.com/maruel/httpjson"
)

// Messages. https://platform.openai.com/docs/api-reference/making-requests

// chatCompletionRequest is documented at
// https://platform.openai.com/docs/api-reference/chat/create
type chatCompletionRequest struct {
	Model       string          `json:"model"`
	MaxTokens   int             `json:"max_tokens,omitempty"`
	Stream      bool            `json:"stream"`
	Messages    []genai.Message `json:"messages"`
	Seed        int             `json:"seed,omitempty"`
	Temperature float64         `json:"temperature,omitempty"`
}

// chatCompletionsResponse is documented at
// https://platform.openai.com/docs/api-reference/chat/object
type chatCompletionsResponse struct {
	Choices []choices `json:"choices"`
	Created int64     `json:"created"`
	ID      string    `json:"id"`
	Model   string    `json:"model"`
	Object  string    `json:"object"`
	Usage   struct {
		PromptTokens        int64 `json:"prompt_tokens"`
		CompletionTokens    int64 `json:"completion_tokens"`
		TotalTokens         int64 `json:"total_tokens"`
		PromptTokensDetails struct {
			CachedTokens int64 `json:"cached_tokens"`
			AudioTokens  int64 `json:"audio_tokens"`
		} `json:"prompt_tokens_details"`
		CompletionTokensDetails struct {
			ReasoningTokens          int64 `json:"reasoning_tokens"`
			AudioTokens              int64 `json:"audio_tokens"`
			AcceptedPredictionTokens int64 `json:"accepted_prediction_tokens"`
			RejectedPredictionTokens int64 `json:"rejected_prediction_tokens"`
		} `json:"completion_tokens_details"`
	} `json:"usage"`
	ServiceTier       string `json:"service_tier"`
	SystemFingerprint string `json:"system_fingerprint"`
}

type choice struct {
	Role    genai.Role `json:"role"`
	Content string     `json:"content"`
	Refusal string     `json:"refusal"`
}

type choices struct {
	// FinishReason is one of "stop", "length", "content_filter" or "tool_calls".
	FinishReason string      `json:"finish_reason"`
	Index        int         `json:"index"`
	Message      choice      `json:"message"`
	Logprobs     interface{} `json:"logprobs"`
}

// chatCompletionsStreamResponse is not documented?
type chatCompletionsStreamResponse struct {
	Choices []streamChoices `json:"choices"`
	Created int64           `json:"created"`
	ID      string          `json:"id"`
	Model   string          `json:"model"`
	Object  string          `json:"object"`
	Usage   struct {
		CompletionTokens int64 `json:"completion_tokens"`
		PromptTokens     int64 `json:"prompt_tokens"`
		TotalTokens      int64 `json:"total_tokens"`
	} `json:"usage"`
}

type streamChoices struct {
	Delta openAIStreamDelta `json:"delta"`
	// FinishReason is one of null, "stop", "length", "content_filter" or "tool_calls".
	FinishReason string `json:"finish_reason"`
	Index        int    `json:"index"`
	// Message      genai.Message `json:"message"`
}

type openAIStreamDelta struct {
	Content string `json:"content"`
}

type errorResponse struct {
	Error errorResponseError `json:"error"`
}

type errorResponseError struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
	Status  string `json:"status"`
}

type Client struct {
	BaseURL string
	ApiKey  string
	Model   string
}

func (c *Client) Completion(ctx context.Context, msgs []genai.Message, maxtoks, seed int, temperature float64) (string, error) {
	data := chatCompletionRequest{
		Model:       c.Model,
		MaxTokens:   maxtoks,
		Messages:    msgs,
		Seed:        seed,
		Temperature: temperature,
	}
	msg := chatCompletionsResponse{}
	if err := c.post(ctx, c.baseURL()+"/v1/chat/completions", data, &msg); err != nil {
		return "", fmt.Errorf("failed to get chat response: %w", err)
	}
	if len(msg.Choices) != 1 {
		return "", fmt.Errorf("server returned an unexpected number of choices, expected 1, got %d", len(msg.Choices))
	}
	return msg.Choices[0].Message.Content, nil
}

func (c *Client) CompletionStream(ctx context.Context, msgs []genai.Message, maxtoks, seed int, temperature float64, words chan<- string) (string, error) {
	start := time.Now()
	data := chatCompletionRequest{
		Model:       c.Model,
		Messages:    msgs,
		MaxTokens:   maxtoks,
		Stream:      true,
		Seed:        seed,
		Temperature: temperature,
	}
	h := make(http.Header)
	h.Add("Authorization", "Bearer "+c.ApiKey)
	resp, err := httpjson.Default.PostRequest(ctx, c.baseURL()+"/v1/chat/completions", h, data)
	if err != nil {
		return "", fmt.Errorf("failed to get server response: %w", err)
	}
	defer resp.Body.Close()
	r := bufio.NewReader(resp.Body)
	reply := ""
	for {
		line, err := r.ReadBytes('\n')
		line = bytes.TrimSpace(line)
		if err == io.EOF {
			err = nil
			if len(line) == 0 {
				return reply, nil
			}
		}
		if err != nil {
			return reply, fmt.Errorf("failed to get server response: %w", err)
		}
		if len(line) == 0 {
			continue
		}
		const prefix = "data: "
		if !bytes.HasPrefix(line, []byte(prefix)) {
			return reply, fmt.Errorf("unexpected line. expected \"data: \", got %q", line)
		}
		suffix := string(line[len(prefix):])
		if suffix == "[DONE]" {
			return reply, nil
		}
		d := json.NewDecoder(strings.NewReader(suffix))
		d.DisallowUnknownFields()
		msg := chatCompletionsStreamResponse{}
		if err = d.Decode(&msg); err != nil {
			return reply, fmt.Errorf("failed to decode server response %q: %w", string(line), err)
		}
		if len(msg.Choices) != 1 {
			return reply, fmt.Errorf("server returned an unexpected number of choices, expected 1, got %d", len(msg.Choices))
		}
		word := msg.Choices[0].Delta.Content
		slog.DebugContext(ctx, "openai", "word", word, "duration", time.Since(start).Round(time.Millisecond))
		// TODO: Remove.
		switch word {
		// Llama-3, Gemma-2, Phi-3
		case "<|eot_id|>", "<end_of_turn>", "<|end|>", "<|endoftext|>":
			return reply, nil
		case "":
		default:
			words <- word
			reply += word
		}
	}
}

func (c *Client) CompletionContent(ctx context.Context, msgs []genai.Message, maxtoks, seed int, temperature float64, mime string, content []byte) (string, error) {
	return "", errors.New("not implemented")
}

func (c *Client) post(ctx context.Context, url string, in, out any) error {
	h := make(http.Header)
	h.Add("Authorization", "Bearer "+c.ApiKey)
	if err := httpjson.Default.Post(ctx, url, h, in, out); err != nil {
		if err2, ok := err.(*httpjson.Error); ok {
			er := errorResponse{}
			d := json.NewDecoder(bytes.NewReader(err2.ResponseBody))
			d.DisallowUnknownFields()
			if err3 := d.Decode(&er); err3 == nil {
				return fmt.Errorf("error %d (%s): %s", er.Error.Code, er.Error.Status, er.Error.Message)
			}
			slog.WarnContext(ctx, "openai", "url", url, "err", err2, "response", string(err2.ResponseBody))
		}
		return err
	}
	return nil
}

func (c *Client) baseURL() string {
	if c.BaseURL != "" {
		return c.BaseURL
	}
	return "https://api.openai.com"
}

var _ genai.Backend = &Client{}
