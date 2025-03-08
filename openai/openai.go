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

	"github.com/maruel/genai/genaiapi"
	"github.com/maruel/httpjson"
)

// Messages. https://platform.openai.com/docs/api-reference/making-requests

// chatCompletionRequest is documented at
// https://platform.openai.com/docs/api-reference/chat/create
type chatCompletionRequest struct {
	Model               string             `json:"model"`
	MaxTokens           int64              `json:"max_tokens,omitzero"` // Deprecated
	MaxCompletionTokens int64              `json:"max_completion_tokens,omitzero"`
	Stream              bool               `json:"stream"`
	Messages            []genaiapi.Message `json:"messages"`
	Seed                int64              `json:"seed,omitzero"`
	Temperature         float64            `json:"temperature,omitzero"` // [0, 2]
	Store               bool               `json:"store,omitzero"`
	ReasoningEffort     string             `json:"reasoning_effort,omitempty"` // low, medium, high
	Metadata            map[string]string  `json:"metadata,omitempty"`
	FrequencyPenalty    float64            `json:"frequency_penalty,omitempty"` // [-2.0, 2.0]
	LogitBias           map[string]float64 `json:"logit_bias,omitempty"`
	LogProbs            bool               `json:"logprobs,omitzero"`
	TopLogProbs         int64              `json:"top_logprobs,omitzero"`     // [0, 20]
	N                   int64              `json:"n,omitzero"`                // Number of choices
	Modalities          []string           `json:"modalities,omitempty"`      // text, audio
	Prediction          any                `json:"prediction,omitempty"`      // TODO
	Audio               any                `json:"audio,omitempty"`           // TODO
	PresencePenalty     float64            `json:"presence_penalty,omitzero"` // [-2.0, 2.0]
	ResponseFormat      any                `json:"response_format,omitempty"` // TODO e.g. json_object with json_schema
	ServiceTier         string             `json:"service_tier,omitzero"`     // "auto", "default"
	Stop                []string           `json:"stop,omitempty"`            // keywords to stop completion
	StreamOptions       any                `json:"stream_options,omitempty"`  // TODO
	TopP                float64            `json:"top_p,omitzero"`            // [0, 1]
	Tools               []any              `json:"tools,omitempty"`           // TODO
	ToolChoices         []any              `json:"tool_choices,omitempty"`    // TODO
	ParallelToolCalls   bool               `json:"parallel_tool_calls,omitzero"`
	User                string             `json:"user,omitzero"`
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
	Role    genaiapi.Role `json:"role"`
	Content string        `json:"content"`
	Refusal string        `json:"refusal"`
}

type choices struct {
	// FinishReason is one of "stop", "length", "content_filter" or "tool_calls".
	FinishReason string      `json:"finish_reason"`
	Index        int64       `json:"index"`
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
	Index        int64  `json:"index"`
	// Message      genaiapi.Message `json:"message"`
}

type openAIStreamDelta struct {
	Content string `json:"content"`
}

//

type errorResponse struct {
	Error errorResponseError `json:"error"`
}

type errorResponseError struct {
	Code    string `json:"code"`
	Message string `json:"message"`
	Status  string `json:"status"`
	Type    string `json:"type"`
	Param   any    `json:"param"`
}

//

type Client struct {
	// BaseURL defaults to OpenAI's API endpoint. See billing information at
	// https://platform.openai.com/settings/organization/billing/overview
	BaseURL string
	// ApiKey can be retrieved from https://platform.openai.com/settings/organization/api-keys
	ApiKey string
	// Model to use, from https://platform.openai.com/docs/api-reference/models
	Model string
}

func (c *Client) Completion(ctx context.Context, msgs []genaiapi.Message, opts any) (string, error) {
	in := chatCompletionRequest{Model: c.Model, Messages: msgs}
	switch v := opts.(type) {
	case *genaiapi.CompletionOptions:
		in.MaxTokens = v.MaxTokens
		in.Seed = v.Seed
		in.Temperature = v.Temperature
	default:
		return "", fmt.Errorf("unsupported options type %T", opts)
	}
	out := chatCompletionsResponse{}
	if err := c.post(ctx, c.baseURL()+"/v1/chat/completions", in, &out); err != nil {
		return "", fmt.Errorf("failed to get chat response: %w", err)
	}
	if len(out.Choices) != 1 {
		return "", fmt.Errorf("server returned an unexpected number of choices, expected 1, got %d", len(out.Choices))
	}
	return out.Choices[0].Message.Content, nil
}

func (c *Client) CompletionStream(ctx context.Context, msgs []genaiapi.Message, opts any, words chan<- string) (string, error) {
	start := time.Now()
	in := chatCompletionRequest{Model: c.Model, Messages: msgs, Stream: true}
	switch v := opts.(type) {
	case *genaiapi.CompletionOptions:
		in.MaxTokens = v.MaxTokens
		in.Seed = v.Seed
		in.Temperature = v.Temperature
	default:
		return "", fmt.Errorf("unsupported options type %T", opts)
	}
	h := make(http.Header)
	h.Add("Authorization", "Bearer "+c.ApiKey)
	p := httpjson.DefaultClient
	// OpenAI doesn't support any compression. lol.
	p.Compress = ""
	resp, err := p.PostRequest(ctx, c.baseURL()+"/v1/chat/completions", h, in)
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
		out := chatCompletionsStreamResponse{}
		if err = d.Decode(&out); err != nil {
			return reply, fmt.Errorf("failed to decode server response %q: %w", string(line), err)
		}
		if len(out.Choices) != 1 {
			return reply, fmt.Errorf("server returned an unexpected number of choices, expected 1, got %d", len(out.Choices))
		}
		word := out.Choices[0].Delta.Content
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

func (c *Client) CompletionContent(ctx context.Context, msgs []genaiapi.Message, opts any, mime string, content []byte) (string, error) {
	return "", errors.New("not implemented")
}

func (c *Client) post(ctx context.Context, url string, in, out any) error {
	if c.ApiKey == "" {
		return errors.New("openai ApiKey is required; get one at " + apiKeyURL)
	}
	h := make(http.Header)
	h.Add("Authorization", "Bearer "+c.ApiKey)
	p := httpjson.DefaultClient
	// OpenAI doesn't support any compression. lol.
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
		// OpenAI error message prints the URL already.
		if er.Error.Code == "" {
			return fmt.Errorf("error %s: %s", er.Error.Type, er.Error.Message)
		}
		return fmt.Errorf("error %s (%s): %s", er.Error.Code, er.Error.Status, er.Error.Message)
	default:
		var herr *httpjson.Error
		if errors.As(err, &herr) {
			slog.WarnContext(ctx, "openai", "url", url, "err", err, "response", string(herr.ResponseBody), "status", herr.StatusCode)
		} else {
			slog.WarnContext(ctx, "openai", "url", url, "err", err)
		}
		return err
	}
}

func (c *Client) baseURL() string {
	if c.BaseURL != "" {
		return c.BaseURL
	}
	return "https://api.openai.com"
}

const apiKeyURL = "https://platform.openai.com/settings/organization/api-keys"

var _ genaiapi.CompletionProvider = &Client{}
