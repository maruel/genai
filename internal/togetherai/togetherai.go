// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Package togetherai implements a client for the Together.ai API.
//
// It is described at https://docs.together.ai/docs/
package togetherai

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"math/big"
	"net/http"
	"strings"
	"time"

	"github.com/maruel/genai/genaiapi"
	"github.com/maruel/httpjson"
)

// https://docs.together.ai/reference/chat-completions-1
type CompletionRequest struct {
	Model                         string             `json:"model"`
	Stream                        bool               `json:"stream"`
	Messages                      []Message          `json:"messages"`
	MaxTokens                     int64              `json:"max_tokens,omitzero"`
	Stop                          []string           `json:"stop,omitzero"`
	Temperature                   float64            `json:"temperature,omitzero"` // [0, 1]
	TopP                          float64            `json:"top_p,omitzero"`       // [0, 1]
	TopK                          int32              `json:"top_k,omitzero"`
	ContextLengthExceededBehavior string             `json:"context_length_exceeded_behavior,omitzero"` // "error", "truncate"
	RepetitionPenalty             float64            `json:"repetition_penalty,omitzero"`
	Logprobs                      int32              `json:"logprobs,omitzero"` // bool as 0 or 1
	Echo                          bool               `json:"echo,omitzero"`
	N                             int32              `json:"n,omitzero"`                 // Number of completions to generate
	PresencePenalty               float64            `json:"presence_penalty,omitzero"`  // [-2.0, 2.0]
	FrequencyPenalty              float64            `json:"frequency_penalty,omitzero"` // [-2.0, 2.0]
	LogitBias                     map[string]float64 `json:"logit_bias,omitzero"`
	Seed                          int64              `json:"seed,omitzero"`
	ResponseFormat                any                `json:"response_format,omitzero"` // TODO
	Tools                         []any              `json:"tools,omitzero"`           // TODO
	ToolChoices                   []any              `json:"tool_choices,omitzero"`    // TODO
	SafetyModel                   string             `json:"safety_model,omitzero"`    // https://docs.together.ai/docs/inference-models#moderation-models
}

func (c *CompletionRequest) fromOpts(opts any) error {
	switch v := opts.(type) {
	case *genaiapi.CompletionOptions:
		c.MaxTokens = v.MaxTokens
		c.Seed = v.Seed
		c.Temperature = v.Temperature
	default:
		return fmt.Errorf("unsupported options type %T", opts)
	}
	return nil
}

func (c *CompletionRequest) fromMsgs(msgs []genaiapi.Message) error {
	c.Messages = make([]Message, len(msgs))
	for i := range msgs {
		c.Messages[i].Role = msgs[i].Role
		c.Messages[i].Content.Type = "text"
		c.Messages[i].Content.Text = msgs[i].Content
	}
	return nil
}

type Message struct {
	Role    genaiapi.Role `json:"role"`
	Content struct {
		Type     string `json:"content"` // text,image_url,video_url
		Text     string `json:"text"`
		ImageURL string `json:"image_url"`
		VideoURL string `json:"video_url"`
	} `json:"content"`
}

type CompletionResponse struct {
	ID      string   `json:"id"`
	Prompt  []string `json:"prompt"`
	Choices []struct {
		Text  string `json:"text"`
		Index int64  `json:"index"`
		// The seed is returned as a int128.
		Seed big.Int `json:"seed"`
		// FinishReason is one of "stop", "eos", "length", "function_call" or "tool_calls".
		FinishReason string `json:"finish_reason"`
		Message      struct {
			Role      genaiapi.Role `json:"role"`
			Content   string        `json:"content"`
			ToolCalls []struct {
				Index    int64  `json:"index"`
				ID       string `json:"id"`
				Type     string `json:"type"` // function
				Function struct {
					Name      string   `json:"name"`
					Arguments []string `json:"arguments"`
				} `json:"function"`
			} `json:"tool_calls"`
			FunctionCall struct {
				Name      string   `json:"name"`
				Arguments []string `json:"arguments"`
			} `json:"function_call"`
		} `json:"message"`
		Logprobs struct {
			TokenIDs      []int64   `json:"token_ids"`
			Tokens        []string  `json:"tokens"`
			TokenLogprobs []float64 `json:"token_logprobs"`
		} `json:"logprobs"`
	} `json:"choices"`
	Usage struct {
		PromptTokens     int64 `json:"prompt_tokens"`
		CompletionTokens int64 `json:"completion_tokens"`
		TotalTokens      int64 `json:"total_tokens"`
	} `json:"usage"`
	Created Time   `json:"created"`
	Model   string `json:"model"`
	Object  string `json:"object"` // chat.completion
}

type CompletionStreamChunkResponse struct {
	/*
		ID                string `json:"id"`
		Object            string `json:"object"`
		Created           Time   `json:"created"`
		Model             string `json:"model"`
		SystemFingerprint string `json:"system_fingerprint"`
		Choices           []struct {
			Index int64 `json:"index"`
			Delta struct {
				Role    genaiapi.Role `json:"role"`
				Content string        `json:"content"`
			} `json:"delta"`
			Logprobs     any    `json:"logprobs"`
			FinishReason string `json:"finish_reason"` // stop
		} `json:"choices"`
	*/
}

//

type errorResponse struct {
	ID    string `json:"id"`
	Error struct {
		Message string `json:"message"`
		Type    string `json:"type"`
		Code    string `json:"code"`
		Param   string `json:"param"`
	} `json:"error"`
}

type Client struct {
	// ApiKey can be retrieved from https://api.together.ai/settings/api-keys
	ApiKey string
	// Model to use, from https://docs.together.ai/docs/serverless-models
	Model string
}

func (c *Client) Completion(ctx context.Context, msgs []genaiapi.Message, opts any) (string, error) {
	// https://docs.together.ai/docs/chat-overview
	in := CompletionRequest{Model: c.Model}
	if err := in.fromOpts(opts); err != nil {
		return "", err
	}
	if err := in.fromMsgs(msgs); err != nil {
		return "", err
	}
	out := CompletionResponse{}
	if err := c.CompletionRaw(ctx, &in, &out); err != nil {
		return "", fmt.Errorf("failed to get chat response: %w", err)
	}
	if len(out.Choices) != 1 {
		return "", fmt.Errorf("server returned an unexpected number of choices, expected 1, got %d", len(out.Choices))
	}
	return out.Choices[0].Message.Content, nil
}

func (c *Client) CompletionRaw(ctx context.Context, in *CompletionRequest, out *CompletionResponse) error {
	if err := c.validate(true); err != nil {
		return err
	}
	return c.post(ctx, "https://api.together.xyz/v1/chat/completions", in, out)
}

func (c *Client) CompletionStream(ctx context.Context, msgs []genaiapi.Message, opts any, words chan<- string) error {
	in := CompletionRequest{Model: c.Model, Stream: true}
	if err := in.fromOpts(opts); err != nil {
		return err
	}
	if err := in.fromMsgs(msgs); err != nil {
		return err
	}
	ch := make(chan CompletionStreamChunkResponse)
	end := make(chan struct{})
	go func() {
		/* TODO
		for msg := range ch {
			if len(msg.Choices) != 1 {
				continue
			}
			word := msg.Choices[0].Delta.Content
			if word != "" {
				words <- word
			}
		}
		*/
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
	h.Add("Authorization", "Bearer "+c.ApiKey)
	// Together.ai doesn't HTTP POST support compression.
	// TODO Test
	// httpjson.DefaultClient.PostCompress = "gzip"
	resp, err := httpjson.DefaultClient.PostRequest(ctx, "https://api.together.xyz/v1/chat/completions", h, in)
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
		const prefix = "data: "
		if !bytes.HasPrefix(line, []byte(prefix)) {
			return fmt.Errorf("unexpected line. expected \"data: \", got %q", line)
		}
		suffix := string(line[len(prefix):])
		d := json.NewDecoder(strings.NewReader(suffix))
		d.DisallowUnknownFields()
		d.UseNumber()
		msg := CompletionStreamChunkResponse{}
		if err = d.Decode(&msg); err != nil {
			return fmt.Errorf("failed to decode server response %q: %w", string(line), err)
		}
		out <- msg
	}
}

func (c *Client) CompletionContent(ctx context.Context, msgs []genaiapi.Message, opts any, mime string, content []byte) (string, error) {
	return "", errors.New("not implemented")
}

type Time int64

func (t *Time) AsTime() time.Time {
	return time.Unix(int64(*t), 0)
}

type Model struct {
	ID            string `json:"id"`
	Object        string `json:"object"`
	Created       Time   `json:"created"`
	Type          string `json:"type"` // chat,moderation,image,
	Running       bool   `json:"running"`
	DisplayName   string `json:"display_name"`
	Organization  string `json:"organization"`
	Link          string `json:"link"`
	License       string `json:"license"`
	ContextLength int64  `json:"context_length"`
	Config        struct {
		ChatTemplate string   `json:"chat_template"`
		Stop         []string `json:"stop"`
		BosToken     string   `json:"bos"`
		EosToken     string   `json:"eos"`
	} `json:"config"`
	Pricing struct {
		Hourly   float64 `json:"hourly"`
		Input    float64 `json:"input"`
		Output   float64 `json:"output"`
		Base     float64 `json:"base"`
		Finetune float64 `json:"finetune"`
	} `json:"pricing"`
}

func (m *Model) GetID() string {
	return m.ID
}

func (m *Model) String() string {
	return fmt.Sprintf("%s (%s): %s Context: %d; in: %.1f$/Mt out: %.1f$/Mt", m.ID, m.Created.AsTime().Format("2006-01-02"), m.Type, m.ContextLength, m.Pricing.Input, m.Pricing.Output)
}

func (c *Client) ListModels(ctx context.Context) ([]genaiapi.Model, error) {
	// https://docs.together.ai/reference/models-1
	if err := c.validate(false); err != nil {
		return nil, err
	}
	h := make(http.Header)
	h.Add("Authorization", "Bearer "+c.ApiKey)
	var out []Model
	err := httpjson.DefaultClient.Get(ctx, "https://api.together.xyz/v1/models", h, &out)
	if err != nil {
		return nil, err
	}
	models := make([]genaiapi.Model, len(out))
	for i := range out {
		models[i] = &out[i]
	}
	return models, err
}

func (c *Client) validate(needModel bool) error {
	if c.ApiKey == "" {
		return errors.New("together.ai ApiKey is required; get one at " + apiKeyURL)
	}
	if needModel && c.Model == "" {
		return errors.New("a Model is required")
	}
	return nil
}

func (c *Client) post(ctx context.Context, url string, in, out any) error {
	h := make(http.Header)
	h.Add("Authorization", "Bearer "+c.ApiKey)
	// Together.AI doesn't HTTP POST support compression.
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
				return fmt.Errorf("%w: error %s (%s): %s. You can get a new API key at %s", herr, er.Error.Code, er.Error.Type, er.Error.Message, apiKeyURL)
			}
			return fmt.Errorf("%w: error %s (%s): %s", herr, er.Error.Code, er.Error.Type, er.Error.Message)
		}
		return fmt.Errorf("error %s (%s): %s", er.Error.Code, er.Error.Type, er.Error.Message)
	default:
		var herr *httpjson.Error
		if errors.As(err, &herr) {
			slog.WarnContext(ctx, "togetherai", "url", url, "err", err, "response", string(herr.ResponseBody), "status", herr.StatusCode)
		} else {
			slog.WarnContext(ctx, "togetherai", "url", url, "err", err)
		}
		return err
	}
}

const apiKeyURL = "https://api.together.ai/settings/api-keys"

var _ genaiapi.CompletionProvider = &Client{}
