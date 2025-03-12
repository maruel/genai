// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Package perplexity implements a client for the Perplexity API.
//
// It is described at https://docs.perplexity.ai/api-reference
package perplexity

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

// https://docs.perplexity.ai/api-reference/chat-completions
type CompletionRequest struct {
	Model                  string    `json:"model"`
	Messages               []Message `json:"messages"`
	MaxTokens              int64     `json:"max_tokens,omitzero"`
	Temperature            float64   `json:"temperature,omitzero"`
	TopP                   float64   `json:"top_p,omitzero"` // [0, 1.0]
	SearchDomainFilter     []string  `json:"search_domain_filter,omitzero"`
	ReturnImages           bool      `json:"return_images,omitzero"`
	ReturnRelatedQuestions bool      `json:"return_related_questions,omitzero"`
	SearchRecencyFilter    string    `json:"search_recency_filter,omitzero"` // month, week, day, hour
	TopK                   int64     `json:"top_k,omitzero"`                 // [0, 2048^]
	Stream                 bool      `json:"stream"`
	PresencePenalty        float64   `json:"presence_penalty,omitzero"` // [-2.0, 2.0]
	FrequencyPenalty       float64   `json:"frequency_penalty,omitzero"`
	// Only available in higher tiers, see https://docs.perplexity.ai/guides/usage-tiers
	ResponseFormat JSONSchema `json:"response_format,omitzero"`
}

func (c *CompletionRequest) fromOpts(opts any) error {
	if opts != nil {
		switch v := opts.(type) {
		case *genaiapi.CompletionOptions:
			c.MaxTokens = v.MaxTokens
			c.Temperature = v.Temperature
			if v.Seed != 0 {
				return errors.New("perplexity doesn't support seed")
			}
		default:
			return fmt.Errorf("unsupported options type %T", opts)
		}
	}
	return nil
}

func (c *CompletionRequest) fromMsgs(msgs []genaiapi.Message) error {
	c.Messages = make([]Message, len(msgs))
	for i, m := range msgs {
		if err := m.Validate(); err != nil {
			return fmt.Errorf("message %d: %w", i, err)
		}
		switch m.Role {
		case genaiapi.System:
			if i != 0 {
				return fmt.Errorf("message %d: system message must be first message", i)
			}
		case genaiapi.User, genaiapi.Assistant:
		default:
			return fmt.Errorf("message %d: unsupported role %q", i, m.Role)
		}
		switch m.Type {
		case genaiapi.Text:
		default:
			return fmt.Errorf("message %d: unsupported content type %s", i, m.Type)
		}
		c.Messages[i].Role = m.Role
		c.Messages[i].Content = m.Text
	}
	return nil
}

type Message struct {
	Role    genaiapi.Role `json:"role"`
	Content string        `json:"content"`
}

type JSONSchema any

type CompletionResponse struct {
	ID        string   `json:"id"`
	Model     string   `json:"model"`
	Object    string   `json:"object"`
	Created   Time     `json:"created"`
	Citations []string `json:"citations"`
	Choices   []struct {
		Index        int64  `json:"index"`
		FinishReason string `json:"finish_reason"` // stop, length
		Message      struct {
			Content string `json:"content"`
			Role    string `json:"role"`
		} `json:"message"`
		Delta struct {
			Content string `json:"content"`
			Role    string `json:"role"`
		} `json:"delta"`
	} `json:"choices"`
	Usage struct {
		PromptTokens     int64 `json:"prompt_tokens"`
		CompletionTokens int64 `json:"completion_tokens"`
		TotalTokens      int64 `json:"total_tokens"`
	} `json:"usage"`
}

type CompletionStreamChunkResponse = CompletionResponse

type Time int64

func (t *Time) AsTime() time.Time {
	return time.Unix(int64(*t), 0)
}

//

type errorResponse1 struct {
	Detail string `json:"detail"`
}

type errorResponse2 struct {
	Error struct {
		Message string `json:"message"`
		Type    string `json:"type"`
		Code    int    `json:"code"`
	} `json:"error"`
}

type Client struct {
	// ApiKey can be retrieved from https://www.perplexity.ai/settings/api
	ApiKey string
	// Model to use, defaults to "sonar".
	Model string
}

func (c *Client) Completion(ctx context.Context, msgs []genaiapi.Message, opts any) (string, error) {
	// https://docs.perplexity.ai/api-reference/chat-completions
	if err := c.validate(); err != nil {
		return "", err
	}
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
		return "", errors.New("expected 1 choice")
	}
	return out.Choices[0].Message.Content, nil
}

func (c *Client) CompletionRaw(ctx context.Context, in *CompletionRequest, out *CompletionResponse) error {
	if err := c.validate(); err != nil {
		return err
	}
	return c.post(ctx, "https://api.perplexity.ai/chat/completions", in, out)
}

func (c *Client) CompletionStream(ctx context.Context, msgs []genaiapi.Message, opts any, words chan<- string) error {
	if err := c.validate(); err != nil {
		return err
	}
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
		for msg := range ch {
			if len(msg.Choices) != 1 {
				continue
			}
			word := msg.Choices[0].Message.Content
			if word != "" {
				words <- word
			}
		}
		end <- struct{}{}
	}()
	err := c.CompletionStreamRaw(ctx, &in, ch)
	close(ch)
	<-end
	return err
}

func (c *Client) CompletionStreamRaw(ctx context.Context, in *CompletionRequest, out chan<- CompletionStreamChunkResponse) error {
	if err := c.validate(); err != nil {
		return err
	}
	h := make(http.Header)
	h.Add("Authorization", "Bearer "+c.ApiKey)
	resp, err := httpjson.DefaultClient.PostRequest(ctx, "https://api.perplexity.ai/chat/completions", h, in)
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
		const dataPrefix = "data: "
		if !bytes.HasPrefix(line, []byte(dataPrefix)) {
			d := json.NewDecoder(bytes.NewReader(line))
			d.DisallowUnknownFields()
			d.UseNumber()
			er1 := errorResponse1{}
			if err = d.Decode(&er1); err != nil {
				d = json.NewDecoder(bytes.NewReader(line))
				d.DisallowUnknownFields()
				d.UseNumber()
				er2 := errorResponse2{}
				if err = d.Decode(&er2); err != nil {
					return fmt.Errorf("unexpected line. expected \"data: \", got %q", line)
				}
				return fmt.Errorf("server error: %s", er2.Error.Message)
			}
			return fmt.Errorf("server error: %s", er1.Detail)
		}
		suffix := string(line[len(dataPrefix):])
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

func (c *Client) validate() error {
	if c.ApiKey == "" {
		return errors.New("perplexity ApiKey is required; get one at " + apiKeyURL)
	}
	if c.Model == "" {
		c.Model = "sonar"
	}
	return nil
}

func (c *Client) post(ctx context.Context, url string, in, out any) error {
	h := make(http.Header)
	h.Add("Authorization", "Bearer "+c.ApiKey)
	resp, err := httpjson.DefaultClient.PostRequest(ctx, url, h, in)
	if err != nil {
		return err
	}
	er1 := errorResponse1{}
	switch i, err := httpjson.DecodeResponse(resp, out, &er1); i {
	case 0:
		return nil
	case 1:
		var herr *httpjson.Error
		if errors.As(err, &herr) {
			if herr.StatusCode == http.StatusUnauthorized {
				return fmt.Errorf("%w: error: %s. You can get a new API key at %s", herr, er1.Detail, apiKeyURL)
			}
			return fmt.Errorf("%w: error: %s", herr, er1.Detail)
		}
		return fmt.Errorf("error: %s", er1.Detail)
	default:
		var herr *httpjson.Error
		if errors.As(err, &herr) {
			slog.WarnContext(ctx, "perplexity", "url", url, "err", err, "response", string(herr.ResponseBody), "status", herr.StatusCode)
		} else {
			slog.WarnContext(ctx, "perplexity", "url", url, "err", err)
		}
		return err
	}
}

const apiKeyURL = "https://www.perplexity.ai/settings/api"

var _ genaiapi.CompletionProvider = &Client{}
