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
	"os"
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
	// Only available in higher tiers, see
	// https://docs.perplexity.ai/guides/usage-tiers and
	// https://docs.perplexity.ai/guides/structured-outputs
	ResponseFormat struct {
		Type       string `json:"type,omitzero"` // "json_schema", "regex"
		JSONSchema struct {
			Schema genaiapi.JSONSchema `json:"schema,omitzero"`
		} `json:"json_schema,omitzero"`
		Regex struct {
			Regex string `json:"regex,omitzero"`
		} `json:"regex,omitzero"`
	} `json:"response_format,omitzero"`
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
			if v.ReplyAsJSON && !v.JSONSchema.IsZero() {
				// Doesn't seem to work in practice.
				c.ResponseFormat.Type = "json_schema"
				c.ResponseFormat.JSONSchema.Schema = v.JSONSchema
			} else if v.ReplyAsJSON || !v.JSONSchema.IsZero() {
				return errors.New("to be implemented")
			}
			if len(v.Tools) != 0 {
				return errors.New("perplexity doesn't support tools yet")
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

// Client implements the REST JSON based API.
type Client struct {
	apiKey string
	model  string
}

// New creates a new client to talk to the Perplexity platform API.
//
// If apiKey is not provided, it tries to load it from the PERPLEXITY_API_KEY environment variable.
// If none is found, it returns an error.
// Get your API key at https://www.perplexity.ai/settings/api
func New(apiKey string) (*Client, error) {
	if apiKey == "" {
		if apiKey = os.Getenv("PERPLEXITY_API_KEY"); apiKey == "" {
			return nil, errors.New("perplexity API key is required; get one at " + apiKeyURL)
		}
	}
	return &Client{apiKey: apiKey, model: "sonar"}, nil
}

func (c *Client) Completion(ctx context.Context, msgs []genaiapi.Message, opts any) (genaiapi.Message, error) {
	// https://docs.perplexity.ai/api-reference/chat-completions
	msg := genaiapi.Message{}
	in := CompletionRequest{Model: c.model}
	if err := in.fromOpts(opts); err != nil {
		return msg, err
	}
	if err := in.fromMsgs(msgs); err != nil {
		return msg, err
	}
	out := CompletionResponse{}
	if err := c.CompletionRaw(ctx, &in, &out); err != nil {
		return msg, fmt.Errorf("failed to get chat response: %w", err)
	}
	if len(out.Choices) != 1 {
		return msg, errors.New("expected 1 choice")
	}
	msg.Type = genaiapi.Text
	msg.Text = out.Choices[0].Message.Content
	switch role := out.Choices[0].Message.Role; role {
	case "system", "assistant", "user":
		msg.Role = genaiapi.Role(role)
	default:
		return msg, fmt.Errorf("unsupported role %q", role)
	}
	return msg, nil
}

func (c *Client) CompletionRaw(ctx context.Context, in *CompletionRequest, out *CompletionResponse) error {
	return c.post(ctx, "https://api.perplexity.ai/chat/completions", in, out)
}

func (c *Client) CompletionStream(ctx context.Context, msgs []genaiapi.Message, opts any, chunks chan<- genaiapi.MessageChunk) error {
	in := CompletionRequest{Model: c.model, Stream: true}
	if err := in.fromOpts(opts); err != nil {
		return err
	}
	if err := in.fromMsgs(msgs); err != nil {
		return err
	}
	ch := make(chan CompletionStreamChunkResponse)
	end := make(chan error)
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()
	go func() {
		lastRole := genaiapi.System
		for pkt := range ch {
			if len(pkt.Choices) != 1 {
				continue
			}
			switch role := pkt.Choices[0].Delta.Role; role {
			case "system", "assistant", "user":
				lastRole = genaiapi.Role(role)
			case "":
			default:
				cancel()
				// We need to empty the channel to avoid blocking the goroutine.
				for range ch {
				}
				end <- fmt.Errorf("unexpected role %q", role)
				return
			}
			if word := pkt.Choices[0].Delta.Content; word != "" {
				chunks <- genaiapi.MessageChunk{Role: lastRole, Type: genaiapi.Text, Text: word}
			}
		}
		end <- nil
	}()
	err := c.CompletionStreamRaw(ctx, &in, ch)
	close(ch)
	if err2 := <-end; err2 != nil {
		err = err2
	}
	return err
}

func (c *Client) CompletionStreamRaw(ctx context.Context, in *CompletionRequest, out chan<- CompletionStreamChunkResponse) error {
	h := make(http.Header)
	h.Add("Authorization", "Bearer "+c.apiKey)
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

func (c *Client) post(ctx context.Context, url string, in, out any) error {
	h := make(http.Header)
	h.Add("Authorization", "Bearer "+c.apiKey)
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
