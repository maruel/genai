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

	"github.com/invopop/jsonschema"
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
			Schema *jsonschema.Schema `json:"schema,omitzero"`
		} `json:"json_schema,omitzero"`
		Regex struct {
			Regex string `json:"regex,omitzero"`
		} `json:"regex,omitzero"`
	} `json:"response_format,omitzero"`
}

func (c *CompletionRequest) Init(msgs []genaiapi.Message, opts genaiapi.Validatable) error {
	var errs []error
	if opts != nil {
		if err := opts.Validate(); err != nil {
			errs = append(errs, err)
		} else {
			switch v := opts.(type) {
			case *genaiapi.CompletionOptions:
				c.MaxTokens = v.MaxTokens
				c.Temperature = v.Temperature
				if v.Seed != 0 {
					errs = append(errs, errors.New("perplexity doesn't support seed"))
				}
				c.TopP = v.TopP
				c.TopK = v.TopK
				if len(v.Stop) != 0 {
					errs = append(errs, errors.New("perplexity doesn't support stop tokens"))
				}
				if v.DecodeAs != nil {
					// Requires Tier 3 to work in practice.
					c.ResponseFormat.Type = "json_schema"
					c.ResponseFormat.JSONSchema.Schema = jsonschema.Reflect(v.DecodeAs)
				} else if v.ReplyAsJSON {
					errs = append(errs, errors.New("perplexity client doesn't support unstructured JSON yet; use structured JSON"))
				}
				if len(v.Tools) != 0 {
					errs = append(errs, errors.New("perplexity doesn't support tools"))
				}
			default:
				errs = append(errs, fmt.Errorf("unsupported options type %T", opts))
			}
		}
	}

	if err := genaiapi.ValidateMessages(msgs); err != nil {
		errs = append(errs, err)
	} else {
		c.Messages = make([]Message, len(msgs))
		for i, m := range msgs {
			if err := c.Messages[i].From(m); err != nil {
				return fmt.Errorf("message %d: %w", i, err)
			}
		}
	}
	return errors.Join(errs...)
}

type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

func (msg *Message) From(m genaiapi.Message) error {
	switch m.Role {
	case genaiapi.System, genaiapi.User, genaiapi.Assistant:
		msg.Role = string(m.Role)
	default:
		return fmt.Errorf("unsupported role %q", m.Role)
	}
	switch m.Type {
	case genaiapi.Text:
		msg.Content = m.Text
	default:
		return fmt.Errorf("unsupported content type %s", m.Type)
	}
	return nil
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

func (c *CompletionResponse) ToResult() (genaiapi.CompletionResult, error) {
	out := genaiapi.CompletionResult{}
	out.InputTokens = c.Usage.PromptTokens
	out.OutputTokens = c.Usage.CompletionTokens
	if len(c.Choices) != 1 {
		return out, errors.New("expected 1 choice")
	}
	out.Type = genaiapi.Text
	out.Text = c.Choices[0].Message.Content
	switch role := c.Choices[0].Message.Role; role {
	case "system", "assistant", "user":
		out.Role = genaiapi.Role(role)
	default:
		return out, fmt.Errorf("unsupported role %q", role)
	}
	return out, nil
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

func (c *Client) Completion(ctx context.Context, msgs []genaiapi.Message, opts genaiapi.Validatable) (genaiapi.CompletionResult, error) {
	// https://docs.perplexity.ai/api-reference/chat-completions
	rpcin := CompletionRequest{Model: c.model}
	if err := rpcin.Init(msgs, opts); err != nil {
		return genaiapi.CompletionResult{}, err
	}
	rpcout := CompletionResponse{}
	if err := c.CompletionRaw(ctx, &rpcin, &rpcout); err != nil {
		return genaiapi.CompletionResult{}, fmt.Errorf("failed to get chat response: %w", err)
	}
	return rpcout.ToResult()
}

func (c *Client) CompletionRaw(ctx context.Context, in *CompletionRequest, out *CompletionResponse) error {
	in.Stream = false
	return c.post(ctx, "https://api.perplexity.ai/chat/completions", in, out)
}

func (c *Client) CompletionStream(ctx context.Context, msgs []genaiapi.Message, opts genaiapi.Validatable, chunks chan<- genaiapi.MessageChunk) error {
	in := CompletionRequest{Model: c.model}
	if err := in.Init(msgs, opts); err != nil {
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
	in.Stream = true
	h := make(http.Header)
	h.Add("Authorization", "Bearer "+c.apiKey)
	resp, err := httpjson.DefaultClient.PostRequest(ctx, "https://api.perplexity.ai/chat/completions", h, in)
	if err != nil {
		return fmt.Errorf("failed to get server response: %w", err)
	}
	defer resp.Body.Close()
	for r := bufio.NewReader(resp.Body); ; {
		line, err := r.ReadBytes('\n')
		if line = bytes.TrimSpace(line); err == io.EOF {
			if len(line) == 0 {
				return nil
			}
		} else if err != nil {
			return fmt.Errorf("failed to get server response: %w", err)
		}
		if len(line) != 0 {
			if err := parseStreamLine(line, out); err != nil {
				return err
			}
		}
	}
}

func parseStreamLine(line []byte, out chan<- CompletionStreamChunkResponse) error {
	const dataPrefix = "data: "
	if !bytes.HasPrefix(line, []byte(dataPrefix)) {
		d := json.NewDecoder(bytes.NewReader(line))
		d.DisallowUnknownFields()
		d.UseNumber()
		er1 := errorResponse1{}
		if err := d.Decode(&er1); err != nil {
			d = json.NewDecoder(bytes.NewReader(line))
			d.DisallowUnknownFields()
			d.UseNumber()
			er2 := errorResponse2{}
			if err := d.Decode(&er2); err != nil {
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
	if err := d.Decode(&msg); err != nil {
		return fmt.Errorf("failed to decode server response %q: %w", string(line), err)
	}
	out <- msg
	return nil
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
			// Perplexity may return an HTML page on invalid API key.
			if bytes.HasPrefix(herr.ResponseBody, []byte("<html>")) {
				return fmt.Errorf("%w: You can get a new API key at %s", herr, apiKeyURL)
			}
			slog.WarnContext(ctx, "perplexity", "url", url, "err", err, "response", string(herr.ResponseBody), "status", herr.StatusCode)
		} else {
			slog.WarnContext(ctx, "perplexity", "url", url, "err", err)
		}
		return err
	}
}

const apiKeyURL = "https://www.perplexity.ai/settings/api"

var _ genaiapi.CompletionProvider = &Client{}
