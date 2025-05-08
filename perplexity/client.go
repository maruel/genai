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
	"github.com/maruel/genai"
	"github.com/maruel/genai/internal"
	"github.com/maruel/httpjson"
	"github.com/maruel/roundtrippers"
	"golang.org/x/sync/errgroup"
)

// https://docs.perplexity.ai/api-reference/chat-completions
type ChatRequest struct {
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

// Init initializes the provider specific completion request with the generic completion request.
func (c *ChatRequest) Init(msgs genai.Messages, opts genai.Validatable, model string) error {
	c.Model = model
	var errs []error
	var unsupported []string
	sp := ""
	if opts != nil {
		if err := opts.Validate(); err != nil {
			errs = append(errs, err)
		} else {
			switch v := opts.(type) {
			case *genai.ChatOptions:
				c.MaxTokens = v.MaxTokens
				c.Temperature = v.Temperature
				c.TopP = v.TopP
				sp = v.SystemPrompt
				if v.Seed != 0 {
					unsupported = append(unsupported, "Seed")
				}
				c.TopK = v.TopK
				if len(v.Stop) != 0 {
					unsupported = append(unsupported, "Stop tokens")
				}
				if v.DecodeAs != nil {
					// Requires Tier 3 to work in practice.
					c.ResponseFormat.Type = "json_schema"
					c.ResponseFormat.JSONSchema.Schema = jsonschema.Reflect(v.DecodeAs)
				} else if v.ReplyAsJSON {
					unsupported = append(unsupported, "ReplyAsJSON")
				}
				if len(v.Tools) != 0 {
					unsupported = append(unsupported, "Tools")
				}
				if v.ThinkingBudget > 0 {
					unsupported = append(unsupported, "ThinkingBudget")
				}
			default:
				errs = append(errs, fmt.Errorf("unsupported options type %T", opts))
			}
		}
	}

	if err := msgs.Validate(); err != nil {
		errs = append(errs, err)
	} else {
		offset := 0
		if sp != "" {
			offset = 1
		}
		c.Messages = make([]Message, len(msgs)+offset)
		if sp != "" {
			c.Messages[0].Role = "system"
			c.Messages[0].Content = sp
		}
		for i := range msgs {
			if err := c.Messages[i+offset].From(&msgs[i]); err != nil {
				errs = append(errs, fmt.Errorf("message %d: %w", i, err))
			}
		}
	}
	if len(unsupported) > 0 {
		// If we have unsupported features but no other errors, return a continuable error
		if len(errs) == 0 {
			return &genai.UnsupportedContinuableError{Unsupported: unsupported}
		}
		// Otherwise, add the unsupported features to the error list
		errs = append(errs, &genai.UnsupportedContinuableError{Unsupported: unsupported})
	}
	return errors.Join(errs...)
}

// https://docs.perplexity.ai/api-reference/chat-completions
type Message struct {
	Role    string `json:"role"` // "system", "assistant", "user"
	Content string `json:"content"`
}

func (m *Message) From(in *genai.Message) error {
	switch in.Role {
	case genai.User, genai.Assistant:
		m.Role = string(in.Role)
	default:
		return fmt.Errorf("unsupported role %q", in.Role)
	}
	if len(in.Contents) > 1 {
		return errors.New("perplexity doesn't support multiple content blocks; TODO split transparently")
	}
	if len(in.ToolCalls) != 0 {
		return errors.New("perplexity doesn't support tools")
	}
	if in.Contents[0].Text != "" {
		m.Content = in.Contents[0].Text
	} else {
		return fmt.Errorf("unsupported content type %v", in.Contents[0])
	}
	return nil
}

func (m *Message) To(out *genai.Message) error {
	switch role := m.Role; role {
	case "assistant":
		out.Role = genai.Role(role)
	default:
		return fmt.Errorf("unsupported role %q", role)
	}
	out.Contents = []genai.Content{{Text: m.Content}}
	return nil
}

type ChatResponse struct {
	ID        string   `json:"id"`
	Model     string   `json:"model"`
	Object    string   `json:"object"`
	Created   Time     `json:"created"`
	Citations []string `json:"citations"`
	Choices   []struct {
		Index        int64   `json:"index"`
		FinishReason string  `json:"finish_reason"` // stop, length
		Message      Message `json:"message"`
		Delta        struct {
			Content string `json:"content"`
			Role    string `json:"role"`
		} `json:"delta"`
	} `json:"choices"`
	Usage struct {
		PromptTokens int64 `json:"prompt_tokens"`
		ChatTokens   int64 `json:"completion_tokens"`
		TotalTokens  int64 `json:"total_tokens"`
	} `json:"usage"`
}

func (c *ChatResponse) ToResult() (genai.ChatResult, error) {
	out := genai.ChatResult{
		Usage: genai.Usage{
			InputTokens:  c.Usage.PromptTokens,
			OutputTokens: c.Usage.ChatTokens,
		},
	}
	if len(c.Choices) != 1 {
		return out, errors.New("expected 1 choice")
	}
	out.FinishReason = c.Choices[0].FinishReason
	err := c.Choices[0].Message.To(&out.Message)
	return out, err
}

type ChatStreamChunkResponse = ChatResponse

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
	// Client is exported for testing replay purposes.
	Client httpjson.Client

	model string
}

// New creates a new client to talk to the Perplexity platform API.
//
// If apiKey is not provided, it tries to load it from the PERPLEXITY_API_KEY environment variable.
// If none is found, it returns an error.
// Get your API key at https://www.perplexity.ai/settings/api
//
// Models are listed at https://docs.perplexity.ai/guides/model-cards
func New(apiKey, model string) (*Client, error) {
	if apiKey == "" {
		if apiKey = os.Getenv("PERPLEXITY_API_KEY"); apiKey == "" {
			return nil, errors.New("perplexity API key is required; get one at " + apiKeyURL)
		}
	}
	return &Client{
		model: model,
		Client: httpjson.Client{
			Client: &http.Client{Transport: &roundtrippers.Header{
				Transport: &roundtrippers.Retry{Transport: http.DefaultTransport},
				Header:    http.Header{"Authorization": {"Bearer " + apiKey}},
			}},
			Lenient: internal.BeLenient,
		},
	}, nil
}

func (c *Client) Chat(ctx context.Context, msgs genai.Messages, opts genai.Validatable) (genai.ChatResult, error) {
	// https://docs.perplexity.ai/api-reference/chat-completions
	rpcin := ChatRequest{}
	var continuableErr error
	if err := rpcin.Init(msgs, opts, c.model); err != nil {
		// If it's an UnsupportedContinuableError, we can continue
		if uce, ok := err.(*genai.UnsupportedContinuableError); ok {
			// Store the error to return later if no other error occurs
			continuableErr = uce
			// Otherwise log the error but continue
		} else {
			return genai.ChatResult{}, err
		}
	}
	rpcout := ChatResponse{}
	if err := c.ChatRaw(ctx, &rpcin, &rpcout); err != nil {
		return genai.ChatResult{}, fmt.Errorf("failed to get chat response: %w", err)
	}
	result, err := rpcout.ToResult()
	if err != nil {
		return result, err
	}
	// Return the continuable error if no other error occurred
	if continuableErr != nil {
		return result, continuableErr
	}
	return result, nil
}

func (c *Client) ChatRaw(ctx context.Context, in *ChatRequest, out *ChatResponse) error {
	in.Stream = false
	return c.post(ctx, "https://api.perplexity.ai/chat/completions", in, out)
}

func (c *Client) ChatStream(ctx context.Context, msgs genai.Messages, opts genai.Validatable, chunks chan<- genai.MessageFragment) error {
	in := ChatRequest{}
	var continuableErr error
	if err := in.Init(msgs, opts, c.model); err != nil {
		// If it's an UnsupportedContinuableError, we can continue
		if uce, ok := err.(*genai.UnsupportedContinuableError); ok {
			// Store the error to return later if no other error occurs
			continuableErr = uce
			// Otherwise log the error but continue
		} else {
			return err
		}
	}
	ch := make(chan ChatStreamChunkResponse)
	eg, ctx := errgroup.WithContext(ctx)
	eg.Go(func() error {
		return processStreamPackets(ch, chunks)
	})
	err := c.ChatStreamRaw(ctx, &in, ch)
	close(ch)
	if err2 := eg.Wait(); err2 != nil {
		err = err2
	}
	// Return the continuable error if no other error occurred
	if err == nil && continuableErr != nil {
		return continuableErr
	}
	return err
}

func processStreamPackets(ch <-chan ChatStreamChunkResponse, chunks chan<- genai.MessageFragment) error {
	defer func() {
		// We need to empty the channel to avoid blocking the goroutine.
		for range ch {
		}
	}()
	for pkt := range ch {
		if len(pkt.Choices) != 1 {
			continue
		}
		switch role := pkt.Choices[0].Delta.Role; role {
		case "", "assistant":
		default:
			return fmt.Errorf("unexpected role %q", role)
		}
		if word := pkt.Choices[0].Delta.Content; word != "" {
			fragment := genai.MessageFragment{TextFragment: word}
			// Include FinishReason if available
			if pkt.Choices[0].FinishReason != "" {
				fragment.FinishReason = pkt.Choices[0].FinishReason
			}
			chunks <- fragment
		}
	}
	return nil
}

func (c *Client) ChatStreamRaw(ctx context.Context, in *ChatRequest, out chan<- ChatStreamChunkResponse) error {
	in.Stream = true
	resp, err := c.Client.PostRequest(ctx, "https://api.perplexity.ai/chat/completions", nil, in)
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

func parseStreamLine(line []byte, out chan<- ChatStreamChunkResponse) error {
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
	msg := ChatStreamChunkResponse{}
	if err := d.Decode(&msg); err != nil {
		return fmt.Errorf("failed to decode server response %q: %w", string(line), err)
	}
	out <- msg
	return nil
}

func (c *Client) post(ctx context.Context, url string, in, out any) error {
	resp, err := c.Client.PostRequest(ctx, url, nil, in)
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

var _ genai.ChatProvider = &Client{}
