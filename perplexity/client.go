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
	"golang.org/x/sync/errgroup"
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

// Init initializes the provider specific completion request with the generic completion request.
func (c *CompletionRequest) Init(msgs genai.Messages, opts genai.Validatable) error {
	var errs []error
	sp := ""
	if opts != nil {
		if err := opts.Validate(); err != nil {
			errs = append(errs, err)
		} else {
			switch v := opts.(type) {
			case *genai.CompletionOptions:
				c.MaxTokens = v.MaxTokens
				c.Temperature = v.Temperature
				c.TopP = v.TopP
				sp = v.SystemPrompt
				if v.Seed != 0 {
					errs = append(errs, errors.New("perplexity doesn't support seed"))
				}
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

type CompletionResponse struct {
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
		PromptTokens     int64 `json:"prompt_tokens"`
		CompletionTokens int64 `json:"completion_tokens"`
		TotalTokens      int64 `json:"total_tokens"`
	} `json:"usage"`
}

func (c *CompletionResponse) ToResult() (genai.CompletionResult, error) {
	out := genai.CompletionResult{
		Usage: genai.Usage{
			InputTokens:  c.Usage.PromptTokens,
			OutputTokens: c.Usage.CompletionTokens,
		},
	}
	if len(c.Choices) != 1 {
		return out, errors.New("expected 1 choice")
	}
	err := c.Choices[0].Message.To(&out.Message)
	return out, err
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
	model string
	c     httpjson.Client
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
		c: httpjson.Client{
			Client: &http.Client{
				Transport: &internal.TransportHeaders{
					R: http.DefaultTransport,
					H: map[string]string{"Authorization": "Bearer " + apiKey},
				},
			},
			// Perplexity doesn't support HTTP POST compression.
			PostCompress: "",
		},
	}, nil
}

func (c *Client) Completion(ctx context.Context, msgs genai.Messages, opts genai.Validatable) (genai.CompletionResult, error) {
	// https://docs.perplexity.ai/api-reference/chat-completions
	rpcin := CompletionRequest{Model: c.model}
	if err := rpcin.Init(msgs, opts); err != nil {
		return genai.CompletionResult{}, err
	}
	rpcout := CompletionResponse{}
	if err := c.CompletionRaw(ctx, &rpcin, &rpcout); err != nil {
		return genai.CompletionResult{}, fmt.Errorf("failed to get chat response: %w", err)
	}
	return rpcout.ToResult()
}

func (c *Client) CompletionRaw(ctx context.Context, in *CompletionRequest, out *CompletionResponse) error {
	in.Stream = false
	return c.post(ctx, "https://api.perplexity.ai/chat/completions", in, out)
}

func (c *Client) CompletionStream(ctx context.Context, msgs genai.Messages, opts genai.Validatable, chunks chan<- genai.MessageFragment) error {
	in := CompletionRequest{Model: c.model}
	if err := in.Init(msgs, opts); err != nil {
		return err
	}
	ch := make(chan CompletionStreamChunkResponse)
	eg, ctx := errgroup.WithContext(ctx)
	eg.Go(func() error {
		return processStreamPackets(ch, chunks)
	})
	err := c.CompletionStreamRaw(ctx, &in, ch)
	close(ch)
	if err2 := eg.Wait(); err2 != nil {
		err = err2
	}
	return err
}

func processStreamPackets(ch <-chan CompletionStreamChunkResponse, chunks chan<- genai.MessageFragment) error {
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
			chunks <- genai.MessageFragment{TextFragment: word}
		}
	}
	return nil
}

func (c *Client) CompletionStreamRaw(ctx context.Context, in *CompletionRequest, out chan<- CompletionStreamChunkResponse) error {
	in.Stream = true
	resp, err := c.c.PostRequest(ctx, "https://api.perplexity.ai/chat/completions", nil, in)
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
	resp, err := c.c.PostRequest(ctx, url, nil, in)
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

var _ genai.CompletionProvider = &Client{}
