// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Package openrouter implements a client for the OpenRouter API.
//
// OpenRouter is a unified interface to 300+ LLM models from 60+ providers. It provides
// an OpenAI-compatible API with additional features like provider routing, model fallbacks,
// and transparent pricing.
//
// It is described at https://openrouter.ai/docs/api/reference/overview
package openrouter

import (
	"bytes"
	"context"
	_ "embed"
	"encoding/json"
	"fmt"
	"iter"
	"net/http"
	"os"
	"slices"

	"github.com/maruel/roundtrippers"

	"github.com/maruel/genai"
	"github.com/maruel/genai/base"
	"github.com/maruel/genai/internal"
	"github.com/maruel/genai/scoreboard"
)

//go:embed scoreboard.json
var scoreboardJSON []byte

// Scoreboard for OpenRouter.
func Scoreboard() scoreboard.Score {
	var s scoreboard.Score
	d := json.NewDecoder(bytes.NewReader(scoreboardJSON))
	d.DisallowUnknownFields()
	if err := d.Decode(&s); err != nil {
		panic(fmt.Errorf("failed to unmarshal scoreboard.json: %w", err))
	}
	return s
}

// GenOption is the OpenRouter-specific options.
type GenOption struct {
	// Models specifies fallback models to try if the primary model fails.
	// OpenRouter will try each model in order on failure (rate limit, content moderation, etc).
	Models []string
	// Route specifies the routing strategy. Use "fallback" to try models in order.
	Route Route
	// Provider configures provider routing preferences.
	Provider *ProviderPreferences
}

// Validate implements genai.Validatable.
func (o *GenOption) Validate() error {
	return nil
}

// Route is the routing strategy for multi-model requests.
type Route string

const (
	// RouteFallback tries providers in order, falling back on failure.
	RouteFallback Route = "fallback"
)

// Client implements genai.Provider for OpenRouter.
type Client struct {
	base.NotImplemented
	impl base.Provider[*ErrorResponse, *ChatRequest, *ChatResponse, ChatStreamChunkResponse]
}

// New creates a new client to talk to the OpenRouter API.
//
// If apiKey is not provided, it tries to load it from the OPENROUTER_API_KEY environment variable.
// If none is found, it will still return a client coupled with a base.ErrAPIKeyRequired error.
// Get your API key at https://openrouter.ai/settings/keys
//
// OpenRouter model IDs use the format "provider/model", e.g. "anthropic/claude-3.5-sonnet".
// See https://openrouter.ai/models for available models.
func New(ctx context.Context, opts ...genai.ProviderOption) (*Client, error) {
	var apiKey, model string
	var modalities genai.Modalities
	var preloadedModels []genai.Model
	var wrapper func(http.RoundTripper) http.RoundTripper
	if err := base.CheckDuplicateOptions(opts); err != nil {
		return nil, err
	}
	for _, opt := range opts {
		if err := opt.Validate(); err != nil {
			return nil, err
		}
		switch v := opt.(type) {
		case genai.ProviderOptionAPIKey:
			apiKey = string(v)
		case genai.ProviderOptionModel:
			model = string(v)
		case genai.ProviderOptionModalities:
			modalities = genai.Modalities(v)
		case genai.ProviderOptionPreloadedModels:
			preloadedModels = []genai.Model(v)
		case genai.ProviderOptionTransportWrapper:
			wrapper = v
		default:
			return nil, fmt.Errorf("unsupported option type %T", opt)
		}
	}
	const apiKeyURL = "https://openrouter.ai/settings/keys"
	var err error
	if apiKey == "" {
		if apiKey = os.Getenv("OPENROUTER_API_KEY"); apiKey == "" {
			err = &base.ErrAPIKeyRequired{EnvVar: "OPENROUTER_API_KEY", URL: apiKeyURL}
		}
	}
	mod := genai.Modalities{genai.ModalityText}
	if len(modalities) != 0 && !slices.Equal(modalities, mod) {
		return nil, fmt.Errorf("unexpected option Modalities %s, only text is supported", mod)
	}
	t := base.DefaultTransport
	if wrapper != nil {
		t = wrapper(t)
	}
	c := &Client{
		impl: base.Provider[*ErrorResponse, *ChatRequest, *ChatResponse, ChatStreamChunkResponse]{
			GenSyncURL:      "https://openrouter.ai/api/v1/chat/completions",
			ProcessStream:   ProcessStream,
			PreloadedModels: preloadedModels,
			ProviderBase: base.ProviderBase[*ErrorResponse]{
				APIKeyURL: apiKeyURL,
				Lenient:   internal.BeLenient,
				Client: http.Client{
					Transport: &roundtrippers.Header{
						Header:    http.Header{"Authorization": {"Bearer " + apiKey}},
						Transport: &roundtrippers.RequestID{Transport: t},
					},
				},
			},
		},
	}
	if err == nil {
		switch model {
		case "":
		case string(genai.ModelCheap), string(genai.ModelGood), string(genai.ModelSOTA):
			c.impl.Model = c.selectBestTextModel(model)
			c.impl.OutputModalities = mod
		default:
			c.impl.Model = model
			c.impl.OutputModalities = mod
		}
	}
	return c, err
}

// selectBestTextModel selects the most appropriate model based on the preference.
func (c *Client) selectBestTextModel(preference string) string {
	switch preference {
	case string(genai.ModelCheap):
		return "qwen/qwen3.5-35b-a3b"
	case string(genai.ModelGood):
		return "qwen/qwen3.5-122b-a10b"
	default:
		// SOTA
		return "qwen/qwen3.5-397b-a17b"
	}
}

// Name implements genai.Provider.
func (c *Client) Name() string {
	return "openrouter"
}

// ModelID implements genai.Provider.
func (c *Client) ModelID() string {
	return c.impl.Model
}

// OutputModalities implements genai.Provider.
func (c *Client) OutputModalities() genai.Modalities {
	return c.impl.OutputModalities
}

// Scoreboard implements genai.Provider.
func (c *Client) Scoreboard() scoreboard.Score {
	return Scoreboard()
}

// HTTPClient returns the HTTP client.
func (c *Client) HTTPClient() *http.Client {
	return &c.impl.Client
}

// GenSync implements genai.Provider.
func (c *Client) GenSync(ctx context.Context, msgs genai.Messages, opts ...genai.GenOption) (genai.Result, error) {
	return c.impl.GenSync(ctx, msgs, opts...)
}

// GenSyncRaw provides access to the raw API.
func (c *Client) GenSyncRaw(ctx context.Context, in *ChatRequest, out *ChatResponse) error {
	return c.impl.GenSyncRaw(ctx, in, out)
}

// GenStream implements genai.Provider.
func (c *Client) GenStream(ctx context.Context, msgs genai.Messages, opts ...genai.GenOption) (iter.Seq[genai.Reply], func() (genai.Result, error)) {
	return c.impl.GenStream(ctx, msgs, opts...)
}

// GenStreamRaw provides access to the raw API.
func (c *Client) GenStreamRaw(ctx context.Context, in *ChatRequest) (iter.Seq[ChatStreamChunkResponse], func() error) {
	return c.impl.GenStreamRaw(ctx, in)
}

// ListModels implements genai.Provider.
func (c *Client) ListModels(ctx context.Context) ([]genai.Model, error) {
	if c.impl.PreloadedModels != nil {
		return c.impl.PreloadedModels, nil
	}
	var resp ModelsResponse
	if err := c.impl.DoRequest(ctx, "GET", "https://openrouter.ai/api/v1/models", nil, &resp); err != nil {
		return nil, err
	}
	return resp.ToModels(), nil
}

// ProcessStream converts the raw packets from the streaming API into Reply fragments.
func ProcessStream(chunks iter.Seq[ChatStreamChunkResponse]) (iter.Seq[genai.Reply], func() (genai.Usage, [][]genai.Logprob, error)) {
	var finalErr error
	var logprobs [][]genai.Logprob
	u := genai.Usage{}

	return func(yield func(genai.Reply) bool) {
			pendingToolCall := ToolCall{}
			for pkt := range chunks {
				if len(pkt.Choices) != 1 {
					continue
				}
				if pkt.Usage.TotalTokens != 0 {
					u.InputTokens = pkt.Usage.PromptTokens
					u.InputCachedTokens = pkt.Usage.PromptTokensDetails.CachedTokens
					u.ReasoningTokens = pkt.Usage.CompletionTokensDetails.ReasoningTokens
					u.OutputTokens = pkt.Usage.CompletionTokens
					u.TotalTokens = pkt.Usage.TotalTokens
				}
				if pkt.ServiceTier != "" {
					u.ServiceTier = pkt.ServiceTier
				}
				if pkt.Choices[0].FinishReason != "" {
					u.FinishReason = pkt.Choices[0].FinishReason.ToFinishReason()
				}
				if pkt.Choices[0].Logprobs != nil {
					logprobs = append(logprobs, pkt.Choices[0].Logprobs.To()...)
				}
				switch role := pkt.Choices[0].Delta.Role; role {
				case "assistant", "":
				default:
					finalErr = &internal.BadError{Err: fmt.Errorf("unexpected role %q", role)}
					return
				}
				f := genai.Reply{}
				for _, c := range pkt.Choices[0].Delta.Content {
					switch c.Type {
					case ContentText:
						f.Text += c.Text
					default:
						finalErr = &internal.BadError{Err: fmt.Errorf("implement content type %q", c.Type)}
						return
					}
				}
				// OpenRouter streams tool call arguments incrementally. Buffer them and
				// yield the complete tool call once a new call starts or the stream ends.
				if len(pkt.Choices[0].Delta.ToolCalls) > 1 {
					finalErr = &internal.BadError{Err: fmt.Errorf("implement multiple tool calls: %#v", pkt)}
					return
				}
				if len(pkt.Choices[0].Delta.ToolCalls) == 1 {
					t := pkt.Choices[0].Delta.ToolCalls[0]
					if t.ID != "" {
						// A new call.
						if pendingToolCall.ID == "" {
							pendingToolCall = t
							if !f.IsZero() {
								if !yield(f) {
									return
								}
							}
							continue
						}
						// Flush previous.
						pendingToolCall.To(&f.ToolCall)
						pendingToolCall = t
					} else if pendingToolCall.ID != "" {
						// Continuation: accumulate arguments.
						pendingToolCall.Function.Arguments += t.Function.Arguments
						if !f.IsZero() {
							finalErr = &internal.BadError{Err: fmt.Errorf("implement tool call with metadata: %#v", pkt)}
							return
						}
						continue
					}
				} else if pendingToolCall.ID != "" {
					// Flush pending tool call.
					pendingToolCall.To(&f.ToolCall)
					pendingToolCall = ToolCall{}
				}
				if !f.IsZero() {
					if !yield(f) {
						return
					}
				}
			}
		}, func() (genai.Usage, [][]genai.Logprob, error) {
			return u, logprobs, finalErr
		}
}

var _ genai.Provider = &Client{}
