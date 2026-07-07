// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Package github implements a client for the GitHub Models API.
//
// It is described at https://docs.github.com/en/rest/models
package github

import (
	"bytes"
	"context"
	_ "embed"
	"encoding/json"
	"fmt"
	"iter"
	"net/http"
	"os"
	"os/exec"
	"slices"
	"strconv"
	"strings"
	"time"

	"github.com/maruel/roundtrippers"

	"github.com/maruel/genai"
	"github.com/maruel/genai/base"
	"github.com/maruel/genai/internal"
	"github.com/maruel/genai/scoreboard"
)

//go:embed scoreboard.json
var scoreboardJSON []byte

// Scoreboard for GitHub Models.
func Scoreboard() scoreboard.Score {
	var s scoreboard.Score
	d := json.NewDecoder(bytes.NewReader(scoreboardJSON))
	d.DisallowUnknownFields()
	if err := d.Decode(&s); err != nil {
		panic(fmt.Errorf("failed to unmarshal scoreboard.json: %w", err))
	}
	return s
}

//

// Client implements genai.Provider.
type Client struct {
	base.NotImplemented
	impl base.Provider[*ErrorResponse, *ChatRequest, *ChatResponse, ChatStreamChunkResponse]
}

// getGHToken retrieves the GitHub token from the gh CLI if available.
func getGHToken(ctx context.Context) (string, error) {
	cmd := exec.CommandContext(ctx, "gh", "auth", "token")
	out, err := cmd.Output()
	if err != nil {
		return "", err
	}
	return strings.TrimSpace(string(out)), nil
}

// New creates a new client to talk to the GitHub Models API.
//
// If apiKey is not provided via ProviderOptionAPIKey, it tries to load it from the GITHUB_TOKEN environment
// variable. If that fails, it attempts to get the token from the gh CLI. If none is found, it will still
// return a client coupled with a base.ErrAPIKeyRequired error.
// Get a token at https://github.com/settings/tokens (needs models:read scope) or use the default
// GITHUB_TOKEN in Actions.
//
// To use multiple models, create multiple clients.
// Use one of the models from https://github.com/marketplace/models
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
	const apiKeyURL = "https://github.com/settings/tokens"
	var err error
	if apiKey == "" {
		if apiKey = os.Getenv("GITHUB_TOKEN"); apiKey == "" {
			// Try to get token from gh CLI
			if token, err2 := getGHToken(ctx); err2 == nil {
				apiKey = token
			} else {
				err = &base.ErrAPIKeyRequired{EnvVar: "GITHUB_TOKEN", URL: apiKeyURL}
			}
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
			GenSyncURL:      "https://models.github.ai/inference/chat/completions",
			ProcessStream:   ProcessStream,
			ProcessHeaders:  processHeaders,
			PreloadedModels: preloadedModels,
			ProviderBase: base.ProviderBase[*ErrorResponse]{
				APIKeyURL: apiKeyURL,
				Lenient:   internal.BeLenient,
				Client: http.Client{
					Transport: &roundtrippers.Header{
						Header: http.Header{
							"Authorization":        {"Bearer " + apiKey},
							"Accept":               {"application/vnd.github+json"},
							"X-GitHub-Api-Version": {"2026-03-10"},
						},
						Transport: &roundtrippers.RequestID{Transport: t},
					},
				},
			},
		},
	}
	if err == nil {
		switch model {
		case "":
		case string(genai.ModelCheap):
			c.impl.Model = "openai/gpt-4.1-nano"
			c.impl.OutputModalities = mod
		case string(genai.ModelGood):
			c.impl.Model = "openai/gpt-4.1-mini"
			c.impl.OutputModalities = mod
		case string(genai.ModelSOTA):
			c.impl.Model = "openai/gpt-4.1"
			c.impl.OutputModalities = mod
		default:
			c.impl.Model = model
			c.impl.OutputModalities = mod
		}
	}
	return c, err
}

// Name implements genai.Provider.
func (c *Client) Name() string {
	return "github"
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

// HTTPClient returns the HTTP client to fetch results generated by the provider.
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
	// https://docs.github.com/en/rest/models/catalog
	var resp []CatalogModel
	if err := c.impl.DoRequest(ctx, "GET", "https://models.github.ai/catalog/models", nil, &resp); err != nil {
		return nil, err
	}
	models := make([]genai.Model, len(resp))
	for i := range resp {
		models[i] = &resp[i]
	}
	return models, nil
}

// ProcessStream converts the raw packets from the streaming API into Reply fragments.
func ProcessStream(chunks iter.Seq[ChatStreamChunkResponse]) (iter.Seq[genai.Reply], func() (genai.Usage, [][]genai.Logprob, error)) {
	var finalErr error
	u := genai.Usage{}

	return func(yield func(genai.Reply) bool) {
			pendingToolCall := ToolCall{}
			for pkt := range chunks {
				if len(pkt.Choices) == 0 {
					// Usage-only packet at end of stream.
					if pkt.Usage.PromptTokens != 0 || pkt.Usage.CompletionTokens != 0 {
						u.InputTokens = pkt.Usage.PromptTokens
						u.OutputTokens = pkt.Usage.CompletionTokens
						u.TotalTokens = pkt.Usage.TotalTokens
						if u.TotalTokens == 0 {
							u.TotalTokens = u.InputTokens + u.OutputTokens
						}
						u.InputCachedTokens = pkt.Usage.PromptTokensDetails.CachedTokens
						u.ReasoningTokens = pkt.Usage.CompletionTokensDetails.ReasoningTokens
					}
					continue
				}
				if len(pkt.Choices) != 1 {
					continue
				}
				switch role := pkt.Choices[0].Delta.Role; role {
				case "", "assistant":
				default:
					finalErr = &internal.BadError{Err: fmt.Errorf("unexpected role %q", role)}
					return
				}
				if pkt.Usage.PromptTokens != 0 || pkt.Usage.CompletionTokens != 0 {
					u.InputTokens = pkt.Usage.PromptTokens
					u.OutputTokens = pkt.Usage.CompletionTokens
					u.TotalTokens = pkt.Usage.TotalTokens
					if u.TotalTokens == 0 {
						u.TotalTokens = u.InputTokens + u.OutputTokens
					}
					u.InputCachedTokens = pkt.Usage.PromptTokensDetails.CachedTokens
					u.ReasoningTokens = pkt.Usage.CompletionTokensDetails.ReasoningTokens
					if len(pkt.Choices) > 0 {
						u.FinishReason = pkt.Choices[0].FinishReason.ToFinishReason()
					}
				}
				for _, nt := range pkt.Choices[0].Delta.ToolCalls {
					if pendingToolCall.ID != "" {
						if nt.ID == "" {
							// Continuation of current tool call arguments.
							pendingToolCall.Function.Arguments += nt.Function.Arguments
						} else {
							// New tool call: flush the previous one.
							f := genai.Reply{}
							pendingToolCall.To(&f.ToolCall)
							if !yield(f) {
								return
							}
							pendingToolCall = nt
						}
					} else {
						pendingToolCall = nt
					}
				}
				for _, content := range pkt.Choices[0].Delta.Content {
					switch content.Type {
					case ContentText:
						if !yield(genai.Reply{Text: content.Text}) {
							return
						}
					default:
						finalErr = &internal.BadError{Err: fmt.Errorf("implement content type %q", content.Type)}
						return
					}
				}
			}
			if pendingToolCall.ID != "" {
				f := genai.Reply{}
				pendingToolCall.To(&f.ToolCall)
				if !yield(f) {
					return
				}
			}
		}, func() (genai.Usage, [][]genai.Logprob, error) {
			return u, nil, finalErr
		}
}

func processHeaders(h http.Header) []genai.RateLimit {
	// GitHub Models uses standard GitHub rate limit headers.
	requestsLimit, _ := strconv.ParseInt(h.Get("X-RateLimit-Limit"), 10, 64)
	requestsRemaining, _ := strconv.ParseInt(h.Get("X-RateLimit-Remaining"), 10, 64)
	resetUnix, _ := strconv.ParseInt(h.Get("X-RateLimit-Reset"), 10, 64)

	var limits []genai.RateLimit
	if requestsLimit > 0 {
		var reset time.Time
		if resetUnix > 0 {
			reset = time.Unix(resetUnix, 0)
		}
		limits = append(limits, genai.RateLimit{
			Type:      genai.Requests,
			Period:    genai.PerOther,
			Limit:     requestsLimit,
			Remaining: requestsRemaining,
			Reset:     reset,
		})
	}
	return limits
}

var _ genai.Provider = &Client{}
