// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Package huggingface implements a client for the HuggingFace serverless
// inference API.
//
// It is described at https://huggingface.co/docs/api-inference/
package huggingface

import (
	"bytes"
	"context"
	_ "embed"
	"encoding/json"
	"errors"
	"fmt"
	"iter"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
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

// Official python SDK: https://github.com/huggingface/huggingface_hub
//
// But the real spec source of truth is
// https://github.com/huggingface/huggingface.js/tree/main/packages/tasks/src/tasks/chat-completion

//go:embed scoreboard.json
var scoreboardJSON []byte

// Scoreboard for Huggingface.
func Scoreboard() scoreboard.Score {
	var s scoreboard.Score
	d := json.NewDecoder(bytes.NewReader(scoreboardJSON))
	d.DisallowUnknownFields()
	if err := d.Decode(&s); err != nil {
		panic(fmt.Errorf("failed to unmarshal scoreboard.json: %w", err))
	}
	return s
}

// Client implements genai.Provider.
type Client struct {
	base.NotImplemented
	impl base.Provider[*ErrorResponse, *ChatRequest, *ChatResponse, ChatStreamChunkResponse]
}

// TODO: Investigate https://huggingface.co/blog/inference-providers and https://huggingface.co/docs/inference-endpoints/

// New creates a new client to talk to the HuggingFace serverless inference API.
//
// If ProviderOptionAPIKey is not provided, it tries to load it from the HUGGINGFACE_API_KEY environment variable.
// Otherwise, it tries to load it from the huggingface python client's cache.
// If none is found, it will still return a client coupled with an base.ErrAPIKeyRequired error.
// Get your API key at https://huggingface.co/settings/tokens
//
// To use multiple models, create multiple clients.
// Use one of the tens of thousands of models to chose from at https://huggingface.co/models?inference=warm&sort=trending
//
// ProviderOptionTransportWrapper can be used to add the HTTP header "X-HF-Bill-To" via roundtrippers.Header. See
// https://huggingface.co/docs/inference-providers/pricing#organization-billing
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
	const apiKeyURL = "https://huggingface.co/settings/tokens"
	var err error
	if apiKey == "" {
		if apiKey = os.Getenv("HUGGINGFACE_API_KEY"); apiKey == "" {
			// Fallback to loading from the python client's cache.
			h, errHome := os.UserHomeDir()
			if errHome != nil {
				err = &base.ErrAPIKeyRequired{EnvVar: "HUGGINGFACE_API_KEY", URL: apiKeyURL}
			} else {
				// TODO: Windows.
				b, errRead := os.ReadFile(filepath.Join(h, ".cache", "huggingface", "token"))
				if errRead != nil {
					err = &base.ErrAPIKeyRequired{EnvVar: "HUGGINGFACE_API_KEY", URL: apiKeyURL}
				} else {
					if apiKey = strings.TrimSpace(string(b)); apiKey == "" {
						err = &base.ErrAPIKeyRequired{EnvVar: "HUGGINGFACE_API_KEY", URL: apiKeyURL}
					}
				}
			}
		}
	}
	mod := genai.Modalities{genai.ModalityText}
	if len(modalities) != 0 && !slices.Equal(modalities, mod) {
		// https://huggingface.co/docs/inference-providers/index
		return nil, fmt.Errorf("unexpected option Modalities %s, only text is implemented (send PR to add support)", mod)
	}
	t := base.DefaultTransport
	if wrapper != nil {
		t = wrapper(t)
	}
	c := &Client{
		impl: base.Provider[*ErrorResponse, *ChatRequest, *ChatResponse, ChatStreamChunkResponse]{
			GenSyncURL:      "https://router.huggingface.co/v1/chat/completions",
			ProcessStream:   ProcessStream,
			PreloadedModels: preloadedModels,
			ProcessHeaders:  processHeaders,
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
			if c.impl.Model, err = c.selectBestTextModel(ctx, model); err != nil {
				return nil, err
			}
			c.impl.OutputModalities = mod
		default:
			c.impl.Model = model
			c.impl.OutputModalities = mod
		}
	}
	return c, err
}

// selectBestTextModel selects the most recent model based on the preference (cheap, good, or SOTA).
//
// We may want to make this function overridable in the future by the client since this is going to break one
// day or another.
func (c *Client) selectBestTextModel(ctx context.Context, preference string) (string, error) {
	// Warning: listing models from Huggingface takes a while.
	mdls, err := c.ListModels(ctx)
	if err != nil {
		return "", fmt.Errorf("failed to automatically select the model: %w", err)
	}
	cheap := preference == string(genai.ModelCheap)
	good := preference == string(genai.ModelGood) || preference == ""
	selectedModel := ""
	trending := 0.
	weights := 0
	re := regexp.MustCompile(`(\d+)B`)
	for _, mdl := range mdls {
		m := mdl.(*Model)
		if m.TrendingScore < 2 {
			continue
		}
		// TODO: This algorithm would gain to be improved.
		switch {
		case cheap:
			// HF doesn't report the number of weights in the model. Try to guess it.
			matches := re.FindAllStringSubmatch(m.ID, 1)
			if len(matches) != 1 {
				continue
			}
			w, err2 := strconv.Atoi(matches[0][1])
			if err2 != nil {
				continue
			}
			if strings.HasPrefix(m.ID, "meta-llama/Llama") && strings.HasSuffix(m.ID, "-Instruct") && !strings.Contains(m.ID, "Llama-3.2-") && (weights == 0 || w < weights) {
				weights = w
				selectedModel = m.ID
			}
		case good:
			// HF doesn't report the number of weights in the model. Try to guess it.
			matches := re.FindAllStringSubmatch(m.ID, 1)
			if len(matches) != 1 {
				continue
			}
			w, err2 := strconv.Atoi(matches[0][1])
			if err2 != nil {
				continue
			}
			if strings.HasPrefix(m.ID, "Qwen/Qwen") && (weights == 0 || w > weights) {
				weights = w
				selectedModel = m.ID
			}
		default:
			if strings.HasPrefix(m.ID, "deepseek-ai/") && !strings.Contains(m.ID, "Qwen") && !strings.Contains(m.ID, "Prover") && !strings.Contains(m.ID, "Distill") && (trending == 0 || trending < m.TrendingScore) {
				// Make it a popularity contest.
				trending = m.TrendingScore
				selectedModel = m.ID
			}
		}
	}
	if selectedModel == "" {
		return "", errors.New("failed to find a model automatically")
	}
	return selectedModel, nil
}

// Name implements genai.Provider.
//
// It returns the name of the provider.
func (c *Client) Name() string {
	return "huggingface"
}

// ModelID implements genai.Provider.
//
// It returns the selected model ID.
func (c *Client) ModelID() string {
	return c.impl.Model
}

// OutputModalities implements genai.Provider.
//
// It returns the output modalities, i.e. what kind of output the model will generate (text, audio, image,
// video, etc).
func (c *Client) OutputModalities() genai.Modalities {
	return c.impl.OutputModalities
}

// Scoreboard implements genai.Provider.
func (c *Client) Scoreboard() scoreboard.Score {
	return Scoreboard()
}

// HTTPClient returns the HTTP client to fetch results (e.g. videos) generated by the provider.
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
	// https://huggingface.co/docs/hub/api
	// There's 20k models warm as of March 2025. There's no way to sort by
	// trending. Sorting by download is not useful. There's no pagination.
	var resp ModelsResponse
	if err := c.impl.DoRequest(ctx, "GET", "https://huggingface.co/api/models?inference=warm", nil, &resp); err != nil {
		return nil, err
	}
	return resp.ToModels(), nil
}

// ProcessStream converts the raw packets from the streaming API into Reply fragments.
func ProcessStream(chunks iter.Seq[ChatStreamChunkResponse]) (iter.Seq[genai.Reply], func() (genai.Usage, [][]genai.Logprob, error)) {
	var finalErr error
	u := genai.Usage{}
	var l [][]genai.Logprob

	return func(yield func(genai.Reply) bool) {
			pendingToolCall := ToolCall{}
			for pkt := range chunks {
				if pkt.Usage.PromptTokens != 0 {
					u.InputTokens = pkt.Usage.PromptTokens
					u.InputCachedTokens = pkt.Usage.PromptTokensDetails.CachedTokens
					u.ReasoningTokens = pkt.Usage.CompletionTokensDetails.ReasoningTokens
					u.OutputTokens = pkt.Usage.CompletionTokens
					u.TotalTokens = pkt.Usage.TotalTokens
				}
				if len(pkt.Choices) != 1 {
					continue
				}
				if pkt.Choices[0].FinishReason != "" {
					u.FinishReason = pkt.Choices[0].FinishReason.ToFinishReason()
				}
				if !pkt.Choices[0].Logprobs.IsZero() {
					l = append(l, pkt.Choices[0].Logprobs.To()...)
				}
				switch role := pkt.Choices[0].Delta.Role; role {
				case "assistant", "":
				default:
					finalErr = &internal.BadError{Err: fmt.Errorf("unexpected role %q", role)}
					return
				}
				// There's only one at a time ever.
				if len(pkt.Choices[0].Delta.ToolCalls) > 1 {
					finalErr = &internal.BadError{Err: fmt.Errorf("implement multiple tool calls: %#v", pkt.Choices[0].Delta.ToolCalls)}
					return
				}
				f := genai.Reply{Text: pkt.Choices[0].Delta.Content}
				// Huggingface streams the arguments. Buffer the arguments to send the fragment as a whole tool call.
				if len(pkt.Choices[0].Delta.ToolCalls) == 1 {
					// ID is not consistently set. Use Name for now but that's risky.
					if t := pkt.Choices[0].Delta.ToolCalls[0]; t.Function.Name == pendingToolCall.Function.Name {
						// Continuation.
						pendingToolCall.Function.Arguments += t.Function.Arguments
						if !f.IsZero() {
							finalErr = &internal.BadError{Err: fmt.Errorf("implement tool call with metadata: %#v", pkt)}
							return
						}
						continue
					} else {
						// A new call.
						if pendingToolCall.Function.Name == "" {
							pendingToolCall = t
							if !f.IsZero() {
								finalErr = &internal.BadError{Err: fmt.Errorf("implement tool call with metadata: %#v", pkt)}
								return
							}
							continue
						}
						// Flush.
						pendingToolCall.To(&f.ToolCall)
						pendingToolCall = ToolCall{}
					}
				} else if pendingToolCall.Function.Name != "" {
					// Flush.
					pendingToolCall.To(&f.ToolCall)
					pendingToolCall = ToolCall{}
				}
				if !yield(f) {
					return
				}
			}
			// Hugginface doesn't send an "ending" packet, FinishReason isn't even set on the last packet.
			if pendingToolCall.Function.Name != "" {
				// Flush.
				f := genai.Reply{}
				pendingToolCall.To(&f.ToolCall)
				if !yield(f) {
					return
				}
			}
		}, func() (genai.Usage, [][]genai.Logprob, error) {
			return u, l, finalErr
		}
}

func processHeaders(h http.Header) []genai.RateLimit {
	var limits []genai.RateLimit
	requestsLimit, _ := strconv.ParseInt(h.Get("X-Ratelimit-Limit-Requests"), 10, 64)
	requestsRemaining, _ := strconv.ParseInt(h.Get("X-Ratelimit-Remaining-Requests"), 10, 64)
	tokensLimit, _ := strconv.ParseInt(h.Get("X-Ratelimit-Limit-Tokens"), 10, 64)
	tokensRemaining, _ := strconv.ParseInt(h.Get("X-Ratelimit-Remaining-Tokens"), 10, 64)
	reset, _ := time.ParseDuration(h.Get("X-Ratelimit-Dynamic-Period-Remaining"))

	if requestsLimit > 0 {
		limits = append(limits, genai.RateLimit{
			Type:      genai.Requests,
			Period:    genai.PerOther,
			Limit:     requestsLimit,
			Remaining: requestsRemaining,
			Reset:     time.Now().Add(reset).Round(10 * time.Millisecond),
		})
	}
	if tokensLimit > 0 {
		limits = append(limits, genai.RateLimit{
			Type:      genai.Tokens,
			Period:    genai.PerOther,
			Limit:     tokensLimit,
			Remaining: tokensRemaining,
			Reset:     time.Now().Add(reset).Round(10 * time.Millisecond),
		})
	}
	return limits
}

var _ genai.Provider = &Client{}
