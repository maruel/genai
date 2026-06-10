// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Package alibaba implements a client for the Alibaba Cloud DashScope API (Model Studio).
//
// It uses the OpenAI-compatible endpoint documented at
// https://www.alibabacloud.com/help/en/model-studio/compatibility-of-openai-with-dashscope
//
// Regional endpoints:
//   - International: https://dashscope-intl.aliyuncs.com/compatible-mode/v1
//   - US: https://dashscope-us.aliyuncs.com/compatible-mode/v1
//   - China: https://dashscope.aliyuncs.com/compatible-mode/v1
package alibaba

import (
	"bytes"
	"context"
	_ "embed"
	"encoding/json"
	"errors"
	"fmt"
	"iter"
	"math"
	"net/http"
	"os"
	"regexp"
	"slices"
	"strconv"

	"github.com/maruel/roundtrippers"

	"github.com/maruel/genai"
	"github.com/maruel/genai/base"
	"github.com/maruel/genai/internal"
	"github.com/maruel/genai/scoreboard"
)

//go:embed scoreboard_us.json
var scoreboardUSJSON []byte

//go:embed scoreboard_intl.json
var scoreboardIntlJSON []byte

// ScoreboardForBackend returns the scoreboard for a specific backend.
//
// CN falls back to Intl since the model catalog is shared.
func ScoreboardForBackend(b ProviderOptionBackend) scoreboard.Score {
	raw := scoreboardUSJSON
	name := "scoreboard_us.json"
	if b == BackendIntl || b == BackendCN {
		raw = scoreboardIntlJSON
		name = "scoreboard_intl.json"
	}
	var s scoreboard.Score
	d := json.NewDecoder(bytes.NewReader(raw))
	d.DisallowUnknownFields()
	if err := d.Decode(&s); err != nil {
		panic(fmt.Errorf("failed to unmarshal %s: %w", name, err))
	}
	return s
}

// Client implements genai.Provider for Alibaba Cloud DashScope.
type Client struct {
	base.NotImplemented
	impl    base.Provider[*ErrorResponse, *ChatRequest, *ChatResponse, ChatStreamChunkResponse]
	baseURL string
	backend ProviderOptionBackend
}

// ProviderOptionBackend selects a DashScope regional endpoint.
type ProviderOptionBackend string

const (
	// BackendIntl is the international (Singapore) endpoint (default).
	BackendIntl ProviderOptionBackend = "dashscope-intl"
	// BackendUS is the US (Virginia) endpoint.
	BackendUS ProviderOptionBackend = "dashscope-us"
	// BackendCN is the China (Beijing) endpoint.
	BackendCN ProviderOptionBackend = "dashscope"
)

// GenOption defines Alibaba DashScope specific generation options.
type GenOption struct {
	// Thinking controls the thinking mode. Qwen3.5 models default to thinking enabled.
	Thinking bool
	// ThinkingBudget limits the maximum number of reasoning tokens. 0 means no limit.
	ThinkingBudget int64
}

// Validate implements genai.Validatable.
func (o *GenOption) Validate() error {
	return nil
}

// Validate implements genai.ProviderOption.
func (p ProviderOptionBackend) Validate() error {
	switch p {
	case BackendIntl, BackendUS, BackendCN:
		return nil
	default:
		return fmt.Errorf("unknown backend %q, use BackendIntl, BackendUS, or BackendCN", string(p))
	}
}

// New creates a new client for the Alibaba Cloud DashScope API.
//
// If ProviderOptionAPIKey is not provided, it tries DASHSCOPE_API_KEY_INTL,
// DASHSCOPE_API_KEY_US, DASHSCOPE_API_KEY_CN (auto-selecting the backend),
// then DASHSCOPE_API_KEY.
//
// ProviderOptionBackend selects a named regional endpoint (e.g. BackendUS).
// When set, the matching DASHSCOPE_API_KEY_<region> is tried first.
// ProviderOptionRemote overrides all other endpoint selection with a full URL.
func New(ctx context.Context, opts ...genai.ProviderOption) (*Client, error) {
	var apiKey, model, remote string
	var backend ProviderOptionBackend
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
		case genai.ProviderOptionRemote:
			remote = string(v)
		case ProviderOptionBackend:
			backend = v
		default:
			return nil, fmt.Errorf("unsupported option type %T", opt)
		}
	}
	const apiKeyURL = "https://modelstudio.console.alibabacloud.com/"
	var err error
	if apiKey == "" {
		// When backend is set, try its matching region key first.
		switch backend {
		case BackendIntl:
			apiKey = os.Getenv("DASHSCOPE_API_KEY_INTL")
		case BackendUS:
			apiKey = os.Getenv("DASHSCOPE_API_KEY_US")
		case BackendCN:
			apiKey = os.Getenv("DASHSCOPE_API_KEY_CN")
		default:
			// Auto-detect: first region key found sets the backend.
			if v := os.Getenv("DASHSCOPE_API_KEY_INTL"); v != "" {
				apiKey = v
				backend = BackendIntl
			} else if v := os.Getenv("DASHSCOPE_API_KEY_US"); v != "" {
				apiKey = v
				backend = BackendUS
			} else if v := os.Getenv("DASHSCOPE_API_KEY_CN"); v != "" {
				apiKey = v
				backend = BackendCN
			}
		}
		if apiKey == "" {
			if apiKey = os.Getenv("DASHSCOPE_API_KEY"); apiKey == "" {
				err = &base.ErrAPIKeyRequired{EnvVar: "DASHSCOPE_API_KEY", URL: apiKeyURL}
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
	if remote == "" {
		switch backend {
		case BackendUS:
			remote = "https://dashscope-us.aliyuncs.com/compatible-mode/v1"
		case BackendCN:
			remote = "https://dashscope.aliyuncs.com/compatible-mode/v1"
		default:
			remote = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
		}
	}
	c := &Client{
		baseURL: remote,
		backend: backend,
		impl: base.Provider[*ErrorResponse, *ChatRequest, *ChatResponse, ChatStreamChunkResponse]{
			GenSyncURL:      remote + "/chat/completions",
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

// Name implements genai.Provider.
func (c *Client) Name() string {
	return "alibaba"
}

// selectBestTextModel selects the most appropriate text model based on the preference (cheap, good, or SOTA).
//
// It calls ListModels and analyzes model IDs by string to find the most recent version family,
// then picks the smallest, middle, or largest model within that family for cheap, good, or SOTA.
// Only canonical base text models matching `qwen{V}-{N}b-a{M}b` or `qwen{V}-max` are considered.
//
// We may want to make this function overridable in the future by the client since this is going to break one
// day or another.
func (c *Client) selectBestTextModel(ctx context.Context, preference string) (string, error) {
	mdls, err := c.ListModels(ctx)
	if err != nil {
		return "", fmt.Errorf("failed to automatically select the model: %w", err)
	}
	// modelRe matches only canonical base text models, capturing version and param count.
	// Examples: "qwen3-30b-a3b", "qwen3.5-122b-a10b", "qwen3-max".
	modelRe := regexp.MustCompile(`^qwen(\d+(?:\.\d+)?)-(?:(\d+)b-a\d+b|max)$`)
	type candidate struct {
		id      string
		version float64
		params  int  // math.MaxInt for "-max" models
		isMax   bool // true for "-max" models
	}
	var candidates []candidate
	for _, mdl := range mdls {
		m := modelRe.FindStringSubmatch(mdl.GetID())
		if m == nil {
			continue
		}
		ver, _ := strconv.ParseFloat(m[1], 64)
		var params int
		isMax := m[2] == ""
		if isMax {
			params = math.MaxInt
		} else {
			params, _ = strconv.Atoi(m[2])
		}
		candidates = append(candidates, candidate{mdl.GetID(), ver, params, isMax})
	}
	if len(candidates) == 0 {
		return "", errors.New("failed to find a model automatically")
	}
	// "-max" models are only appropriate for SOTA, not for cheap or good.
	allowMax := preference == string(genai.ModelSOTA)
	// Find the most recent version family with at least one allowed candidate.
	maxVer := 0.0
	for _, cand := range candidates {
		if (allowMax || !cand.isMax) && cand.version > maxVer {
			maxVer = cand.version
		}
	}
	if maxVer == 0 {
		return "", errors.New("failed to find a model automatically")
	}
	// Filter to only the most recent version family, excluding max models unless SOTA.
	latest := candidates[:0]
	for _, cand := range candidates {
		if cand.version == maxVer && (allowMax || !cand.isMax) {
			latest = append(latest, cand)
		}
	}
	// Sort by parameter count ascending.
	slices.SortFunc(latest, func(a, b candidate) int {
		if a.params < b.params {
			return -1
		}
		if a.params > b.params {
			return 1
		}
		return 0
	})
	switch preference {
	case string(genai.ModelCheap):
		return latest[0].id, nil
	case string(genai.ModelSOTA):
		return latest[len(latest)-1].id, nil
	default: // ModelGood
		return latest[len(latest)/2].id, nil
	}
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
	return ScoreboardForBackend(c.backend)
}

// ScoreboardVariants implements genai.ProviderScoreboardVariants.
func (c *Client) ScoreboardVariants() []genai.ScoreboardVariant {
	return []genai.ScoreboardVariant{
		{Name: "Intl", Score: ScoreboardForBackend(BackendIntl)},
		{Name: "US", Score: ScoreboardForBackend(BackendUS)},
	}
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
	if err := c.impl.DoRequest(ctx, "GET", c.baseURL+"/models", nil, &resp); err != nil {
		return nil, err
	}
	return resp.ToModels(), nil
}

// ProcessStream converts raw stream chunks to genai.Reply fragments.
func ProcessStream(chunks iter.Seq[ChatStreamChunkResponse]) (iter.Seq[genai.Reply], func() (genai.Usage, [][]genai.Logprob, error)) {
	var finalErr error
	u := genai.Usage{}

	return func(yield func(genai.Reply) bool) {
			// pending tracks in-flight tool calls by their stream Index.
			pending := map[int64]*ToolCall{}
			// pendingOrder preserves insertion order so we flush deterministically.
			pendingOrder := []int64{}
			for pkt := range chunks {
				if len(pkt.Choices) != 1 {
					continue
				}
				if pkt.Usage.CompletionTokens != 0 {
					u.InputTokens = pkt.Usage.PromptTokens
					u.InputCachedTokens = pkt.Usage.PromptTokensDetails.CachedTokens
					u.OutputTokens = pkt.Usage.CompletionTokens
					u.FinishReason = pkt.Choices[0].FinishReason.ToFinishReason()
				}
				switch role := pkt.Choices[0].Delta.Role; role {
				case "assistant", "":
				default:
					finalErr = &internal.BadError{Err: fmt.Errorf("unexpected role %q", role)}
					return
				}
				f := genai.Reply{}
				if s := pkt.Choices[0].Delta.ReasoningContent; s != "" {
					f.Reasoning = s
				}
				for _, c := range pkt.Choices[0].Delta.Content {
					switch c.Type {
					case ContentText:
						f.Text += c.Text
					default:
						finalErr = &internal.BadError{Err: fmt.Errorf("unsupported stream content type %q", c.Type)}
						return
					}
				}
				// Buffer tool call arguments per Index and flush completed calls.
				for _, tc := range pkt.Choices[0].Delta.ToolCalls {
					if tc.ID != "" {
						// New tool call starting.
						pending[tc.Index] = &tc
						pendingOrder = append(pendingOrder, tc.Index)
					} else if p := pending[tc.Index]; p != nil {
						// Continuation of an existing tool call.
						p.Function.Arguments += tc.Function.Arguments
					}
				}
				if len(pkt.Choices[0].Delta.ToolCalls) == 0 && len(pending) > 0 {
					// No more tool call deltas; flush all pending in order.
					for _, idx := range pendingOrder {
						p := pending[idx]
						r := genai.Reply{}
						p.To(&r.ToolCall)
						if !yield(r) {
							return
						}
					}
					pending = map[int64]*ToolCall{}
					pendingOrder = nil
				}
				if !f.IsZero() {
					if !yield(f) {
						return
					}
				}
			}
			// Flush any remaining pending tool calls at end of stream.
			for _, idx := range pendingOrder {
				p := pending[idx]
				r := genai.Reply{}
				p.To(&r.ToolCall)
				if !yield(r) {
					return
				}
			}
		}, func() (genai.Usage, [][]genai.Logprob, error) {
			return u, nil, finalErr
		}
}

var _ genai.Provider = &Client{}
