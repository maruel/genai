// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Package baseten implements a client for the Baseten inference API.
//
// It is described at https://docs.baseten.co/reference/inference-api/chat-completions
//
// Baseten offers an OpenAI-compatible chat completions endpoint hosting a limited
// set of high-performance models (DeepSeek, GLM, Kimi, MiniMax, gpt-oss).
package baseten

import (
	"bytes"
	"cmp"
	"context"
	_ "embed"
	"encoding/json"
	"errors"
	"fmt"
	"iter"
	"net/http"
	"os"
	"slices"
	"strconv"
	"strings"
	"sync"

	"github.com/maruel/roundtrippers"

	"github.com/maruel/genai"
	"github.com/maruel/genai/base"
	"github.com/maruel/genai/internal"
	"github.com/maruel/genai/scoreboard"
)

//go:embed scoreboard.json
var scoreboardJSON []byte

//go:embed models.json
var modelsJSON []byte

// Scoreboard for Baseten.
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

// New creates a new client to talk to the Baseten inference API.
//
// If apiKey is not provided via ProviderOptionAPIKey, it tries to load it from the BASETEN_API_KEY environment
// variable. If none is found, it will still return a client coupled with a base.ErrAPIKeyRequired error.
// Get an API key at https://app.baseten.co/settings/account/api_keys
//
// To use multiple models, create multiple clients.
// Use one of the models from https://docs.baseten.co/development/model-apis/overview
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
	const apiKeyURL = "https://app.baseten.co/settings/account/api_keys"
	var err error
	if apiKey == "" {
		if apiKey = os.Getenv("BASETEN_API_KEY"); apiKey == "" {
			err = &base.ErrAPIKeyRequired{EnvVar: "BASETEN_API_KEY", URL: apiKeyURL}
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
			GenSyncURL:      "https://inference.baseten.co/v1/chat/completions",
			ProcessStream:   ProcessStream,
			ProcessHeaders:  processHeaders,
			PreloadedModels: preloadedModels,
			LieToolCalls:    true,
			ProviderBase: base.ProviderBase[*ErrorResponse]{
				APIKeyURL: apiKeyURL,
				Lenient:   internal.BeLenient,
				Client: http.Client{
					// Baseten uses "Api-Key" prefix instead of "Bearer".
					Transport: &roundtrippers.Header{
						Header:    http.Header{"Authorization": {"Api-Key " + apiKey}},
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

// selectBestTextModel selects the most appropriate model based on the preference (cheap, good, or SOTA).
//
// It reads the scoreboard to find the reference model for the preference, extracts its family prefix,
// then picks the model with the highest version number from the available models.
func (c *Client) selectBestTextModel(ctx context.Context, preference string) (string, error) {
	mdls, err := c.ListModels(ctx)
	if err != nil {
		return "", fmt.Errorf("failed to automatically select the model: %w", err)
	}
	// Find the reference model from the scoreboard.
	s := Scoreboard()
	var ref string
	for _, sc := range s.Scenarios {
		if len(sc.Models) == 0 {
			continue
		}
		switch preference {
		case string(genai.ModelCheap):
			if sc.Cheap {
				ref = sc.Models[0]
			}
		case string(genai.ModelGood):
			if sc.Good {
				ref = sc.Models[0]
			}
		default:
			if sc.SOTA {
				ref = sc.Models[0]
			}
		}
		if ref != "" {
			break
		}
	}
	if ref == "" {
		return "", errors.New("no reference model found in scoreboard")
	}
	// Extract the model family prefix and find the highest version.
	prefix := modelFamilyPrefix(ref)
	selected := ""
	highestVer := -1.0
	for _, mdl := range mdls {
		id := mdl.GetID()
		if !strings.HasPrefix(id, prefix) {
			continue
		}
		if v := parseLeadingFloat(id[len(prefix):]); v > highestVer {
			highestVer = v
			selected = id
		}
	}
	if selected == "" {
		return "", errors.New("failed to find a model automatically")
	}
	return selected, nil
}

// modelFamilyPrefix extracts the model family prefix by stripping the trailing version number.
//
// E.g., "zai-org/GLM-5" -> "zai-org/GLM-", "MiniMaxAI/MiniMax-M2.5" -> "MiniMaxAI/MiniMax-M".
func modelFamilyPrefix(id string) string {
	i := len(id)
	// Skip trailing non-digit suffix (e.g., 'b' in "120b").
	for i > 0 && id[i-1] != '.' && (id[i-1] < '0' || id[i-1] > '9') {
		i--
	}
	// Skip trailing digits and dots (the version number).
	for i > 0 && (id[i-1] >= '0' && id[i-1] <= '9' || id[i-1] == '.') {
		i--
	}
	if i == 0 {
		return id
	}
	return id[:i]
}

// parseLeadingFloat parses a leading float from the string (e.g., "5" -> 5, "4.7" -> 4.7, "120b" -> 120).
func parseLeadingFloat(s string) float64 {
	end := 0
	for end < len(s) && (s[end] >= '0' && s[end] <= '9' || s[end] == '.') {
		end++
	}
	v, _ := strconv.ParseFloat(s[:end], 64)
	return v
}

// Name implements genai.Provider.
func (c *Client) Name() string {
	return "baseten"
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
		cacheModelFeatures(c.impl.PreloadedModels)
		return c.impl.PreloadedModels, nil
	}
	// https://docs.baseten.co/reference/inference-api/models
	var resp ModelsResponse
	if err := c.impl.DoRequest(ctx, "GET", "https://inference.baseten.co/v1/models", nil, &resp); err != nil {
		return nil, err
	}
	mdls := resp.ToModels()
	cacheModelFeatures(mdls)
	return mdls, nil
}

// ProcessStream converts the raw packets from the streaming API into Reply fragments.
func ProcessStream(chunks iter.Seq[ChatStreamChunkResponse]) (iter.Seq[genai.Reply], func() (genai.Usage, [][]genai.Logprob, error)) {
	var finalErr error
	u := genai.Usage{}
	var l [][]genai.Logprob

	return func(yield func(genai.Reply) bool) {
			pendingToolCall := ToolCall{}
			for pkt := range chunks {
				if len(pkt.Choices) != 1 {
					continue
				}
				switch role := pkt.Choices[0].Delta.Role; role {
				case "", "assistant":
				default:
					finalErr = &internal.BadError{Err: fmt.Errorf("unexpected role %q", role)}
					return
				}
				if pkt.Usage.TotalTokens != 0 {
					u.InputTokens = pkt.Usage.PromptTokens
					u.InputCachedTokens = pkt.Usage.PromptTokensDetails.CachedTokens
					u.OutputTokens = pkt.Usage.CompletionTokens
					u.ReasoningTokens = pkt.Usage.CompletionTokensDetails.ReasoningTokens
					u.TotalTokens = pkt.Usage.TotalTokens
				}
				if fr := pkt.Choices[0].FinishReason; fr != "" {
					u.FinishReason = fr.ToFinishReason()
				}

				for _, nt := range pkt.Choices[0].Delta.ToolCalls {
					if pendingToolCall.ID != "" {
						// Detect continuation: empty ID (OpenAI), same ID (Kimi),
						// or empty name with different ID (MiniMax).
						isContinuation := nt.ID == "" || nt.ID == pendingToolCall.ID || nt.Function.Name == ""
						if isContinuation {
							pendingToolCall.Function.Arguments += nt.Function.Arguments
							if nt.Function.Name != "" {
								pendingToolCall.Function.Name = nt.Function.Name
							}
						} else {
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
				if r := cmp.Or(pkt.Choices[0].Delta.Reasoning, pkt.Choices[0].Delta.ReasoningContent); r != "" {
					if !yield(genai.Reply{Reasoning: r}) {
						return
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
				if len(pkt.Choices[0].Logprobs.Content) != 0 {
					l = append(l, pkt.Choices[0].Logprobs.To()...)
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
			return u, l, finalErr
		}
}

func processHeaders(h http.Header) []genai.RateLimit {
	return nil
}

type modelsData struct {
	Models map[string]modelData `json:"models"`
}

type modelData struct {
	EnableThinking bool `json:"enable_thinking"`
}

var modelsEnableThinking sync.Map

func init() {
	data := modelsData{}
	if err := json.Unmarshal(modelsJSON, &data); err != nil {
		panic(fmt.Sprintf("failed to parse embedded models.json: %v", err))
	}
	for k, v := range data.Models {
		modelsEnableThinking.Store(k, v.EnableThinking)
	}
}

func cacheModelFeatures(mdls []genai.Model) {
	for _, m := range mdls {
		bm, ok := m.(*Model)
		if !ok {
			continue
		}
		if _, ok = modelsEnableThinking.Load(bm.ID); ok {
			continue
		}
		if slices.Contains(bm.SupportedFeatures, "reasoning") {
			modelsEnableThinking.Store(bm.ID, true)
		}
	}
}

func enableThinkingByDefault(model string) bool {
	v, ok := modelsEnableThinking.Load(model)
	return ok && v.(bool)
}

var _ genai.Provider = &Client{}
