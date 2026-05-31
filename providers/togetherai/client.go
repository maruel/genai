// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Package togetherai implements a client for the Together.ai API.
//
// It is described at https://docs.together.ai/docs/
package togetherai

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
	"strconv"
	"strings"
	"time"

	"github.com/maruel/genai"
	"github.com/maruel/genai/base"
	"github.com/maruel/genai/internal"
	"github.com/maruel/genai/internal/bb"
	"github.com/maruel/genai/scoreboard"
	"github.com/maruel/roundtrippers"
)

// Official python client library at https://github.com/togethercomputer/together-python/tree/main/src/together

//go:embed scoreboard.json
var scoreboardJSON []byte

// Scoreboard for TogetherAI.
//
// See https://docs.together.ai/docs/serverless-models and https://api.together.ai/models
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

// New creates a new client to talk to the Together.AI platform API.
//
// If ProviderOptionAPIKey is not provided, it tries to load it from the TOGETHER_API_KEY environment variable.
// If none is found, it will still return a client coupled with an base.ErrAPIKeyRequired error.
// Get your API key at https://api.together.ai/settings/api-keys
//
// To use multiple models, create multiple clients.
// Use one of the model from https://docs.together.ai/docs/serverless-models
//
// # Vision
//
// We must select a model that supports video.
// https://docs.together.ai/docs/serverless-models#vision-models
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
	const apiKeyURL = "https://api.together.ai/settings/api-keys"
	var err error
	if apiKey == "" {
		if apiKey = os.Getenv("TOGETHER_API_KEY"); apiKey == "" {
			err = &base.ErrAPIKeyRequired{EnvVar: "TOGETHER_API_KEY", URL: apiKeyURL}
		}
	}
	switch len(modalities) {
	case 0:
	case 1:
		switch modalities[0] {
		case genai.ModalityImage, genai.ModalityText:
		case genai.ModalityAudio:
			// TODO: Add support for audio.
			return nil, fmt.Errorf("unexpected option Modalities %s, only image or text are implemented (send PR to add support)", modalities)
		case genai.ModalityDocument, genai.ModalityVideo:
			return nil, fmt.Errorf("unexpected option Modalities %s, only image or text are implemented", modalities)
		default:
			return nil, fmt.Errorf("unexpected option Modalities %s, only image or text are implemented", modalities)
		}
	default:
		return nil, fmt.Errorf("unexpected option Modalities %s, only image or text are implemented (send PR to add support)", modalities)
	}
	t := base.DefaultTransport
	if wrapper != nil {
		t = wrapper(t)
	}
	c := &Client{
		impl: base.Provider[*ErrorResponse, *ChatRequest, *ChatResponse, ChatStreamChunkResponse]{
			GenSyncURL:      "https://api.together.xyz/v1/chat/completions",
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
			if len(modalities) == 0 || modalities[0] == genai.ModalityText {
				if c.impl.Model, err = c.selectBestTextModel(ctx, model); err != nil {
					return nil, err
				}
				c.impl.OutputModalities = genai.Modalities{genai.ModalityText}
			} else {
				if c.impl.Model, err = c.selectBestImageModel(ctx, model); err != nil {
					return nil, err
				}
				c.impl.OutputModalities = genai.Modalities{genai.ModalityImage}
			}
		default:
			c.impl.Model = model
			if len(modalities) == 0 {
				c.impl.OutputModalities, err = c.detectModelModalities(ctx, model)
			} else {
				c.impl.OutputModalities = modalities
			}
		}
	}
	return c, err
}

// detectModelModalities tries its best to figure out the modality of a model
//
// We may want to make this function overridable in the future by the client since this is going to break one
// day or another.
func (c *Client) detectModelModalities(ctx context.Context, model string) (genai.Modalities, error) {
	// Detect if it is an image model.
	mdls, err2 := c.ListModels(ctx)
	if err2 != nil {
		return nil, fmt.Errorf("failed to detect the output modality for the model %s: %w", model, err2)
	}
	for _, mdl := range mdls {
		if m := mdl.(*Model); m.ID == model {
			switch m.Type {
			case "chat", "":
				// Some models don't have a type set; default to text.
				return genai.Modalities{genai.ModalityText}, nil
			case "image":
				return genai.Modalities{genai.ModalityImage}, nil
			default:
				return nil, fmt.Errorf("failed to detect the output modality for the model %s: found type %s", model, m.Type)
			}
		}
	}
	return nil, fmt.Errorf("failed to automatically detect the model modality: model %s not found", model)
}

// selectBestTextModel selects the most appropriate model based on the preference (cheap, good, or SOTA).
//
// We may want to make this function overridable in the future by the client since this is going to break one
// day or another.
func (c *Client) selectBestTextModel(ctx context.Context, preference string) (string, error) {
	mdls, err := c.ListModels(ctx)
	if err != nil {
		return "", fmt.Errorf("failed to automatically select the model: %w", err)
	}
	cheap := preference == string(genai.ModelCheap)
	good := preference == string(genai.ModelGood) || preference == ""
	selectedModel := ""
	bestVer := 0.
	price := math.Inf(1)
	cutoff := time.Now().Add(-365 * 25 * time.Hour)
	for _, mdl := range mdls {
		m := mdl.(*Model)
		if m.Type != "chat" || strings.Contains(m.ID, "-VL-") || strings.Contains(m.ID, "-Vision-") || (m.Created > 1000 && m.Created.AsTime().Before(cutoff)) {
			continue
		}
		switch {
		case cheap:
			if strings.HasPrefix(m.ID, "openai/gpt") && (m.Pricing.Output == 0 || (price > m.Pricing.Output)) {
				price = m.Pricing.Output
				selectedModel = m.ID
			}
		case good:
			if v := deepseekVersion(m.ID); v > bestVer {
				bestVer = v
				selectedModel = m.ID
			}
		default:
			if v := glmVersion(m.ID); v > bestVer {
				bestVer = v
				selectedModel = m.ID
			}
		}
	}
	if selectedModel == "" {
		return "", errors.New("failed to find a model automatically")
	}
	return selectedModel, nil
}

// selectBestImageModel selects the most appropriate model based on the preference (cheap, good, or SOTA).
//
// We may want to make this function overridable in the future by the client since this is going to break one
// day or another.
func (c *Client) selectBestImageModel(ctx context.Context, preference string) (string, error) {
	mdls, err := c.ListModels(ctx)
	if err != nil {
		return "", fmt.Errorf("failed to automatically select the model: %w", err)
	}
	cheap := preference == string(genai.ModelCheap)
	good := preference == string(genai.ModelGood) || preference == ""
	selectedModel := ""
	// As of May 2026, pricing is still not set for image models, so we rely on naming conventions.
	// FLUX.2 models are preferred for SOTA over arbitrary -pro models.
	flux2BestID, flux2BestRank := "", 0
	for _, mdl := range mdls {
		m := mdl.(*Model)
		if m.Type != "image" {
			continue
		}
		switch {
		case cheap:
			if strings.HasSuffix(m.ID, "-schnell") && (selectedModel == "" || m.ID > selectedModel) {
				selectedModel = m.ID
			}
		case good:
			if strings.HasSuffix(m.ID, "-dev") && !strings.Contains(m.ID, "kontext") && (selectedModel == "" || m.ID > selectedModel) {
				selectedModel = m.ID
			}
		default:
			// Track the best FLUX.2 model by rank: max=3, pro=2, flex/dev=1.
			if r := flux2Rank(m.ID); r > flux2BestRank {
				flux2BestRank = r
				flux2BestID = m.ID
			}
		}
	}
	if selectedModel == "" {
		// SOTA: prefer FLUX.2 over any other -pro model.
		if flux2BestID != "" {
			return flux2BestID, nil
		}
		// Fallback to best -pro model (excluding kontext).
		for _, mdl := range mdls {
			m := mdl.(*Model)
			if m.Type != "image" {
				continue
			}
			if strings.HasSuffix(m.ID, "-pro") && !strings.Contains(m.ID, "kontext") && (selectedModel == "" || m.ID > selectedModel) {
				selectedModel = m.ID
			}
		}
	}
	if selectedModel == "" {
		return "", errors.New("failed to find a model automatically")
	}
	return selectedModel, nil
}

// flux2Rank returns a quality rank for FLUX.2 image models, or 0 for non-FLUX.2 models.
//
// max=3, pro=2, flex/dev=1.
func flux2Rank(id string) int {
	if !strings.HasPrefix(id, "black-forest-labs/FLUX.2") {
		return 0
	}
	if strings.Contains(id, "-max") {
		return 3
	}
	if strings.Contains(id, "-pro") {
		return 2
	}
	return 1
}

// parseVersion extracts a leading version number (digits and dots) from s.
// Returns the version and the remaining string. Returns 0 if s doesn't start with a digit.
func parseVersion(s string) (float64, string) {
	i := 0
	for i < len(s) && (s[i] >= '0' && s[i] <= '9' || s[i] == '.') {
		i++
	}
	if i == 0 {
		return 0, s
	}
	v, err := strconv.ParseFloat(s[:i], 64)
	if err != nil {
		return 0, s
	}
	return v, s[i:]
}

// glmVersion returns the version of a base GLM model (e.g. GLM-5.1 → 5.1) or 0 for
// non-base variants (e.g. GLM-OCR, GLM-4.5V, GLM-5-FP4).
func glmVersion(modelID string) float64 {
	s, ok := strings.CutPrefix(modelID, "zai-org/GLM-")
	if !ok {
		return 0
	}
	v, rem := parseVersion(s)
	if rem != "" {
		return 0
	}
	return v
}

// deepseekVersion returns the version of a DeepSeek chat model (e.g.
// DeepSeek-V4-Pro → 4.0) or 0 for non-DeepSeek models or non-chat variants
// (e.g. DeepSeek-R1, DeepSeek-OCR, DeepSeek-Coder).
func deepseekVersion(modelID string) float64 {
	s, ok := strings.CutPrefix(modelID, "deepseek-ai/DeepSeek-V")
	if !ok {
		return 0
	}
	v, rem := parseVersion(s)
	// Accept plain version numbers (V4, V4.1) and reject suffixed variants (R1, Coder, OCR).
	if rem != "" && rem != "-Pro" {
		return 0
	}
	return v
}

// qwenVersion returns the version of a base Qwen model (e.g. Qwen3.5-397B-A17B → 3.5)
// or 0 for variants (e.g. Qwen3-Coder, Qwen3-VL, Qwen3-Next) and quantized
// models (e.g. Qwen3.6-35B-A3B-FP8).
func qwenVersion(modelID string) float64 {
	s, ok := strings.CutPrefix(modelID, "Qwen/Qwen")
	if !ok {
		return 0
	}
	v, rem := parseVersion(s)
	// After the version, expect "-<digit>" (size like 397B), not "-<letter>" (variant like Coder).
	if len(rem) < 2 || rem[0] != '-' || rem[1] < '0' || rem[1] > '9' {
		return 0
	}
	// Reject quantized variants (-FP8, -FP4, -INT4) and fine-tuned variants (-Lora).
	for _, suffix := range []string{"-FP8", "-FP4", "-INT4", "-Lora"} {
		if strings.Contains(rem, suffix) {
			return 0
		}
	}
	return v
}

// Name implements genai.Provider.
//
// It returns the name of the provider.
func (c *Client) Name() string {
	return "togetherai"
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
	if c.impl.OutputModalities[0] == genai.ModalityText {
		return c.impl.GenSync(ctx, msgs, opts...)
	}
	if len(msgs) != 1 {
		return genai.Result{}, errors.New("must pass exactly one Message")
	}
	return c.genImage(ctx, &msgs[0], opts...)
}

// GenSyncRaw provides access to the raw API.
func (c *Client) GenSyncRaw(ctx context.Context, in *ChatRequest, out *ChatResponse) error {
	return c.impl.GenSyncRaw(ctx, in, out)
}

// GenStream implements genai.Provider.
func (c *Client) GenStream(ctx context.Context, msgs genai.Messages, opts ...genai.GenOption) (iter.Seq[genai.Reply], func() (genai.Result, error)) {
	if c.impl.OutputModalities[0] == genai.ModalityText {
		return c.impl.GenStream(ctx, msgs, opts...)
	}
	return base.SimulateStream(ctx, c, msgs, opts...)
}

// GenStreamRaw provides access to the raw API.
func (c *Client) GenStreamRaw(ctx context.Context, in *ChatRequest) (iter.Seq[ChatStreamChunkResponse], func() error) {
	return c.impl.GenStreamRaw(ctx, in)
}

// genImage generates non-text content (images or audio).
func (c *Client) genImage(ctx context.Context, msg *genai.Message, opts ...genai.GenOption) (genai.Result, error) {
	if c.impl.OutputModalities[0] == genai.ModalityAudio {
		// https://docs.together.ai/reference/audio-speech
		return genai.Result{}, errors.New("audio not implemented yet")
	}
	// https://docs.together.ai/reference/post-images-generations
	res := genai.Result{}
	if err := c.impl.Validate(); err != nil {
		return genai.Result{}, err
	}
	req := ImageRequest{}
	if err := req.Init(msg, c.impl.Model, opts...); err != nil {
		return res, err
	}
	resp := ImageResponse{}
	if err := c.impl.DoRequest(ctx, "POST", "https://api.together.xyz/v1/images/generations", &req, &resp); err != nil {
		return res, err
	}
	res.Replies = make([]genai.Reply, len(resp.Data))
	for i := range resp.Data {
		n := "content.jpg"
		if len(resp.Data) > 1 {
			n = fmt.Sprintf("content%d.jpg", i+1)
		}
		if url := resp.Data[i].URL; url != "" {
			res.Replies[i].Doc = genai.Doc{Filename: n, URL: url}
		} else if d := resp.Data[i].B64JSON; len(d) != 0 {
			res.Replies[i].Doc = genai.Doc{Filename: n, Src: &bb.BytesBuffer{D: resp.Data[i].B64JSON}}
		} else {
			return res, errors.New("internal error")
		}
	}
	if err := res.Validate(); err != nil {
		return res, err
	}
	return res, nil
}

// ListModels implements genai.Provider.
func (c *Client) ListModels(ctx context.Context) ([]genai.Model, error) {
	if c.impl.PreloadedModels != nil {
		return c.impl.PreloadedModels, nil
	}
	// https://docs.together.ai/reference/models-1
	var resp ModelsResponse
	if err := c.impl.DoRequest(ctx, "GET", "https://api.together.xyz/v1/models", nil, &resp); err != nil {
		return nil, err
	}
	return resp.ToModels(), nil
}

// ProcessStream converts the raw packets from the streaming API into Reply fragments.
func ProcessStream(chunks iter.Seq[ChatStreamChunkResponse]) (iter.Seq[genai.Reply], func() (genai.Usage, [][]genai.Logprob, error)) {
	var finalErr error
	var warnings []string
	u := genai.Usage{}
	var l [][]genai.Logprob

	return func(yield func(genai.Reply) bool) {
			pendingToolCall := ToolCall{}
			// flushToolCall emits the buffered tool call, if any. It returns false only when the consumer
			// asked to stop. Incomplete parallel tool calls (a model truncating the trailing call
			// mid-arguments) are dropped rather than emitted as invalid.
			flushToolCall := func() bool {
				if pendingToolCall.ID == "" {
					return true
				}
				args := pendingToolCall.Function.Arguments
				if args == "" {
					args = "{}"
				}
				tc := pendingToolCall
				pendingToolCall = ToolCall{}
				if !json.Valid([]byte(args)) {
					return true
				}
				return yield(genai.Reply{ToolCall: genai.ToolCall{
					ID:        tc.ID,
					Name:      tc.Function.Name,
					Arguments: args,
				}})
			}
			for pkt := range chunks {
				if pkt.Usage.TotalTokens != 0 {
					u.InputTokens = pkt.Usage.PromptTokens
					u.InputCachedTokens = max(pkt.Usage.CachedTokens, pkt.Usage.PromptTokensDetails.CachedTokens)
					u.ReasoningTokens = pkt.Usage.ReasoningTokens
					u.OutputTokens = pkt.Usage.CompletionTokens
					u.TotalTokens = pkt.Usage.TotalTokens
				}
				for _, w := range pkt.Warnings {
					warnings = append(warnings, w.Message)
				}
				if len(pkt.Choices) != 1 {
					continue
				}
				// Check for streaming errors.
				if pkt.Choices[0].Error != nil {
					finalErr = fmt.Errorf("streaming error: %v", pkt.Choices[0].Error)
					return
				}
				if pkt.Choices[0].FinishReason != "" {
					u.FinishReason = pkt.Choices[0].FinishReason.ToFinishReason()
				}
				switch role := pkt.Choices[0].Delta.Role; role {
				case "assistant", "":
				default:
					finalErr = &internal.BadError{Err: fmt.Errorf("unexpected role %q", role)}
					return
				}
				if len(pkt.Choices[0].ToolCalls) > 1 {
					finalErr = &internal.BadError{Err: fmt.Errorf("implement multiple tool calls: %#v", pkt.Choices[0].ToolCalls)}
					return
				}
				// TogetherAI streams the arguments of one tool call at a time, switching to the next
				// parallel call by sending a new ID. Buffer the arguments to send each call as a whole.
				if len(pkt.Choices[0].Delta.ToolCalls) > 1 {
					finalErr = &internal.BadError{Err: fmt.Errorf("implement multiple delta tool calls: %#v", pkt.Choices[0].Delta.ToolCalls)}
					return
				}
				if len(pkt.Choices[0].Delta.ToolCalls) == 1 {
					t := pkt.Choices[0].Delta.ToolCalls[0]
					if t.ID != "" {
						// A new (possibly parallel) call: flush the previous one first.
						if !flushToolCall() {
							return
						}
						pendingToolCall = t
						continue
					}
					if pendingToolCall.ID != "" {
						// Continuation of the buffered call's arguments.
						pendingToolCall.Function.Arguments += t.Function.Arguments
						continue
					}
				} else if !flushToolCall() {
					// No more tool call deltas: flush the buffered call.
					return
				}
				if len(pkt.Choices[0].ToolCalls) == 1 {
					// That's a non-streamed tool call.
					f := genai.Reply{}
					pkt.Choices[0].ToolCalls[0].To(&f.ToolCall)
					if !yield(f) {
						return
					}
				}
				f := genai.Reply{
					Text:      pkt.Choices[0].Delta.Content,
					Reasoning: pkt.Choices[0].Delta.Reasoning,
				}
				if !yield(f) {
					return
				}
				if len(pkt.Choices[0].Logprobs.Tokens) != 0 {
					l = append(l, pkt.Choices[0].Logprobs.To()...)
				}
			}
		}, func() (genai.Usage, [][]genai.Logprob, error) {
			if len(warnings) != 0 {
				uce := &base.ErrNotSupported{}
				for _, w := range warnings {
					if strings.Contains(w, "tool_choice") {
						uce.Options = append(uce.Options, "GenOptionTools.Force")
					} else {
						uce.Options = append(uce.Options, w)
					}
				}
				return u, l, uce
			}
			return u, l, finalErr
		}
}

func processHeaders(h http.Header) []genai.RateLimit {
	var limits []genai.RateLimit
	limitReq, _ := strconv.ParseInt(h.Get("X-Ratelimit-Limit"), 10, 64)
	remainingReq, _ := strconv.ParseInt(h.Get("X-Ratelimit-Remaining"), 10, 64)

	limitTok, _ := strconv.ParseInt(h.Get("X-Ratelimit-Limit-Tokens"), 10, 64)
	remainingTok, _ := strconv.ParseInt(h.Get("X-Ratelimit-Remaining-Tokens"), 10, 64)

	reset, _ := time.ParseDuration(h.Get("X-Ratelimit-Reset"))

	if limitReq > 0 {
		limits = append(limits, genai.RateLimit{
			Type:      genai.Requests,
			Period:    genai.PerOther,
			Limit:     limitReq,
			Remaining: remainingReq,
			Reset:     time.Now().Add(reset).Round(10 * time.Millisecond),
		})
	}
	if limitTok > 0 {
		limits = append(limits, genai.RateLimit{
			Type:      genai.Tokens,
			Period:    genai.PerOther,
			Limit:     limitTok,
			Remaining: remainingTok,
			Reset:     time.Now().Add(reset).Round(10 * time.Millisecond),
		})
	}
	return limits
}

var _ genai.Provider = &Client{}
