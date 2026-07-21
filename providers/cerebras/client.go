// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Package cerebras implements a client for the Cerebras API.
//
// It is described at https://inference-docs.cerebras.ai/api-reference/
package cerebras

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
	"slices"
	"strconv"
	"time"

	"github.com/maruel/roundtrippers"

	"github.com/maruel/genai"
	"github.com/maruel/genai/base"
	"github.com/maruel/genai/internal"
	"github.com/maruel/genai/scoreboard"
)

//go:embed scoreboard.json
var scoreboardJSON []byte

// Scoreboard for Cerebras.
func Scoreboard() scoreboard.Score {
	var s scoreboard.Score
	d := json.NewDecoder(bytes.NewReader(scoreboardJSON))
	d.DisallowUnknownFields()
	if err := d.Decode(&s); err != nil {
		panic(fmt.Errorf("failed to unmarshal scoreboard.json: %w", err))
	}
	return s
}

// Official python client: https://github.com/Cerebras/cerebras-cloud-sdk-python
//
// CompletionsResource.create() at
// https://github.com/Cerebras/cerebras-cloud-sdk-python/blob/main/src/cerebras/cloud/sdk/resources/chat/completions.py

// ProviderOptionQueueThreshold sets the default queue time at which requests using auto or flex service tiers are rejected.
// Cerebras accepts values from 50ms through 20s.
type ProviderOptionQueueThreshold time.Duration

// Validate implements genai.ProviderOption.
func (q ProviderOptionQueueThreshold) Validate() error {
	return validateQueueThreshold(time.Duration(q))
}

func validateQueueThreshold(q time.Duration) error {
	if q != 0 && (q < 50*time.Millisecond || q > 20*time.Second) {
		return fmt.Errorf("QueueThreshold must be between 50ms and 20s, got %s", q)
	}
	return nil
}

// GenOption controls Cerebras-specific generation settings.
type GenOption struct {
	// Prediction provides known output content for the model to reuse.
	Prediction Prediction
	// PromptCacheKey groups requests that share a prompt prefix.
	PromptCacheKey string
	// ReasoningEffort controls how much reasoning the model performs.
	ReasoningEffort ReasoningEffort
	// ReasoningFormat controls how reasoning content appears in a response.
	ReasoningFormat ReasoningFormat
	// ServiceTier controls request priority.
	ServiceTier ServiceTier
	// ToolChoice controls how the model may call tools.
	ToolChoice ToolChoice
	// QueueThreshold controls the queue time at which flex or auto requests are rejected.
	// Cerebras accepts values from 50ms through 20s.
	QueueThreshold time.Duration
}

// Validate implements genai.Validatable.
func (o *GenOption) Validate() error {
	if len(o.PromptCacheKey) > 1024 {
		return errors.New("prompt cache key exceeds the maximum length of 1024 bytes")
	}
	if err := o.Prediction.Validate(); err != nil {
		return err
	}
	if err := o.ReasoningEffort.Validate(); err != nil {
		return err
	}
	if err := o.ReasoningFormat.Validate(); err != nil {
		return err
	}
	if err := o.ServiceTier.Validate(); err != nil {
		return err
	}
	if err := o.ToolChoice.Validate(); err != nil {
		return err
	}
	return validateQueueThreshold(o.QueueThreshold)
}

type queueThresholdContextKey struct{}

func ctxWithQueueThreshold(ctx context.Context, opts []genai.GenOption) context.Context {
	var q time.Duration
	for _, opt := range opts {
		if v, ok := opt.(*GenOption); ok && v.QueueThreshold != 0 {
			q = v.QueueThreshold
		}
	}
	if q == 0 {
		return ctx
	}
	return context.WithValue(ctx, queueThresholdContextKey{}, q)
}

type queueThresholdHeader struct {
	transport http.RoundTripper
}

func (q *queueThresholdHeader) RoundTrip(req *http.Request) (*http.Response, error) {
	threshold, ok := req.Context().Value(queueThresholdContextKey{}).(time.Duration)
	if !ok {
		return q.transport.RoundTrip(req)
	}
	req = req.Clone(req.Context())
	req.Header.Set("queue_threshold", strconv.FormatInt(threshold.Milliseconds(), 10))
	return q.transport.RoundTrip(req)
}

func (q *queueThresholdHeader) Unwrap() http.RoundTripper {
	return q.transport
}

// Client implements genai.Provider.
type Client struct {
	base.NotImplemented
	impl base.Provider[*ErrorResponse, *ChatRequest, *ChatResponse, ChatStreamChunkResponse]
}

// New creates a new client to talk to the Cerebras platform API.
//
// If apiKey is not provided via ProviderOptionAPIKey, it tries to load it from the CEREBRAS_API_KEY environment
// variable. If none is found, it will still return a client coupled with an base.ErrAPIKeyRequired error.
// Get an API key at http://cloud.cerebras.ai/
//
// To use multiple models, create multiple clients.
// Use one of the model from https://cerebras.ai/inference
func New(ctx context.Context, opts ...genai.ProviderOption) (*Client, error) {
	var apiKey, model string
	var queueThreshold time.Duration
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
		case ProviderOptionQueueThreshold:
			queueThreshold = time.Duration(v)
		default:
			return nil, fmt.Errorf("unsupported option type %T", opt)
		}
	}
	const apiKeyURL = "https://cloud.cerebras.ai/platform/"
	var err error
	if apiKey == "" {
		if apiKey = os.Getenv("CEREBRAS_API_KEY"); apiKey == "" {
			err = &base.ErrAPIKeyRequired{EnvVar: "CEREBRAS_API_KEY", URL: apiKeyURL}
		}
	}
	h := http.Header{"Authorization": {"Bearer " + apiKey}}
	if queueThreshold != 0 {
		h.Set("queue_threshold", strconv.FormatInt(queueThreshold.Milliseconds(), 10))
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
			GenSyncURL:      "https://api.cerebras.ai/v1/chat/completions",
			ProcessStream:   ProcessStream,
			ProcessHeaders:  processHeaders,
			PreloadedModels: preloadedModels,
			LieToolCalls:    true,
			ProviderBase: base.ProviderBase[*ErrorResponse]{
				APIKeyURL: apiKeyURL,
				Lenient:   internal.BeLenient,
				Client: http.Client{
					Transport: &roundtrippers.Header{
						Header: h,
						Transport: &queueThresholdHeader{
							transport: &roundtrippers.RequestID{Transport: t},
						},
					},
				},
			},
		},
	}
	if err == nil {
		switch model {
		case "":
		case string(genai.ModelCheap), string(genai.ModelGood), string(genai.ModelSOTA):
			if c.impl.Model, err = c.selectBestTextModel(ctx); err != nil {
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

// selectBestTextModel selects Gemma 4 31B, Cerebras' preferred model for text generation.
func (c *Client) selectBestTextModel(ctx context.Context) (string, error) {
	const model = "gemma-4-31b"
	mdls, err := c.ListModels(ctx)
	if err != nil {
		return "", fmt.Errorf("failed to automatically select the model: %w", err)
	}
	for _, mdl := range mdls {
		if mdl.GetID() == model {
			return model, nil
		}
	}
	return "", fmt.Errorf("failed to find preferred model %q", model)
}

// Name implements genai.Provider.
//
// It returns the name of the provider.
func (c *Client) Name() string {
	return "cerebras"
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
//
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
	return c.impl.GenSync(ctxWithQueueThreshold(ctx, opts), msgs, opts...)
}

// GenSyncRaw provides access to the raw API.
func (c *Client) GenSyncRaw(ctx context.Context, in *ChatRequest, out *ChatResponse) error {
	return c.impl.GenSyncRaw(ctx, in, out)
}

// GenStream implements genai.Provider.
func (c *Client) GenStream(ctx context.Context, msgs genai.Messages, opts ...genai.GenOption) (iter.Seq[genai.Reply], func() (genai.Result, error)) {
	return c.impl.GenStream(ctxWithQueueThreshold(ctx, opts), msgs, opts...)
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
	// https://inference-docs.cerebras.ai/api-reference/models
	var resp ModelsResponse
	if err := c.impl.DoRequest(ctx, "GET", "https://api.cerebras.ai/v1/models", nil, &resp); err != nil {
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
			// gpt-oss-* streams the tool call arguments but not the other models. Fun.
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
					u.ReasoningTokens = pkt.Usage.CompletionTokensDetails.ReasoningTokens
					u.OutputTokens = pkt.Usage.CompletionTokens
					u.TotalTokens = pkt.Usage.TotalTokens
					u.FinishReason = pkt.Choices[0].FinishReason.ToFinishReason()
				}

				for _, nt := range pkt.Choices[0].Delta.ToolCalls {
					// We need to determine if the model streams too calls or not. gpt-oss-* streams the tool call arguments
					// but not others.
					if pendingToolCall.ID != "" {
						if nt.ID == "" {
							// Continuation.
							pendingToolCall.Function.Arguments += nt.Function.Arguments
						} else {
							// Flush first.
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
				if !yield(genai.Reply{Reasoning: pkt.Choices[0].Delta.Reasoning}) {
					return
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
	requestsLimit, _ := strconv.ParseInt(h.Get("X-Ratelimit-Limit-Requests-Day"), 10, 64)
	requestsRemaining, _ := strconv.ParseInt(h.Get("X-Ratelimit-Remaining-Requests-Day"), 10, 64)
	requestsReset, _ := time.ParseDuration(h.Get("X-Ratelimit-Reset-Requests-Day") + "s")

	tokensLimit, _ := strconv.ParseInt(h.Get("X-Ratelimit-Limit-Tokens-Minute"), 10, 64)
	tokensRemaining, _ := strconv.ParseInt(h.Get("X-Ratelimit-Remaining-Tokens-Minute"), 10, 64)
	tokensReset, _ := time.ParseDuration(h.Get("X-Ratelimit-Reset-Tokens-Minute") + "s")

	var limits []genai.RateLimit
	if requestsLimit > 0 {
		limits = append(limits, genai.RateLimit{
			Type:      genai.Requests,
			Period:    genai.PerDay,
			Limit:     requestsLimit,
			Remaining: requestsRemaining,
			Reset:     time.Now().Add(requestsReset).Round(10 * time.Millisecond),
		})
	}
	if tokensLimit > 0 {
		limits = append(limits, genai.RateLimit{
			Type:      genai.Tokens,
			Period:    genai.PerMinute,
			Limit:     tokensLimit,
			Remaining: tokensRemaining,
			Reset:     time.Now().Add(tokensReset).Round(10 * time.Millisecond),
		})
	}
	return limits
}

var _ genai.Provider = &Client{}
