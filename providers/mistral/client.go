// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Package mistral implements a client for the Mistral API.
//
// It is described at https://docs.mistral.ai/api/
package mistral

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

// Scoreboard for Mistral.
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

// TODO:
// https://codestral.mistral.ai/v1/fim/completions
// https://codestral.mistral.ai/v1/chat/completions

// New creates a new client to talk to the Mistral platform API.
//
// If ProviderOptionAPIKey is not provided, it tries to load it from the MISTRAL_API_KEY environment variable.
// If none is found, it will still return a client coupled with an base.ErrAPIKeyRequired error.
// Get your API key at https://console.mistral.ai/api-keys or https://console.mistral.ai/codestral
//
// To use multiple models, create multiple clients.
// Use one of the model from https://docs.mistral.ai/getting-started/models/models_overview/
//
// # PDF understanding
//
// PDF understanding requires a model which has the "OCR" or the "Document understanding" capability. There's
// a subtle difference between the two; from what I understand, the document understanding will only parse the
// text, while the OCR will try to understand the pictures.
//
// https://docs.mistral.ai/capabilities/document/
// https://docs.mistral.ai/capabilities/vision/
//
// # Tool use
//
// Tool use requires a model which has the tool capability. See
// https://docs.mistral.ai/capabilities/function_calling/
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
	const apiKeyURL = "https://console.mistral.ai/api-keys"
	var err error
	if apiKey == "" {
		if apiKey = os.Getenv("MISTRAL_API_KEY"); apiKey == "" {
			err = &base.ErrAPIKeyRequired{EnvVar: "MISTRAL_API_KEY", URL: apiKeyURL}
		}
	}
	mod := genai.Modalities{genai.ModalityText}
	if len(modalities) != 0 && !slices.Equal(modalities, mod) {
		// https://docs.mistral.ai/agents/connectors/image_generation/
		return nil, fmt.Errorf("unexpected option Modalities %s, only text is implemented (send PR to add support)", mod)
	}
	t := base.DefaultTransport
	if wrapper != nil {
		t = wrapper(t)
	}
	c := &Client{
		impl: base.Provider[*ErrorResponse, *ChatRequest, *ChatResponse, ChatStreamChunkResponse]{
			GenSyncURL:      "https://api.mistral.ai/v1/chat/completions",
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
	for _, mdl := range mdls {
		m := mdl.(*Model)
		// TODO: Support magistral.
		if !strings.HasSuffix(m.ID, "latest") || strings.HasPrefix(m.ID, "devstral") || strings.HasPrefix(m.ID, "magistral") || strings.HasPrefix(m.ID, "pixtral") || strings.HasPrefix(m.ID, "voxtral") {
			continue
		}
		switch {
		case cheap:
			if strings.Contains(m.ID, "small") {
				selectedModel = m.ID
			}
		case good:
			if strings.Contains(m.ID, "medium") {
				selectedModel = m.ID
			}
		default:
			if strings.Contains(m.ID, "large") {
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
	return "mistral"
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
	// https://docs.mistral.ai/api/#tag/models
	var resp ModelsResponse
	if err := c.impl.DoRequest(ctx, "GET", "https://api.mistral.ai/v1/models", nil, &resp); err != nil {
		return nil, err
	}
	return resp.ToModels(), nil
}

// ProcessStream converts the raw packets from the streaming API into Reply fragments.
func ProcessStream(chunks iter.Seq[ChatStreamChunkResponse]) (iter.Seq[genai.Reply], func() (genai.Usage, [][]genai.Logprob, error)) {
	var finalErr error
	u := genai.Usage{}

	return func(yield func(genai.Reply) bool) {
			for pkt := range chunks {
				if len(pkt.Choices) != 1 {
					continue
				}
				if pkt.Usage.PromptTokens != 0 {
					u.InputTokens = pkt.Usage.PromptTokens
					u.OutputTokens = pkt.Usage.CompletionTokens
					u.TotalTokens = pkt.Usage.TotalTokens
					u.FinishReason = pkt.Choices[0].FinishReason.ToFinishReason()
				}
				switch role := pkt.Choices[0].Delta.Role; role {
				case "assistant", "":
				default:
					finalErr = &internal.BadError{Err: fmt.Errorf("unexpected role %q", role)}
					return
				}
				if !yield(genai.Reply{Text: pkt.Choices[0].Delta.Content}) {
					return
				}
				// Mistral is one of the rare provider that can stream multiple tool calls all at once. It's probably
				// because it's buffering server-side.
				for i := range pkt.Choices[0].Delta.ToolCalls {
					f := genai.Reply{}
					pkt.Choices[0].Delta.ToolCalls[i].To(&f.ToolCall)
					if !yield(f) {
						return
					}
				}
			}
		}, func() (genai.Usage, [][]genai.Logprob, error) {
			return u, nil, finalErr
		}
}

func processHeaders(h http.Header) []genai.RateLimit {
	var limits []genai.RateLimit
	requestsLimit, _ := strconv.ParseInt(h.Get("X-Ratelimit-Limit-Req-10-Second"), 10, 64)
	requestsRemaining, _ := strconv.ParseInt(h.Get("X-Ratelimit-Remaining-Req-10-Second"), 10, 64)

	tokensPerMinLimit, _ := strconv.ParseInt(h.Get("X-Ratelimit-Limit-Tokens-Minute"), 10, 64)
	tokensPerMinRemaining, _ := strconv.ParseInt(h.Get("X-Ratelimit-Remaining-Tokens-Minute"), 10, 64)

	tokensPerMonthLimit, _ := strconv.ParseInt(h.Get("X-Ratelimit-Limit-Tokens-Month"), 10, 64)
	tokensPerMonthRemaining, _ := strconv.ParseInt(h.Get("X-Ratelimit-Remaining-Tokens-Month"), 10, 64)

	if requestsLimit > 0 {
		limits = append(limits, genai.RateLimit{
			Type:      genai.Requests,
			Period:    genai.PerOther, // 10 seconds is not a standard period
			Limit:     requestsLimit,
			Remaining: requestsRemaining,
			Reset:     time.Now().Add(10 * time.Second).Round(10 * time.Millisecond),
		})
	}
	if tokensPerMinLimit > 0 {
		limits = append(limits, genai.RateLimit{
			Type:      genai.Tokens,
			Period:    genai.PerMinute,
			Limit:     tokensPerMinLimit,
			Remaining: tokensPerMinRemaining,
			Reset:     time.Now().Add(time.Minute).Round(10 * time.Millisecond),
		})
	}
	if tokensPerMonthLimit > 0 {
		limits = append(limits, genai.RateLimit{
			Type:      genai.Tokens,
			Period:    genai.PerMonth,
			Limit:     tokensPerMonthLimit,
			Remaining: tokensPerMonthRemaining,
			// This is not accurate, but there's no reset header.
			Reset: time.Now().Add(30 * 24 * time.Hour).Round(10 * time.Millisecond),
		})
	}
	return limits
}

var _ genai.Provider = &Client{}
