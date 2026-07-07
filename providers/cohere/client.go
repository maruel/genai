// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Package cohere implements a client for the Cohere API.
//
// It is described at https://docs.cohere.com/reference/
package cohere

// See official client at https://github.com/cohere-ai/cohere-go

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
	"strings"

	"github.com/maruel/roundtrippers"

	"github.com/maruel/genai"
	"github.com/maruel/genai/base"
	"github.com/maruel/genai/internal"
	"github.com/maruel/genai/scoreboard"
)

//go:embed scoreboard.json
var scoreboardJSON []byte

// Scoreboard for Cohere.
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

// New creates a new client to talk to the Cohere platform API.
//
// If ProviderOptionAPIKey is not provided, it tries to load it from the COHERE_API_KEY environment variable.
// If none is found, it will still return a client coupled with an base.ErrAPIKeyRequired error.
// Get your API key at https://dashboard.cohere.com/api-keys
//
// Use one of the model from https://cohere.com/pricing and https://docs.cohere.com/v2/docs/models
// To use multiple models, create multiple clients.
//
// # Tool use
//
// Tool use requires the use a model that supports structured output.
// https://docs.cohere.com/v2/docs/structured-outputs
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
	const apiKeyURL = "https://dashboard.cohere.com/api-keys"
	var err error
	if apiKey == "" {
		if apiKey = os.Getenv("COHERE_API_KEY"); apiKey == "" {
			err = &base.ErrAPIKeyRequired{EnvVar: "COHERE_API_KEY", URL: apiKeyURL}
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
			GenSyncURL:      "https://api.cohere.com/v2/chat",
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
	var ctxLen int64

	for _, mdl := range mdls {
		m := mdl.(*Model)
		if !slices.Contains(m.Endpoints, "chat") || strings.Contains(m.Name, "nightly") {
			continue
		}
		if !slices.Contains(m.Features, "strict_tools") {
			continue
		}
		switch {
		case cheap:
			if strings.Contains(m.Name, "r7b") && !strings.Contains(m.Name, "arabic") && (ctxLen == 0 || ctxLen < m.ContextLength) {
				ctxLen = m.ContextLength
				selectedModel = m.Name
			}
		case good:
			// Prefer reasoning models as they're more capable.
			if strings.Contains(m.Name, "reasoning") && (ctxLen == 0 || ctxLen < m.ContextLength) {
				ctxLen = m.ContextLength
				selectedModel = m.Name
			}
		default:
			if strings.Contains(m.Name, "reasoning") && (ctxLen == 0 || ctxLen < m.ContextLength) {
				// For the greatest, we want the largest context.
				ctxLen = m.ContextLength
				selectedModel = m.Name
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
	return "cohere"
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
	// https://docs.cohere.com/reference/list-models
	var resp ModelsResponse
	if err := c.impl.DoRequest(ctx, "GET", "https://api.cohere.com/v1/models?page_size=1000", nil, &resp); err != nil {
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
			wasThinking := false
			pendingToolCall := ToolCall{}
			for pkt := range chunks {
				// These can't happen.
				if len(pkt.Delta.Message.Content) > 1 {
					finalErr = &internal.BadError{Err: errors.New("implement multiple content")}
					return
				}
				if len(pkt.Delta.Message.ToolCalls) > 1 {
					finalErr = &internal.BadError{Err: errors.New("implement multiple tool calls")}
					return
				}
				switch role := pkt.Delta.Message.Role; role {
				case "assistant", "":
				default:
					finalErr = &internal.BadError{Err: fmt.Errorf("unexpected role %q", role)}
					return
				}
				if pkt.Logprobs.Text != "" {
					l = append(l, pkt.Logprobs.To())
				}
				f := genai.Reply{}
				switch pkt.Type {
				case ChunkMessageStart:
					// Nothing useful.
					if len(pkt.Delta.Message.Content) != 0 {
						finalErr = &internal.BadError{Err: fmt.Errorf("expected no content %#v", pkt)}
						return
					}
				case ChunkMessageEnd:
					// Contain usage and finish reason.
					if pkt.Delta.FinishReason == FinishError {
						finalErr = errors.New(pkt.Delta.Error)
						return
					}
					if len(pkt.Delta.Message.Content) != 0 {
						finalErr = &internal.BadError{Err: fmt.Errorf("expected no content %#v", pkt)}
						return
					}
					u.InputTokens = pkt.Delta.Usage.Tokens.InputTokens
					u.InputCachedTokens = pkt.Delta.Usage.CachedTokens
					u.OutputTokens = pkt.Delta.Usage.Tokens.OutputTokens
					u.FinishReason = pkt.Delta.FinishReason.ToFinishReason()
				case ChunkContentStart:
					if len(pkt.Delta.Message.Content) != 1 {
						finalErr = &internal.BadError{Err: fmt.Errorf("expected content %#v", pkt)}
						return
					}
					switch t := pkt.Delta.Message.Content[0].Type; t {
					case ContentText:
						f.Text = pkt.Delta.Message.Content[0].Text
						wasThinking = false
					case ContentThinking:
						f.Reasoning = pkt.Delta.Message.Content[0].Text
						wasThinking = true
					case ContentDocument, ContentImageURL:
						finalErr = &internal.BadError{Err: fmt.Errorf("implement content %q", t)}
					default:
						finalErr = &internal.BadError{Err: fmt.Errorf("implement content %q", t)}
						return
					}
				case ChunkContentDelta:
					// Sometimes the delta is empty?
					if len(pkt.Delta.Message.Content) == 1 {
						t := pkt.Delta.Message.Content[0].Text
						if t == "" {
							finalErr = &internal.BadError{Err: fmt.Errorf("empty content %#v", pkt)}
							return
						}
						// Type is not set. We need to remember the value from ChunkContentStart.
						if wasThinking {
							f.Reasoning = t
						} else {
							f.Text = t
						}
					}
				case ChunkContentEnd:
					// Will be useful when there's multiple index.
					// Sometimes the delta is empty?
					if len(pkt.Delta.Message.Content) != 0 {
						finalErr = &internal.BadError{Err: fmt.Errorf("unexpected content %#v", pkt)}
						return
					}
				case ChunkToolPlanDelta:
					f.Reasoning = pkt.Delta.Message.ToolPlan
				case ChunkToolCallStart:
					if len(pkt.Delta.Message.ToolCalls) != 1 {
						finalErr = &internal.BadError{Err: fmt.Errorf("expected tool call %#v", pkt)}
						return
					}
					pendingToolCall = pkt.Delta.Message.ToolCalls[0]
				case ChunkToolCallDelta:
					pendingToolCall.Function.Arguments += pkt.Delta.Message.ToolCalls[0].Function.Arguments
				case ChunkToolCallEnd:
					pendingToolCall.To(&f.ToolCall)
					pendingToolCall = ToolCall{}
				case ChunkCitationStart:
					if len(pkt.Delta.Message.Citations) != 1 {
						finalErr = &internal.BadError{Err: fmt.Errorf("expected one citation, got %v", pkt)}
						return
					}
					if err := pkt.Delta.Message.Citations[0].To(&f.Citation); err != nil {
						finalErr = err
						return
					}
				case ChunkCitationEnd:
					if len(pkt.Delta.Message.Citations) != 0 {
						finalErr = &internal.BadError{Err: fmt.Errorf("expected no citations, got %v", pkt)}
						return
					}
				default:
					if !internal.BeLenient {
						finalErr = &internal.BadError{Err: fmt.Errorf("unknown packet %q", pkt.Type)}
						return
					}
				}
				if !yield(f) {
					return
				}
			}
		}, func() (genai.Usage, [][]genai.Logprob, error) {
			return u, l, finalErr
		}
}

var _ genai.Provider = &Client{}
