// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Package openaicompatible implements a lenient client for "OpenAI-compatible" providers.
//
// It supports text and Chat Completions function tools without provider-specific schema
// restrictions. Unknown response fields and finish reasons are accepted. Tool arguments
// must still be valid JSON. Tool support depends on the remote model.
//
// Alibaba, Baseten, Cerebras, DeepSeek, Groq, HuggingFace, Mistral, OpenAI Chat,
// OpenRouter, Pollinations, TogetherAI, and Xiaomi share this function-tool protocol.
// This client does not translate Anthropic or Cohere tools, Cloudflare's model-specific
// requests, or provider-specific reasoning modes. Perplexity does not support function
// tools. Use a dedicated provider when a model requires its specific fields.
//
// See https://platform.openai.com/docs/api-reference/chat/create.
package openaicompatible

import (
	"bytes"
	"context"
	_ "embed"
	"encoding/json"
	"errors"
	"fmt"
	"iter"
	"net/http"
	"slices"

	"github.com/maruel/roundtrippers"

	"github.com/maruel/genai"
	"github.com/maruel/genai/base"
	"github.com/maruel/genai/internal"
	"github.com/maruel/genai/scoreboard"
)

//go:embed scoreboard.json
var scoreboardJSON []byte

// Scoreboard for generic OpenAI compatible API.
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

// New creates a new client to talk to an "OpenAI-compatible" platform API.
//
// It supports text exchanges (no multi-modal) and Chat Completions function tools.
// Tool support and tool choice modes depend on the remote model.
//
// Option ProviderOptionRemote must be set.
//
// Automatic model selection via ModelCheap, ModelGood, ModelSOTA is not supported and it will specify no
// model in this case.
//
// Exceptionally it will interpret Model set to "" as no model to specify since there is no automatic model
// selection.
func New(ctx context.Context, opts ...genai.ProviderOption) (*Client, error) {
	var apiKey, model, remote string
	var modalities genai.Modalities
	var preloadedModels []genai.Model
	var wrapper func(http.RoundTripper) http.RoundTripper
	if err := base.CheckDuplicateProviderOptions(opts); err != nil {
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
		default:
			return nil, fmt.Errorf("unsupported option type %T", opt)
		}
	}
	if apiKey != "" {
		return nil, errors.New("unexpected option ProviderOptionAPIKey")
	}
	if remote == "" {
		return nil, errors.New("option ProviderOptionRemote is required")
	}
	mod := genai.Modalities{genai.ModalityText}
	if len(modalities) != 0 && !slices.Equal(modalities, mod) {
		return nil, fmt.Errorf("unexpected option Modalities %s, only text is supported", mod)
	}
	switch model {
	case "", string(genai.ModelCheap), string(genai.ModelGood), string(genai.ModelSOTA):
		model = ""
	}
	t := base.DefaultTransport
	if wrapper != nil {
		t = wrapper(t)
	}
	return &Client{
		impl: base.Provider[*ErrorResponse, *ChatRequest, *ChatResponse, ChatStreamChunkResponse]{
			GenSyncURL:      remote,
			ProcessStream:   ProcessStream,
			PreloadedModels: preloadedModels,
			ProviderBase: base.ProviderBase[*ErrorResponse]{
				Model:            model,
				ModelOptional:    true,
				OutputModalities: mod,
				// It is always lenient by definition.
				Lenient: true,
				Client: http.Client{
					Transport: &roundtrippers.RequestID{Transport: t},
				},
			},
		},
	}, nil
}

// Close implements io.Closer. It currently does nothing.
func (c *Client) Close() error {
	return nil
}

// Name implements genai.Provider.
//
// It returns the name of the provider.
func (c *Client) Name() string {
	return "openaicompatible"
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

// ProcessStream converts the raw packets from the streaming API into Reply fragments.
//
// Tool calls stay pending across text and usage packets and flush at stream end.
// Providers can repeat or omit IDs and names. A new ID starts a separate call even
// when it reuses a stream index. TogetherAI can send whole calls outside the delta.
func ProcessStream(chunks iter.Seq[ChatStreamChunkResponse]) (iter.Seq[genai.Reply], func() (genai.Usage, [][]genai.Logprob, error)) {
	var finalErr error
	u := genai.Usage{}

	return func(yield func(genai.Reply) bool) {
			var pending []ToolCall
			for pkt := range chunks {
				if pkt.Usage.TotalTokens != 0 {
					u.InputTokens = pkt.Usage.PromptTokens
					u.OutputTokens = pkt.Usage.CompletionTokens
					u.TotalTokens = pkt.Usage.TotalTokens
				}
				if len(pkt.Choices) > 1 {
					finalErr = &internal.BadError{Err: fmt.Errorf("expected at most one choice, got %d", len(pkt.Choices))}
					return
				}
				m := pkt.Delta.Message
				var whole []ToolCall
				if len(pkt.Choices) == 1 {
					m = pkt.Choices[0].Delta.Message
					whole = pkt.Choices[0].ToolCalls
					if pkt.Choices[0].FinishReason != "" {
						u.FinishReason = pkt.Choices[0].FinishReason.ToFinishReason()
					}
				} else if pkt.FinishReason != "" {
					u.FinishReason = pkt.FinishReason.ToFinishReason()
				}
				switch role := m.Role; role {
				case "", "assistant":
				default:
					finalErr = &internal.BadError{Err: fmt.Errorf("unexpected role %q", role)}
					return
				}
				if len(pkt.Choices) == 0 && m.IsZero() && pkt.Delta.Text != "" {
					if !yield(genai.Reply{Text: pkt.Delta.Text}) {
						return
					}
				}
				for _, content := range m.Content {
					if content.Type != "" && content.Type != ContentText {
						finalErr = &internal.BadError{Err: fmt.Errorf("unsupported content type %q", content.Type)}
						return
					}
					if content.Text != "" {
						if !yield(genai.Reply{Text: content.Text}) {
							return
						}
					}
				}
				for i, tc := range m.ToolCalls {
					idx := -1
					for j := len(pending) - 1; j >= 0; j-- {
						p := &pending[j]
						if tc.ID != "" && tc.ID == p.ID || tc.Index != nil && p.Index != nil && *tc.Index == *p.Index && (tc.ID == "" || p.ID == "") {
							idx = j
							break
						}
					}
					// Without indexes or IDs, use packet position for parallel deltas,
					// and the last unfinished call for serial deltas.
					if idx == -1 && tc.Index == nil && tc.ID == "" && len(pending) > 0 {
						j := len(pending) - 1
						if len(m.ToolCalls) > 1 {
							j = i
						}
						if j < len(pending) && (tc.Function.Name == "" || tc.Function.Name == pending[j].Function.Name) && (tc.Function.Name == "" || tc.Function.Arguments == "" || !json.Valid([]byte(pending[j].Function.Arguments))) {
							idx = j
						}
					}
					if idx == -1 {
						pending = append(pending, tc)
						continue
					}
					p := &pending[idx]
					if tc.ID != "" {
						p.ID = tc.ID
					}
					if tc.Function.Name != "" && tc.Function.Name != p.Function.Name {
						p.Function.Name += tc.Function.Name
					}
					p.Function.Arguments += tc.Function.Arguments
				}
				// Whole choice-level calls replace deltas for the same call.
				for _, tc := range whole {
					idx := slices.IndexFunc(pending, func(p ToolCall) bool {
						return tc.ID != "" && tc.ID == p.ID || tc.Index != nil && p.Index != nil && *tc.Index == *p.Index && (tc.ID == "" || p.ID == "")
					})
					if idx == -1 {
						pending = append(pending, tc)
					} else {
						pending[idx] = tc
					}
				}
			}
			for i := range pending {
				var t genai.ToolCall
				if err := pending[i].To(&t); err != nil {
					finalErr = &internal.BadError{Err: fmt.Errorf("tool call: %w", err)}
					return
				}
				if !yield(genai.Reply{ToolCall: t}) {
					return
				}
			}
			normalizeToolFinish(&u, len(pending) != 0)
		}, func() (genai.Usage, [][]genai.Logprob, error) {
			return u, nil, finalErr
		}
}

var _ genai.Provider = &Client{}
