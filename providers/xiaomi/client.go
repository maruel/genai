// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Package xiaomi implements a client for the Xiaomi MiMo platform API.
//
// It uses the OpenAI-compatible endpoint which supports reasoning, tool calling, multimodal input,
// and structured output.
//
// See https://platform.xiaomimimo.com/docs/en-US/api/chat/openai-api
package xiaomi

import (
	"bytes"
	"context"
	_ "embed"
	"encoding/base64"
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
	"github.com/maruel/genai/internal/bb"
	"github.com/maruel/genai/scoreboard"
)

//go:embed scoreboard.json
var scoreboardJSON []byte

// Scoreboard for Xiaomi MiMo.
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

// New creates a new client to talk to the Xiaomi MiMo platform API.
//
// If ProviderOptionAPIKey is not provided, it tries to load it from the MIMO_API_KEY environment variable.
// If none is found, it will still return a client coupled with a base.ErrAPIKeyRequired error.
// Get your API key at https://platform.xiaomimimo.com/#/console
//
// To use multiple models, create multiple clients.
// Use one of the models from https://platform.xiaomimimo.com/docs/en-US/api/chat/openai-api
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
	const apiKeyURL = "https://platform.xiaomimimo.com/#/console"
	var err error
	if apiKey == "" {
		if apiKey = os.Getenv("MIMO_API_KEY"); apiKey == "" {
			err = &base.ErrAPIKeyRequired{EnvVar: "MIMO_API_KEY", URL: apiKeyURL}
		}
	}
	mod := genai.Modalities{genai.ModalityText}
	if len(modalities) != 0 {
		switch {
		case slices.Equal(modalities, genai.Modalities{genai.ModalityText}):
		case slices.Equal(modalities, genai.Modalities{genai.ModalityAudio}):
			mod = genai.Modalities{genai.ModalityAudio}
		default:
			return nil, fmt.Errorf("unexpected option Modalities %s, only text or audio is supported", modalities)
		}
	}
	t := base.DefaultTransport
	if wrapper != nil {
		t = wrapper(t)
	}
	c := &Client{
		impl: base.Provider[*ErrorResponse, *ChatRequest, *ChatResponse, ChatStreamChunkResponse]{
			GenSyncURL:      "https://api.xiaomimimo.com/v1/chat/completions",
			ProcessStream:   makeProcessStream(""),
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
			if mod[0] == genai.ModalityAudio {
				if c.impl.Model, err = c.selectBestAudioModel(ctx); err != nil {
					return nil, err
				}
			} else {
				if c.impl.Model, err = c.selectBestTextModel(ctx, model); err != nil {
					return nil, err
				}
			}
			c.impl.OutputModalities = mod
		default:
			c.impl.Model = model
			switch {
			case len(modalities) != 0:
				c.impl.OutputModalities = mod
			case strings.Contains(model, "tts"):
				c.impl.OutputModalities = genai.Modalities{genai.ModalityAudio}
			default:
				c.impl.OutputModalities = mod
			}
		}
	}
	return c, err
}

// selectBestTextModel selects the most appropriate model based on the preference (cheap, good, or SOTA).
func (c *Client) selectBestTextModel(ctx context.Context, preference string) (string, error) {
	mdls, err := c.ListModels(ctx)
	if err != nil {
		return "", fmt.Errorf("failed to automatically select the model: %w", err)
	}
	// ModelGood and ModelSOTA both select mimo-v2.5-pro.
	want := "mimo-v2.5-pro"
	if preference == string(genai.ModelCheap) {
		want = "mimo-v2.5"
	}
	for _, mdl := range mdls {
		if mdl.(*Model).ID == want {
			return want, nil
		}
	}
	return "", errors.New("failed to find a model automatically")
}

// selectBestAudioModel selects the most appropriate audio model based on the preference (cheap, good, or SOTA).
//
// Audio models are identified by "tts" in their name.
func (c *Client) selectBestAudioModel(ctx context.Context) (string, error) {
	mdls, err := c.ListModels(ctx)
	if err != nil {
		return "", fmt.Errorf("failed to automatically select the model: %w", err)
	}
	// All TTS models are equivalent; pick the base one.
	want := "mimo-v2.5-tts"
	for _, mdl := range mdls {
		if mdl.(*Model).ID == want {
			return want, nil
		}
	}
	return "", errors.New("failed to find an audio model automatically")
}

// Name implements genai.Provider.
//
// It returns the name of the provider.
func (c *Client) Name() string {
	return "xiaomi"
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
	// Build the request ourselves so GenSyncRaw can set audioFormat on the response.
	in := &ChatRequest{}
	if err := in.Init(msgs, c.impl.Model, opts...); err != nil {
		return genai.Result{}, err
	}
	out := &ChatResponse{}
	if err := c.GenSyncRaw(ctx, in, out); err != nil {
		return genai.Result{}, err
	}
	return out.ToResult()
}

// GenSyncRaw provides access to the raw API.
func (c *Client) GenSyncRaw(ctx context.Context, in *ChatRequest, out *ChatResponse) error {
	out.audioFormat = in.Audio.Format
	return c.impl.GenSyncRaw(ctx, in, out)
}

// GenStream implements genai.Provider.
func (c *Client) GenStream(ctx context.Context, msgs genai.Messages, opts ...genai.GenOption) (iter.Seq[genai.Reply], func() (genai.Result, error)) {
	// Build the request ourselves so we can extract the audio format for stream processing.
	in := &ChatRequest{}
	if err := in.Init(msgs, c.impl.Model, opts...); err != nil {
		return func(yield func(genai.Reply) bool) {}, func() (genai.Result, error) { return genai.Result{}, err }
	}
	format := in.Audio.Format

	chunks, finishRaw := c.impl.GenStreamRaw(ctx, in)
	processStream := makeProcessStream(format)
	fragments, finishUsage := processStream(chunks)

	res := genai.Result{}
	fnFragments := func(yield func(genai.Reply) bool) {
		for f := range fragments {
			if f.IsZero() {
				continue
			}
			if err := f.Validate(); err != nil {
				break
			}
			if err := res.Accumulate(&f); err != nil {
				break
			}
			if !yield(f) {
				break
			}
		}
	}
	fnFinish := func() (genai.Result, error) {
		err := finishRaw()
		var usageErr error
		res.Usage, res.Logprobs, usageErr = finishUsage()
		err = errors.Join(err, usageErr)
		if err != nil {
			return res, err
		}
		if err := res.Validate(); err != nil {
			return res, &internal.BadError{Err: err}
		}
		return res, nil
	}
	return fnFragments, fnFinish
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
	if err := c.impl.DoRequest(ctx, "GET", "https://api.xiaomimimo.com/v1/models", nil, &resp); err != nil {
		return nil, err
	}
	return resp.ToModels(), nil
}

// makeProcessStream returns a stream processor that uses the given audio format for Doc filenames.
func makeProcessStream(audioFormat string) func(iter.Seq[ChatStreamChunkResponse]) (iter.Seq[genai.Reply], func() (genai.Usage, [][]genai.Logprob, error)) {
	return func(chunks iter.Seq[ChatStreamChunkResponse]) (iter.Seq[genai.Reply], func() (genai.Usage, [][]genai.Logprob, error)) {
		var finalErr error
		u := genai.Usage{}

		return func(yield func(genai.Reply) bool) {
				pendingToolCall := ToolCall{}
				var audioBuf []byte
				for pkt := range chunks {
					// Extract usage from the final chunk (which has empty choices but populated usage).
					if pkt.Usage.CompletionTokens != 0 {
						u.InputTokens = pkt.Usage.PromptTokens
						u.InputCachedTokens = pkt.Usage.PromptTokensDetails.CachedTokens
						u.ReasoningTokens = pkt.Usage.CompletionTokensDetails.ReasoningTokens
						u.OutputTokens = pkt.Usage.CompletionTokens
					}
					if len(pkt.Choices) != 1 {
						continue
					}
					// Extract finish reason from the chunk that has it.
					if pkt.Choices[0].FinishReason != "" {
						u.FinishReason = pkt.Choices[0].FinishReason.ToFinishReason()
					}
					if len(pkt.Choices[0].Delta.ToolCalls) > 1 {
						finalErr = &internal.BadError{Err: fmt.Errorf("implement multiple tool calls: %#v", pkt)}
						return
					}
					switch role := pkt.Choices[0].Delta.Role; role {
					case "assistant", "":
					default:
						finalErr = &internal.BadError{Err: fmt.Errorf("unexpected role %q", role)}
						return
					}
					// Handle TTS audio streaming. Audio data arrives in delta.audio.data as base64 chunks.
					if pkt.Choices[0].Delta.Audio.Data != "" {
						chunk, err := base64.StdEncoding.DecodeString(pkt.Choices[0].Delta.Audio.Data)
						if err != nil {
							finalErr = &internal.BadError{Err: fmt.Errorf("failed to decode audio chunk: %w", err)}
							return
						}
						audioBuf = append(audioBuf, chunk...)
						continue
					}
					var text strings.Builder
					for _, c := range pkt.Choices[0].Delta.Content {
						if c.Type == ContentText {
							text.WriteString(c.Text)
						}
					}
					f := genai.Reply{
						Text:      text.String(),
						Reasoning: pkt.Choices[0].Delta.ReasoningContent,
					}
					// Handle web search citations.
					for _, a := range pkt.Choices[0].Delta.Annotations {
						if a.Type != "url_citation" {
							if !internal.BeLenient {
								finalErr = &internal.BadError{Err: fmt.Errorf("unsupported annotation type %q", a.Type)}
								return
							}
							continue
						}
						c := genai.Citation{
							Sources: []genai.CitationSource{{Type: genai.CitationWeb, Title: a.Title, URL: a.URL}},
						}
						if !yield(genai.Reply{Citation: c}) {
							return
						}
					}
					// MiMo streams the arguments. Buffer the arguments to send the fragment as a whole tool call.
					if len(pkt.Choices[0].Delta.ToolCalls) == 1 {
						if t := pkt.Choices[0].Delta.ToolCalls[0]; t.ID != "" {
							// A new call.
							if pendingToolCall.ID == "" {
								pendingToolCall = t
								if !f.IsZero() {
									finalErr = &internal.BadError{Err: fmt.Errorf("implement tool call with metadata: %#v", pkt)}
									return
								}
								continue
							}
							// Flush.
							pendingToolCall.To(&f.ToolCall)
							pendingToolCall = t
						} else if pendingToolCall.ID != "" {
							// Continuation.
							pendingToolCall.Function.Arguments += t.Function.Arguments
							if !f.IsZero() {
								finalErr = &internal.BadError{Err: fmt.Errorf("implement tool call with metadata: %#v", pkt)}
								return
							}
							continue
						}
					} else if pendingToolCall.ID != "" {
						// Flush.
						pendingToolCall.To(&f.ToolCall)
						pendingToolCall = ToolCall{}
					}
					if !yield(f) {
						return
					}
				}
				// Flush accumulated audio data as a single document.
				if len(audioBuf) > 0 {
					if !yield(genai.Reply{
						Doc: genai.Doc{
							Filename: "audio." + audioFormat,
							Src:      &bb.BytesBuffer{D: audioBuf},
						},
					}) {
						return
					}
				}
			}, func() (genai.Usage, [][]genai.Logprob, error) {
				return u, nil, finalErr
			}
	}
}

var _ genai.Provider = &Client{}
