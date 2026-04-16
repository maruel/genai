// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Package ollama implements a client for the Ollama API.
//
// It is described at https://github.com/ollama/ollama/blob/main/docs/api.md
// and https://pkg.go.dev/github.com/ollama/ollama/api
package ollama

import (
	"bufio"
	"bytes"
	"context"
	_ "embed"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"iter"
	"net/http"
	"slices"
	"strings"

	"github.com/maruel/genai"
	"github.com/maruel/genai/base"
	"github.com/maruel/genai/internal"
	"github.com/maruel/genai/scoreboard"
	"github.com/maruel/roundtrippers"
	"golang.org/x/sync/errgroup"
)

//go:embed scoreboard.json
var scoreboardJSON []byte

// Scoreboard for Ollama.
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

// We cannot use ClientChat because GenSync and GenStream try to pull on first failure, and GenStream receives
// line separated JSON instead of SSE.

// Client implements genai.Provider.
type Client struct {
	base.NotImplemented
	impl            base.ProviderBase[*ErrorResponse]
	preloadedModels []genai.Model
	baseURL         string
	chatURL         string
}

// New creates a new client to talk to the Ollama API.
//
// ProviderOptionRemote defaults to "http://localhost:11434".
//
// Ollama doesn't have any mean of authentication so ProviderOptionAPIKey is not supported.
//
// To use multiple models, create multiple clients.
// Use one of the model from https://ollama.com/library
//
// Automatic model selection via ModelCheap, ModelGood, ModelSOTA is using hardcoded models. Before using an
// hardcoded model ID, it will ask ollama to determine if a model is already loaded and it will use that
// instead.
func New(ctx context.Context, opts ...genai.ProviderOption) (*Client, error) {
	var baseURL, model string
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
		case genai.ProviderOptionRemote:
			baseURL = string(v)
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
	if baseURL == "" {
		baseURL = "http://localhost:11434"
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
		impl: base.ProviderBase[*ErrorResponse]{
			Lenient: internal.BeLenient,
			Client: http.Client{
				Transport: &roundtrippers.RequestID{Transport: t},
			},
		},
		preloadedModels: preloadedModels,
		baseURL:         baseURL,
		chatURL:         baseURL + "/api/chat",
	}
	switch model {
	case "":
	case string(genai.ModelCheap), string(genai.ModelGood), string(genai.ModelSOTA):
		c.impl.Model = c.selectBestTextModel(ctx, model)
		c.impl.OutputModalities = mod
	default:
		c.impl.Model = model
		c.impl.OutputModalities = mod
	}
	return c, nil
}

// selectBestTextModel selects the most appropriate model based on the preference (cheap, good, or SOTA).
//
// We may want to make this function overridable in the future by the client since this is going to break one
// day or another.
func (c *Client) selectBestTextModel(ctx context.Context, preference string) string {
	// There's no way to list what's the current best models and no way to list the models in the library:
	// https://github.com/ollama/ollama/issues/8241

	// Figure out the model loaded if any. Ignore the error.
	m, _ := c.ListModels(ctx)
	if len(m) > 0 {
		return m[0].GetID()
	}
	// Hard code some popular models, it's more useful than failing hard. The model is not immediately pulled,
	// it will be pulled upon first use.
	switch preference {
	case string(genai.ModelCheap):
		return "gemma4:e2b"
	case string(genai.ModelSOTA):
		return "qwen3.5:2b"
	case string(genai.ModelGood), "":
		return "qwen3.5:2b"
	default:
		return "qwen3.5:2b"
	}
}

// Name implements genai.Provider.
//
// It returns the name of the provider.
func (c *Client) Name() string {
	return "ollama"
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
	res := genai.Result{}
	in := ChatRequest{}
	if err := in.Init(msgs, c.impl.Model, opts...); err != nil {
		return res, err
	}
	var out ChatResponse
	if err := c.GenSyncRaw(ctx, &in, &out); err != nil {
		return res, err
	}
	res, err := out.ToResult()
	if err != nil {
		return res, err
	}
	if err = res.Validate(); err != nil {
		return res, &internal.BadError{Err: err}
	}
	return res, nil
}

// GenSyncRaw provides access to the raw API.
func (c *Client) GenSyncRaw(ctx context.Context, in *ChatRequest, out *ChatResponse) error {
	if err := c.Validate(); err != nil {
		return err
	}
	in.Stream = false
	err := c.impl.DoRequest(ctx, "POST", c.chatURL, in, out)
	if err != nil {
		// TODO: Cheezy.
		if strings.Contains(err.Error(), "not found") {
			if err := c.PullModel(ctx, c.impl.Model); err != nil {
				return err
			}
			// Retry.
			err = c.impl.DoRequest(ctx, "POST", c.chatURL, in, out)
		}
	}
	return err
}

// GenStream implements genai.Provider.
func (c *Client) GenStream(ctx context.Context, msgs genai.Messages, opts ...genai.GenOption) (iter.Seq[genai.Reply], func() (genai.Result, error)) {
	res := genai.Result{}
	var finalErr error

	fnFragments := func(yield func(genai.Reply) bool) {
		in := ChatRequest{}
		if err := in.Init(msgs, c.impl.Model, opts...); err != nil {
			finalErr = err
			return
		}
		chunks, finish1 := c.GenStreamRaw(ctx, &in)
		fragments, finish2 := ProcessStream(chunks)
		for f := range fragments {
			if f.IsZero() {
				continue
			}
			if err := f.Validate(); err != nil {
				// Catch provider implementation bugs.
				finalErr = &internal.BadError{Err: err}
				break
			}
			if err := res.Accumulate(&f); err != nil {
				finalErr = &internal.BadError{Err: err}
				return
			}
			if !yield(f) {
				break
			}
		}
		if err := finish1(); finalErr == nil {
			finalErr = err
		}
		var err error
		res.Usage, res.Logprobs, err = finish2()
		if finalErr == nil {
			finalErr = err
		}
	}
	fnFinish := func() (genai.Result, error) {
		if res.Usage.FinishReason == genai.FinishedStop && slices.ContainsFunc(res.Replies, func(r genai.Reply) bool { return !r.ToolCall.IsZero() }) {
			// Lie for the benefit of everyone.
			res.Usage.FinishReason = genai.FinishedToolCalls
		}
		return res, finalErr
	}
	return fnFragments, fnFinish
}

// GenStreamRaw provides access to the raw API.
func (c *Client) GenStreamRaw(ctx context.Context, in *ChatRequest) (iter.Seq[ChatStreamChunkResponse], func() error) {
	var finalError error
	finish := func() error {
		return finalError
	}
	if finalError = c.Validate(); finalError != nil {
		finalError = &internal.BadError{Err: finalError}
		return yieldNothing[ChatStreamChunkResponse], finish
	}
	in.Stream = true
	// Try first, if it immediately errors out requesting to pull, pull then try again.
	resp, err1 := c.impl.JSONRequest(ctx, "POST", c.chatURL, in)
	if err1 != nil {
		finalError = &internal.BadError{Err: fmt.Errorf("failed to get server response: %w", err1)}
		return yieldNothing[ChatStreamChunkResponse], finish
	}

	// Process the stream in a separate goroutine to make sure that when the client iterate, there is already a
	// packet waiting for it. This reduces the overall latency.
	out := make(chan ChatStreamChunkResponse, 16)
	eg := errgroup.Group{}
	eg.Go(func() error {
		defer close(out)
		// Ollama doesn't use SSE.
		err2 := processJSONStream(resp.Body, out, c.impl.Lenient)
		_ = resp.Body.Close()
		if err2 == nil || !strings.Contains(err2.Error(), "not found") {
			return err2
		}
		// Model was not present. Try to pull then rerun again.
		if err2 = c.PullModel(ctx, c.impl.Model); err2 != nil {
			return &internal.BadError{Err: err2}
		}
		// Try a second time now that the model was pulled successfully.
		if resp, err2 = c.impl.JSONRequest(ctx, "POST", c.chatURL, in); err2 != nil {
			return &internal.BadError{Err: fmt.Errorf("failed to get server response: %w", err2)}
		}
		defer func() { _ = resp.Body.Close() }()
		if resp.StatusCode != http.StatusOK {
			return c.impl.DecodeError(c.chatURL, resp)
		}
		// Ollama doesn't use SSE.
		return processJSONStream(resp.Body, out, c.impl.Lenient)
	})

	return func(yield func(ChatStreamChunkResponse) bool) {
		for pkt := range out {
			if !yield(pkt) {
				break
			}
		}
		// Drain remaining messages to unblock the producer goroutine so
		// eg.Wait() doesn't deadlock.
		for range out {
		}
	}, eg.Wait
}

// ListModels implements genai.Provider.
func (c *Client) ListModels(ctx context.Context) ([]genai.Model, error) {
	if c.preloadedModels != nil {
		return c.preloadedModels, nil
	}
	// https://github.com/ollama/ollama/blob/main/docs/api.md#list-local-models
	var resp ModelsResponse
	if err := c.impl.DoRequest(ctx, "GET", c.baseURL+"/api/tags", nil, &resp); err != nil {
		return nil, err
	}
	return resp.ToModels(), nil
}

// PullModel is the equivalent of "ollama pull".
//
// Files are cached under $HOME/.ollama/models/manifests/registry.ollama.ai/library/ or $OLLAMA_MODELS.
func (c *Client) PullModel(ctx context.Context, model string) error {
	in := pullModelRequest{Model: model}
	// TODO: Stream updates instead of hanging for several minutes.
	out := pullModelResponse{}
	if err := c.impl.DoRequest(ctx, "POST", c.baseURL+"/api/pull", &in, &out); err != nil {
		return fmt.Errorf("pull failed: %w", err)
	} else if out.Status != "success" {
		return fmt.Errorf("pull failed: %s", out.Status)
	}
	return nil
}

// Version returns the Ollama server version.
func (c *Client) Version(ctx context.Context) (string, error) {
	v := Version{}
	if err := c.impl.DoRequest(ctx, "GET", c.baseURL+"/api/version", nil, &v); err != nil {
		return v.Version, fmt.Errorf("failed to get version: %w", err)
	}
	return v.Version, nil
}

// Ping checks that the Ollama server is reachable.
func (c *Client) Ping(ctx context.Context) error {
	_, err := c.Version(ctx)
	return err
}

// Validate returns an error if the client is not properly configured.
func (c *Client) Validate() error {
	if c.impl.Model == "" {
		return errors.New("a model is required")
	}
	return nil
}

// processJSONStream processes a \n separated JSON stream. This is different from other backends which use
// SSE.
func processJSONStream(body io.Reader, out chan<- ChatStreamChunkResponse, lenient bool) error {
	for r := bufio.NewReader(body); ; {
		line, err := r.ReadBytes('\n')
		if line = bytes.TrimSpace(line); err == io.EOF {
			if len(line) == 0 {
				return nil
			}
		} else if err != nil {
			return &internal.BadError{Err: fmt.Errorf("failed to get server response: %w", err)}
		}
		if len(line) == 0 {
			continue
		}
		d := json.NewDecoder(bytes.NewReader(line))
		if !lenient {
			d.DisallowUnknownFields()
		}
		d.UseNumber()
		msg := ChatStreamChunkResponse{}
		if err := d.Decode(&msg); err != nil {
			d := json.NewDecoder(bytes.NewReader(line))
			if !lenient {
				d.DisallowUnknownFields()
			}
			d.UseNumber()
			er := ErrorResponse{}
			if err := d.Decode(&er); err != nil {
				return &internal.BadError{Err: fmt.Errorf("failed to decode server response %q: %w", string(line), err)}
			}
			return &er
		}
		out <- msg
	}
}

// ProcessStream converts the raw packets from the streaming API into Reply fragments.
func ProcessStream(chunks iter.Seq[ChatStreamChunkResponse]) (iter.Seq[genai.Reply], func() (genai.Usage, [][]genai.Logprob, error)) {
	var finalErr error
	u := genai.Usage{}
	var l [][]genai.Logprob

	return func(yield func(genai.Reply) bool) {
			for pkt := range chunks {
				if pkt.EvalCount != 0 {
					u.InputTokens = pkt.PromptEvalCount
					u.OutputTokens = pkt.EvalCount
					u.FinishReason = pkt.DoneReason.ToFinishReason()
				}
				l = append(l, ToGenaiLogprobs(pkt.Logprobs)...)
				switch role := pkt.Message.Role; role {
				case "", "assistant":
				default:
					finalErr = &internal.BadError{Err: fmt.Errorf("unexpected role %q", role)}
					return
				}
				if pkt.Message.Thinking != "" {
					if !yield(genai.Reply{Reasoning: pkt.Message.Thinking}) {
						return
					}
				}
				for i := range pkt.Message.ToolCalls {
					f := genai.Reply{}
					if err := pkt.Message.ToolCalls[i].To(&f.ToolCall); err != nil {
						finalErr = &internal.BadError{Err: err}
						return
					}
					if !yield(f) {
						return
					}
				}
				if pkt.Message.Content != "" {
					if !yield(genai.Reply{Text: pkt.Message.Content}) {
						return
					}
				}
			}
		}, func() (genai.Usage, [][]genai.Logprob, error) {
			return u, l, finalErr
		}
}

func yieldNothing[T any](yield func(T) bool) {
}

var _ genai.Provider = &Client{}
