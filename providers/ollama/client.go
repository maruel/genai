// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Package ollama implements a client for the Ollama API.
//
// It is described at https://github.com/ollama/ollama/blob/main/docs/api.md
// and https://pkg.go.dev/github.com/ollama/ollama/api
//
// Decision models, including Clef Flash, use [Client.SystemOne] through /v1/systemone
// with Ollama v0.35.1 or later. See https://docs.ollama.com/api/systemone.
//
// # Embeddings
//
// [Client.Embed] embeds text batches with an explicitly configured embedding model.
// Install the model first with [Client.PullModel]. Dimensions defaults to the
// full vector size; compatible models support a requested size. Include any
// model-required query or document prefixes in text. [Client.EmbedRaw] exposes
// explicit model and dimension controls.
// Preloaded model capabilities identify embedding output; use
// genai.ProviderOptionModalities when model metadata is unavailable.
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

	"github.com/maruel/roundtrippers"
	"golang.org/x/sync/errgroup"

	"github.com/maruel/genai"
	"github.com/maruel/genai/base"
	"github.com/maruel/genai/internal"
	"github.com/maruel/genai/scoreboard"
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
	if err := base.CheckDuplicateProviderOptions(opts); err != nil {
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
	if slices.ContainsFunc(modalities, func(m genai.Modality) bool {
		return m != genai.ModalityText && m != genai.ModalityDecision && m != genai.ModalityEmbedding
	}) {
		return nil, fmt.Errorf("unexpected option Modalities %s, only text, decision and embedding are supported", modalities)
	}
	if slices.Contains(modalities, genai.ModalityEmbedding) && (model == string(genai.ModelCheap) || model == string(genai.ModelGood) || model == string(genai.ModelSOTA)) {
		return nil, errors.New("embedding models must be selected explicitly")
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
	mod := genai.Modalities{modelOutputModality(&Model{Name: model})}
	switch model {
	case "":
	case string(genai.ModelCheap), string(genai.ModelGood), string(genai.ModelSOTA):
		var output genai.Modality
		c.impl.Model, output = c.selectBestModel(ctx, model, modalities)
		mod = genai.Modalities{output}
	default:
		c.impl.Model = model
		for _, m := range preloadedModels {
			if strings.TrimSuffix(m.GetID(), ":latest") == strings.TrimSuffix(model, ":latest") {
				mod = genai.Modalities{modelOutputModality(m)}
				break
			}
		}
	}
	if slices.Equal(modalities, genai.Modalities{genai.ModalityEmbedding}) {
		mod = modalities
	}
	if model != "" {
		if len(modalities) != 0 && !slices.Equal(modalities, mod) {
			return nil, fmt.Errorf("unexpected option Modalities %s, model supports %s", modalities, mod)
		}
		c.impl.OutputModalities = mod
	} else if len(modalities) != 0 {
		c.impl.OutputModalities = modalities
	}
	return c, nil
}

// selectBestModel selects an installed model with the requested output modality or a popular default.
//
// We may want to make this function overridable in the future by the client since this is going to break one
// day or another.
func (c *Client) selectBestModel(ctx context.Context, preference string, modalities genai.Modalities) (string, genai.Modality) {
	// There's no way to list what's the current best models and no way to list the models in the library:
	// https://github.com/ollama/ollama/issues/8241

	// Figure out the model loaded if any. Ignore the error.
	m, _ := c.ListModels(ctx)
	for _, model := range m {
		mod := modelOutputModality(model)
		if mod != genai.ModalityEmbedding && (len(modalities) == 0 || slices.Equal(modalities, genai.Modalities{mod})) {
			return model.GetID(), mod
		}
	}
	// Hard code some popular models, it's more useful than failing hard. The model is not immediately pulled,
	// it will be pulled upon first use.
	if slices.Equal(modalities, genai.Modalities{genai.ModalityDecision}) {
		return "clef-flash:latest", genai.ModalityDecision
	}
	switch preference {
	case string(genai.ModelCheap):
		return "gemma4:e2b", genai.ModalityText
	case string(genai.ModelSOTA):
		return "qwen3.5:2b", genai.ModalityText
	case string(genai.ModelGood), "":
		return "qwen3.5:2b", genai.ModalityText
	default:
		return "qwen3.5:2b", genai.ModalityText
	}
}

func modelOutputModality(m genai.Model) genai.Modality {
	if native, ok := m.(*Model); ok {
		if slices.Contains(native.Capabilities, "embedding") && !slices.Contains(native.Capabilities, "completion") {
			return genai.ModalityEmbedding
		}
	}
	model := m.GetID()
	if strings.HasPrefix(model[strings.LastIndexByte(model, '/')+1:], "clef") {
		return genai.ModalityDecision
	}
	return genai.ModalityText
}

// Close implements io.Closer. It currently does nothing.
func (c *Client) Close() error {
	return nil
}

// Capabilities implements genai.Provider.
func (c *Client) Capabilities() genai.ProviderCapabilities {
	return genai.ProviderCapabilities{Embed: true, SystemOne: true}
}

// SystemOne answers typed questions using a local decision model.
// Inline images are encoded as base64; each document read is bounded to 10 MiB.
func (c *Client) SystemOne(ctx context.Context, in *genai.SystemOneRequest) (*genai.SystemOneResponse, error) {
	req := SystemOneRequest{Model: c.impl.Model}
	if err := req.From(in); err != nil {
		return nil, err
	}
	out := SystemOneResponse{}
	if err := c.SystemOneRaw(ctx, &req, &out); err != nil {
		return nil, err
	}
	res := &genai.SystemOneResponse{}
	if err := out.To(res); err != nil {
		return nil, err
	}
	if err := res.ValidateQuestions(&req.Questions); err != nil {
		return nil, &internal.BadError{Err: err}
	}
	return res, nil
}

// SystemOneRaw provides access to Ollama's native /v1/systemone endpoint.
func (c *Client) SystemOneRaw(ctx context.Context, in *SystemOneRequest, out *SystemOneResponse) error {
	if err := in.Validate(); err != nil {
		return err
	}
	err := c.impl.DoRequest(ctx, "POST", c.baseURL+"/v1/systemone", in, out)
	if e, ok := errors.AsType[*ErrorResponse](err); ok && strings.Contains(e.Error(), "not found") {
		if err := c.PullModel(ctx, in.Model); err != nil {
			return err
		}
		return c.impl.DoRequest(ctx, "POST", c.baseURL+"/v1/systemone", in, out)
	}
	return err
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

// Embed implements genai.Provider with the configured model.
// Document inputs return base.ErrNotSupported.
func (c *Client) Embed(ctx context.Context, in *genai.EmbeddingRequest) (*genai.EmbeddingResponse, error) {
	if in == nil {
		return nil, errors.New("embedding request is required")
	}
	if err := in.Validate(); err != nil {
		return nil, err
	}
	texts := make([]string, len(in.Inputs))
	for i := range in.Inputs {
		if !in.Inputs[i].Doc.IsZero() {
			return nil, fmt.Errorf("embedding input #%d: %w", i, &base.ErrNotSupported{Options: []string{"Request.Doc"}})
		}
		texts[i] = in.Inputs[i].Text
	}
	req := EmbeddingRequest{Model: c.ModelID(), Input: texts, Dimensions: in.Dimensions}
	var raw EmbeddingResponse
	if err := c.EmbedRaw(ctx, &req, &raw); err != nil {
		return nil, err
	}
	if raw.Object != "list" || len(raw.Data) != len(in.Inputs) {
		return nil, &internal.BadError{Err: fmt.Errorf("invalid embedding response: object %q, %d results for %d inputs", raw.Object, len(raw.Data), len(in.Inputs))}
	}
	out := &genai.EmbeddingResponse{Embeddings: make([][]float32, len(raw.Data)), Usage: genai.Usage{InputTokens: raw.Usage.PromptTokens, TotalTokens: raw.Usage.TotalTokens}}
	for i := range raw.Data {
		e := &raw.Data[i]
		if e.Object != "embedding" || e.Index < 0 || e.Index >= len(out.Embeddings) || out.Embeddings[e.Index] != nil {
			return nil, &internal.BadError{Err: fmt.Errorf("invalid embedding #%d: object %q, index %d", i, e.Object, e.Index)}
		}
		out.Embeddings[e.Index] = e.Embedding
	}
	if err := out.Validate(); err != nil {
		return nil, &internal.BadError{Err: fmt.Errorf("invalid embedding response: %w", err)}
	}
	if in.Dimensions != 0 && len(out.Embeddings[0]) != in.Dimensions {
		return nil, &internal.BadError{Err: fmt.Errorf("got %d embedding dimensions, want %d", len(out.Embeddings[0]), in.Dimensions)}
	}
	return out, nil
}

// EmbedRaw exposes pooled text embeddings through /v1/embeddings.
// The model must be installed; use PullModel to download it.
// The request is not modified. HTTP or decoding errors clear the response.
func (c *Client) EmbedRaw(ctx context.Context, in *EmbeddingRequest, out *EmbeddingResponse) error {
	if in == nil || out == nil {
		return errors.New("embedding request and response are required")
	}
	if err := in.Validate(); err != nil {
		return err
	}
	req := embeddingRequest{EmbeddingRequest: *in, EncodingFormat: "base64"}
	if err := c.impl.DoRequest(ctx, http.MethodPost, c.baseURL+"/v1/embeddings", &req, out); err != nil {
		*out = EmbeddingResponse{}
		return err
	}
	return nil
}
