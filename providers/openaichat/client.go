// Copyright 2024 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Package openaichat implements a client for the OpenAI Chat Completion API.
//
// It is described at https://platform.openai.com/docs/api-reference/
package openaichat

// See official client at https://github.com/openai/openai-go

import (
	"bytes"
	"context"
	_ "embed"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"iter"
	"net/http"
	"net/url"
	"os"
	"strings"
	"time"

	"github.com/maruel/genai"
	"github.com/maruel/genai/base"
	"github.com/maruel/genai/internal"
	"github.com/maruel/genai/internal/bb"
	"github.com/maruel/genai/providers/openaibase"
	"github.com/maruel/genai/scoreboard"
	"github.com/maruel/roundtrippers"
)

//go:embed scoreboard.json
var scoreboardJSON []byte

// Scoreboard for OpenAI.
func Scoreboard() scoreboard.Score {
	var s scoreboard.Score
	d := json.NewDecoder(bytes.NewReader(scoreboardJSON))
	d.DisallowUnknownFields()
	if err := d.Decode(&s); err != nil {
		panic(fmt.Errorf("failed to unmarshal scoreboard.json: %w", err))
	}
	return s
}

// GenOptionText defines OpenAI specific options.
type GenOptionText struct {
	// ReasoningEffort is the amount of effort (number of tokens) the LLM can use to think about the answer.
	//
	// When unspecified, defaults to medium.
	ReasoningEffort ReasoningEffort
	// ServiceTier specify the priority.
	ServiceTier ServiceTier
}

// Validate implements genai.Validatable.
func (o *GenOptionText) Validate() error {
	if err := o.ReasoningEffort.Validate(); err != nil {
		return err
	}
	return o.ServiceTier.Validate()
}

// GenOptionAudio specifies audio generation options for audio models like gpt-audio.
//
// See https://platform.openai.com/docs/guides/audio
type GenOptionAudio struct {
	// Voice is the voice to use for speech synthesis.
	//
	// Supported voices: "alloy", "ash", "ballad", "coral", "echo", "fable", "nova", "onyx", "sage", "shimmer".
	// When empty, defaults to "alloy".
	Voice string
	// Format is the output audio format.
	//
	// Supported formats: "mp3", "wav", "flac", "opus", "pcm16", "aac".
	// When empty, defaults to "mp3".
	Format string
}

// Validate implements genai.Validatable.
func (o *GenOptionAudio) Validate() error {
	return nil
}

//

// In May 2025, OpenAI started pushing for Response API. They say it's the only way to keep reasoning items.
// It's interesting because Anthropic did that with the old API but OpenAI can't. Shrug.
// https://cookbook.openai.com/examples/responses_api/reasoning_items
// https://platform.openai.com/docs/api-reference/responses/create
// TODO: Switch over.

// Client implements genai.Provider.
type Client struct {
	base.NotImplemented
	impl   base.Provider[*ErrorResponse, *ChatRequest, *ChatResponse, ChatStreamChunkResponse]
	shared openaibase.Client
}

// New creates a new client to talk to the OpenAI platform API.
//
// If ProviderOptionAPIKey is not provided, it tries to load it from the OPENAI_API_KEY environment variable.
// If none is found, it will still return a client coupled with an base.ErrAPIKeyRequired error.
// Get your API key at https://platform.openai.com/settings/organization/api-keys
//
// To use multiple models, create multiple clients.
// Use one of the model from https://platform.openai.com/docs/models
//
// # Documents
//
// OpenAI supports many types of documents, listed at
// https://platform.openai.com/docs/assistants/tools/file-search#supported-files
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
	const apiKeyURL = "https://platform.openai.com/settings/organization/api-keys"
	var err error
	if apiKey == "" {
		if apiKey = os.Getenv("OPENAI_API_KEY"); apiKey == "" {
			err = &base.ErrAPIKeyRequired{EnvVar: "OPENAI_API_KEY", URL: apiKeyURL}
		}
	}
	switch len(modalities) {
	case 0:
		// Auto-detect below.
	case 1:
		switch modalities[0] {
		case genai.ModalityAudio, genai.ModalityImage, genai.ModalityText, genai.ModalityVideo:
		case genai.ModalityDocument:
			return nil, fmt.Errorf("unexpected option Modalities %s, only audio, image or text are supported", modalities)
		default:
			return nil, fmt.Errorf("unexpected option Modalities %s, only audio, image or text are supported", modalities)
		}
	default:
		return nil, fmt.Errorf("unexpected option Modalities %s, only audio, image or text are supported", modalities)
	}
	t := base.DefaultTransport
	if wrapper != nil {
		t = wrapper(t)
	}
	const baseURL = "https://api.openai.com/v1"
	c := &Client{
		impl: base.Provider[*ErrorResponse, *ChatRequest, *ChatResponse, ChatStreamChunkResponse]{
			GenSyncURL:      baseURL + "/chat/completions",
			ProcessStream:   ProcessStream,
			PreloadedModels: preloadedModels,
			ProcessHeaders:  openaibase.ProcessHeaders,
			ProviderBase: base.ProviderBase[*ErrorResponse]{
				// OpenAI error message prints the api key URL already.
				APIKeyURL: "",
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
	c.shared = openaibase.Client{
		Impl:            &c.impl.ProviderBase,
		BaseURL:         baseURL,
		PreloadedModels: preloadedModels,
	}
	if err == nil {
		switch model {
		case "":
		case string(genai.ModelCheap), string(genai.ModelGood), string(genai.ModelSOTA):
			var mod genai.Modality
			switch len(modalities) {
			case 0:
				mod = genai.ModalityText
			case 1:
				mod = modalities[0]
			default:
				// TODO: Maybe it's possible, need to double check.
				return nil, fmt.Errorf("can't use model %s with option Modalities %s", model, modalities)
			}
			switch mod {
			case genai.ModalityText:
				if c.impl.Model, err = c.shared.SelectBestTextModel(ctx, model); err != nil {
					return nil, err
				}
				c.impl.OutputModalities = genai.Modalities{mod}
			case genai.ModalityImage:
				if c.impl.Model, err = c.shared.SelectBestImageModel(ctx, model); err != nil {
					return nil, err
				}
				c.impl.OutputModalities = genai.Modalities{mod}
			case genai.ModalityVideo:
				if c.impl.Model, err = c.shared.SelectBestVideoModel(ctx, model); err != nil {
					return nil, err
				}
				c.impl.OutputModalities = genai.Modalities{mod}
			case genai.ModalityAudio:
				if c.impl.Model, err = c.selectBestAudioModel(ctx, model); err != nil {
					return nil, err
				}
				c.impl.OutputModalities = genai.Modalities{mod}
			case genai.ModalityDocument:
				// TODO: Soon, because it's cool.
				return nil, fmt.Errorf("automatic model selection is not implemented yet for modality %s (send PR to add support)", modalities)
			default:
				// TODO: Soon, because it's cool.
				return nil, fmt.Errorf("automatic model selection is not implemented yet for modality %s (send PR to add support)", modalities)
			}
		default:
			c.impl.Model = model
			switch len(modalities) {
			case 0:
				c.impl.OutputModalities, err = c.shared.DetectModelModalities(ctx, model)
			case 1:
				c.impl.OutputModalities = modalities
			default:
				// TODO: Maybe it's possible, need to double check.
				return nil, fmt.Errorf("can't use model %s with option Modalities %s", model, modalities)
			}
		}
	}
	return c, err
}

// selectBestAudioModel selects the most appropriate audio model based on the preference (cheap, good, or SOTA).
//
// Audio models are identified by the "audio" in their name.
func (c *Client) selectBestAudioModel(ctx context.Context, preference string) (string, error) {
	mdls, err := c.shared.ListModels(ctx)
	if err != nil {
		return "", fmt.Errorf("failed to automatically select the model: %w", err)
	}
	cheap := preference == string(genai.ModelCheap)
	good := preference == string(genai.ModelGood) || preference == ""
	sota := preference == string(genai.ModelSOTA)
	selectedModel := ""
	for _, mdl := range mdls {
		m := mdl.(*Model)
		// Only consider models with "audio" in their name
		if !strings.Contains(m.ID, "audio") {
			continue
		}
		// Categorize based on model characteristics
		isGPT4O := strings.HasPrefix(m.ID, "gpt-4o-") && strings.Contains(m.ID, "audio")
		isGPT4OMini := isGPT4O && strings.Contains(m.ID, "mini")
		isMini := strings.Contains(m.ID, "mini")
		matches := false
		switch {
		case cheap:
			matches = isMini
		case good:
			matches = !isMini && !isGPT4O
		case sota:
			// SOTA: prefer gpt-4o-audio models (but not mini)
			matches = isGPT4O && !isGPT4OMini
		}
		if !matches {
			continue
		}
		// Select the best available model, preferring newer versions lexicographically
		if selectedModel == "" || m.ID > selectedModel {
			selectedModel = m.ID
		}
	}
	if selectedModel == "" {
		return "", errors.New("failed to find an audio model automatically")
	}
	return selectedModel, nil
}

// Name implements genai.Provider.
//
// It returns the name of the provider.
func (c *Client) Name() string {
	return "openaichat"
}

// GenSyncRaw provides access to the raw API.
func (c *Client) GenSyncRaw(ctx context.Context, in *ChatRequest, out *ChatResponse) error {
	out.audioFormat = in.Audio.Format
	return c.impl.GenSyncRaw(ctx, in, out)
}

// GenStreamRaw provides access to the raw API.
func (c *Client) GenStreamRaw(ctx context.Context, in *ChatRequest) (iter.Seq[ChatStreamChunkResponse], func() error) {
	return c.impl.GenStreamRaw(ctx, in)
}

// GenAsync implements genai.ProviderGenAsync.
//
// It requests the providers' batch API and returns the job ID. It can take up to 24 hours to complete.
func (c *Client) GenAsync(ctx context.Context, msgs genai.Messages, opts ...genai.GenOption) (genai.Job, error) {
	fileID, err := c.CacheAddRequest(ctx, msgs, "TODO", "batch.json", 24*time.Hour, opts...)
	if err != nil {
		return "", err
	}
	b := BatchRequest{CompletionWindow: "24h", Endpoint: "/v1/chat/completions", InputFileID: fileID}
	resp, err := c.GenAsyncRaw(ctx, b)
	if len(resp.Errors.Data) != 0 {
		var errs []error
		for _, d := range resp.Errors.Data {
			errs = append(errs, fmt.Errorf("batch error on line %d: %s (%s)", d.Line, d.Message, d.Code))
		}
		err = errors.Join(err, errors.Join(errs...))
	}
	return genai.Job(resp.ID), err
}

// GenAsyncRaw runs an asynchronous generation request.
func (c *Client) GenAsyncRaw(ctx context.Context, b BatchRequest) (Batch, error) {
	resp := Batch{}
	err := c.impl.DoRequest(ctx, "POST", c.shared.BaseURL+"/batches", &b, &resp)
	return resp, err
}

// PokeResult implements genai.ProviderGenAsync.
//
// It retrieves the result for a job ID.
func (c *Client) PokeResult(ctx context.Context, id genai.Job) (genai.Result, error) {
	res := genai.Result{}
	resp, err := c.PokeResultRaw(ctx, id)
	if len(resp.Errors.Data) != 0 {
		var errs []error
		for _, d := range resp.Errors.Data {
			errs = append(errs, fmt.Errorf("batch error on line %d: %s (%s)", d.Line, d.Message, d.Code))
		}
		err = errors.Join(err, errors.Join(errs...))
	}
	if resp.Status == "validating" || resp.Status == "in_progress" || resp.Status == "finalizing" {
		res.Usage.FinishReason = genai.Pending
	}
	if resp.OutputFileID != "" {
		f, err2 := c.shared.FileGet(ctx, resp.OutputFileID)
		if f != nil {
			defer func() { _ = f.Close() }()
			out := BatchRequestOutput{}
			d := json.NewDecoder(f)
			d.UseNumber()
			if !c.impl.Lenient {
				d.DisallowUnknownFields()
			}
			if err3 := d.Decode(&out); err3 != nil {
				return res, errors.Join(err, err2, err3)
			}
			var err4 error
			res, err4 = out.Response.Body.ToResult()
			if err4 == nil && out.Error.Message != "" {
				err4 = fmt.Errorf("error %s: %s", out.Error.Code, out.Error.Message)
			}
			err = errors.Join(err, err2, err4)
		} else {
			err = errors.Join(err, err2)
		}
	}
	// TODO: Delete the input and output files.
	return res, err
}

// PokeResultRaw polls an asynchronous generation request.
func (c *Client) PokeResultRaw(ctx context.Context, id genai.Job) (Batch, error) {
	out := Batch{}
	u := c.shared.BaseURL + "/batches/" + url.PathEscape(string(id))
	err := c.impl.DoRequest(ctx, "GET", u, nil, &out)
	return out, err
}

// Cancel cancels an in-progress batch. The batch will be in status cancelling for up to 10 minutes, before
// changing to cancelled, where it will have partial results (if any) available in the output file.
func (c *Client) Cancel(ctx context.Context, id genai.Job) error {
	_, err := c.CancelRaw(ctx, id)
	return err
}

// CancelRaw cancels a batch request.
func (c *Client) CancelRaw(ctx context.Context, id genai.Job) (Batch, error) {
	u := c.shared.BaseURL + "/batches/" + url.PathEscape(string(id)) + "/cancel"
	resp := Batch{}
	err := c.impl.DoRequest(ctx, "POST", u, nil, &resp)
	// TODO: Delete the file too.
	return resp, err
}

// CacheAddRequest adds a cache entry.
func (c *Client) CacheAddRequest(ctx context.Context, msgs genai.Messages, name, displayName string, ttl time.Duration, opts ...genai.GenOption) (string, error) {
	if err := c.impl.Validate(); err != nil {
		return "", err
	}
	// Upload the messages and options as a file.
	b := BatchRequestInput{CustomID: name, Method: "POST", URL: "/v1/chat/completions"}
	if err := b.Body.Init(msgs, c.impl.Model, opts...); err != nil {
		return "", err
	}
	raw, err := json.Marshal(b)
	if err != nil {
		return "", err
	}
	return c.shared.FileAdd(ctx, displayName, bytes.NewReader(raw))
}

// CacheList lists cache entries.
func (c *Client) CacheList(ctx context.Context) ([]genai.CacheEntry, error) {
	l, err := c.shared.FilesListRaw(ctx)
	if err != nil {
		return nil, err
	}
	out := make([]genai.CacheEntry, len(l))
	for i := range l {
		out[i] = &l[i]
	}
	return out, nil
}

// CacheDelete deletes a cache entry.
func (c *Client) CacheDelete(ctx context.Context, name string) error {
	return c.shared.FileDel(ctx, name)
}

// FileAdd uploads a file. The TTL is one month.
func (c *Client) FileAdd(ctx context.Context, filename string, r io.ReadSeeker) (string, error) {
	return c.shared.FileAdd(ctx, filename, r)
}

// FileGet retrieves a file.
func (c *Client) FileGet(ctx context.Context, id string) (io.ReadCloser, error) {
	return c.shared.FileGet(ctx, id)
}

// FileDel deletes a file.
func (c *Client) FileDel(ctx context.Context, id string) error {
	return c.shared.FileDel(ctx, id)
}

// FilesListRaw lists files.
func (c *Client) FilesListRaw(ctx context.Context) ([]File, error) {
	return c.shared.FilesListRaw(ctx)
}

// ProcessStream converts the raw packets from the streaming API into Reply fragments.
func makeProcessStream(audioFormat string) func(iter.Seq[ChatStreamChunkResponse]) (iter.Seq[genai.Reply], func() (genai.Usage, [][]genai.Logprob, error)) {
	return func(chunks iter.Seq[ChatStreamChunkResponse]) (iter.Seq[genai.Reply], func() (genai.Usage, [][]genai.Logprob, error)) {
		var finalErr error
		u := genai.Usage{}
		var l [][]genai.Logprob
		var audioBuf string

		return func(yield func(genai.Reply) bool) {
				pendingToolCall := ToolCall{}
				for pkt := range chunks {
				if pkt.Usage.PromptTokens != 0 {
					u.InputTokens = pkt.Usage.PromptTokens
					u.InputCachedTokens = pkt.Usage.PromptTokensDetails.CachedTokens
					u.ReasoningTokens = pkt.Usage.CompletionTokensDetails.ReasoningTokens
					u.OutputTokens = pkt.Usage.CompletionTokens
					u.ServiceTier = pkt.ServiceTier
				}
				if len(pkt.Choices) != 1 {
					continue
				}
				l = append(l, pkt.Choices[0].Logprobs.To()...)
				if fr := pkt.Choices[0].FinishReason; fr != "" {
					u.FinishReason = fr.ToFinishReason()
				}
				switch role := pkt.Choices[0].Delta.Role; role {
				case "", "assistant":
				default:
					finalErr = &internal.BadError{Err: fmt.Errorf("unexpected role %q", role)}
					return
				}
				if len(pkt.Choices[0].Delta.ToolCalls) > 1 {
					finalErr = &internal.BadError{Err: fmt.Errorf("implement multiple tool calls: %#v", pkt)}
					return
				}
				if r := pkt.Choices[0].Delta.Refusal; r != "" {
					finalErr = &internal.BadError{Err: fmt.Errorf("refused: %q", r)}
					return
				}

				f := genai.Reply{}
				for _, a := range pkt.Choices[0].Delta.Annotations {
					f.Citation.StartIndex = a.URLCitation.StartIndex
					f.Citation.EndIndex = a.URLCitation.EndIndex
					f.Citation.Sources = []genai.CitationSource{{Type: genai.CitationWeb, URL: a.URLCitation.URL}}
					if !yield(f) {
						return
					}
					f = genai.Reply{}
				}

				f.Text = pkt.Choices[0].Delta.Content
				// gpt-audio streams transcript in delta.audio.transcript, not delta.content.
				if tr := pkt.Choices[0].Delta.Audio.Transcript; tr != "" {
					if f.Text == "" {
						f.Text = tr
					} else {
						f.Text += tr
					}
				}
				// Accumulate streaming audio data; yield when complete.
				if d := pkt.Choices[0].Delta.Audio.Data; d != "" {
					audioBuf += d
				}
				if audioBuf != "" && pkt.Choices[0].FinishReason != "" {
					audioData, err := base64.StdEncoding.DecodeString(audioBuf)
					if err != nil {
						finalErr = &internal.BadError{Err: fmt.Errorf("failed to decode streamed audio: %w", err)}
						return
					}
					audioBuf = ""
					fn := "audio.bin"
					if audioFormat != "" {
						fn = "audio." + audioFormat
					}
					// Yield the audio Doc; reset f so tool call processing can continue.
					if tr := pkt.Choices[0].Delta.Audio.Transcript; tr != "" {
						if !f.IsZero() {
							if !yield(f) {
								return
							}
							f = genai.Reply{}
						}
						if !yield(genai.Reply{Text: tr}) {
							return
						}
					}
					af := genai.Reply{Doc: genai.Doc{Filename: fn, Src: &bb.BytesBuffer{D: audioData}}}
					if !f.IsZero() {
						if !yield(f) {
							return
						}
					}
					if !yield(af) {
						return
					}
					f = genai.Reply{}
				}
				// OpenAI streams the arguments. Buffer the arguments to send the fragment as a whole tool call.
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
			// Flush pending tool call at end of stream.
			if pendingToolCall.ID != "" {
				tf := genai.Reply{}
				pendingToolCall.To(&tf.ToolCall)
				if !yield(tf) {
					return
				}
			}
			// Flush accumulated audio at end of stream (gpt-audio doesn't set finish_reason).
			if audioBuf != "" {
				audioData, err := base64.StdEncoding.DecodeString(audioBuf)
				if err != nil {
					finalErr = &internal.BadError{Err: fmt.Errorf("failed to decode streamed audio: %w", err)}
					return
				}
				fn := "audio.bin"
				if audioFormat != "" {
					fn = "audio." + audioFormat
				}
				af := genai.Reply{Doc: genai.Doc{Filename: fn, Src: &bb.BytesBuffer{D: audioData}}}
				if !yield(af) {
					return
				}
			}
		}, func() (genai.Usage, [][]genai.Logprob, error) {
			return u, l, finalErr
		}
	}
}

// ProcessStream is the default stream processor (no audio format).
func ProcessStream(chunks iter.Seq[ChatStreamChunkResponse]) (iter.Seq[genai.Reply], func() (genai.Usage, [][]genai.Logprob, error)) {
	return makeProcessStream("")(chunks)
}

// Capabilities implements genai.Provider.
func (c *Client) Capabilities() genai.ProviderCapabilities {
	return genai.ProviderCapabilities{
		GenAsync: true,
		Caching:  true,
	}
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

// ListModels implements genai.Provider.
func (c *Client) ListModels(ctx context.Context) ([]genai.Model, error) {
	return c.shared.ListModels(ctx)
}

// GenSync implements genai.Provider.
func (c *Client) GenSync(ctx context.Context, msgs genai.Messages, opts ...genai.GenOption) (genai.Result, error) {
	if c.shared.IsImage() || c.shared.IsVideo() {
		if len(msgs) != 1 {
			return genai.Result{}, errors.New("must pass exactly one Message")
		}
		return c.shared.GenDoc(ctx, &msgs[0], opts...)
	}
	// Build the request ourselves so GenSyncRaw can track audioFormat on the response.
	in := &ChatRequest{}
	if err := in.Init(msgs, c.impl.Model, opts...); err != nil {
		return genai.Result{}, err
	}
	out := &ChatResponse{}
	if err := c.GenSyncRaw(ctx, in, out); err != nil {
		return genai.Result{}, err
	}
	res, err := out.ToResult()
	if err != nil {
		return res, err
	}
	if err := res.Validate(); err != nil {
		return res, &internal.BadError{Err: err}
	}
	if lastResp := c.impl.LastResponseHeaders(); lastResp != nil {
		res.Usage.Limits = openaibase.ProcessHeaders(lastResp)
	}
	return res, nil
}

// GenStream implements genai.Provider.
func (c *Client) GenStream(ctx context.Context, msgs genai.Messages, opts ...genai.GenOption) (iter.Seq[genai.Reply], func() (genai.Result, error)) {
	if c.shared.IsImage() || c.shared.IsVideo() {
		return base.SimulateStream(ctx, c, msgs, opts...)
	}
	// Build the request ourselves so makeProcessStream can use the audio format.
	in := &ChatRequest{}
	if err := in.Init(msgs, c.impl.Model, opts...); err != nil {
		return func(yield func(genai.Reply) bool) {}, func() (genai.Result, error) { return genai.Result{}, err }
	}
	// Streaming only supports pcm16 audio format.
	if in.Audio.Format == "mp3" {
		in.Audio.Format = "pcm16"
	}

	res := genai.Result{}
	var finalErr error
	fnFragments := func(yield func(genai.Reply) bool) {
		chunks, finish := c.GenStreamRaw(ctx, in)
		// Capture headers immediately after the HTTP call, before iterating.
		lastResp := c.impl.LastResponseHeaders()
		fragments, finish2 := makeProcessStream(in.Audio.Format)(chunks)
		sent := false
		for f := range fragments {
			if f.IsZero() {
				continue
			}
			if err := f.Validate(); err != nil {
				finalErr = &internal.BadError{Err: err}
				break
			}
			if err := res.Accumulate(&f); err != nil {
				finalErr = &internal.BadError{Err: err}
				break
			}
			sent = true
			if !yield(f) {
				break
			}
		}
		if err := finish(); finalErr == nil {
			finalErr = err
		}
		var err error
		res.Usage, res.Logprobs, err = finish2()
		if finalErr == nil {
			finalErr = err
		}
		if !sent && finalErr == nil {
			finalErr = errors.New("model sent no reply")
		}
		if lastResp != nil {
			res.Usage.Limits = openaibase.ProcessHeaders(lastResp)
		}
	}
	return fnFragments, func() (genai.Result, error) {
		return res, finalErr
	}
}

var _ genai.Provider = &Client{}
