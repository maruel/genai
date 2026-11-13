// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Package gemini implements a client for Google's Gemini API.
//
// Not to be confused with Google's Vertex AI.
//
// It is described at https://ai.google.dev/api/?lang=rest but the doc is weirdly organized.
//
// # Speech
//
// [Client.InteractionRaw] and [Client.InteractionStreamRaw] expose native
// speech configuration, annotations and lifecycle events. These APIs are
// independent of Live conversations; agent/tool and non-speech unions are omitted.
//
// [Client.ConnectLiveRaw] sends native Live setup and exposes complete server
// messages without transcript or turn-commit policy.
//
// # Embeddings
//
// [Client.Embed] embeds text batches with an explicitly configured embedding model.
// Dimensions defaults to the full vector size; compatible models support a
// requested size. Reduced-dimensional vectors can require normalization before
// similarity comparisons. Token usage is included when the model reports it.
// [Client.EmbedRaw] exposes task types, titles and
// native content inputs. Include any model-required task prefixes in text.
// Preloaded model metadata can identify embedding output without qualification;
// [Client.Scoreboard] reports measured support for exact model IDs.
package gemini

// See official client at https://github.com/google/generative-ai-go

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
	"math"
	"mime"
	"net/http"
	"net/url"
	"os"
	"reflect"
	"slices"
	"strconv"
	"strings"
	"time"

	"github.com/maruel/genai/internal/realtime"

	"github.com/maruel/roundtrippers"

	"github.com/maruel/genai"
	"github.com/maruel/genai/base"
	"github.com/maruel/genai/internal"
	"github.com/maruel/genai/internal/bb"
	"github.com/maruel/genai/scoreboard"
)

//go:embed scoreboard.json
var scoreboardJSON []byte

// Scoreboard for Gemini.
//
// See the following multi-modal sources:
//
// https://ai.google.dev/gemini-api/docs/image-understanding?hl=en&lang=rest#supported-formats
// https://ai.google.dev/gemini-api/docs/video-understanding?hl=en#supported-formats
// https://ai.google.dev/gemini-api/docs/audio?hl=en#supported-formats
// https://ai.google.dev/gemini-api/docs/document-processing?hl=en&lang=rest#technical-details
func Scoreboard() scoreboard.Score {
	var s scoreboard.Score
	d := json.NewDecoder(bytes.NewReader(scoreboardJSON))
	d.DisallowUnknownFields()
	if err := d.Decode(&s); err != nil {
		panic(fmt.Errorf("failed to unmarshal scoreboard.json: %w", err))
	}
	return s
}

// GenOption defines Gemini specific options.
type GenOption struct {
	// CodeExecution enables the code execution tool, allowing the model to generate and run Python code.
	//
	// https://ai.google.dev/gemini-api/docs/code-execution
	CodeExecution bool

	// URLContext enables the URL context tool, allowing the model to fetch and process web pages.
	//
	// https://ai.google.dev/gemini-api/docs/url-context
	URLContext bool

	// FileSearch enables the file search tool with the specified configuration.
	//
	// https://ai.google.dev/gemini-api/docs/file-search
	FileSearch *FileSearch
}

// Validate implements genai.Validatable.
func (o *GenOption) Validate() error {
	return nil
}

// functionResponseSchema is the fixed JSON Schema for the tool function response wrapper.
var functionResponseSchema = genai.JSONSchema(`{"type":"object","properties":{"response":{"type":"string"}},"required":["response"]}`)

// Client implements genai.Provider.
type Client struct {
	base.NotImplemented

	impl base.Provider[*ErrorResponse, *ChatRequest, *ChatResponse, ChatStreamChunkResponse]
}

// New creates a new client to talk to Google's Gemini platform API.
//
// If ProviderOptionAPIKey is not provided, it tries to load it from the GEMINI_API_KEY environment variable.
// If none is found, it will still return a client coupled with an base.ErrAPIKeyRequired error.
// Get your API key at https://ai.google.dev/gemini-api/docs/getting-started
//
// To use multiple models, create multiple clients.
// Use one of the model from https://ai.google.dev/gemini-api/docs/models/gemini
//
// See https://ai.google.dev/gemini-api/docs/file-prompting-strategies?hl=en
// for good ideas on how to prompt with images.
//
// Using large files requires a pinned model with caching support.
//
// Visit https://ai.google.dev/gemini-api/docs/pricing for up to date information.
//
// As of May 2025, price on Pro model increases when more than 200k input tokens are used.
// Cached input tokens are 25% of the price of new tokens.
func New(ctx context.Context, opts ...genai.ProviderOption) (*Client, error) {
	var apiKey, model string
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
		case genai.ProviderOptionBare:
			// No agent harness to strip.
		case genai.ProviderOptionTransportWrapper:
			wrapper = v
		default:
			return nil, fmt.Errorf("unsupported option type %T", opt)
		}
	}
	const apiKeyURL = "https://aistudio.google.com/apikey"
	var err error
	if apiKey == "" {
		if apiKey = os.Getenv("GEMINI_API_KEY"); apiKey == "" {
			err = &base.ErrAPIKeyRequired{EnvVar: "GEMINI_API_KEY", URL: apiKeyURL}
		}
	}
	switch len(modalities) {
	case 0:
		// Auto-detect below.
	case 1:
		switch modalities[0] {
		case genai.ModalityAudio, genai.ModalityEmbedding, genai.ModalityImage, genai.ModalityText, genai.ModalityVideo:
		case genai.ModalityDocument:
			return nil, fmt.Errorf("unexpected option Modalities %s, only audio, embedding, image, text, or video are supported", modalities)
		default:
			return nil, fmt.Errorf("unexpected option Modalities %s, only audio, embedding, image, text, or video are supported", modalities)
		}
	case 2:
		// The only combination supported is image + text.
		mods := slices.Clone(modalities)
		slices.Sort(mods)
		if !slices.Equal(mods, []genai.Modality{genai.ModalityImage, genai.ModalityText}) {
			return nil, fmt.Errorf("unexpected option Modalities %s, only image+text are supported when grouped together", mods)
		}
	default:
		return nil, fmt.Errorf("unexpected option Modalities %s, only audio, embedding, image, text, video, or image+text are supported", modalities)
	}
	// Google supports HTTP POST gzip compression!
	var t http.RoundTripper = &roundtrippers.PostCompressed{
		Transport: base.DefaultTransport,
		Encoding:  "gzip",
	}
	if wrapper != nil {
		t = wrapper(t)
	}
	// Eventually, use OAuth https://ai.google.dev/gemini-api/docs/oauth#curl
	c := &Client{
		impl: base.Provider[*ErrorResponse, *ChatRequest, *ChatResponse, ChatStreamChunkResponse]{
			ProcessStream:   ProcessStream,
			PreloadedModels: preloadedModels,
			LieToolCalls:    true,
			ProviderBase: base.ProviderBase[*ErrorResponse]{
				APIKeyURL: apiKeyURL,
				Lenient:   internal.BeLenient,
				Client: http.Client{
					Transport: &roundtrippers.Header{
						Header:    http.Header{"x-goog-api-key": {apiKey}},
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
				if c.impl.Model, err = c.selectBestTextModel(ctx, model); err != nil {
					return nil, err
				}
				c.impl.GenSyncURL = "https://generativelanguage.googleapis.com/v1beta/models/" + url.PathEscape(c.impl.Model) + ":generateContent"
				c.impl.GenStreamURL = "https://generativelanguage.googleapis.com/v1beta/models/" + url.PathEscape(c.impl.Model) + ":streamGenerateContent?alt=sse"
				c.impl.OutputModalities = genai.Modalities{mod}
			case genai.ModalityImage:
				if c.impl.Model, err = c.selectBestImageModel(ctx, model); err != nil {
					return nil, err
				}
				c.impl.OutputModalities = genai.Modalities{mod}
			case genai.ModalityVideo:
				if c.impl.Model, err = c.selectBestVideoModel(ctx, model); err != nil {
					return nil, err
				}
				c.impl.OutputModalities = genai.Modalities{mod}
			case genai.ModalityAudio:
				if c.impl.Model, err = c.selectBestAudioModel(ctx); err != nil {
					return nil, err
				}
				c.impl.OutputModalities = genai.Modalities{mod}
			case genai.ModalityDocument:
				// TODO: Implement document modality model selection.
				return nil, fmt.Errorf("automatic model selection is not implemented yet for modality %s (send PR to add support)", modalities)
			default:
				return nil, fmt.Errorf("automatic model selection is not implemented yet for modality %s (send PR to add support)", modalities)
			}
		default:
			c.impl.Model = model
			if len(modalities) == 0 {
				c.impl.OutputModalities, err = c.detectModelModalities(ctx, c.impl.Model)
			} else {
				c.impl.OutputModalities = modalities
			}
			c.impl.GenSyncURL = "https://generativelanguage.googleapis.com/v1beta/models/" + url.PathEscape(c.impl.Model) + ":generateContent"
			c.impl.GenStreamURL = "https://generativelanguage.googleapis.com/v1beta/models/" + url.PathEscape(c.impl.Model) + ":streamGenerateContent?alt=sse"
		}
	}
	return c, err
}

// Close implements io.Closer. It currently does nothing.
func (c *Client) Close() error {
	return nil
}

// Name implements genai.Provider.
//
// It returns the name of the provider.
func (c *Client) Name() string {
	return "gemini"
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

// ConnectRealtime opens a Gemini Live audio conversation using this client's model
// and HTTP authentication. The model must explicitly support Live; use a bare
// model ID, such as gemini-3.8-live. No model-list request occurs. The
// configuration defaults to server VAD, 16 kHz mono PCM input and 24 kHz mono PCM
// output. RealtimeOptionAudio.InputFormat selects another input rate; Gemini
// resamples it. Native-audio models output audio only; request transcripts separately.
//
// Server VAD detects turn ends and interrupts the current response. Manual
// detection disables it: send ActivityStart, audio chunks, then ActivityEnd.
// Manual turns start activity with the first Audio and end it when Commit or
// Generate is set. Gemini responds automatically, so Generate is advisory.
// Server turns send an audio stream end for Commit or Generate. Explicit cancellation and playback truncation are
// unsupported.
//
// Text with Generate=false appends passive context; Generate=true submits a
// completed user turn. Empty text generation is ignored. Standard models use
// blocking calls. Extended-thinking
// models require non-blocking calls, which may emit filler turns before the
// tool call. Return one RealtimeInputContent batch with ToolResults; the provider resumes
// automatically, so Generate is ignored for tool results. A batch that includes a provider-cancelled
// call is rejected in full with *genai.RealtimeToolCallCancelledError, which lists the
// cancelled calls. Cancelled tool IDs cannot be reused.
//
// RealtimeOption adds trusted proxy endpoints, idle ping timing, resumption
// handles and optional sliding-window compression. Interaction
// status, resumption and go-away observations are RealtimeEventProvider events
// with *InteractionStatus, *Resumption and *GoAway payloads; no replay occurs
// automatically. Usage snapshots cover one response. Frames and
// unfinished transcripts are bounded to 16 MiB, and at most 1024 tool
// correlations remain pending. A malformed recognized payload terminates the
// session. Keep Receive running to process ping replies.
//
// Protocol source:
// https://github.com/googleapis/go-genai/blob/9cf04deac32f2418781170b3cd41c09abf060564/live.go
func (c *Client) ConnectRealtime(ctx context.Context, opts ...genai.RealtimeOption) (genai.RealtimeSession, error) {
	setup := &LiveSetup{
		GenerationConfig:    LiveGenerationConfig{ResponseModalities: []Modality{ModalityAudio}},
		RealtimeInputConfig: LiveRealtimeInputConfig{ActivityHandling: LiveActivityInterrupts, TurnCoverage: LiveTurnOnlyActivity},
	}
	var inputFormat genai.AudioFormat
	var providerOpts []genai.RealtimeOption
	seen := make(map[reflect.Type]struct{}, len(opts))
	for _, opt := range opts {
		if opt == nil || (reflect.ValueOf(opt).Kind() == reflect.Pointer && reflect.ValueOf(opt).IsNil()) {
			return nil, errors.New("realtime option must not be nil")
		}
		t := reflect.TypeOf(opt)
		if _, ok := seen[t]; ok {
			return nil, fmt.Errorf("duplicate realtime option type %s", t)
		}
		seen[t] = struct{}{}
		switch o := opt.(type) {
		case genai.EffortLevel:
			if o == "" {
				continue
			}
			cfg, err := thinkingConfig(c.impl.Model, o)
			if err != nil {
				return nil, err
			}
			if setup.GenerationConfig.ThinkingConfig != nil {
				cfg.IncludeThoughts = setup.GenerationConfig.ThinkingConfig.IncludeThoughts
			}
			setup.GenerationConfig.ThinkingConfig = &cfg
		case genai.ReasoningOutput:
			if setup.GenerationConfig.ThinkingConfig == nil {
				setup.GenerationConfig.ThinkingConfig = &ThinkingConfig{}
			}
			setup.GenerationConfig.ThinkingConfig.IncludeThoughts = true
		case genai.RealtimeOptionInstructions:
			setup.SystemInstruction = &LiveContent{Parts: []LivePart{{Text: string(o)}}}
		case *genai.RealtimeOptionTools:
			if err := setup.Tools.Init(o.Tools, c.impl.Model); err != nil {
				return nil, err
			}
		case *genai.RealtimeOptionAudio:
			// Gemini Live resamples input at any rate declared in the blob MIME type:
			// https://aistudio.google.com/docs/live-api/capabilities
			inputFormat = o.InputFormat
			if o.Voice != "" {
				setup.GenerationConfig.SpeechConfig = &LiveSpeechConfig{VoiceConfig: LiveVoiceConfig{PrebuiltVoiceConfig: LivePrebuiltVoiceConfig{VoiceName: o.Voice}}}
			}
			if o.InputTranscription {
				setup.InputAudioTranscription = &LiveTranscriptionConfig{}
			}
			if o.OutputTranscription {
				setup.OutputAudioTranscription = &LiveTranscriptionConfig{}
			}
		case genai.Modalities:
			if len(o) > 1 || (len(o) == 1 && o[0] != genai.ModalityAudio) {
				return nil, fmt.Errorf("gemini live native-audio output: %w", &base.ErrNotSupported{Options: []string{"output modalities"}})
			}
		case genai.RealtimeTurnDetection:
			if o == genai.RealtimeTurnManual {
				setup.RealtimeInputConfig.AutomaticActivityDetection = &LiveAutomaticActivityDetection{Disabled: true}
			}
		default:
			providerOpts = append(providerOpts, opt)
		}
	}
	for _, opt := range opts {
		if err := opt.Validate(); err != nil {
			return nil, fmt.Errorf("realtime option %T: %w", opt, err)
		}
	}
	opts = providerOpts
	if c.impl.Model == "" {
		return nil, errors.New("gemini live requires an explicit model")
	}
	o := RealtimeOption{}
	for _, opt := range opts {
		v, ok := opt.(*RealtimeOption)
		if !ok {
			return nil, fmt.Errorf("unsupported Gemini Live option %T: %w", opt, &base.ErrNotSupported{Options: []string{fmt.Sprintf("%T", opt)}})
		}
		o = *v
	}
	if len(o.InputLanguages) != 0 && setup.InputAudioTranscription == nil {
		return nil, errors.New("gemini live input language hints require input transcription")
	}
	if o.Endpoint == "" {
		o.Endpoint = "wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent"
	}
	if o.PingInterval == 0 {
		o.PingInterval = 30 * time.Second
	}
	if o.PingTimeout == 0 {
		o.PingTimeout = 10 * time.Second
	}
	setup.Model = c.impl.Model
	if !strings.HasPrefix(setup.Model, "models/") {
		setup.Model = "models/" + setup.Model
	}
	if setup.InputAudioTranscription != nil {
		setup.InputAudioTranscription.LanguageCodes = o.InputLanguages
	}
	if strings.Contains(c.impl.Model, "extended-thinking") {
		if setup.GenerationConfig.ThinkingConfig == nil {
			setup.GenerationConfig.ThinkingConfig = &ThinkingConfig{}
		}
		if setup.GenerationConfig.ThinkingConfig.ThinkingLevel == "" && setup.GenerationConfig.ThinkingConfig.ThinkingBudget == nil {
			setup.GenerationConfig.ThinkingConfig.ThinkingLevel = ThinkingLevelLow
		}
	}
	if o.SessionResumption || o.ResumeHandle != "" {
		setup.SessionResumption = &LiveSessionResumption{Handle: o.ResumeHandle}
	}
	if o.Compression != nil {
		setup.ContextWindowCompression = &LiveCompression{TriggerTokens: o.Compression.TriggerTokens, SlidingWindow: LiveSlidingWindow{TargetTokens: o.Compression.TargetTokens}}
	}
	data, err := json.Marshal(LiveClientMessage{Setup: setup})
	if err != nil {
		return nil, fmt.Errorf("encode Gemini Live setup: %w", err)
	}
	ws, err := realtime.Dial(ctx, o.Endpoint, &c.impl.Client, maxMessageBytes, o.PingInterval, o.PingTimeout)
	if err != nil {
		return nil, err
	}
	if inputFormat == (genai.AudioFormat{}) {
		inputFormat = genai.AudioFormat{SampleRate: 16000}
	}
	info := genai.RealtimeSessionInfo{
		InputFormat:   inputFormat,
		OutputFormat:  genai.AudioFormat{SampleRate: 24000},
		TurnDetection: genai.RealtimeTurnServer,
	}
	if setup.RealtimeInputConfig.AutomaticActivityDetection != nil {
		info.TurnDetection = genai.RealtimeTurnManual
	}
	info.InputModalities = []genai.Modality{genai.ModalityAudio, genai.ModalityImage, genai.ModalityText}
	info.Capabilities.AudioCommit = true
	info.Capabilities.ToolResults = len(setup.Tools) > 0
	// contextUpdate is preview. It updates instructions and tools only; output
	// modality, token limits, voice and turn detection stay connection-scoped.
	info.Capabilities.Update = genai.RealtimeUpdateCapabilities{Instructions: true, Tools: true, ToolChoice: true}
	s := &RealtimeSession{ws: ws, info: info, model: c.impl.Model, liveTools: setup.Tools, toolChoice: genai.ToolCallAny, tools: make(map[string]realtimePendingTool), sendMu: make(chan struct{}, 1)}
	if err := ws.Write(ctx, data); err != nil {
		return nil, errors.Join(err, ws.Close())
	}
	data, err = ws.Read(ctx)
	if err != nil {
		return nil, errors.Join(err, ws.Close())
	}
	var msg LiveServerMessage
	if err := internal.UnmarshalJSON(data, &msg); err != nil {
		return nil, errors.Join(fmt.Errorf("decode Gemini Live setup response: %w", err), ws.Close())
	}
	if msg.Error != nil {
		return nil, errors.Join(&genai.RealtimeError{Code: strconv.Itoa(msg.Error.Code), Message: "gemini live setup rejected"}, ws.Close())
	}
	if msg.SetupComplete == nil {
		return nil, errors.Join(errors.New("gemini live expected setupComplete"), ws.Close())
	}
	s.info.ID = msg.SetupComplete.SessionID
	if err := s.translate(&msg); err != nil {
		return nil, errors.Join(err, ws.Close())
	}
	return s, nil
}

// GenSync implements genai.Provider.
func (c *Client) GenSync(ctx context.Context, msgs genai.Messages, opts ...genai.GenOption) (res genai.Result, err error) {
	if err := base.CheckDuplicateGenOptions(opts); err != nil {
		return genai.Result{}, err
	}
	if !slices.Contains(c.impl.OutputModalities, genai.ModalityText) {
		if len(msgs) != 1 {
			return genai.Result{}, errors.New("must pass exactly one Message")
		}
		return c.genDoc(ctx, &msgs[0], opts...)
	}
	// GenSync must be inlined because we need to call our GenSyncRaw.
	in := &ChatRequest{}
	if err := in.Init(msgs, c.impl.Model, opts...); err != nil {
		return res, err
	}
	out := &ChatResponse{}
	if err := c.GenSyncRaw(ctx, in, out); err != nil {
		return res, err
	}
	res, err = out.ToResult()
	if err != nil {
		return res, err
	}
	if err := res.Validate(); err != nil {
		// Catch provider implementation bugs.
		return res, err
	}

	lastResp := c.impl.LastResponseHeaders()
	if c.impl.ProcessHeaders != nil && lastResp != nil {
		res.Usage.Limits = c.impl.ProcessHeaders(lastResp)
	}
	return res, nil
}

// GenSyncRaw provides access to the raw API.
//
// It calls the Gemini API method generateContent.
func (c *Client) GenSyncRaw(ctx context.Context, in *ChatRequest, out *ChatResponse) error {
	if len(in.GenerationConfig.ResponseModalities) == 0 {
		in.GenerationConfig.ResponseModalities = make([]Modality, len(c.impl.OutputModalities))
		for i, m := range c.impl.OutputModalities {
			switch m {
			case genai.ModalityAudio:
				in.GenerationConfig.ResponseModalities[i] = ModalityAudio
			case genai.ModalityImage:
				in.GenerationConfig.ResponseModalities[i] = ModalityImage
			case genai.ModalityText:
				in.GenerationConfig.ResponseModalities[i] = ModalityText
			case genai.ModalityDocument, genai.ModalityVideo:
				return fmt.Errorf("unsupported modality %s", m)
			default:
				return fmt.Errorf("unsupported modality %s", m)
			}
		}
	}
	return c.impl.GenSyncRaw(ctx, in, out)
}

// GenStream implements genai.Provider.
func (c *Client) GenStream(ctx context.Context, msgs genai.Messages, opts ...genai.GenOption) (iter.Seq[genai.Reply], func() (genai.Result, error)) {
	if err := base.CheckDuplicateGenOptions(opts); err != nil {
		return func(yield func(genai.Reply) bool) {}, func() (genai.Result, error) { return genai.Result{}, err }
	}
	if !slices.Contains(c.impl.OutputModalities, genai.ModalityText) {
		return base.SimulateStream(ctx, c, msgs, opts...)
	}
	// GenStream must be inlined because we need to call our GenStreamRaw.
	res := genai.Result{}
	var finalErr error

	fnFragments := func(yield func(genai.Reply) bool) {
		in := &ChatRequest{}
		if err := in.Init(msgs, c.impl.Model, opts...); err != nil {
			finalErr = &internal.BadError{Err: err}
			return
		}
		// Generate parsed chunks from the raw JSON SSE stream.
		chunks, finish1 := c.GenStreamRaw(ctx, in)
		// Converts raw chunks into fragments.
		fragments, finish2 := c.impl.ProcessStream(chunks)
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
				break
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
		lastResp := c.impl.LastResponseHeaders()
		if c.impl.ProcessHeaders != nil && lastResp != nil {
			res.Usage.Limits = c.impl.ProcessHeaders(lastResp)
		}
		if c.impl.LieToolCalls && res.Usage.FinishReason == genai.FinishedStop {
			for i := range res.Replies {
				if !res.Replies[i].ToolCall.IsZero() {
					// Lie for the benefit of everyone.
					res.Usage.FinishReason = genai.FinishedToolCalls
					break
				}
			}
		}
	}
	fnFinish := func() (genai.Result, error) {
		if finalErr != nil {
			return res, finalErr
		}
		if err := res.Validate(); err != nil {
			// Catch provider implementation bugs.
			return res, err
		}
		return res, nil
	}
	return fnFragments, fnFinish
}

// GenStreamRaw provides access to the raw API.
//
// It calls the Gemini API method streamGenerateContent?alt=sse.
func (c *Client) GenStreamRaw(ctx context.Context, in *ChatRequest) (iter.Seq[ChatStreamChunkResponse], func() error) {
	if len(in.GenerationConfig.ResponseModalities) == 0 {
		in.GenerationConfig.ResponseModalities = make([]Modality, len(c.impl.OutputModalities))
		for i, m := range c.impl.OutputModalities {
			switch m {
			case genai.ModalityAudio:
				in.GenerationConfig.ResponseModalities[i] = ModalityAudio
			case genai.ModalityImage:
				in.GenerationConfig.ResponseModalities[i] = ModalityImage
			case genai.ModalityText:
				in.GenerationConfig.ResponseModalities[i] = ModalityText
			case genai.ModalityDocument, genai.ModalityVideo:
				return yieldNothing[ChatStreamChunkResponse], func() error {
					return &internal.BadError{Err: fmt.Errorf("unsupported modality %s", m)}
				}
			default:
				return yieldNothing[ChatStreamChunkResponse], func() error {
					return &internal.BadError{Err: fmt.Errorf("unsupported modality %s", m)}
				}
			}
		}
	}
	return c.impl.GenStreamRaw(ctx, in)
}

// CacheAddRequest caches the content for later use.
//
// Includes tools and systemprompt.
//
// Default time to live (ttl) is 1 hour.
//
// The minimum cacheable object is 4096 tokens.
//
// Visit https://ai.google.dev/gemini-api/docs/pricing for up to date information.
//
// As of May 2025, price on Pro model cache cost is 4.50$ per 1MTok/hour and Flash is 1$ per 1Mok/hour.
//
// At certain volumes, using cached tokens is lower cost than passing in the same corpus of tokens repeatedly.
// The cost for caching depends on the input token size and how long you want the tokens to persist.
func (c *Client) CacheAddRequest(ctx context.Context, msgs genai.Messages, name, displayName string, ttl time.Duration, opts ...genai.GenOption) (string, error) {
	// See https://ai.google.dev/gemini-api/docs/caching?hl=en&lang=rest#considerations
	// Useful when reusing the same large data multiple times to reduce token usage.
	// This requires a pinned model, with trailing -001.
	if err := c.impl.Validate(); err != nil {
		return "", err
	}
	in := CachedContent{}
	if err := in.Init(msgs, c.impl.Model, name, displayName, ttl, opts...); err != nil {
		return "", err
	}
	out := CachedContent{}
	u := "https://generativelanguage.googleapis.com/v1beta/cachedContents"
	if err := c.impl.DoRequest(ctx, "POST", u, &in, &out); err != nil {
		return "", err
	}
	name = strings.TrimPrefix(out.Name, "cachedContents/")
	return name, nil
}

// CacheExtend extends the TTL of a cached content entry.
func (c *Client) CacheExtend(ctx context.Context, name string, ttl time.Duration) error {
	// https://ai.google.dev/api/caching#method:-cachedcontents.patch
	u := "https://generativelanguage.googleapis.com/v1beta/cachedContents/" + url.PathEscape(name)
	// Model is required.
	in := CachedContent{Model: "models/" + c.impl.Model, TTL: Duration(ttl)}
	out := CachedContent{}
	return c.impl.DoRequest(ctx, "PATCH", u, &in, &out)
}

// CacheList lists cache entries.
func (c *Client) CacheList(ctx context.Context) ([]genai.CacheEntry, error) {
	l, err := c.CacheListRaw(ctx)
	if err != nil {
		return nil, err
	}
	out := make([]genai.CacheEntry, len(l))
	for i := range l {
		out[i] = &l[i]
	}
	return out, nil
}

// CacheListRaw retrieves the list of cached items.
func (c *Client) CacheListRaw(ctx context.Context) ([]CachedContent, error) {
	// https://ai.google.dev/api/caching#method:-cachedcontents.list
	// pageSize, pageToken
	var data struct {
		CachedContents []CachedContent `json:"cachedContents"`
		NextPageToken  string          `json:"nextPageToken"`
	}
	baseURL := "https://generativelanguage.googleapis.com/v1beta/cachedContents?pageSize=100"
	var out []CachedContent
	for ctx.Err() == nil {
		u := baseURL
		if data.NextPageToken != "" {
			u += "&pageToken=" + data.NextPageToken
		}
		if data.CachedContents != nil {
			data.CachedContents = data.CachedContents[:0]
		}
		data.NextPageToken = ""
		if err := c.impl.DoRequest(ctx, "GET", u, nil, &data); err != nil {
			return nil, err
		}
		for i := range data.CachedContents {
			data.CachedContents[i].Name = strings.TrimPrefix(data.CachedContents[i].Name, "cachedContents/")
		}
		out = append(out, data.CachedContents...)
		if len(data.CachedContents) == 0 || data.NextPageToken == "" {
			break
		}
	}
	return out, nil
}

// CacheGetRaw retrieves a cached content entry.
func (c *Client) CacheGetRaw(ctx context.Context, name string) (CachedContent, error) {
	// https://ai.google.dev/api/caching#method:-cachedcontents.get
	u := "https://generativelanguage.googleapis.com/v1beta/cachedContents/" + url.PathEscape(name)
	out := CachedContent{}
	err := c.impl.DoRequest(ctx, "GET", u, nil, &out)
	return out, err
}

// CacheDelete deletes a cached file.
func (c *Client) CacheDelete(ctx context.Context, name string) error {
	// https://ai.google.dev/api/caching#method:-cachedcontents.delete
	u := "https://generativelanguage.googleapis.com/v1beta/cachedContents/" + url.PathEscape(name)
	var out struct{}
	return c.impl.DoRequest(ctx, "DELETE", u, nil, &out)
}

// GenAsync implements genai.ProviderGenAsync.
//
// It requests the providers' asynchronous API and returns the job ID.
//
// GenAsync is only supported for models that have "predictLongRunning" reported in their
// Model.SupportedGenerationMethods.
//
// The resulting file is available for 48 hours. It requires the API key in the HTTP header to be fetched, so
// use the client's HTTP client.
func (c *Client) GenAsync(ctx context.Context, msgs genai.Messages, opts ...genai.GenOption) (genai.Job, error) {
	if err := base.CheckDuplicateGenOptions(opts); err != nil {
		return "", err
	}
	// GenAsync only works with video generation models (predictLongRunning endpoint).
	// Text models use generateContent which doesn't support async operations.
	if !slices.Contains(c.impl.OutputModalities, genai.ModalityVideo) {
		return "", &base.ErrNotSupported{}
	}
	if err := c.impl.Validate(); err != nil {
		return "", err
	}
	if len(msgs) != 1 {
		return "", errors.New("only one message can be passed as input")
	}
	req := ImageRequest{}
	if err := req.Init(&msgs[0], c.impl.Model, c.impl.OutputModalities, opts...); err != nil {
		return "", err
	}
	resp, err := c.PredictLongRunningRaw(ctx, &req)
	return genai.Job(resp.Name), err
}

// PokeResult implements genai.ProviderGenAsync.
//
// It retrieves the result for a job ID.
func (c *Client) PokeResult(ctx context.Context, id genai.Job) (genai.Result, error) {
	res := genai.Result{}
	op, err := c.PokeResultRaw(ctx, id)
	if err != nil {
		return res, err
	}
	if !op.Done {
		res.Usage.FinishReason = genai.Pending
		return res, nil
	}
	res.Usage.FinishReason = genai.FinishedStop
	for _, p := range op.Response.GenerateVideoResponse.GeneratedSamples {
		// This requires the Google API key to fetch!
		res.Replies = []genai.Reply{{Doc: genai.Doc{Filename: "content.mp4", URL: p.Video.URI}}}
	}
	return res, nil
}

// PredictRaw requests the providers' synchronous API to generate an image.
//
// The official documentation https://ai.google.dev/api/models?hl=en#method:-models.predict is not really
// helpful.
func (c *Client) PredictRaw(ctx context.Context, req *ImageRequest) (ImageResponse, error) {
	res := ImageResponse{}
	if err := c.impl.Validate(); err != nil {
		return res, err
	}
	// https://ai.google.dev/api/models?hl=en#method:-models.predict
	u := "https://generativelanguage.googleapis.com/v1beta/models/" + url.PathEscape(c.impl.Model) + ":predict"
	err := c.impl.DoRequest(ctx, "POST", u, &req, &res)
	return res, err
}

// PredictLongRunningRaw requests the providers' asynchronous API to generate a video.
//
// The official documentation https://ai.google.dev/api/models?hl=en#method:-models.predictlongrunning is not really
// helpful.
func (c *Client) PredictLongRunningRaw(ctx context.Context, req *ImageRequest) (Operation, error) {
	res := Operation{}
	if err := c.impl.Validate(); err != nil {
		return res, err
	}
	u := "https://generativelanguage.googleapis.com/v1beta/models/" + url.PathEscape(c.impl.Model) + ":predictLongRunning"
	err := c.impl.DoRequest(ctx, "POST", u, &req, &res)
	return res, err
}

// PokeResultRaw retrieves the result for a job ID if already available.
func (c *Client) PokeResultRaw(ctx context.Context, id genai.Job) (Operation, error) {
	res := Operation{}
	u := "https://generativelanguage.googleapis.com/v1beta/" + string(id)
	err := c.impl.DoRequest(ctx, "GET", u, nil, &res)
	if err != nil {
		return res, fmt.Errorf("failed to get job %q: %w", id, err)
	}
	return res, err
}

// ListModels implements genai.Provider.
func (c *Client) ListModels(ctx context.Context) ([]genai.Model, error) {
	if c.impl.PreloadedModels != nil {
		return c.impl.PreloadedModels, nil
	}
	// https://ai.google.dev/api/models?hl=en#method:-models.list
	var resp ModelsResponse
	if err := c.impl.DoRequest(ctx, "GET", "https://generativelanguage.googleapis.com/v1beta/models?pageSize=1000", nil, &resp); err != nil {
		return nil, err
	}
	return resp.ToModels(), nil
}

// ListVoicesRequest filters the Gemini voice catalog. Multiple
// values within a filter combine with OR; different filters combine with AND.
// Filters use case-insensitive exact matches, except Search, which matches
// substrings of display names and descriptions.
type ListVoicesRequest struct {
	Accent       []string
	Context      []string
	Gender       []VoiceGender
	LanguageCode []string
	Persona      []string
	Pitch        []VoicePitch
	RegionCode   []string
	Search       string
	Type         []VoiceType
	// PageSize defaults to 50. The server caps values above 1000.
	PageSize int
}

// ListVoices lazily iterates Google's voice catalog with all metadata, fetching
// subsequent pages as needed. Breaking iteration stops without fetching another
// page. A zero request uses the service's default filters and page size.
//
// https://ai.google.dev/gemini-api/docs/speech-generation#voices
func (c *Client) ListVoices(ctx context.Context, req *ListVoicesRequest) iter.Seq2[Voice, error] {
	return func(yield func(Voice, error) bool) {
		q := url.Values{}
		q["accent"] = req.Accent
		q["context"] = req.Context
		q["language_code"] = req.LanguageCode
		q["persona"] = req.Persona
		q["region_code"] = req.RegionCode
		for _, g := range req.Gender {
			q.Add("gender", string(g))
		}
		for _, p := range req.Pitch {
			q.Add("pitch", string(p))
		}
		for _, typ := range req.Type {
			q.Add("type", string(typ))
		}
		if req.Search != "" {
			q.Set("search", req.Search)
		}
		if req.PageSize != 0 {
			q.Set("page_size", strconv.Itoa(req.PageSize))
		}
		var resp ListVoicesResponse
		for {
			if err := ctx.Err(); err != nil {
				yield(Voice{}, err)
				return
			}
			u := "https://generativelanguage.googleapis.com/v1beta/voices"
			if query := q.Encode(); query != "" {
				u += "?" + query
			}
			clear(resp.Voices)
			resp.Voices = resp.Voices[:0]
			resp.NextPageToken = ""
			if err := c.impl.DoRequest(ctx, http.MethodGet, u, nil, &resp); err != nil {
				yield(Voice{}, err)
				return
			}
			for i := range resp.Voices {
				if err := ctx.Err(); err != nil {
					yield(Voice{}, err)
					return
				}
				if !yield(resp.Voices[i], nil) {
					return
				}
			}
			if resp.NextPageToken == "" {
				return
			}
			q.Set("page_token", resp.NextPageToken)
		}
	}
}

// GetModel returns the details for a single model.
//
// https://ai.google.dev/api/models?hl=en#method:-models.get
func (c *Client) GetModel(ctx context.Context, id string) (*Model, error) {
	var resp Model
	u := "https://generativelanguage.googleapis.com/v1beta/models/" + url.PathEscape(id)
	if err := c.impl.DoRequest(ctx, "GET", u, nil, &resp); err != nil {
		return nil, err
	}
	return &resp, nil
}

// CountTokens counts the number of tokens in the given messages.
//
// https://ai.google.dev/api/tokens#method:-models.counttokens
func (c *Client) CountTokens(ctx context.Context, msgs genai.Messages, opts ...genai.GenOption) (*CountTokensResponse, error) {
	var req ChatRequest
	if err := req.Init(msgs, c.impl.Model, opts...); err != nil {
		return nil, err
	}
	var resp CountTokensResponse
	u := "https://generativelanguage.googleapis.com/v1beta/models/" + url.PathEscape(c.impl.Model) + ":countTokens"
	if err := c.impl.DoRequest(ctx, "POST", u, &req, &resp); err != nil {
		return nil, err
	}
	return &resp, nil
}

// FileUpload uploads a file using the Gemini resumable upload protocol.
//
// The file is uploaded in two phases:
//  1. Initiate a resumable upload session with metadata.
//  2. Upload the file bytes to the returned upload URL.
//
// https://ai.google.dev/api/files#method:-media.upload
func (c *Client) FileUpload(ctx context.Context, displayName, mimeType string, r io.Reader) (*FileMetadata, error) {
	// Phase 1: Start the resumable upload.
	meta := struct {
		File struct {
			DisplayName string `json:"displayName"`
		} `json:"file"`
	}{}
	meta.File.DisplayName = displayName
	body, err := json.Marshal(&meta)
	if err != nil {
		return nil, err
	}
	startURL := "https://generativelanguage.googleapis.com/upload/v1beta/files"
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, startURL, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json; charset=utf-8")
	req.Header.Set("X-Goog-Upload-Protocol", "resumable")
	req.Header.Set("X-Goog-Upload-Command", "start")
	req.Header.Set("X-Goog-Upload-Header-Content-Type", mimeType)
	resp, err := c.impl.Client.Do(req)
	if err != nil {
		return nil, err
	}
	_ = resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("gemini: file upload start failed with status %d", resp.StatusCode)
	}
	uploadURL := resp.Header.Get("X-Goog-Upload-Url")
	if uploadURL == "" {
		return nil, errors.New("gemini: missing X-Goog-Upload-Url in response")
	}

	// Phase 2: Upload the file bytes.
	data, err := io.ReadAll(r)
	if err != nil {
		return nil, err
	}
	req, err = http.NewRequestWithContext(ctx, http.MethodPut, uploadURL, bytes.NewReader(data))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Length", strconv.Itoa(len(data)))
	req.Header.Set("X-Goog-Upload-Offset", "0")
	req.Header.Set("X-Goog-Upload-Command", "upload, finalize")
	resp, err = c.impl.Client.Do(req)
	if err != nil {
		if resp != nil {
			_ = resp.Body.Close()
		}
		return nil, err
	}
	var result struct {
		File FileMetadata `json:"file"`
	}
	if err := c.impl.DecodeResponse(resp, uploadURL, &result); err != nil {
		return nil, err
	}
	return &result.File, nil
}

// FileGetMetadata retrieves metadata for a single file.
//
// The name parameter should be in the form "files/{id}".
//
// https://ai.google.dev/api/files#method:-files.get
func (c *Client) FileGetMetadata(ctx context.Context, name string) (*FileMetadata, error) {
	u := "https://generativelanguage.googleapis.com/v1beta/" + name
	var resp FileMetadata
	if err := c.impl.DoRequest(ctx, "GET", u, nil, &resp); err != nil {
		return nil, err
	}
	return &resp, nil
}

// FileList returns all files.
//
// https://ai.google.dev/api/files#method:-files.list
func (c *Client) FileList(ctx context.Context) ([]FileMetadata, error) {
	resp, err := c.FileListRaw(ctx)
	if err != nil {
		return nil, err
	}
	return resp.Files, nil
}

// FileListRaw returns the raw paginated response for files.
//
// https://ai.google.dev/api/files#method:-files.list
func (c *Client) FileListRaw(ctx context.Context) (*FileListResponse, error) {
	u := "https://generativelanguage.googleapis.com/v1beta/files?pageSize=100"
	resp := &FileListResponse{}
	if err := c.impl.DoRequest(ctx, "GET", u, nil, resp); err != nil {
		return nil, err
	}
	return resp, nil
}

// FileDelete deletes a file.
//
// The name parameter should be in the form "files/{id}".
//
// https://ai.google.dev/api/files#method:-files.delete
func (c *Client) FileDelete(ctx context.Context, name string) error {
	u := "https://generativelanguage.googleapis.com/v1beta/" + name
	return c.impl.DoRequest(ctx, "DELETE", u, nil, &struct{}{})
}

// FileSearchStoreCreate creates a new file search store.
//
// https://ai.google.dev/gemini-api/docs/file-search
func (c *Client) FileSearchStoreCreate(ctx context.Context, displayName string) (*FileSearchStore, error) {
	req := struct {
		DisplayName string `json:"displayName"`
	}{DisplayName: displayName}
	var resp FileSearchStore
	u := "https://generativelanguage.googleapis.com/v1beta/fileSearchStores"
	if err := c.impl.DoRequest(ctx, "POST", u, &req, &resp); err != nil {
		return nil, err
	}
	return &resp, nil
}

// FileSearchStoreGet retrieves a file search store.
//
// The name parameter should be in the form "fileSearchStores/{id}".
//
// https://ai.google.dev/gemini-api/docs/file-search
func (c *Client) FileSearchStoreGet(ctx context.Context, name string) (*FileSearchStore, error) {
	u := "https://generativelanguage.googleapis.com/v1beta/" + name
	var resp FileSearchStore
	if err := c.impl.DoRequest(ctx, "GET", u, nil, &resp); err != nil {
		return nil, err
	}
	return &resp, nil
}

// FileSearchStoreList returns all file search stores.
//
// https://ai.google.dev/gemini-api/docs/file-search
func (c *Client) FileSearchStoreList(ctx context.Context) ([]FileSearchStore, error) {
	resp, err := c.FileSearchStoreListRaw(ctx)
	if err != nil {
		return nil, err
	}
	return resp.FileSearchStores, nil
}

// FileSearchStoreListRaw returns the raw paginated response for file search stores.
//
// https://ai.google.dev/gemini-api/docs/file-search
func (c *Client) FileSearchStoreListRaw(ctx context.Context) (*FileSearchStoreListResponse, error) {
	u := "https://generativelanguage.googleapis.com/v1beta/fileSearchStores?pageSize=20"
	resp := &FileSearchStoreListResponse{}
	if err := c.impl.DoRequest(ctx, "GET", u, nil, resp); err != nil {
		return nil, err
	}
	return resp, nil
}

// FileSearchStoreDelete deletes a file search store.
//
// The name parameter should be in the form "fileSearchStores/{id}".
// Set force to true to delete the store even if it contains documents.
//
// https://ai.google.dev/gemini-api/docs/file-search
func (c *Client) FileSearchStoreDelete(ctx context.Context, name string, force bool) error {
	u := "https://generativelanguage.googleapis.com/v1beta/" + name + "?force=" + strconv.FormatBool(force)
	return c.impl.DoRequest(ctx, "DELETE", u, nil, &struct{}{})
}

// FileSearchStoreUploadDocument uploads a document to a file search store using the resumable upload protocol.
//
// The store parameter should be in the form "fileSearchStores/{id}".
// Returns an Operation since document processing is asynchronous.
//
// https://ai.google.dev/gemini-api/docs/file-search
func (c *Client) FileSearchStoreUploadDocument(ctx context.Context, store, displayName, mimeType string, r io.Reader) (*Operation, error) {
	// Phase 1: Start the resumable upload.
	meta := struct {
		DisplayName string `json:"displayName,omitzero"`
	}{DisplayName: displayName}
	body, err := json.Marshal(&meta)
	if err != nil {
		return nil, err
	}
	startURL := "https://generativelanguage.googleapis.com/upload/v1beta/" + store + ":uploadToFileSearchStore"
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, startURL, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json; charset=utf-8")
	req.Header.Set("X-Goog-Upload-Protocol", "resumable")
	req.Header.Set("X-Goog-Upload-Command", "start")
	req.Header.Set("X-Goog-Upload-Header-Content-Type", mimeType)
	resp, err := c.impl.Client.Do(req)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(resp.Body)
		_ = resp.Body.Close()
		return nil, fmt.Errorf("gemini: file search store upload start failed with status %d: %s", resp.StatusCode, b)
	}
	_ = resp.Body.Close()
	uploadURL := resp.Header.Get("X-Goog-Upload-Url")
	if uploadURL == "" {
		return nil, errors.New("gemini: missing X-Goog-Upload-Url in response")
	}

	// Phase 2: Upload the file bytes.
	data, err := io.ReadAll(r)
	if err != nil {
		return nil, err
	}
	req, err = http.NewRequestWithContext(ctx, http.MethodPut, uploadURL, bytes.NewReader(data))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Length", strconv.Itoa(len(data)))
	req.Header.Set("X-Goog-Upload-Offset", "0")
	req.Header.Set("X-Goog-Upload-Command", "upload, finalize")
	resp, err = c.impl.Client.Do(req)
	if err != nil {
		if resp != nil {
			_ = resp.Body.Close()
		}
		return nil, err
	}
	var result Operation
	if err := c.impl.DecodeResponse(resp, uploadURL, &result); err != nil {
		return nil, err
	}
	return &result, nil
}

// FileSearchStoreImportFile imports an already-uploaded file into a file search store.
//
// The store parameter should be in the form "fileSearchStores/{id}".
// The fileName parameter should be in the form "files/{id}".
//
// https://ai.google.dev/gemini-api/docs/file-search
func (c *Client) FileSearchStoreImportFile(ctx context.Context, store, fileName string) (*Operation, error) {
	req := struct {
		FileName string `json:"fileName"`
	}{FileName: fileName}
	var resp Operation
	u := "https://generativelanguage.googleapis.com/v1beta/" + store + ":importFile"
	if err := c.impl.DoRequest(ctx, "POST", u, &req, &resp); err != nil {
		return nil, err
	}
	return &resp, nil
}

// FileSearchStoreDocumentGet retrieves a document from a file search store.
//
// The name parameter should be in the form "fileSearchStores/{storeId}/documents/{docId}".
//
// https://ai.google.dev/gemini-api/docs/file-search
func (c *Client) FileSearchStoreDocumentGet(ctx context.Context, name string) (*FileSearchStoreDocument, error) {
	u := "https://generativelanguage.googleapis.com/v1beta/" + name
	var resp FileSearchStoreDocument
	if err := c.impl.DoRequest(ctx, "GET", u, nil, &resp); err != nil {
		return nil, err
	}
	return &resp, nil
}

// FileSearchStoreDocumentList returns all documents in a file search store.
//
// The store parameter should be in the form "fileSearchStores/{id}".
//
// https://ai.google.dev/gemini-api/docs/file-search
func (c *Client) FileSearchStoreDocumentList(ctx context.Context, store string) ([]FileSearchStoreDocument, error) {
	resp, err := c.FileSearchStoreDocumentListRaw(ctx, store)
	if err != nil {
		return nil, err
	}
	return resp.Documents, nil
}

// FileSearchStoreDocumentListRaw returns the raw paginated response for documents in a file search store.
//
// The store parameter should be in the form "fileSearchStores/{id}".
//
// https://ai.google.dev/gemini-api/docs/file-search
func (c *Client) FileSearchStoreDocumentListRaw(ctx context.Context, store string) (*FileSearchStoreDocumentListResponse, error) {
	u := "https://generativelanguage.googleapis.com/v1beta/" + store + "/documents?pageSize=20"
	resp := &FileSearchStoreDocumentListResponse{}
	if err := c.impl.DoRequest(ctx, "GET", u, nil, resp); err != nil {
		return nil, err
	}
	return resp, nil
}

// FileSearchStoreDocumentDelete deletes a document from a file search store.
//
// The name parameter should be in the form "fileSearchStores/{storeId}/documents/{docId}".
// Set force to true to force deletion even if the document is still being processed.
//
// https://ai.google.dev/gemini-api/docs/file-search
func (c *Client) FileSearchStoreDocumentDelete(ctx context.Context, name string, force bool) error {
	u := "https://generativelanguage.googleapis.com/v1beta/" + name + "?force=" + strconv.FormatBool(force)
	return c.impl.DoRequest(ctx, "DELETE", u, nil, &struct{}{})
}

// TODO: To implement ProviderGenAsync, we need to use the Vertex API, not the API key based Gemini one.
// https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/batch-prediction-api
// This may require creating a whole new provider with Vertex AI API surface.

// Capabilities implements genai.Provider.
func (c *Client) Capabilities() genai.ProviderCapabilities {
	// GenAsync (predictLongRunning) is only supported for video generation models.
	// Text models use generateContent, which doesn't support async operations.
	return genai.ProviderCapabilities{Realtime: true, Embed: true,
		GenAsync: slices.Contains(c.impl.OutputModalities, genai.ModalityVideo),
		Caching:  true,
	}
}

// isPinnedModel returns true if the model name ends with a version number (e.g., -001, -002).
func isPinnedModel(name string) bool {
	parts := strings.Split(name, "-")
	if len(parts) == 0 {
		return false
	}
	lastPart := parts[len(parts)-1]
	// A pinned model ends with 3 digits and is not "-latest"
	if lastPart == "latest" {
		return false
	}
	return len(lastPart) == 3 && lastPart[0] >= '0' && lastPart[0] <= '9'
}

// compareVersions compares two model names by their version numbers.
// Returns >0 if a > b, <0 if a < b, 0 if equal or incomparable.
// Example: "gemini-3-flash" > "gemini-2.5-flash".
func compareVersions(a, b string) int {
	// Extract version parts (e.g., "2.5" from "gemini-2.5-flash")
	getVersionParts := func(name string) (major, minor int) {
		parts := strings.Split(name, "-")
		for i, part := range parts {
			if part == "gemini" && i+1 < len(parts) {
				versionStr := parts[i+1]
				// Check if the next meaningful part is a type (flash, pro, etc)
				var isVersionNumber bool
				for j := i + 2; j < len(parts); j++ {
					if parts[j] == "flash" || parts[j] == "pro" || parts[j] == "lite" {
						isVersionNumber = true
						break
					}
				}
				if isVersionNumber {
					// Parse major.minor version
					versionParts := strings.Split(versionStr, ".")
					if len(versionParts) >= 1 {
						if v, err := strconv.Atoi(versionParts[0]); err == nil {
							major = v
						}
					}
					if len(versionParts) >= 2 {
						if v, err := strconv.Atoi(versionParts[1]); err == nil {
							minor = v
						}
					}
					return major, minor
				}
			}
		}
		return major, minor
	}

	aMajor, aMinor := getVersionParts(a)
	bMajor, bMinor := getVersionParts(b)
	if aMajor != bMajor {
		return aMajor - bMajor
	}
	return aMinor - bMinor
}

// ProcessStream converts the raw packets from the streaming API into Reply fragments.
func ProcessStream(chunks iter.Seq[ChatStreamChunkResponse]) (iter.Seq[genai.Reply], func() (genai.Usage, [][]genai.Logprob, error)) {
	var finalErr error
	u := genai.Usage{}

	return func(yield func(genai.Reply) bool) {
			for pkt := range chunks {
				if len(pkt.Candidates) != 1 {
					continue
				}
				if pkt.UsageMetadata.TotalTokenCount != 0 {
					// Not 100% sure.
					u.InputTokens = pkt.UsageMetadata.PromptTokenCount
					u.ReasoningTokens = pkt.UsageMetadata.ThoughtsTokenCount
					u.OutputTokens = pkt.UsageMetadata.CandidatesTokenCount + pkt.UsageMetadata.ToolUsePromptTokenCount + pkt.UsageMetadata.ThoughtsTokenCount
					u.TotalTokens = pkt.UsageMetadata.TotalTokenCount
				}
				if pkt.Candidates[0].FinishReason != "" {
					u.FinishReason = pkt.Candidates[0].FinishReason.ToFinishReason()
				}
				switch role := pkt.Candidates[0].Content.Role; role {
				case "model", "":
				default:
					finalErr = &internal.BadError{Err: fmt.Errorf("unexpected role %q", role)}
					return
				}

				f := genai.Reply{}

				if !pkt.Candidates[0].GroundingMetadata.IsZero() {
					replies, err := pkt.Candidates[0].GroundingMetadata.To()
					if err != nil {
						finalErr = &internal.BadError{Err: err}
						return
					}
					// Handle citations as a separate packet.
					for i := range replies {
						f.Citation = replies[i].Citation
						if !yield(f) {
							return
						}
					}
					f = genai.Reply{}
				}

				if len(pkt.Candidates[0].UrlContextMetadata.UrlMetadata) > 0 {
					urlReplies := pkt.Candidates[0].UrlContextMetadata.To()
					for i := range urlReplies {
						f.Citation = urlReplies[i].Citation
						if !yield(f) {
							return
						}
					}
					f = genai.Reply{}
				}

				for i := range pkt.Candidates[0].Content.Parts {
					part := &pkt.Candidates[0].Content.Parts[i]
					if len(part.ThoughtSignature) != 0 {
						f.Opaque = map[string]any{"signature": part.ThoughtSignature}
					}
					if part.Thought {
						f.Reasoning += part.Text
					} else {
						f.Text += part.Text
					}
					if part.InlineData.MimeType != "" || len(part.InlineData.Data) != 0 {
						exts, err := mime.ExtensionsByType(part.InlineData.MimeType)
						if err != nil {
							finalErr = &internal.BadError{Err: fmt.Errorf("failed to get extension for mime type %q: %w", part.InlineData.MimeType, err)}
							return
						}
						if len(exts) == 0 {
							finalErr = &internal.BadError{Err: fmt.Errorf("mime type %q has no extension", part.InlineData.MimeType)}
							return
						}
						f.Doc.Filename = "content" + exts[0]
						f.Doc.Src = &bb.BytesBuffer{D: part.InlineData.Data}
					}
					if part.FunctionCall.ID != "" || part.FunctionCall.Name != "" {
						f.Opaque = nil
						// https://ai.google.dev/api/caching?hl=en#FunctionCall
						if len(part.ThoughtSignature) != 0 {
							f.ToolCall.Opaque = map[string]any{"signature": part.ThoughtSignature}
						}
						if err := part.FunctionCall.To(&f.ToolCall); err != nil {
							finalErr = err
							return
						}
					}
					if part.FunctionResponse.ID != "" {
						// https://ai.google.dev/api/caching?hl=en#FunctionResponse
						finalErr = &internal.BadError{Err: fmt.Errorf("implement function response %#v", part)}
						return
					}
					if part.FileData.MimeType != "" || part.FileData.FileURI != "" {
						exts, err := mime.ExtensionsByType(part.FileData.MimeType)
						if err != nil {
							finalErr = &internal.BadError{Err: fmt.Errorf("failed to get extension for mime type %q: %w", part.FileData.MimeType, err)}
							return
						}
						if len(exts) == 0 {
							finalErr = &internal.BadError{Err: fmt.Errorf("mime type %q has no extension", part.FileData.MimeType)}
							return
						}
						f.Doc.Filename = "content" + exts[0]
						f.Doc.URL = part.FileData.FileURI
					}
					if part.ExecutableCode.Language != "" || part.ExecutableCode.Code != "" {
						// https://ai.google.dev/api/caching?hl=en#ExecutableCode
						f.Text = part.ExecutableCode.Code
						f.Opaque = map[string]any{"type": "executable_code", "language": part.ExecutableCode.Language}
						if !yield(f) {
							return
						}
						f = genai.Reply{}
					}
					if part.CodeExecutionResult.Outcome != "" || part.CodeExecutionResult.Output != "" {
						// https://ai.google.dev/api/caching?hl=en#CodeExecutionResult
						f.Text = part.CodeExecutionResult.Output
						f.Opaque = map[string]any{"type": "code_execution_result", "outcome": part.CodeExecutionResult.Outcome}
						if !yield(f) {
							return
						}
						f = genai.Reply{}
					}
					if !f.IsZero() && !yield(f) {
						return
					}
					f = genai.Reply{}
				}
			}
		}, func() (genai.Usage, [][]genai.Logprob, error) {
			return u, nil, finalErr
		}
}

func yieldNothing[T any](yield func(T) bool) {
}

var _ genai.Provider = &Client{}

// Embed implements genai.Provider without selecting a task type.
// Document inputs use native content conversion and require a compatible model;
// a text-only model returns base.ErrNotSupported.
func (c *Client) Embed(ctx context.Context, in *genai.EmbeddingRequest) (*genai.EmbeddingResponse, error) {
	if in == nil {
		return nil, errors.New("embedding request is required")
	}
	if err := in.Validate(); err != nil {
		return nil, err
	}
	if in.Dimensions > math.MaxInt32 {
		return nil, errors.New("embedding dimensions exceed int32 range")
	}
	model := "models/" + strings.TrimPrefix(c.ModelID(), "models/")
	req := BatchEmbeddingRequest{Requests: make([]EmbeddingRequest, len(in.Inputs))}
	hasDoc := false
	for i := range in.Inputs {
		input := in.Inputs[i]
		hasDoc = hasDoc || !input.Doc.IsZero()
		var part Part
		if err := part.FromRequest(&input); err != nil {
			return nil, fmt.Errorf("embedding input #%d: %w", i, err)
		}
		req.Requests[i] = EmbeddingRequest{Model: model, Content: Content{Parts: []Part{part}}, EmbedContentConfig: EmbedContentConfig{OutputDimensionality: int32(in.Dimensions)}}
	}
	var raw BatchEmbeddingResponse
	if err := c.EmbedRaw(ctx, &req, &raw); err != nil {
		// Text-only models reject a valid media-only input as empty text.
		if e, ok := errors.AsType[*ErrorResponse](err); ok && hasDoc && e.ErrorVal.Code == http.StatusBadRequest && e.ErrorVal.Status == "INVALID_ARGUMENT" && e.ErrorVal.Message == "The text content is empty." {
			return nil, fmt.Errorf("gemini embedding model rejects media: %w", &base.ErrNotSupported{Options: []string{"Request.Doc"}})
		}
		return nil, err
	}
	out := &genai.EmbeddingResponse{Embeddings: make([][]float32, len(raw.Embeddings)), Usage: genai.Usage{InputTokens: raw.UsageMetadata.PromptTokenCount, TotalTokens: raw.UsageMetadata.PromptTokenCount}}
	for i := range raw.Embeddings {
		out.Embeddings[i] = raw.Embeddings[i].Values
	}
	if err := out.Validate(); err != nil {
		return nil, &internal.BadError{Err: fmt.Errorf("invalid embedding response: %w", err)}
	}
	if len(out.Embeddings) != len(in.Inputs) {
		return nil, &internal.BadError{Err: fmt.Errorf("got %d embedding vectors for %d inputs", len(out.Embeddings), len(in.Inputs))}
	}
	if in.Dimensions != 0 && len(out.Embeddings[0]) != in.Dimensions {
		return nil, &internal.BadError{Err: fmt.Errorf("got %d embedding dimensions, want %d", len(out.Embeddings[0]), in.Dimensions)}
	}
	return out, nil
}

// EmbedRaw exposes the native batchEmbedContents endpoint. Models and
// per-content task controls come from the requests, not generation configuration.
func (c *Client) EmbedRaw(ctx context.Context, in *BatchEmbeddingRequest, out *BatchEmbeddingResponse) error {
	if in == nil || out == nil {
		return errors.New("embedding request and response are required")
	}
	if err := in.Validate(); err != nil {
		return err
	}
	u := "https://generativelanguage.googleapis.com/v1beta/models/" + url.PathEscape(strings.TrimPrefix(in.Requests[0].Model, "models/")) + ":batchEmbedContents"
	if err := c.impl.DoRequest(ctx, http.MethodPost, u, in, out); err != nil {
		*out = BatchEmbeddingResponse{}
		return err
	}
	return nil
}

// InteractionRaw creates a batch Interactions speech request. It preserves native
// statuses, annotations and usage; a failed interaction resource is not an HTTP
// error. Callers decide whether the returned status satisfies their operation.
// Responses are limited to 1 MiB unless in.MaxResponseBytes sets an explicit bound.
// Requests are limited to 20 MB JSON; Source inputs are consumed synchronously
// into a private temporary file. Its lifetime extends until all transport-owned
// request bodies close, including uploads continuing after an early response.
func (c *Client) InteractionRaw(ctx context.Context, in *InteractionRequest, out *InteractionResponse) (err error) {
	if out == nil {
		return errors.New("interaction response is required")
	}
	*out = InteractionResponse{}
	resp, err := c.interactionRequest(ctx, in, false)
	if err != nil {
		return err
	}
	if resp.StatusCode != http.StatusOK {
		return c.impl.DecodeError(speechEndpoint, resp)
	}
	defer func() {
		err = errors.Join(err, resp.Body.Close())
		if err != nil {
			*out = InteractionResponse{}
		}
	}()
	limit := in.MaxResponseBytes
	if limit == 0 {
		limit = 1024 * 1024
	}
	decoder := json.NewDecoder(http.MaxBytesReader(nil, resp.Body, limit))
	if err := decoder.Decode(out); err != nil {
		return err
	}
	var extra any
	if err = decoder.Decode(&extra); !errors.Is(err, io.EOF) {
		if err == nil {
			err = errors.New("multiple speech response values")
		}
		return err
	}
	return nil
}

// InteractionStreamRaw creates a streaming Interactions speech request. It
// returns a single-use, pull-driven event sequence and an idempotent finalizer.
// Call finish after iteration to obtain protocol/provider/close errors. Breaking
// iteration closes the response without draining it; call finish also if never
// iterating. Finish must not race iteration. Error and terminal events are yielded
// before their errors are returned by finish. SSE events are limited to 1 MiB.
// The request's Stream field is overridden without mutating it. Source ownership
// and request size limits match InteractionRaw.
func (c *Client) InteractionStreamRaw(ctx context.Context, in *InteractionRequest) (iter.Seq[InteractionEvent], func() error) {
	resp, err := c.interactionRequest(ctx, in, true)
	if err != nil {
		return yieldNothing[InteractionEvent], func() error { return err }
	}
	if resp.StatusCode != http.StatusOK {
		err = c.impl.DecodeError(speechEndpoint, resp)
		return yieldNothing[InteractionEvent], func() error { return err }
	}
	if typ, _, err := mime.ParseMediaType(resp.Header.Get("Content-Type")); err != nil || typ != "text/event-stream" {
		err = errors.Join(errors.New("speech response must be an event stream"), resp.Body.Close())
		return yieldNothing[InteractionEvent], func() error { return err }
	}
	scan := bufio.NewScanner(resp.Body)
	scan.Buffer(make([]byte, 4096), 1024*1024)
	stream := &interactionStream{body: resp.Body, scan: scan}
	return stream.iterate, stream.finish
}

// ConnectLiveRaw opens the native Gemini Live protocol with the supplied setup.
// Setup is sent unchanged; model and feature configuration belong to the caller.
// Initial preserves the native setupComplete resource. Subsequent server frames,
// including transcription segments, errors and goAway, are returned uncollapsed.
// Authentication, bounded reads, cancellation and teardown use the same transport
// as other Gemini realtime operations. This method does not list models.
func (c *Client) ConnectLiveRaw(ctx context.Context, setup *LiveSetup) (*LiveConn, error) {
	if setup == nil || setup.Model == "" {
		return nil, errors.New("gemini live setup and model are required")
	}
	data, err := json.Marshal(LiveClientMessage{Setup: setup})
	if err != nil {
		return nil, err
	}
	ws, err := realtime.Dial(ctx, "wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent", &c.impl.Client, maxMessageBytes, 0, 0)
	if err != nil {
		return nil, err
	}
	conn := &LiveConn{ws: ws}
	if err = ws.Write(ctx, data); err == nil {
		conn.Initial, err = conn.Receive(ctx)
	}
	if err == nil && (conn.Initial.SetupComplete == nil || conn.Initial.Error != nil) {
		err = errors.New("gemini live setup rejected")
	}
	if err != nil {
		return nil, errors.Join(err, conn.Close())
	}
	return conn, nil
}

// detectModelModalities tries its best to figure out the modality of a model
//
// We may want to make this function overridable in the future by the client since this is going to break one
// day or another.
func (c *Client) detectModelModalities(ctx context.Context, model string) (genai.Modalities, error) {
	id := strings.TrimPrefix(model, "models/")
	if sb := c.Scoreboard(); sb.Embedding(id) != nil {
		return genai.Modalities{genai.ModalityEmbedding}, nil
	}
	for _, mdl := range c.impl.PreloadedModels {
		if m, ok := mdl.(*Model); ok && m.GetID() == id && m.embeddingOnly() {
			return genai.Modalities{genai.ModalityEmbedding}, nil
		}
	}

	// It's tricky because modalities are not directly returned by ListModels.
	switch {
	case strings.HasPrefix(model, "gemini-") && strings.HasSuffix(model, "-image"):
		return genai.Modalities{genai.ModalityImage}, nil
	case strings.HasPrefix(model, "gem"):
		return genai.Modalities{genai.ModalityText}, nil
	case strings.HasPrefix(model, "ima"):
		return genai.Modalities{genai.ModalityImage}, nil
	case strings.HasPrefix(model, "veo"):
		return genai.Modalities{genai.ModalityVideo}, nil
	}
	// We probably want to fetch SupportedGenerationMethods from the model anyway to make sure the right
	// API is used. We can keep a predefined table for known model, I'd be surprised the generation
	// methods would change per model.
	mdls, err := c.ListModels(ctx)
	if err != nil {
		return nil, fmt.Errorf("failed to automatically detect the model modality: %w", err)
	}
	for _, mdl := range mdls {
		if m := mdl.(*Model); m.GetID() == model { //nolint:forcetypeassert // ListModels constructs only provider Model values.
			switch {
			case m.embeddingOnly():
				return genai.Modalities{genai.ModalityEmbedding}, nil
			case slices.Contains(m.SupportedGenerationMethods, "generateContent"):
				return genai.Modalities{genai.ModalityText}, nil
			case slices.Contains(m.SupportedGenerationMethods, "predict"):
				return genai.Modalities{genai.ModalityImage}, nil
			case slices.Contains(m.SupportedGenerationMethods, "predictLongRunning"):
				return genai.Modalities{genai.ModalityVideo}, nil
			}
			return nil, fmt.Errorf("failed to automatically detect the model modality with methods: %s", m.SupportedGenerationMethods)
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

	// Build a map of available models for quick lookup.
	availableModels := map[string]struct{}{}
	for _, mdl := range mdls {
		availableModels[strings.TrimPrefix(mdl.(*Model).Name, "models/")] = struct{}{} //nolint:forcetypeassert // ListModels constructs only provider Model values.
	}
	cheap := preference == string(genai.ModelCheap)
	good := preference == string(genai.ModelGood) || preference == ""
	switch {
	case cheap:
		if _, ok := availableModels["gemini-flash-lite-latest"]; ok {
			return "gemini-flash-lite-latest", nil
		}
	case good:
		if _, ok := availableModels["gemini-flash-latest"]; ok {
			return "gemini-flash-latest", nil
		}
	default:
		if _, ok := availableModels["gemini-pro-latest"]; ok {
			return "gemini-pro-latest", nil
		}
	}

	// If the preferred "-latest" model is not available, find the best from the list.
	selectedModel := ""
	var tokens int64
	selectedPinned := false
	for _, mdl := range mdls {
		m := mdl.(*Model) //nolint:forcetypeassert // ListModels constructs only provider Model values.
		// Skip experimental models
		if !slices.Contains(m.SupportedGenerationMethods, "generateContent") || strings.Contains(m.Name, "tts") || (tokens != 0 && tokens > m.OutputTokenLimit) || strings.Contains(m.Name, "-exp") {
			continue
		}
		name := strings.TrimPrefix(m.Name, "models/")
		var matches bool
		switch {
		case cheap:
			matches = strings.HasPrefix(name, "gemini") && strings.Contains(name, "-flash-lite")
		case good:
			matches = strings.HasPrefix(name, "gemini") && strings.Contains(name, "-flash") && !strings.Contains(name, "-lite")
		default:
			matches = strings.HasPrefix(name, "gemini") && strings.Contains(name, "-pro")
		}
		if !matches {
			continue
		}
		// Check if this model is better than the current selection.
		isCandidatePinned := isPinnedModel(name)
		shouldUpdate := false
		switch {
		case selectedModel == "":
			shouldUpdate = true
		case isCandidatePinned && !selectedPinned:
			// Prefer pinned models (with -NNN suffix) over unpinned ones
			shouldUpdate = true
		case isCandidatePinned == selectedPinned:
			// Both pinned or both unpinned; compare versions
			if compareVersions(name, selectedModel) > 0 {
				shouldUpdate = true
			}
		}
		if shouldUpdate {
			selectedModel = name
			selectedPinned = isCandidatePinned
			tokens = m.OutputTokenLimit
		}
	}
	if selectedModel == "" {
		return "", errors.New("failed to find a model automatically")
	}
	return selectedModel, nil
}

// selectBestImageModel selects the most appropriate model based on the preference (cheap, good, or SOTA).
//
// Image generation models are the gemini-*-image models (a.k.a. Nano Banana).
// The SOTA tier is the pro model, the good tier is the flash model, and the cheap tier is the flash-lite model.
// Preview models are skipped. The newest version wins.
func (c *Client) selectBestImageModel(ctx context.Context, preference string) (string, error) {
	mdls, err := c.ListModels(ctx)
	if err != nil {
		return "", fmt.Errorf("failed to automatically select the model: %w", err)
	}
	cheap := preference == string(genai.ModelCheap)
	good := preference == string(genai.ModelGood) || preference == ""
	selectedModel := ""
	for _, mdl := range mdls {
		m := mdl.(*Model) //nolint:forcetypeassert // ListModels constructs only provider Model values.
		if !slices.Contains(m.SupportedGenerationMethods, "generateContent") || strings.Contains(m.Name, "tts") {
			continue
		}
		name := strings.TrimPrefix(m.Name, "models/")
		if !strings.HasSuffix(name, "-image") || strings.Contains(name, "-preview") {
			continue
		}
		switch {
		case cheap:
			if !strings.Contains(name, "flash-lite") {
				continue
			}
		case good:
			if !strings.Contains(name, "flash") || strings.Contains(name, "flash-lite") {
				continue
			}
		default:
			if !strings.Contains(name, "pro") {
				continue
			}
		}
		if selectedModel == "" || compareVersions(name, selectedModel) > 0 {
			selectedModel = name
		}
	}
	if selectedModel == "" {
		return "", errors.New("failed to find a model automatically")
	}
	return selectedModel, nil
}

// selectBestVideoModel selects the most appropriate video model based on the preference (cheap, good, or SOTA).
//
// Video models are identified by the "veo-" prefix.
func (c *Client) selectBestVideoModel(ctx context.Context, preference string) (string, error) {
	mdls, err := c.ListModels(ctx)
	if err != nil {
		return "", fmt.Errorf("failed to automatically select the model: %w", err)
	}
	cheap := preference == string(genai.ModelCheap)
	good := preference == string(genai.ModelGood) || preference == ""
	selectedModel := ""
	for _, mdl := range mdls {
		m := mdl.(*Model) //nolint:forcetypeassert // ListModels constructs only provider Model values.
		name := strings.TrimPrefix(m.Name, "models/")
		// Only consider models starting with "veo-"
		if !strings.HasPrefix(name, "veo-") {
			continue
		}
		isFast := strings.Contains(name, "fast")
		matches := isFast
		if !cheap && !good {
			matches = !matches
		}
		if !matches {
			continue
		}
		// Select the best available model, preferring newer versions lexicographically
		if selectedModel == "" || name > selectedModel {
			selectedModel = name
		}
	}
	if selectedModel == "" {
		return "", errors.New("failed to find a video model automatically")
	}
	return selectedModel, nil
}

// selectBestAudioModel selects the most appropriate audio model.
//
// Audio models are identified by the "native-audio" suffix in their names.
func (c *Client) selectBestAudioModel(ctx context.Context) (string, error) {
	mdls, err := c.ListModels(ctx)
	if err != nil {
		return "", fmt.Errorf("failed to automatically select the model: %w", err)
	}
	selectedModel := ""
	for _, mdl := range mdls {
		m := mdl.(*Model) //nolint:forcetypeassert // ListModels constructs only provider Model values.
		name := strings.TrimPrefix(m.Name, "models/")
		// Only consider models with "native-audio" in their name
		if !strings.Contains(name, "native-audio") {
			continue
		}
		// Select the best available model, preferring newer versions lexicographically
		if selectedModel == "" || name > selectedModel {
			selectedModel = name
		}
	}
	if selectedModel == "" {
		return "", errors.New("failed to find an audio model automatically")
	}
	return selectedModel, nil
}

// genDoc is a simplified version of GenSync.
//
// Use it to generate images.
//
// genDoc is only supported for models that have "predict" reported in their Model.SupportedGenerationMethods.
func (c *Client) genDoc(ctx context.Context, msg *genai.Message, opts ...genai.GenOption) (genai.Result, error) {
	// TODO: Smartly decide the method to use instead of hardcoding on the modality.
	if slices.Contains(c.impl.OutputModalities, genai.ModalityVideo) {
		waitForPoll := time.Second
		filtered := make([]genai.GenOption, 0, len(opts))
		for _, opt := range opts {
			if v, ok := opt.(genai.GenOptionPollInterval); ok {
				waitForPoll = time.Duration(v)
			} else {
				filtered = append(filtered, opt)
			}
		}
		id, err := c.GenAsync(ctx, genai.Messages{*msg}, filtered...)
		if err != nil {
			return genai.Result{}, err
		}
		// Loop until the result is available.
		for {
			select {
			case <-ctx.Done():
				return genai.Result{}, ctx.Err()
			case <-time.After(waitForPoll):
				if res, err := c.PokeResult(ctx, id); res.Usage.FinishReason != genai.Pending {
					return res, err
				}
			}
		}
	}
	res := genai.Result{}
	req := ImageRequest{}
	if err := req.Init(msg, c.impl.Model, c.impl.OutputModalities, opts...); err != nil {
		return res, err
	}
	resp, err := c.PredictRaw(ctx, &req)
	if err != nil {
		return res, err
	}
	// As of now, the last item in the list contains ContentType = "Positive Prompt".
	nbImages := 0
	for i := range resp.Predictions {
		if len(resp.Predictions[i].BytesBase64Encoded) > 0 {
			nbImages++
		}
	}
	for i := range resp.Predictions {
		if len(resp.Predictions[i].BytesBase64Encoded) == 0 {
			continue
		}
		if resp.Predictions[i].MimeType != "image/jpeg" {
			return res, fmt.Errorf("unsupported mime type %q", resp.Predictions[i].MimeType)
		}
		n := "content.jpg"
		if nbImages > 1 {
			n = fmt.Sprintf("content%d.jpg", i+1)
		}
		res.Replies = append(res.Replies, genai.Reply{Doc: genai.Doc{Filename: n, Src: &bb.BytesBuffer{D: resp.Predictions[i].BytesBase64Encoded}}})
	}
	if err := res.Validate(); err != nil {
		return res, err
	}
	return res, nil
}

func (c *Client) interactionRequest(ctx context.Context, in *InteractionRequest, stream bool) (resp *http.Response, err error) {
	if err := c.impl.Validate(); err != nil {
		return nil, err
	}
	if in == nil {
		return nil, errors.New("interaction request is required")
	}
	if err := in.Validate(); err != nil {
		return nil, err
	}
	reqBody := *in
	reqBody.Stream = stream
	hasSource := false
	for i := range in.Input {
		hasSource = hasSource || in.Input[i].Source != nil
	}
	var req *http.Request
	if !hasSource {
		data, e := json.Marshal(&reqBody)
		err = e
		if err != nil {
			return nil, err
		}
		if len(data) > 20_000_000 {
			return nil, errors.New("inline speech request exceeds 20 MB")
		}
		data = append(data, '\n')
		req, err = http.NewRequestWithContext(ctx, http.MethodPost, speechEndpoint, bytes.NewReader(data))
		if err == nil {
			req.Header.Set("Content-Type", "application/json; charset=utf-8")
		}
	} else {
		var upload *interactionUpload
		upload, err = spoolInteractionRequest(ctx, &reqBody)
		if err != nil {
			return nil, err
		}
		defer func() {
			err = errors.Join(err, upload.release())
			if err != nil && resp != nil {
				err = errors.Join(err, resp.Body.Close())
				resp = nil
			}
		}()
		var body io.ReadCloser
		body, err = upload.open()
		if err != nil {
			return nil, err
		}
		req, err = http.NewRequestWithContext(ctx, http.MethodPost, speechEndpoint, body)
		if err != nil {
			return nil, errors.Join(err, body.Close())
		}
		req.Header.Set("Content-Type", "application/json")
		req.ContentLength = upload.size
		req.GetBody = upload.open
	}
	if err != nil {
		return nil, err
	}
	resp, err = c.impl.Client.Do(req)
	if err != nil && resp != nil {
		err = errors.Join(err, resp.Body.Close())
		resp = nil
	}
	return resp, err
}
