// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// OpenAI realtime session setup shared by the Chat and Responses clients.

package openaibase

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"net/url"
	"reflect"
	"strings"
	"time"

	"github.com/maruel/genai"
	"github.com/maruel/genai/base"
)

// RealtimeOption configures OpenAI Realtime beyond the portable session contract.
type RealtimeOption struct {
	// Endpoint replaces the WebSocket URL with a trusted proxy. The client's
	// authentication transport is used for this endpoint too. No query is allowed;
	// ConnectRealtime adds the client's explicit model.
	Endpoint string
	// InputTranscriptionModel is required when InputTranscription is requested.
	// It is explicit because transcription model availability changes separately
	// from realtime models. InputLanguage is an optional language hint. A full
	// BCP 47 locale such as fr-CA is accepted and reduced to its ISO-639-1 primary
	// language.
	InputTranscriptionModel string
	InputLanguage           string
	// MaxOutputTokens bounds each model response; zero leaves the provider default.
	MaxOutputTokens int64
	// PingInterval and PingTimeout default to 30 and 10 seconds respectively.
	PingInterval time.Duration
	PingTimeout  time.Duration
}

// Validate checks provider-specific controls before opening a socket.
func (o *RealtimeOption) Validate() error {
	if o.Endpoint != "" {
		u, err := url.Parse(o.Endpoint)
		if err != nil || (u.Scheme != "ws" && u.Scheme != "wss") || u.Host == "" || u.User != nil || u.RawQuery != "" || u.Fragment != "" {
			return errors.New("OpenAI realtime endpoint must be an absolute ws/wss URL without credentials, query or fragment")
		}
	}
	if o.MaxOutputTokens < 0 {
		return errors.New("OpenAI realtime MaxOutputTokens must not be negative")
	}
	if o.PingInterval < 0 || o.PingTimeout < 0 {
		return errors.New("OpenAI realtime keepalive durations must not be negative")
	}
	if o.InputLanguage != "" {
		if _, err := normalizeInputLanguage(o.InputLanguage); err != nil {
			return err
		}
	}
	return nil
}

var _ genai.RealtimeOption = (*RealtimeOption)(nil)

// ConnectRealtimeSession opens an OpenAI GA realtime conversation with an
// explicit model and HTTP authentication transport. It waits for
// session.updated to acknowledge the setup. Native input and output are mono
// PCM16LE at 24 kHz. No model-list request occurs. Audio output includes its
// transcript; text-only output is a separate modality and cannot request an
// audio voice or output transcription. Input transcription needs an explicit
// model in RealtimeOption and is asynchronous guidance, not an exact record.
// The caller owns tools and playback.
//
// Server VAD is the default and automatically generates and interrupts
// responses. Manual turns send audio chunks, then set Commit to commit without
// replying or Generate to commit pending audio and request the reply. Text with
// Generate=false appends context; Generate=true also requests a response. Empty
// Text with Generate=true requests a response from the current context.
//
// A successful response emits one complete parallel tool batch. Submit every
// call's ID and name in RealtimeInputContent.ToolResults. Tool results always
// request the continuation. Cancelled,
// failed and incomplete responses never authorize tools. Function-call argument
// deltas and completions are informational provider events; execute only the
// complete batch. Sessions retain at most 1024 pending calls and 4096 historical
// call IDs.
//
// Cancel stops the current response and suppresses its late audio and
// transcripts. An explicit ResponseID must match the current observed response.
// Use Truncate with the assistant item ID, content index zero and played duration
// to reconcile unheard audio; playback remains application-owned. Recoverable
// provider errors are events; malformed recognized events terminate the session.
// Frames, input transactions and pending events are bounded.
//
// Protocol: GA /v1/realtime. See [Session] for the wire session.
func ConnectRealtimeSession(ctx context.Context, hc *http.Client, model string, opts ...genai.RealtimeOption) (*Session, error) {
	parallelToolCalls := true
	setup := &SessionConfig{
		Type:              SessionRealtime,
		OutputModalities:  Modalities{genai.ModalityAudio},
		Audio:             AudioConfig{Input: AudioInput{Format: AudioFormat{Type: AudioPCM, Rate: 24000}}, Output: AudioOutput{Format: AudioFormat{Type: AudioPCM, Rate: 24000}}},
		Tools:             ToolList{},
		ParallelToolCalls: &parallelToolCalls,
	}
	var turnDetection genai.RealtimeTurnDetection
	var outputTranscription bool
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
			if o != "" {
				e, err := ThinkingEffort(o, model)
				if err != nil {
					return nil, err
				}
				if o == genai.EffortOff || o == genai.EffortMinimal {
					e = ReasoningEffortMinimal
				}
				if e == ReasoningEffortMax {
					e = ReasoningEffortXHigh
				}
				setup.Reasoning = &ReasoningConfig{Effort: e}
			}
		case genai.ReasoningOutput:
			return nil, &base.ErrNotSupported{Options: []string{"IncludeReasoning"}}
		case genai.RealtimeOptionInstructions:
			setup.Instructions = string(o)
		case *genai.RealtimeOptionTools:
			if err := setup.Tools.Init(o.Tools); err != nil {
				return nil, err
			}
		case *genai.RealtimeOptionAudio:
			setup.Audio.Output.Voice.Name = o.Voice
			outputTranscription = o.OutputTranscription
			if o.InputTranscription {
				setup.Audio.Input.Transcription = &Transcription{}
			}
		case genai.Modalities:
			for i, m := range o {
				if m != genai.ModalityAudio && m != genai.ModalityText {
					return nil, fmt.Errorf("realtime option %T: unsupported realtime output modality %q", opt, m)
				}
				for _, prev := range o[:i] {
					if prev == m {
						return nil, fmt.Errorf("realtime option %T: duplicate output modality %q", opt, m)
					}
				}
			}
			setup.OutputModalities = Modalities(o)
		case genai.RealtimeTurnDetection:
			turnDetection = o
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
	if model == "" {
		return nil, errors.New("OpenAI realtime requires an explicit model")
	}
	o := RealtimeOption{}
	for _, opt := range opts {
		v, ok := opt.(*RealtimeOption)
		if !ok {
			return nil, fmt.Errorf("unsupported OpenAI realtime option %T: %w", opt, &base.ErrNotSupported{Options: []string{fmt.Sprintf("%T", opt)}})
		}
		o = *v
	}
	if o.Endpoint == "" {
		o.Endpoint = "wss://api.openai.com/v1/realtime"
	}
	if o.PingInterval == 0 {
		o.PingInterval = 30 * time.Second
	}
	if o.PingTimeout == 0 {
		o.PingTimeout = 10 * time.Second
	}
	u, err := url.Parse(o.Endpoint)
	if err != nil {
		return nil, errors.New("invalid OpenAI realtime endpoint")
	}
	q := u.Query()
	q.Set("model", model)
	u.RawQuery = q.Encode()
	if len(setup.OutputModalities) == 0 {
		setup.OutputModalities = Modalities{genai.ModalityAudio}
	}
	if len(setup.OutputModalities) > 1 {
		return nil, fmt.Errorf("OpenAI realtime accepts audio or text output, not both: %w", &base.ErrNotSupported{Options: []string{"mixed output modalities"}})
	}
	textOutput := setup.OutputModalities[0] == genai.ModalityText
	if textOutput && (outputTranscription || setup.Audio.Output.Voice.Name != "") {
		return nil, fmt.Errorf("OpenAI text output cannot request audio voice or transcription: %w", &base.ErrNotSupported{Options: []string{"voice or output transcription with text output"}})
	}
	if setup.Audio.Input.Transcription == nil && (o.InputLanguage != "" || o.InputTranscriptionModel != "") {
		return nil, errors.New("input transcription options require InputTranscription")
	}
	if o.MaxOutputTokens != 0 {
		setup.MaxOutputTokens = MaxOutputTokens{Value: o.MaxOutputTokens}
	}
	if turnDetection != genai.RealtimeTurnManual {
		setup.Audio.Input.TurnDetection = &TurnDetection{Type: TurnServerVAD, CreateResponse: true, InterruptResponse: true}
	}
	if setup.Audio.Input.Transcription != nil {
		if o.InputTranscriptionModel == "" {
			return nil, errors.New("InputTranscription requires an explicit OpenAI transcription model")
		}
		language, err := normalizeInputLanguage(o.InputLanguage)
		if err != nil {
			return nil, err
		}
		setup.Audio.Input.Transcription.Model = o.InputTranscriptionModel
		setup.Audio.Input.Transcription.Language = language
	}
	return ConnectRealtime(ctx, hc, u.String(), o.PingInterval, o.PingTimeout, setup)
}

// normalizeInputLanguage accepts a language or locale and returns its ISO-639-1
// primary language. Both '-' and '_' separators are accepted.
func normalizeInputLanguage(value string) (string, error) {
	if value == "" {
		return "", nil
	}
	parts := strings.Split(value, "-")
	if len(parts[0]) != 2 {
		return "", errors.New("OpenAI realtime input language must start with an ISO-639-1 code")
	}
	for _, c := range strings.ToLower(parts[0]) {
		if c < 'a' || c > 'z' {
			return "", errors.New("OpenAI realtime input language must start with an ISO-639-1 code")
		}
	}
	for _, part := range parts[1:] {
		if part == "" {
			return "", errors.New("OpenAI realtime input language has an empty locale segment")
		}
		for _, c := range strings.ToLower(part) {
			if (c < 'a' || c > 'z') && (c < '0' || c > '9') {
				return "", errors.New("OpenAI realtime input language has an invalid locale segment")
			}
		}
	}
	return strings.ToLower(parts[0]), nil
}
