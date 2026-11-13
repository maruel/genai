// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// llama.cpp realtime setup, session ownership and error handling.

package llamacpp

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"iter"
	"math"
	"net/url"
	"reflect"
	"slices"
	"strconv"
	"sync"
	"time"

	"github.com/maruel/genai"
	"github.com/maruel/genai/base"
	"github.com/maruel/genai/internal"
	"github.com/maruel/genai/internal/realtime"
)

// ErrNoActiveResponse means a cancel was rejected locally because no response is
// active or reserved for generation. No wire effects occurred. A terminal event
// may still be queued; observe completion before starting a dependent
// generation.
var ErrNoActiveResponse = errors.New("llama.cpp realtime has no active response")

// RealtimeOption controls response limits, tool choice and WebSocket health for
// llama.cpp.
// Configure the server endpoint with genai.ProviderOptionRemote, or replace the
// derived realtime URL with Endpoint for a trusted proxy. The server owns model
// selection; a persistent socket reserves one model slot, including idle.
type RealtimeOption struct {
	// Endpoint replaces the derived /v1/realtime WebSocket URL, for a trusted
	// proxy. Leave empty to derive it from the provider's remote URL.
	Endpoint string
	// MaxOutputTokens limits each response. Zero leaves the server default.
	MaxOutputTokens int64
	// RequireToolCall forces the first response of a session with tools to call
	// one. Later responses use automatic tool choice. It helps models that
	// otherwise answer instead of calling a tool when generating audio.
	RequireToolCall bool
	// PingInterval enables optional WebSocket keepalive. Zero disables it.
	// llama.cpp processes input synchronously on its WebSocket reader, so a
	// healthy server cannot answer pings while prefilling a large tool result.
	// PingTimeout defaults to 10 seconds when PingInterval is set.
	PingInterval time.Duration
	PingTimeout  time.Duration
}

// Validate rejects invalid limits before connecting to the runtime.
func (o *RealtimeOption) Validate() error {
	if o.Endpoint != "" {
		u, err := url.Parse(o.Endpoint)
		if err != nil || (u.Scheme != "ws" && u.Scheme != "wss") || u.Host == "" || u.User != nil || u.RawQuery != "" || u.Fragment != "" {
			return errors.New("llama.cpp realtime endpoint must be an absolute ws/wss URL without credentials, query or fragment")
		}
	}
	if o.MaxOutputTokens < 0 || o.MaxOutputTokens > math.MaxInt32 {
		return errors.New("llama.cpp realtime MaxOutputTokens must be between zero and 2147483647")
	}
	if o.PingInterval < 0 || o.PingTimeout < 0 {
		return errors.New("llama.cpp realtime keepalive durations must not be negative")
	}
	if o.PingInterval == 0 && o.PingTimeout != 0 {
		return errors.New("llama.cpp realtime PingTimeout requires PingInterval")
	}
	return nil
}

func acceptRealtimeTools(want, got ToolList) error {
	if len(got) != len(want) {
		return errors.New("realtime provider did not accept tool declarations")
	}
	for i := range want {
		if got[i].Name != want[i].Name || got[i].Type != ToolFunction || got[i].Description != want[i].Description {
			return errors.New("realtime provider did not accept tools")
		}
		a, err := parseSchema(got[i].Parameters)
		if err != nil {
			return errors.New("realtime provider acknowledged invalid tool schema")
		}
		b, err := parseSchema(want[i].Parameters)
		if err != nil || !reflect.DeepEqual(a, b) {
			return errors.New("realtime provider did not accept tool schemas")
		}
	}
	return nil
}

func nativeFormat(got AudioFormat) (genai.AudioFormat, error) {
	f := genai.AudioFormat{SampleRate: got.Rate}
	if got.Type != AudioPCM || f.Validate() != nil {
		return genai.AudioFormat{}, errors.New("llama.cpp realtime audio must be native PCM")
	}
	return f, nil
}

// Input sample rates the server resamples from, in tools/server/server-realtime.cpp.
const (
	minInputRate = 8000
	maxInputRate = 192000
)

const maxRealtimeTools = 1024
const maxRealtimeToolHistory = 4096

// maxMessageBytes bounds one inbound llama.cpp realtime wire message. llama.cpp
// does not document a WebSocket message limit.
const maxMessageBytes = 16 << 20

type realtimePendingTool struct {
	index      int
	name       string
	responseID string
}

// RealtimeSession owns a llama.cpp conversation. It duplicates the OpenAI-wire
// session because llama.cpp rejects commands with stricter ownership rules and
// different error paths.
//
// It implements the portable session contract over llama.cpp's /v1/realtime
// protocol. Provider-specific behavior:
//   - Turns are manual. Stream audio, commit it, then request generation. Server
//     VAD, voice selection, transcription, activity signals, audio stream end
//     and truncation are unsupported.
//   - Only native PCM16 mono input and output are accepted. The server
//     negotiates the sample rates at setup.
//   - session.update changes instructions and tools only before the first turn.
//     Tool choice, max output tokens and output modalities remain updatable.
//   - New user input requires an idle response and no pending tool calls.
//     Set Commit to commit audio without generating, or Generate to commit
//     nonempty audio and request the response. Commit and Generate together
//     commit buffered audio, then request the response. Empty text with
//     Generate=true requests a response from the current turn.
//   - RequireToolCall forces a tool call in the first response when the session
//     declares tools. Submit one complete result batch; call order is restored.
//     Tool results always request the continuation.
//   - Cancellation suppresses late output but does not roll back prefilled audio
//     or model history.
//   - There is no automatic history surgery. Context or output-limit exhaustion
//     can end the connection.
//   - SendRaw bypasses portable validation and state tracking.
//
// See [Client.ConnectRealtime] for LFM3 guidance and the protocol source.
type RealtimeSession struct {
	ws      *realtime.Conn
	info    genai.RealtimeSessionInfo
	pending []genai.RealtimeEvent
	names   map[string]struct{}
	sendMu  chan struct{}

	mu    sync.Mutex
	tools map[string]realtimePendingTool
	// History prevents replayed tool IDs from executing twice. Sessions are bounded
	// to maxRealtimeToolHistory calls; reconnecting is caller policy.
	history       map[string]struct{}
	current       string
	suppressed    map[string]struct{}
	cancelNext    bool
	cancelEventID string
	nextID        uint64
	// Manual turn ownership mirrors the ordered local control queue, not audio
	// buffering or speculative model output.
	audio             bool
	turn              bool
	results           bool
	generationPending bool
	started           bool
	requireToolCall   bool
}

func newRealtimeSession(ws *realtime.Conn, info *genai.RealtimeSessionInfo, names []string) *RealtimeSession {
	set := make(map[string]struct{}, len(names))
	for _, n := range names {
		set[n] = struct{}{}
	}
	return &RealtimeSession{ws: ws, info: *info, names: set, sendMu: make(chan struct{}, 1), tools: map[string]realtimePendingTool{}, history: map[string]struct{}{}, suppressed: map[string]struct{}{}}
}

// Info implements genai.RealtimeSession.
func (s *RealtimeSession) Info() genai.RealtimeSessionInfo { return s.info }

// Close implements io.Closer.
func (s *RealtimeSession) Close() error { return s.ws.Close() }

// SendRaw writes one complete client message verbatim. It bypasses portable
// validation and session state tracking. The caller owns the message and any
// resulting state changes. Use Send for portable commands.
func (s *RealtimeSession) SendRaw(ctx context.Context, message []byte) error {
	return realtime.WriteRaw(ctx, s.ws, s.sendMu, message)
}

// Send implements genai.RealtimeSession.
func (s *RealtimeSession) Send(ctx context.Context, in genai.RealtimeInput) error {
	if in == nil {
		return errors.New("realtime input is nil")
	}
	if err := in.Validate(); err != nil {
		return err
	}
	select {
	case s.sendMu <- struct{}{}:
		defer func() { <-s.sendMu }()
	case <-ctx.Done():
		return ctx.Err()
	case <-s.ws.Done():
		return s.ws.Err()
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	if err := s.ws.Err(); err != nil {
		return err
	}
	// Serialize state changes with event translation so cancellation and tool
	// consumption become visible before a concurrently received frame is handled.
	s.mu.Lock()
	defer s.mu.Unlock()
	if err := s.validateLocked(in); err != nil {
		return err
	}
	if u, ok := in.(genai.RealtimeUpdate); ok {
		upd, names, err := s.updateLocked(&u)
		if err != nil {
			return err
		}
		data, err := json.Marshal(upd)
		if err != nil {
			return fmt.Errorf("encode llama.cpp session update: %w", err)
		}
		if err := s.ws.Write(ctx, data); err != nil {
			return err
		}
		if names != nil {
			set := make(map[string]struct{}, len(names))
			for _, n := range names {
				set[n] = struct{}{}
			}
			s.names = set
			s.info.Capabilities.ToolResults = len(names) != 0 || len(s.tools) != 0
		}
		if u.ToolChoice != nil {
			s.requireToolCall = false
		}
		return nil
	}
	var msgs []ClientMessage
	var results []genai.ToolCallResult
	generate := false
	switch v := in.(type) {
	case genai.RealtimeInputContent:
		switch {
		case v.Audio != nil || v.Commit:
			if len(v.Audio) != 0 {
				msgs = append(msgs, ClientMessage{Type: EventAudioAppend, Audio: v.Audio})
			}
			if (v.Commit || (v.Audio != nil && v.Generate)) && s.info.TurnDetection == genai.RealtimeTurnManual {
				msgs = append(msgs, ClientMessage{Type: EventAudioCommit})
			}
		case v.ToolResults != nil:
			if len(v.ToolResults) > maxRealtimeTools {
				return errors.New("realtime tool batch exceeds limit")
			}
			var batch string
			for i := range v.ToolResults {
				r := &v.ToolResults[i]
				p, ok := s.tools[r.ID]
				if !ok || p.name != r.Name {
					return errors.New("realtime tool result does not match a pending call")
				}
				if batch != "" && batch != p.responseID {
					return errors.New("realtime tool results must complete one response batch")
				}
				batch = p.responseID
			}
			n := 0
			for _, p := range s.tools {
				if p.responseID == batch {
					n++
				}
			}
			if n != len(v.ToolResults) {
				return errors.New("realtime tool results must complete one response batch")
			}
			msgs = make([]ClientMessage, len(v.ToolResults), len(v.ToolResults)+1)
			for i := range v.ToolResults {
				r := &v.ToolResults[i]
				// The runtime appends results in arrival order. Render a complete batch
				// in call order so the model sees its original argument order.
				msgs[s.tools[r.ID].index] = ClientMessage{Type: EventItemCreate, Item: &Item{Type: ItemFunctionOutput, CallID: r.ID, Output: r.Result}}
			}
			results = v.ToolResults
		default:
			if v.Text != "" {
				msgs = append(msgs, ClientMessage{Type: EventItemCreate, Item: &Item{Type: ItemMessage, Role: RoleUser, Content: []RealtimeMessageContent{{Type: ContentInputText, Text: v.Text}}}})
			}
		}
		generate = v.Generate
		if v.ToolResults != nil {
			generate = true
		}
	case genai.RealtimeCancel:
		if v.ResponseID != "" && v.ResponseID != s.current {
			return errors.New("realtime cancel target must be the current observed response")
		}
		msg := ClientMessage{Type: EventCancel, ResponseID: v.ResponseID}
		if s.current == "" {
			msg.EventID = "genai_cancel_" + strconv.FormatUint(s.nextID, 10)
		}
		msgs = append(msgs, msg)
	default:
		return &base.ErrNotSupported{Options: []string{fmt.Sprintf("%T", in)}}
	}
	if generate {
		msgs = append(msgs, s.createMessageLocked())
	}
	// Materialize the complete transaction before the first wire effect. Separate
	// frames remain ordered; a partial write failure is terminal and never retried.
	frames := make([][]byte, len(msgs))
	for i := range msgs {
		b, err := json.Marshal(&msgs[i])
		if err != nil {
			return err
		}
		frames[i] = b
	}
	for _, b := range frames {
		if err := s.ws.Write(ctx, b); err != nil {
			return err
		}
	}
	s.sentLocked(in)
	for i := range results {
		delete(s.tools, results[i].ID)
	}
	if _, ok := in.(genai.RealtimeCancel); ok {
		if s.current != "" {
			s.suppressed[s.current] = struct{}{}
		} else {
			s.cancelNext = true
			s.cancelEventID = msgs[0].EventID
			s.nextID++
		}
	}
	return nil
}

func toolChoiceFromPortable(c genai.ToolCallRequest) ToolChoice {
	switch c {
	case genai.ToolCallAny:
		return ToolChoiceAuto
	case genai.ToolCallNone:
		return ToolChoiceNone
	case genai.ToolCallRequired:
		return ToolChoiceRequired
	default:
		return ""
	}
}

// Receive implements genai.RealtimeSession.
func (s *RealtimeSession) Receive(ctx context.Context) iter.Seq2[genai.RealtimeEvent, error] {
	return func(yield func(genai.RealtimeEvent, error) bool) {
		for {
			if err := ctx.Err(); err != nil {
				s.ws.Abort(err)
				yield(genai.RealtimeEvent{}, err)
				return
			}
			if err := s.ws.Err(); err != nil {
				yield(genai.RealtimeEvent{}, err)
				return
			}
			if len(s.pending) != 0 {
				e := s.pending[0]
				s.pending[0] = genai.RealtimeEvent{}
				s.pending = s.pending[1:]
				if !yield(e, nil) {
					return
				}
				continue
			}
			b, err := s.ws.Read(ctx)
			if err != nil {
				yield(genai.RealtimeEvent{}, err)
				return
			}
			var m ServerMessage
			if err := internal.UnmarshalJSON(b, &m); err != nil {
				err = fmt.Errorf("invalid realtime event payload: %w", err)
				s.ws.Abort(err)
				yield(genai.RealtimeEvent{}, err)
				return
			}
			s.mu.Lock()
			err = s.translateLocked(&m)
			s.mu.Unlock()
			if err != nil {
				s.ws.Abort(err)
				yield(genai.RealtimeEvent{}, err)
				return
			}
		}
	}
}

func (s *RealtimeSession) validateLocked(in genai.RealtimeInput) error {
	var generate bool
	switch v := in.(type) {
	case genai.RealtimeInputContent:
		switch {
		case v.Audio != nil || v.Commit:
			generate = v.Generate
			if v.Commit && s.info.TurnDetection == genai.RealtimeTurnManual && !s.info.Capabilities.AudioCommit {
				return &base.ErrNotSupported{Options: []string{"audio commit"}}
			}
			if v.Commit && s.info.TurnDetection == genai.RealtimeTurnManual && len(v.Audio) == 0 && !s.audio {
				return errors.New("llama.cpp cannot commit an empty audio buffer")
			}
			_, cancelled := s.suppressed[s.current]
			if (s.generationPending && !s.cancelNext) || (s.current != "" && !cancelled) {
				return errors.New("llama.cpp must cancel its active response before new user input")
			}
			if len(s.tools) != 0 {
				return errors.New("llama.cpp pending tool calls require results before another user turn")
			}
		case v.ToolResults != nil:
			if !s.info.Capabilities.ToolResults {
				return &base.ErrNotSupported{Options: []string{"tool results"}}
			}
			generate = true
			if s.turn {
				return errors.New("llama.cpp tool results cannot follow a new user turn")
			}
		case !v.Doc.IsZero():
			return &base.ErrNotSupported{Options: []string{"document input"}}
		default:
			generate = v.Generate
			if v.Text == "" && !s.turn && !s.results {
				return errors.New("llama.cpp empty text generation requires a committed user turn or tool results")
			}
			_, cancelled := s.suppressed[s.current]
			if (s.generationPending && !s.cancelNext) || (s.current != "" && !cancelled) {
				return errors.New("llama.cpp must cancel its active response before new user input")
			}
			if len(s.tools) != 0 {
				return errors.New("llama.cpp pending tool calls require results before another user turn")
			}
		}
	case genai.RealtimeCancel:
		if !s.info.Capabilities.CancelResponse {
			return &base.ErrNotSupported{Options: []string{"cancel"}}
		}
		if v.ResponseID == "" && s.current == "" && !s.generationPending {
			return ErrNoActiveResponse
		}
	case genai.RealtimeUpdate:
	case genai.RealtimeTruncate:
		if !s.info.Capabilities.Truncate {
			return &base.ErrNotSupported{Options: []string{"truncate"}}
		}
	default:
		return &base.ErrNotSupported{Options: []string{fmt.Sprintf("%T", in)}}
	}
	if generate {
		if s.generationPending || s.current != "" {
			return errors.New("llama.cpp response generation is already active")
		}
		if s.audio {
			c, ok := in.(genai.RealtimeInputContent)
			if !ok {
				return errors.New("llama.cpp audio requires commit before generation")
			}
			if !c.Commit && (c.Audio == nil || !c.Generate) {
				return errors.New("llama.cpp audio requires commit before generation")
			}
		}
	}
	return nil
}

func (s *RealtimeSession) sentLocked(in genai.RealtimeInput) {
	c, ok := in.(genai.RealtimeInputContent)
	if !ok {
		return
	}
	switch {
	case c.Audio != nil || c.Commit:
		s.audio = true
		s.turn = true
		s.started = true
		if (c.Commit || (c.Audio != nil && c.Generate)) && s.info.TurnDetection == genai.RealtimeTurnManual {
			s.audio = false
			if c.Generate {
				s.turn = false
				s.results = false
				s.generationPending = true
			}
		}
	case c.ToolResults != nil:
		s.turn = false
		s.audio = false
		s.results = false
		s.started = true
		s.generationPending = true
	default:
		s.turn = true
		s.started = true
		if c.Generate {
			s.turn = false
			s.audio = false
			s.results = false
			s.generationPending = true
		}
	}
}

func (s *RealtimeSession) translateLocked(m *ServerMessage) error {
	e := genai.RealtimeEvent{ResponseID: m.ResponseID, ItemID: m.ItemID, ContentIndex: m.ContentIndex}
	_, suppressed := s.suppressed[m.ResponseID]
	switch m.Type {
	case EventResponseCreated:
		if m.Response == nil || (m.Response.ID == "" || len(m.Response.ID) > 256) {
			return errors.New("invalid realtime response creation")
		}
		if s.current != "" {
			return errors.New("overlapping realtime responses are unsupported")
		}
		s.current = m.Response.ID
		if s.cancelNext {
			s.suppressed[s.current] = struct{}{}
			s.cancelNext = false
			s.cancelEventID = ""
		}
	case EventOutputAudioDelta:
		if suppressed {
			return nil
		}
		b, err := base64.StdEncoding.DecodeString(m.Delta)
		if err != nil || len(b) == 0 || len(b)%2 != 0 || m.ResponseID == "" || m.ItemID == "" {
			return errors.New("invalid realtime PCM audio")
		}
		e.Kind = genai.RealtimeEventContentAudio
		e.Content.Audio = b
		s.pending = append(s.pending, e)
	case EventOutputTextDelta, EventOutputTextDone, EventOutputTranscriptDelta, EventOutputTranscriptDone, EventInputTranscriptDelta, EventInputTranscriptDone:
		if m.Type != EventInputTranscriptDelta && m.Type != EventInputTranscriptDone && m.ResponseID == "" {
			return errors.New("realtime output transcript omitted response identity")
		}
		if suppressed {
			return nil
		}
		switch m.Type { //nolint:exhaustive // The outer switch admits only these six transcript event types.
		case EventOutputTextDelta:
			e.Kind = genai.RealtimeEventContentText
			e.Content.Text = m.Delta
		case EventOutputTextDone:
			e.Kind = genai.RealtimeEventContentText
			e.Content.Text = m.Text
			e.Content.Final = true
		case EventOutputTranscriptDelta:
			e.Kind = genai.RealtimeEventTranscriptOutput
			e.Content.Text = m.Delta
		case EventOutputTranscriptDone:
			e.Kind = genai.RealtimeEventTranscriptOutput
			e.Content.Text = m.Transcript
			e.Content.Final = true
		case EventInputTranscriptDelta:
			e.Kind = genai.RealtimeEventTranscriptInput
			e.Content.Text = m.Delta
		case EventInputTranscriptDone:
			e.Kind = genai.RealtimeEventTranscriptInput
			e.Content.Text = m.Transcript
			e.Content.Final = true
		}
		if m.ItemID == "" {
			return errors.New("realtime transcript omitted item identity")
		}
		s.pending = append(s.pending, e)
	case EventResponseDone:
		if m.Response == nil || (m.Response.ID == "" || len(m.Response.ID) > 256) {
			return errors.New("invalid realtime response completion")
		}
		r := m.Response
		e.ResponseID = r.ID
		_, cancelled := s.suppressed[r.ID]
		switch r.Status {
		case StatusCompleted:
			if !cancelled {
				if err := s.toolsLocked(r); err != nil {
					return err
				}
			}
		case StatusCancelled:
			e.Kind = genai.RealtimeEventInterrupted
			s.pending = append(s.pending, e)
		case StatusFailed, StatusIncomplete:
			e.Kind = genai.RealtimeEventError
			e.Error = &genai.RealtimeError{Code: string(r.Status), Message: "Realtime response did not complete"}
			s.pending = append(s.pending, e)
			e.Error = nil
		case StatusInProgress:
			return errors.New("realtime response.done has nonterminal status")
		default:
			return errors.New("unknown realtime response completion status")
		}
		if r.Usage != nil {
			u := r.Usage
			if u.InputTokens < 0 || u.OutputTokens < 0 || u.TotalTokens < 0 || u.InputDetails.CachedTokens < 0 {
				return errors.New("invalid realtime usage")
			}
			e.Kind = genai.RealtimeEventUsage
			e.Usage = &genai.Usage{InputTokens: u.InputTokens, OutputTokens: u.OutputTokens, TotalTokens: u.TotalTokens, InputCachedTokens: u.InputDetails.CachedTokens}
			s.pending = append(s.pending, e)
			e.Usage = nil
		}
		e.Kind = genai.RealtimeEventGenerationDone
		s.pending = append(s.pending, e)
		hasTools := false
		for i := range r.Output {
			if r.Output[i].Type == ItemFunctionCall {
				hasTools = true
				break
			}
		}
		if r.Status != StatusCompleted || cancelled || !hasTools {
			e.Kind = genai.RealtimeEventTurnDone
			s.pending = append(s.pending, e)
		}
		if s.current == r.ID {
			s.current = ""
		}
		delete(s.suppressed, r.ID)
	case EventFunctionCallArgsDelta:
		if suppressed {
			return nil
		}
		if m.ResponseID == "" || m.ItemID == "" {
			return errors.New("invalid realtime function call arguments delta")
		}
		e.Kind = genai.RealtimeEventProvider
		e.ProviderEvent = &FunctionCallArgsDelta{ResponseID: m.ResponseID, ItemID: m.ItemID, OutputIndex: m.OutputIndex, Delta: m.Delta}
		s.pending = append(s.pending, e)
	case EventFunctionCallArgsDone:
		if suppressed {
			return nil
		}
		if m.ResponseID == "" || m.CallID == "" || m.Name == "" || len(m.CallID) > 256 {
			return errors.New("invalid realtime function call arguments completion")
		}
		e.Kind = genai.RealtimeEventProvider
		e.ProviderEvent = &FunctionCallArgsDone{ResponseID: m.ResponseID, ItemID: m.ItemID, OutputIndex: m.OutputIndex, CallID: m.CallID, Name: m.Name, Arguments: m.Arguments}
		s.pending = append(s.pending, e)
	case EventError, EventInputTranscriptFailed:
		if m.Error == nil {
			return errors.New("realtime error event omitted error")
		}
		// Provider messages can echo user input or credentials. Preserve bounded
		// machine correlation, with a stable message rather than raw server text.
		if len(m.Error.Code) > 128 || len(m.Error.EventID) > 256 {
			return errors.New("realtime error correlation exceeds limit")
		}
		e.Kind = genai.RealtimeEventError
		e.Error = &genai.RealtimeError{Code: m.Error.Code, EventID: m.Error.EventID, Message: "Realtime provider rejected an operation"}
		s.pending = append(s.pending, e)
		if s.cancelEventID != "" && m.Error.EventID == s.cancelEventID {
			s.cancelNext = false
			s.cancelEventID = ""
		}
	default:
		// Unknown events do not authorize execution. Only response.done commits a
		// complete tool batch.
	}
	switch m.Type {
	case EventResponseCreated, EventResponseDone:
		s.generationPending = false
	case EventError:
		// The runtime does not correlate rejected event IDs. With no observed
		// response there is no generation to reserve or cancel.
		if s.current == "" {
			s.generationPending = false
			s.cancelNext = false
			s.cancelEventID = ""
		}
	default:
	}
	return nil
}

func (s *RealtimeSession) toolsLocked(r *Response) error {
	calls := make([]genai.ToolCall, 0)
	for i := range r.Output {
		v := &r.Output[i]
		if v.Type != ItemFunctionCall {
			continue
		}
		if len(calls) >= maxRealtimeTools || len(s.tools)+len(calls) >= maxRealtimeTools || len(s.history)+len(calls) >= maxRealtimeToolHistory {
			return errors.New("realtime tool correlation limit reached")
		}
		if (v.CallID == "" || len(v.CallID) > 256) || v.Name == "" || !json.Valid([]byte(v.Arguments)) {
			return errors.New("invalid realtime function call")
		}
		if _, ok := s.names[v.Name]; !ok {
			return errors.New("realtime function call names an undeclared tool")
		}
		if _, ok := s.history[v.CallID]; ok {
			return errors.New("realtime function call repeats an ID")
		}
		for j := range calls {
			if calls[j].ID == v.CallID {
				return errors.New("realtime function call repeats an ID")
			}
		}
		calls = append(calls, genai.ToolCall{ID: v.CallID, Name: v.Name, Arguments: v.Arguments})
	}
	if len(calls) != 0 {
		for i := range calls {
			v := &calls[i]
			s.tools[v.ID] = realtimePendingTool{index: i, name: v.Name, responseID: r.ID}
			s.history[v.ID] = struct{}{}
		}
		s.pending = append(s.pending, genai.RealtimeEvent{Kind: genai.RealtimeEventContentToolCalls, ResponseID: r.ID, Content: genai.RealtimeContent{ToolCalls: calls}})
	}
	return nil
}

var _ genai.RealtimeSession = (*RealtimeSession)(nil)

func (s *RealtimeSession) updateLocked(u *genai.RealtimeUpdate) (*SessionUpdateMessage, []string, error) {
	var options []string
	c := &s.info.Capabilities.Update
	if u.Instructions != nil && !c.Instructions {
		options = append(options, "instructions")
	}
	if u.Tools != nil && !c.Tools {
		options = append(options, "tools")
	}
	if u.ToolChoice != nil {
		if *u.ToolChoice == genai.ToolCallRequired {
			if !c.ToolChoiceRequired {
				options = append(options, "required tool choice")
			}
		} else if !c.ToolChoice {
			options = append(options, "tool choice")
		}
	}
	if u.MaxOutputTokens != nil && !c.MaxOutputTokens {
		options = append(options, "max output tokens")
	}
	if u.OutputModalities != nil && !c.OutputModalities {
		options = append(options, "output modalities")
	}
	if u.TurnDetection != nil && !c.TurnDetection {
		options = append(options, "turn detection")
	}
	if u.Voice != nil && !c.VoiceBeforeOutput {
		options = append(options, "voice")
	}
	if len(options) != 0 {
		return nil, nil, &base.ErrNotSupported{Options: options}
	}
	upd := &SessionUpdate{}
	var names []string
	if u.Instructions != nil {
		if s.started {
			return nil, nil, errors.New("llama.cpp session instructions can only change before the first turn")
		}
		upd.Instructions = u.Instructions
	}
	if u.Tools != nil {
		if s.started {
			return nil, nil, errors.New("llama.cpp session tools can only change before the first turn")
		}
		if err := upd.Tools.Init(*u.Tools); err != nil {
			return nil, nil, err
		}
		names = upd.Tools.Names()
	}
	if u.ToolChoice != nil {
		upd.ToolChoice = toolChoiceFromPortable(*u.ToolChoice)
	}
	if u.MaxOutputTokens != nil {
		if *u.MaxOutputTokens > math.MaxInt32 {
			return nil, nil, errors.New("llama.cpp max output tokens exceeds 2147483647")
		}
		upd.MaxOutputTokens = MaxOutputTokens{Value: *u.MaxOutputTokens}
	}
	if u.OutputModalities != nil {
		if s.info.OutputFormat == (genai.AudioFormat{}) && slices.Contains(*u.OutputModalities, genai.ModalityAudio) {
			return nil, nil, &base.ErrNotSupported{Options: []string{"audio output"}}
		}
		upd.OutputModalities = Modalities(*u.OutputModalities)
	}
	if u.Voice != nil {
		if s.started {
			return nil, nil, errors.New("llama.cpp session voice can only change before the first turn")
		}
		upd.Audio.Output.Voice = *u.Voice
	}
	return &SessionUpdateMessage{Type: EventSessionUpdate, Session: upd}, names, nil
}

// createMessageLocked returns response.create, optionally forcing the first tool
// call when the session declares tools.
func (s *RealtimeSession) createMessageLocked() ClientMessage {
	msg := ClientMessage{Type: EventCreate}
	if s.requireToolCall && len(s.names) != 0 && len(s.history) == 0 {
		msg.Response = &ResponseConfig{ToolChoice: ToolChoiceRequired}
	}
	return msg
}
