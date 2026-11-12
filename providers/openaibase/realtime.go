// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// OpenAI GA realtime session engine and correlation.
//
// It uses the protocol described at
// https://developers.openai.com/api/reference/resources/realtime/client-events.
// Provider adapters own authentication and model selection; [ConnectRealtimeSession]
// owns setup and options.

package openaibase

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"iter"
	"math"
	"math/big"
	"net/http"
	"reflect"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/maruel/genai"
	"github.com/maruel/genai/base"
	"github.com/maruel/genai/internal"
	"github.com/maruel/genai/internal/realtime"
)

// RealtimeVoices returns OpenAI's documented built-in realtime voice names.
//
// https://developers.openai.com/api/docs/guides/realtime-conversations#voice-options
func RealtimeVoices() []string {
	return []string{"alloy", "ash", "ballad", "cedar", "coral", "echo", "marin", "sage", "shimmer", "verse"}
}

const maxTools = 1024
const maxToolHistory = 4096

// maxDocBytes bounds one inline document; base64 inflates the bound by one third
// on the wire.
const maxDocBytes = 16 << 20

// maxMessageBytes bounds one inbound OpenAI realtime wire message.
const maxMessageBytes = 16 << 20

// Session owns an OpenAI GA realtime conversation and its WebSocket connection.
//
// It implements the portable session contract for the OpenAI /v1/realtime
// protocol. Provider adapters own the endpoint, authentication, model selection
// and setup; use [ConnectRealtime] instead of constructing a session.
//
// OpenAI-specific behavior:
//   - Server VAD and manual turns are both supported. Manual turns send audio,
//     commit it, then request generation.
//   - session.update accepts instructions, tools, tool choice, max output
//     tokens, output modalities, turn detection and voice. Voice changes are
//     rejected after the first audio output.
//   - Truncate is available only while the output modality is audio. Playback
//     remains application-owned.
//   - Input and output are mono PCM16 at 24 kHz.
//   - Document input is one user message carrying the base64 data URL of the
//     document and the text that follows it, bounded to 16 MiB.
//   - A completed response authorizes one complete tool-call batch. Sessions
//     retain at most 1024 pending calls and 4096 historical call IDs.
//   - Cancellation suppresses late audio and transcripts for the cancelled
//     response.
//   - Usage events replace the latest non-cumulative snapshot.
//     rate_limits.updated replaces the latest rate-limit snapshot.
//   - SendRaw bypasses portable validation and state tracking.
//
// See [ConnectRealtimeSession] for setup options and the protocol source.
type Session struct {
	ws   *realtime.Conn
	info genai.RealtimeSessionInfo
	// pending is receiver-owned and holds only events from one bounded frame.
	pending []genai.RealtimeEvent
	names   map[string]struct{}
	sendMu  chan struct{}

	mu    sync.Mutex
	tools map[string]pendingTool
	// History prevents replayed tool IDs from executing twice. Sessions are bounded
	// to maxToolHistory calls; reconnecting is caller policy, never implicit replay.
	history       map[string]struct{}
	current       string
	suppressed    map[string]struct{}
	cancelNext    bool
	cancelEventID string
	nextID        uint64
	audioOutput   bool
}

// newSession wraps an accepted OpenAI-wire connection. info must describe the
// negotiated session.
func newSession(ws *realtime.Conn, info *genai.RealtimeSessionInfo, names []string) *Session {
	set := make(map[string]struct{}, len(names))
	for _, n := range names {
		set[n] = struct{}{}
	}
	return &Session{ws: ws, info: *info, names: set, sendMu: make(chan struct{}, 1), tools: map[string]pendingTool{}, history: map[string]struct{}{}, suppressed: map[string]struct{}{}}
}

// ConnectRealtime opens an OpenAI-wire session, sends setup and waits for the
// provider acknowledgement. The caller owns endpoint and keepalive selection.
func ConnectRealtime(ctx context.Context, hc *http.Client, endpoint string, pingInterval, pingTimeout time.Duration, setup *SessionConfig) (*Session, error) {
	if setup == nil {
		return nil, errors.New("realtime setup is required")
	}
	data, err := json.Marshal(ClientMessage{Type: EventSessionUpdate, Session: setup})
	if err != nil {
		return nil, fmt.Errorf("encode realtime setup: %w", err)
	}
	ws, err := realtime.Dial(ctx, endpoint, hc, maxMessageBytes, pingInterval, pingTimeout)
	if err != nil {
		return nil, err
	}
	// Setup has no asynchronous user events: require created followed by updated.
	// Unknown provider extensions are ignored within a bounded handshake budget.
	created := false
	info := genai.RealtimeSessionInfo{}
	for range 32 {
		b, e := ws.Read(ctx)
		if e != nil {
			return nil, errors.Join(e, ws.Close())
		}
		var m ServerMessage
		if e := internal.UnmarshalJSON(b, &m); e != nil {
			return nil, errors.Join(fmt.Errorf("invalid realtime setup payload: %w", e), ws.Close())
		}
		switch m.Type {
		case EventSessionCreated:
			if created || m.Session == nil || (m.Session.ID == "" || len(m.Session.ID) > 256) || m.Session.Type != SessionRealtime {
				return nil, errors.Join(errors.New("invalid realtime session creation"), ws.Close())
			}
			created = true
			info.ID = m.Session.ID
			if e := ws.Write(ctx, data); e != nil {
				return nil, errors.Join(e, ws.Close())
			}
		case EventSessionUpdated:
			if !created || m.Session == nil {
				return nil, errors.Join(errors.New("realtime setup acknowledgement missing"), ws.Close())
			}
			if e := acceptSetup(setup, m.Session, &info); e != nil {
				return nil, errors.Join(e, ws.Close())
			}
			return newSession(ws, &info, setup.Tools.Names()), nil
		case EventError:
			return nil, errors.Join(errors.New("realtime provider rejected setup"), ws.Close())
		default:
		}
	}
	return nil, errors.Join(errors.New("realtime setup exceeded event limit"), ws.Close())
}

// Info implements genai.RealtimeSession.
func (s *Session) Info() genai.RealtimeSessionInfo { return s.info }

// Close implements io.Closer.
func (s *Session) Close() error { return s.ws.Close() }

// SendRaw writes one complete client message verbatim. It bypasses portable
// validation and session state tracking. The caller owns the message and any
// resulting state changes. Use Send for portable commands.
func (s *Session) SendRaw(ctx context.Context, message []byte) error {
	return realtime.WriteRaw(ctx, s.ws, s.sendMu, message)
}

// Send implements genai.RealtimeSession.
func (s *Session) Send(ctx context.Context, in genai.RealtimeInput) error {
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
	// Document encoding reads and base64-encodes caller data before taking the
	// state lock that event translation shares.
	var content []Content
	if v, ok := in.(genai.RealtimeInputContent); ok && !v.Doc.IsZero() {
		var err error
		if content, err = docContent(&v.Doc, v.Text); err != nil {
			return err
		}
	}
	// Serialize state changes with event translation so cancellation and tool
	// consumption become visible before a concurrently received frame is handled.
	s.mu.Lock()
	defer s.mu.Unlock()
	if u, ok := in.(genai.RealtimeUpdate); ok {
		upd, names, err := s.updateLocked(&u)
		if err != nil {
			return err
		}
		data, err := json.Marshal(upd)
		if err != nil {
			return fmt.Errorf("encode realtime session update: %w", err)
		}
		if err := s.ws.Write(ctx, data); err != nil {
			return err
		}
		if names != nil {
			s.names = names
			s.info.Capabilities.ToolResults = len(names) != 0 || len(s.tools) != 0
		}
		if u.OutputModalities != nil {
			s.info.Capabilities.Truncate = (*u.OutputModalities)[0] == genai.ModalityAudio
		}
		if u.TurnDetection != nil {
			s.info.TurnDetection = *u.TurnDetection
			s.info.Capabilities.AudioCommit = *u.TurnDetection == genai.RealtimeTurnManual
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
			if v.Commit && s.info.TurnDetection != genai.RealtimeTurnManual {
				return &base.ErrNotSupported{Options: []string{"audio commit with server turn detection"}}
			}
			if (v.Commit || (v.Audio != nil && v.Generate)) && s.info.TurnDetection == genai.RealtimeTurnManual {
				if !s.info.Capabilities.AudioCommit {
					return &base.ErrNotSupported{Options: []string{"audio commit"}}
				}
				msgs = append(msgs, ClientMessage{Type: EventAudioCommit})
			}
		case v.ToolResults != nil:
			if len(v.ToolResults) > maxTools {
				return errors.New("realtime tool batch exceeds limit")
			}
			if !s.info.Capabilities.ToolResults {
				return &base.ErrNotSupported{Options: []string{"tool results"}}
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
				// Render a complete batch in call order so runtimes that append results
				// without resolving call IDs keep the model's original argument order.
				msgs[s.tools[r.ID].index] = ClientMessage{Type: EventItemCreate, Item: &Item{Type: ItemFunctionOutput, CallID: r.ID, Output: r.Result}}
			}
			results = v.ToolResults
		case !v.Doc.IsZero():
			msgs = append(msgs, ClientMessage{Type: EventItemCreate, Item: &Item{Type: ItemMessage, Role: RoleUser, Content: content}})
		default:
			if v.Text != "" {
				msgs = append(msgs, ClientMessage{Type: EventItemCreate, Item: &Item{Type: ItemMessage, Role: RoleUser, Content: []Content{{Type: ContentInputText, Text: v.Text}}}})
			}
		}
		generate = v.Generate
		if v.ToolResults != nil {
			generate = true
		}
	case genai.RealtimeCancel:
		if !s.info.Capabilities.CancelResponse {
			return &base.ErrNotSupported{Options: []string{"cancel"}}
		}
		if v.ResponseID != "" && v.ResponseID != s.current {
			return errors.New("realtime cancel target must be the current observed response")
		}
		msg := ClientMessage{Type: EventCancel, ResponseID: v.ResponseID}
		if s.current == "" {
			msg.EventID = "genai_cancel_" + strconv.FormatUint(s.nextID, 10)
		}
		msgs = append(msgs, msg)
	case genai.RealtimeTruncate:
		if !s.info.Capabilities.Truncate {
			return &base.ErrNotSupported{Options: []string{"truncate"}}
		}
		if v.ContentIndex != 0 {
			return errors.New("OpenAI audio truncation requires content index zero")
		}
		msgs = append(msgs, ClientMessage{Type: EventTruncate, ItemID: v.ItemID, ContentIndex: &v.ContentIndex, AudioEndMS: &v.AudioEndMS})
	default:
		return &base.ErrNotSupported{Options: []string{fmt.Sprintf("%T", in)}}
	}
	if generate {
		msgs = append(msgs, ClientMessage{Type: EventCreate})
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

// docContent renders one user message carrying the document and the text that
// follows it.
func docContent(d *genai.Doc, text string) ([]Content, error) {
	if d.URL != "" {
		return nil, &base.ErrNotSupported{Options: []string{"remote documents"}}
	}
	mimeType, data, err := d.Read(maxDocBytes)
	if err != nil {
		return nil, fmt.Errorf("read document: %w", err)
	}
	content := make([]Content, 1, 2)
	content[0] = Content{Type: ContentInputImage, ImageURL: "data:" + mimeType + ";base64," + base64.StdEncoding.EncodeToString(data)}
	if text != "" {
		content = append(content, Content{Type: ContentInputText, Text: text})
	}
	return content, nil
}

func toolChoiceFromPortable(c genai.ToolCallRequest) ToolChoice {
	switch c {
	case genai.ToolCallAny:
		return ToolChoice{Mode: ToolChoiceAuto}
	case genai.ToolCallNone:
		return ToolChoice{Mode: ToolChoiceNone}
	case genai.ToolCallRequired:
		return ToolChoice{Mode: ToolChoiceRequired}
	default:
		return ToolChoice{}
	}
}

// Receive implements genai.RealtimeSession.
func (s *Session) Receive(ctx context.Context) iter.Seq2[genai.RealtimeEvent, error] {
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

type pendingTool struct {
	index      int
	name       string
	responseID string
}

func (s *Session) translateLocked(m *ServerMessage) error {
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
		s.audioOutput = true
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
	case EventSpeechStarted:
		if s.info.TurnDetection != genai.RealtimeTurnServer {
			return nil
		}
		if s.current != "" {
			s.suppressed[s.current] = struct{}{}
		}
		e.Kind = genai.RealtimeEventInputSpeechStart
		s.pending = append(s.pending, e)
	case EventSpeechStopped:
		if s.info.TurnDetection != genai.RealtimeTurnServer {
			return nil
		}
		e.Kind = genai.RealtimeEventInputSpeechEnd
		s.pending = append(s.pending, e)
	case EventRateLimits:
		if len(m.RateLimits) == 0 {
			return errors.New("realtime rate limits event is empty")
		}
		limits := make([]genai.RateLimit, 0, len(m.RateLimits))
		now := time.Now()
		for _, r := range m.RateLimits {
			if r.Limit <= 0 || r.Remaining < 0 || r.ResetSeconds < 0 || math.IsNaN(r.ResetSeconds) || math.IsInf(r.ResetSeconds, 0) {
				return errors.New("invalid realtime rate limit")
			}
			var typ genai.RateLimitType
			switch r.Name {
			case "requests":
				typ = genai.Requests
			case "tokens":
				typ = genai.Tokens
			default:
				return fmt.Errorf("unknown realtime rate limit %q", r.Name)
			}
			limits = append(limits, genai.RateLimit{
				Type:      typ,
				Period:    genai.PerOther,
				Limit:     r.Limit,
				Remaining: r.Remaining,
				Reset:     now.Add(time.Duration(r.ResetSeconds * float64(time.Second))).Round(10 * time.Millisecond),
			})
		}
		e.Kind = genai.RealtimeEventUsage
		e.Usage = &genai.Usage{Limits: limits}
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
			if u.InputTokens < 0 || u.OutputTokens < 0 || u.TotalTokens < 0 || u.InputDetails.CachedTokens < 0 || u.OutputDetails.ReasoningTokens < 0 {
				return errors.New("invalid realtime usage")
			}
			e.Kind = genai.RealtimeEventUsage
			e.Usage = &genai.Usage{InputTokens: u.InputTokens, InputCachedTokens: u.InputDetails.CachedTokens, ReasoningTokens: u.OutputDetails.ReasoningTokens, OutputTokens: u.OutputTokens, TotalTokens: u.TotalTokens}
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
	return nil
}

func (s *Session) toolsLocked(r *Response) error {
	calls := make([]genai.ToolCall, 0)
	for i := range r.Output {
		v := &r.Output[i]
		if v.Type != ItemFunctionCall {
			continue
		}
		if len(calls) >= maxTools || len(s.tools)+len(calls) >= maxTools || len(s.history)+len(calls) >= maxToolHistory {
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
			s.tools[v.ID] = pendingTool{index: i, name: v.Name, responseID: r.ID}
			s.history[v.ID] = struct{}{}
		}
		s.pending = append(s.pending, genai.RealtimeEvent{Kind: genai.RealtimeEventContentToolCalls, ResponseID: r.ID, Content: genai.RealtimeContent{ToolCalls: calls}})
	}
	return nil
}

func acceptSetup(want, got *SessionConfig, info *genai.RealtimeSessionInfo) error {
	if got.ID != info.ID || got.Type != SessionRealtime {
		return errors.New("realtime acknowledgement changed session identity")
	}
	if got.Audio.Input.Format != want.Audio.Input.Format || got.Audio.Output.Format != want.Audio.Output.Format {
		return errors.New("realtime provider did not accept PCM24 audio formats")
	}
	if len(got.OutputModalities) != 1 || got.OutputModalities[0] != want.OutputModalities[0] {
		return errors.New("realtime provider did not accept output modality")
	}
	if want.Audio.Input.TurnDetection == nil {
		if got.Audio.Input.TurnDetection != nil {
			return errors.New("realtime provider did not disable turn detection")
		}
		info.TurnDetection = genai.RealtimeTurnManual
	} else {
		v := got.Audio.Input.TurnDetection
		if v == nil || v.Type != TurnServerVAD || !v.CreateResponse || !v.InterruptResponse {
			return errors.New("realtime provider did not accept server turns")
		}
		info.TurnDetection = genai.RealtimeTurnServer
	}
	if (want.Audio.Input.Transcription == nil) != (got.Audio.Input.Transcription == nil) {
		return errors.New("realtime provider did not accept transcription enablement")
	}
	if want.Audio.Input.Transcription != nil && (got.Audio.Input.Transcription.Model != want.Audio.Input.Transcription.Model || got.Audio.Input.Transcription.Language != want.Audio.Input.Transcription.Language) {
		return errors.New("realtime provider did not accept input transcription")
	}
	if want.MaxOutputTokens != (MaxOutputTokens{}) && want.MaxOutputTokens != got.MaxOutputTokens {
		return errors.New("realtime provider did not accept output token limit")
	}
	if w := want.Truncation; w != nil {
		g := got.Truncation
		if g == nil || g.Strategy != w.Strategy || g.Ratio != w.Ratio || (w.PostInstructionsTokens != 0 && g.PostInstructionsTokens != w.PostInstructionsTokens) {
			return errors.New("realtime provider did not accept truncation")
		}
	}
	if want.Reasoning != nil && (got.Reasoning == nil || got.Reasoning.Effort != want.Reasoning.Effort) {
		return errors.New("realtime provider did not accept reasoning effort")
	}
	if want.Audio.Output.Voice != (Voice{}) && want.Audio.Output.Voice != got.Audio.Output.Voice {
		return errors.New("realtime provider did not accept voice")
	}
	if got.Instructions != want.Instructions {
		return errors.New("realtime provider did not accept instructions")
	}
	if err := acceptTools(want.Tools, got.Tools); err != nil {
		return err
	}
	f := genai.AudioFormat{SampleRate: 24000}
	info.InputFormat = f
	info.OutputFormat = f
	info.InputModalities = []genai.Modality{genai.ModalityAudio, genai.ModalityImage, genai.ModalityText}
	info.Capabilities = genai.RealtimeCapabilities{
		AudioCommit:    info.TurnDetection == genai.RealtimeTurnManual,
		CancelResponse: true,
		ToolResults:    len(want.Tools) != 0,
		Truncate:       got.OutputModalities[0] == genai.ModalityAudio,
		Update: genai.RealtimeUpdateCapabilities{
			Instructions:       true,
			Tools:              true,
			ToolChoice:         true,
			ToolChoiceRequired: true,
			MaxOutputTokens:    true,
			OutputModalities:   true,
			TurnDetection:      true,
			VoiceBeforeOutput:  true,
		},
	}
	return nil
}

func acceptTools(want, got []Tool) error {
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

// decimalNumber is a canonical decimal value, independent of JSON spelling.
// Keeping its exponent as text avoids expanding a short number like 1e999999999.
type decimalNumber struct {
	negative    bool
	coefficient string
	exponent    string
}

func parseSchema(b []byte) (any, error) {
	d := json.NewDecoder(bytes.NewReader(b))
	d.UseNumber()
	var v any
	if err := d.Decode(&v); err != nil {
		return nil, err
	}
	return normalizeSchema(v)
}

func normalizeSchema(v any) (any, error) {
	switch x := v.(type) {
	case json.Number:
		s := string(x)
		negative := strings.HasPrefix(s, "-")
		s = strings.TrimPrefix(s, "-")
		exp := "0"
		if i := strings.IndexAny(s, "eE"); i >= 0 {
			s, exp = s[:i], s[i+1:]
		}
		integer, fraction, _ := strings.Cut(s, ".")
		coefficient := strings.TrimLeft(integer+fraction, "0")
		trimmed := strings.TrimRight(coefficient, "0")
		if trimmed == "" {
			return decimalNumber{}, nil
		}
		var exponent big.Int
		if _, ok := exponent.SetString(exp, 10); !ok {
			return nil, errors.New("invalid JSON number exponent")
		}
		shift := len(coefficient) - len(trimmed) - len(fraction)
		exponent.Add(&exponent, big.NewInt(int64(shift)))
		return decimalNumber{negative: negative, coefficient: trimmed, exponent: exponent.String()}, nil
	case []any:
		for i := range x {
			n, err := normalizeSchema(x[i])
			if err != nil {
				return nil, err
			}
			x[i] = n
		}
	case map[string]any:
		for k, v := range x {
			n, err := normalizeSchema(v)
			if err != nil {
				return nil, err
			}
			x[k] = n
		}
	default:
	}
	return v, nil
}

func (s *Session) updateLocked(u *genai.RealtimeUpdate) (*SessionUpdateMessage, map[string]struct{}, error) {
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
	var names map[string]struct{}
	if u.Instructions != nil {
		upd.Instructions = u.Instructions
	}
	if u.Tools != nil {
		var tools ToolList
		if err := tools.Init(*u.Tools); err != nil {
			return nil, nil, err
		}
		upd.Tools = tools
		names = make(map[string]struct{}, len(tools))
		for _, n := range tools.Names() {
			names[n] = struct{}{}
		}
	}
	if u.ToolChoice != nil {
		upd.ToolChoice = toolChoiceFromPortable(*u.ToolChoice)
	}
	if u.MaxOutputTokens != nil {
		upd.MaxOutputTokens = MaxOutputTokens{Value: *u.MaxOutputTokens}
	}
	if u.OutputModalities != nil {
		if len(*u.OutputModalities) != 1 {
			return nil, nil, errors.New("OpenAI realtime accepts one output modality")
		}
		upd.OutputModalities = Modalities(*u.OutputModalities)
	}
	if u.TurnDetection != nil {
		upd.Audio.Input.TurnDetection.Set = true
		if *u.TurnDetection == genai.RealtimeTurnServer {
			upd.Audio.Input.TurnDetection.Value = &TurnDetection{Type: TurnServerVAD, CreateResponse: true, InterruptResponse: true}
		}
	}
	if u.Voice != nil {
		if s.audioOutput {
			return nil, nil, errors.New("OpenAI realtime voice cannot change after audio output")
		}
		upd.Audio.Output.Voice = &Voice{Name: *u.Voice}
	}
	return &SessionUpdateMessage{Type: EventSessionUpdate, Session: upd}, names, nil
}
