// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Gemini Live sessions and normalized realtime event translation.

package gemini

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"iter"
	"maps"
	"net/url"
	"slices"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/maruel/genai"
	"github.com/maruel/genai/base"
	"github.com/maruel/genai/internal"
	"github.com/maruel/genai/internal/realtime"
)

// maxMessageBytes bounds one inbound Gemini Live wire message and transcript.
// Gemini does not document a WebSocket message limit.
const maxMessageBytes = 16 << 20

// maxDocBytes bounds one inline document.
const maxDocBytes = 16 << 20

// RealtimeOption configures Gemini Live features beyond the portable contract.
// Affective dialogue and proactivity fields are deliberately omitted.
type RealtimeOption struct {
	// Endpoint replaces the Live WebSocket URL, for a trusted provider proxy.
	// The client's HTTP authentication transport is used for this endpoint too.
	Endpoint string
	// InputLanguages supplies BCP-47 transcription hints; output language belongs
	// in the caller's system instruction, not speechConfig.languageCode.
	InputLanguages []string
	// SessionResumption requests resumable-state events. ResumeHandle restores a
	// previous session and implies SessionResumption. Pass the latest nonempty
	// Handle from a *Resumption provider event. No replay is automatic.
	SessionResumption bool
	ResumeHandle      string
	// Compression enables Gemini's sliding-window context compression.
	Compression *RealtimeCompression
	// PingInterval and PingTimeout control idle transport health. Zero selects
	// 30 seconds and 10 seconds respectively; values must not be negative.
	PingInterval time.Duration
	PingTimeout  time.Duration
}

// Validate checks Gemini-specific options before connection establishment.
func (o *RealtimeOption) Validate() error {
	if o.Endpoint != "" {
		u, err := url.Parse(o.Endpoint)
		if err != nil || (u.Scheme != "ws" && u.Scheme != "wss") || u.Host == "" || u.User != nil || u.RawQuery != "" || u.Fragment != "" {
			return errors.New("gemini live endpoint must be an absolute ws/wss URL without credentials, query or fragment")
		}
	}
	if slices.Contains(o.InputLanguages, "") {
		return errors.New("gemini live input language hints must not be empty")
	}
	if o.PingInterval < 0 || o.PingTimeout < 0 {
		return errors.New("gemini live keepalive durations must not be negative")
	}
	if c := o.Compression; c != nil {
		if c.TriggerTokens < 0 || c.TargetTokens < 0 {
			return errors.New("gemini live compression token counts must not be negative")
		}
		if c.TriggerTokens != 0 && c.TargetTokens >= c.TriggerTokens {
			return errors.New("gemini live compression target tokens must be below trigger tokens")
		}
	}
	return nil
}

// RealtimeCompression configures Gemini's sliding-window context reduction.
//
// Compression lifts the 15-minute limit on audio-only sessions. A zero
// TriggerTokens selects 80% of the model's context window; a zero TargetTokens
// selects half the trigger. An explicit TargetTokens must be below an explicit
// TriggerTokens.
//
// Documented at https://ai.google.dev/api/live#ContextWindowCompressionConfig
type RealtimeCompression struct {
	TriggerTokens int64
	TargetTokens  int64
}

// RealtimeSession is the concrete Gemini Live session returned by
// ConnectRealtime.
//
// It implements the portable session contract over BidiGenerateContent.
// Provider-specific behavior:
//   - Server VAD is the default. Manual turns require ActivityStart before audio
//     and ActivityEnd after it. AudioEnd pauses a server audio stream.
//   - Native-audio models output audio only. Transcripts arrive as events and do
//     not make text an output modality.
//   - ContextUpdate is preview. It updates instructions and tools. Tool choice
//     none clears declarations; auto restores the setup declarations; required
//     is rejected. Output modalities, token limits, voice and turn detection are
//     connection-scoped.
//   - Explicit cancellation, audio commit, explicit generation and truncation
//     are unsupported.
//   - Tool results resume the provider automatically. Cancelled calls are
//     reported as provider events, and a batch containing one fails with
//     [genai.RealtimeToolCallCancelledError].
//   - Extended-thinking models require non-blocking calls and may emit filler
//     turns before the tool call.
//   - Usage snapshots cover one response. Interaction status, resumption and
//     go-away are provider events.
//   - Response IDs are adapter-local and span model output through turnComplete.
//     Output transcription before observed model output, or after turnComplete,
//     has no response identity. Only usage on turnComplete without tool calls
//     is correlated; standalone, intermediate and tool-turn usage stays unowned
//     because the wire protocol does not identify its response or snapshot scope.
//     Interrupted turns retain their identity through their final turnComplete.
//   - Input is PCM16 at 16 kHz. Output is PCM16 at 24 kHz.
//   - Document input is a user turn whose parts carry the inline document followed
//     by the text, bounded to 16 MiB.
//   - SendRaw bypasses portable validation and state tracking.
//
// See [Client.ConnectRealtime] for setup options and the protocol source.
type RealtimeSession struct {
	ws   *realtime.Conn
	info genai.RealtimeSessionInfo
	// pending holds events from one bounded wire message and is receiver-owned.
	pending []genai.RealtimeEvent
	// model, liveTools and toolChoice are owned by Send under sendMu. liveTools
	// are the declarations that toolChoice auto restores after a none update.
	model      string
	liveTools  LiveTools
	toolChoice genai.ToolCallRequest
	// Transcript builders are receiver-owned and bounded by the message limit.
	inputTranscript  strings.Builder
	outputTranscript strings.Builder
	// Response correlation is receiver-owned. Gemini supplies no wire response
	// IDs (go-genai types.go at 89c10e9d4b39f956d7d145a9086c70b94713d725).
	// Model output establishes a turn; turnComplete closes its identity.
	responseID     string
	responseTools  bool
	nextResponseID uint64

	sendMu chan struct{}
	// activity is owned by Send, under sendMu.
	activity bool

	toolMu sync.Mutex
	tools  map[string]realtimePendingTool
	// cancelled retains exact correlation for late results. IDs cannot be reused
	// within this session; exhaustion fails closed rather than losing provenance.
	cancelled      map[string]string
	cancelledBytes int
	nextID         uint64
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
	if err := s.ws.Err(); err != nil {
		return err
	}
	if u, ok := in.(genai.RealtimeUpdate); ok {
		update, tools, choice, err := s.updateMessage(&u)
		if err != nil {
			return err
		}
		data, err := json.Marshal(LiveClientMessage{ContextUpdate: update})
		if err != nil {
			return fmt.Errorf("encode Gemini Live context update: %w", err)
		}
		if err := s.ws.Write(ctx, data); err != nil {
			return err
		}
		s.liveTools, s.toolChoice = tools, choice
		if u.Tools != nil {
			s.info.Capabilities.ToolResults = len(tools) != 0 || len(s.tools) != 0
		}
		return nil
	}
	var msgs []LiveClientMessage
	var results []genai.ToolCallResult
	started, ended := false, false
	switch v := in.(type) {
	case genai.RealtimeInputContent:
		switch {
		case v.Audio != nil || v.Commit:
			if s.info.TurnDetection == genai.RealtimeTurnManual && !s.activity {
				if len(v.Audio) == 0 && v.Commit {
					return errors.New("gemini live cannot end an empty activity")
				}
				msgs = append(msgs, LiveClientMessage{RealtimeInput: &LiveRealtimeInput{ActivityStart: &LiveActivity{}}})
				started = true
			}
			if len(v.Audio) != 0 {
				msgs = append(msgs, LiveClientMessage{RealtimeInput: &LiveRealtimeInput{Audio: &LiveBlob{MIMEType: "audio/pcm;rate=" + strconv.Itoa(s.info.InputFormat.SampleRate), Data: v.Audio}}})
			}
			if v.Commit || (v.Audio != nil && v.Generate) {
				if s.info.TurnDetection == genai.RealtimeTurnManual {
					// Gemini responds automatically after activityEnd. Generate is advisory.
					msgs = append(msgs, LiveClientMessage{RealtimeInput: &LiveRealtimeInput{ActivityEnd: &LiveActivity{}}})
					ended = true
				} else {
					msgs = append(msgs, LiveClientMessage{RealtimeInput: &LiveRealtimeInput{AudioStreamEnd: true}})
				}
			}
		case v.ToolResults != nil:
			if !s.info.Capabilities.ToolResults {
				return &base.ErrNotSupported{Options: []string{"tool results"}}
			}
			// A result batch is validated atomically while protected from cancellation.
			s.toolMu.Lock()
			defer s.toolMu.Unlock()
			var cancelled []string
			for i := range v.ToolResults {
				r := &v.ToolResults[i]
				if p, ok := s.tools[r.ID]; ok {
					if p.name != r.Name {
						return errors.New("gemini live tool result does not match a pending call")
					}
				} else if name, ok := s.cancelled[r.ID]; ok && name != "" && name == r.Name {
					cancelled = append(cancelled, r.ID)
				} else {
					return errors.New("gemini live tool result does not match a pending call")
				}
			}
			if len(cancelled) != 0 {
				return &genai.RealtimeToolCallCancelledError{ToolCallIDs: cancelled}
			}
			responses := make([]LiveFunctionResponse, len(v.ToolResults))
			for i := range v.ToolResults {
				r := &v.ToolResults[i]
				p := s.tools[r.ID]
				var obj map[string]json.RawMessage
				response := json.RawMessage(r.Result)
				if json.Unmarshal(response, &obj) != nil || obj == nil {
					// Gemini requires an object; preserve opaque text as a response value.
					wrapped, err := json.Marshal(LiveTextResult{Response: r.Result})
					if err != nil {
						return err
					}
					response = wrapped
				}
				responses[i] = LiveFunctionResponse{ID: p.wireID, Name: p.name, Response: response}
			}
			msgs = append(msgs, LiveClientMessage{ToolResponse: &LiveToolResponse{FunctionResponses: responses}})
			results = v.ToolResults
		case !v.Doc.IsZero():
			blob, err := inlineDoc(&v.Doc, maxDocBytes)
			if err != nil {
				return err
			}
			parts := make([]LivePart, 1, 2)
			parts[0] = LivePart{InlineData: blob}
			if v.Text != "" {
				parts = append(parts, LivePart{Text: v.Text})
			}
			msgs = append(msgs, LiveClientMessage{ClientContent: &LiveClientContent{Turns: []LiveContent{{Role: "user", Parts: parts}}, TurnComplete: v.Generate}})
		default:
			if v.Text == "" {
				// Gemini has no explicit generation command. It responds automatically.
				return nil
			}
			msgs = append(msgs, LiveClientMessage{ClientContent: &LiveClientContent{Turns: []LiveContent{{Role: "user", Parts: []LivePart{{Text: v.Text}}}}, TurnComplete: v.Generate}})
		}
	default:
		return &base.ErrNotSupported{Options: []string{fmt.Sprintf("%T", in)}}
	}
	frames := make([][]byte, len(msgs))
	for i := range msgs {
		data, err := json.Marshal(&msgs[i])
		if err != nil {
			return fmt.Errorf("encode Gemini Live input: %w", err)
		}
		frames[i] = data
	}
	for _, b := range frames {
		if err := s.ws.Write(ctx, b); err != nil {
			return err
		}
	}
	if started {
		s.activity = true
	}
	if ended {
		s.activity = false
	}
	for i := range results {
		delete(s.tools, results[i].ID)
	}
	return nil
}

// inlineDoc renders one inline document as a content part payload. Gemini Live
// reads still images from a client turn; realtimeInput.video carries streamed
// video.
func inlineDoc(d *genai.Doc, limit int64) (*LiveBlob, error) {
	if d.URL != "" {
		return nil, &base.ErrNotSupported{Options: []string{"remote documents"}}
	}
	mimeType, data, err := d.Read(limit)
	if err != nil {
		return nil, fmt.Errorf("read document: %w", err)
	}
	return &LiveBlob{MIMEType: mimeType, Data: data}, nil
}

// Receive implements genai.RealtimeSession.
func (s *RealtimeSession) Receive(ctx context.Context) iter.Seq2[genai.RealtimeEvent, error] {
	return func(yield func(genai.RealtimeEvent, error) bool) {
		for {
			if err := s.ws.Err(); err != nil {
				yield(genai.RealtimeEvent{}, err)
				return
			}
			if err := ctx.Err(); err != nil {
				s.ws.Abort(err)
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
			data, err := s.ws.Read(ctx)
			if err != nil {
				yield(genai.RealtimeEvent{}, err)
				return
			}
			var msg LiveServerMessage
			if err := internal.UnmarshalJSON(data, &msg); err != nil {
				err = fmt.Errorf("decode Gemini Live server message: %w", err)
				s.ws.Abort(err)
				yield(genai.RealtimeEvent{}, err)
				return
			}
			if err := s.translate(&msg); err != nil {
				s.pending = nil
				s.ws.Abort(err)
				yield(genai.RealtimeEvent{}, err)
				return
			}
		}
	}
}

func (s *RealtimeSession) translate(msg *LiveServerMessage) error {
	if msg.Error != nil {
		// Gemini reports protocol failures terminally. Do not echo a provider
		// message that could contain caller content or authentication material.
		return &genai.RealtimeError{Code: strconv.Itoa(msg.Error.Code), Message: "gemini live provider error"}
	}
	first := len(s.pending)
	if (msg.ServerContent != nil && len(msg.ServerContent.ModelTurn.Parts) != 0 || msg.ToolCall != nil) && s.responseID == "" {
		s.nextResponseID++
		s.responseID = "genai-gemini-response-" + strconv.FormatUint(s.nextResponseID, 10)
	}
	if msg.ToolCall != nil {
		s.responseTools = true
	}
	if c := msg.ServerContent; c != nil {
		if c.Interrupted {
			s.pending = append(s.pending, genai.RealtimeEvent{Kind: genai.RealtimeEventInterrupted})
		}
		for i := range c.ModelTurn.Parts {
			p := &c.ModelTurn.Parts[i]
			if p.InlineData != nil {
				if p.InlineData.MIMEType != "audio/pcm;rate=24000" || len(p.InlineData.Data) == 0 || len(p.InlineData.Data)%2 != 0 {
					return errors.New("gemini live returned invalid 24 kHz PCM16 audio")
				}
				s.pending = append(s.pending, genai.RealtimeEvent{Kind: genai.RealtimeEventContentAudio, Content: genai.RealtimeContent{Audio: p.InlineData.Data}})
			}
			if p.Text != "" {
				kind := genai.RealtimeEventContentText
				c := genai.RealtimeContent{Text: p.Text}
				if p.Thought {
					kind = genai.RealtimeEventContentReasoning
					c = genai.RealtimeContent{Reasoning: p.Text}
				}
				s.pending = append(s.pending, genai.RealtimeEvent{Kind: kind, Content: c})
			}
		}
		if c.InterimInputTranscription != nil {
			s.pending = append(s.pending, genai.RealtimeEvent{Kind: genai.RealtimeEventTranscriptInputInterim, Content: genai.RealtimeContent{Text: c.InterimInputTranscription.Text}})
		}
		if c.InputTranscription != nil {
			e, err := liveTranscriptEvent(genai.RealtimeEventTranscriptInput, c.InputTranscription, &s.inputTranscript)
			if err != nil {
				return err
			}
			s.pending = append(s.pending, e)
		}
		if c.OutputTranscription != nil {
			e, err := liveTranscriptEvent(genai.RealtimeEventTranscriptOutput, c.OutputTranscription, &s.outputTranscript)
			if err != nil {
				return err
			}
			s.pending = append(s.pending, e)
		}
		if c.InteractionStatus != "" {
			s.pending = append(s.pending, genai.RealtimeEvent{Kind: genai.RealtimeEventProvider, ProviderEvent: &InteractionStatus{Status: c.InteractionStatus}})
		}
		if c.GenerationComplete || c.Interrupted {
			s.pending = append(s.pending, genai.RealtimeEvent{Kind: genai.RealtimeEventGenerationDone})
		}
		// Extended-thinking sessions emit turnComplete for filler utterances while
		// interactionStatus is IN_PROGRESS. Only the final status completes the turn.
		if c.TurnComplete && c.InteractionStatus != InteractionStatusInProgress {
			s.pending = append(s.pending, genai.RealtimeEvent{Kind: genai.RealtimeEventTurnDone})
		}
	}
	if t := msg.ToolCall; t != nil {
		calls, err := s.registerCalls(t.FunctionCalls)
		if err != nil {
			return err
		}
		s.pending = append(s.pending, genai.RealtimeEvent{Kind: genai.RealtimeEventContentToolCalls, Content: genai.RealtimeContent{ToolCalls: calls}})
	}
	if c := msg.ToolCallCancellation; c != nil {
		s.toolMu.Lock()
		// Reserve the whole batch before changing correlations. Unknown IDs do
		// not establish provenance and cannot classify a later result as cancelled.
		additional := make(map[string]string)
		additionalBytes := 0
		for _, id := range c.IDs {
			if _, exists := s.cancelled[id]; exists {
				continue
			}
			if _, exists := additional[id]; exists {
				continue
			}
			// An empty name reserves unknown or already-resulted IDs against reuse,
			// without inventing exact provenance for future late-result rejection.
			name := s.tools[id].name
			if len(s.cancelled)+len(additional) >= 1024 || len(id)+len(name) > 64*1024-s.cancelledBytes-additionalBytes {
				s.toolMu.Unlock()
				return errors.New("gemini live exceeded cancelled tool call history limit")
			}
			additional[id] = name
			additionalBytes += len(id) + len(name)
		}
		if s.cancelled == nil {
			s.cancelled = make(map[string]string)
		}
		maps.Copy(s.cancelled, additional)
		s.cancelledBytes += additionalBytes
		for _, id := range c.IDs {
			delete(s.tools, id)
		}
		s.toolMu.Unlock()
		s.pending = append(s.pending, genai.RealtimeEvent{Kind: genai.RealtimeEventProvider, ProviderEvent: &ToolCallCancellation{IDs: c.IDs}})
	}
	if u := msg.UsageMetadata; u != nil {
		id := ""
		if c := msg.ServerContent; c != nil && c.TurnComplete && !s.responseTools {
			id = s.responseID
		}
		s.pending = append(s.pending, genai.RealtimeEvent{Kind: genai.RealtimeEventUsage, ResponseID: id, Usage: &genai.Usage{
			InputTokens: u.PromptTokenCount, InputCachedTokens: u.CachedContentTokenCount,
			OutputTokens: u.ResponseTokenCount, ReasoningTokens: u.ThoughtsTokenCount, TotalTokens: u.TotalTokenCount,
			ServiceTier: string(u.ServiceTier),
		}})
	}
	if g := msg.GoAway; g != nil {
		left, err := time.ParseDuration(g.TimeLeft)
		if g.TimeLeft == "" {
			left, err = 0, nil
		}
		if err != nil || left < 0 {
			return errors.New("gemini live returned invalid go-away duration")
		}
		s.pending = append(s.pending, genai.RealtimeEvent{Kind: genai.RealtimeEventProvider, ProviderEvent: &GoAway{TimeLeft: left}})
	}
	if r := msg.SessionResumptionUpdate; r != nil {
		s.pending = append(s.pending, genai.RealtimeEvent{Kind: genai.RealtimeEventProvider, ProviderEvent: &Resumption{Handle: r.NewHandle}})
	}
	if s.info.TurnDetection == genai.RealtimeTurnServer {
		if v := msg.VoiceActivity; v != nil {
			switch v.Type {
			case LiveVoiceActivityStart:
				s.pending = append(s.pending, genai.RealtimeEvent{Kind: genai.RealtimeEventInputSpeechStart})
			case LiveVoiceActivityEnd:
				s.pending = append(s.pending, genai.RealtimeEvent{Kind: genai.RealtimeEventInputSpeechEnd})
			default:
			}
		}
		if v := msg.VoiceActivityDetectionSignal; v != nil {
			switch v.VADSignalType {
			case LiveVADSignalSOS:
				s.pending = append(s.pending, genai.RealtimeEvent{Kind: genai.RealtimeEventInputSpeechStart})
			case LiveVADSignalEOS:
				s.pending = append(s.pending, genai.RealtimeEvent{Kind: genai.RealtimeEventInputSpeechEnd})
			default:
			}
		}
	}
	// Output transcription belongs to the generation and its last chunk precedes
	// generationComplete/interrupted, then turnComplete. Input transcription is
	// independent. See https://ai.google.dev/api/live#BidiGenerateContentServerContent.
	for i := first; i < len(s.pending); i++ {
		e := &s.pending[i]
		switch e.Kind {
		case genai.RealtimeEventContentAudio, genai.RealtimeEventContentText, genai.RealtimeEventContentReasoning,
			genai.RealtimeEventContentToolCalls, genai.RealtimeEventTranscriptOutput, genai.RealtimeEventGenerationDone,
			genai.RealtimeEventTurnDone, genai.RealtimeEventInterrupted:
			e.ResponseID = s.responseID
		default:
		}
	}
	if c := msg.ServerContent; c != nil && c.TurnComplete {
		s.responseID = ""
		s.responseTools = false
	}
	return nil
}

func (s *RealtimeSession) registerCalls(wire []LiveFunctionCall) ([]genai.ToolCall, error) {
	s.toolMu.Lock()
	defer s.toolMu.Unlock()
	// Blocking tools normally leave only one small batch outstanding. Bound
	// retained correlations even if a peer keeps sending calls without results.
	if len(s.tools)+len(wire) > 1024 {
		return nil, errors.New("gemini live exceeded pending tool call limit")
	}
	if len(wire) == 0 {
		return nil, errors.New("gemini live returned an empty tool call batch")
	}
	calls := make([]genai.ToolCall, len(wire))
	for i := range wire {
		c := &wire[i]
		args := c.Args
		if len(args) == 0 {
			args = json.RawMessage(`{}`)
		}
		var obj map[string]json.RawMessage
		if c.Name == "" || json.Unmarshal(args, &obj) != nil || obj == nil {
			return nil, errors.New("gemini live returned invalid tool call")
		}
		id := c.ID
		if id == "" {
			for {
				s.nextID++
				id = fmt.Sprintf("genai-gemini-%d", s.nextID)
				_, cancelled := s.cancelled[id]
				if _, exists := s.tools[id]; !exists && !cancelled {
					break
				}
			}
		}
		if _, exists := s.cancelled[id]; exists {
			return nil, errors.New("gemini live reused a cancelled tool ID")
		}
		if _, exists := s.tools[id]; exists {
			return nil, errors.New("gemini live reused a pending tool ID")
		}
		s.tools[id] = realtimePendingTool{name: c.Name, wireID: c.ID}
		calls[i] = genai.ToolCall{ID: id, Name: c.Name, Arguments: string(args)}
	}
	return calls, nil
}

func liveTranscriptEvent(kind genai.RealtimeEventKind, in *LiveTranscription, text *strings.Builder) (genai.RealtimeEvent, error) {
	if text.Len()+len(in.Text) > maxMessageBytes {
		return genai.RealtimeEvent{}, errors.New("gemini live transcript exceeds size limit")
	}
	text.WriteString(in.Text)
	e := genai.RealtimeEvent{Kind: kind, Content: genai.RealtimeContent{Text: in.Text, Final: in.Finished}}
	if in.Finished {
		e.Content.Text = text.String()
		text.Reset()
	}
	return e, nil
}

type realtimePendingTool struct {
	name   string
	wireID string
}

var _ genai.RealtimeSession = (*RealtimeSession)(nil)
var _ genai.RealtimeOption = (*RealtimeOption)(nil)

func (s *RealtimeSession) updateMessage(u *genai.RealtimeUpdate) (LiveContextUpdate, LiveTools, genai.ToolCallRequest, error) {
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
		return LiveContextUpdate{}, nil, genai.ToolCallAny, &base.ErrNotSupported{Options: options}
	}
	tools, choice := s.liveTools, s.toolChoice
	if u.Tools != nil {
		if err := tools.Init(*u.Tools, s.model); err != nil {
			return LiveContextUpdate{}, nil, genai.ToolCallAny, err
		}
	}
	if u.ToolChoice != nil {
		choice = *u.ToolChoice
		if choice == genai.ToolCallRequired {
			return LiveContextUpdate{}, nil, genai.ToolCallAny, &base.ErrNotSupported{Options: []string{"required tool choice"}}
		}
	}
	if u.MaxOutputTokens != nil || u.OutputModalities != nil {
		return LiveContextUpdate{}, nil, genai.ToolCallAny, &base.ErrNotSupported{Options: []string{"Gemini Live output configuration"}}
	}
	ctx := LiveContextUpdate{}
	if u.Instructions != nil {
		ctx.SystemInstruction = LiveContent{Parts: []LivePart{{Text: *u.Instructions}}}
	}
	if u.Tools != nil || u.ToolChoice != nil {
		if choice == genai.ToolCallNone {
			ctx.Tools = &LiveContextUpdateTools{}
		} else {
			ctx.Tools = &LiveContextUpdateTools{Tools: tools}
		}
	}
	return ctx, tools, choice, nil
}
