// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Stateful realtime conversation contracts independent of provider transports.
//
// [Provider.ConnectRealtime] opens a bidirectional [RealtimeSession] alongside
// the request-oriented API. The session owns provider conversation state. The
// application owns capture, playback, voice activity detection, tool execution,
// and reconnection.

package genai

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"iter"
)

// RealtimeSession is one provider-owned conversation and its transport.
//
// One goroutine receives events while other goroutines may call Send. Concurrent
// sends are serialized; callers must order dependent commands themselves. Close
// may run concurrently and wakes blocked Send and Receive calls. Call it once.
// The session does not reconnect, replay, execute tools, detect voice activity,
// or resample.
type RealtimeSession interface {
	io.Closer
	// Info returns immutable native formats, configured features, and accepted
	// operations after setup.
	Info() RealtimeSessionInfo
	// Send applies one complete command to the session and writes it to the
	// provider. It rejects invalid, unsupported, and out-of-order commands before
	// the first wire write. It serializes concurrent calls, translates the command
	// into the provider's wire messages, and updates local correlation only after
	// every frame is written. Input buffers are borrowed until Send returns and are
	// not retained. A successful return means written, not accepted by the
	// provider. A write failure may leave a partial transaction and terminates the
	// session; do not retry it.
	Send(ctx context.Context, input RealtimeInput) error
	// Receive returns the events as they arrive from the server. Slow consumers
	// apply backpressure; providers must bound frame sizes and any pending events
	// rather than queue indefinitely. Unknown wire events may be skipped or returned
	// as RealtimeEventProvider events; malformed recognized events are errors. Yielded
	// content must satisfy RealtimeContent.Validate. The sequence stops after the
	// first error, including io.EOF for normal peer closure. Breaking out stops
	// receiving without closing the session. Canceling ctx terminates the transport.
	Receive(ctx context.Context) iter.Seq2[RealtimeEvent, error]
}

// RealtimeOption configures a realtime session.
//
// Portable options are RealtimeOptionInstructions, RealtimeOptionAudio,
// RealtimeOptionTools, RealtimeTurnDetection, and Modalities. Modalities selects
// output modalities and accepts only ModalityAudio and ModalityText; empty
// selects audio. Input is per command: audio, text, or tool results.
//
// A provider may define its own option type for settings that have no portable
// meaning. A provider rejects options it cannot honor rather than ignoring them.
type RealtimeOption interface {
	Validate() error
}

// AudioFormat describes raw mono PCM16LE at SampleRate Hz. genai never
// resamples; callers convert at the capture and playback boundaries.
type AudioFormat struct {
	SampleRate int
}

// Validate checks that SampleRate is positive.
func (f *AudioFormat) Validate() error {
	if f.SampleRate <= 0 {
		return errors.New("sample rate must be positive")
	}
	return nil
}

// RealtimeTurnDetection selects who decides that microphone input is complete.
type RealtimeTurnDetection string

// Validate implements RealtimeOption.
func (o RealtimeTurnDetection) Validate() error {
	if o != "" && o != RealtimeTurnManual && o != RealtimeTurnServer {
		return fmt.Errorf("unknown turn detection %q", o)
	}
	return nil
}

// RealtimeTurnDetection values. Empty selects the provider's documented default.
const (
	RealtimeTurnManual RealtimeTurnDetection = "manual"
	RealtimeTurnServer RealtimeTurnDetection = "server"
)

// RealtimeOptionInstructions supplies session instructions.
type RealtimeOptionInstructions string

// Validate implements RealtimeOption.
func (RealtimeOptionInstructions) Validate() error { return nil }

// RealtimeOptionAudio requests voice selection and audio transcripts.
// Transcription is opt-in and independent of generated text output. A false
// transcription field requests no transcription but does not promise that the
// provider sends no transcript events; callers must handle them either way.
type RealtimeOptionAudio struct {
	Voice               string
	InputTranscription  bool
	OutputTranscription bool
}

// Validate implements RealtimeOption.
func (*RealtimeOptionAudio) Validate() error { return nil }

// RealtimeOptionTools declares session tools. The application owns their execution.
type RealtimeOptionTools struct {
	Tools []ToolDef
}

// Validate implements RealtimeOption.
func (o *RealtimeOptionTools) Validate() error {
	return validateRealtimeTools(o.Tools)
}

func validateRealtimeTools(tools []ToolDef) error {
	for i := range tools {
		if err := tools[i].Validate(); err != nil {
			return fmt.Errorf("tool %d: %w", i, err)
		}
		if tools[i].Callback == nil && len(tools[i].InputSchemaOverride) == 0 {
			return fmt.Errorf("tool %q requires a callback or input schema", tools[i].Name)
		}
		schema, err := tools[i].GetInputSchema()
		if err != nil {
			return fmt.Errorf("tool %q schema: %w", tools[i].Name, err)
		}
		var object map[string]json.RawMessage
		if err := json.Unmarshal(schema, &object); err != nil || object == nil {
			return fmt.Errorf("tool %q requires an object input schema", tools[i].Name)
		}
		for j := range i {
			if tools[j].Name == tools[i].Name {
				return fmt.Errorf("duplicate tool name %q", tools[i].Name)
			}
		}
	}
	return nil
}

// RealtimeCapabilities describes the operations this configured session accepts.
// A false field means the provider rejects that command with
// base.ErrNotSupported. Configured features are reported by RealtimeSessionInfo,
// not here.
//
// AudioCommit reports that RealtimeInputContent.Commit commits manual-turn audio.
type RealtimeCapabilities struct {
	AudioCommit    bool
	CancelResponse bool
	ToolResults    bool
	Truncate       bool
	Update         RealtimeUpdateCapabilities
}

// RealtimeUpdateCapabilities describes the session settings this configured
// session accepts at runtime. ToolChoice covers auto and none; required has its
// own field because some providers can clear tools but cannot force a call.
// VoiceBeforeOutput rejects a voice change after the session produced audio.
type RealtimeUpdateCapabilities struct {
	Instructions       bool
	Tools              bool
	ToolChoice         bool
	ToolChoiceRequired bool
	MaxOutputTokens    bool
	OutputModalities   bool
	TurnDetection      bool
	VoiceBeforeOutput  bool
}

// RealtimeSessionInfo contains the accepted session identity, native formats,
// turn mode, and capabilities. IDs are provider-supplied when available; an
// empty ID means none was supplied.
type RealtimeSessionInfo struct {
	ID           string
	InputFormat  AudioFormat
	OutputFormat AudioFormat
	// TurnDetection is the accepted server or manual turn mode.
	TurnDetection RealtimeTurnDetection
	Capabilities  RealtimeCapabilities
}

// RealtimeInput is one command sent to RealtimeSession. Only the types in this
// package implement it.
type RealtimeInput interface {
	realtimeInput()
	Validate() error
}

// RealtimeInputContent sends one content command. Audio is selected by Audio or
// Commit; otherwise exactly one of Text or ToolResults is set. The zero value
// is invalid.
//
// Audio appends PCM at Info().InputFormat to the buffered user audio. Commit
// commits the buffer without requesting a reply and may close the provider audio
// stream; an empty Commit commits audio sent earlier. Generate requests a reply
// and commits nonempty audio first when needed. With server turn detection, the
// provider may ignore Generate and reply on its own.
//
// Text adds a user text turn. Generate=false adds passive context; Generate=true
// submits the turn and requests a response. An empty Text with Generate=true
// requests a response from the current context. Providers without an explicit
// generation command may ignore it.
//
// ToolResults answers completed calls. Send complete batches to preserve
// parallel call ordering. Generate requests a response after the results;
// providers that resume automatically treat it as advisory.
type RealtimeInputContent struct {
	// Audio appends PCM at Info().InputFormat. It may be empty only with Commit.
	Audio []byte
	// Text adds a user text turn. It may be empty only with Generate.
	Text string
	// ToolResults answers completed calls.
	ToolResults []ToolCallResult
	// Commit commits buffered audio without requesting a reply.
	Commit bool
	// Generate requests a response. It commits nonempty audio first.
	Generate bool
}

func (RealtimeInputContent) realtimeInput() {}

// Validate checks that exactly one payload is set and that its flags are valid.
func (i RealtimeInputContent) Validate() error {
	switch {
	case i.Audio != nil || i.Commit:
		if i.Text != "" || i.ToolResults != nil {
			return errors.New("realtime input content sets more than one payload")
		}
		if len(i.Audio)%2 != 0 {
			return errors.New("audio must contain complete PCM16 samples")
		}
		if len(i.Audio) == 0 && !i.Commit {
			return errors.New("audio requires samples unless Commit")
		}
	case i.ToolResults != nil:
		if i.Text != "" {
			return errors.New("realtime input content sets more than one payload")
		}
		if len(i.ToolResults) == 0 {
			return errors.New("tool results are required")
		}
		for j := range i.ToolResults {
			r := &i.ToolResults[j]
			if err := r.Validate(); err != nil {
				return fmt.Errorf("tool result %d: %w", j, err)
			}
			if r.ID == "" || r.Name == "" {
				return errors.New("realtime tool results require both ID and Name")
			}
			for k := range j {
				if i.ToolResults[k].ID == r.ID {
					return fmt.Errorf("duplicate tool result ID %q", r.ID)
				}
			}
		}
	default:
		if i.Text == "" && !i.Generate {
			return errors.New("realtime input content requires audio, text, or tool results")
		}
	}
	return nil
}

// RealtimeCancel stops generation, optionally targeting ResponseID. It does not
// remove conversation history or queued playback.
type RealtimeCancel struct {
	ResponseID string
}

func (RealtimeCancel) realtimeInput() {}

// Validate implements RealtimeInput.
func (RealtimeCancel) Validate() error { return nil }

// RealtimeTruncate reconciles unheard output audio in ItemID at ContentIndex and
// AudioEndMS. It requires Capabilities.Truncate and does not stop generation.
type RealtimeTruncate struct {
	ItemID       string
	ContentIndex int
	AudioEndMS   int64
}

func (RealtimeTruncate) realtimeInput() {}

// Validate checks the truncation target.
func (i RealtimeTruncate) Validate() error {
	if i.ItemID == "" || i.ContentIndex < 0 || i.AudioEndMS < 0 {
		return errors.New("truncate requires ItemID and nonnegative ContentIndex and AudioEndMS")
	}
	return nil
}

// RealtimeUpdate changes live session settings. A nil field is unchanged.
// Tools may be an empty list to clear declarations. Instructions may be empty
// to clear them. MaxOutputTokens must be positive. TurnDetection selects manual
// or server turns. Voice changes only before output audio on providers that
// report VoiceBeforeOutput.
type RealtimeUpdate struct {
	Instructions     *string
	Tools            *[]ToolDef
	ToolChoice       *ToolCallRequest
	MaxOutputTokens  *int64
	OutputModalities *Modalities
	TurnDetection    *RealtimeTurnDetection
	Voice            *string
}

func (RealtimeUpdate) realtimeInput() {}

// Validate rejects empty, malformed, or ambiguous updates.
func (u RealtimeUpdate) Validate() error {
	if u.Instructions == nil && u.Tools == nil && u.ToolChoice == nil && u.MaxOutputTokens == nil && u.OutputModalities == nil && u.TurnDetection == nil && u.Voice == nil {
		return errors.New("realtime update requires at least one setting")
	}
	if u.Tools != nil {
		if err := validateRealtimeTools(*u.Tools); err != nil {
			return err
		}
	}
	if u.ToolChoice != nil {
		switch *u.ToolChoice {
		case ToolCallAny, ToolCallNone, ToolCallRequired:
		default:
			return fmt.Errorf("unknown realtime tool choice %d", *u.ToolChoice)
		}
	}
	if u.MaxOutputTokens != nil && *u.MaxOutputTokens <= 0 {
		return errors.New("max output tokens must be positive")
	}
	if u.TurnDetection != nil && *u.TurnDetection != RealtimeTurnManual && *u.TurnDetection != RealtimeTurnServer {
		return fmt.Errorf("unknown turn detection %q", *u.TurnDetection)
	}
	if u.OutputModalities != nil {
		if len(*u.OutputModalities) == 0 {
			return errors.New("output modalities must not be empty")
		}
		for i, m := range *u.OutputModalities {
			if m != ModalityAudio && m != ModalityText {
				return fmt.Errorf("unsupported realtime output modality %q", m)
			}
			for _, p := range (*u.OutputModalities)[:i] {
				if p == m {
					return fmt.Errorf("duplicate output modality %q", m)
				}
			}
		}
	}
	return nil
}

// RealtimeEventKind identifies an event emitted by a session.
type RealtimeEventKind string

// RealtimeEventKind values. Each kind defines the RealtimeEvent fields that
// apply to it.
const (
	// RealtimeEventContentAudio carries one chunk of generated PCM audio at
	// Info().OutputFormat. Chunks may split words and utterances arbitrarily.
	RealtimeEventContentAudio RealtimeEventKind = "audio"
	// RealtimeEventContentText carries generated text. Content.Text is a delta unless
	// Final is true, in which case it replaces the accumulated deltas.
	RealtimeEventContentText RealtimeEventKind = "text"
	// RealtimeEventContentToolCalls carries complete function calls from one response.
	// Arguments are validated JSON, and IDs are unique and nonempty. The caller
	// executes the calls and sends the results in RealtimeInputContent.ToolResults. The
	// turn cannot complete until it receives those results.
	RealtimeEventContentToolCalls RealtimeEventKind = "tool_calls"
	// RealtimeEventError reports a recoverable provider error. The session
	// remains usable.
	RealtimeEventError RealtimeEventKind = "error"
	// RealtimeEventGenerationDone reports that the provider ended generation for
	// one response. Playback may continue, and a response that only calls tools
	// also emits it. It does not end the turn. An interrupted response also emits
	// it before TurnDone.
	RealtimeEventGenerationDone RealtimeEventKind = "generation_done"
	// RealtimeEventInputSpeechEnd reports that provider voice activity detection
	// stopped detecting user speech. It occurs only with server turn detection;
	// manual turns disable provider VAD. ItemID may identify the user item. The
	// provider may separately commit audio or start a response when configured;
	// this event guarantees neither.
	RealtimeEventInputSpeechEnd RealtimeEventKind = "input_speech_end"
	// RealtimeEventInputSpeechStart reports that provider voice activity detection
	// detected user speech. It occurs only with server turn detection; manual
	// turns disable provider VAD. ItemID may identify the user item that speech
	// will create. It does not guarantee that a turn produces a response.
	RealtimeEventInputSpeechStart RealtimeEventKind = "input_speech_start"
	// RealtimeEventInterrupted reports that the provider stopped the active
	// response before normal completion. Stop and discard queued playback. The
	// session remains usable.
	RealtimeEventInterrupted RealtimeEventKind = "interrupted"
	// RealtimeEventProvider carries an event with no portable kind. ProviderEvent
	// holds the concrete provider type. Callers must ignore provider event types
	// they do not handle.
	RealtimeEventProvider RealtimeEventKind = "provider"
	// RealtimeEventTranscriptInput carries a transcription of user audio.
	// Content.Text is a delta unless Final is true, in which case it replaces the
	// accumulated deltas. Transcription is independent of model output and may
	// arrive out of order with respect to response events.
	RealtimeEventTranscriptInput RealtimeEventKind = "input_transcript"
	// RealtimeEventTranscriptOutput carries a transcription of generated audio.
	// Content.Text is a delta unless Final is true, in which case it replaces the
	// accumulated deltas. It is not generated text output; RealtimeEventContentText
	// carries that.
	RealtimeEventTranscriptOutput RealtimeEventKind = "output_transcript"
	// RealtimeEventTurnDone reports that the provider completed the current turn.
	// No more output follows for the current input unless the caller submits tool
	// results. It also follows an interrupted, cancelled, or failed response. The
	// session remains open.
	RealtimeEventTurnDone RealtimeEventKind = "turn_done"
	// RealtimeEventUsage reports token usage for one response and may include a
	// rate-limit snapshot. It is not cumulative; sum the token fields across
	// events for session totals. When Usage.Limits is set, it replaces the
	// previous limits.
	RealtimeEventUsage RealtimeEventKind = "usage"
)

// RealtimeProviderEvent is a provider-specific event payload. Providers define
// the concrete types and document the event names they represent. Providers use
// it when no portable event kind applies; callers ignore event kinds they do not
// handle.
type RealtimeProviderEvent interface {
	// RealtimeEventName returns the provider's wire event name.
	RealtimeEventName() string
}

// RealtimeContent is the payload of a RealtimeEvent. The zero value means the
// event carries no content. At most one of Audio, Text, or ToolCalls is set.
// Final is valid only with Text.
type RealtimeContent struct {
	// Audio is owned PCM at Info().OutputFormat for Audio events.
	Audio []byte
	// Text is a delta for Text and transcript events. When Final is true, Text
	// is the complete text and replaces accumulated deltas.
	Text string
	// Final marks Text as complete for Text and transcript events.
	Final bool
	// ToolCalls contains complete calls with validated JSON arguments and unique
	// nonempty IDs for ToolCalls events. Providers that omit call IDs must assign
	// session-unique IDs and preserve the original provider correlation internally.
	ToolCalls []ToolCall
}

// Validate rejects multiple payload fields or Final without Text.
func (c *RealtimeContent) Validate() error {
	n := 0
	if len(c.Audio) != 0 {
		n++
	}
	if c.Text != "" {
		n++
	}
	if len(c.ToolCalls) != 0 {
		n++
	}
	if n > 1 {
		return errors.New("realtime content sets more than one of Audio, Text, or ToolCalls")
	}
	if c.Final && c.Text == "" {
		return errors.New("realtime content Final requires Text")
	}
	return nil
}

// RealtimeEvent is one normalized event. Kind identifies the event; Content
// carries its payload. Provider-specific events use RealtimeEventProvider.
// Empty IDs mean unspecified, never a fabricated provider identity.
type RealtimeEvent struct {
	// Kind identifies the event. GenerationDone means output generation ended,
	// not playback ended. TurnDone means the provider completed its turn, which
	// need not mean session idleness. Interrupted signals the caller to stop
	// playback.
	Kind RealtimeEventKind
	// ResponseID identifies the response when the provider supplies it.
	ResponseID string
	// ItemID identifies the conversation item when the provider supplies it.
	ItemID string
	// ContentIndex identifies the content part within ItemID.
	ContentIndex int
	// Content is the event payload. Its zero value means the event carries no
	// content.
	Content RealtimeContent
	// Usage is a non-cumulative snapshot for one response. Each Usage event
	// describes its own response; sum them to compute session usage. ResponseID
	// identifies the response when the provider supplies it. Limits, when present,
	// replaces the previous rate-limit snapshot.
	Usage *Usage
	// Error carries a recoverable provider error for Error events. The session
	// remains usable. Terminal failures are returned by Receive instead.
	Error *RealtimeError
	// ProviderEvent is provider-specific information for this event. It is nil
	// when the provider supplies none. For RealtimeEventProvider, it is the
	// complete payload.
	ProviderEvent RealtimeProviderEvent
}

// RealtimeError is an error reported by the provider. RealtimeEventError carries
// a recoverable one; setup and transport failures are returned directly.
type RealtimeError struct {
	// Code is the provider error code when reported.
	Code string
	// Message is a short, caller-safe description of the failure.
	Message string
	// EventID identifies the rejected client event when reported.
	EventID string
}

// Error implements error.
func (e *RealtimeError) Error() string {
	if e.Code == "" {
		return e.Message
	}
	return e.Code + ": " + e.Message
}
