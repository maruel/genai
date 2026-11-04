// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// llama.cpp realtime wire types for the OpenAI-compatible conversation protocol.
//
// Modeled from llama.cpp v0.9.0 tools/server/server-realtime.cpp at commit
// cd959bae99ff4bed22bef2604f45241c09bbb8c8. The types cover server events and the client commands the session
// runtime uses.

package llamacpp

import (
	"bytes"
	"encoding/json"
	"errors"
	"math/big"
	"strings"

	"github.com/maruel/genai"
)

// EventType identifies a llama.cpp realtime wire event.
type EventType string

// Client event types.
const (
	EventAudioAppend                EventType = "input_audio_buffer.append"
	EventAudioClear                 EventType = "input_audio_buffer.clear"
	EventAudioCommit                EventType = "input_audio_buffer.commit"
	EventCancel                     EventType = "response.cancel"
	EventCreate                     EventType = "response.create"
	EventItemCreate                 EventType = "conversation.item.create"
	EventTranscriptionSessionUpdate EventType = "transcription_session.update"
	EventTruncate                   EventType = "conversation.item.truncate"
	EventSessionUpdate              EventType = "session.update"
)

// Server event types.
const (
	EventAudioCleared          EventType = "input_audio_buffer.cleared"
	EventAudioCommitted        EventType = "input_audio_buffer.committed"
	EventContentPartAdded      EventType = "response.content_part.added"
	EventContentPartDone       EventType = "response.content_part.done"
	EventError                 EventType = "error"
	EventFunctionCallArgsDelta EventType = "response.function_call_arguments.delta"
	EventFunctionCallArgsDone  EventType = "response.function_call_arguments.done"
	EventInputTranscriptDone   EventType = "conversation.item.input_audio_transcription.completed"
	EventInputTranscriptDelta  EventType = "conversation.item.input_audio_transcription.delta"
	EventInputTranscriptFailed EventType = "conversation.item.input_audio_transcription.failed"
	EventItemAdded             EventType = "conversation.item.added"
	EventItemDone              EventType = "conversation.item.done"
	EventOutputAudioDelta      EventType = "response.output_audio.delta"
	EventOutputAudioDone       EventType = "response.output_audio.done"
	EventOutputItemAdded       EventType = "response.output_item.added"
	EventOutputItemDone        EventType = "response.output_item.done"
	EventOutputTextDelta       EventType = "response.output_text.delta"
	EventOutputTextDone        EventType = "response.output_text.done"
	EventOutputTranscriptDelta EventType = "response.output_audio_transcript.delta"
	EventOutputTranscriptDone  EventType = "response.output_audio_transcript.done"
	EventResponseCreated       EventType = "response.created"
	EventResponseDone          EventType = "response.done"
	EventSessionCreated        EventType = "session.created"
	EventSessionUpdated        EventType = "session.updated"
)

// ItemType identifies a conversation item.
type ItemType string

// Item types.
const (
	ItemFunctionCall   ItemType = "function_call"
	ItemFunctionOutput ItemType = "function_call_output"
	ItemMessage        ItemType = "message"
)

// ResponseStatus is the terminal state of a response.
type ResponseStatus string

// Response status values.
const (
	StatusCancelled  ResponseStatus = "cancelled"
	StatusCompleted  ResponseStatus = "completed"
	StatusFailed     ResponseStatus = "failed"
	StatusIncomplete ResponseStatus = "incomplete"
	StatusInProgress ResponseStatus = "in_progress"
)

// SessionKind identifies a session object.
type SessionKind string

// Session kind values.
const (
	SessionRealtime      SessionKind = "realtime"
	SessionTranscription SessionKind = "transcription"
)

// ObjectKind identifies a wire object.
type ObjectKind string

// Wire object values.
const (
	ObjectItem                 ObjectKind = "realtime.item"
	ObjectResponse             ObjectKind = "realtime.response"
	ObjectSession              ObjectKind = "realtime.session"
	ObjectTranscriptionSession ObjectKind = "realtime.transcription_session"
)

// ItemStatus is a conversation item lifecycle state.
type ItemStatus string

// Item status values.
const (
	ItemCompleted  ItemStatus = "completed"
	ItemInProgress ItemStatus = "in_progress"
	ItemCancelled  ItemStatus = "cancelled"
)

// AudioEncoding identifies a PCM wire encoding.
type AudioEncoding string

// AudioPCM is signed 16-bit little-endian PCM.
const AudioPCM AudioEncoding = "audio/pcm"

// TurnKind identifies a server turn detector.
type TurnKind string

// TurnServerVAD is server-side voice activity detection.
const TurnServerVAD TurnKind = "server_vad"

// ToolKind identifies a tool declaration kind.
type ToolKind string

// ToolFunction is a function tool.
const ToolFunction ToolKind = "function"

// Role identifies a message author.
type Role string

// Role values.
const (
	RoleAssistant Role = "assistant"
	RoleUser      Role = "user"
)

// ContentKind identifies a message content part.
type ContentKind string

// Message content values.
const (
	ContentInputAudio  ContentKind = "input_audio"
	ContentInputText   ContentKind = "input_text"
	ContentOutputAudio ContentKind = "output_audio"
	ContentOutputText  ContentKind = "output_text"
)

// ClientMessage is one outbound wire event.
type ClientMessage struct {
	EventID      string          `json:"event_id,omitzero"`
	Type         EventType       `json:"type"`
	Session      *SessionConfig  `json:"session,omitzero"`
	Audio        []byte          `json:"audio,omitzero"`
	Item         *Item           `json:"item,omitzero"`
	ResponseID   string          `json:"response_id,omitzero"`
	ItemID       string          `json:"item_id,omitzero"`
	ContentIndex *int            `json:"content_index,omitzero"`
	AudioEndMS   *int64          `json:"audio_end_ms,omitzero"`
	Response     *ResponseConfig `json:"response,omitzero"`
}

// SessionUpdateMessage is the session.update command payload. SessionConfig is
// the setup and acknowledgement payload; it cannot describe a partial update.
type SessionUpdateMessage struct {
	Type    EventType      `json:"type"`
	Session *SessionUpdate `json:"session"`
}

// SessionUpdate changes only the fields that are set. An empty non-nil tool
// list clears declarations; a pointer to an empty instruction string clears
// instructions.
//
// Source: llama.cpp tools/server/server-realtime.cpp.
type SessionUpdate struct {
	Type             SessionKind        `json:"type,omitzero"`
	Instructions     *string            `json:"instructions,omitempty"`
	Tools            ToolList           `json:"tools,omitzero"`
	Audio            SessionUpdateAudio `json:"audio,omitzero"`
	ToolChoice       ToolChoice         `json:"tool_choice,omitzero"`
	MaxOutputTokens  MaxOutputTokens    `json:"max_output_tokens,omitzero"`
	OutputModalities Modalities         `json:"output_modalities,omitzero"`
}

// SessionUpdateAudio is the partial audio configuration of session.update.
type SessionUpdateAudio struct {
	Input SessionUpdateAudioInput `json:"input,omitzero"`
}

// SessionUpdateAudioInput is the partial input-audio configuration.
type SessionUpdateAudioInput struct {
	Format        SessionUpdateAudioFormat   `json:"format,omitzero"`
	Transcription SessionUpdateTranscription `json:"transcription,omitzero"`
}

// SessionUpdateAudioFormat is the partial PCM format accepted by
// session.update. Only the rate is validated by the server.
type SessionUpdateAudioFormat struct {
	Type AudioEncoding `json:"type,omitzero"`
	Rate int           `json:"rate,omitzero"`
}

// SessionUpdateTranscription is the partial input-transcription configuration.
type SessionUpdateTranscription struct {
	Language string `json:"language,omitzero"`
}

// ResponseConfig is the response.create payload.
type ResponseConfig struct {
	OutputModalities Modalities       `json:"output_modalities,omitzero"`
	MaxOutputTokens  *MaxOutputTokens `json:"max_output_tokens,omitzero"`
	ToolChoice       ToolChoice       `json:"tool_choice,omitzero"`
}

// ToolChoice selects how the model chooses tools.
type ToolChoice string

// Tool choice values.
const (
	ToolChoiceAuto     ToolChoice = "auto"
	ToolChoiceNone     ToolChoice = "none"
	ToolChoiceRequired ToolChoice = "required"
)

// MaxOutputTokens is an integer token limit or "inf".
type MaxOutputTokens struct {
	Value    int64
	Infinite bool
}

// UnmarshalJSON accepts an integer or "inf".
func (m *MaxOutputTokens) UnmarshalJSON(b []byte) error {
	*m = MaxOutputTokens{}
	if bytes.Equal(b, []byte(`"inf"`)) {
		m.Infinite = true
		return nil
	}
	if err := json.Unmarshal(b, &m.Value); err != nil {
		return errors.New("realtime max_output_tokens must be an integer or \"inf\"")
	}
	return nil
}

// MarshalJSON emits an integer or "inf".
func (m MaxOutputTokens) MarshalJSON() ([]byte, error) {
	if m.Infinite {
		return []byte(`"inf"`), nil
	}
	return json.Marshal(m.Value)
}

// SessionConfig is the session setup and acknowledgement payload.
type SessionConfig struct {
	ID               string           `json:"id,omitzero"`
	Object           ObjectKind       `json:"object,omitzero"`
	Type             SessionKind      `json:"type"`
	Model            string           `json:"model,omitzero"`
	Instructions     string           `json:"instructions"`
	OutputModalities Modalities       `json:"output_modalities"`
	Audio            AudioConfig      `json:"audio"`
	Tools            ToolList         `json:"tools"`
	ToolChoice       ToolChoice       `json:"tool_choice,omitzero"`
	MaxOutputTokens  *MaxOutputTokens `json:"max_output_tokens,omitzero"`
}

// AudioConfig groups input and output audio settings.
type AudioConfig struct {
	Input  AudioInput  `json:"input"`
	Output AudioOutput `json:"output"`
}

// AudioInput configures microphone audio.
type AudioInput struct {
	Format        AudioFormat    `json:"format"`
	TurnDetection *TurnDetection `json:"turn_detection"`
	Transcription *Transcription `json:"transcription"`
}

// AudioOutput configures generated audio.
type AudioOutput struct {
	Format AudioFormat `json:"format"`
	Voice  string      `json:"voice,omitzero"`
}

// AudioFormat describes PCM on the wire.
type AudioFormat struct {
	Type AudioEncoding `json:"type"`
	Rate int           `json:"rate"`
}

// TurnDetection configures server-side turn detection.
type TurnDetection struct {
	Type              TurnKind `json:"type"`
	CreateResponse    bool     `json:"create_response"`
	InterruptResponse bool     `json:"interrupt_response"`
}

// Transcription configures input transcription.
type Transcription struct {
	Model    string `json:"model"`
	Language string `json:"language,omitzero"`
}

// RealtimeTool is a function tool declaration.
type RealtimeTool struct {
	Type        ToolKind        `json:"type"`
	Name        string          `json:"name"`
	Description string          `json:"description,omitzero"`
	Parameters  json.RawMessage `json:"parameters"`
}

// From converts a portable tool declaration to a llama.cpp wire tool.
func (t *RealtimeTool) From(in *genai.ToolDef) error {
	schema, err := in.GetInputSchema()
	if err != nil {
		return err
	}
	*t = RealtimeTool{Type: ToolFunction, Name: in.Name, Description: in.Description, Parameters: schema}
	return nil
}

// Item is a conversation item included in a client message.
type Item struct {
	Type      ItemType                 `json:"type"`
	ID        string                   `json:"id,omitzero"`
	Object    ObjectKind               `json:"object,omitzero"`
	Status    ItemStatus               `json:"status,omitzero"`
	Role      Role                     `json:"role,omitzero"`
	Content   []RealtimeMessageContent `json:"content,omitzero"`
	CallID    string                   `json:"call_id,omitzero"`
	Output    string                   `json:"output,omitzero"`
	Name      string                   `json:"name,omitzero"`
	Arguments string                   `json:"arguments,omitzero"`
}

// RealtimeMessageContent is one message content part.
type RealtimeMessageContent struct {
	Type       ContentKind `json:"type"`
	Text       string      `json:"text,omitzero"`
	Transcript string      `json:"transcript,omitzero"`
}

// ServerMessage is one inbound wire event.
type ServerMessage struct {
	Type           EventType               `json:"type"`
	EventID        string                  `json:"event_id"`
	Session        *SessionConfig          `json:"session"`
	Response       *Response               `json:"response"`
	ResponseID     string                  `json:"response_id"`
	ItemID         string                  `json:"item_id"`
	OutputIndex    int                     `json:"output_index"`
	ContentIndex   int                     `json:"content_index"`
	Delta          string                  `json:"delta"`
	Text           string                  `json:"text"`
	Transcript     string                  `json:"transcript"`
	CallID         string                  `json:"call_id"`
	Name           string                  `json:"name"`
	Arguments      string                  `json:"arguments"`
	Error          *WireError              `json:"error"`
	PreviousItemID string                  `json:"previous_item_id"`
	Item           *Item                   `json:"item"`
	Part           *RealtimeMessageContent `json:"part"`
}

// FunctionCallArgsDelta is a streamed fragment of a function call's JSON
// arguments. It does not authorize execution; ToolCalls does.
type FunctionCallArgsDelta struct {
	ResponseID  string
	ItemID      string
	OutputIndex int
	Delta       string
}

// RealtimeEventName implements genai.RealtimeProviderEvent.
func (*FunctionCallArgsDelta) RealtimeEventName() string {
	return string(EventFunctionCallArgsDelta)
}

// FunctionCallArgsDone is a completed function call before response.done. It
// does not authorize execution; ToolCalls does.
type FunctionCallArgsDone struct {
	ResponseID  string
	ItemID      string
	OutputIndex int
	CallID      string
	Name        string
	Arguments   string
}

// RealtimeEventName implements genai.RealtimeProviderEvent.
func (*FunctionCallArgsDone) RealtimeEventName() string {
	return string(EventFunctionCallArgsDone)
}

// Response is a completed or failed model response.
type Response struct {
	ID               string            `json:"id"`
	Object           ObjectKind        `json:"object,omitzero"`
	Status           ResponseStatus    `json:"status"`
	Output           boundedList[Item] `json:"output"`
	Usage            *RealtimeUsage    `json:"usage"`
	OutputModalities Modalities        `json:"output_modalities,omitzero"`
}

// RealtimeUsage is response token accounting.
type RealtimeUsage struct {
	InputTokens  int64        `json:"input_tokens,omitzero"`
	OutputTokens int64        `json:"output_tokens,omitzero"`
	TotalTokens  int64        `json:"total_tokens,omitzero"`
	InputDetails TokenDetails `json:"input_token_details,omitzero"`
}

// TokenDetails is input token accounting detail.
type TokenDetails struct {
	CachedTokens int64 `json:"cached_tokens,omitzero"`
}

// WireError is a provider error payload.
type WireError struct {
	Type    string `json:"type,omitzero"`
	Code    string `json:"code,omitzero"`
	Message string `json:"message,omitzero"`
	Param   string `json:"param,omitzero"`
	EventID string `json:"event_id,omitzero"`
}

// Modalities bounds the two output modalities a provider may negotiate.
// Each provider rejects combinations it does not implement.
type Modalities []genai.Modality

// UnmarshalJSON bounds the negotiated output modalities.
func (m *Modalities) UnmarshalJSON(b []byte) error {
	v, err := parseList[genai.Modality](b, 2)
	*m = v
	return err
}

// ToolList bounds tool declarations.
type ToolList []RealtimeTool

// Init converts portable tool declarations to llama.cpp wire tools.
func (l *ToolList) Init(tools []genai.ToolDef) error {
	if len(tools) > maxRealtimeTools {
		return errors.New("realtime tool declarations exceed limit")
	}
	decls := make([]RealtimeTool, 0, len(tools))
	for i := range tools {
		var tool RealtimeTool
		if err := tool.From(&tools[i]); err != nil {
			return err
		}
		decls = append(decls, tool)
	}
	*l = decls
	return nil
}

// Names returns the declared tool names.
func (l ToolList) Names() []string {
	names := make([]string, len(l))
	for i := range l {
		names[i] = l[i].Name
	}
	return names
}

// UnmarshalJSON bounds peer-supplied tool declarations.
func (l *ToolList) UnmarshalJSON(b []byte) error {
	v, err := parseList[RealtimeTool](b, maxRealtimeTools)
	*l = v
	return err
}

// boundedList stops allocation amplification from peer-supplied arrays before
// decoding each next element. Both output items and declarations have 1024 slots.
type boundedList[T any] []T

func (l *boundedList[T]) UnmarshalJSON(b []byte) error {
	v, err := parseList[T](b, maxRealtimeTools)
	*l = v
	return err
}

func parseList[T any](b []byte, limit int) ([]T, error) {
	if bytes.Equal(b, []byte("null")) {
		return nil, nil
	}
	d := json.NewDecoder(bytes.NewReader(b))
	tok, err := d.Token()
	if err != nil {
		return nil, err
	}
	if tok != json.Delim('[') {
		return nil, errors.New("realtime list must be an array")
	}
	var values []T
	for d.More() {
		if len(values) >= limit {
			return nil, errors.New("realtime list exceeds item limit")
		}
		var v T
		if err := d.Decode(&v); err != nil {
			return nil, err
		}
		values = append(values, v)
	}
	if _, err := d.Token(); err != nil {
		return nil, err
	}
	return values, nil
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
