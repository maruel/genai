// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// OpenAI GA realtime wire types.
//
// The types model the server events and the client commands used by the session
// runtime. Server events are modeled so strict JSON decoding accepts messages
// the official SDKs can produce, including events the runtime ignores. Client
// command payloads model the fields the runtime sends; unused command fields are
// omitted. The runtime implements a subset of the protocol; see realtime.go.

package openaibase

import (
	"bytes"
	"encoding/json"
	"errors"

	"github.com/maruel/genai"
	"github.com/maruel/genai/internal"
)

// EventType identifies an OpenAI GA realtime wire event.
type EventType string

// Client event types.
const (
	EventAudioAppend            EventType = "input_audio_buffer.append"
	EventAudioClear             EventType = "input_audio_buffer.clear"
	EventAudioCommit            EventType = "input_audio_buffer.commit"
	EventCancel                 EventType = "response.cancel"
	EventCreate                 EventType = "response.create"
	EventItemCreate             EventType = "conversation.item.create"
	EventItemDelete             EventType = "conversation.item.delete"
	EventItemRetrieve           EventType = "conversation.item.retrieve"
	EventOutputAudioBufferClear EventType = "output_audio_buffer.clear"
	EventSessionUpdate          EventType = "session.update"
	EventTruncate               EventType = "conversation.item.truncate"
)

// Server event types.
const (
	EventAudioCleared             EventType = "input_audio_buffer.cleared"
	EventAudioCommitted           EventType = "input_audio_buffer.committed"
	EventContentPartAdded         EventType = "response.content_part.added"
	EventContentPartDone          EventType = "response.content_part.done"
	EventConversationCreated      EventType = "conversation.created"
	EventDTMFReceived             EventType = "input_audio_buffer.dtmf_event_received"
	EventError                    EventType = "error"
	EventFunctionCallArgsDelta    EventType = "response.function_call_arguments.delta"
	EventFunctionCallArgsDone     EventType = "response.function_call_arguments.done"
	EventInputTranscriptDone      EventType = "conversation.item.input_audio_transcription.completed"
	EventInputTranscriptDelta     EventType = "conversation.item.input_audio_transcription.delta"
	EventInputTranscriptFailed    EventType = "conversation.item.input_audio_transcription.failed"
	EventInputTranscriptSegment   EventType = "conversation.item.input_audio_transcription.segment"
	EventItemAdded                EventType = "conversation.item.added"
	EventItemCreated              EventType = "conversation.item.created"
	EventItemDeleted              EventType = "conversation.item.deleted"
	EventItemDone                 EventType = "conversation.item.done"
	EventItemRetrieved            EventType = "conversation.item.retrieved"
	EventItemTruncated            EventType = "conversation.item.truncated"
	EventMcpCallArgsDelta         EventType = "response.mcp_call_arguments.delta"
	EventMcpCallArgsDone          EventType = "response.mcp_call_arguments.done"
	EventMcpCallCompleted         EventType = "response.mcp_call.completed"
	EventMcpCallFailed            EventType = "response.mcp_call.failed"
	EventMcpCallInProgress        EventType = "response.mcp_call.in_progress"
	EventMcpListToolsCompleted    EventType = "mcp_list_tools.completed"
	EventMcpListToolsFailed       EventType = "mcp_list_tools.failed"
	EventMcpListToolsInProgress   EventType = "mcp_list_tools.in_progress"
	EventOutputAudioBufferCleared EventType = "output_audio_buffer.cleared"
	EventOutputAudioBufferStarted EventType = "output_audio_buffer.started"
	EventOutputAudioBufferStopped EventType = "output_audio_buffer.stopped"
	EventOutputAudioDelta         EventType = "response.output_audio.delta"
	EventOutputAudioDone          EventType = "response.output_audio.done"
	EventOutputItemAdded          EventType = "response.output_item.added"
	EventOutputItemDone           EventType = "response.output_item.done"
	EventOutputTextDelta          EventType = "response.output_text.delta"
	EventOutputTextDone           EventType = "response.output_text.done"
	EventOutputTranscriptDelta    EventType = "response.output_audio_transcript.delta"
	EventOutputTranscriptDone     EventType = "response.output_audio_transcript.done"
	EventRateLimits               EventType = "rate_limits.updated"
	EventResponseCreated          EventType = "response.created"
	EventResponseDone             EventType = "response.done"
	EventSessionCreated           EventType = "session.created"
	EventSessionUpdated           EventType = "session.updated"
	EventSpeechStarted            EventType = "input_audio_buffer.speech_started"
	EventSpeechStopped            EventType = "input_audio_buffer.speech_stopped"
	EventTimeoutTriggered         EventType = "input_audio_buffer.timeout_triggered"
)

// ItemType identifies a conversation item.
type ItemType string

// Item types.
const (
	ItemFunctionCall        ItemType = "function_call"
	ItemFunctionOutput      ItemType = "function_call_output"
	ItemMcpApprovalRequest  ItemType = "mcp_approval_request"
	ItemMcpApprovalResponse ItemType = "mcp_approval_response"
	ItemMcpCall             ItemType = "mcp_call"
	ItemMcpListTools        ItemType = "mcp_list_tools"
	ItemMessage             ItemType = "message"
	ItemReference           ItemType = "item_reference"
)

// ItemStatus is a conversation item lifecycle state.
type ItemStatus string

// Item status values.
const (
	ItemCompleted  ItemStatus = "completed"
	ItemIncomplete ItemStatus = "incomplete"
	ItemInProgress ItemStatus = "in_progress"
)

// ObjectKind identifies an OpenAI wire object.
type ObjectKind string

// Wire object identifiers.
const (
	ObjectConversation         ObjectKind = "realtime.conversation"
	ObjectItem                 ObjectKind = "realtime.item"
	ObjectResponse             ObjectKind = "realtime.response"
	ObjectSession              ObjectKind = "realtime.session"
	ObjectTranscriptionSession ObjectKind = "realtime.transcription_session"
)

// Role identifies a message author.
type Role string

// Message roles.
const (
	RoleAssistant Role = "assistant"
	RoleSystem    Role = "system"
	RoleUser      Role = "user"
)

// ContentKind identifies a message content part.
type ContentKind string

// Message content kinds.
const (
	ContentInputAudio  ContentKind = "input_audio"
	ContentInputImage  ContentKind = "input_image"
	ContentInputText   ContentKind = "input_text"
	ContentOutputAudio ContentKind = "output_audio"
	ContentOutputText  ContentKind = "output_text"
)

// PartKind identifies a streamed response content part.
type PartKind string

// Response content part kinds.
const (
	PartAudio PartKind = "audio"
	PartText  PartKind = "text"
)

// ImageDetail is the input image resolution hint.
type ImageDetail string

// Input image detail levels.
const (
	DetailAuto ImageDetail = "auto"
	DetailHigh ImageDetail = "high"
	DetailLow  ImageDetail = "low"
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

// IncompleteReason explains a response that did not complete.
type IncompleteReason string

// Incomplete response reasons.
const (
	ReasonClientCancelled IncompleteReason = "client_cancelled"
	ReasonContentFilter   IncompleteReason = "content_filter"
	ReasonMaxOutputTokens IncompleteReason = "max_output_tokens"
	ReasonTurnDetected    IncompleteReason = "turn_detected"
)

// UsageKind identifies a transcription usage accounting form.
type UsageKind string

// Transcription usage kinds.
const (
	UsageDuration UsageKind = "duration"
	UsageTokens   UsageKind = "tokens"
)

// RateLimitName identifies a rate limit bucket.
type RateLimitName string

// Rate limit buckets.
const (
	RateLimitRequests RateLimitName = "requests"
	RateLimitTokens   RateLimitName = "tokens"
)

// SessionKind identifies a session object.
type SessionKind string

// Session kinds.
const (
	SessionRealtime      SessionKind = "realtime"
	SessionTranscription SessionKind = "transcription"
)

// AudioEncoding identifies an audio wire encoding.
type AudioEncoding string

// Audio encodings.
const (
	AudioPCM  AudioEncoding = "audio/pcm"
	AudioPCMA AudioEncoding = "audio/pcma"
	AudioPCMU AudioEncoding = "audio/pcmu"
)

// TurnKind identifies a server turn detector.
type TurnKind string

// Turn detectors.
const (
	TurnSemanticVAD TurnKind = "semantic_vad"
	TurnServerVAD   TurnKind = "server_vad"
)

// Eagerness is the semantic voice activity detection responsiveness.
type Eagerness string

// Semantic turn detection eagerness levels.
const (
	EagernessAuto   Eagerness = "auto"
	EagernessHigh   Eagerness = "high"
	EagernessLow    Eagerness = "low"
	EagernessMedium Eagerness = "medium"
)

// NoiseReductionKind identifies the microphone noise profile.
type NoiseReductionKind string

// Noise reduction profiles.
const (
	NoiseFarField  NoiseReductionKind = "far_field"
	NoiseNearField NoiseReductionKind = "near_field"
)

// TranscriptDelay is the transcription emission latency.
type TranscriptDelay string

// Transcription delays.
const (
	TranscriptDelayHigh    TranscriptDelay = "high"
	TranscriptDelayLow     TranscriptDelay = "low"
	TranscriptDelayMedium  TranscriptDelay = "medium"
	TranscriptDelayMinimal TranscriptDelay = "minimal"
	TranscriptDelayXHigh   TranscriptDelay = "xhigh"
)

// ToolKind identifies a tool declaration kind.
type ToolKind string

// Tool kinds.
const (
	ToolFunction ToolKind = "function"
	ToolMCP      ToolKind = "mcp"
)

// ToolChoiceMode is a tool choice string mode.
type ToolChoiceMode string

// Tool choice modes.
const (
	ToolChoiceAuto     ToolChoiceMode = "auto"
	ToolChoiceNone     ToolChoiceMode = "none"
	ToolChoiceRequired ToolChoiceMode = "required"
)

// TruncationStrategy identifies a conversation truncation strategy.
type TruncationStrategy string

// Truncation strategies.
const (
	TruncationAuto           TruncationStrategy = "auto"
	TruncationDisabled       TruncationStrategy = "disabled"
	TruncationRetentionRatio TruncationStrategy = "retention_ratio"
)

// ConversationMode selects whether a response is written to the default
// conversation.
type ConversationMode string

// Response conversation modes.
const (
	ConversationAuto ConversationMode = "auto"
	ConversationNone ConversationMode = "none"
)

// IncludeKind requests an extra field in server outputs.
type IncludeKind string

// IncludeLogprobs requests input transcription log probabilities.
const IncludeLogprobs IncludeKind = "item.input_audio_transcription.logprobs"

// McpErrorKind identifies an MCP failure class.
type McpErrorKind string

// MCP failure classes.
const (
	McpHTTPError          McpErrorKind = "http_error"
	McpProtocolError      McpErrorKind = "protocol_error"
	McpToolExecutionError McpErrorKind = "tool_execution_error"
)

// MaxOutputTokens bounds one response: a token count or the "inf" sentinel.
type MaxOutputTokens struct {
	// Infinite reports the "inf" sentinel.
	Infinite bool
	// Value is the token bound when Infinite is false.
	Value int64
}

// UnmarshalJSON accepts an integer or "inf".
func (m *MaxOutputTokens) UnmarshalJSON(b []byte) error {
	*m = MaxOutputTokens{}
	if bytes.Equal(b, []byte("null")) {
		return nil
	}
	if err := json.Unmarshal(b, &m.Value); err == nil {
		return nil
	}
	m.Value = 0
	var s string
	if err := json.Unmarshal(b, &s); err != nil || s != "inf" {
		return errors.New("realtime max_output_tokens must be an integer or \"inf\"")
	}
	m.Infinite = true
	return nil
}

// MarshalJSON emits an integer or "inf".
func (m MaxOutputTokens) MarshalJSON() ([]byte, error) {
	if m.Infinite {
		return []byte(`"inf"`), nil
	}
	return json.Marshal(m.Value)
}

// Voice names an output voice: a built-in voice name or a custom voice
// identifier.
type Voice struct {
	// Name is a built-in voice name such as "marin".
	Name string
	// ID is a custom voice identifier.
	ID string
}

// UnmarshalJSON accepts a built-in name or a custom voice object.
func (v *Voice) UnmarshalJSON(b []byte) error {
	*v = Voice{}
	if bytes.Equal(b, []byte("null")) {
		return nil
	}
	if err := json.Unmarshal(b, &v.Name); err == nil {
		return nil
	}
	v.Name = ""
	var o struct {
		ID string `json:"id"`
	}
	if err := internal.UnmarshalJSON(b, &o); err != nil || o.ID == "" {
		return errors.New("realtime voice must be a voice name or a voice id object")
	}
	*v = Voice{ID: o.ID}
	return nil
}

// MarshalJSON emits a built-in name or a custom voice object.
func (v Voice) MarshalJSON() ([]byte, error) {
	if v.ID != "" {
		return json.Marshal(struct {
			ID string `json:"id"`
		}{v.ID})
	}
	return json.Marshal(v.Name)
}

// ToolChoice selects how the model chooses tools: a mode string, a forced
// function tool, or a forced MCP tool.
type ToolChoice struct {
	// Mode is set for the string forms.
	Mode ToolChoiceMode
	// Kind is set for the forced tool object forms.
	Kind ToolKind
	// Name is the forced function or MCP tool name.
	Name string
	// ServerLabel is the forced MCP server label.
	ServerLabel string
}

// UnmarshalJSON accepts a mode string or a forced tool object.
func (t *ToolChoice) UnmarshalJSON(b []byte) error {
	*t = ToolChoice{}
	if bytes.Equal(b, []byte("null")) {
		return nil
	}
	if err := json.Unmarshal(b, &t.Mode); err == nil {
		return nil
	}
	t.Mode = ""
	var o struct {
		Type        ToolKind `json:"type"`
		Name        string   `json:"name"`
		ServerLabel string   `json:"server_label"`
	}
	if err := internal.UnmarshalJSON(b, &o); err != nil {
		return errors.New("realtime tool_choice must be a mode or a forced tool object")
	}
	t.Kind = o.Type
	t.Name = o.Name
	t.ServerLabel = o.ServerLabel
	return nil
}

// MarshalJSON emits a mode string or a forced tool object.
func (t ToolChoice) MarshalJSON() ([]byte, error) {
	if t.Mode != "" {
		return json.Marshal(t.Mode)
	}
	return json.Marshal(struct {
		Type        ToolKind `json:"type"`
		Name        string   `json:"name,omitzero"`
		ServerLabel string   `json:"server_label,omitzero"`
	}{t.Kind, t.Name, t.ServerLabel})
}

// Truncation configures conversation truncation: a strategy string or the
// retention ratio object.
type Truncation struct {
	// Strategy is the truncation strategy.
	Strategy TruncationStrategy
	// Ratio is the retained fraction of post-instruction tokens.
	Ratio float64
	// PostInstructionsTokens bounds the tokens kept after instructions.
	PostInstructionsTokens int64
}

// UnmarshalJSON accepts a strategy string or the retention ratio object.
func (t *Truncation) UnmarshalJSON(b []byte) error {
	*t = Truncation{}
	if bytes.Equal(b, []byte("null")) {
		return nil
	}
	if err := json.Unmarshal(b, &t.Strategy); err == nil {
		return nil
	}
	t.Strategy = ""
	var o struct {
		Type           TruncationStrategy `json:"type"`
		RetentionRatio float64            `json:"retention_ratio"`
		TokenLimits    struct {
			PostInstructions int64 `json:"post_instructions"`
		} `json:"token_limits"`
	}
	if err := internal.UnmarshalJSON(b, &o); err != nil {
		return errors.New("realtime truncation must be a strategy or a retention ratio object")
	}
	*t = Truncation{Strategy: o.Type, Ratio: o.RetentionRatio, PostInstructionsTokens: o.TokenLimits.PostInstructions}
	return nil
}

// MarshalJSON emits a strategy string or the retention ratio object.
//
// The object form is emitted whenever a ratio or a token limit is set, even
// with an empty strategy, since the string form cannot carry them.
func (t Truncation) MarshalJSON() ([]byte, error) {
	// Only the retention ratio object can carry a ratio or a token limit.
	object := t.Strategy == "" || t.Strategy == TruncationRetentionRatio
	if !object || (t.Ratio == 0 && t.PostInstructionsTokens == 0) {
		return json.Marshal(t.Strategy)
	}
	strategy := t.Strategy
	if strategy == "" {
		strategy = TruncationRetentionRatio
	}
	return json.Marshal(struct {
		Type           TruncationStrategy `json:"type"`
		RetentionRatio float64            `json:"retention_ratio"`
		TokenLimits    *tokenLimits       `json:"token_limits,omitzero"`
	}{strategy, t.Ratio, tokenLimitsFor(t.PostInstructionsTokens)})
}

type tokenLimits struct {
	PostInstructions int64 `json:"post_instructions"`
}

func tokenLimitsFor(postInstructions int64) *tokenLimits {
	if postInstructions == 0 {
		return nil
	}
	return &tokenLimits{PostInstructions: postInstructions}
}

// Tracing configures session traces: the "auto" sentinel or a granular
// configuration. A nil *Tracing disables tracing.
type Tracing struct {
	// Auto requests the default trace.
	Auto bool
	// WorkflowName names the trace in the Traces Dashboard.
	WorkflowName string
	// GroupID groups traces in the Traces Dashboard.
	GroupID string
	// Metadata is arbitrary trace metadata.
	Metadata json.RawMessage
}

// UnmarshalJSON accepts "auto" or a tracing configuration object.
func (t *Tracing) UnmarshalJSON(b []byte) error {
	*t = Tracing{}
	if bytes.Equal(b, []byte("null")) {
		return nil
	}
	var s string
	if err := json.Unmarshal(b, &s); err == nil {
		if s != "auto" {
			return errors.New("realtime tracing must be \"auto\" or a tracing configuration")
		}
		t.Auto = true
		return nil
	}
	var o struct {
		WorkflowName string          `json:"workflow_name"`
		GroupID      string          `json:"group_id"`
		Metadata     json.RawMessage `json:"metadata"`
	}
	if err := internal.UnmarshalJSON(b, &o); err != nil {
		return errors.New("realtime tracing must be \"auto\" or a tracing configuration")
	}
	*t = Tracing{WorkflowName: o.WorkflowName, GroupID: o.GroupID, Metadata: o.Metadata}
	return nil
}

// MarshalJSON emits "auto" or a tracing configuration object.
func (t Tracing) MarshalJSON() ([]byte, error) {
	if t.Auto {
		return []byte(`"auto"`), nil
	}
	return json.Marshal(struct {
		WorkflowName string          `json:"workflow_name,omitzero"`
		GroupID      string          `json:"group_id,omitzero"`
		Metadata     json.RawMessage `json:"metadata,omitzero"`
	}{t.WorkflowName, t.GroupID, t.Metadata})
}

// Prompt references a prompt template and its variables.
type Prompt struct {
	// ID is the prompt template identifier.
	ID string `json:"id"`
	// Version pins a prompt version.
	Version string `json:"version,omitzero"`
	// Variables are open-ended substitution values: strings or response input
	// objects such as images and files.
	Variables map[string]json.RawMessage `json:"variables,omitzero"`
}

// McpToolFilter restricts MCP tools by name or by read-only annotation.
type McpToolFilter struct {
	// ToolNames lists matching tool names.
	ToolNames []string `json:"tool_names,omitzero"`
	// ReadOnly matches tools annotated as read-only.
	ReadOnly *bool `json:"read_only,omitzero"`
}

// McpAllowedTools lists the permitted tools of an MCP server: a name list or a
// filter object.
type McpAllowedTools struct {
	// Names is set for the name list form.
	Names []string
	// Filter is set for the filter object form.
	Filter *McpToolFilter
}

// UnmarshalJSON accepts a name list or a filter object.
func (m *McpAllowedTools) UnmarshalJSON(b []byte) error {
	*m = McpAllowedTools{}
	if bytes.Equal(b, []byte("null")) {
		return nil
	}
	var names []string
	if err := json.Unmarshal(b, &names); err == nil {
		m.Names = names
		return nil
	}
	var f McpToolFilter
	if err := internal.UnmarshalJSON(b, &f); err != nil {
		return errors.New("realtime allowed_tools must be a name list or a filter object")
	}
	m.Filter = &f
	return nil
}

// MarshalJSON emits a name list or a filter object.
func (m McpAllowedTools) MarshalJSON() ([]byte, error) {
	if m.Filter != nil {
		return json.Marshal(m.Filter)
	}
	return json.Marshal(m.Names)
}

// McpRequireApproval declares which MCP tools require approval: a setting
// string or a filter pair.
type McpRequireApproval struct {
	// Setting is set for the "always" and "never" string forms.
	Setting McpApprovalSetting
	// Filter is set for the filter pair form.
	Filter *McpApprovalFilter
}

// UnmarshalJSON accepts a setting string or a filter pair.
func (m *McpRequireApproval) UnmarshalJSON(b []byte) error {
	*m = McpRequireApproval{}
	if bytes.Equal(b, []byte("null")) {
		return nil
	}
	if err := json.Unmarshal(b, &m.Setting); err == nil {
		return nil
	}
	m.Setting = ""
	var f McpApprovalFilter
	if err := internal.UnmarshalJSON(b, &f); err != nil {
		return errors.New("realtime require_approval must be a setting or a filter object")
	}
	m.Filter = &f
	return nil
}

// MarshalJSON emits a setting string or a filter pair.
func (m McpRequireApproval) MarshalJSON() ([]byte, error) {
	if m.Filter != nil {
		return json.Marshal(m.Filter)
	}
	return json.Marshal(m.Setting)
}

// McpApprovalSetting is a single approval policy for all MCP tools.
type McpApprovalSetting string

// MCP approval settings.
const (
	ApprovalAlways McpApprovalSetting = "always"
	ApprovalNever  McpApprovalSetting = "never"
)

// McpApprovalFilter pairs the tools that always and never require approval.
type McpApprovalFilter struct {
	Always *McpToolFilter `json:"always,omitzero"`
	Never  *McpToolFilter `json:"never,omitzero"`
}

// ClientMessage is one outbound wire event.
type ClientMessage struct {
	EventID        string          `json:"event_id,omitzero"`
	Type           EventType       `json:"type"`
	Session        *SessionConfig  `json:"session,omitzero"`
	Audio          []byte          `json:"audio,omitzero"`
	Item           *Item           `json:"item,omitzero"`
	ResponseID     string          `json:"response_id,omitzero"`
	ItemID         string          `json:"item_id,omitzero"`
	ContentIndex   *int            `json:"content_index,omitzero"`
	AudioEndMS     *int64          `json:"audio_end_ms,omitzero"`
	PreviousItemID string          `json:"previous_item_id,omitzero"`
	Response       *ResponseConfig `json:"response,omitzero"`
}

// SessionUpdateMessage is the session.update command payload. SessionConfig is
// the setup and acknowledgement payload; it cannot describe a partial update.
type SessionUpdateMessage struct {
	EventID string         `json:"event_id,omitzero"`
	Type    EventType      `json:"type"`
	Session *SessionUpdate `json:"session"`
}

// SessionUpdate changes only the fields that are set. An empty non-nil tool
// list or include list clears it; a pointer to an empty instruction string
// clears instructions. It models the OpenAI GA session.update payload except
// model, which cannot change during a session.
//
// Documented at
// https://developers.openai.com/api/reference/resources/realtime/client-events#session.update
type SessionUpdate struct {
	Type              SessionKind        `json:"type,omitzero"`
	Instructions      *string            `json:"instructions,omitempty"`
	ParallelToolCalls *bool              `json:"parallel_tool_calls,omitzero"`
	Prompt            *Prompt            `json:"prompt,omitzero"`
	Tracing           Nullable[Tracing]  `json:"tracing,omitzero"`
	Audio             SessionUpdateAudio `json:"audio,omitzero"`
	Include           []IncludeKind      `json:"include,omitzero"`
	MaxOutputTokens   MaxOutputTokens    `json:"max_output_tokens,omitzero"`
	OutputModalities  Modalities         `json:"output_modalities,omitzero"`
	Reasoning         *ReasoningConfig   `json:"reasoning,omitzero"`
	ToolChoice        ToolChoice         `json:"tool_choice,omitzero"`
	Tools             ToolList           `json:"tools,omitzero"`
	Truncation        *Truncation        `json:"truncation,omitzero"`
}

// SessionUpdateAudio is the partial audio configuration of session.update.
type SessionUpdateAudio struct {
	Input  SessionUpdateAudioInput  `json:"input,omitzero"`
	Output SessionUpdateAudioOutput `json:"output,omitzero"`
}

// SessionUpdateAudioInput is the partial input-audio configuration. The three
// Nullable fields distinguish no update from an explicit JSON null.
type SessionUpdateAudioInput struct {
	Format         AudioFormat              `json:"format,omitzero"`
	NoiseReduction Nullable[NoiseReduction] `json:"noise_reduction,omitzero"`
	Transcription  Nullable[Transcription]  `json:"transcription,omitzero"`
	TurnDetection  Nullable[TurnDetection]  `json:"turn_detection,omitzero"`
}

// SessionUpdateAudioOutput is the partial output-audio configuration.
type SessionUpdateAudioOutput struct {
	Format AudioFormat `json:"format,omitzero"`
	Speed  *float64    `json:"speed,omitzero"`
	Voice  *Voice      `json:"voice,omitzero"`
}

// Nullable is an optional JSON value that can also be explicitly null. The
// field is omitted when Set is false. When Set is true and Value is nil, it
// marshals as null.
type Nullable[T any] struct {
	Set   bool
	Value *T
}

// IsZero reports whether the value is absent.
func (n Nullable[T]) IsZero() bool { return !n.Set }

// MarshalJSON emits the value or null.
func (n Nullable[T]) MarshalJSON() ([]byte, error) {
	if n.Value == nil {
		return []byte("null"), nil
	}
	return json.Marshal(n.Value)
}

// ResponseConfig is the response.create command payload.
type ResponseConfig struct {
	Instructions      string            `json:"instructions,omitzero"`
	Audio             *ResponseAudio    `json:"audio,omitzero"`
	Tools             ToolList          `json:"tools,omitzero"`
	ToolChoice        *ToolChoice       `json:"tool_choice,omitzero"`
	ParallelToolCalls *bool             `json:"parallel_tool_calls,omitzero"`
	Reasoning         *ReasoningConfig  `json:"reasoning,omitzero"`
	MaxOutputTokens   MaxOutputTokens   `json:"max_output_tokens,omitzero"`
	Conversation      ConversationMode  `json:"conversation,omitzero"`
	Metadata          map[string]string `json:"metadata,omitzero"`
	Prompt            *Prompt           `json:"prompt,omitzero"`
	Input             []Item            `json:"input,omitzero"`
	OutputModalities  Modalities        `json:"output_modalities,omitzero"`
}

// SessionConfig is the session setup and acknowledgement payload. The
// acknowledgement echoes fields the session.update command does not accept,
// such as the session identifier, object and expiry.
type SessionConfig struct {
	ID                string           `json:"id,omitzero"`
	Object            ObjectKind       `json:"object,omitzero"`
	Type              SessionKind      `json:"type"`
	Model             string           `json:"model,omitzero"`
	Instructions      string           `json:"instructions"`
	OutputModalities  Modalities       `json:"output_modalities"`
	Audio             AudioConfig      `json:"audio"`
	Tools             ToolList         `json:"tools"`
	ToolChoice        *ToolChoice      `json:"tool_choice,omitzero"`
	Reasoning         *ReasoningConfig `json:"reasoning,omitzero"`
	MaxOutputTokens   MaxOutputTokens  `json:"max_output_tokens,omitzero"`
	ExpiresAt         *int64           `json:"expires_at,omitzero"`
	Include           []IncludeKind    `json:"include,omitzero"`
	Prompt            *Prompt          `json:"prompt,omitzero"`
	Tracing           *Tracing         `json:"tracing,omitzero"`
	Truncation        *Truncation      `json:"truncation,omitzero"`
	ParallelToolCalls *bool            `json:"parallel_tool_calls,omitzero"`
}

// ReasoningConfig configures session reasoning effort.
type ReasoningConfig struct {
	Effort ReasoningEffort `json:"effort,omitzero"`
}

// AudioConfig groups input and output audio settings. Transcription sessions
// carry only the input section.
type AudioConfig struct {
	Input  AudioInput  `json:"input"`
	Output AudioOutput `json:"output"`
}

// AudioInput configures microphone audio.
type AudioInput struct {
	Format         AudioFormat     `json:"format"`
	TurnDetection  *TurnDetection  `json:"turn_detection"`
	Transcription  *Transcription  `json:"transcription"`
	NoiseReduction *NoiseReduction `json:"noise_reduction,omitzero"`
}

// AudioOutput configures generated audio.
type AudioOutput struct {
	Format AudioFormat `json:"format"`
	Voice  Voice       `json:"voice,omitzero"`
	Speed  *float64    `json:"speed,omitzero"`
}

// AudioFormat describes the negotiated wire encoding.
type AudioFormat struct {
	Type AudioEncoding `json:"type"`
	Rate int           `json:"rate,omitzero"`
}

// NoiseReduction configures input audio noise filtering.
type NoiseReduction struct {
	Type NoiseReductionKind `json:"type"`
	// Strength is the suppression strength, from 0 to 1.
	Strength *float64 `json:"strength,omitzero"`
}

// TurnDetection configures server-side turn detection. Server and semantic
// detection share this shape; each sets its own fields.
type TurnDetection struct {
	Type              TurnKind  `json:"type"`
	CreateResponse    bool      `json:"create_response"`
	InterruptResponse bool      `json:"interrupt_response"`
	Threshold         *float64  `json:"threshold,omitzero"`
	PrefixPaddingMS   *int64    `json:"prefix_padding_ms,omitzero"`
	SilenceDurationMS *int64    `json:"silence_duration_ms,omitzero"`
	IdleTimeoutMS     *int64    `json:"idle_timeout_ms,omitzero"`
	Eagerness         Eagerness `json:"eagerness,omitzero"`
}

// Transcription configures input transcription.
type Transcription struct {
	Model     string          `json:"model"`
	Language  string          `json:"language,omitzero"`
	Languages []string        `json:"languages,omitzero"`
	Keywords  []string        `json:"keywords,omitzero"`
	Prompt    string          `json:"prompt,omitzero"`
	Delay     TranscriptDelay `json:"delay,omitzero"`
}

// Tool is a function or MCP tool declaration.
type Tool struct {
	Type        ToolKind        `json:"type"`
	Name        string          `json:"name,omitzero"`
	Description string          `json:"description,omitzero"`
	Parameters  json.RawMessage `json:"parameters,omitzero"`
	// MCP tool fields, set when Type is ToolMCP.
	ServerLabel       string              `json:"server_label,omitzero"`
	ServerURL         string              `json:"server_url,omitzero"`
	ServerDescription string              `json:"server_description,omitzero"`
	Authorization     string              `json:"authorization,omitzero"`
	TunnelID          string              `json:"tunnel_id,omitzero"`
	ConnectorID       string              `json:"connector_id,omitzero"`
	Headers           map[string]string   `json:"headers,omitzero"`
	AllowedTools      *McpAllowedTools    `json:"allowed_tools,omitzero"`
	RequireApproval   *McpRequireApproval `json:"require_approval,omitzero"`
	AllowedCallers    []string            `json:"allowed_callers,omitzero"`
	DeferLoading      *bool               `json:"defer_loading,omitzero"`
}

// Item is one conversation item: a message, a function call, its output, an
// MCP item, or a response input reference.
type Item struct {
	Type      ItemType             `json:"type"`
	ID        string               `json:"id,omitzero"`
	Object    ObjectKind           `json:"object,omitzero"`
	Status    ItemStatus           `json:"status,omitzero"`
	Role      Role                 `json:"role,omitzero"`
	Content   boundedList[Content] `json:"content,omitzero"`
	CallID    string               `json:"call_id,omitzero"`
	Output    string               `json:"output,omitzero"`
	Name      string               `json:"name,omitzero"`
	Arguments string               `json:"arguments,omitzero"`
	// Phase is the assistant message phase the server assigns to response
	// items. The published SDK types omit it.
	Phase string `json:"phase,omitzero"`
	// MCP item fields.
	ServerLabel       string               `json:"server_label,omitzero"`
	ApprovalRequestID string               `json:"approval_request_id,omitzero"`
	Approve           *bool                `json:"approve,omitzero"`
	Reason            string               `json:"reason,omitzero"`
	Tools             boundedList[McpTool] `json:"tools,omitzero"`
	Error             *McpCallError        `json:"error,omitzero"`
}

// OutputItem is a conversation item emitted by a response. It has the same wire
// shape as Item.
type OutputItem = Item

// Content is one message content part.
type Content struct {
	Type       ContentKind `json:"type"`
	Text       string      `json:"text,omitzero"`
	Transcript string      `json:"transcript,omitzero"`
	Audio      string      `json:"audio,omitzero"`
	ImageURL   string      `json:"image_url,omitzero"`
	Detail     ImageDetail `json:"detail,omitzero"`
}

// ContentPart is one streamed response content part.
type ContentPart struct {
	Type       PartKind `json:"type"`
	Text       string   `json:"text,omitzero"`
	Transcript string   `json:"transcript,omitzero"`
	Audio      string   `json:"audio,omitzero"`
}

// McpTool is one tool exposed by an MCP server.
type McpTool struct {
	Name        string `json:"name"`
	Description string `json:"description,omitzero"`
	// InputSchema is the JSON schema of the tool input.
	InputSchema json.RawMessage `json:"input_schema,omitzero"`
	// Annotations are open-ended MCP tool annotations.
	Annotations json.RawMessage `json:"annotations,omitzero"`
}

// McpCallError is an MCP tool call failure.
type McpCallError struct {
	Type    McpErrorKind `json:"type"`
	Code    int64        `json:"code,omitzero"`
	Message string       `json:"message,omitzero"`
}

// ServerMessage is one inbound wire event.
type ServerMessage struct {
	Type           EventType               `json:"type"`
	EventID        string                  `json:"event_id"`
	Obfuscation    string                  `json:"obfuscation,omitzero"`
	Conversation   *Conversation           `json:"conversation,omitzero"`
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
	RateLimits     []WireRateLimit         `json:"rate_limits"`
	PreviousItemID string                  `json:"previous_item_id"`
	Item           *Item                   `json:"item"`
	Part           *ContentPart            `json:"part"`
	Usage          *TranscriptionUsage     `json:"usage"`
	Languages      []TranscriptionLanguage `json:"languages"`
	Logprobs       []LogProb               `json:"logprobs"`
	AudioStartMS   *int64                  `json:"audio_start_ms,omitzero"`
	AudioEndMS     *int64                  `json:"audio_end_ms,omitzero"`
	// DTMF and transcription segment fields.
	Event      string   `json:"event,omitzero"`
	ReceivedAt int64    `json:"received_at,omitzero"`
	ID         string   `json:"id,omitzero"`
	Speaker    string   `json:"speaker,omitzero"`
	Start      *float64 `json:"start,omitzero"`
	End        *float64 `json:"end,omitzero"`
}

// Conversation is the conversation resource.
type Conversation struct {
	ID     string     `json:"id"`
	Object ObjectKind `json:"object,omitzero"`
}

// Response is a model response.
type Response struct {
	ID               string                  `json:"id"`
	Object           ObjectKind              `json:"object,omitzero"`
	Status           ResponseStatus          `json:"status"`
	StatusDetails    *StatusDetails          `json:"status_details,omitzero"`
	Output           boundedList[OutputItem] `json:"output"`
	Usage            *Usage                  `json:"usage"`
	ConversationID   string                  `json:"conversation_id,omitzero"`
	OutputModalities Modalities              `json:"output_modalities,omitzero"`
	Audio            *ResponseAudio          `json:"audio,omitzero"`
	MaxOutputTokens  MaxOutputTokens         `json:"max_output_tokens,omitzero"`
	Metadata         map[string]string       `json:"metadata,omitzero"`
}

// StatusDetails explains why a response did not reach a clean completion.
type StatusDetails struct {
	Type   ResponseStatus   `json:"type"`
	Reason IncompleteReason `json:"reason,omitzero"`
	Error  *StatusError     `json:"error,omitzero"`
}

// StatusError describes a failed response.
type StatusError struct {
	Type    string `json:"type,omitzero"`
	Code    string `json:"code,omitzero"`
	Message string `json:"message,omitzero"`
}

// ResponseAudio is the audio configuration a response used.
type ResponseAudio struct {
	Output ResponseAudioOutput `json:"output"`
}

// ResponseAudioOutput is the generated audio configuration.
type ResponseAudioOutput struct {
	Format AudioFormat `json:"format"`
	Voice  Voice       `json:"voice,omitzero"`
}

// Usage is response token accounting.
type Usage struct {
	InputTokens   int64              `json:"input_tokens"`
	OutputTokens  int64              `json:"output_tokens"`
	TotalTokens   int64              `json:"total_tokens"`
	InputDetails  InputTokenDetails  `json:"input_token_details"`
	OutputDetails OutputTokenDetails `json:"output_token_details"`
}

// InputTokenDetails splits response input tokens.
type InputTokenDetails struct {
	CachedTokens  int64               `json:"cached_tokens,omitzero"`
	AudioTokens   int64               `json:"audio_tokens,omitzero"`
	ImageTokens   int64               `json:"image_tokens,omitzero"`
	TextTokens    int64               `json:"text_tokens,omitzero"`
	CachedDetails *CachedTokenDetails `json:"cached_tokens_details,omitzero"`
}

// CachedTokenDetails splits cached input tokens.
type CachedTokenDetails struct {
	AudioTokens int64 `json:"audio_tokens,omitzero"`
	ImageTokens int64 `json:"image_tokens,omitzero"`
	TextTokens  int64 `json:"text_tokens,omitzero"`
}

// OutputTokenDetails splits response output tokens.
type OutputTokenDetails struct {
	AudioTokens int64 `json:"audio_tokens,omitzero"`
	TextTokens  int64 `json:"text_tokens,omitzero"`
	// ReasoningTokens is reported by reasoning-capable realtime models. The
	// published SDK types omit it.
	ReasoningTokens int64 `json:"reasoning_tokens,omitzero"`
}

// TranscriptionUsage is the input transcription billing of one item: either
// token or duration accounting.
type TranscriptionUsage struct {
	Type         UsageKind                  `json:"type"`
	InputTokens  int64                      `json:"input_tokens,omitzero"`
	OutputTokens int64                      `json:"output_tokens,omitzero"`
	TotalTokens  int64                      `json:"total_tokens,omitzero"`
	InputDetails *TranscriptionInputDetails `json:"input_token_details,omitzero"`
	Seconds      float64                    `json:"seconds,omitzero"`
}

// TranscriptionInputDetails splits the tokens billed for input transcription.
type TranscriptionInputDetails struct {
	AudioTokens int64 `json:"audio_tokens,omitzero"`
	TextTokens  int64 `json:"text_tokens,omitzero"`
}

// TranscriptionLanguage is a language detected in transcribed audio.
type TranscriptionLanguage struct {
	Code string `json:"code"`
}

// LogProb is one transcription token log probability.
type LogProb struct {
	Token   string  `json:"token"`
	Logprob float64 `json:"logprob"`
	Bytes   []int64 `json:"bytes"`
}

// WireRateLimit is one rate-limit snapshot from rate_limits.updated.
type WireRateLimit struct {
	Name         RateLimitName `json:"name"`
	Limit        int64         `json:"limit"`
	Remaining    int64         `json:"remaining"`
	ResetSeconds float64       `json:"reset_seconds"`
}

// WireError is a provider error payload.
type WireError struct {
	Type    string `json:"type,omitzero"`
	Code    string `json:"code,omitzero"`
	Message string `json:"message,omitzero"`
	Param   string `json:"param,omitzero"`
	EventID string `json:"event_id,omitzero"`
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

// Modalities bounds the two output modalities a provider may negotiate.
// Each provider rejects combinations it does not implement.
type Modalities []genai.Modality

// UnmarshalJSON bounds the negotiated output modalities.
func (m *Modalities) UnmarshalJSON(b []byte) error {
	v, err := parseList[genai.Modality](b, 2)
	*m = v
	return err
}

// ToolList bounds peer-supplied tool declarations.
type ToolList []Tool

// Init converts portable tool declarations to OpenAI-wire tools.
func (l *ToolList) Init(tools []genai.ToolDef) error {
	if len(tools) > maxTools {
		return errors.New("realtime tool declarations exceed limit")
	}
	decls := make([]Tool, 0, len(tools))
	for i := range tools {
		v := &tools[i]
		schema, err := v.GetInputSchema()
		if err != nil {
			return err
		}
		decls = append(decls, Tool{Type: ToolFunction, Name: v.Name, Description: v.Description, Parameters: schema})
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
	v, err := parseList[Tool](b, maxTools)
	*l = v
	return err
}

// boundedList stops allocation amplification from peer-supplied arrays before
// decoding each next element. Both output items and declarations have 1024 slots.
type boundedList[T any] []T

func (l *boundedList[T]) UnmarshalJSON(b []byte) error {
	v, err := parseList[T](b, maxTools)
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
