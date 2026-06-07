// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Wire types for Pi's custom JSONL protocol over stdin/stdout.
//
// Pi uses a type-dispatched JSONL protocol (not JSON-RPC 2.0). Commands are
// sent on stdin, responses and events are emitted on stdout, each as a single
// JSON line terminated by LF.
//
// Type names follow the upstream definitions in:
//
//   - packages/coding-agent/src/modes/rpc/rpc-types.ts — RPC command/response types
//   - packages/agent/src/types.ts — AgentEvent types
//   - packages/ai/src/types.ts — AssistantMessage, AssistantMessageEvent, Model
//
// Source: https://github.com/badlogic/pi-mono

package pi

import (
	"bytes"
	"encoding/json"
	"strings"
)

// ============================================================
// Shared types: enums, routing probes.
// ============================================================

// CommandType is the type discriminator for commands sent to Pi on stdin.
type CommandType string

// Command type constants.
const (
	// Prompting.
	CmdPrompt   CommandType = "prompt"
	CmdSteer    CommandType = "steer"
	CmdFollowUp CommandType = "follow_up"
	CmdAbort    CommandType = "abort"

	// Session.
	CmdNewSession           CommandType = "new_session"
	CmdGetState             CommandType = "get_state"
	CmdGetSessionStats      CommandType = "get_session_stats"
	CmdExportHTML           CommandType = "export_html"
	CmdSwitchSession        CommandType = "switch_session"
	CmdFork                 CommandType = "fork"
	CmdGetForkMessages      CommandType = "get_fork_messages"
	CmdGetLastAssistantText CommandType = "get_last_assistant_text"
	CmdSetSessionName       CommandType = "set_session_name"
	CmdGetMessages          CommandType = "get_messages"
	CmdGetCommands          CommandType = "get_commands"

	// Model.
	CmdSetModel   CommandType = "set_model"
	CmdCycleModel CommandType = "cycle_model"
	CmdGetModels  CommandType = "get_available_models"

	// Thinking.
	CmdSetThinking   CommandType = "set_thinking_level"
	CmdCycleThinking CommandType = "cycle_thinking_level"

	// Queue modes.
	CmdSetSteeringMode CommandType = "set_steering_mode"
	CmdSetFollowUpMode CommandType = "set_follow_up_mode"

	// Compaction.
	CmdCompact           CommandType = "compact"
	CmdSetAutoCompaction CommandType = "set_auto_compaction"

	// Retry.
	CmdSetAutoRetry CommandType = "set_auto_retry"
	CmdAbortRetry   CommandType = "abort_retry"

	// Bash.
	CmdBash      CommandType = "bash"
	CmdAbortBash CommandType = "abort_bash"
)

// EventType is the type discriminator for events emitted by Pi on stdout.
type EventType string

// Event type constants.
const (
	EventAgentStart     EventType = "agent_start"
	EventAgentEnd       EventType = "agent_end"
	EventTurnStart      EventType = "turn_start"
	EventTurnEnd        EventType = "turn_end"
	EventMessageStart   EventType = "message_start"
	EventMessageUpdate  EventType = "message_update"
	EventMessageEnd     EventType = "message_end"
	EventToolExecStart  EventType = "tool_execution_start"
	EventToolExecUpdate EventType = "tool_execution_update"
	EventToolExecEnd    EventType = "tool_execution_end"
	EventResponse       EventType = "response"
	EventExtensionUI    EventType = "extension_ui_request"
)

// DeltaType is the type discriminator for AssistantMessageEvent deltas.
type DeltaType string

// Delta type constants.
const (
	DeltaStart      DeltaType = "start"
	DeltaTextStart  DeltaType = "text_start"
	DeltaTextDelta  DeltaType = "text_delta"
	DeltaTextEnd    DeltaType = "text_end"
	DeltaThinkStart DeltaType = "thinking_start"
	DeltaThinkDelta DeltaType = "thinking_delta"
	DeltaThinkEnd   DeltaType = "thinking_end"
	DeltaToolStart  DeltaType = "toolcall_start"
	DeltaToolDelta  DeltaType = "toolcall_delta"
	DeltaToolEnd    DeltaType = "toolcall_end"
	DeltaDone       DeltaType = "done"
	DeltaError      DeltaType = "error"
)

// StopReason is the reason the model stopped generating.
type StopReason string

// Stop reason constants.
const (
	StopReasonStop    StopReason = "stop"
	StopReasonLength  StopReason = "length"
	StopReasonToolUse StopReason = "toolUse"
	StopReasonError   StopReason = "error"
	StopReasonAborted StopReason = "aborted"
)

// ThinkingLevel controls reasoning depth.
type ThinkingLevel string

// Thinking level constants.
const (
	ThinkingOff     ThinkingLevel = "off"
	ThinkingMinimal ThinkingLevel = "minimal"
	ThinkingLow     ThinkingLevel = "low"
	ThinkingMedium  ThinkingLevel = "medium"
	ThinkingHigh    ThinkingLevel = "high"
	ThinkingXHigh   ThinkingLevel = "xhigh"
)

// ExtensionUIMethod is the method discriminator for extension UI requests.
type ExtensionUIMethod string

// Extension UI method constants.
const (
	UIMethodSelect        ExtensionUIMethod = "select"
	UIMethodConfirm       ExtensionUIMethod = "confirm"
	UIMethodInput         ExtensionUIMethod = "input"
	UIMethodEditor        ExtensionUIMethod = "editor"
	UIMethodNotify        ExtensionUIMethod = "notify"
	UIMethodSetStatus     ExtensionUIMethod = "setStatus"
	UIMethodSetWidget     ExtensionUIMethod = "setWidget"
	UIMethodSetTitle      ExtensionUIMethod = "setTitle"
	UIMethodSetEditorText ExtensionUIMethod = "set_editor_text"
)

// QueueMode controls how steering or follow-up messages are processed.
type QueueMode string

// Queue mode constants.
const (
	QueueModeAll        QueueMode = "all"
	QueueModeOneAtATime QueueMode = "one-at-a-time"
)

// Role is the message role discriminator.
type Role string

// Role constants.
const (
	RoleUser       Role = "user"
	RoleAssistant  Role = "assistant"
	RoleToolResult Role = "toolResult"
)

// ContentBlockType is the content block type discriminator.
type ContentBlockType string

// Content block type constants.
const (
	ContentText     ContentBlockType = "text"
	ContentThinking ContentBlockType = "thinking"
	ContentToolCall ContentBlockType = "toolCall"
	ContentImage    ContentBlockType = "image"
)

// StreamingBehavior controls how a prompt interacts with an ongoing generation.
type StreamingBehavior string

// Streaming behavior constants.
const (
	StreamSteer    StreamingBehavior = "steer"
	StreamFollowUp StreamingBehavior = "followUp"
)

// NotifyType is the severity level for notify extension UI requests.
type NotifyType string

// Notify type constants.
const (
	NotifyInfo    NotifyType = "info"
	NotifyWarning NotifyType = "warning"
	NotifyError   NotifyType = "error"
)

// WidgetPlacement controls where a widget is placed relative to the editor.
type WidgetPlacement string

// Widget placement constants.
const (
	WidgetAboveEditor WidgetPlacement = "aboveEditor"
	WidgetBelowEditor WidgetPlacement = "belowEditor"
)

// SlashCommandSource is the origin of a slash command.
type SlashCommandSource string

// Slash command source constants.
const (
	CommandSourceExtension SlashCommandSource = "extension"
	CommandSourcePrompt    SlashCommandSource = "prompt"
	CommandSourceSkill     SlashCommandSource = "skill"
)

// ExtensionUIResponseType is the fixed type discriminator for all extension UI responses.
const ExtensionUIResponseType = "extension_ui_response"

// ---------- Routing probe ----------

// LineProbe extracts routing fields from a JSONL line to determine its kind.
type LineProbe struct {
	Type    EventType   `json:"type"`
	Command CommandType `json:"command,omitzero"`
	ID      string      `json:"id,omitzero"`
	Success *bool       `json:"success,omitzero"`
}

// ============================================================
// Input types: commands sent to Pi (stdin).
// ============================================================

// ---------- Prompting ----------

// PromptCmd sends a user message.
type PromptCmd struct {
	ID                string            `json:"id,omitzero"`
	Type              CommandType       `json:"type"`
	Message           string            `json:"message"`
	Images            []ImageContent    `json:"images,omitzero"`
	StreamingBehavior StreamingBehavior `json:"streamingBehavior,omitzero"`
}

// ImageContent is an inline image in base64.
type ImageContent struct {
	Type     ContentBlockType `json:"type"`
	Data     string           `json:"data"`
	MimeType string           `json:"mimeType"`
}

// SteerCmd sends a steering message mid-run.
type SteerCmd struct {
	ID      string         `json:"id,omitzero"`
	Type    CommandType    `json:"type"`
	Message string         `json:"message"`
	Images  []ImageContent `json:"images,omitzero"`
}

// FollowUpCmd sends a follow-up message after the agent finishes.
type FollowUpCmd struct {
	ID      string         `json:"id,omitzero"`
	Type    CommandType    `json:"type"`
	Message string         `json:"message"`
	Images  []ImageContent `json:"images,omitzero"`
}

// AbortCmd cancels the current generation.
type AbortCmd struct {
	ID   string      `json:"id,omitzero"`
	Type CommandType `json:"type"`
}

// ---------- Session ----------

// NewSessionCmd starts a fresh session.
type NewSessionCmd struct {
	ID            string      `json:"id,omitzero"`
	Type          CommandType `json:"type"`
	ParentSession string      `json:"parentSession,omitzero"`
}

// GetStateCmd requests current session state.
type GetStateCmd struct {
	ID   string      `json:"id,omitzero"`
	Type CommandType `json:"type"`
}

// GetSessionStatsCmd requests session statistics.
type GetSessionStatsCmd struct {
	ID   string      `json:"id,omitzero"`
	Type CommandType `json:"type"`
}

// ExportHTMLCmd exports the session as HTML.
type ExportHTMLCmd struct {
	ID         string      `json:"id,omitzero"`
	Type       CommandType `json:"type"`
	OutputPath string      `json:"outputPath,omitzero"`
}

// SwitchSessionCmd switches to a different session.
type SwitchSessionCmd struct {
	ID          string      `json:"id,omitzero"`
	Type        CommandType `json:"type"`
	SessionPath string      `json:"sessionPath"`
}

// ForkCmd forks the session at a specific entry.
type ForkCmd struct {
	ID      string      `json:"id,omitzero"`
	Type    CommandType `json:"type"`
	EntryID string      `json:"entryId"`
}

// GetForkMessagesCmd gets messages available for forking.
type GetForkMessagesCmd struct {
	ID   string      `json:"id,omitzero"`
	Type CommandType `json:"type"`
}

// GetLastAssistantTextCmd gets the last assistant message text.
type GetLastAssistantTextCmd struct {
	ID   string      `json:"id,omitzero"`
	Type CommandType `json:"type"`
}

// SetSessionNameCmd sets the session name.
type SetSessionNameCmd struct {
	ID   string      `json:"id,omitzero"`
	Type CommandType `json:"type"`
	Name string      `json:"name"`
}

// GetMessagesCmd gets all messages in the session.
type GetMessagesCmd struct {
	ID   string      `json:"id,omitzero"`
	Type CommandType `json:"type"`
}

// GetCommandsCmd gets available slash commands.
type GetCommandsCmd struct {
	ID   string      `json:"id,omitzero"`
	Type CommandType `json:"type"`
}

// ---------- Model ----------

// SetModelCmd switches the active model.
type SetModelCmd struct {
	ID       string      `json:"id,omitzero"`
	Type     CommandType `json:"type"`
	Provider string      `json:"provider"`
	ModelID  string      `json:"modelId"`
}

// CycleModelCmd cycles to the next model.
type CycleModelCmd struct {
	ID   string      `json:"id,omitzero"`
	Type CommandType `json:"type"`
}

// GetModelsCmd requests the list of available models.
type GetModelsCmd struct {
	ID   string      `json:"id,omitzero"`
	Type CommandType `json:"type"`
}

// ---------- Thinking ----------

// SetThinkingCmd sets the thinking level.
type SetThinkingCmd struct {
	ID    string        `json:"id,omitzero"`
	Type  CommandType   `json:"type"`
	Level ThinkingLevel `json:"level"`
}

// CycleThinkingCmd cycles to the next thinking level.
type CycleThinkingCmd struct {
	ID   string      `json:"id,omitzero"`
	Type CommandType `json:"type"`
}

// ---------- Queue modes ----------

// SetSteeringModeCmd sets the steering queue mode.
type SetSteeringModeCmd struct {
	ID   string      `json:"id,omitzero"`
	Type CommandType `json:"type"`
	Mode QueueMode   `json:"mode"`
}

// SetFollowUpModeCmd sets the follow-up queue mode.
type SetFollowUpModeCmd struct {
	ID   string      `json:"id,omitzero"`
	Type CommandType `json:"type"`
	Mode QueueMode   `json:"mode"`
}

// ---------- Compaction ----------

// CompactCmd triggers compaction with optional custom instructions.
type CompactCmd struct {
	ID                 string      `json:"id,omitzero"`
	Type               CommandType `json:"type"`
	CustomInstructions string      `json:"customInstructions,omitzero"`
}

// SetAutoCompactionCmd enables or disables automatic compaction.
type SetAutoCompactionCmd struct {
	ID      string      `json:"id,omitzero"`
	Type    CommandType `json:"type"`
	Enabled bool        `json:"enabled"`
}

// ---------- Retry ----------

// SetAutoRetryCmd enables or disables automatic retry.
type SetAutoRetryCmd struct {
	ID      string      `json:"id,omitzero"`
	Type    CommandType `json:"type"`
	Enabled bool        `json:"enabled"`
}

// AbortRetryCmd aborts the current retry.
type AbortRetryCmd struct {
	ID   string      `json:"id,omitzero"`
	Type CommandType `json:"type"`
}

// ---------- Bash ----------

// BashCmd executes a bash command.
type BashCmd struct {
	ID      string      `json:"id,omitzero"`
	Type    CommandType `json:"type"`
	Command string      `json:"command"`
}

// AbortBashCmd aborts the current bash command.
type AbortBashCmd struct {
	ID   string      `json:"id,omitzero"`
	Type CommandType `json:"type"`
}

// ---------- Extension UI responses (stdin) ----------

// ExtensionUIResponseValue is sent back for select/input/editor requests.
type ExtensionUIResponseValue struct {
	Type  string `json:"type"`
	ID    string `json:"id"`
	Value string `json:"value"`
}

// ExtensionUIResponseConfirm is sent back for confirm requests.
type ExtensionUIResponseConfirm struct {
	Type      string `json:"type"`
	ID        string `json:"id"`
	Confirmed bool   `json:"confirmed"`
}

// ExtensionUIResponseCancelled is sent back when a UI request is cancelled.
type ExtensionUIResponseCancelled struct {
	Type      string `json:"type"`
	ID        string `json:"id"`
	Cancelled bool   `json:"cancelled"`
}

// ============================================================
// Output types: responses and events from Pi (stdout).
// ============================================================

// ---------- Response envelope ----------

// Response is the generic response wrapper. Dispatch on Command field.
type Response struct {
	ID      string          `json:"id,omitzero"`
	Type    EventType       `json:"type"`
	Command CommandType     `json:"command"`
	Success bool            `json:"success"`
	Error   string          `json:"error,omitzero"`
	Data    json.RawMessage `json:"data,omitzero"`
}

// ---------- Response data payloads ----------

// ModelsData is the data payload for get_available_models response.
type ModelsData struct {
	Models []Model `json:"models"`
}

// StateData is the data payload for get_state response.
type StateData struct {
	Model                 *Model        `json:"model,omitzero"`
	ThinkingLevel         ThinkingLevel `json:"thinkingLevel"`
	IsStreaming           bool          `json:"isStreaming"`
	IsCompacting          bool          `json:"isCompacting"`
	SteeringMode          QueueMode     `json:"steeringMode"`
	FollowUpMode          QueueMode     `json:"followUpMode"`
	SessionFile           string        `json:"sessionFile,omitzero"`
	SessionID             string        `json:"sessionId"`
	SessionName           string        `json:"sessionName,omitzero"`
	AutoCompactionEnabled bool          `json:"autoCompactionEnabled"`
	MessageCount          int           `json:"messageCount"`
	PendingMessageCount   int           `json:"pendingMessageCount"`
}

// NewSessionData is the data payload for new_session response.
type NewSessionData struct {
	Cancelled bool `json:"cancelled"`
}

// CycleModelData is the data payload for cycle_model response.
type CycleModelData struct {
	Model         *Model        `json:"model,omitzero"`
	ThinkingLevel ThinkingLevel `json:"thinkingLevel,omitzero"`
	IsScoped      bool          `json:"isScoped,omitzero"`
}

// CycleThinkingData is the data payload for cycle_thinking_level response.
type CycleThinkingData struct {
	Level ThinkingLevel `json:"level"`
}

// BashData is the data payload for bash response. Fields are opaque.
type BashData struct {
	json.RawMessage
}

// ExportHTMLData is the data payload for export_html response.
type ExportHTMLData struct {
	Path string `json:"path"`
}

// SwitchSessionData is the data payload for switch_session response.
type SwitchSessionData struct {
	Cancelled bool `json:"cancelled"`
}

// ForkData is the data payload for fork response.
type ForkData struct {
	Text      string `json:"text"`
	Cancelled bool   `json:"cancelled"`
}

// ForkMessagesData is the data payload for get_fork_messages response.
type ForkMessagesData struct {
	Messages []ForkMessage `json:"messages"`
}

// ForkMessage is a single entry in the fork messages list.
type ForkMessage struct {
	EntryID string `json:"entryId"`
	Text    string `json:"text"`
}

// LastAssistantTextData is the data payload for get_last_assistant_text response.
type LastAssistantTextData struct {
	Text *string `json:"text"` // nullable
}

// GetMessagesData is the data payload for get_messages response.
type GetMessagesData struct {
	Messages []AgentMessage `json:"messages"`
}

// GetCommandsData is the data payload for get_commands response.
type GetCommandsData struct {
	Commands []SlashCommand `json:"commands"`
}

// SlashCommand is a command available for invocation via prompt.
type SlashCommand struct {
	Name        string             `json:"name"`
	Description string             `json:"description,omitzero"`
	Source      SlashCommandSource `json:"source"`
	SourceInfo  SourceInfo         `json:"sourceInfo"`
}

// SourceInfo describes the origin of a slash command. Fields are opaque.
type SourceInfo struct {
	json.RawMessage
}

// SessionStatsData is the data payload for get_session_stats response.
type SessionStatsData struct {
	SessionFile   string        `json:"sessionFile,omitzero"`
	SessionID     string        `json:"sessionId"`
	UserMessages  int           `json:"userMessages"`
	AssistantMsgs int           `json:"assistantMessages"`
	ToolCalls     int           `json:"toolCalls"`
	ToolResults   int           `json:"toolResults"`
	TotalMessages int           `json:"totalMessages"`
	Tokens        SessionTokens `json:"tokens"`
	Cost          float64       `json:"cost"`
	ContextUsage  ContextUsage  `json:"contextUsage,omitzero"`
}

// SessionTokens holds aggregated token counts for a session.
type SessionTokens struct {
	Input      int64 `json:"input"`
	Output     int64 `json:"output"`
	CacheRead  int64 `json:"cacheRead"`
	CacheWrite int64 `json:"cacheWrite"`
	Total      int64 `json:"total"`
}

// ContextUsage reports the estimated context window utilization.
type ContextUsage struct {
	// Tokens is the estimated number of tokens consumed in the current
	// session context. Zero when unavailable (e.g. after compaction).
	Tokens int64 `json:"tokens,omitzero"`
	// ContextWindow is the model's maximum context window size.
	ContextWindow int64 `json:"contextWindow"`
	// Percent is the context usage as a percentage (0-100). Zero when
	// Tokens is unavailable.
	Percent float64 `json:"percent,omitzero"`
}

// CompactData is the data payload for compact response. Fields are opaque.
type CompactData struct {
	json.RawMessage
}

// ---------- Model ----------

// Model matches the upstream Model<Api> shape.
//
// It implements genai.Model.
type Model struct {
	ID            string    `json:"id"`
	Name          string    `json:"name"`
	API           string    `json:"api"`
	Provider      string    `json:"provider"`
	BaseURL       string    `json:"baseUrl"`
	Reasoning     bool      `json:"reasoning"`
	Input         []string  `json:"input"`
	ContextWindow int64     `json:"contextWindow"`
	MaxTokens     int64     `json:"maxTokens"`
	Cost          ModelCost `json:"cost"`
}

// GetID returns the provider-qualified model ID (e.g. "cerebras/gpt-oss-120b").
func (m *Model) GetID() string { return m.Provider + "/" + m.ID }

// String returns the model's display name.
func (m *Model) String() string { return m.Name }

// Context returns the context window size in tokens.
func (m *Model) Context() int64 { return m.ContextWindow }

// ModelCost holds per-million-token costs.
type ModelCost struct {
	Input      float64 `json:"input"`
	Output     float64 `json:"output"`
	CacheRead  float64 `json:"cacheRead"`
	CacheWrite float64 `json:"cacheWrite"`
}

// ---------- Agent events ----------

// AgentStartEvent is emitted when the agent begins processing.
type AgentStartEvent struct {
	Type EventType `json:"type"`
}

// AgentEndEvent is emitted when the agent finishes. Contains accumulated messages.
type AgentEndEvent struct {
	Type     EventType      `json:"type"`
	Messages []AgentMessage `json:"messages"`
}

// TurnStartEvent is emitted when a turn begins.
type TurnStartEvent struct {
	Type EventType `json:"type"`
}

// TurnEndEvent is emitted when a turn finishes.
type TurnEndEvent struct {
	Type        EventType         `json:"type"`
	Message     AgentMessage      `json:"message"`
	ToolResults []json.RawMessage `json:"toolResults,omitzero"`
}

// MessageStartEvent is emitted when a message begins.
type MessageStartEvent struct {
	Type    EventType    `json:"type"`
	Message AgentMessage `json:"message"`
}

// MessageUpdateEvent is emitted during streaming with a delta.
type MessageUpdateEvent struct {
	Type                  EventType             `json:"type"`
	Message               AgentMessage          `json:"message"`
	AssistantMessageEvent AssistantMessageEvent `json:"assistantMessageEvent"`
}

// MessageEndEvent is emitted when a message is complete.
type MessageEndEvent struct {
	Type    EventType    `json:"type"`
	Message AgentMessage `json:"message"`
}

// ToolExecStartEvent is emitted when a tool begins execution.
type ToolExecStartEvent struct {
	Type       EventType       `json:"type"`
	ToolCallID string          `json:"toolCallId"`
	ToolName   string          `json:"toolName"`
	Args       json.RawMessage `json:"args"`
}

// ToolExecUpdateEvent is emitted during tool execution with progress.
type ToolExecUpdateEvent struct {
	Type          EventType       `json:"type"`
	ToolCallID    string          `json:"toolCallId"`
	ToolName      string          `json:"toolName"`
	Args          json.RawMessage `json:"args"`
	PartialResult ToolExecResult  `json:"partialResult"`
}

// ToolExecEndEvent is emitted when a tool finishes execution.
type ToolExecEndEvent struct {
	Type       EventType      `json:"type"`
	ToolCallID string         `json:"toolCallId"`
	ToolName   string         `json:"toolName"`
	Result     ToolExecResult `json:"result"`
	IsError    bool           `json:"isError"`
}

// ToolExecResult is the result payload for tool_execution_update and
// tool_execution_end events.
//
// It contains an array of content blocks with the tool output.
type ToolExecResult struct {
	Content ContentBlocks `json:"content"`
}

// Text extracts and concatenates all text content from the result blocks.
func (r *ToolExecResult) Text() string {
	var b strings.Builder
	for i := range r.Content {
		if r.Content[i].Text != "" {
			b.WriteString(r.Content[i].Text)
		}
	}
	return b.String()
}

// ---------- Extension UI events ----------

// ExtensionUIRequest is emitted when an extension needs user input.
type ExtensionUIRequest struct {
	Type            EventType         `json:"type"`
	ID              string            `json:"id"`
	Method          ExtensionUIMethod `json:"method"`
	Title           string            `json:"title,omitzero"`
	Message         string            `json:"message,omitzero"`
	Options         []string          `json:"options,omitzero"`
	Placeholder     string            `json:"placeholder,omitzero"`
	Prefill         string            `json:"prefill,omitzero"`
	Timeout         int               `json:"timeout,omitzero"`
	NotifyType      NotifyType        `json:"notifyType,omitzero"`
	StatusKey       string            `json:"statusKey,omitzero"`
	StatusText      *string           `json:"statusText,omitzero"`
	WidgetKey       string            `json:"widgetKey,omitzero"`
	WidgetLines     []string          `json:"widgetLines,omitzero"`
	WidgetPlacement WidgetPlacement   `json:"widgetPlacement,omitzero"`
	Text            string            `json:"text,omitzero"` // for set_editor_text
}

// ============================================================
// Message types (shared between events and responses).
// ============================================================

// AgentMessage is the union of user/assistant/toolResult messages.
// We only care about assistant messages for building genai.Result.
type AgentMessage struct {
	Role         Role          `json:"role"`
	Content      ContentBlocks `json:"content,omitzero"`
	API          string        `json:"api,omitzero"`
	Provider     string        `json:"provider,omitzero"`
	Model        string        `json:"model,omitzero"`
	ResponseID   string        `json:"responseId,omitzero"`
	Usage        MessageUsage  `json:"usage,omitzero"`
	StopReason   StopReason    `json:"stopReason,omitzero"`
	ErrorMessage string        `json:"errorMessage,omitzero"`
	Timestamp    float64       `json:"timestamp,omitzero"`
	// ToolResult-specific fields.
	ToolCallID string `json:"toolCallId,omitzero"`
	ToolName   string `json:"toolName,omitzero"`
	IsError    bool   `json:"isError,omitzero"`
}

// MessageUsage holds token usage from an AssistantMessage.
type MessageUsage struct {
	Input       int64     `json:"input"`
	Output      int64     `json:"output"`
	CacheRead   int64     `json:"cacheRead"`
	CacheWrite  int64     `json:"cacheWrite"`
	TotalTokens int64     `json:"totalTokens"`
	Cost        UsageCost `json:"cost,omitzero"`
}

// UsageCost holds cost information.
type UsageCost struct {
	Input      float64 `json:"input"`
	Output     float64 `json:"output"`
	CacheRead  float64 `json:"cacheRead"`
	CacheWrite float64 `json:"cacheWrite"`
	Total      float64 `json:"total"`
}

// ---------- Content blocks ----------

// ContentBlock is one entry in AssistantMessage.content.
// Discriminated by Type: "text", "thinking", "toolCall", "image".
type ContentBlock struct {
	Type ContentBlockType `json:"type"`
	// text block
	Text          string `json:"text,omitzero"`
	TextSignature string `json:"textSignature,omitzero"`
	// thinking block
	Thinking          string `json:"thinking,omitzero"`
	ThinkingSignature string `json:"thinkingSignature,omitzero"`
	Redacted          bool   `json:"redacted,omitzero"`
	// toolCall block
	ID               string                     `json:"id,omitzero"`
	Name             string                     `json:"name,omitzero"`
	Arguments        map[string]json.RawMessage `json:"arguments,omitzero"`
	ThoughtSignature string                     `json:"thoughtSignature,omitzero"`
	// image block
	Data     string `json:"data,omitzero"`
	MimeType string `json:"mimeType,omitzero"`
}

// ContentBlocks is a []ContentBlock with custom JSON unmarshaling.
//
// User messages may carry content as a plain string instead of an array of
// blocks, so we accept both forms.
type ContentBlocks []ContentBlock

// UnmarshalJSON handles both string and array JSON content.
func (c *ContentBlocks) UnmarshalJSON(data []byte) error {
	data = bytes.TrimSpace(data)
	if len(data) == 0 || bytes.Equal(data, []byte("null")) {
		return nil
	}
	if data[0] == '[' {
		return json.Unmarshal(data, (*[]ContentBlock)(c))
	}
	var s string
	if err := json.Unmarshal(data, &s); err != nil {
		return err
	}
	*c = ContentBlocks{{Type: ContentText, Text: s}}
	return nil
}

// ---------- Assistant message event (delta) ----------

// AssistantMessageEvent is a streaming delta inside MessageUpdateEvent.
type AssistantMessageEvent struct {
	Type         DeltaType     `json:"type"`
	ContentIndex int           `json:"contentIndex,omitzero"`
	Delta        string        `json:"delta,omitzero"`
	Content      string        `json:"content,omitzero"`
	Reason       StopReason    `json:"reason,omitzero"`
	ToolCall     *ContentBlock `json:"toolCall,omitzero"`
	// Partial carries the accumulated message during streaming.
	Partial *AgentMessage `json:"partial,omitzero"`
	// Message carries the final message on done.
	Message *AgentMessage `json:"message,omitzero"`
	// Error carries the final message on error/abort.
	Error *AgentMessage `json:"error,omitzero"`
}
