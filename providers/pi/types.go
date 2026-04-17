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

// ---------- Routing probe ----------

// LineProbe extracts routing fields from a JSONL line to determine its kind.
type LineProbe struct {
	Type    EventType `json:"type"`
	Command string    `json:"command,omitzero"`
	ID      string    `json:"id,omitzero"`
	Success *bool     `json:"success,omitzero"`
}

// ============================================================
// Input types: commands sent to Pi (stdin).
// ============================================================

// ---------- Prompting ----------

// PromptCmd sends a user message.
type PromptCmd struct {
	ID                string         `json:"id,omitzero"`
	Type              CommandType    `json:"type"`
	Message           string         `json:"message"`
	Images            []ImageContent `json:"images,omitzero"`
	StreamingBehavior string         `json:"streamingBehavior,omitzero"` // "steer" or "followUp"
}

// ImageContent is an inline image in base64.
type ImageContent struct {
	Type     string `json:"type"`
	Data     string `json:"data"`
	MimeType string `json:"mimeType"`
}

// cmdSteer sends a steering message mid-run.
type cmdSteer struct {
	ID      string         `json:"id,omitzero"`
	Type    CommandType    `json:"type"`
	Message string         `json:"message"`
	Images  []ImageContent `json:"images,omitzero"`
}

// cmdFollowUp sends a follow-up message after the agent finishes.
type cmdFollowUp struct {
	ID      string         `json:"id,omitzero"`
	Type    CommandType    `json:"type"`
	Message string         `json:"message"`
	Images  []ImageContent `json:"images,omitzero"`
}

// cmdAbort cancels the current generation.
type cmdAbort struct {
	ID   string      `json:"id,omitzero"`
	Type CommandType `json:"type"`
}

// ---------- Session ----------

// cmdNewSession starts a fresh session.
type cmdNewSession struct {
	ID            string      `json:"id,omitzero"`
	Type          CommandType `json:"type"`
	ParentSession string      `json:"parentSession,omitzero"`
}

// cmdGetState requests current session state.
type cmdGetState struct {
	ID   string      `json:"id,omitzero"`
	Type CommandType `json:"type"`
}

// cmdGetSessionStats requests session statistics.
type cmdGetSessionStats struct {
	ID   string      `json:"id,omitzero"`
	Type CommandType `json:"type"`
}

// cmdExportHTML exports the session as HTML.
type cmdExportHTML struct {
	ID         string      `json:"id,omitzero"`
	Type       CommandType `json:"type"`
	OutputPath string      `json:"outputPath,omitzero"`
}

// cmdSwitchSession switches to a different session.
type cmdSwitchSession struct {
	ID          string      `json:"id,omitzero"`
	Type        CommandType `json:"type"`
	SessionPath string      `json:"sessionPath"`
}

// cmdFork forks the session at a specific entry.
type cmdFork struct {
	ID      string      `json:"id,omitzero"`
	Type    CommandType `json:"type"`
	EntryID string      `json:"entryId"`
}

// cmdGetForkMessages gets messages available for forking.
type cmdGetForkMessages struct {
	ID   string      `json:"id,omitzero"`
	Type CommandType `json:"type"`
}

// cmdGetLastAssistantText gets the last assistant message text.
type cmdGetLastAssistantText struct {
	ID   string      `json:"id,omitzero"`
	Type CommandType `json:"type"`
}

// cmdSetSessionName sets the session name.
type cmdSetSessionName struct {
	ID   string      `json:"id,omitzero"`
	Type CommandType `json:"type"`
	Name string      `json:"name"`
}

// cmdGetMessages gets all messages in the session.
type cmdGetMessages struct {
	ID   string      `json:"id,omitzero"`
	Type CommandType `json:"type"`
}

// cmdGetCommands gets available slash commands.
type cmdGetCommands struct {
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

// cmdCycleModel cycles to the next model.
type cmdCycleModel struct {
	ID   string      `json:"id,omitzero"`
	Type CommandType `json:"type"`
}

// cmdGetModels requests the list of available models.
type cmdGetModels struct {
	ID   string      `json:"id,omitzero"`
	Type CommandType `json:"type"`
}

// ---------- Thinking ----------

// cmdSetThinking sets the thinking level.
type cmdSetThinking struct {
	ID    string        `json:"id,omitzero"`
	Type  CommandType   `json:"type"`
	Level ThinkingLevel `json:"level"`
}

// cmdCycleThinking cycles to the next thinking level.
type cmdCycleThinking struct {
	ID   string      `json:"id,omitzero"`
	Type CommandType `json:"type"`
}

// ---------- Queue modes ----------

// cmdSetSteeringMode sets the steering queue mode.
type cmdSetSteeringMode struct {
	ID   string      `json:"id,omitzero"`
	Type CommandType `json:"type"`
	Mode QueueMode   `json:"mode"`
}

// cmdSetFollowUpMode sets the follow-up queue mode.
type cmdSetFollowUpMode struct {
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

// cmdSetAutoCompaction enables or disables automatic compaction.
type cmdSetAutoCompaction struct {
	ID      string      `json:"id,omitzero"`
	Type    CommandType `json:"type"`
	Enabled bool        `json:"enabled"`
}

// ---------- Retry ----------

// cmdSetAutoRetry enables or disables automatic retry.
type cmdSetAutoRetry struct {
	ID      string      `json:"id,omitzero"`
	Type    CommandType `json:"type"`
	Enabled bool        `json:"enabled"`
}

// cmdAbortRetry aborts the current retry.
type cmdAbortRetry struct {
	ID   string      `json:"id,omitzero"`
	Type CommandType `json:"type"`
}

// ---------- Bash ----------

// cmdBash executes a bash command.
type cmdBash struct {
	ID      string      `json:"id,omitzero"`
	Type    CommandType `json:"type"`
	Command string      `json:"command"`
}

// cmdAbortBash aborts the current bash command.
type cmdAbortBash struct {
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
	Command string          `json:"command"`
	Success bool            `json:"success"`
	Error   string          `json:"error,omitzero"`
	Data    json.RawMessage `json:"data,omitzero"`
}

// ---------- Response data payloads ----------

// modelsData is the data payload for get_available_models response.
type modelsData struct {
	Models []Model `json:"models"`
}

// stateData is the data payload for get_state response.
type stateData struct {
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

// newSessionData is the data payload for new_session response.
type newSessionData struct {
	Cancelled bool `json:"cancelled"`
}

// cycleModelData is the data payload for cycle_model response.
type cycleModelData struct {
	Model         *Model        `json:"model,omitzero"`
	ThinkingLevel ThinkingLevel `json:"thinkingLevel,omitzero"`
	IsScoped      bool          `json:"isScoped,omitzero"`
}

// cycleThinkingData is the data payload for cycle_thinking_level response.
type cycleThinkingData struct {
	Level ThinkingLevel `json:"level"`
}

// bashData is the data payload for bash response.
type bashData struct {
	// BashResult fields are opaque; we only need to know it succeeded.
	json.RawMessage
}

// exportHTMLData is the data payload for export_html response.
type exportHTMLData struct {
	Path string `json:"path"`
}

// switchSessionData is the data payload for switch_session response.
type switchSessionData struct {
	Cancelled bool `json:"cancelled"`
}

// forkData is the data payload for fork response.
type forkData struct {
	Text      string `json:"text"`
	Cancelled bool   `json:"cancelled"`
}

// forkMessagesData is the data payload for get_fork_messages response.
type forkMessagesData struct {
	Messages []forkMessage `json:"messages"`
}

// forkMessage is a single entry in the fork messages list.
type forkMessage struct {
	EntryID string `json:"entryId"`
	Text    string `json:"text"`
}

// lastAssistantTextData is the data payload for get_last_assistant_text response.
type lastAssistantTextData struct {
	Text *string `json:"text"` // nullable
}

// getMessagesData is the data payload for get_messages response.
type getMessagesData struct {
	Messages []AgentMessage `json:"messages"`
}

// getCommandsData is the data payload for get_commands response.
type getCommandsData struct {
	Commands []rpcSlashCommand `json:"commands"`
}

// rpcSlashCommand is a command available for invocation via prompt.
type rpcSlashCommand struct {
	Name        string     `json:"name"`
	Description string     `json:"description,omitzero"`
	Source      string     `json:"source"` // "extension", "prompt", or "skill"
	SourceInfo  sourceInfo `json:"sourceInfo"`
}

// sourceInfo describes the origin of a slash command.
type sourceInfo struct {
	json.RawMessage
}

// sessionStatsData is the data payload for get_session_stats response.
// SessionStats fields are opaque.
type sessionStatsData struct {
	json.RawMessage
}

// compactData is the data payload for compact response.
// CompactionResult fields are opaque.
type compactData struct {
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

// eventAgentStart is emitted when the agent begins processing.
type eventAgentStart struct {
	Type EventType `json:"type"`
}

// AgentEndEvent is emitted when the agent finishes. Contains accumulated messages.
type AgentEndEvent struct {
	Type     EventType      `json:"type"`
	Messages []AgentMessage `json:"messages"`
}

// eventTurnStart is emitted when a turn begins.
type eventTurnStart struct {
	Type EventType `json:"type"`
}

// TurnEndEvent is emitted when a turn finishes.
type TurnEndEvent struct {
	Type        EventType         `json:"type"`
	Message     AgentMessage      `json:"message"`
	ToolResults []json.RawMessage `json:"toolResults,omitzero"`
}

// eventMessageStart is emitted when a message begins.
type eventMessageStart struct {
	Type    EventType    `json:"type"`
	Message AgentMessage `json:"message"`
}

// MessageUpdateEvent is emitted during streaming with a delta.
type MessageUpdateEvent struct {
	Type                  EventType             `json:"type"`
	Message               AgentMessage          `json:"message"`
	AssistantMessageEvent AssistantMessageEvent `json:"assistantMessageEvent"`
}

// eventMessageEnd is emitted when a message is complete.
type eventMessageEnd struct {
	Type    EventType    `json:"type"`
	Message AgentMessage `json:"message"`
}

// ToolExecStartEvent is emitted when a tool begins execution.
type ToolExecStartEvent struct {
	Type       EventType `json:"type"`
	ToolCallID string    `json:"toolCallId"`
	ToolName   string    `json:"toolName"`
	Args       any       `json:"args"`
}

// ToolExecUpdateEvent is emitted during tool execution with progress.
type ToolExecUpdateEvent struct {
	Type          EventType       `json:"type"`
	ToolCallID    string          `json:"toolCallId"`
	ToolName      string          `json:"toolName"`
	Args          any             `json:"args"`
	PartialResult json.RawMessage `json:"partialResult"`
}

// ToolExecEndEvent is emitted when a tool finishes execution.
type ToolExecEndEvent struct {
	Type       EventType       `json:"type"`
	ToolCallID string          `json:"toolCallId"`
	ToolName   string          `json:"toolName"`
	Result     json.RawMessage `json:"result"`
	IsError    bool            `json:"isError"`
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
	NotifyType      string            `json:"notifyType,omitzero"` // "info", "warning", "error"
	StatusKey       string            `json:"statusKey,omitzero"`
	StatusText      *string           `json:"statusText,omitzero"`
	WidgetKey       string            `json:"widgetKey,omitzero"`
	WidgetLines     []string          `json:"widgetLines,omitzero"`
	WidgetPlacement string            `json:"widgetPlacement,omitzero"` // "aboveEditor", "belowEditor"
	Text            string            `json:"text,omitzero"`            // for set_editor_text
}

// ============================================================
// Message types (shared between events and responses).
// ============================================================

// AgentMessage is the union of user/assistant/toolResult messages.
// We only care about assistant messages for building genai.Result.
type AgentMessage struct {
	Role         string        `json:"role"`
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
	Type string `json:"type"`
	// text block
	Text          string `json:"text,omitzero"`
	TextSignature string `json:"textSignature,omitzero"`
	// thinking block
	Thinking          string `json:"thinking,omitzero"`
	ThinkingSignature string `json:"thinkingSignature,omitzero"`
	Redacted          bool   `json:"redacted,omitzero"`
	// toolCall block
	ID               string         `json:"id,omitzero"`
	Name             string         `json:"name,omitzero"`
	Arguments        map[string]any `json:"arguments,omitzero"`
	ThoughtSignature string         `json:"thoughtSignature,omitzero"`
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
	*c = ContentBlocks{{Type: "text", Text: s}}
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
