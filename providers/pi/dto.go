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
// These DTOs are defined against Pi Coding Agent v0.84.2.
//
// Source: https://github.com/earendil-works/pi

package pi

import (
	"bytes"
	"encoding/json"
	"strings"
)

// ============================================================
// Shared types: enums, routing probes.
// ============================================================

// EventType is Pi's wire type discriminator for events emitted on stdout and
// commands sent on stdin.
type EventType string

// Command type constants.
const (
	// Prompting.
	CmdPrompt   EventType = "prompt"
	CmdSteer    EventType = "steer"
	CmdFollowUp EventType = "follow_up"
	CmdAbort    EventType = "abort"

	// Session.
	CmdNewSession           EventType = "new_session"
	CmdGetState             EventType = "get_state"
	CmdGetSessionStats      EventType = "get_session_stats"
	CmdExportHTML           EventType = "export_html"
	CmdSwitchSession        EventType = "switch_session"
	CmdFork                 EventType = "fork"
	CmdClone                EventType = "clone"
	CmdGetForkMessages      EventType = "get_fork_messages"
	CmdGetEntries           EventType = "get_entries"
	CmdGetTree              EventType = "get_tree"
	CmdGetLastAssistantText EventType = "get_last_assistant_text"
	CmdSetSessionName       EventType = "set_session_name"
	CmdGetMessages          EventType = "get_messages"
	CmdGetCommands          EventType = "get_commands"

	// Model.
	CmdSetModel   EventType = "set_model"
	CmdCycleModel EventType = "cycle_model"
	CmdGetModels  EventType = "get_available_models"

	// Thinking.
	CmdSetThinking       EventType = "set_thinking_level"
	CmdCycleThinking     EventType = "cycle_thinking_level"
	CmdGetThinkingLevels EventType = "get_available_thinking_levels"

	// Queue modes.
	CmdSetSteeringMode EventType = "set_steering_mode"
	CmdSetFollowUpMode EventType = "set_follow_up_mode"

	// Compaction.
	CmdCompact           EventType = "compact"
	CmdSetAutoCompaction EventType = "set_auto_compaction"

	// Retry.
	CmdSetAutoRetry EventType = "set_auto_retry"
	CmdAbortRetry   EventType = "abort_retry"

	// Bash.
	CmdBash      EventType = "bash"
	CmdAbortBash EventType = "abort_bash"
)

// Event type constants.
const (
	EventAgentStart                     EventType = "agent_start"
	EventAgentEnd                       EventType = "agent_end"
	EventAgentSettled                   EventType = "agent_settled"
	EventTurnStart                      EventType = "turn_start"
	EventTurnEnd                        EventType = "turn_end"
	EventMessageStart                   EventType = "message_start"
	EventMessageUpdate                  EventType = "message_update"
	EventMessageEnd                     EventType = "message_end"
	EventToolExecStart                  EventType = "tool_execution_start"
	EventToolExecUpdate                 EventType = "tool_execution_update"
	EventToolExecEnd                    EventType = "tool_execution_end"
	EventQueueUpdate                    EventType = "queue_update"
	EventCompactionStart                EventType = "compaction_start"
	EventCompactionEnd                  EventType = "compaction_end"
	EventAutoRetryStart                 EventType = "auto_retry_start"
	EventAutoRetryEnd                   EventType = "auto_retry_end"
	EventSummarizationRetryScheduled    EventType = "summarization_retry_scheduled"
	EventSummarizationRetryAttemptStart EventType = "summarization_retry_attempt_start"
	EventSummarizationRetryFinished     EventType = "summarization_retry_finished"
	EventThinkingLevelChanged           EventType = "thinking_level_changed"
	EventEntryAppended                  EventType = "entry_appended"
	EventResponse                       EventType = "response"
	EventExtensionUI                    EventType = "extension_ui_request"
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
	StopReasonPending  StopReason = "pending"
	StopReasonStop     StopReason = "stop"
	StopReasonLength   StopReason = "length"
	StopReasonToolUse  StopReason = "toolUse"
	StopReasonError    StopReason = "error"
	StopReasonAborted  StopReason = "aborted"
	StopReasonDeferred StopReason = "deferred"
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
	ThinkingMax     ThinkingLevel = "max"
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
	Type    EventType `json:"type"`
	Command EventType `json:"command,omitzero"`
	ID      string    `json:"id,omitzero"`
	Success *bool     `json:"success,omitzero"`
}

// ============================================================
// Input types: commands sent to Pi (stdin).
// ============================================================

// ---------- Prompting ----------

// PromptCmd sends a user message.
type PromptCmd struct {
	ID                string            `json:"id,omitzero"`
	Type              EventType         `json:"type"`
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
	Type    EventType      `json:"type"`
	Message string         `json:"message"`
	Images  []ImageContent `json:"images,omitzero"`
}

// FollowUpCmd sends a follow-up message after the agent finishes.
type FollowUpCmd struct {
	ID      string         `json:"id,omitzero"`
	Type    EventType      `json:"type"`
	Message string         `json:"message"`
	Images  []ImageContent `json:"images,omitzero"`
}

// AbortCmd cancels the current generation.
type AbortCmd struct {
	ID   string    `json:"id,omitzero"`
	Type EventType `json:"type"`
}

// ---------- Session ----------

// NewSessionCmd starts a fresh session.
type NewSessionCmd struct {
	ID            string    `json:"id,omitzero"`
	Type          EventType `json:"type"`
	ParentSession string    `json:"parentSession,omitzero"`
}

// GetStateCmd requests current session state.
type GetStateCmd struct {
	ID   string    `json:"id,omitzero"`
	Type EventType `json:"type"`
}

// GetSessionStatsCmd requests session statistics.
type GetSessionStatsCmd struct {
	ID   string    `json:"id,omitzero"`
	Type EventType `json:"type"`
}

// ExportHTMLCmd exports the session as HTML.
type ExportHTMLCmd struct {
	ID         string    `json:"id,omitzero"`
	Type       EventType `json:"type"`
	OutputPath string    `json:"outputPath,omitzero"`
}

// SwitchSessionCmd switches to a different session.
type SwitchSessionCmd struct {
	ID          string    `json:"id,omitzero"`
	Type        EventType `json:"type"`
	SessionPath string    `json:"sessionPath"`
}

// ForkCmd forks the session at a specific entry.
type ForkCmd struct {
	ID      string    `json:"id,omitzero"`
	Type    EventType `json:"type"`
	EntryID string    `json:"entryId"`
}

// GetForkMessagesCmd gets messages available for forking.
type GetForkMessagesCmd struct {
	ID   string    `json:"id,omitzero"`
	Type EventType `json:"type"`
}

// GetLastAssistantTextCmd gets the last assistant message text.
type GetLastAssistantTextCmd struct {
	ID   string    `json:"id,omitzero"`
	Type EventType `json:"type"`
}

// SetSessionNameCmd sets the session name.
type SetSessionNameCmd struct {
	ID   string    `json:"id,omitzero"`
	Type EventType `json:"type"`
	Name string    `json:"name"`
}

// GetMessagesCmd gets all messages in the session.
type GetMessagesCmd struct {
	ID   string    `json:"id,omitzero"`
	Type EventType `json:"type"`
}

// GetCommandsCmd gets available slash commands.
type GetCommandsCmd struct {
	ID   string    `json:"id,omitzero"`
	Type EventType `json:"type"`
}

// ---------- Model ----------

// SetModelCmd switches the active model.
type SetModelCmd struct {
	ID       string    `json:"id,omitzero"`
	Type     EventType `json:"type"`
	Provider string    `json:"provider"`
	ModelID  string    `json:"modelId"`
}

// CycleModelCmd cycles to the next model.
type CycleModelCmd struct {
	ID   string    `json:"id,omitzero"`
	Type EventType `json:"type"`
}

// GetModelsCmd requests the list of available models.
type GetModelsCmd struct {
	ID   string    `json:"id,omitzero"`
	Type EventType `json:"type"`
}

// ---------- Thinking ----------

// SetThinkingCmd sets the thinking level.
type SetThinkingCmd struct {
	ID    string        `json:"id,omitzero"`
	Type  EventType     `json:"type"`
	Level ThinkingLevel `json:"level"`
}

// CycleThinkingCmd cycles to the next thinking level.
type CycleThinkingCmd struct {
	ID   string    `json:"id,omitzero"`
	Type EventType `json:"type"`
}

// GetThinkingLevelsCmd requests the thinking levels available for the active model.
type GetThinkingLevelsCmd struct {
	ID   string    `json:"id,omitzero"`
	Type EventType `json:"type"`
}

// ---------- Queue modes ----------

// SetSteeringModeCmd sets the steering queue mode.
type SetSteeringModeCmd struct {
	ID   string    `json:"id,omitzero"`
	Type EventType `json:"type"`
	Mode QueueMode `json:"mode"`
}

// SetFollowUpModeCmd sets the follow-up queue mode.
type SetFollowUpModeCmd struct {
	ID   string    `json:"id,omitzero"`
	Type EventType `json:"type"`
	Mode QueueMode `json:"mode"`
}

// ---------- Compaction ----------

// CompactCmd triggers compaction with optional custom instructions.
type CompactCmd struct {
	ID                 string    `json:"id,omitzero"`
	Type               EventType `json:"type"`
	CustomInstructions string    `json:"customInstructions,omitzero"`
}

// SetAutoCompactionCmd enables or disables automatic compaction.
type SetAutoCompactionCmd struct {
	ID      string    `json:"id,omitzero"`
	Type    EventType `json:"type"`
	Enabled bool      `json:"enabled"`
}

// ---------- Retry ----------

// SetAutoRetryCmd enables or disables automatic retry.
type SetAutoRetryCmd struct {
	ID      string    `json:"id,omitzero"`
	Type    EventType `json:"type"`
	Enabled bool      `json:"enabled"`
}

// AbortRetryCmd aborts the current retry.
type AbortRetryCmd struct {
	ID   string    `json:"id,omitzero"`
	Type EventType `json:"type"`
}

// ---------- Bash ----------

// BashCmd executes a bash command.
type BashCmd struct {
	ID                 string    `json:"id,omitzero"`
	Type               EventType `json:"type"`
	Command            string    `json:"command"`
	ExcludeFromContext bool      `json:"excludeFromContext,omitzero"`
}

// CloneCmd clones the current session.
type CloneCmd struct {
	ID   string    `json:"id,omitzero"`
	Type EventType `json:"type"`
}

// GetEntriesCmd gets session entries, optionally after the given entry ID.
type GetEntriesCmd struct {
	ID    string    `json:"id,omitzero"`
	Type  EventType `json:"type"`
	Since string    `json:"since,omitzero"`
}

// GetTreeCmd gets the session-entry tree.
type GetTreeCmd struct {
	ID   string    `json:"id,omitzero"`
	Type EventType `json:"type"`
}

// AbortBashCmd aborts the current bash command.
type AbortBashCmd struct {
	ID   string    `json:"id,omitzero"`
	Type EventType `json:"type"`
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
	Command EventType       `json:"command"`
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

// ThinkingLevelsData is the data payload for get_available_thinking_levels response.
type ThinkingLevelsData struct {
	Levels []ThinkingLevel `json:"levels"`
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

// CloneData is the data payload for clone response.
type CloneData struct {
	Cancelled bool `json:"cancelled"`
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

// EntriesData is the data payload for get_entries response. Session entry
// details are protocol-defined extension data and are retained as raw JSON.
type EntriesData struct {
	Entries []SessionEntry `json:"entries"`
	LeafID  *string        `json:"leafId"`
}

// TreeData is the data payload for get_tree response.
type TreeData struct {
	Tree   []SessionTreeNode `json:"tree"`
	LeafID *string           `json:"leafId"`
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
	ID        string `json:"id"`
	Name      string `json:"name"`
	API       string `json:"api"`
	Provider  string `json:"provider"`
	BaseURL   string `json:"baseUrl"`
	Reasoning bool   `json:"reasoning"`
	// ThinkingLevelMap maps Pi thinking levels to provider-specific values. JSON null values decode as empty strings.
	ThinkingLevelMap map[ThinkingLevel]string   `json:"thinkingLevelMap,omitzero"`
	Input            []string                   `json:"input"`
	ContextWindow    int64                      `json:"contextWindow"`
	MaxTokens        int64                      `json:"maxTokens"`
	Cost             ModelCost                  `json:"cost"`
	SamplingParams   map[string]json.RawMessage `json:"samplingParams,omitzero"`
	Headers          map[string]string          `json:"headers,omitzero"`
	Compat           json.RawMessage            `json:"compat,omitzero"`
}

// GetID returns the provider-qualified model ID (e.g. "cerebras/gpt-oss-120b").
func (m *Model) GetID() string { return m.Provider + "/" + m.ID }

// String returns the model's display name.
func (m *Model) String() string { return m.Name }

// Context returns the context window size in tokens.
func (m *Model) Context() int64 { return m.ContextWindow }

// ModelCost holds per-million-token costs.
type ModelCost struct {
	Input      float64         `json:"input"`
	Output     float64         `json:"output"`
	CacheRead  float64         `json:"cacheRead"`
	CacheWrite float64         `json:"cacheWrite"`
	Tiers      []ModelCostTier `json:"tiers,omitzero"`
}

// ModelCostTier is a pricing tier that applies when input use exceeds its threshold.
type ModelCostTier struct {
	InputTokensAbove int64   `json:"inputTokensAbove"`
	Input            float64 `json:"input"`
	Output           float64 `json:"output"`
	CacheRead        float64 `json:"cacheRead"`
	CacheWrite       float64 `json:"cacheWrite"`
}

// ---------- Agent events ----------

// AgentStartEvent is emitted when the agent begins processing.
type AgentStartEvent struct {
	Type EventType `json:"type"`
}

// AgentEndEvent is emitted when one low-level agent run completes. WillRetry
// indicates whether Pi will automatically retry before the agent settles.
type AgentEndEvent struct {
	Type      EventType      `json:"type"`
	Messages  []AgentMessage `json:"messages"`
	WillRetry bool           `json:"willRetry"`
}

// AgentSettledEvent is emitted when Pi has no automatic retry, compaction
// retry, or queued continuation remaining.
type AgentSettledEvent struct {
	Type EventType `json:"type"`
}

// AutoRetryStartEvent reports the start of an automatic retry after a transient error.
type AutoRetryStartEvent struct {
	Type         EventType `json:"type"`
	Attempt      int       `json:"attempt"`
	MaxAttempts  int       `json:"maxAttempts"`
	DelayMS      int64     `json:"delayMs"`
	ErrorMessage string    `json:"errorMessage"`
}

// AutoRetryEndEvent reports automatic retry completion or final failure.
type AutoRetryEndEvent struct {
	Type       EventType `json:"type"`
	Success    bool      `json:"success"`
	Attempt    int       `json:"attempt"`
	FinalError string    `json:"finalError,omitzero"`
}

// QueueUpdateEvent reports the current pending steering and follow-up queues.
type QueueUpdateEvent struct {
	Type     EventType `json:"type"`
	Steering []string  `json:"steering"`
	FollowUp []string  `json:"followUp"`
}

// CompactionReason identifies why Pi compacted its context.
type CompactionReason string

// Compaction reason constants.
const (
	CompactionManual    CompactionReason = "manual"
	CompactionOverflow  CompactionReason = "overflow"
	CompactionThreshold CompactionReason = "threshold"
)

// CompactionStartEvent reports the beginning of manual or automatic compaction.
type CompactionStartEvent struct {
	Type   EventType        `json:"type"`
	Reason CompactionReason `json:"reason"`
}

// CompactionResult contains the summary and usage generated by compaction.
type CompactionResult struct {
	Summary              string          `json:"summary"`
	FirstKeptEntryID     string          `json:"firstKeptEntryId"`
	TokensBefore         int64           `json:"tokensBefore"`
	EstimatedTokensAfter int64           `json:"estimatedTokensAfter,omitzero"`
	Usage                *MessageUsage   `json:"usage,omitzero"`
	Details              json.RawMessage `json:"details,omitzero"`
}

// CompactionEndEvent reports compaction completion, cancellation, or failure.
type CompactionEndEvent struct {
	Type         EventType         `json:"type"`
	Reason       CompactionReason  `json:"reason"`
	Result       *CompactionResult `json:"result,omitzero"`
	Aborted      bool              `json:"aborted"`
	WillRetry    bool              `json:"willRetry"`
	ErrorMessage string            `json:"errorMessage,omitzero"`
}

// SummarizationRetrySource identifies the operation whose summary Pi is retrying.
type SummarizationRetrySource string

// Summarization retry source constants.
const (
	SummarizationRetryBranchSummary SummarizationRetrySource = "branchSummary"
	SummarizationRetryCompaction    SummarizationRetrySource = "compaction"
)

// SummarizationRetryScheduledEvent reports a delayed summarization retry.
type SummarizationRetryScheduledEvent struct {
	Type         EventType `json:"type"`
	Attempt      int       `json:"attempt"`
	MaxAttempts  int       `json:"maxAttempts"`
	DelayMS      int64     `json:"delayMs"`
	ErrorMessage string    `json:"errorMessage"`
}

// SummarizationRetryAttemptStartEvent reports the start of a retried summary request.
type SummarizationRetryAttemptStartEvent struct {
	Type   EventType                `json:"type"`
	Source SummarizationRetrySource `json:"source"`
	Reason CompactionReason         `json:"reason,omitzero"`
}

// SummarizationRetryFinishedEvent reports completion of the summarization retry loop.
type SummarizationRetryFinishedEvent struct {
	Type EventType `json:"type"`
}

// ThinkingLevelChangedEvent reports a thinking-level change made during a session.
type ThinkingLevelChangedEvent struct {
	Type  EventType     `json:"type"`
	Level ThinkingLevel `json:"level"`
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
	Usage                 MessageUsage          `json:"usage"`
	AssistantMessageEvent AssistantMessageEvent `json:"assistantMessageEvent"`
}

// MessageUpdateDeltaEvent is the minimal message_update event shape needed by
// streaming consumers that only need the assistantMessageEvent payload.
type MessageUpdateDeltaEvent struct {
	Type                  EventType          `json:"type"`
	Usage                 MessageUsage       `json:"usage"`
	AssistantMessageEvent MessageUpdateDelta `json:"assistantMessageEvent"`
}

// MessageUpdateDelta is the compact assistantMessageEvent payload emitted in a
// message_update event.
type MessageUpdateDelta struct {
	Type         DeltaType              `json:"type"`
	ContentIndex int                    `json:"contentIndex,omitzero"`
	ID           string                 `json:"id,omitzero"`
	ToolName     string                 `json:"toolName,omitzero"`
	Delta        string                 `json:"delta,omitzero"`
	Content      string                 `json:"content,omitzero"`
	Reason       StopReason             `json:"reason,omitzero"`
	ToolCall     *MessageUpdateToolCall `json:"toolCall,omitzero"`
	Error        *MessageUpdateError    `json:"error,omitzero"`
}

// MessageUpdateToolCall is the tool call payload in a message_update delta.
type MessageUpdateToolCall struct {
	ID        string                     `json:"id"`
	Name      string                     `json:"name"`
	Arguments map[string]json.RawMessage `json:"arguments"`
}

// MessageUpdateError is the error payload in a message_update delta.
type MessageUpdateError struct {
	ErrorMessage string `json:"errorMessage"`
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

// EditToolArgs is the args shape for Pi's edit tool.
type EditToolArgs struct {
	Path    string        `json:"path"`
	OldText string        `json:"oldText"`
	NewText string        `json:"newText"`
	Edits   []ReplaceEdit `json:"edits"`
}

// ReplaceEdit is one old/new text replacement in an edit tool call.
type ReplaceEdit struct {
	OldText string `json:"oldText"`
	NewText string `json:"newText"`
}

// SubagentToolArgs is the args shape for Pi's subagent tool.
type SubagentToolArgs struct {
	SubagentToolStep

	Action string                  `json:"action"`
	Tasks  []SubagentToolStep      `json:"tasks"`
	Chain  []SubagentToolChainStep `json:"chain"`
}

// SubagentToolStep is one subagent invocation in a subagent tool call.
type SubagentToolStep struct {
	Agent string `json:"agent"`
	Label string `json:"label"`
	Phase string `json:"phase"`
	Task  string `json:"task"`
}

// SubagentToolChainStep is one chain step in a subagent tool call.
type SubagentToolChainStep struct {
	SubagentToolStep

	Parallel []SubagentToolStep `json:"parallel"`
}

// ToolExecResult is the result payload for tool_execution_update and
// tool_execution_end events.
//
// It contains an array of content blocks with the tool output.
type ToolExecResult struct {
	Content ContentBlocks   `json:"content"`
	IsError bool            `json:"isError,omitzero"`
	Details json.RawMessage `json:"details,omitzero"`
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
	Role           Role              `json:"role"`
	Content        ContentBlocks     `json:"content,omitzero"`
	API            string            `json:"api,omitzero"`
	Provider       string            `json:"provider,omitzero"`
	Model          string            `json:"model,omitzero"`
	ResponseModel  string            `json:"responseModel,omitzero"`
	ResponseID     string            `json:"responseId,omitzero"`
	Diagnostics    []json.RawMessage `json:"diagnostics,omitzero"`
	Usage          MessageUsage      `json:"usage,omitzero"`
	StopReason     StopReason        `json:"stopReason,omitzero"`
	Deferred       *DeferredHandle   `json:"deferred,omitzero"`
	ErrorMessage   string            `json:"errorMessage,omitzero"`
	RawStopReason  string            `json:"rawStopReason,omitzero"`
	Timestamp      float64           `json:"timestamp,omitzero"`
	ToolCallID     string            `json:"toolCallId,omitzero"`
	ToolName       string            `json:"toolName,omitzero"`
	Details        json.RawMessage   `json:"details,omitzero"`
	AddedToolNames []string          `json:"addedToolNames,omitzero"`
	IsError        bool              `json:"isError,omitzero"`
	CustomType     string            `json:"customType,omitzero"`
	Display        bool              `json:"display,omitzero"`
}

// DeferredHandle identifies a provider-managed deferred response.
type DeferredHandle struct {
	Provider    string          `json:"provider"`
	ModelID     string          `json:"modelId"`
	API         string          `json:"api"`
	ID          string          `json:"id"`
	ExpiresAt   int64           `json:"expiresAt,omitzero"`
	PollAfterMS int64           `json:"pollAfterMs,omitzero"`
	Data        json.RawMessage `json:"data,omitzero"`
}

// MessageUsage holds token usage from an AssistantMessage.
type MessageUsage struct {
	Input        int64     `json:"input"`
	Output       int64     `json:"output"`
	CacheRead    int64     `json:"cacheRead"`
	CacheWrite   int64     `json:"cacheWrite"`
	CacheWrite1h int64     `json:"cacheWrite1h,omitzero"`
	Reasoning    int64     `json:"reasoning,omitzero"`
	TotalTokens  int64     `json:"totalTokens"`
	Cost         UsageCost `json:"cost,omitzero"`
}

// EntryAppendedEvent reports an entry added to Pi's session history.
type EntryAppendedEvent struct {
	Type  EventType    `json:"type"`
	Entry SessionEntry `json:"entry"`
}

// SessionEntry is a session-history entry. Details are protocol-defined
// extension data and are retained as raw JSON.
type SessionEntry struct {
	Type                 string          `json:"type"`
	ID                   string          `json:"id"`
	ParentID             *string         `json:"parentId"`
	Timestamp            string          `json:"timestamp"`
	Message              *AgentMessage   `json:"message,omitzero"`
	Content              ContentBlocks   `json:"content,omitzero"`
	ThinkingLevel        ThinkingLevel   `json:"thinkingLevel,omitzero"`
	Provider             string          `json:"provider,omitzero"`
	ModelID              string          `json:"modelId,omitzero"`
	Summary              string          `json:"summary,omitzero"`
	FirstKeptEntryID     string          `json:"firstKeptEntryId,omitzero"`
	TokensBefore         int64           `json:"tokensBefore,omitzero"`
	EstimatedTokensAfter int64           `json:"estimatedTokensAfter,omitzero"`
	Details              json.RawMessage `json:"details,omitzero"`
	FromHook             bool            `json:"fromHook,omitzero"`
	FromID               string          `json:"fromId,omitzero"`
	CustomType           string          `json:"customType,omitzero"`
	Data                 json.RawMessage `json:"data,omitzero"`
	TargetID             string          `json:"targetId,omitzero"`
	Label                *string         `json:"label"`
	Name                 string          `json:"name,omitzero"`
	Display              bool            `json:"display,omitzero"`
	Usage                *MessageUsage   `json:"usage,omitzero"`
}

// SessionTreeNode is one node in the session-entry tree.
type SessionTreeNode struct {
	Entry          SessionEntry      `json:"entry"`
	Children       []SessionTreeNode `json:"children"`
	Label          string            `json:"label,omitzero"`
	LabelTimestamp string            `json:"labelTimestamp,omitzero"`
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
	PartialArgs      string                     `json:"partialArgs,omitzero"`
	StreamIndex      int                        `json:"streamIndex,omitzero"`
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
	ID           string        `json:"id,omitzero"`
	ToolName     string        `json:"toolName,omitzero"`
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
