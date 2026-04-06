// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Wire types for the Claude Code NDJSON streaming protocol.

package claudecode

import "encoding/json"

// ============================================================================
// Input types (sent TO the agent via stdin)
// ============================================================================
//
// Claude Code accepts five NDJSON message types on stdin when running with
// --input-format stream-json. The StdinMessage union covers all of them.
// See controlSchemas.ts StdinMessageSchema in the Claude Code source.

// InputType is the top-level "type" discriminator for Claude Code stdin NDJSON.
type InputType string

const (
	// InputUser sends a user turn.
	InputUser InputType = "user"
	// InputControlRequest sends a control request to Claude Code.
	InputControlRequest InputType = "control_request"
	// InputControlResponse responds to a control request from Claude Code.
	InputControlResponse InputType = "control_response"
	// InputKeepAlive is a heartbeat.
	InputKeepAlive InputType = "keep_alive"
	// InputUpdateEnvVars pushes env vars at runtime.
	InputUpdateEnvVars InputType = "update_environment_variables"
)

// ---------- user message ----------

// InputUserMsg sends a user turn to Claude Code (type:"user").
type InputUserMsg struct {
	Type            InputType        `json:"type"` // InputUser
	Message         InputUserContent `json:"message"`
	UUID            string           `json:"uuid,omitempty"`
	SessionID       string           `json:"session_id,omitempty"`
	ParentToolUseID string           `json:"parent_tool_use_id,omitempty"`
	IsSynthetic     bool             `json:"isSynthetic,omitempty"`
	ToolUseResult   json.RawMessage  `json:"tool_use_result,omitempty"`
	Priority        string           `json:"priority,omitempty"` // "now", "next", "later"
	Timestamp       string           `json:"timestamp,omitempty"`
}

// InputUserContent is the message body within an InputUserMsg.
type InputUserContent struct {
	Role    string              `json:"role"` // always "user"
	Content []InputContentBlock `json:"content"`
}

// InputContentBlock is a single block in the content array sent to Claude Code.
type InputContentBlock struct {
	Type   string           `json:"type"`
	Source InputImageSource `json:"source,omitzero"`
	Text   string           `json:"text,omitempty"`
}

// InputImageSource is an image source block sent to Claude Code.
type InputImageSource struct {
	Type      string `json:"type"`                 // "base64" or "url"
	MediaType string `json:"media_type,omitempty"` // e.g. "image/png"
	Data      string `json:"data,omitempty"`       // base64-encoded bytes
	URL       string `json:"url,omitempty"`        // Type == "url"
}

// ---------- control request ----------

// InputControlRequestMsg sends a control request to Claude Code (type:"control_request").
// The Request field is a JSON object whose "subtype" discriminator determines
// its schema. Use one of the ControlReq* structs below as the Request value.
type InputControlRequestMsg struct {
	Type      InputType `json:"type"` // InputControlRequest
	RequestID string    `json:"request_id"`
	Request   any       `json:"request"`
}

// ControlSubtype is the "subtype" discriminator for control requests.
type ControlSubtype string

// ControlSubtype values for control request subtypes.
const (
	ControlInitialize         ControlSubtype = "initialize"
	ControlInterrupt          ControlSubtype = "interrupt"
	ControlCanUseTool         ControlSubtype = "can_use_tool"
	ControlSetPermissionMode  ControlSubtype = "set_permission_mode"
	ControlSetModel           ControlSubtype = "set_model"
	ControlSetMaxThinking     ControlSubtype = "set_max_thinking_tokens"
	ControlMcpStatus          ControlSubtype = "mcp_status"
	ControlGetContextUsage    ControlSubtype = "get_context_usage"
	ControlHookCallback       ControlSubtype = "hook_callback"
	ControlMcpMessage         ControlSubtype = "mcp_message"
	ControlRewindFiles        ControlSubtype = "rewind_files"
	ControlCancelAsyncMessage ControlSubtype = "cancel_async_message"
	ControlSeedReadState      ControlSubtype = "seed_read_state"
	ControlMcpSetServers      ControlSubtype = "mcp_set_servers"
	ControlReloadPlugins      ControlSubtype = "reload_plugins"
	ControlMcpReconnect       ControlSubtype = "mcp_reconnect"
	ControlMcpToggle          ControlSubtype = "mcp_toggle"
	ControlStopTask           ControlSubtype = "stop_task"
	ControlApplyFlagSettings  ControlSubtype = "apply_flag_settings"
	ControlGetSettings        ControlSubtype = "get_settings"
	ControlElicitation        ControlSubtype = "elicitation"
)

// ControlReqInitialize initializes the SDK session.
type ControlReqInitialize struct {
	Subtype                ControlSubtype  `json:"subtype"` // ControlInitialize
	Hooks                  json.RawMessage `json:"hooks,omitempty"`
	SDKMcpServers          []string        `json:"sdkMcpServers,omitempty"`
	JSONSchema             json.RawMessage `json:"jsonSchema,omitempty"`
	SystemPrompt           string          `json:"systemPrompt,omitempty"`
	AppendSystemPrompt     string          `json:"appendSystemPrompt,omitempty"`
	Agents                 json.RawMessage `json:"agents,omitempty"`
	PromptSuggestions      bool            `json:"promptSuggestions,omitempty"`
	AgentProgressSummaries bool            `json:"agentProgressSummaries,omitempty"`
}

// ControlReqInterrupt interrupts the currently running conversation turn.
type ControlReqInterrupt struct {
	Subtype ControlSubtype `json:"subtype"` // ControlInterrupt
}

// ControlReqCanUseTool requests permission to use a tool.
type ControlReqCanUseTool struct {
	Subtype               ControlSubtype  `json:"subtype"` // ControlCanUseTool
	ToolName              string          `json:"tool_name"`
	Input                 json.RawMessage `json:"input"`
	PermissionSuggestions json.RawMessage `json:"permission_suggestions,omitempty"`
	BlockedPath           string          `json:"blocked_path,omitempty"`
	DecisionReason        string          `json:"decision_reason,omitempty"`
	Title                 string          `json:"title,omitempty"`
	DisplayName           string          `json:"display_name,omitempty"`
	ToolUseID             string          `json:"tool_use_id"`
	AgentID               string          `json:"agent_id,omitempty"`
	Description           string          `json:"description,omitempty"`
}

// ControlReqSetPermissionMode changes the tool permission mode.
type ControlReqSetPermissionMode struct {
	Subtype   ControlSubtype `json:"subtype"` // ControlSetPermissionMode
	Mode      string         `json:"mode"`
	Ultraplan bool           `json:"ultraplan,omitempty"`
}

// ControlReqSetModel switches the model for subsequent turns.
type ControlReqSetModel struct {
	Subtype ControlSubtype `json:"subtype"`         // ControlSetModel
	Model   string         `json:"model,omitempty"` // empty = reset to default
}

// ControlReqSetMaxThinkingTokens configures extended thinking token limit.
type ControlReqSetMaxThinkingTokens struct {
	Subtype           ControlSubtype `json:"subtype"`             // ControlSetMaxThinking
	MaxThinkingTokens *int           `json:"max_thinking_tokens"` // null = unlimited
}

// ControlReqMcpStatus queries status of all MCP server connections.
type ControlReqMcpStatus struct {
	Subtype ControlSubtype `json:"subtype"` // ControlMcpStatus
}

// ControlReqGetContextUsage returns a context window usage breakdown.
type ControlReqGetContextUsage struct {
	Subtype ControlSubtype `json:"subtype"` // ControlGetContextUsage
}

// ControlReqHookCallback delivers a hook callback with its input data.
type ControlReqHookCallback struct {
	Subtype    ControlSubtype  `json:"subtype"` // ControlHookCallback
	CallbackID string          `json:"callback_id"`
	Input      json.RawMessage `json:"input"`
	ToolUseID  string          `json:"tool_use_id,omitempty"`
}

// ControlReqMcpMessage sends a JSON-RPC message to a specific MCP server.
type ControlReqMcpMessage struct {
	Subtype    ControlSubtype  `json:"subtype"` // ControlMcpMessage
	ServerName string          `json:"server_name"`
	Message    json.RawMessage `json:"message"`
}

// ControlReqRewindFiles reverts file changes since a given user message.
type ControlReqRewindFiles struct {
	Subtype       ControlSubtype `json:"subtype"` // ControlRewindFiles
	UserMessageID string         `json:"user_message_id"`
	DryRun        bool           `json:"dry_run,omitempty"`
}

// ControlReqCancelAsyncMessage drops a pending async user message from
// the command queue by UUID.
type ControlReqCancelAsyncMessage struct {
	Subtype     ControlSubtype `json:"subtype"` // ControlCancelAsyncMessage
	MessageUUID string         `json:"message_uuid"`
}

// ControlReqSeedReadState seeds the readFileState cache with a path+mtime
// entry so Edit validation succeeds after a prior Read was removed from context.
type ControlReqSeedReadState struct {
	Subtype ControlSubtype `json:"subtype"` // ControlSeedReadState
	Path    string         `json:"path"`
	Mtime   int64          `json:"mtime"`
}

// ControlReqMcpSetServers replaces the set of dynamically managed MCP servers.
type ControlReqMcpSetServers struct {
	Subtype ControlSubtype  `json:"subtype"` // ControlMcpSetServers
	Servers json.RawMessage `json:"servers"`
}

// ControlReqReloadPlugins reloads plugins from disk.
type ControlReqReloadPlugins struct {
	Subtype ControlSubtype `json:"subtype"` // ControlReloadPlugins
}

// ControlReqMcpReconnect reconnects a disconnected or failed MCP server.
type ControlReqMcpReconnect struct {
	Subtype    ControlSubtype `json:"subtype"` // ControlMcpReconnect
	ServerName string         `json:"serverName"`
}

// ControlReqMcpToggle enables or disables an MCP server.
type ControlReqMcpToggle struct {
	Subtype    ControlSubtype `json:"subtype"` // ControlMcpToggle
	ServerName string         `json:"serverName"`
	Enabled    bool           `json:"enabled"`
}

// ControlReqStopTask stops a running background task.
type ControlReqStopTask struct {
	Subtype ControlSubtype `json:"subtype"` // ControlStopTask
	TaskID  string         `json:"task_id"`
}

// ControlReqApplyFlagSettings merges settings into the flag settings layer.
type ControlReqApplyFlagSettings struct {
	Subtype  ControlSubtype  `json:"subtype"` // ControlApplyFlagSettings
	Settings json.RawMessage `json:"settings"`
}

// ControlReqGetSettings returns the effective and per-source settings.
type ControlReqGetSettings struct {
	Subtype ControlSubtype `json:"subtype"` // ControlGetSettings
}

// ControlReqElicitation requests the SDK consumer to handle an MCP elicitation.
type ControlReqElicitation struct {
	Subtype         ControlSubtype  `json:"subtype"` // ControlElicitation
	MCPServerName   string          `json:"mcp_server_name"`
	Message         string          `json:"message"`
	Mode            string          `json:"mode,omitempty"` // "form" or "url"
	URL             string          `json:"url,omitempty"`
	ElicitationID   string          `json:"elicitation_id,omitempty"`
	RequestedSchema json.RawMessage `json:"requested_schema,omitempty"`
}

// ---------- control response ----------

// ControlResponseSubtype is the "subtype" discriminator for control responses.
type ControlResponseSubtype string

// ControlResponseSubtype values.
const (
	ControlResponseSuccess ControlResponseSubtype = "success"
	ControlResponseError   ControlResponseSubtype = "error"
)

// InputControlResponseMsg responds to a control request from Claude Code (type:"control_response").
type InputControlResponseMsg struct {
	Type     InputType       `json:"type"` // InputControlResponse
	Response ControlResponse `json:"response"`
}

// ControlResponse is the inner response, either success or error.
type ControlResponse struct {
	Subtype                   ControlResponseSubtype `json:"subtype"` // ControlResponseSuccess or ControlResponseError
	RequestID                 string                 `json:"request_id"`
	Response                  json.RawMessage        `json:"response,omitempty"`                    // success only
	Error                     string                 `json:"error,omitempty"`                       // error only
	PendingPermissionRequests json.RawMessage        `json:"pending_permission_requests,omitempty"` // error only
}

// ---------- keep alive / env vars ----------

// InputKeepAliveMsg is a heartbeat (type:"keep_alive").
type InputKeepAliveMsg struct {
	Type InputType `json:"type"` // InputKeepAlive
}

// InputUpdateEnvVarsMsg pushes env vars at runtime (type:"update_environment_variables").
type InputUpdateEnvVarsMsg struct {
	Type      InputType         `json:"type"` // InputUpdateEnvVars
	Variables map[string]string `json:"variables"`
}

// ---------- slash commands in -p mode ----------
//
// Slash commands can be sent as user message content (e.g. "/compact").
// In -p (print/headless) mode, only a subset is available:
//   - type="prompt" (skills like /review, /commit) — always intercepted
//   - type="local" with supportsNonInteractive=true — listed below
//
// Unrecognized commands (disabled or not in the filtered list) are passed
// through to the model as plain text, which typically fails with
// "Unknown skill: <name>".
//
// Available local commands in -p mode:
//   /compact       — shrink context (clear history, keep summary)
//   /context       — show context window usage breakdown
//   /cost          — show session cost (subscription users only)
//   /advisor       — configure advisor model (when available)
//   /release-notes — view changelog
//   /extra-usage   — extra usage info (subscription users only)
//
// Anthropic-internal only (USER_TYPE=ant):
//   /version       — print running version
//   /files         — list files in context
//   /heapdump      — dump JS heap (hidden)
//
// NOT available in -p mode (supportsNonInteractive=false or local-jsx):
//   /model, /clear, /config, /permissions, /help, /voice, /rewind,
//   /reload-plugins, /vim, /stickers, /install-slack-app, /bridge-kick
//
// For unavailable commands, use control request subtypes instead:
//   /model          → ControlSetModel
//   /reload-plugins → ControlReloadPlugins
//   (no equivalent for /clear — start a new session instead)

// ============================================================================
// Output types (received FROM the agent via stdout)
// ============================================================================

// OutputType is the top-level "type" discriminator for Claude Code stdout NDJSON.
type OutputType string

const (
	// Core message types (SDKMessageSchema).

	// OutputAssistant is a complete assistant turn with content blocks.
	OutputAssistant OutputType = "assistant"
	// OutputUser is an echoed user message (input or tool result).
	OutputUser OutputType = "user"
	// OutputResult is a terminal message with final status and usage.
	OutputResult OutputType = "result"
	// OutputSystem is a system event; dispatch further on SystemSubtype.
	OutputSystem OutputType = "system"
	// OutputStreamEvent is a partial assistant message (streaming delta).
	OutputStreamEvent OutputType = "stream_event"
	// OutputRateLimitEvent is emitted when rate limit status transitions.
	OutputRateLimitEvent OutputType = "rate_limit_event"
	// OutputToolProgress reports elapsed time for a running tool.
	OutputToolProgress OutputType = "tool_progress"
	// OutputAuthStatus reports authentication state changes.
	OutputAuthStatus OutputType = "auth_status"
	// OutputToolUseSummary summarizes preceding tool calls.
	OutputToolUseSummary OutputType = "tool_use_summary"
	// OutputPromptSuggestion is a predicted next user prompt.
	OutputPromptSuggestion OutputType = "prompt_suggestion"

	// Streamlined output types (only with --streamlined-output).

	// OutputStreamlinedText replaces assistant messages with text only.
	OutputStreamlinedText OutputType = "streamlined_text"
	// OutputStreamlinedToolUseSummary replaces tool_use blocks with a summary.
	OutputStreamlinedToolUseSummary OutputType = "streamlined_tool_use_summary"

	// Control protocol types.

	// OutputControlRequest is a control request from Claude Code to the host.
	OutputControlRequest OutputType = "control_request"
	// OutputControlResponse is a response to a control request sent by the host.
	OutputControlResponse OutputType = "control_response"
	// OutputControlCancelRequest cancels a pending control request.
	OutputControlCancelRequest OutputType = "control_cancel_request"

	// Misc.

	// OutputKeepAlive is a heartbeat.
	OutputKeepAlive OutputType = "keep_alive"
)

// SystemSubtype is the "subtype" discriminator for type="system" messages.
type SystemSubtype string

const (
	// SystemInit is the first message in a session.
	SystemInit SystemSubtype = "init"
	// SystemTaskStarted signals a background subagent has started.
	SystemTaskStarted SystemSubtype = "task_started"
	// SystemTaskNotification signals a background subagent completed/failed/stopped.
	SystemTaskNotification SystemSubtype = "task_notification"
	// SystemTaskProgress reports progress of a background subagent.
	SystemTaskProgress SystemSubtype = "task_progress"
	// SystemCompactBoundary marks where context was compacted.
	SystemCompactBoundary SystemSubtype = "compact_boundary"
	// SystemStatus reports idle/running/requires_action transitions.
	SystemStatus SystemSubtype = "status"
	// SystemSessionStateChanged mirrors notifySessionStateChanged;
	// authoritative turn-over signal ("idle" fires after result is flushed).
	SystemSessionStateChanged SystemSubtype = "session_state_changed"
	// SystemAPIRetry is emitted when an API request fails with a retryable
	// error and will be retried after a delay.
	SystemAPIRetry SystemSubtype = "api_retry"
	// SystemLocalCommandOutput is output from a local slash command.
	SystemLocalCommandOutput SystemSubtype = "local_command_output"
	// SystemHookStarted signals a hook has started executing.
	SystemHookStarted SystemSubtype = "hook_started"
	// SystemHookProgress reports partial output from a running hook.
	SystemHookProgress SystemSubtype = "hook_progress"
	// SystemHookResponse reports a hook's final result.
	SystemHookResponse SystemSubtype = "hook_response"
	// SystemFilesPersisted reports files uploaded to cloud storage.
	SystemFilesPersisted SystemSubtype = "files_persisted"
	// SystemElicitationComplete signals an MCP URL-mode elicitation is done.
	SystemElicitationComplete SystemSubtype = "elicitation_complete"
	// SystemPostTurnSummary is a background summary emitted after each
	// assistant turn (summarizes_uuid points to the assistant message).
	SystemPostTurnSummary SystemSubtype = "post_turn_summary"
)

// ---------- envelope probe ----------

// OutputTypeProbe extracts the type discriminator from a Claude Code JSONL record.
type OutputTypeProbe struct {
	Type OutputType `json:"type"`
	// Subtype is untyped because its meaning varies by Type: SystemSubtype for
	// system messages, but free-form strings like "success" or "error_max_turns"
	// for result messages.
	Subtype string `json:"subtype"`
}

// ---------- system/init ----------

// OutputInitMsg is the wire representation of a system/init record.
type OutputInitMsg struct {
	Type      OutputType    `json:"type"`
	Subtype   SystemSubtype `json:"subtype"`
	Cwd       string        `json:"cwd"`
	SessionID string        `json:"session_id"`
	Tools     []string      `json:"tools"`
	Model     string        `json:"model"`
	Version   string        `json:"claude_code_version"`
	UUID      string        `json:"uuid"`
	Timestamp string        `json:"timestamp,omitempty"`

	Agents         []string        `json:"agents,omitempty"`
	APIKeySource   string          `json:"apiKeySource,omitempty"`
	FastModeState  string          `json:"fast_mode_state,omitempty"`
	MCPServers     []InitMCPServer `json:"mcp_servers,omitempty"`
	OutputStyle    string          `json:"output_style,omitempty"`
	PermissionMode string          `json:"permissionMode,omitempty"`
	Plugins        []InitPlugin    `json:"plugins,omitempty"`
	Skills         []string        `json:"skills,omitempty"`
	SlashCommands  []string        `json:"slash_commands,omitempty"`
}

// InitMCPServer is an MCP server entry in the system/init message.
type InitMCPServer struct {
	Name   string `json:"name"`
	Status string `json:"status"`
}

// InitPlugin is a plugin entry in the system/init message.
type InitPlugin struct {
	Name   string `json:"name"`
	Path   string `json:"path"`
	Source string `json:"source"`
}

// ---------- system (non-init) ----------

// OutputSystemMsg is the wire representation of a non-init system record.
type OutputSystemMsg struct {
	Type      OutputType    `json:"type"`
	Subtype   SystemSubtype `json:"subtype"`
	SessionID string        `json:"session_id"`
	UUID      string        `json:"uuid"`
	Timestamp string        `json:"timestamp,omitempty"`

	// task_started / task_progress / task_notification fields.
	Description  string        `json:"description,omitempty"`
	TaskID       string        `json:"task_id,omitempty"`
	TaskType     string        `json:"task_type,omitempty"`
	ToolUseID    string        `json:"tool_use_id,omitempty"`
	LastToolName string        `json:"last_tool_name,omitempty"`
	Status       string        `json:"status,omitempty"`
	UsageExtra   TaskUsageWire `json:"usage,omitempty"`
	OutputFile   string        `json:"output_file,omitempty"`
	Summary      string        `json:"summary,omitempty"`

	// api_retry fields.
	Attempt      int             `json:"attempt,omitempty"`
	MaxRetries   int             `json:"max_retries,omitempty"`
	RetryDelayMs int             `json:"retry_delay_ms,omitempty"`
	ErrorStatus  int             `json:"error_status,omitempty"`
	Error        json.RawMessage `json:"error,omitempty"`

	// Other optional fields.
	PermissionMode  string              `json:"permissionMode,omitempty"`
	CompactMetadata CompactMetadataWire `json:"compact_metadata,omitempty"`
	Prompt          json.RawMessage     `json:"prompt,omitempty"`
}

// CompactMetadataWire is the compact_metadata object on compact_boundary system messages.
type CompactMetadataWire struct {
	Trigger   string `json:"trigger"`    // "auto" or "manual".
	PreTokens int    `json:"pre_tokens"` // Token count before compaction.
}

// TaskUsageWire is the usage object on task_progress/task_notification system messages.
type TaskUsageWire struct {
	TotalTokens int `json:"total_tokens"`
	ToolUses    int `json:"tool_uses"`
	DurationMs  int `json:"duration_ms"`
}

// ---------- assistant ----------

// OutputAssistantMsg is the wire representation of an assistant record.
type OutputAssistantMsg struct {
	Type            OutputType           `json:"type"`
	SessionID       string               `json:"session_id"`
	UUID            string               `json:"uuid"`
	Timestamp       string               `json:"timestamp,omitempty"`
	Message         AssistantMessageBody `json:"message"`
	ParentToolUseID string               `json:"parent_tool_use_id"`
	Error           string               `json:"error"`
}

// AssistantMessageBody is the inner message object within an assistant record.
type AssistantMessageBody struct {
	ID           string               `json:"id"`
	Type         string               `json:"type,omitempty"`
	Role         string               `json:"role"`
	Model        string               `json:"model"`
	Content      []OutputContentBlock `json:"content"`
	Usage        MsgUsage             `json:"usage"`
	StopReason   string               `json:"stop_reason"`
	StopSequence string               `json:"stop_sequence"`
	StopDetails  json.RawMessage      `json:"stop_details,omitempty"`

	Container         json.RawMessage `json:"container,omitempty"`
	ContextManagement json.RawMessage `json:"context_management,omitempty"`
}

// ContentBlockStart is the content_block field in a content_block_start streaming event.
type ContentBlockStart struct {
	Type string `json:"type"`
	ID   string `json:"id,omitempty"`
	Name string `json:"name,omitempty"`
}

// OutputContentBlock is a single content block inside an assistant message.
// This is a flat union: fields are populated depending on Type.
//
//   - "text":                Text
//   - "thinking":            Thinking, Signature
//   - "tool_use":            ID, Name, Input
//   - "tool_result":         ToolUseID, Content, IsError
//   - "server_tool_use":     ID, Name, Input, Caller
//   - "web_search_tool_use": ID, Name, Input, Caller
type OutputContentBlock struct {
	Type      string          `json:"type"`
	Text      string          `json:"text,omitempty"`
	ID        string          `json:"id,omitempty"`
	Name      string          `json:"name,omitempty"`
	Input     json.RawMessage `json:"input,omitempty"`
	Thinking  string          `json:"thinking,omitempty"`
	Signature string          `json:"signature,omitempty"`
	Caller    json.RawMessage `json:"caller,omitempty"`
	// tool_result fields (inline MCP tool results).
	ToolUseID string          `json:"tool_use_id,omitempty"`
	Content   json.RawMessage `json:"content,omitempty"`
	IsError   bool            `json:"is_error,omitempty"`
}

// ---------- user (echoed) ----------

// OutputUserMsg is the wire representation of a user record.
type OutputUserMsg struct {
	Type            OutputType      `json:"type"`
	UUID            string          `json:"uuid"`
	SessionID       string          `json:"session_id,omitempty"`
	Timestamp       string          `json:"timestamp,omitempty"`
	Message         json.RawMessage `json:"message"`
	ParentToolUseID string          `json:"parent_tool_use_id"`
	ToolUseResult   json.RawMessage `json:"tool_use_result,omitempty"`
	IsSynthetic     bool            `json:"isSynthetic,omitempty"`
	IsReplay        bool            `json:"isReplay,omitempty"`
}

// ---------- result ----------

// OutputResultMsg is the wire representation of a result record.
type OutputResultMsg struct {
	Type             OutputType `json:"type"`
	Subtype          string     `json:"subtype"`
	IsError          bool       `json:"is_error"`
	DurationMs       int64      `json:"duration_ms"`
	DurationAPIMs    int64      `json:"duration_api_ms"`
	NumTurns         int        `json:"num_turns"`
	Result           string     `json:"result"`
	Errors           []string   `json:"errors,omitempty"`
	SessionID        string     `json:"session_id"`
	TotalCostUSD     float64    `json:"total_cost_usd"`
	Usage            MsgUsage   `json:"usage"`
	UUID             string     `json:"uuid"`
	StructuredOutput string     `json:"structured_output"`
	Timestamp        string     `json:"timestamp,omitempty"`

	FastModeState     string                     `json:"fast_mode_state,omitempty"`
	ModelUsage        map[string]ModelUsageEntry `json:"modelUsage,omitempty"`
	PermissionDenials []json.RawMessage          `json:"permission_denials,omitempty"`
	StopReason        string                     `json:"stop_reason,omitempty"`
	TerminalReason    string                     `json:"terminal_reason,omitempty"`
}

// ModelUsageEntry is per-model usage stats in the result message.
type ModelUsageEntry struct {
	InputTokens              int     `json:"inputTokens"`
	OutputTokens             int     `json:"outputTokens"`
	CacheReadInputTokens     int     `json:"cacheReadInputTokens"`
	CacheCreationInputTokens int     `json:"cacheCreationInputTokens"`
	WebSearchRequests        int     `json:"webSearchRequests"`
	CostUSD                  float64 `json:"costUSD"`
	ContextWindow            int     `json:"contextWindow"`
	MaxOutputTokens          int     `json:"maxOutputTokens"`
}

// ---------- Token usage ----------

// MsgUsage holds token counts from the model.
type MsgUsage struct {
	InputTokens              int64  `json:"input_tokens"`
	OutputTokens             int64  `json:"output_tokens"`
	CacheCreationInputTokens int64  `json:"cache_creation_input_tokens"`
	CacheReadInputTokens     int64  `json:"cache_read_input_tokens"`
	ServiceTier              string `json:"service_tier,omitzero"`
	InferenceGeo             string `json:"inference_geo,omitzero"`
	Speed                    string `json:"speed,omitzero"`

	// Iterations is an int or an array, depending on the model.
	Iterations json.RawMessage `json:"iterations,omitzero"`

	ServerToolUse *ServerToolUse `json:"server_tool_use,omitzero"`
	CacheCreation *CacheCreation `json:"cache_creation,omitzero"`
}

// ServerToolUse tracks server-side tool use counts.
type ServerToolUse struct {
	WebSearchRequests int `json:"web_search_requests"`
	WebFetchRequests  int `json:"web_fetch_requests"`
}

// CacheCreation breaks down cache creation by time bucket.
type CacheCreation struct {
	Ephemeral1hInputTokens int `json:"ephemeral_1h_input_tokens"`
	Ephemeral5mInputTokens int `json:"ephemeral_5m_input_tokens"`
}

// ---------- stream_event ----------

// OutputStreamEventMsg is the wire representation of a stream_event record.
type OutputStreamEventMsg struct {
	Type            OutputType      `json:"type"`
	UUID            string          `json:"uuid"`
	SessionID       string          `json:"session_id"`
	Timestamp       string          `json:"timestamp,omitempty"`
	ParentToolUseID string          `json:"parent_tool_use_id"`
	Event           StreamEventData `json:"event"`
}

// StreamEventData is the nested event body inside a stream_event record.
type StreamEventData struct {
	Type         string          `json:"type"`
	Index        int             `json:"index"`
	Delta        *StreamDelta    `json:"delta,omitempty"`
	ContentBlock json.RawMessage `json:"content_block,omitempty"`
	// message_start carries the full message object; message_delta carries
	// stop_reason and usage in a delta wrapper.
	Message json.RawMessage `json:"message,omitempty"`
	Usage   json.RawMessage `json:"usage,omitempty"`
}

// StreamDelta is a delta object inside a stream event.
type StreamDelta struct {
	Type        string `json:"type"`
	Text        string `json:"text"`
	PartialJSON string `json:"partial_json"`
	Thinking    string `json:"thinking"`
	Signature   string `json:"signature"`
	// message_delta carries stop_reason.
	StopReason string `json:"stop_reason,omitempty"`
}

// ---------- rate_limit_event ----------

// OutputRateLimitEventMsg is the wire representation of a rate_limit_event record.
// Emitted when the CLI's rate limit status transitions (e.g. allowed → allowed_warning).
type OutputRateLimitEventMsg struct {
	Type          OutputType    `json:"type"`
	UUID          string        `json:"uuid"`
	SessionID     string        `json:"session_id"`
	Timestamp     string        `json:"timestamp,omitempty"`
	RateLimitInfo RateLimitInfo `json:"rate_limit_info"`
}

// RateLimitInfo is the nested rate limit info inside a rate_limit_event.
// Wire format uses camelCase (matches Claude Code CLI JSON output).
type RateLimitInfo struct {
	Status                string  `json:"status"`
	ResetsAt              float64 `json:"resetsAt,omitempty"`
	RateLimitType         string  `json:"rateLimitType,omitempty"`
	Utilization           float64 `json:"utilization,omitempty"`
	OverageStatus         string  `json:"overageStatus,omitempty"`
	OverageResetsAt       float64 `json:"overageResetsAt,omitempty"`
	OverageDisabledReason string  `json:"overageDisabledReason,omitempty"`
	IsUsingOverage        bool    `json:"isUsingOverage,omitempty"`
}

// ---------- tool_progress ----------

// OutputToolProgressMsg is emitted periodically while a tool is running.
type OutputToolProgressMsg struct {
	Type               OutputType `json:"type"` // OutputToolProgress
	ToolUseID          string     `json:"tool_use_id"`
	ToolName           string     `json:"tool_name"`
	ParentToolUseID    string     `json:"parent_tool_use_id"` // nullable
	ElapsedTimeSeconds int        `json:"elapsed_time_seconds"`
	TaskID             string     `json:"task_id,omitempty"`
	UUID               string     `json:"uuid"`
	SessionID          string     `json:"session_id"`
}

// ---------- auth_status ----------

// OutputAuthStatusMsg reports authentication state changes.
type OutputAuthStatusMsg struct {
	Type             OutputType `json:"type"` // OutputAuthStatus
	IsAuthenticating bool       `json:"isAuthenticating"`
	Output           []string   `json:"output"`
	Error            string     `json:"error,omitempty"`
	UUID             string     `json:"uuid"`
	SessionID        string     `json:"session_id"`
}

// ---------- tool_use_summary ----------

// OutputToolUseSummaryMsg summarizes a group of preceding tool calls.
type OutputToolUseSummaryMsg struct {
	Type                OutputType `json:"type"` // OutputToolUseSummary
	Summary             string     `json:"summary"`
	PrecedingToolUseIDs []string   `json:"preceding_tool_use_ids"`
	UUID                string     `json:"uuid"`
	SessionID           string     `json:"session_id"`
}

// ---------- prompt_suggestion ----------

// OutputPromptSuggestionMsg is a predicted next user prompt.
type OutputPromptSuggestionMsg struct {
	Type       OutputType `json:"type"` // OutputPromptSuggestion
	Suggestion string     `json:"suggestion"`
	UUID       string     `json:"uuid"`
	SessionID  string     `json:"session_id"`
}

// ---------- streamlined output ----------

// OutputStreamlinedTextMsg replaces assistant messages in streamlined output mode.
type OutputStreamlinedTextMsg struct {
	Type      OutputType `json:"type"` // OutputStreamlinedText
	Text      string     `json:"text"`
	SessionID string     `json:"session_id"`
	UUID      string     `json:"uuid"`
}

// OutputStreamlinedToolUseSummaryMsg replaces tool_use blocks in streamlined output.
type OutputStreamlinedToolUseSummaryMsg struct {
	Type        OutputType `json:"type"` // OutputStreamlinedToolUseSummary
	ToolSummary string     `json:"tool_summary"`
	SessionID   string     `json:"session_id"`
	UUID        string     `json:"uuid"`
}

// ---------- control_cancel_request ----------

// OutputControlCancelRequestMsg cancels a pending control request.
type OutputControlCancelRequestMsg struct {
	Type      OutputType `json:"type"` // OutputControlCancelRequest
	RequestID string     `json:"request_id"`
}

// ---------- system subtype output types ----------

// OutputSessionStateChangedMsg reports idle/running/requires_action transitions.
type OutputSessionStateChangedMsg struct {
	Type      OutputType    `json:"type"`    // OutputSystem
	Subtype   SystemSubtype `json:"subtype"` // SystemSessionStateChanged
	State     string        `json:"state"`   // "idle", "running", "requires_action"
	UUID      string        `json:"uuid"`
	SessionID string        `json:"session_id"`
}

// OutputPostTurnSummaryMsg is an AI-generated summary after each assistant turn.
type OutputPostTurnSummaryMsg struct {
	Type           OutputType    `json:"type"`    // OutputSystem
	Subtype        SystemSubtype `json:"subtype"` // SystemPostTurnSummary
	SummarizesUUID string        `json:"summarizes_uuid"`
	StatusCategory string        `json:"status_category"` // "blocked","waiting","completed","review_ready","failed"
	StatusDetail   string        `json:"status_detail"`
	IsNoteworthy   bool          `json:"is_noteworthy"`
	Title          string        `json:"title"`
	Description    string        `json:"description"`
	RecentAction   string        `json:"recent_action"`
	NeedsAction    string        `json:"needs_action"`
	ArtifactURLs   []string      `json:"artifact_urls"`
	UUID           string        `json:"uuid"`
	SessionID      string        `json:"session_id"`
}

// OutputLocalCommandOutputMsg is output from a local slash command (e.g. /cost).
type OutputLocalCommandOutputMsg struct {
	Type      OutputType    `json:"type"`    // OutputSystem
	Subtype   SystemSubtype `json:"subtype"` // SystemLocalCommandOutput
	Content   string        `json:"content"`
	UUID      string        `json:"uuid"`
	SessionID string        `json:"session_id"`
}

// OutputHookStartedMsg signals a hook has started executing.
type OutputHookStartedMsg struct {
	Type      OutputType    `json:"type"`    // OutputSystem
	Subtype   SystemSubtype `json:"subtype"` // SystemHookStarted
	HookID    string        `json:"hook_id"`
	HookName  string        `json:"hook_name"`
	HookEvent string        `json:"hook_event"`
	UUID      string        `json:"uuid"`
	SessionID string        `json:"session_id"`
}

// OutputHookProgressMsg reports partial output from a running hook.
type OutputHookProgressMsg struct {
	Type      OutputType    `json:"type"`    // OutputSystem
	Subtype   SystemSubtype `json:"subtype"` // SystemHookProgress
	HookID    string        `json:"hook_id"`
	HookName  string        `json:"hook_name"`
	HookEvent string        `json:"hook_event"`
	Stdout    string        `json:"stdout"`
	Stderr    string        `json:"stderr"`
	Output    string        `json:"output"`
	UUID      string        `json:"uuid"`
	SessionID string        `json:"session_id"`
}

// OutputHookResponseMsg reports a hook's final result.
type OutputHookResponseMsg struct {
	Type      OutputType    `json:"type"`    // OutputSystem
	Subtype   SystemSubtype `json:"subtype"` // SystemHookResponse
	HookID    string        `json:"hook_id"`
	HookName  string        `json:"hook_name"`
	HookEvent string        `json:"hook_event"`
	Output    string        `json:"output"`
	Stdout    string        `json:"stdout"`
	Stderr    string        `json:"stderr"`
	ExitCode  int           `json:"exit_code,omitempty"`
	Outcome   string        `json:"outcome"` // "success", "error", "cancelled"
	UUID      string        `json:"uuid"`
	SessionID string        `json:"session_id"`
}

// OutputFilesPersistedMsg reports files uploaded to cloud storage.
type OutputFilesPersistedMsg struct {
	Type    OutputType    `json:"type"`    // OutputSystem
	Subtype SystemSubtype `json:"subtype"` // SystemFilesPersisted
	Files   []struct {
		Filename string `json:"filename"`
		FileID   string `json:"file_id"`
	} `json:"files"`
	Failed []struct {
		Filename string `json:"filename"`
		Error    string `json:"error"`
	} `json:"failed"`
	ProcessedAt string `json:"processed_at"`
	UUID        string `json:"uuid"`
	SessionID   string `json:"session_id"`
}

// OutputElicitationCompleteMsg signals an MCP URL-mode elicitation is done.
type OutputElicitationCompleteMsg struct {
	Type          OutputType    `json:"type"`    // OutputSystem
	Subtype       SystemSubtype `json:"subtype"` // SystemElicitationComplete
	MCPServerName string        `json:"mcp_server_name"`
	ElicitationID string        `json:"elicitation_id"`
	UUID          string        `json:"uuid"`
	SessionID     string        `json:"session_id"`
}

// ---------- output helper types ----------

// OutputUserText is a plain-text user message body.
type OutputUserText struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// OutputUserBlock is a block-style user message body.
type OutputUserBlock struct {
	Role    string                   `json:"role"`
	Content []OutputUserContentBlock `json:"content"`
}

// OutputUserContentBlock is a single content block in a user message.
type OutputUserContentBlock struct {
	Type      string             `json:"type"`
	Text      string             `json:"text,omitempty"`
	Source    *OutputImageSource `json:"source,omitempty"`
	ToolUseID string             `json:"tool_use_id,omitempty"`
	// Nested content and error flag for inline tool_result blocks (MCP tools).
	Content []ToolResultContent `json:"content,omitempty"`
	IsError bool                `json:"is_error,omitempty"`
}

// OutputImageSource is an image source in an output user content block.
type OutputImageSource struct {
	Type      string `json:"type"`
	MediaType string `json:"media_type"`
	Data      string `json:"data"`
}

// OutputToolResult is the message body format for tool results delivered via
// the top-level parent_tool_use_id path (standard Claude Code tools).
type OutputToolResult struct {
	Content []ToolResultContent `json:"content"`
	IsError bool                `json:"is_error"`
}

// ToolResultContent is a content entry in a tool result.
type ToolResultContent struct {
	Type string `json:"type"`
	Text string `json:"text"`
}
