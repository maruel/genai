// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Wire types for the Claude Code NDJSON streaming protocol.

package claudecode

import "encoding/json"

// NDJSON message types for the Claude Code CLI stream-json protocol.
//
// Each line on stdout is one of these types, discriminated by the "type" field.
// Each line on stdin is an inputUser.
//
// References:
//   - https://platform.claude.com/docs/en/agent-sdk/streaming-output
//   - https://github.com/anthropics/claude-agent-sdk-python (src/claude_agent_sdk/types.py)

// ============================================================================
// Input types (sent TO the agent via stdin)
// ============================================================================
//
// Claude Code accepts five NDJSON message types on stdin when running with
// --input-format stream-json. See controlSchemas.ts StdinMessageSchema in the
// Claude Code source.

// inputType is the top-level "type" discriminator for Claude Code stdin NDJSON.
type inputType string

const (
	// inputTypeUser sends a user turn.
	inputTypeUser inputType = "user"
	// inputTypeControlRequest sends a control request to Claude Code.
	inputTypeControlRequest inputType = "control_request"
	// inputTypeControlResponse responds to a control request from Claude Code.
	inputTypeControlResponse inputType = "control_response"
	// inputTypeKeepAlive is a heartbeat.
	inputTypeKeepAlive inputType = "keep_alive"
	// inputTypeUpdateEnvVars pushes env vars at runtime.
	inputTypeUpdateEnvVars inputType = "update_environment_variables"
)

// ---------- user message ----------

// inputUser sends a user turn to Claude Code (type:"user").
type inputUser struct {
	Type    inputType        `json:"type"` // inputTypeUser
	Message inputUserContent `json:"message"`
}

// inputUserContent holds the user message payload.
// Content is a plain string for text-only messages, or []inputContentBlock for
// multi-modal messages that include images.
type inputUserContent struct {
	Role    string `json:"role"`    // always "user"
	Content any    `json:"content"` // string or []inputContentBlock
}

// inputContentBlock is a single content block inside a multi-modal user message.
type inputContentBlock struct {
	Type   string            `json:"type"`             // "text" or "image"
	Text   string            `json:"text,omitempty"`   // Type == "text"
	Source *inputImageSource `json:"source,omitempty"` // Type == "image"
}

// inputImageSource describes the data source for an image content block.
type inputImageSource struct {
	Type      string `json:"type"`                 // "base64" or "url"
	MediaType string `json:"media_type,omitempty"` // e.g. "image/png"
	Data      string `json:"data,omitempty"`       // base64-encoded bytes
	URL       string `json:"url,omitempty"`        // Type == "url"
}

// ---------- control request ----------

// inputControlRequest sends a control request to Claude Code (type:"control_request").
// The Request field is a JSON object whose "subtype" discriminator determines
// its schema. Use one of the controlReq* structs below as the Request value.
type inputControlRequest struct {
	Type      inputType `json:"type"` // inputTypeControlRequest
	RequestID string    `json:"request_id"`
	Request   any       `json:"request"`
}

// controlSubtype is the "subtype" discriminator for control requests.
type controlSubtype string

// controlSubtype values for control request subtypes.
const (
	controlInitialize         controlSubtype = "initialize"
	controlInterrupt          controlSubtype = "interrupt"
	controlCanUseTool         controlSubtype = "can_use_tool"
	controlSetPermissionMode  controlSubtype = "set_permission_mode"
	controlSetModel           controlSubtype = "set_model"
	controlSetMaxThinking     controlSubtype = "set_max_thinking_tokens"
	controlMcpStatus          controlSubtype = "mcp_status"
	controlGetContextUsage    controlSubtype = "get_context_usage"
	controlHookCallback       controlSubtype = "hook_callback"
	controlMcpMessage         controlSubtype = "mcp_message"
	controlRewindFiles        controlSubtype = "rewind_files"
	controlCancelAsyncMessage controlSubtype = "cancel_async_message"
	controlSeedReadState      controlSubtype = "seed_read_state"
	controlMcpSetServers      controlSubtype = "mcp_set_servers"
	controlReloadPlugins      controlSubtype = "reload_plugins"
	controlMcpReconnect       controlSubtype = "mcp_reconnect"
	controlMcpToggle          controlSubtype = "mcp_toggle"
	controlStopTask           controlSubtype = "stop_task"
	controlApplyFlagSettings  controlSubtype = "apply_flag_settings"
	controlGetSettings        controlSubtype = "get_settings"
	controlElicitation        controlSubtype = "elicitation"
)

// controlReqInitialize initializes the SDK session.
type controlReqInitialize struct {
	Subtype            controlSubtype  `json:"subtype"` // controlInitialize
	Hooks              json.RawMessage `json:"hooks,omitempty"`
	SDKMcpServers      []string        `json:"sdkMcpServers,omitempty"`
	JSONSchema         json.RawMessage `json:"jsonSchema,omitempty"`
	SystemPrompt       string          `json:"systemPrompt,omitempty"`
	AppendSystemPrompt string          `json:"appendSystemPrompt,omitempty"`
}

// controlReqInterrupt interrupts the currently running conversation turn.
type controlReqInterrupt struct {
	Subtype controlSubtype `json:"subtype"` // controlInterrupt
}

// controlReqSetModel switches the model for subsequent turns.
type controlReqSetModel struct {
	Subtype controlSubtype `json:"subtype"`         // controlSetModel
	Model   string         `json:"model,omitempty"` // empty = reset to default
}

// controlReqGetContextUsage returns a context window usage breakdown.
type controlReqGetContextUsage struct {
	Subtype controlSubtype `json:"subtype"` // controlGetContextUsage
}

// controlReqMcpToggle enables or disables an MCP server.
type controlReqMcpToggle struct {
	Subtype    controlSubtype `json:"subtype"` // controlMcpToggle
	ServerName string         `json:"serverName"`
	Enabled    bool           `json:"enabled"`
}

// controlReqMcpReconnect reconnects a disconnected or failed MCP server.
type controlReqMcpReconnect struct {
	Subtype    controlSubtype `json:"subtype"` // controlMcpReconnect
	ServerName string         `json:"serverName"`
}

// ---------- control response ----------

// controlResponseSubtype is the "subtype" discriminator for control responses.
type controlResponseSubtype string

const (
	controlResponseSuccess controlResponseSubtype = "success"
	controlResponseError   controlResponseSubtype = "error"
)

// inputControlResponse responds to a control request from Claude Code (type:"control_response").
type inputControlResponse struct {
	Type     inputType       `json:"type"` // inputTypeControlResponse
	Response controlResponse `json:"response"`
}

// controlResponse is the inner response, either success or error.
type controlResponse struct {
	Subtype   controlResponseSubtype `json:"subtype"` // controlResponseSuccess or controlResponseError
	RequestID string                 `json:"request_id"`
	Response  json.RawMessage        `json:"response,omitempty"` // success only
	Error     string                 `json:"error,omitempty"`    // error only
}

// ---------- keep alive / env vars ----------

// inputKeepAlive is a heartbeat (type:"keep_alive").
type inputKeepAlive struct {
	Type inputType `json:"type"` // inputTypeKeepAlive
}

// inputUpdateEnvVars pushes env vars at runtime (type:"update_environment_variables").
type inputUpdateEnvVars struct {
	Type      inputType         `json:"type"` // inputTypeUpdateEnvVars
	Variables map[string]string `json:"variables"`
}

// Compile-time assertions for input wire types.
var (
	_ = inputControlRequest{}
	_ = inputControlResponse{}
	_ = inputKeepAlive{}
	_ = inputUpdateEnvVars{}

	_ = controlReqInitialize{}
	_ = controlReqInterrupt{}
	_ = controlReqSetModel{}
	_ = controlReqGetContextUsage{}
	_ = controlReqMcpToggle{}
	_ = controlReqMcpReconnect{}
)

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
// NOT available in -p mode (supportsNonInteractive=false or local-jsx):
//   /model, /clear, /config, /permissions, /help, /voice, /rewind,
//   /reload-plugins, /vim, /stickers, /install-slack-app, /bridge-kick
//
// For unavailable commands, use control request subtypes instead:
//   /model          → controlSetModel
//   /reload-plugins → controlReloadPlugins
//   (no equivalent for /clear — start a new session instead)

// ============================================================================
// Output types (received FROM the agent via stdout)
// ============================================================================

// outputType is the top-level "type" discriminator for Claude Code stdout NDJSON.
type outputType string

const (
	// Core message types.

	// outputTypeAssistant is a complete assistant turn with content blocks.
	outputTypeAssistant outputType = "assistant"
	// outputTypeUser is an echoed user message (input or tool result).
	outputTypeUser outputType = "user"
	// outputTypeResult is a terminal message with final status and usage.
	outputTypeResult outputType = "result"
	// outputTypeSystem is a system event; dispatch further on systemSubtype.
	outputTypeSystem outputType = "system"
	// outputTypeStreamEvent is a partial assistant message (streaming delta).
	outputTypeStreamEvent outputType = "stream_event"
	// outputTypeRateLimitEvent is emitted when rate limit status transitions.
	outputTypeRateLimitEvent outputType = "rate_limit_event"
	// outputTypeToolProgress reports elapsed time for a running tool.
	outputTypeToolProgress outputType = "tool_progress"
	// outputTypeAuthStatus reports authentication state changes.
	outputTypeAuthStatus outputType = "auth_status"
	// outputTypeToolUseSummary summarizes preceding tool calls.
	outputTypeToolUseSummary outputType = "tool_use_summary"
	// outputTypePromptSuggestion is a predicted next user prompt.
	outputTypePromptSuggestion outputType = "prompt_suggestion"

	// Streamlined output types (only with --streamlined-output).

	// outputTypeStreamlinedText replaces assistant messages with text only.
	outputTypeStreamlinedText outputType = "streamlined_text"
	// outputTypeStreamlinedToolUseSummary replaces tool_use blocks with a summary.
	outputTypeStreamlinedToolUseSummary outputType = "streamlined_tool_use_summary"

	// Control protocol types.

	// outputTypeControlRequest is a control request from Claude Code to the host.
	outputTypeControlRequest outputType = "control_request"
	// outputTypeControlResponse is a response to a control request sent by the host.
	outputTypeControlResponse outputType = "control_response"
	// outputTypeControlCancelRequest cancels a pending control request.
	outputTypeControlCancelRequest outputType = "control_cancel_request"

	// Misc.

	// outputTypeKeepAlive is a heartbeat.
	outputTypeKeepAlive outputType = "keep_alive"
)

// systemSubtype is the "subtype" discriminator for type="system" messages.
type systemSubtype string

const (
	// systemInit is the first message in a session.
	systemInit systemSubtype = "init"
	// systemTaskStarted signals a background subagent has started.
	systemTaskStarted systemSubtype = "task_started"
	// systemTaskNotification signals a background subagent completed/failed/stopped.
	systemTaskNotification systemSubtype = "task_notification"
	// systemTaskProgress reports progress of a background subagent.
	systemTaskProgress systemSubtype = "task_progress"
	// systemCompactBoundary marks where context was compacted.
	systemCompactBoundary systemSubtype = "compact_boundary"
	// systemStatus reports idle/running/requires_action transitions.
	systemStatus systemSubtype = "status"
	// systemSessionStateChanged mirrors notifySessionStateChanged;
	// authoritative turn-over signal ("idle" fires after result is flushed).
	systemSessionStateChanged systemSubtype = "session_state_changed"
	// systemAPIRetry is emitted when an API request fails with a retryable
	// error and will be retried after a delay.
	systemAPIRetry systemSubtype = "api_retry"
	// systemLocalCommandOutput is output from a local slash command.
	systemLocalCommandOutput systemSubtype = "local_command_output"
	// systemHookStarted signals a hook has started executing.
	systemHookStarted systemSubtype = "hook_started"
	// systemHookProgress reports partial output from a running hook.
	systemHookProgress systemSubtype = "hook_progress"
	// systemHookResponse reports a hook's final result.
	systemHookResponse systemSubtype = "hook_response"
	// systemFilesPersisted reports files uploaded to cloud storage.
	systemFilesPersisted systemSubtype = "files_persisted"
	// systemElicitationComplete signals an MCP URL-mode elicitation is done.
	systemElicitationComplete systemSubtype = "elicitation_complete"
	// systemPostTurnSummary is a background summary emitted after each
	// assistant turn (summarizes_uuid points to the assistant message).
	systemPostTurnSummary systemSubtype = "post_turn_summary"
)

// ---------- envelope probe ----------

// outputTypeProbe extracts the type discriminator from a Claude Code JSONL record.
type outputTypeProbe struct {
	Type outputType `json:"type"`
	// Subtype is untyped because its meaning varies by Type: systemSubtype for
	// system messages, but free-form strings like "success" or "error_max_turns"
	// for result messages.
	Subtype string `json:"subtype,omitempty"`
}

// ---------- system/init ----------

// outputInit is the first message emitted on stdout, type="system" subtype="init".
type outputInit struct {
	Type      outputType      `json:"type"`
	Subtype   systemSubtype   `json:"subtype"`
	SessionID string          `json:"session_id"`
	Cwd       string          `json:"cwd,omitzero"`
	Model     string          `json:"model"`
	Tools     []string        `json:"tools"`
	Version   string          `json:"claude_code_version"`
	UUID      string          `json:"uuid,omitzero"`
	Timestamp json.RawMessage `json:"timestamp,omitzero"`

	// Optional fields present in verbose mode.
	Agents         json.RawMessage `json:"agents,omitzero"`
	APIKeySource   json.RawMessage `json:"apiKeySource,omitzero"`
	FastModeState  json.RawMessage `json:"fast_mode_state,omitzero"`
	MCPServers     json.RawMessage `json:"mcp_servers,omitzero"`
	OutputStyle    json.RawMessage `json:"output_style,omitzero"`
	PermissionMode json.RawMessage `json:"permissionMode,omitzero"`
	Plugins        json.RawMessage `json:"plugins,omitzero"`
	Skills         json.RawMessage `json:"skills,omitzero"`
	SlashCommands  json.RawMessage `json:"slash_commands,omitzero"`
}

// ---------- system (non-init) ----------

// outputSystem is emitted for non-init system records, type="system" with various subtypes.
//
// Known subtypes include: task_started, task_progress, task_notification,
// api_retry, compact_boundary, permission_mode_change.
// Fields are populated depending on subtype.
type outputSystem struct {
	Type      outputType      `json:"type"`
	Subtype   systemSubtype   `json:"subtype"`
	SessionID string          `json:"session_id"`
	UUID      string          `json:"uuid,omitzero"`
	Timestamp json.RawMessage `json:"timestamp,omitzero"`

	// task_started / task_progress / task_notification fields.
	Description  json.RawMessage `json:"description,omitzero"`
	TaskID       json.RawMessage `json:"task_id,omitzero"`
	TaskType     json.RawMessage `json:"task_type,omitzero"`
	ToolUseID    json.RawMessage `json:"tool_use_id,omitzero"`
	LastToolName json.RawMessage `json:"last_tool_name,omitzero"`
	Status       json.RawMessage `json:"status,omitzero"`
	Usage        json.RawMessage `json:"usage,omitzero"`
	OutputFile   json.RawMessage `json:"output_file,omitzero"`
	Summary      json.RawMessage `json:"summary,omitzero"`

	// api_retry fields.
	Attempt      json.RawMessage `json:"attempt,omitzero"`
	MaxRetries   json.RawMessage `json:"max_retries,omitzero"`
	RetryDelayMs json.RawMessage `json:"retry_delay_ms,omitzero"`
	ErrorStatus  json.RawMessage `json:"error_status,omitzero"`
	Error        json.RawMessage `json:"error,omitzero"`

	// compact_boundary fields.
	CompactMetadata json.RawMessage `json:"compact_metadata,omitzero"`

	// Other optional fields.
	PermissionMode json.RawMessage `json:"permissionMode,omitzero"`
	Prompt         json.RawMessage `json:"prompt,omitzero"`
}

// ---------- assistant ----------

// outputAssistant carries a complete assistant response, type="assistant".
type outputAssistant struct {
	Type            outputType           `json:"type"`
	SessionID       string               `json:"session_id,omitzero"`
	UUID            string               `json:"uuid,omitzero"`
	Timestamp       json.RawMessage      `json:"timestamp,omitzero"`
	Message         assistantMessageBody `json:"message"`
	ParentToolUseID string               `json:"parent_tool_use_id,omitzero"`
	Error           string               `json:"error,omitzero"`
}

// assistantMessageBody is the inner message from the model.
type assistantMessageBody struct {
	ID         string               `json:"id,omitzero"`
	Type       string               `json:"type,omitzero"` // "message"
	Role       string               `json:"role,omitzero"`
	Model      string               `json:"model,omitzero"`
	Content    []outputContentBlock `json:"content"`
	StopReason string               `json:"stop_reason,omitzero"`
	StopSeq    *string              `json:"stop_sequence,omitzero"`
	Usage      msgUsage             `json:"usage"`
	Container  json.RawMessage      `json:"container,omitzero"`
	CtxMgmt    json.RawMessage      `json:"context_management,omitzero"`
}

// outputContentBlock is a single block within an assistantMessageBody.
// This is a flat union: fields are populated depending on Type.
//
//   - "text":        Text
//   - "thinking":    Thinking, Signature
//   - "tool_use":    ID, Name, Input
//   - "tool_result": ToolUseID, Content, IsError
type outputContentBlock struct {
	Type string `json:"type"`

	// text fields.
	Text string `json:"text,omitempty"`

	// thinking fields.
	Thinking  string `json:"thinking,omitempty"`
	Signature string `json:"signature,omitempty"`

	// tool_use fields.
	ID    string          `json:"id,omitempty"`
	Name  string          `json:"name,omitempty"`
	Input json.RawMessage `json:"input,omitempty"`

	// tool_result fields (inline MCP tool results).
	ToolUseID string          `json:"tool_use_id,omitempty"`
	Content   json.RawMessage `json:"content,omitempty"` // []outputContentBlock
	IsError   bool            `json:"is_error,omitempty"`
}

// ---------- user (echoed) ----------

// outputUser is emitted for user records, type="user".
// Not sent by the genai provider, but appears on stdout during multi-turn.
type outputUser struct {
	Type            outputType      `json:"type"`
	UUID            string          `json:"uuid,omitzero"`
	SessionID       string          `json:"session_id,omitzero"`
	Timestamp       json.RawMessage `json:"timestamp,omitzero"`
	Message         json.RawMessage `json:"message"` // outputUserText or outputUserBlock
	ParentToolUseID *string         `json:"parent_tool_use_id,omitzero"`
	ToolUseResult   json.RawMessage `json:"tool_use_result,omitzero"`
	IsSynthetic     bool            `json:"isSynthetic,omitzero"`
	IsReplay        bool            `json:"isReplay,omitzero"`
}

// outputUserText is the user message body when content is a plain string.
type outputUserText struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// outputUserBlock is the user message body when content is an array of blocks.
type outputUserBlock struct {
	Role    string                   `json:"role"`
	Content []outputUserContentBlock `json:"content"`
}

// outputUserContentBlock is a single content block in a user message.
type outputUserContentBlock struct {
	Type      string             `json:"type"`
	Text      string             `json:"text,omitempty"`
	Source    *outputImageSource `json:"source,omitempty"`
	ToolUseID string             `json:"tool_use_id,omitempty"`
	// Nested content and error flag for inline tool_result blocks (MCP tools).
	Content []toolResultContent `json:"content,omitempty"`
	IsError bool                `json:"is_error,omitempty"`
}

// outputImageSource is the image source inside a user content block.
type outputImageSource struct {
	Type      string `json:"type"`
	MediaType string `json:"media_type"`
	Data      string `json:"data"`
}

// toolResultContent is a single content item inside a tool result.
type toolResultContent struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

// ---------- result ----------

// outputResult is the terminal message emitted when the session ends, type="result".
type outputResult struct {
	Type             outputType      `json:"type"`
	Subtype          string          `json:"subtype"`
	IsError          bool            `json:"is_error"`
	Result           string          `json:"result"`
	Errors           []string        `json:"errors,omitzero"`
	StopReason       string          `json:"stop_reason,omitzero"`
	SessionID        string          `json:"session_id"`
	Usage            msgUsage        `json:"usage"`
	TotalCostUSD     float64         `json:"total_cost_usd"`
	DurationMs       int64           `json:"duration_ms,omitzero"`
	DurationAPIMs    int64           `json:"duration_api_ms,omitzero"`
	NumTurns         int             `json:"num_turns,omitzero"`
	UUID             string          `json:"uuid,omitzero"`
	StructuredOutput *string         `json:"structured_output,omitzero"`
	Timestamp        json.RawMessage `json:"timestamp,omitzero"`

	FastModeState     json.RawMessage `json:"fast_mode_state,omitzero"`
	ModelUsage        json.RawMessage `json:"modelUsage,omitzero"`
	PermissionDenials json.RawMessage `json:"permission_denials,omitzero"`
}

// ---------- Token usage ----------

// msgUsage holds token counts from the model.
type msgUsage struct {
	InputTokens              int64  `json:"input_tokens"`
	OutputTokens             int64  `json:"output_tokens"`
	CacheCreationInputTokens int64  `json:"cache_creation_input_tokens"`
	CacheReadInputTokens     int64  `json:"cache_read_input_tokens"`
	ServiceTier              string `json:"service_tier,omitzero"`
	InferenceGeo             string `json:"inference_geo,omitzero"`

	// Iterations is an int or an array, depending on the model.
	Iterations json.RawMessage `json:"iterations,omitzero"`

	ServerToolUse *serverToolUse `json:"server_tool_use,omitzero"`
	CacheCreation *cacheCreation `json:"cache_creation,omitzero"`
}

// serverToolUse tracks server-side tool use counts.
type serverToolUse struct {
	WebSearchRequests int `json:"web_search_requests"`
	WebFetchRequests  int `json:"web_fetch_requests"`
}

// cacheCreation breaks down cache creation by time bucket.
type cacheCreation struct {
	Ephemeral1hInputTokens int `json:"ephemeral_1h_input_tokens"`
	Ephemeral5mInputTokens int `json:"ephemeral_5m_input_tokens"`
}

// ---------- stream_event ----------

// outputStreamEvent carries a partial streaming update, type="stream_event".
// Only emitted when --include-partial-messages is set.
type outputStreamEvent struct {
	Type            outputType      `json:"type"`
	UUID            string          `json:"uuid,omitzero"`
	SessionID       string          `json:"session_id,omitzero"`
	Timestamp       json.RawMessage `json:"timestamp,omitzero"`
	ParentToolUseID string          `json:"parent_tool_use_id,omitzero"`
	Event           streamEventData `json:"event"`
}

// streamEventData holds the event data for a single streaming update.
type streamEventData struct {
	Type         string          `json:"type"`                    // content_block_start, content_block_delta, content_block_stop, message_start, message_delta, message_stop, error
	Index        int             `json:"index"`                   // content block index
	Delta        *streamDelta    `json:"delta,omitempty"`         // present for content_block_delta and message_delta
	ContentBlock json.RawMessage `json:"content_block,omitempty"` // present for content_block_start
	// message_start carries the full message object.
	Message json.RawMessage `json:"message,omitempty"`
	// message_delta carries stop_reason and usage in a delta wrapper.
	Usage json.RawMessage `json:"usage,omitempty"`
}

// streamDelta is the payload inside a content_block_delta or message_delta event.
type streamDelta struct {
	Type        string `json:"type"` // text_delta, thinking_delta, input_json_delta, signature_delta
	Text        string `json:"text,omitempty"`
	Thinking    string `json:"thinking,omitempty"`
	PartialJSON string `json:"partial_json,omitempty"`
	Signature   string `json:"signature,omitempty"`
	// message_delta carries stop_reason.
	StopReason string `json:"stop_reason,omitempty"`
}

// contentBlockStart is the content_block field in a content_block_start event.
type contentBlockStart struct {
	Type string `json:"type"`
	ID   string `json:"id,omitempty"`
	Name string `json:"name,omitempty"`
}

// ---------- rate_limit_event ----------

// outputRateLimitEvent is emitted when the CLI's rate limit status transitions
// (e.g. allowed → allowed_warning), type="rate_limit_event".
type outputRateLimitEvent struct {
	Type          outputType      `json:"type"`
	UUID          string          `json:"uuid,omitzero"`
	SessionID     string          `json:"session_id,omitzero"`
	Timestamp     json.RawMessage `json:"timestamp,omitzero"`
	RateLimitInfo rateLimitInfo   `json:"rate_limit_info"`
}

// rateLimitInfo is the nested rate limit info inside a rate_limit_event.
type rateLimitInfo struct {
	Status                string          `json:"status"` // "allowed", "allowed_warning", "limited"
	ResetsAt              json.RawMessage `json:"resets_at,omitzero"`
	RateLimitType         json.RawMessage `json:"rate_limit_type,omitzero"`
	Utilization           json.RawMessage `json:"utilization,omitzero"`
	OverageStatus         json.RawMessage `json:"overage_status,omitzero"`
	OverageResetsAt       json.RawMessage `json:"overage_resets_at,omitzero"`
	OverageDisabledReason json.RawMessage `json:"overage_disabled_reason,omitzero"`
}

// ---------- progress ----------

// outputProgress is emitted during tool execution, type="progress".
type outputProgress struct {
	Type            outputType      `json:"type"`
	UUID            string          `json:"uuid,omitzero"`
	SessionID       string          `json:"session_id,omitzero"`
	Timestamp       json.RawMessage `json:"timestamp,omitzero"`
	ParentToolUseID string          `json:"parentToolUseID,omitzero"`
	ToolUseID       string          `json:"toolUseID,omitzero"`
	Data            json.RawMessage `json:"data"`
}

// progressData is the data field inside a progress record.
// The Type field discriminates variants.
type progressData struct {
	Type string `json:"type"` // hook_progress, bash_progress, agent_progress, query_update, search_results_received, waiting_for_task

	// hook_progress fields.
	HookEvent string `json:"hookEvent,omitzero"`
	HookName  string `json:"hookName,omitzero"`
	Command   string `json:"command,omitzero"`

	// bash_progress fields.
	Output             string  `json:"output,omitzero"`
	FullOutput         string  `json:"fullOutput,omitzero"`
	TotalLines         int     `json:"totalLines,omitzero"`
	ElapsedTimeSeconds float64 `json:"elapsedTimeSeconds,omitzero"`
	TimeoutMs          int     `json:"timeoutMs,omitzero"`

	// agent_progress fields.
	AgentID            string          `json:"agentId,omitzero"`
	Message            string          `json:"message,omitzero"`
	Prompt             string          `json:"prompt,omitzero"`
	NormalizedMessages json.RawMessage `json:"normalizedMessages,omitzero"`

	// query_update / search_results_received fields.
	Query       string `json:"query,omitzero"`
	ResultCount int    `json:"resultCount,omitzero"`

	// waiting_for_task fields.
	TaskDescription string `json:"taskDescription,omitzero"`
	TaskType        string `json:"taskType,omitzero"`
}

// ---------- documentation-only output types ----------
//
// These types are not parsed by the genai provider (they fall through to
// a no-op) but document the full stdout wire protocol.

// outputToolProgress is emitted periodically while a tool is running.
type outputToolProgress struct {
	Type               outputType `json:"type"` // outputTypeToolProgress
	ToolUseID          string     `json:"tool_use_id"`
	ToolName           string     `json:"tool_name"`
	ParentToolUseID    string     `json:"parent_tool_use_id"`
	ElapsedTimeSeconds int        `json:"elapsed_time_seconds"`
	TaskID             string     `json:"task_id,omitempty"`
	UUID               string     `json:"uuid"`
	SessionID          string     `json:"session_id"`
}

// outputAuthStatus reports authentication state changes.
type outputAuthStatus struct {
	Type             outputType `json:"type"` // outputTypeAuthStatus
	IsAuthenticating bool       `json:"isAuthenticating"`
	Output           []string   `json:"output"`
	Error            string     `json:"error,omitempty"`
	UUID             string     `json:"uuid"`
	SessionID        string     `json:"session_id"`
}

// outputToolUseSummary summarizes a group of preceding tool calls.
type outputToolUseSummary struct {
	Type                outputType `json:"type"` // outputTypeToolUseSummary
	Summary             string     `json:"summary"`
	PrecedingToolUseIDs []string   `json:"preceding_tool_use_ids"`
	UUID                string     `json:"uuid"`
	SessionID           string     `json:"session_id"`
}

// outputPromptSuggestion is a predicted next user prompt.
type outputPromptSuggestion struct {
	Type       outputType `json:"type"` // outputTypePromptSuggestion
	Suggestion string     `json:"suggestion"`
	UUID       string     `json:"uuid"`
	SessionID  string     `json:"session_id"`
}

// outputStreamlinedText replaces assistant messages in streamlined output mode.
type outputStreamlinedText struct {
	Type      outputType `json:"type"` // outputTypeStreamlinedText
	Text      string     `json:"text"`
	SessionID string     `json:"session_id"`
	UUID      string     `json:"uuid"`
}

// outputStreamlinedToolUseSummary replaces tool_use blocks in streamlined output.
type outputStreamlinedToolUseSummary struct {
	Type        outputType `json:"type"` // outputTypeStreamlinedToolUseSummary
	ToolSummary string     `json:"tool_summary"`
	SessionID   string     `json:"session_id"`
	UUID        string     `json:"uuid"`
}

// outputControlCancelRequest cancels a pending control request.
type outputControlCancelRequest struct {
	Type      outputType `json:"type"` // outputTypeControlCancelRequest
	RequestID string     `json:"request_id"`
}

// outputSessionStateChanged reports idle/running/requires_action transitions.
type outputSessionStateChanged struct {
	Type      outputType    `json:"type"`    // outputTypeSystem
	Subtype   systemSubtype `json:"subtype"` // systemSessionStateChanged
	State     string        `json:"state"`   // "idle", "running", "requires_action"
	UUID      string        `json:"uuid"`
	SessionID string        `json:"session_id"`
}

// outputPostTurnSummary is an AI-generated summary after each assistant turn.
type outputPostTurnSummary struct {
	Type           outputType    `json:"type"`    // outputTypeSystem
	Subtype        systemSubtype `json:"subtype"` // systemPostTurnSummary
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

// Compile-time assertions for documentation-only output wire types.
var (
	_ = outputToolProgress{}
	_ = outputAuthStatus{}
	_ = outputToolUseSummary{}
	_ = outputPromptSuggestion{}
	_ = outputStreamlinedText{}
	_ = outputStreamlinedToolUseSummary{}
	_ = outputControlCancelRequest{}
	_ = outputSessionStateChanged{}
	_ = outputPostTurnSummary{}
)
