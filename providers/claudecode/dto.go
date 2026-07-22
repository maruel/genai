// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Wire types for the Claude Code NDJSON streaming protocol.

package claudecode

import (
	"bytes"
	"encoding/json"
	"errors"
	"slices"
	"strings"
	"time"

	"github.com/maruel/genai/base"
	"github.com/maruel/genai/providers/anthropic"
)

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
	Source anthropic.Source `json:"source,omitzero"`
	Text   string           `json:"text,omitempty"`
}

// ---------- control request ----------

// InputControlRequestMsg sends a control request to Claude Code (type:"control_request").
// The Request field is a JSON object whose "subtype" discriminator determines
// its schema. Use one of the ControlReq* structs below as the Request value.
type InputControlRequestMsg struct {
	Type      InputType       `json:"type"` // InputControlRequest
	RequestID string          `json:"request_id"`
	Request   json.RawMessage `json:"request"`
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
	Hooks                  HooksConfig     `json:"hooks,omitempty"`
	SDKMcpServers          []string        `json:"sdkMcpServers,omitempty"`
	JSONSchema             JSONSchema      `json:"jsonSchema,omitempty"`
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
	Subtype               ControlSubtype             `json:"subtype"` // ControlCanUseTool
	ToolName              string                     `json:"tool_name"`
	Input                 map[string]json.RawMessage `json:"input"`
	PermissionSuggestions []PermissionUpdate         `json:"permission_suggestions,omitempty"`
	BlockedPath           string                     `json:"blocked_path,omitempty"`
	DecisionReason        string                     `json:"decision_reason,omitempty"`
	Title                 string                     `json:"title,omitempty"`
	DisplayName           string                     `json:"display_name,omitempty"`
	ToolUseID             string                     `json:"tool_use_id"`
	AgentID               string                     `json:"agent_id,omitempty"`
	Description           string                     `json:"description,omitempty"`
}

// ControlCanUseToolBehavior is the can_use_tool control response behavior.
type ControlCanUseToolBehavior string

// ControlCanUseToolBehavior values.
const (
	ControlCanUseToolBehaviorAllow ControlCanUseToolBehavior = "allow"
	ControlCanUseToolBehaviorDeny  ControlCanUseToolBehavior = "deny"
)

// ControlResponsePayload is the can_use_tool permission decision payload.
type ControlResponsePayload struct {
	Behavior           ControlCanUseToolBehavior `json:"behavior"`
	UpdatedInput       json.RawMessage           `json:"updatedInput,omitempty"`       // Required for allow: original or modified tool input.
	UpdatedPermissions []PermissionUpdate        `json:"updatedPermissions,omitempty"` // Optional for allow.
	Message            string                    `json:"message,omitempty"`            // Required for deny.
	Interrupt          bool                      `json:"interrupt,omitempty"`          // Optional for deny.
	ToolUseID          string                    `json:"toolUseID,omitempty"`
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
	Subtype    ControlSubtype    `json:"subtype"` // ControlHookCallback
	CallbackID string            `json:"callback_id"`
	Input      HookCallbackInput `json:"input"`
	ToolUseID  string            `json:"tool_use_id,omitempty"`
}

// ControlReqMcpMessage sends a JSON-RPC message to a specific MCP server.
type ControlReqMcpMessage struct {
	Subtype    ControlSubtype `json:"subtype"` // ControlMcpMessage
	ServerName string         `json:"server_name"`
	Message    JSONRPCMessage `json:"message"`
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
	Subtype ControlSubtype             `json:"subtype"` // ControlMcpSetServers
	Servers map[string]json.RawMessage `json:"servers"`
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
	Subtype  ControlSubtype             `json:"subtype"` // ControlApplyFlagSettings
	Settings map[string]json.RawMessage `json:"settings"`
}

// ControlReqGetSettings returns the effective and per-source settings.
type ControlReqGetSettings struct {
	Subtype ControlSubtype `json:"subtype"` // ControlGetSettings
}

// ControlReqElicitation requests the SDK consumer to handle an MCP elicitation.
type ControlReqElicitation struct {
	Subtype         ControlSubtype `json:"subtype"` // ControlElicitation
	MCPServerName   string         `json:"mcp_server_name"`
	Message         string         `json:"message"`
	Mode            string         `json:"mode,omitempty"` // "form" or "url"
	URL             string         `json:"url,omitempty"`
	ElicitationID   string         `json:"elicitation_id,omitempty"`
	RequestedSchema JSONSchema     `json:"requested_schema,omitempty"`
}

// JSONSchema is an open JSON Schema object.
type JSONSchema map[string]json.RawMessage

// HookEvent identifies a Claude Code hook event.
type HookEvent string

// HookEvent values.
const (
	HookPreToolUse        HookEvent = "PreToolUse"
	HookPostToolUse       HookEvent = "PostToolUse"
	HookPostToolUseFail   HookEvent = "PostToolUseFailure"
	HookUserPromptSubmit  HookEvent = "UserPromptSubmit"
	HookStop              HookEvent = "Stop"
	HookSubagentStop      HookEvent = "SubagentStop"
	HookPreCompact        HookEvent = "PreCompact"
	HookNotification      HookEvent = "Notification"
	HookSubagentStart     HookEvent = "SubagentStart"
	HookPermissionRequest HookEvent = "PermissionRequest"
)

// HooksConfig maps hook event names to hook matcher configs.
type HooksConfig map[HookEvent][]HookMatcherConfig

// HookMatcherConfig is one initialized SDK hook matcher.
type HookMatcherConfig struct {
	Matcher         string   `json:"matcher,omitempty"`
	HookCallbackIDs []string `json:"hookCallbackIds,omitempty"`
	Timeout         float64  `json:"timeout,omitempty"`
}

// HookCallbackInput carries a hook callback payload from Claude Code.
type HookCallbackInput struct {
	HookEventName         string                     `json:"hook_event_name,omitempty"`
	SessionID             string                     `json:"session_id,omitempty"`
	TranscriptPath        string                     `json:"transcript_path,omitempty"`
	Cwd                   string                     `json:"cwd,omitempty"`
	PermissionMode        string                     `json:"permission_mode,omitempty"`
	AgentID               string                     `json:"agent_id,omitempty"`
	AgentType             string                     `json:"agent_type,omitempty"`
	ToolName              string                     `json:"tool_name,omitempty"`
	ToolInput             map[string]json.RawMessage `json:"tool_input,omitempty"`
	ToolResponse          json.RawMessage            `json:"tool_response,omitempty"`
	ToolUseID             string                     `json:"tool_use_id,omitempty"`
	Error                 string                     `json:"error,omitempty"`
	IsInterrupt           bool                       `json:"is_interrupt,omitempty"`
	Prompt                string                     `json:"prompt,omitempty"`
	StopHookActive        bool                       `json:"stop_hook_active,omitempty"`
	AgentTranscriptPath   string                     `json:"agent_transcript_path,omitempty"`
	Trigger               string                     `json:"trigger,omitempty"`
	CustomInstructions    string                     `json:"custom_instructions,omitempty"`
	Message               string                     `json:"message,omitempty"`
	Title                 string                     `json:"title,omitempty"`
	NotificationType      string                     `json:"notification_type,omitempty"`
	PermissionSuggestions []PermissionUpdate         `json:"permission_suggestions,omitempty"`
}

// PermissionUpdate is a permission update suggestion.
type PermissionUpdate struct {
	Type        PermissionUpdateType        `json:"type"`
	Rules       []PermissionRuleValue       `json:"rules,omitempty"`
	Behavior    PermissionBehavior          `json:"behavior,omitempty"`
	Mode        string                      `json:"mode,omitempty"`
	Directories []string                    `json:"directories,omitempty"`
	Destination PermissionUpdateDestination `json:"destination,omitempty"`
}

// PermissionUpdateType is the permission update variant discriminator.
type PermissionUpdateType string

// PermissionUpdateType values.
const (
	PermissionUpdateAddRules          PermissionUpdateType = "addRules"
	PermissionUpdateReplaceRules      PermissionUpdateType = "replaceRules"
	PermissionUpdateRemoveRules       PermissionUpdateType = "removeRules"
	PermissionUpdateSetMode           PermissionUpdateType = "setMode"
	PermissionUpdateAddDirectories    PermissionUpdateType = "addDirectories"
	PermissionUpdateRemoveDirectories PermissionUpdateType = "removeDirectories"
)

// PermissionRuleValue is one permission rule value.
type PermissionRuleValue struct {
	ToolName    string  `json:"toolName"`
	RuleContent *string `json:"ruleContent,omitempty"`
}

// PermissionBehavior is the behavior assigned to permission rules.
type PermissionBehavior string

// PermissionBehavior values.
const (
	PermissionBehaviorAllow PermissionBehavior = "allow"
	PermissionBehaviorDeny  PermissionBehavior = "deny"
	PermissionBehaviorAsk   PermissionBehavior = "ask"
)

// PermissionUpdateDestination is where a permission update should apply.
type PermissionUpdateDestination string

// PermissionUpdateDestination values.
const (
	PermissionUpdateUserSettings    PermissionUpdateDestination = "userSettings"
	PermissionUpdateProjectSettings PermissionUpdateDestination = "projectSettings"
	PermissionUpdateLocalSettings   PermissionUpdateDestination = "localSettings"
	PermissionUpdateSession         PermissionUpdateDestination = "session"
)

// JSONRPCMessage is a JSON-RPC 2.0 MCP envelope.
type JSONRPCMessage struct {
	JSONRPC string          `json:"jsonrpc"`
	ID      json.RawMessage `json:"id,omitempty"`
	Method  string          `json:"method,omitempty"`
	Params  json.RawMessage `json:"params,omitempty"`
	Result  json.RawMessage `json:"result,omitempty"`
	Error   *JSONRPCError   `json:"error,omitempty"`
}

// JSONRPCError is a JSON-RPC 2.0 error object.
type JSONRPCError struct {
	Code    int             `json:"code"`
	Message string          `json:"message"`
	Data    json.RawMessage `json:"data,omitempty"`
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
	Response                  ControlResponsePayload `json:"response,omitzero"`                     // success only
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
	// SystemTaskUpdated reports a patch to background subagent state.
	SystemTaskUpdated SystemSubtype = "task_updated"
	// SystemBackgroundTasksChanged reports the full live background task set.
	SystemBackgroundTasksChanged SystemSubtype = "background_tasks_changed"
	// SystemThinkingTokens reports incremental estimated thinking token usage.
	SystemThinkingTokens SystemSubtype = "thinking_tokens"
	// SystemTurnDuration reports per-turn duration, budget, and pending-work counts.
	SystemTurnDuration SystemSubtype = "turn_duration"
	// SystemCompactBoundary marks where context was compacted.
	SystemCompactBoundary SystemSubtype = "compact_boundary"
	// SystemStatus reports idle/running/requires_action transitions.
	SystemStatus SystemSubtype = "status"
	// SystemCommandsChanged reports the full available slash command set.
	SystemCommandsChanged SystemSubtype = "commands_changed"
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
	// SystemModelRefusalFallback reports that Claude Code retried a refused
	// request with a fallback model.
	SystemModelRefusalFallback SystemSubtype = "model_refusal_fallback"
	// SystemModelRefusalNoFallback reports that Claude Code has no further
	// fallback after a model refusal.
	SystemModelRefusalNoFallback SystemSubtype = "model_refusal_no_fallback"
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

// OutputControlRequestMsg is a host control request emitted by Claude Code.
type OutputControlRequestMsg struct {
	Type      OutputType      `json:"type"` // OutputControlRequest
	RequestID string          `json:"request_id"`
	Request   json.RawMessage `json:"request"`
}

// DecodeCanUseTool decodes a can_use_tool control request.
func (m *OutputControlRequestMsg) DecodeCanUseTool() (ControlReqCanUseTool, error) {
	var out ControlReqCanUseTool
	if len(m.Request) == 0 {
		return out, errors.New("empty control request")
	}
	if err := json.Unmarshal(m.Request, &out); err != nil {
		return ControlReqCanUseTool{}, err
	}
	return out, nil
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
	Plugins        []InitPlugin    `json:"plugins,omitempty,omitzero"`
	Skills         []string        `json:"skills,omitempty"`
	SlashCommands  []string        `json:"slash_commands,omitempty"`
	Betas          []string        `json:"betas,omitempty"`
	PluginErrors   []InitPluginErr `json:"plugin_errors,omitempty,omitzero"`
	PluginWarnings []InitPluginErr `json:"plugin_warnings,omitempty,omitzero"`
	Capabilities   []string        `json:"capabilities,omitempty"`
	MemoryPaths    InitMemPaths    `json:"memory_paths,omitzero"`

	AnalyticsDisabled       bool `json:"analytics_disabled,omitempty"`
	ProductFeedbackDisabled bool `json:"product_feedback_disabled,omitempty"`
}

// InitMCPServer is an MCP server entry in the system/init message.
type InitMCPServer struct {
	Name   string `json:"name"`
	Status string `json:"status"`
}

// InitPlugin is a plugin entry in the system/init message.
type InitPlugin struct {
	Name    string `json:"name"`
	Path    string `json:"path"`
	Source  string `json:"source"`
	Version string `json:"version,omitempty"`
}

// InitPluginErr is a plugin load error in the system/init message.
type InitPluginErr struct {
	Plugin  string `json:"plugin"`
	Type    string `json:"type"`
	Message string `json:"message"`
}

// InitMemPaths holds absolute directory paths for memory stores.
type InitMemPaths struct {
	Auto string `json:"auto,omitempty"`
	Team string `json:"team,omitempty"`
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
	SubagentType string        `json:"subagent_type,omitempty"`
	TaskID       string        `json:"task_id,omitempty"`
	TaskType     string        `json:"task_type,omitempty"`
	ToolUseID    string        `json:"tool_use_id,omitempty"`
	LastToolName string        `json:"last_tool_name,omitempty"`
	Status       string        `json:"status,omitempty"`
	UsageExtra   TaskUsageWire `json:"usage,omitzero"`
	OutputFile   string        `json:"output_file,omitempty"`
	Summary      string        `json:"summary,omitempty"`
	Tasks        []TaskWire    `json:"tasks,omitempty"`

	// api_retry fields.
	Attempt     int             `json:"attempt,omitempty"`
	MaxRetries  int             `json:"max_retries,omitempty"`
	RetryDelay  base.DurationMS `json:"retry_delay_ms,omitempty"`
	ErrorStatus int             `json:"error_status,omitempty"`
	Error       json.RawMessage `json:"error,omitempty"`

	// Patch carries task completion state.
	Patch PatchWire `json:"patch,omitzero"`

	// model_refusal_fallback / model_refusal_no_fallback fields.
	Trigger                string `json:"trigger,omitempty"`
	Direction              string `json:"direction,omitempty"`
	OriginalModel          string `json:"original_model,omitempty"`
	FallbackModel          string `json:"fallback_model,omitempty"`
	RequestID              string `json:"request_id,omitempty"`
	APIRefusalCategory     string `json:"api_refusal_category,omitempty"`
	APIRefusalExplanation  string `json:"api_refusal_explanation,omitempty"`
	RefusedUserMessageUUID string `json:"refused_user_message_uuid,omitempty"`
	Content                string `json:"content,omitempty"`

	// Other optional fields.
	PermissionMode  string              `json:"permissionMode,omitempty"`
	Commands        []CommandWire       `json:"commands,omitempty"`
	CompactMetadata CompactMetadataWire `json:"compact_metadata,omitzero"`
	CompactResult   string              `json:"compact_result,omitempty"`
	Prompt          json.RawMessage     `json:"prompt,omitempty"`

	// thinking_tokens fields.
	EstimatedTokens      int64 `json:"estimated_tokens,omitzero"`
	EstimatedTokensDelta int64 `json:"estimated_tokens_delta,omitzero"`
}

// PatchWire is the patch object on task system messages.
type PatchWire struct {
	Status         TaskPatchStatus `json:"status"`   // "completed" or "killed".
	EndTime        base.TimeMS     `json:"end_time"` // Unix epoch milliseconds.
	IsBackgrounded bool            `json:"is_backgrounded,omitempty"`
}

// TaskPatchStatus is the status field in task_updated patch objects.
type TaskPatchStatus string

// TaskPatchStatus values.
const (
	TaskPatchStatusCompleted TaskPatchStatus = "completed"
	TaskPatchStatusKilled    TaskPatchStatus = "killed"
)

// CompactMetadataWire is the compact_metadata object on compact_boundary system messages.
type CompactMetadataWire struct {
	Trigger   string `json:"trigger"`    // "auto" or "manual".
	PreTokens int    `json:"pre_tokens"` // Token count before compaction.
}

// TaskUsageWire is the usage object on task_progress/task_notification system messages.
type TaskUsageWire struct {
	TotalTokens int             `json:"total_tokens"`
	ToolUses    int             `json:"tool_uses"`
	Duration    base.DurationMS `json:"duration_ms"`
}

// TaskWire is one entry in a background_tasks_changed system message.
type TaskWire struct {
	TaskID      string `json:"task_id"`
	TaskType    string `json:"task_type"`
	Description string `json:"description"`
}

// CommandWire is one entry in a commands_changed system message.
type CommandWire struct {
	Name         string   `json:"name"`
	Description  string   `json:"description"`
	ArgumentHint string   `json:"argumentHint"`
	Aliases      []string `json:"aliases,omitempty"`
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
	RequestID       string               `json:"request_id,omitempty"`
	SubagentType    string               `json:"subagent_type,omitempty"`
	TaskDescription string               `json:"task_description,omitempty"`
}

// AssistantMessageBody is the inner message object within an assistant record.
type AssistantMessageBody struct {
	ID           string                       `json:"id"`
	Type         string                       `json:"type,omitempty"`
	Role         string                       `json:"role"`
	Model        string                       `json:"model"`
	Content      []OutputContentBlock         `json:"content"`
	Usage        MsgUsage                     `json:"usage"`
	StopReason   string                       `json:"stop_reason"`
	StopSequence string                       `json:"stop_sequence"`
	StopDetails  anthropic.RefusalStopDetails `json:"stop_details,omitzero"`

	Container         Container         `json:"container,omitzero"`
	ContextManagement ContextManagement `json:"context_management,omitzero"`
	Diagnostics       Diagnostics       `json:"diagnostics,omitzero"`
}

// ContentBlockStart is the content_block field in a content_block_start streaming event.
type ContentBlockStart struct {
	Type      string                     `json:"type"`
	Text      string                     `json:"text,omitempty"`
	ID        string                     `json:"id,omitempty"`
	Name      string                     `json:"name,omitempty"`
	Input     map[string]json.RawMessage `json:"input,omitempty"`
	Thinking  string                     `json:"thinking,omitempty"`
	Signature []byte                     `json:"signature,omitempty"`
	Caller    anthropic.Caller           `json:"caller,omitzero"`
	From      FallbackModelRef           `json:"from,omitzero"`
	To        FallbackModelRef           `json:"to,omitzero"`
}

// OutputContentBlock is a single content block inside an assistant message.
// This is a flat union: fields are populated depending on Type.
//
//   - "text":                Text
//   - "thinking":            Thinking, Signature
//   - "tool_use":            ID, Name, Input
//   - "tool_result":         ToolUseID, Content, IsError
//   - "fallback":            From, To
//   - "server_tool_use":     ID, Name, Input, Caller
//   - "web_search_tool_use": ID, Name, Input, Caller
type OutputContentBlock struct {
	Type      string                     `json:"type"`
	Text      string                     `json:"text,omitempty"`
	ID        string                     `json:"id,omitempty"`
	Name      string                     `json:"name,omitempty"`
	Input     map[string]json.RawMessage `json:"input,omitempty"`
	Thinking  string                     `json:"thinking,omitempty"`
	Signature []byte                     `json:"signature,omitempty"`
	Caller    anthropic.Caller           `json:"caller,omitzero"`
	From      FallbackModelRef           `json:"from,omitzero"`
	To        FallbackModelRef           `json:"to,omitzero"`
	// tool_result fields (inline MCP tool results).
	ToolUseID string            `json:"tool_use_id,omitempty"`
	Content   ToolResultPayload `json:"content,omitempty"`
	IsError   bool              `json:"is_error,omitempty"`
}

// FallbackModelRef identifies a model in a fallback content block.
type FallbackModelRef struct {
	Model string `json:"model"`
}

// IsZero reports whether f carries no model reference.
func (f FallbackModelRef) IsZero() bool {
	return f.Model == ""
}

// BashInput is the input for the Bash tool.
type BashInput struct {
	Command     string          `json:"command"`
	Description string          `json:"description,omitempty"`
	Timeout     base.DurationMS `json:"timeout_ms,omitempty"`
}

// ReadInput is the input for the Read tool.
type ReadInput struct {
	FilePath string `json:"file_path"`
	Offset   int    `json:"offset,omitempty"`
	Limit    int    `json:"limit,omitempty"`
}

// EditInput is the input for the Edit tool.
type EditInput struct {
	FilePath   string `json:"file_path"`
	OldString  string `json:"old_string"`
	NewString  string `json:"new_string"`
	ReplaceAll bool   `json:"replace_all,omitempty"`
}

// MultiEditInput is the input for the MultiEdit tool.
type MultiEditInput struct {
	FilePath string            `json:"file_path"`
	Edits    []EditReplacement `json:"edits"`
}

// EditReplacement is one old/new text replacement in a MultiEdit tool call.
type EditReplacement struct {
	OldString string `json:"old_string"`
	NewString string `json:"new_string"`
}

// WriteInput is the input for the Write tool.
type WriteInput struct {
	FilePath string `json:"file_path"`
	Content  string `json:"content"`
}

// GrepInput is the input for the Grep tool.
type GrepInput struct {
	Pattern string `json:"pattern"`
	Path    string `json:"path,omitempty"`
	Glob    string `json:"glob,omitempty"`
	Output  string `json:"output_mode,omitempty"`
	Head    int    `json:"head_limit,omitempty"`
}

// GlobInput is the input for the Glob tool.
type GlobInput struct {
	Pattern string `json:"pattern"`
	Path    string `json:"path,omitempty"`
}

// AskUserQuestionInput is the input for the AskUserQuestion tool.
type AskUserQuestionInput struct {
	Questions []AskUserQuestion `json:"questions"`
}

// AskUserQuestionUpdatedInput is the updated AskUserQuestion input returned
// in an approved control response.
type AskUserQuestionUpdatedInput struct {
	Questions []AskUserQuestion `json:"questions"`
	Answers   map[string]string `json:"answers"`
}

// AskUserQuestion is a single question in an AskUserQuestion tool call.
type AskUserQuestion struct {
	Question    string                  `json:"question"`
	Header      string                  `json:"header,omitempty"`
	Options     []AskUserQuestionOption `json:"options"`
	MultiSelect bool                    `json:"multiSelect,omitempty"`
}

// AskUserQuestionOption is a selectable option in an AskUserQuestion.
type AskUserQuestionOption struct {
	Label       string `json:"label"`
	Description string `json:"description,omitempty"`
}

// TodoWriteInput is the input for the TodoWrite tool.
type TodoWriteInput struct {
	Todos []TodoWriteItem `json:"todos"`
}

// TodoWriteItem is a single todo entry in a TodoWrite tool call.
type TodoWriteItem struct {
	Content    string `json:"content"`
	Status     string `json:"status"`
	ActiveForm string `json:"activeForm,omitempty"`
}

// Container carries code-execution container metadata on a message.
type Container struct {
	ID        string    `json:"id"`
	ExpiresAt time.Time `json:"expires_at"`
	Skills    []Skill   `json:"skills,omitempty"`
}

// IsZero reports whether c carries no container metadata.
func (c Container) IsZero() bool {
	return c.ID == "" && c.ExpiresAt.IsZero() && len(c.Skills) == 0
}

// Skill is a skill loaded in a code-execution container.
type Skill struct {
	SkillID string    `json:"skill_id"`
	Type    SkillType `json:"type"`
	Version string    `json:"version"`
}

// SkillType identifies a loaded skill kind.
type SkillType string

// SkillType values.
const (
	SkillAnthropic SkillType = "anthropic"
	SkillCustom    SkillType = "custom"
)

// Diagnostics carries request-level prompt-cache diagnostic metadata.
type Diagnostics struct {
	CacheMissReason CacheMissReason `json:"cache_miss_reason,omitzero"`
}

// IsZero reports whether d carries no diagnostics metadata.
func (d Diagnostics) IsZero() bool {
	return d.CacheMissReason.IsZero()
}

// CacheMissReason explains why a prompt-cache prefix could not be reused.
type CacheMissReason struct {
	Type                   CacheMissReasonType `json:"type"`
	CacheMissedInputTokens int64               `json:"cache_missed_input_tokens,omitempty"`
}

// IsZero reports whether c carries no cache miss reason.
func (c CacheMissReason) IsZero() bool {
	return c.Type == "" && c.CacheMissedInputTokens == 0
}

// CacheMissReasonType identifies why a prompt-cache prefix was missed.
type CacheMissReasonType string

// CacheMissReasonType values.
const (
	CacheMissReasonModelChanged            CacheMissReasonType = "model_changed"
	CacheMissReasonSystemChanged           CacheMissReasonType = "system_changed"
	CacheMissReasonToolsChanged            CacheMissReasonType = "tools_changed"
	CacheMissReasonMessagesChanged         CacheMissReasonType = "messages_changed"
	CacheMissReasonPreviousMessageNotFound CacheMissReasonType = "previous_message_not_found"
	CacheMissReasonUnavailable             CacheMissReasonType = "unavailable"
)

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
	SubagentType    string          `json:"subagent_type,omitempty"`
	TaskDescription string          `json:"task_description,omitempty"`
	IsSynthetic     bool            `json:"isSynthetic,omitempty"`
	IsReplay        bool            `json:"isReplay,omitempty"`
}

// DecodeMessage decodes the raw user message body into its concrete wire shape.
func (m *OutputUserMsg) DecodeMessage() (OutputUserMessage, error) {
	var out OutputUserMessage
	if len(m.Message) == 0 {
		return out, nil
	}
	if err := json.Unmarshal(m.Message, &out); err != nil {
		return OutputUserMessage{}, err
	}
	return out, nil
}

// DecodeToolUseResult decodes the top-level tool_use_result summary.
func (m *OutputUserMsg) DecodeToolUseResult() (OutputToolUseResult, error) {
	var out OutputToolUseResult
	if len(m.ToolUseResult) == 0 {
		return out, nil
	}
	if err := json.Unmarshal(m.ToolUseResult, &out); err != nil {
		return OutputToolUseResult{}, err
	}
	return out, nil
}

// ---------- result ----------

// ResultSubtype is the terminal result discriminator.
type ResultSubtype string

// Result subtypes.
const (
	ResultSuccess                         ResultSubtype = "success"
	ResultErrorDuringExecution            ResultSubtype = "error_during_execution"
	ResultErrorMaxTurns                   ResultSubtype = "error_max_turns"
	ResultErrorMaxBudgetUSD               ResultSubtype = "error_max_budget_usd"
	ResultErrorMaxStructuredOutputRetries ResultSubtype = "error_max_structured_output_retries"
)

// OutputResultMsg is the wire representation of a result record.
type OutputResultMsg struct {
	Type                   OutputType      `json:"type"`
	Subtype                ResultSubtype   `json:"subtype"`
	IsError                bool            `json:"is_error"`
	Duration               base.DurationMS `json:"duration_ms"`
	DurationAPI            base.DurationMS `json:"duration_api_ms"`
	Ttft                   base.DurationMS `json:"ttft_ms,omitempty"`
	TtftStream             base.DurationMS `json:"ttft_stream_ms,omitempty"`
	TimeToRequest          base.DurationMS `json:"time_to_request_ms,omitempty"`
	TimeToRequestFromSpawn base.DurationMS `json:"time_to_request_from_spawn_ms,omitempty"`
	WarmSpareClaimed       bool            `json:"warm_spare_claimed,omitempty"`
	TimeOrigin             base.TimeMS     `json:"time_origin_ms,omitempty"`
	NumTurns               int             `json:"num_turns"`
	Result                 string          `json:"result"`
	Errors                 []string        `json:"errors,omitempty"`
	SessionID              string          `json:"session_id"`
	TotalCostUSD           float64         `json:"total_cost_usd"`
	Usage                  MsgUsage        `json:"usage"`
	UUID                   string          `json:"uuid"`
	StructuredOutput       json.RawMessage `json:"structured_output,omitempty"`
	Timestamp              string          `json:"timestamp,omitempty"`

	FastModeState     string                     `json:"fast_mode_state,omitempty"`
	ModelUsage        map[string]ModelUsageEntry `json:"modelUsage,omitempty"`
	PermissionDenials []json.RawMessage          `json:"permission_denials,omitempty"`
	StopReason        string                     `json:"stop_reason,omitempty"`
	TerminalReason    string                     `json:"terminal_reason,omitempty"`
	APIErrorStatus    int                        `json:"api_error_status,omitzero"`
	DeferredToolUse   DeferredToolUse            `json:"deferred_tool_use,omitzero"`
	Origin            ResultOrigin               `json:"origin,omitzero"`
}

// AsError returns the Claude Code error represented by m, if any.
func (m *OutputResultMsg) AsError() error {
	if !m.IsError {
		return nil
	}
	errMsg := m.Result
	if errMsg == "" && len(m.Errors) > 0 {
		errMsg = strings.Join(m.Errors, "; ")
	}
	return errors.New("claude error (" + string(m.Subtype) + "): " + errMsg)
}

// ResultOriginKind identifies why a result was emitted.
type ResultOriginKind string

// Result origin kinds.
const (
	ResultOriginTaskNotification ResultOriginKind = "task-notification"
)

// ResultOrigin identifies why a result was emitted.
type ResultOrigin struct {
	Kind ResultOriginKind `json:"kind"`
}

// IsZero reports whether r carries no origin metadata.
func (r ResultOrigin) IsZero() bool {
	return r.Kind == ""
}

// ModelUsageEntry holds per-model usage metadata in a result message.
type ModelUsageEntry struct {
	InputTokens              int64   `json:"inputTokens"`
	OutputTokens             int64   `json:"outputTokens"`
	CacheReadInputTokens     int64   `json:"cacheReadInputTokens"`
	CacheCreationInputTokens int64   `json:"cacheCreationInputTokens"`
	WebSearchRequests        int64   `json:"webSearchRequests"`
	CostUSD                  float64 `json:"costUSD"`
	ContextWindow            int64   `json:"contextWindow"`
	MaxOutputTokens          int64   `json:"maxOutputTokens"`
}

// DeferredToolUse is a tool call deferred from a previous turn in the result message.
type DeferredToolUse struct {
	ID    string                     `json:"id"`
	Name  string                     `json:"name"`
	Input map[string]json.RawMessage `json:"input"`
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

	Iterations []IterationUsage `json:"iterations,omitempty"`

	ServerToolUse ServerToolUse `json:"server_tool_use,omitzero"`
	CacheCreation CacheCreation `json:"cache_creation,omitzero"`

	OutputTokensDetails OutputTokensDetails `json:"output_tokens_details,omitzero"`
}

// IsZero reports whether m carries no usage data.
func (m *MsgUsage) IsZero() bool {
	if m == nil {
		return true
	}
	return m.InputTokens == 0 &&
		m.OutputTokens == 0 &&
		m.CacheCreationInputTokens == 0 &&
		m.CacheReadInputTokens == 0 &&
		m.ServiceTier == "" &&
		m.InferenceGeo == "" &&
		m.Speed == "" &&
		len(m.Iterations) == 0 &&
		m.ServerToolUse.IsZero() &&
		m.CacheCreation.IsZero() &&
		m.OutputTokensDetails.IsZero()
}

// OutputTokensDetails breaks down output token usage.
type OutputTokensDetails struct {
	ThinkingTokens int64 `json:"thinking_tokens"`
}

// IsZero reports whether o carries no output token details.
func (o OutputTokensDetails) IsZero() bool {
	return o.ThinkingTokens == 0
}

// ServerToolUse tracks server-side tool use counts.
type ServerToolUse struct {
	WebSearchRequests int `json:"web_search_requests"`
	WebFetchRequests  int `json:"web_fetch_requests"`
}

// IsZero reports whether s carries no server-side tool use counts.
func (s ServerToolUse) IsZero() bool {
	return s.WebSearchRequests == 0 && s.WebFetchRequests == 0
}

// CacheCreation breaks down cache creation by time bucket.
type CacheCreation struct {
	Ephemeral1hInputTokens int `json:"ephemeral_1h_input_tokens"`
	Ephemeral5mInputTokens int `json:"ephemeral_5m_input_tokens"`
}

// IsZero reports whether c carries no cache creation token counts.
func (c CacheCreation) IsZero() bool {
	return c.Ephemeral1hInputTokens == 0 && c.Ephemeral5mInputTokens == 0
}

// IterationUsage holds per-iteration token usage.
type IterationUsage struct {
	Type                     IterationUsageType `json:"type"`
	Model                    string             `json:"model,omitempty"`
	InputTokens              int64              `json:"input_tokens"`
	OutputTokens             int64              `json:"output_tokens"`
	CacheReadInputTokens     int64              `json:"cache_read_input_tokens"`
	CacheCreationInputTokens int64              `json:"cache_creation_input_tokens"`
	CacheCreation            CacheCreation      `json:"cache_creation,omitzero"`
}

// IterationUsageType identifies the kind of usage iteration.
type IterationUsageType string

// IterationUsageType values.
const (
	IterationUsageMessage        IterationUsageType = "message"
	IterationUsageCompaction     IterationUsageType = "compaction"
	IterationUsageAdvisorMessage IterationUsageType = "advisor_message"
)

// ---------- stream_event ----------

// OutputStreamEventMsg is the wire representation of a stream_event record.
type OutputStreamEventMsg struct {
	Type            OutputType      `json:"type"`
	UUID            string          `json:"uuid"`
	SessionID       string          `json:"session_id"`
	Timestamp       string          `json:"timestamp,omitempty"`
	ParentToolUseID string          `json:"parent_tool_use_id"`
	Event           StreamEventData `json:"event"`
	Ttft            base.DurationMS `json:"ttft_ms,omitempty"`
}

// StreamEventData is the nested event body inside a stream_event record.
type StreamEventData struct {
	Type         string            `json:"type"`
	Index        int               `json:"index"`
	Delta        StreamDelta       `json:"delta,omitzero"`
	ContentBlock ContentBlockStart `json:"content_block,omitzero"`
	// message_start carries the full message object; message_delta carries
	// stop_reason and usage in a delta wrapper.
	Message           AssistantMessageBody `json:"message,omitzero"`
	Usage             MsgUsage             `json:"usage,omitzero"`
	ContextManagement ContextManagement    `json:"context_management,omitzero"`
}

// ContextManagement carries Claude Code context edit metadata.
type ContextManagement struct {
	AppliedEdits []AppliedEdit `json:"applied_edits,omitempty"`
}

// IsZero reports whether c carries no context management metadata.
func (c ContextManagement) IsZero() bool {
	return len(c.AppliedEdits) == 0
}

// AppliedEditType is the context management edit discriminator.
type AppliedEditType string

// AppliedEditType values returned in context management metadata.
const (
	AppliedEditClearToolUses AppliedEditType = "clear_tool_uses_20250919"
	AppliedEditClearThinking AppliedEditType = "clear_thinking_20251015"
)

// AppliedEdit is one context management edit applied during a request.
//
// The Type discriminator determines which variant-specific field is populated:
// ClearedToolUses for clear_tool_uses_20250919 or ClearedThinkingTurns for
// clear_thinking_20251015.
type AppliedEdit struct {
	Type                 AppliedEditType `json:"type"`
	ClearedInputTokens   int64           `json:"cleared_input_tokens"`
	ClearedToolUses      int64           `json:"cleared_tool_uses,omitempty"`
	ClearedThinkingTurns int64           `json:"cleared_thinking_turns,omitempty"`
}

// StreamDelta is a delta object inside a stream event.
type StreamDelta struct {
	Type                 string `json:"type"`
	Text                 string `json:"text"`
	PartialJSON          string `json:"partial_json"`
	Thinking             string `json:"thinking"`
	Signature            []byte `json:"signature"`
	EstimatedTokens      int64  `json:"estimated_tokens,omitzero"`
	EstimatedTokensDelta int64  `json:"estimated_tokens_delta,omitzero"`
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

// RateLimitStatus is the current availability of a rate-limited resource.
type RateLimitStatus string

// Rate limit statuses.
const (
	RateLimitAllowed        RateLimitStatus = "allowed"
	RateLimitAllowedWarning RateLimitStatus = "allowed_warning"
	RateLimitRejected       RateLimitStatus = "rejected"
)

// RateLimitType identifies the quota window that produced a rate-limit event.
type RateLimitType string

// Rate limit types.
const (
	RateLimitFiveHour        RateLimitType = "five_hour"
	RateLimitSevenDay        RateLimitType = "seven_day"
	RateLimitSevenDayOpus    RateLimitType = "seven_day_opus"
	RateLimitSevenDaySonnet  RateLimitType = "seven_day_sonnet"
	RateLimitSevenDayOverage RateLimitType = "seven_day_overage_included"
	RateLimitOverage         RateLimitType = "overage"
)

// OverageDisabledReason explains why overage spending is unavailable.
type OverageDisabledReason string

// Overage disabled reasons.
const (
	OverageDisabledNotProvisioned    OverageDisabledReason = "overage_not_provisioned"
	OverageDisabledOrg               OverageDisabledReason = "org_level_disabled"
	OverageDisabledOrgUntil          OverageDisabledReason = "org_level_disabled_until"
	OverageDisabledOutOfCredits      OverageDisabledReason = "out_of_credits"
	OverageDisabledSeatTier          OverageDisabledReason = "seat_tier_level_disabled"
	OverageDisabledMember            OverageDisabledReason = "member_level_disabled"
	OverageDisabledSeatTierZeroLimit OverageDisabledReason = "seat_tier_zero_credit_limit"
	OverageDisabledGroupZeroLimit    OverageDisabledReason = "group_zero_credit_limit"
	OverageDisabledMemberZeroLimit   OverageDisabledReason = "member_zero_credit_limit"
	OverageDisabledOrgService        OverageDisabledReason = "org_service_level_disabled"
	OverageDisabledNoLimits          OverageDisabledReason = "no_limits_configured"
	OverageDisabledFetchError        OverageDisabledReason = "fetch_error"
	OverageDisabledUnknown           OverageDisabledReason = "unknown"
)

// RateLimitErrorCode identifies an actionable rate-limit error.
type RateLimitErrorCode string

// Rate limit error codes.
const (
	RateLimitErrorCreditsRequired RateLimitErrorCode = "credits_required"
)

// RateLimitPeriod reports utilization for one overage spending period.
type RateLimitPeriod struct {
	Utilization float64 `json:"utilization"`
}

// RateLimitInfo is the nested rate limit info inside a rate_limit_event.
// Wire format uses camelCase (matches Claude Code CLI JSON output).
type RateLimitInfo struct {
	Status                          RateLimitStatus       `json:"status"`
	ResetsAt                        float64               `json:"resetsAt,omitempty"`
	RateLimitType                   RateLimitType         `json:"rateLimitType,omitempty"`
	Utilization                     float64               `json:"utilization,omitempty"`
	OverageStatus                   RateLimitStatus       `json:"overageStatus,omitempty"`
	OverageResetsAt                 float64               `json:"overageResetsAt,omitempty"`
	OverageDisabledReason           OverageDisabledReason `json:"overageDisabledReason,omitempty"`
	IsUsingOverage                  bool                  `json:"isUsingOverage,omitempty"`
	OverageInUse                    bool                  `json:"overageInUse,omitempty"`
	SurpassedThreshold              float64               `json:"surpassedThreshold,omitempty"`
	OveragePeriodMonthly            RateLimitPeriod       `json:"overagePeriodMonthly,omitzero"`
	OveragePeriodChannel            RateLimitPeriod       `json:"overagePeriodChannel,omitzero"`
	ErrorCode                       RateLimitErrorCode    `json:"errorCode,omitempty"`
	CanUserPurchaseCredits          bool                  `json:"canUserPurchaseCredits,omitempty"`
	HasChargeableSavedPaymentMethod bool                  `json:"hasChargeableSavedPaymentMethod,omitempty"`
}

// ---------- tool_progress ----------

// OutputToolProgressMsg is emitted periodically while a tool is running.
type OutputToolProgressMsg struct {
	Type            OutputType     `json:"type"` // OutputToolProgress
	ToolUseID       string         `json:"tool_use_id"`
	ToolName        string         `json:"tool_name"`
	ParentToolUseID string         `json:"parent_tool_use_id"` // nullable
	ElapsedTime     base.DurationS `json:"elapsed_time_seconds"`
	TaskID          string         `json:"task_id,omitempty"`
	UUID            string         `json:"uuid"`
	SessionID       string         `json:"session_id"`
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

// Reasoning returns the concise human-readable progress summary.
func (m *OutputPostTurnSummaryMsg) Reasoning() string {
	var parts []string
	for _, s := range []string{m.StatusDetail, m.NeedsAction, m.RecentAction, m.Description, m.Title} {
		s = strings.TrimSpace(s)
		if s != "" && !slices.Contains(parts, s) {
			parts = append(parts, s)
		}
	}
	return strings.Join(parts, "\n")
}

// OutputCommandsChangedMsg reports the full available slash command set.
type OutputCommandsChangedMsg struct {
	Type      OutputType    `json:"type"`    // OutputSystem
	Subtype   SystemSubtype `json:"subtype"` // SystemCommandsChanged
	Commands  []CommandWire `json:"commands"`
	UUID      string        `json:"uuid"`
	SessionID string        `json:"session_id"`
}

// OutputTurnDurationMsg reports per-turn duration, budget, and pending work.
type OutputTurnDurationMsg struct {
	Type                        OutputType      `json:"type"`    // OutputSystem
	Subtype                     SystemSubtype   `json:"subtype"` // SystemTurnDuration
	Duration                    base.DurationMS `json:"duration_ms"`
	BudgetTokens                int             `json:"budget_tokens,omitempty"`
	BudgetLimit                 int             `json:"budget_limit,omitempty"`
	BudgetNudges                int             `json:"budget_nudges,omitempty"`
	MessageCount                int             `json:"message_count,omitempty"`
	PendingBackgroundAgentCount int             `json:"pending_background_agent_count,omitempty"`
	PendingWorkflowCount        int             `json:"pending_workflow_count,omitempty"`
	UUID                        string          `json:"uuid"`
	SessionID                   string          `json:"session_id"`
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

// OutputUserMessageKind identifies the concrete shape of an OutputUserMessage.
type OutputUserMessageKind string

// OutputUserMessageKind values.
const (
	// OutputUserMessageText is a user message with content encoded as a string.
	OutputUserMessageText OutputUserMessageKind = "text"
	// OutputUserMessageBlock is a user message with content encoded as blocks.
	OutputUserMessageBlock OutputUserMessageKind = "block"
	// OutputUserMessageToolResult is a top-level standard tool result body.
	OutputUserMessageToolResult OutputUserMessageKind = "tool_result"
)

// OutputUserMessage is a typed union for OutputUserMsg.Message.
type OutputUserMessage struct {
	Kind       OutputUserMessageKind
	Role       string
	Text       string
	Content    []OutputUserContentBlock
	ToolResult OutputToolResult
}

// UnmarshalJSON implements json.Unmarshaler.
func (m *OutputUserMessage) UnmarshalJSON(data []byte) error {
	data = bytes.TrimSpace(data)
	*m = OutputUserMessage{}
	if len(data) == 0 || string(data) == "null" {
		return nil
	}
	var p struct {
		Role    string          `json:"role"`
		Content json.RawMessage `json:"content"`
	}
	if err := json.Unmarshal(data, &p); err != nil {
		return err
	}
	if len(p.Content) == 0 {
		return errors.New("user message content is required")
	}
	if p.Role == "user" {
		m.Role = p.Role
		content := bytes.TrimSpace(p.Content)
		if len(content) == 0 || string(content) == "null" {
			return errors.New("user message content is required")
		}
		if content[0] == '"' {
			var text string
			if err := json.Unmarshal(content, &text); err != nil {
				return err
			}
			m.Kind = OutputUserMessageText
			m.Text = text
			return nil
		}
		var blocks []OutputUserContentBlock
		if err := json.Unmarshal(content, &blocks); err != nil {
			return err
		}
		m.Kind = OutputUserMessageBlock
		m.Content = blocks
		return nil
	}
	var res OutputToolResult
	if err := json.Unmarshal(data, &res); err != nil {
		return err
	}
	m.Kind = OutputUserMessageToolResult
	m.ToolResult = res
	return nil
}

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
	Type      string           `json:"type"`
	Text      string           `json:"text,omitempty"`
	Source    anthropic.Source `json:"source,omitzero"`
	ToolUseID string           `json:"tool_use_id,omitempty"`
	// Nested content and error flag for inline tool_result blocks (MCP tools).
	Content ToolResultPayload `json:"content,omitempty"`
	IsError bool              `json:"is_error,omitempty"`
}

// OutputToolResult is the message body format for tool results delivered via
// the top-level parent_tool_use_id path (standard Claude Code tools).
type OutputToolResult struct {
	Content ToolResultPayload `json:"content"`
	IsError bool              `json:"is_error"`
}

// OutputToolUseResult is the optional top-level tool_use_result summary on
// echoed user messages. Claude Code currently emits either a string such as
// "Error: ..." or an object.
type OutputToolUseResult struct {
	Text   string
	Object map[string]json.RawMessage
}

// UnmarshalJSON implements json.Unmarshaler.
func (t *OutputToolUseResult) UnmarshalJSON(data []byte) error {
	data = bytes.TrimSpace(data)
	*t = OutputToolUseResult{}
	if len(data) == 0 || string(data) == "null" {
		return nil
	}
	switch data[0] {
	case '"':
		return json.Unmarshal(data, &t.Text)
	case '{':
		return json.Unmarshal(data, &t.Object)
	default:
		return errors.New("tool_use_result must be a string or object")
	}
}

// ToolResultPayload is a tool result content payload.
//
// Claude Code emits content as either a plain string or an array of content
// blocks, depending on the tool path.
type ToolResultPayload struct {
	Text   string
	Blocks []anthropic.Content
}

// UnmarshalJSON implements json.Unmarshaler.
func (t *ToolResultPayload) UnmarshalJSON(data []byte) error {
	data = bytes.TrimSpace(data)
	if len(data) == 0 || string(data) == "null" {
		t.Text = ""
		t.Blocks = nil
		return nil
	}
	if data[0] == '"' {
		var s string
		if err := json.Unmarshal(data, &s); err != nil {
			return err
		}
		t.Text = s
		t.Blocks = nil
		return nil
	}
	var blocks []anthropic.Content
	if err := json.Unmarshal(data, &blocks); err != nil {
		return err
	}
	t.Text = ""
	t.Blocks = blocks
	return nil
}

// MarshalJSON implements json.Marshaler.
func (t ToolResultPayload) MarshalJSON() ([]byte, error) {
	if t.Text != "" || t.Blocks == nil {
		return json.Marshal(t.Text)
	}
	return json.Marshal(t.Blocks)
}

// IsZero reports whether t carries no tool result content.
func (t ToolResultPayload) IsZero() bool {
	return t.Text == "" && len(t.Blocks) == 0
}

// TextBlocks returns content as text blocks.
func (t ToolResultPayload) TextBlocks() []anthropic.Content {
	if t.Text != "" {
		return []anthropic.Content{{Type: "text", Text: t.Text}}
	}
	return t.Blocks
}
