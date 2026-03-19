// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package claudecode

import "encoding/json"

// NDJSON message types for the Claude Code CLI stream-json protocol.
//
// Each line on stdout is one of these types, discriminated by the "type" field.
// Each line on stdin is an inputMsg.
//
// References:
//   - https://platform.claude.com/docs/en/agent-sdk/streaming-output
//   - https://github.com/anthropics/claude-agent-sdk-python (src/claude_agent_sdk/types.py)

// ---------- Stdin types ----------

// inputMsg is written to stdin to send a user message.
type inputMsg struct {
	Type    string       `json:"type"`    // always "user"
	Message inputContent `json:"message"` // the user content
}

// inputContent holds the user message payload.
// Content is a plain string for text-only messages, or []inputBlock for
// multi-modal messages that include images.
type inputContent struct {
	Role    string `json:"role"`    // always "user"
	Content any    `json:"content"` // string or []inputBlock
}

// inputBlock is a single content block inside a multi-modal user message.
type inputBlock struct {
	Type   string       `json:"type"`             // "text" or "image"
	Text   string       `json:"text,omitempty"`   // Type == "text"
	Source *inputSource `json:"source,omitempty"` // Type == "image"
}

// inputSource describes the data source for an image content block.
type inputSource struct {
	Type      string `json:"type"`                 // "base64" or "url"
	MediaType string `json:"media_type,omitempty"` // e.g. "image/png"
	Data      string `json:"data,omitempty"`       // base64-encoded bytes
	URL       string `json:"url,omitempty"`        // Type == "url"
}

// ---------- Stdout envelope ----------

// baseMsg is used to discriminate the type of each stdout line.
type baseMsg struct {
	Type    string `json:"type"`
	Subtype string `json:"subtype,omitempty"`
}

// ---------- system/init ----------

// systemInitMsg is the first message emitted on stdout, type="system" subtype="init".
type systemInitMsg struct {
	Type      string          `json:"type"`
	Subtype   string          `json:"subtype"`
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

// systemMsg is emitted for non-init system records, type="system" with various subtypes.
//
// Known subtypes include: task_started, task_progress, task_notification,
// api_retry, compact_boundary, permission_mode_change.
// Fields are populated depending on subtype.
type systemMsg struct {
	Type      string          `json:"type"`
	Subtype   string          `json:"subtype"`
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

// assistantMsg carries a complete assistant response, type="assistant".
type assistantMsg struct {
	Type            string          `json:"type"`
	SessionID       string          `json:"session_id,omitzero"`
	UUID            string          `json:"uuid,omitzero"`
	Timestamp       json.RawMessage `json:"timestamp,omitzero"`
	Message         apiMessage      `json:"message"`
	ParentToolUseID string          `json:"parent_tool_use_id,omitzero"`
	Error           string          `json:"error,omitzero"`
}

// apiMessage is the inner message from the model.
type apiMessage struct {
	ID         string          `json:"id,omitzero"`
	Type       string          `json:"type,omitzero"` // "message"
	Role       string          `json:"role,omitzero"`
	Model      string          `json:"model,omitzero"`
	Content    []contentBlock  `json:"content"`
	StopReason string          `json:"stop_reason,omitzero"`
	StopSeq    *string         `json:"stop_sequence,omitzero"`
	Usage      msgUsage        `json:"usage"`
	Container  json.RawMessage `json:"container,omitzero"`
	CtxMgmt    json.RawMessage `json:"context_management,omitzero"`
}

// contentBlock is a single block within an apiMessage.
// Type is one of "text", "thinking", "tool_use", "tool_result".
//
//   - "text":        Text
//   - "thinking":    Thinking, Signature
//   - "tool_use":    ID, Name, Input
//   - "tool_result": ToolUseID, Content, IsError
type contentBlock struct {
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
	Content   json.RawMessage `json:"content,omitempty"` // []contentBlock
	IsError   bool            `json:"is_error,omitempty"`
}

// ---------- user ----------

// userMsg is emitted for user records, type="user".
// Not sent by the genai provider, but appears on stdout during multi-turn.
type userMsg struct {
	Type            string          `json:"type"`
	UUID            string          `json:"uuid,omitzero"`
	SessionID       string          `json:"session_id,omitzero"`
	Timestamp       json.RawMessage `json:"timestamp,omitzero"`
	Message         json.RawMessage `json:"message"` // userTextMessage or userBlockMessage
	ParentToolUseID *string         `json:"parent_tool_use_id,omitzero"`
	ToolUseResult   json.RawMessage `json:"tool_use_result,omitzero"`
	IsSynthetic     bool            `json:"isSynthetic,omitzero"`
}

// userTextMessage is the user message body when content is a plain string.
type userTextMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// userBlockMessage is the user message body when content is an array of blocks.
type userBlockMessage struct {
	Role    string             `json:"role"`
	Content []userContentBlock `json:"content"`
}

// userContentBlock is a single content block in a user message.
type userContentBlock struct {
	Type      string           `json:"type"`
	Text      string           `json:"text,omitempty"`
	Source    *imageSourceWire `json:"source,omitempty"`
	ToolUseID string           `json:"tool_use_id,omitempty"`
	// Nested content and error flag for inline tool_result blocks (MCP tools).
	Content []toolResultContent `json:"content,omitempty"`
	IsError bool                `json:"is_error,omitempty"`
}

// imageSourceWire is the image source inside a user content block.
type imageSourceWire struct {
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

// resultMsg is the terminal message emitted when the session ends, type="result".
type resultMsg struct {
	Type             string          `json:"type"`
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

// streamEventMsg carries a partial streaming update, type="stream_event".
// Only emitted when --include-partial-messages is set.
type streamEventMsg struct {
	Type            string          `json:"type"`
	UUID            string          `json:"uuid,omitzero"`
	SessionID       string          `json:"session_id,omitzero"`
	Timestamp       json.RawMessage `json:"timestamp,omitzero"`
	ParentToolUseID string          `json:"parent_tool_use_id,omitzero"`
	Event           streamEvent     `json:"event"`
}

// streamEvent holds the event data for a single streaming update.
type streamEvent struct {
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

// rateLimitEventMsg is emitted when the CLI's rate limit status transitions
// (e.g. allowed → allowed_warning), type="rate_limit_event".
type rateLimitEventMsg struct {
	Type          string          `json:"type"`
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

// progressMsg is emitted during tool execution, type="progress".
type progressMsg struct {
	Type            string          `json:"type"`
	UUID            string          `json:"uuid,omitzero"`
	SessionID       string          `json:"session_id,omitzero"`
	Timestamp       json.RawMessage `json:"timestamp,omitzero"`
	ParentToolUseID string          `json:"parentToolUseID,omitzero"`
	ToolUseID       string          `json:"toolUseID,omitzero"`
	Data            json.RawMessage `json:"data"`
}

// progressPayload is the data field inside a progress record.
// The Type field discriminates variants.
type progressPayload struct {
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
