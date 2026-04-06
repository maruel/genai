// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Wire types for the OpenCode ACP (Agent Client Protocol) JSON-RPC 2.0 protocol.
//
// Type names follow the upstream ACP SDK definitions:
//
//	packages/opencode/src/acp/agent.ts — session update types and request/response handling
//
// Source: https://github.com/anomalyco/opencode
// Spec:   https://agentclientprotocol.com

package opencode

import "encoding/json"

// ============================================================
// Shared types: enums, JSON-RPC envelope, routing probes.
// ============================================================

// Method is a JSON-RPC method string for the ACP protocol.
type Method string

// JSON-RPC method constants for the ACP protocol.
const (
	// Request methods (client → agent).
	MethodInitialize              Method = "initialize"
	MethodSessionNew              Method = "session/new"
	MethodSessionLoad             Method = "session/load"
	MethodSessionPrompt           Method = "session/prompt"
	MethodSessionCancel           Method = "session/cancel"
	MethodSessionSetModel         Method = "session/set_model"
	MethodSessionSetMode          Method = "session/set_mode"
	MethodUnstableSetSessionModel Method = "unstable_setSessionModel"

	// Notification methods (agent → client).
	MethodSessionUpdate            Method = "session/update"
	MethodSessionRequestPermission Method = "session/request_permission"
)

// UpdateType is the session update discriminator (sessionUpdate field).
type UpdateType string

// Session update type constants.
const (
	UpdateAgentMessageChunk       UpdateType = "agent_message_chunk"
	UpdateAgentThoughtChunk       UpdateType = "agent_thought_chunk"
	UpdateUserMessageChunk        UpdateType = "user_message_chunk"
	UpdateToolCall                UpdateType = "tool_call"
	UpdateToolCallUpdate          UpdateType = "tool_call_update"
	UpdatePlan                    UpdateType = "plan"
	UpdateUsageUpdate             UpdateType = "usage_update"
	UpdateCurrentModeUpdate       UpdateType = "current_mode_update"
	UpdateSessionInfoUpdate       UpdateType = "session_info_update"
	UpdateAvailableCommandsUpdate UpdateType = "available_commands_update"
	UpdateConfigOptionUpdate      UpdateType = "config_option_update"
)

// ToolStatus is the status of a tool call.
type ToolStatus string

// Tool call status constants.
const (
	StatusPending    ToolStatus = "pending"
	StatusInProgress ToolStatus = "in_progress"
	StatusCompleted  ToolStatus = "completed"
	StatusFailed     ToolStatus = "failed"
)

// ToolKind is the kind of tool operation.
type ToolKind string

// Tool call kind constants.
const (
	KindRead       ToolKind = "read"
	KindEdit       ToolKind = "edit"
	KindDelete     ToolKind = "delete"
	KindMove       ToolKind = "move"
	KindSearch     ToolKind = "search"
	KindExecute    ToolKind = "execute"
	KindThink      ToolKind = "think"
	KindFetch      ToolKind = "fetch"
	KindSwitchMode ToolKind = "switch_mode"
	KindOther      ToolKind = "other"
)

// PlanStatus is the status of a plan entry.
type PlanStatus string

// Plan entry status constants.
const (
	PlanStatusPending    PlanStatus = "pending"
	PlanStatusInProgress PlanStatus = "in_progress"
	PlanStatusCompleted  PlanStatus = "completed"
	PlanStatusCancelled  PlanStatus = "cancelled"
)

// ContentType is the type discriminator for content blocks and prompt items.
type ContentType string

// Content type constants.
const (
	ContentText         ContentType = "text"
	ContentImage        ContentType = "image"
	ContentResource     ContentType = "resource"
	ContentResourceLink ContentType = "resource_link"
)

// ---------- JSON-RPC envelope ----------

// JSONRPCMessage is the JSON-RPC 2.0 envelope for ACP messages.
type JSONRPCMessage struct {
	JSONRPC string          `json:"jsonrpc"`
	Method  Method          `json:"method,omitzero"`
	ID      json.RawMessage `json:"id,omitzero"`
	Params  json.RawMessage `json:"params,omitzero"`
	Result  json.RawMessage `json:"result,omitzero"`
	Error   *JSONRPCError   `json:"error,omitzero"`
}

// IsResponse returns true if this is a response (has an ID).
func (m *JSONRPCMessage) IsResponse() bool { return m.ID != nil }

// JSONRPCError is a JSON-RPC 2.0 error object.
type JSONRPCError struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
}

// ---------- Routing probes ----------

// MessageProbe extracts routing fields from an ACP line to distinguish
// caic-injected JSON (has "type") from JSON-RPC (has "method"/"id").
type MessageProbe struct {
	Type   string          `json:"type,omitzero"`
	Method Method          `json:"method,omitzero"`
	ID     json.RawMessage `json:"id,omitzero"`
}

// ParamsProbe extracts the raw params field from a JSON-RPC message.
type ParamsProbe struct {
	Params json.RawMessage `json:"params,omitzero"`
}

// UpdateProbe extracts the discriminator from a session update.
type UpdateProbe struct {
	SessionUpdate UpdateType `json:"sessionUpdate"`
}

// ============================================================
// Input types: requests sent to OpenCode (stdin).
// ============================================================

// ---------- JSON-RPC request envelope ----------

// JSONRPCRequest is the envelope for all JSON-RPC 2.0 requests sent to OpenCode.
type JSONRPCRequest struct {
	JSONRPC string `json:"jsonrpc"`
	ID      int64  `json:"id,omitzero"`
	Method  Method `json:"method"`
	Params  any    `json:"params,omitzero"`
}

// ---------- Handshake request params ----------

// InitializeParams holds the params for the initialize request.
type InitializeParams struct {
	ProtocolVersion    int                `json:"protocolVersion"`
	ClientCapabilities ClientCapabilities `json:"clientCapabilities"`
	ClientInfo         ClientInfo         `json:"clientInfo"`
}

// ClientCapabilities holds the client capability flags for the initialize request.
type ClientCapabilities struct {
	Terminal bool `json:"terminal"`
}

// ClientInfo identifies the client in the initialize request.
type ClientInfo struct {
	Name    string `json:"name"`
	Title   string `json:"title"`
	Version string `json:"version"`
}

// ---------- Session management request params ----------

// SessionNewParams holds the params for session/new.
type SessionNewParams struct {
	Cwd        string      `json:"cwd"`
	McpServers []MCPServer `json:"mcpServers"`
}

// SessionLoadParams holds the params for session/load.
type SessionLoadParams struct {
	SessionID  string      `json:"sessionId"`
	Cwd        string      `json:"cwd"`
	McpServers []MCPServer `json:"mcpServers"`
}

// MCPServer describes an MCP server to register with the session.
// ACP supports three variants (stdio, http, sse) discriminated by the Type
// field. Only stdio is used by genai (for testing).
type MCPServer struct {
	Type    string        `json:"type,omitzero"` // "http", "sse", or empty for stdio.
	Name    string        `json:"name"`
	Command string        `json:"command,omitzero"` // Stdio only.
	Args    []string      `json:"args,omitzero"`    // Stdio only.
	Env     []EnvVariable `json:"env,omitzero"`     // Stdio only.
	URL     string        `json:"url,omitzero"`     // HTTP/SSE only.
	Headers []HTTPHeader  `json:"headers,omitzero"` // HTTP/SSE only.
}

// EnvVariable is a name-value pair for MCP server environment variables.
type EnvVariable struct {
	Name  string `json:"name"`
	Value string `json:"value"`
}

// HTTPHeader is a name-value pair for MCP server HTTP headers.
type HTTPHeader struct {
	Name  string `json:"name"`
	Value string `json:"value"`
}

// ---------- Prompt request params ----------

// PromptContent is a single item in the session/prompt content array.
// This is a flat union discriminated by Type:
//
//   - ContentText:         Text
//   - ContentImage:        Data (base64), MimeType
//   - ContentResource:     Resource (embedded resource)
//   - ContentResourceLink: URI, Name, MimeType
//
// Image handling: the ACP agent receives "image" parts and converts them to
// internal "file" parts with data URLs. As of OpenCode 1.2.27 (2026-03), the
// image data is correctly passed through to the model (verified by elevated
// inputTokens), but models accessed via OpenCode may reply "I can't see
// images" despite receiving the data. This appears to be a model routing
// issue in OpenCode, not a protocol problem.
//
// Relevant OpenCode source paths:
//   - ACP prompt handler: packages/opencode/src/acp/agent.ts case "image" (~line 1317)
//   - Internal part processing: packages/opencode/src/session/prompt.ts (~line 1068, data: URL switch)
//   - Model message conversion: packages/opencode/src/session/message-v2.ts toModelMessages() (~line 637)
//   - Related issue: https://github.com/anomalyco/opencode/issues/9217
type PromptContent struct {
	Type     ContentType     `json:"type"`
	Text     string          `json:"text,omitzero"`
	Data     string          `json:"data,omitzero"`     // Base64 image data.
	MimeType string          `json:"mimeType,omitzero"` // e.g. "image/png".
	URI      string          `json:"uri,omitzero"`
	Name     string          `json:"name,omitzero"`
	Resource json.RawMessage `json:"resource,omitzero"` // Embedded resource object.
}

// SessionPromptParams holds the params for session/prompt.
type SessionPromptParams struct {
	SessionID string          `json:"sessionId"`
	Prompt    []PromptContent `json:"prompt"`
}

// ---------- Model switching ----------

// SetSessionModelParams holds the params for unstable_setSessionModel.
type SetSessionModelParams struct {
	SessionID string `json:"sessionId"`
	ModelID   string `json:"modelId"`
}

// ============================================================
// Output types: notifications and responses received from OpenCode (stdout).
//
// Unknown field detection is centralized in unmarshalNotification
// (parse.go) rather than per-struct UnmarshalJSON methods.
// ============================================================

// ---------- Session update envelope ----------

// SessionUpdateParams holds the params for session/update notifications.
type SessionUpdateParams struct {
	SessionID string          `json:"sessionId"`
	Update    json.RawMessage `json:"update"`
}

// ---------- Content types ----------

// ContentBlock is a content block in message chunks. This is a flat union:
// fields are populated depending on Type.
//
//   - ContentText:         Text, Annotations
//   - ContentImage:        Data, MimeType, URI
//   - ContentResource:     Resource
//   - ContentResourceLink: URI, Name, MimeType
type ContentBlock struct {
	Type        ContentType     `json:"type"`
	Text        string          `json:"text,omitzero"`
	Data        string          `json:"data,omitzero"` // Base64 image data.
	MimeType    string          `json:"mimeType,omitzero"`
	URI         string          `json:"uri,omitzero"`
	Name        string          `json:"name,omitzero"`
	Resource    json.RawMessage `json:"resource,omitzero"`
	Annotations json.RawMessage `json:"annotations,omitzero"`
}

// ---------- Session update types ----------

// AgentMessageChunkUpdate is a streaming text chunk from the agent.
type AgentMessageChunkUpdate struct {
	SessionUpdate UpdateType   `json:"sessionUpdate"`
	Content       ContentBlock `json:"content"`
	MessageID     string       `json:"messageId,omitzero"`
}

// AgentThoughtChunkUpdate is a streaming reasoning chunk from the agent.
type AgentThoughtChunkUpdate struct {
	SessionUpdate UpdateType   `json:"sessionUpdate"`
	Content       ContentBlock `json:"content"`
	MessageID     string       `json:"messageId,omitzero"`
}

// UserMessageChunkUpdate is a replayed user message (during session/load).
type UserMessageChunkUpdate struct {
	SessionUpdate UpdateType   `json:"sessionUpdate"`
	Content       ContentBlock `json:"content"`
}

// ToolCallLocation is a file location associated with a tool call.
type ToolCallLocation struct {
	Path string `json:"path,omitzero"`
	Line int    `json:"line,omitzero"`
}

// ToolCallUpdate is the initial tool call announcement.
type ToolCallUpdate struct {
	SessionUpdate UpdateType         `json:"sessionUpdate"`
	ToolCallID    string             `json:"toolCallId"`
	Title         string             `json:"title,omitzero"`
	Kind          ToolKind           `json:"kind,omitzero"`
	Status        ToolStatus         `json:"status,omitzero"`
	Locations     []ToolCallLocation `json:"locations,omitzero"`
	RawInput      json.RawMessage    `json:"rawInput,omitzero"`
}

// ToolCallContent is a content entry in a tool call update result. This is a
// flat union discriminated by Type:
//
//   - "content":       Content (text block)
//   - "diff":          Path, OldText, NewText
//   - "image":         Content.Data, Content.MimeType
//   - "resource":      Content.Resource
//   - "resource_link": Content.URI, Content.Name, Content.MimeType
type ToolCallContent struct {
	Type    string       `json:"type"`
	Content ContentBlock `json:"content,omitzero"`
	// Diff fields.
	Path    string `json:"path,omitzero"`
	OldText string `json:"oldText,omitzero"`
	NewText string `json:"newText,omitzero"`
}

// ToolCallRawOutput is the structured raw output from a tool call.
type ToolCallRawOutput struct {
	Output   string          `json:"output,omitzero"`
	Error    string          `json:"error,omitzero"`
	Metadata json.RawMessage `json:"metadata,omitzero"`
}

// ToolCallUpdateUpdate is a tool call progress/completion update.
type ToolCallUpdateUpdate struct {
	SessionUpdate UpdateType         `json:"sessionUpdate"`
	ToolCallID    string             `json:"toolCallId"`
	Title         string             `json:"title,omitzero"`
	Kind          ToolKind           `json:"kind,omitzero"`
	Status        ToolStatus         `json:"status,omitzero"`
	Locations     []ToolCallLocation `json:"locations,omitzero"`
	RawInput      json.RawMessage    `json:"rawInput,omitzero"`
	RawOutput     *ToolCallRawOutput `json:"rawOutput,omitzero"`
	Content       []ToolCallContent  `json:"content,omitzero"`
}

// PlanEntry is a single entry in a plan update.
type PlanEntry struct {
	Priority string     `json:"priority,omitzero"`
	Status   PlanStatus `json:"status"`
	Content  string     `json:"content"`
}

// PlanUpdate is a todo/plan update from the agent.
type PlanUpdate struct {
	SessionUpdate UpdateType  `json:"sessionUpdate"`
	Entries       []PlanEntry `json:"entries"`
}

// UsageCost describes the cost of usage.
type UsageCost struct {
	Amount   float64 `json:"amount"`
	Currency string  `json:"currency"`
}

// UsageUpdateUpdate is a context window / cost update.
type UsageUpdateUpdate struct {
	SessionUpdate UpdateType `json:"sessionUpdate"`
	Used          int        `json:"used"`
	Size          int        `json:"size"`
	Cost          UsageCost  `json:"cost,omitzero"`
}

// CurrentModeUpdate is a mode change notification.
type CurrentModeUpdate struct {
	SessionUpdate UpdateType `json:"sessionUpdate"`
	ModeID        string     `json:"modeId,omitzero"`
	ModeName      string     `json:"modeName,omitzero"`
}

// AvailableCommand is a single command in an available_commands_update.
type AvailableCommand struct {
	Name        string `json:"name"`
	Description string `json:"description,omitzero"`
}

// AvailableCommandsUpdate lists commands available in the current session.
type AvailableCommandsUpdate struct {
	SessionUpdate     UpdateType         `json:"sessionUpdate"`
	AvailableCommands []AvailableCommand `json:"availableCommands"`
}

// ---------- Permission request ----------

// PermissionToolCall describes the tool call in a permission request.
type PermissionToolCall struct {
	ToolCallID string             `json:"toolCallId"`
	Status     ToolStatus         `json:"status,omitzero"`
	Title      string             `json:"title,omitzero"`
	Kind       ToolKind           `json:"kind,omitzero"`
	RawInput   json.RawMessage    `json:"rawInput,omitzero"`
	Locations  []ToolCallLocation `json:"locations,omitzero"`
}

// PermissionOption is a single option in a permission request.
type PermissionOption struct {
	OptionID string `json:"optionId"`
	Kind     string `json:"kind"` // "allow_once", "allow_always", "reject_once".
	Name     string `json:"name"`
}

// PermissionRequestParams holds params for session/request_permission.
type PermissionRequestParams struct {
	SessionID string             `json:"sessionId"`
	ToolCall  PermissionToolCall `json:"toolCall"`
	Options   []PermissionOption `json:"options"`
}

// ---------- Response types ----------

// InitializeResult is the result of an initialize request.
type InitializeResult struct {
	ProtocolVersion   int               `json:"protocolVersion"`
	AgentCapabilities AgentCapabilities `json:"agentCapabilities,omitzero"`
	AgentInfo         AgentInfo         `json:"agentInfo,omitzero"`
	AuthMethods       json.RawMessage   `json:"authMethods,omitzero"`
}

// AgentCapabilities holds the agent's declared capabilities from the initialize response.
type AgentCapabilities struct {
	PromptCapabilities  PromptCapabilities `json:"promptCapabilities,omitzero"`
	LoadSession         bool               `json:"loadSession,omitzero"`
	McpCapabilities     json.RawMessage    `json:"mcpCapabilities,omitzero"`
	SessionCapabilities json.RawMessage    `json:"sessionCapabilities,omitzero"`
}

// PromptCapabilities describes prompt content types the agent supports.
type PromptCapabilities struct {
	Image           bool `json:"image,omitzero"`
	EmbeddedContext bool `json:"embeddedContext,omitzero"`
}

// AgentInfo identifies the agent in the initialize response.
type AgentInfo struct {
	Name    string `json:"name,omitzero"`
	Version string `json:"version,omitzero"`
}

// SessionNewResult is the result of a session/new request.
type SessionNewResult struct {
	SessionID string          `json:"sessionId"`
	Models    ModelsInfo      `json:"models,omitzero"`
	Modes     ModesInfo       `json:"modes,omitzero"`
	Meta      json.RawMessage `json:"_meta,omitzero"`
}

// ModelsInfo holds the current and available models from a session response.
type ModelsInfo struct {
	CurrentModelID  string      `json:"currentModelId,omitzero"`
	AvailableModels []ModelInfo `json:"availableModels,omitzero"`
}

// ModelInfo describes a single available model.
type ModelInfo struct {
	ModelID string `json:"modelId"`
	Name    string `json:"name,omitzero"`
}

// ModesInfo holds the current and available modes from a session response.
type ModesInfo struct {
	CurrentModeID  string     `json:"currentModeId,omitzero"`
	AvailableModes []ModeInfo `json:"availableModes,omitzero"`
}

// ModeInfo describes a single available mode.
type ModeInfo struct {
	ID          string `json:"id"`
	Name        string `json:"name,omitzero"`
	Description string `json:"description,omitzero"`
}

// PromptResult is the result of a session/prompt response.
type PromptResult struct {
	StopReason string          `json:"stopReason,omitzero"` // "end_turn", "max_tokens", "cancelled", "refusal".
	Usage      PromptUsage     `json:"usage,omitzero"`
	Meta       json.RawMessage `json:"_meta,omitzero"`
}

// PromptUsage holds the token usage from a session/prompt response.
type PromptUsage struct {
	TotalTokens       int `json:"totalTokens,omitzero"`
	InputTokens       int `json:"inputTokens,omitzero"`
	OutputTokens      int `json:"outputTokens,omitzero"`
	ThoughtTokens     int `json:"thoughtTokens,omitzero"`
	CachedReadTokens  int `json:"cachedReadTokens,omitzero"`
	CachedWriteTokens int `json:"cachedWriteTokens,omitzero"`
}

// JSONRPCResponse is a JSON-RPC 2.0 response sent back to the agent (e.g. for
// permission requests).
type JSONRPCResponse struct {
	JSONRPC string `json:"jsonrpc"`
	ID      int64  `json:"id"`
	Result  any    `json:"result"`
}

// PermissionResponseResult is the result sent back for a permission request.
type PermissionResponseResult struct {
	OptionID string `json:"optionId"`
}
