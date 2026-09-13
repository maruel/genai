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
	MethodAuthenticate           Method = "authenticate"
	MethodInitialize             Method = "initialize"
	MethodSessionClose           Method = "session/close"
	MethodSessionFork            Method = "session/fork"
	MethodSessionList            Method = "session/list"
	MethodSessionNew             Method = "session/new"
	MethodSessionLoad            Method = "session/load"
	MethodSessionPrompt          Method = "session/prompt"
	MethodSessionResume          Method = "session/resume"
	MethodSessionCancel          Method = "session/cancel"
	MethodSessionSetModel        Method = "session/set_model"
	MethodSessionSetMode         Method = "session/set_mode"
	MethodSessionSetConfigOption Method = "session/set_config_option"

	// Request and notification methods (agent → client).
	MethodFSWriteTextFile          Method = "fs/write_text_file"
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
)

// PlanPriority is the relative importance of a plan entry.
type PlanPriority string

// Plan entry priority constants.
const (
	PlanPriorityHigh   PlanPriority = "high"
	PlanPriorityLow    PlanPriority = "low"
	PlanPriorityMedium PlanPriority = "medium"
)

// ContentType is the type discriminator for content blocks and prompt items.
type ContentType string

// Content type constants.
const (
	ContentAudio        ContentType = "audio"
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

// IsResponse reports whether this is a response to a client request.
func (m *JSONRPCMessage) IsResponse() bool { return len(m.ID) != 0 && m.Method == "" }

// IsAgentRequest reports whether the agent expects a response from the client.
func (m *JSONRPCMessage) IsAgentRequest() bool { return len(m.ID) != 0 && m.Method != "" }

// JSONRPCError is a JSON-RPC 2.0 error object.
type JSONRPCError struct {
	Code    int             `json:"code"`
	Message string          `json:"message"`
	Data    json.RawMessage `json:"data,omitzero"`
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
	JSONRPC string          `json:"jsonrpc"`
	ID      int64           `json:"id,omitzero"`
	Method  Method          `json:"method"`
	Params  json.RawMessage `json:"params,omitzero"`
}

// ---------- Handshake request params ----------

// InitializeParams holds the params for the initialize request.
type InitializeParams struct {
	ProtocolVersion    int                `json:"protocolVersion"`
	ClientCapabilities ClientCapabilities `json:"clientCapabilities"`
	ClientInfo         ClientInfo         `json:"clientInfo"`
	Meta               json.RawMessage    `json:"_meta,omitzero"`
}

// ClientCapabilities holds the client capability flags for the initialize request.
type ClientCapabilities struct {
	Auth              json.RawMessage        `json:"auth,omitzero"`
	Elicitation       json.RawMessage        `json:"elicitation,omitzero"`
	FS                FileSystemCapabilities `json:"fs,omitzero"`
	Nes               json.RawMessage        `json:"nes,omitzero"`
	PositionEncodings []string               `json:"positionEncodings,omitzero"`
	Terminal          bool                   `json:"terminal"`
	Meta              json.RawMessage        `json:"_meta,omitzero"`
}

// FileSystemCapabilities describes the client-side file operations available to the agent.
type FileSystemCapabilities struct {
	ReadTextFile  bool            `json:"readTextFile,omitzero"`
	WriteTextFile bool            `json:"writeTextFile,omitzero"`
	Meta          json.RawMessage `json:"_meta,omitzero"`
}

// ClientInfo identifies the client in the initialize request.
type ClientInfo struct {
	Name    string          `json:"name"`
	Title   string          `json:"title"`
	Version string          `json:"version"`
	Meta    json.RawMessage `json:"_meta,omitzero"`
}

// ---------- Session management request params ----------

// SessionNewParams holds the params for session/new.
type SessionNewParams struct {
	Cwd                   string          `json:"cwd"`
	McpServers            []MCPServer     `json:"mcpServers"`
	AdditionalDirectories []string        `json:"additionalDirectories,omitzero"`
	Meta                  json.RawMessage `json:"_meta,omitzero"`
}

// SessionLoadParams holds the params for session/load.
type SessionLoadParams struct {
	SessionID             string          `json:"sessionId"`
	Cwd                   string          `json:"cwd"`
	McpServers            []MCPServer     `json:"mcpServers"`
	AdditionalDirectories []string        `json:"additionalDirectories,omitzero"`
	Meta                  json.RawMessage `json:"_meta,omitzero"`
}

// SessionResumeParams holds the params for session/resume.
type SessionResumeParams struct {
	SessionID             string          `json:"sessionId"`
	Cwd                   string          `json:"cwd"`
	McpServers            []MCPServer     `json:"mcpServers,omitzero"`
	AdditionalDirectories []string        `json:"additionalDirectories,omitzero"`
	Meta                  json.RawMessage `json:"_meta,omitzero"`
}

// SessionForkParams holds the params for session/fork.
type SessionForkParams struct {
	SessionID             string          `json:"sessionId"`
	Cwd                   string          `json:"cwd"`
	McpServers            []MCPServer     `json:"mcpServers,omitzero"`
	AdditionalDirectories []string        `json:"additionalDirectories,omitzero"`
	Meta                  json.RawMessage `json:"_meta,omitzero"`
}

// SessionCloseParams holds the params for session/close.
type SessionCloseParams struct {
	SessionID string          `json:"sessionId"`
	Meta      json.RawMessage `json:"_meta,omitzero"`
}

// SessionCancelParams holds the params for the session/cancel notification.
type SessionCancelParams struct {
	SessionID string          `json:"sessionId"`
	Meta      json.RawMessage `json:"_meta,omitzero"`
}

// SessionListParams holds the params for session/list.
type SessionListParams struct {
	AdditionalDirectories []string        `json:"additionalDirectories,omitzero"`
	Cursor                string          `json:"cursor,omitzero"`
	Cwd                   string          `json:"cwd,omitzero"`
	Meta                  json.RawMessage `json:"_meta,omitzero"`
}

// MCPServer describes an MCP server to register with the session.
// ACP supports three variants (stdio, http, sse) discriminated by the Type
// field. Only stdio is used by genai (for testing).
type MCPServer struct {
	Type    string          `json:"type,omitzero"` // "http", "sse", or empty for stdio.
	Name    string          `json:"name"`
	Command string          `json:"command,omitzero"` // Stdio only.
	Args    []string        `json:"args,omitzero"`    // Stdio only.
	Env     []EnvVariable   `json:"env,omitzero"`     // Stdio only.
	URL     string          `json:"url,omitzero"`     // HTTP/SSE only.
	Headers []HTTPHeader    `json:"headers,omitzero"` // HTTP/SSE only.
	Meta    json.RawMessage `json:"_meta,omitzero"`
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
// OpenCode converts these blocks in packages/opencode/src/acp/content.ts.
type PromptContent struct {
	Type     ContentType     `json:"type"`
	Text     string          `json:"text,omitzero"`
	Data     string          `json:"data,omitzero"`     // Base64 image data.
	MimeType string          `json:"mimeType,omitzero"` // e.g. "image/png".
	URI      string          `json:"uri,omitzero"`
	Name     string          `json:"name,omitzero"`
	Resource json.RawMessage `json:"resource,omitzero"` // Embedded resource object.
	Meta     json.RawMessage `json:"_meta,omitzero"`
}

// SessionPromptParams holds the params for session/prompt.
type SessionPromptParams struct {
	SessionID string          `json:"sessionId"`
	Prompt    []PromptContent `json:"prompt"`
	MessageID string          `json:"messageId,omitzero"`
	Meta      json.RawMessage `json:"_meta,omitzero"`
}

// ---------- Session configuration ----------

// SetSessionModelParams holds the params for session/set_model.
type SetSessionModelParams struct {
	SessionID string          `json:"sessionId"`
	ModelID   string          `json:"modelId"`
	Meta      json.RawMessage `json:"_meta,omitzero"`
}

// SetSessionModeParams holds the params for session/set_mode.
type SetSessionModeParams struct {
	SessionID string          `json:"sessionId"`
	ModeID    string          `json:"modeId"`
	Meta      json.RawMessage `json:"_meta,omitzero"`
}

// AuthenticateParams holds the params for authenticate.
type AuthenticateParams struct {
	MethodID string          `json:"methodId"`
	Meta     json.RawMessage `json:"_meta,omitzero"`
}

// WriteTextFileParams is the fs/write_text_file request shape used by OpenCode.
// The client rejects this method because it does not advertise filesystem capabilities.
type WriteTextFileParams struct {
	SessionID string          `json:"sessionId"`
	Path      string          `json:"path"`
	Content   string          `json:"content"`
	Meta      json.RawMessage `json:"_meta,omitzero"`
}

// ConfigOptionID identifies an ACP session configuration option.
type ConfigOptionID string

// Session configuration option IDs exposed by OpenCode.
const (
	ConfigOptionModel  ConfigOptionID = "model"
	ConfigOptionEffort ConfigOptionID = "effort"
	ConfigOptionMode   ConfigOptionID = "mode"
)

// ConfigOptionCategory groups ACP session configuration options for clients.
type ConfigOptionCategory string

// Session configuration option categories exposed by OpenCode.
const (
	ConfigOptionCategoryModel        ConfigOptionCategory = "model"
	ConfigOptionCategoryThoughtLevel ConfigOptionCategory = "thought_level"
	ConfigOptionCategoryMode         ConfigOptionCategory = "mode"
)

// ConfigOptionType is the ACP configuration-option control type.
type ConfigOptionType string

// Session configuration option control types exposed by OpenCode.
const (
	ConfigOptionTypeBoolean ConfigOptionType = "boolean"
	ConfigOptionTypeSelect  ConfigOptionType = "select"
)

// Effort is an OpenCode reasoning-effort value serialized in the ACP effort
// configuration option.
//
// Availability is provider- and model-specific. The provider validates it
// against the effort values returned by the ACP session.
type Effort string

// Reasoning effort levels supported by OpenCode model variants.
const (
	EffortDefault Effort = "default"
	EffortNone    Effort = "none"
	EffortMinimal Effort = "minimal"
	EffortLow     Effort = "low"
	EffortMedium  Effort = "medium"
	EffortHigh    Effort = "high"
	EffortXHigh   Effort = "xhigh"
	EffortMax     Effort = "max"
)

// Mode is an OpenCode session-mode value serialized in the ACP mode
// configuration option.
type Mode string

// SetSessionConfigOptionParams holds the params for session/set_config_option.
type SetSessionConfigOptionParams struct {
	SessionID string          `json:"sessionId"`
	ConfigID  ConfigOptionID  `json:"configId"`
	Value     string          `json:"value"`
	Meta      json.RawMessage `json:"_meta,omitzero"`
}

// SetSessionBooleanConfigOptionParams holds a boolean session/set_config_option request.
type SetSessionBooleanConfigOptionParams struct {
	SessionID string           `json:"sessionId"`
	ConfigID  ConfigOptionID   `json:"configId"`
	Type      ConfigOptionType `json:"type"`
	Value     bool             `json:"value"`
	Meta      json.RawMessage  `json:"_meta,omitzero"`
}

// ============================================================
// Output types: notifications and responses received from OpenCode (stdout).
// ============================================================

// ---------- Session update envelope ----------

// SessionUpdateParams holds the params for session/update notifications.
type SessionUpdateParams struct {
	SessionID string          `json:"sessionId"`
	Update    json.RawMessage `json:"update"`
	Meta      json.RawMessage `json:"_meta,omitzero"`
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
	Meta        json.RawMessage `json:"_meta,omitzero"`
}

// ---------- Session update types ----------

// AgentMessageChunkUpdate is a streaming text chunk from the agent.
type AgentMessageChunkUpdate struct {
	SessionUpdate UpdateType      `json:"sessionUpdate"`
	Content       ContentBlock    `json:"content"`
	MessageID     string          `json:"messageId,omitzero"`
	Meta          json.RawMessage `json:"_meta,omitzero"`
}

// AgentThoughtChunkUpdate is a streaming reasoning chunk from the agent.
type AgentThoughtChunkUpdate struct {
	SessionUpdate UpdateType      `json:"sessionUpdate"`
	Content       ContentBlock    `json:"content"`
	MessageID     string          `json:"messageId,omitzero"`
	Meta          json.RawMessage `json:"_meta,omitzero"`
}

// UserMessageChunkUpdate is a replayed user message (during session/load).
type UserMessageChunkUpdate struct {
	SessionUpdate UpdateType      `json:"sessionUpdate"`
	MessageID     string          `json:"messageId,omitzero"`
	Content       ContentBlock    `json:"content"`
	Meta          json.RawMessage `json:"_meta,omitzero"`
}

// ToolCallLocation is a file location associated with a tool call.
type ToolCallLocation struct {
	Path string          `json:"path,omitzero"`
	Line int             `json:"line,omitzero"`
	Meta json.RawMessage `json:"_meta,omitzero"`
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
	RawOutput     json.RawMessage    `json:"rawOutput,omitzero"`
	Content       []ToolCallContent  `json:"content,omitzero"`
	Meta          json.RawMessage    `json:"_meta,omitzero"`
}

// EditInput is the rawInput shape for OpenCode edit and replace tool calls.
type EditInput struct {
	FilePath  string `json:"filePath"`
	OldString string `json:"oldString"`
	NewString string `json:"newString"`
}

// ToolCallContent is a content entry in a tool call update result. This is a
// flat union discriminated by Type:
//
//   - "content":       Content (standard content block)
//   - "diff":          Path, OldText, NewText
//   - "terminal":      TerminalID
type ToolCallContent struct {
	Type    string       `json:"type"`
	Content ContentBlock `json:"content,omitzero"`
	// Diff fields.
	Path    string `json:"path,omitzero"`
	OldText string `json:"oldText,omitzero"`
	NewText string `json:"newText,omitzero"`
	// Terminal field.
	TerminalID string          `json:"terminalId,omitzero"`
	Meta       json.RawMessage `json:"_meta,omitzero"`
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
	RawOutput     json.RawMessage    `json:"rawOutput,omitzero"`
	Content       []ToolCallContent  `json:"content,omitzero"`
	Meta          json.RawMessage    `json:"_meta,omitzero"`
}

// PlanEntry is a single entry in a plan update.
type PlanEntry struct {
	Priority PlanPriority    `json:"priority"`
	Status   PlanStatus      `json:"status"`
	Content  string          `json:"content"`
	Meta     json.RawMessage `json:"_meta,omitzero"`
}

// PlanUpdate is a todo/plan update from the agent.
type PlanUpdate struct {
	SessionUpdate UpdateType      `json:"sessionUpdate"`
	Entries       []PlanEntry     `json:"entries"`
	Meta          json.RawMessage `json:"_meta,omitzero"`
}

// UsageCost describes the cost of usage.
type UsageCost struct {
	Amount   float64         `json:"amount"`
	Currency string          `json:"currency"`
	Meta     json.RawMessage `json:"_meta,omitzero"`
}

// UsageUpdateUpdate is a context window / cost update.
type UsageUpdateUpdate struct {
	SessionUpdate UpdateType      `json:"sessionUpdate"`
	Used          int             `json:"used"`
	Size          int             `json:"size"`
	Cost          UsageCost       `json:"cost,omitzero"`
	Meta          json.RawMessage `json:"_meta,omitzero"`
}

// CurrentModeUpdate is a mode change notification.
type CurrentModeUpdate struct {
	SessionUpdate UpdateType      `json:"sessionUpdate"`
	CurrentModeID string          `json:"currentModeId"`
	Meta          json.RawMessage `json:"_meta,omitzero"`
}

// AvailableCommand is a single command in an available_commands_update.
type AvailableCommand struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitzero"`
	Input       json.RawMessage `json:"input,omitzero"`
	Meta        json.RawMessage `json:"_meta,omitzero"`
}

// AvailableCommandsUpdate lists commands available in the current session.
type AvailableCommandsUpdate struct {
	SessionUpdate     UpdateType         `json:"sessionUpdate"`
	AvailableCommands []AvailableCommand `json:"availableCommands"`
	Meta              json.RawMessage    `json:"_meta,omitzero"`
}

// ConfigOptionUpdate reports the complete current session configuration.
type ConfigOptionUpdate struct {
	SessionUpdate UpdateType            `json:"sessionUpdate"`
	ConfigOptions []SessionConfigOption `json:"configOptions"`
	Meta          json.RawMessage       `json:"_meta,omitzero"`
}

// SessionInfoUpdate reports partial session metadata changes.
type SessionInfoUpdate struct {
	SessionUpdate UpdateType      `json:"sessionUpdate"`
	Title         string          `json:"title,omitzero"`
	UpdatedAt     string          `json:"updatedAt,omitzero"`
	Meta          json.RawMessage `json:"_meta,omitzero"`
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
	RawOutput  json.RawMessage    `json:"rawOutput,omitzero"`
	Content    []ToolCallContent  `json:"content,omitzero"`
	Meta       json.RawMessage    `json:"_meta,omitzero"`
}

// PermissionOption is a single option in a permission request.
type PermissionOption struct {
	OptionID string               `json:"optionId"`
	Kind     PermissionOptionKind `json:"kind"`
	Name     string               `json:"name"`
	Meta     json.RawMessage      `json:"_meta,omitzero"`
}

// PermissionOptionKind hints at the effect and persistence of a permission option.
type PermissionOptionKind string

// Permission option kinds.
const (
	PermissionAllowAlways  PermissionOptionKind = "allow_always"
	PermissionAllowOnce    PermissionOptionKind = "allow_once"
	PermissionRejectAlways PermissionOptionKind = "reject_always"
	PermissionRejectOnce   PermissionOptionKind = "reject_once"
)

// PermissionRequestParams holds params for session/request_permission.
type PermissionRequestParams struct {
	SessionID string             `json:"sessionId"`
	ToolCall  PermissionToolCall `json:"toolCall"`
	Options   []PermissionOption `json:"options"`
	Meta      json.RawMessage    `json:"_meta,omitzero"`
}

// ---------- Response types ----------

// InitializeResult is the result of an initialize request.
type InitializeResult struct {
	ProtocolVersion   int               `json:"protocolVersion"`
	AgentCapabilities AgentCapabilities `json:"agentCapabilities,omitzero"`
	AgentInfo         AgentInfo         `json:"agentInfo,omitzero"`
	AuthMethods       []AuthMethod      `json:"authMethods,omitzero"`
	Meta              json.RawMessage   `json:"_meta,omitzero"`
}

// AuthMethod describes an agent-managed authentication choice advertised by OpenCode.
type AuthMethod struct {
	ID          string          `json:"id"`
	Name        string          `json:"name"`
	Description string          `json:"description,omitzero"`
	Meta        json.RawMessage `json:"_meta,omitzero"`
}

// AgentCapabilities holds the agent's declared capabilities from the initialize response.
type AgentCapabilities struct {
	Auth                json.RawMessage     `json:"auth,omitzero"`
	LoadSession         bool                `json:"loadSession,omitzero"`
	MCPCapabilities     MCPCapabilities     `json:"mcpCapabilities,omitzero"`
	Nes                 json.RawMessage     `json:"nes,omitzero"`
	PositionEncoding    string              `json:"positionEncoding,omitzero"`
	PromptCapabilities  PromptCapabilities  `json:"promptCapabilities,omitzero"`
	Providers           json.RawMessage     `json:"providers,omitzero"`
	SessionCapabilities SessionCapabilities `json:"sessionCapabilities,omitzero"`
	Meta                json.RawMessage     `json:"_meta,omitzero"`
}

// MCPCapabilities describes the optional MCP transports supported by the agent.
type MCPCapabilities struct {
	HTTP bool            `json:"http,omitzero"`
	SSE  bool            `json:"sse,omitzero"`
	Meta json.RawMessage `json:"_meta,omitzero"`
}

// SessionCapabilities describes optional session lifecycle methods supported by the agent.
type SessionCapabilities struct {
	AdditionalDirectories MarkerCapabilities `json:"additionalDirectories,omitzero"`
	Close                 MarkerCapabilities `json:"close,omitzero"`
	Fork                  MarkerCapabilities `json:"fork,omitzero"`
	List                  MarkerCapabilities `json:"list,omitzero"`
	Resume                MarkerCapabilities `json:"resume,omitzero"`
	Meta                  json.RawMessage    `json:"_meta,omitzero"`
}

// MarkerCapabilities advertises support through an otherwise empty capability object.
type MarkerCapabilities struct {
	Meta json.RawMessage `json:"_meta,omitzero"`
}

// PromptCapabilities describes prompt content types the agent supports.
type PromptCapabilities struct {
	Audio           bool            `json:"audio,omitzero"`
	Image           bool            `json:"image,omitzero"`
	EmbeddedContext bool            `json:"embeddedContext,omitzero"`
	Meta            json.RawMessage `json:"_meta,omitzero"`
}

// AgentInfo identifies the agent in the initialize response.
type AgentInfo struct {
	Name    string          `json:"name,omitzero"`
	Title   string          `json:"title,omitzero"`
	Version string          `json:"version,omitzero"`
	Meta    json.RawMessage `json:"_meta,omitzero"`
}

// SessionNewResult is the result of a session/new request.
type SessionNewResult struct {
	SessionID     string                `json:"sessionId"`
	ConfigOptions []SessionConfigOption `json:"configOptions,omitzero"`
	Models        ModelsInfo            `json:"models,omitzero"`
	Modes         ModesInfo             `json:"modes,omitzero"`
	Meta          json.RawMessage       `json:"_meta,omitzero"`
}

// SessionInfo describes one session returned by session/list.
type SessionInfo struct {
	SessionID             string          `json:"sessionId"`
	Cwd                   string          `json:"cwd"`
	Title                 string          `json:"title,omitzero"`
	UpdatedAt             string          `json:"updatedAt,omitzero"`
	AdditionalDirectories []string        `json:"additionalDirectories,omitzero"`
	Meta                  json.RawMessage `json:"_meta,omitzero"`
}

// SessionListResult is returned by session/list.
type SessionListResult struct {
	Sessions   []SessionInfo   `json:"sessions"`
	NextCursor string          `json:"nextCursor,omitzero"`
	Meta       json.RawMessage `json:"_meta,omitzero"`
}

// SessionStateResult is the session state returned by session/load and session/resume.
type SessionStateResult struct {
	ConfigOptions []SessionConfigOption `json:"configOptions,omitzero"`
	Models        ModelsInfo            `json:"models,omitzero"`
	Modes         ModesInfo             `json:"modes,omitzero"`
	Meta          json.RawMessage       `json:"_meta,omitzero"`
}

// SessionLoadResult is returned by session/load.
type SessionLoadResult struct{ SessionStateResult }

// SessionResumeResult is returned by session/resume.
type SessionResumeResult struct{ SessionStateResult }

// SessionForkResult is returned by session/fork.
type SessionForkResult struct {
	SessionID string `json:"sessionId"`
	SessionStateResult
}

// EmptyResult is returned by successful ACP methods with no result fields.
type EmptyResult struct {
	Meta json.RawMessage `json:"_meta,omitzero"`
}

// WriteTextFileResult is the empty successful fs/write_text_file response shape.
type WriteTextFileResult struct {
	Meta json.RawMessage `json:"_meta,omitzero"`
}

// SessionConfigOption is a configuration control returned with an ACP session.
type SessionConfigOption struct {
	ID           ConfigOptionID       `json:"id"`
	Name         string               `json:"name"`
	Description  string               `json:"description,omitzero"`
	Category     ConfigOptionCategory `json:"category"`
	Type         ConfigOptionType     `json:"type"`
	CurrentValue json.RawMessage      `json:"currentValue"`
	Options      []ConfigOptionValue  `json:"options,omitzero"`
	Meta         json.RawMessage      `json:"_meta,omitzero"`
}

// ConfigOptionValue is a selectable value in a SessionConfigOption.
type ConfigOptionValue struct {
	Value       string          `json:"value"`
	Name        string          `json:"name"`
	Description string          `json:"description,omitzero"`
	Meta        json.RawMessage `json:"_meta,omitzero"`
}

// SetSessionConfigOptionResult is returned by session/set_config_option.
type SetSessionConfigOptionResult struct {
	ConfigOptions []SessionConfigOption `json:"configOptions"`
	Meta          json.RawMessage       `json:"_meta,omitzero"`
}

// ModelsInfo holds the current and available models from a session response.
type ModelsInfo struct {
	CurrentModelID  string          `json:"currentModelId,omitzero"`
	AvailableModels []ModelInfo     `json:"availableModels,omitzero"`
	Meta            json.RawMessage `json:"_meta,omitzero"`
}

// ModelInfo describes a single available model.
type ModelInfo struct {
	ModelID string          `json:"modelId"`
	Name    string          `json:"name,omitzero"`
	Meta    json.RawMessage `json:"_meta,omitzero"`
}

// ModesInfo holds the current and available modes from a session response.
type ModesInfo struct {
	CurrentModeID  string          `json:"currentModeId,omitzero"`
	AvailableModes []ModeInfo      `json:"availableModes,omitzero"`
	Meta           json.RawMessage `json:"_meta,omitzero"`
}

// ModeInfo describes a single available mode.
type ModeInfo struct {
	ID          string          `json:"id"`
	Name        string          `json:"name,omitzero"`
	Description string          `json:"description,omitzero"`
	Meta        json.RawMessage `json:"_meta,omitzero"`
}

// StopReason identifies why the agent stopped processing a prompt turn.
type StopReason string

// Prompt stop reasons.
const (
	StopReasonCancelled       StopReason = "cancelled"
	StopReasonEndTurn         StopReason = "end_turn"
	StopReasonMaxTokens       StopReason = "max_tokens"
	StopReasonMaxTurnRequests StopReason = "max_turn_requests"
	StopReasonRefusal         StopReason = "refusal"
)

// PromptResult is the result of a session/prompt response.
type PromptResult struct {
	StopReason    StopReason      `json:"stopReason"`
	Usage         PromptUsage     `json:"usage,omitzero"`
	UserMessageID string          `json:"userMessageId,omitzero"`
	Meta          json.RawMessage `json:"_meta,omitzero"`
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
	JSONRPC string          `json:"jsonrpc"`
	ID      json.RawMessage `json:"id"`
	Result  json.RawMessage `json:"result,omitzero"`
	Error   JSONRPCError    `json:"error,omitzero"`
}

// PermissionResponseResult is the result sent back for a permission request.
type PermissionResponseResult struct {
	Outcome PermissionOutcome `json:"outcome"`
}

// PermissionOutcomeType identifies whether a permission was selected or cancelled.
type PermissionOutcomeType string

// Permission outcome types.
const (
	PermissionOutcomeCancelled PermissionOutcomeType = "cancelled"
	PermissionOutcomeSelected  PermissionOutcomeType = "selected"
)

// PermissionOutcome reports either the selected option or cancellation.
type PermissionOutcome struct {
	Outcome  PermissionOutcomeType `json:"outcome"`
	OptionID string                `json:"optionId,omitzero"`
	Meta     json.RawMessage       `json:"_meta,omitzero"`
}
