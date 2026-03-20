// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package opencode

import (
	"encoding/json"

	"github.com/maruel/genai"
)

// ACP (Agent Client Protocol) JSON-RPC 2.0 wire types for the OpenCode CLI.
//
// Each line on stdin/stdout is a complete JSON object terminated by newline.
// Requests and responses carry an "id" field; notifications carry "method" only.
// Requests from the agent (e.g. permission requests) carry both "id" and "method".
//
// References:
//   - https://agentclientprotocol.com
//   - https://opencode.ai

// ---------- JSON-RPC notification method constants ----------

const (
	methodSessionUpdate            = "session/update"
	methodSessionRequestPermission = "session/request_permission"
)

// Session update type discriminators (sessionUpdate field).
const (
	updateAgentMessageChunk = "agent_message_chunk"
	updateAgentThoughtChunk = "agent_thought_chunk"
	updateUsageUpdate       = "usage_update"
)

// ---------- JSON-RPC envelope ----------

// jsonrpcRequest is the envelope for all JSON-RPC 2.0 requests sent to OpenCode.
type jsonrpcRequest struct {
	JSONRPC string `json:"jsonrpc"`
	ID      int64  `json:"id,omitzero"`
	Method  string `json:"method"`
	Params  any    `json:"params,omitzero"`
}

// jsonrpcNotification is a JSON-RPC 2.0 notification (no id, no response expected).
type jsonrpcNotification struct {
	JSONRPC string `json:"jsonrpc"`
	Method  string `json:"method"`
}

// jsonrpcResponse is a JSON-RPC 2.0 response sent back to the agent (e.g. for
// permission requests).
type jsonrpcResponse struct {
	JSONRPC string `json:"jsonrpc"`
	ID      int64  `json:"id"`
	Result  any    `json:"result"`
}

// jsonrpcMessage is the JSON-RPC 2.0 envelope for all inbound messages.
type jsonrpcMessage struct {
	JSONRPC string           `json:"jsonrpc"`
	Method  string           `json:"method,omitzero"`
	ID      *json.RawMessage `json:"id,omitzero"`
	Params  json.RawMessage  `json:"params,omitzero"`
	Result  json.RawMessage  `json:"result,omitzero"`
	Error   *jsonrpcError    `json:"error,omitzero"`
}

func (m *jsonrpcMessage) isResponse() bool { return m.ID != nil && m.Method == "" }
func (m *jsonrpcMessage) isRequest() bool  { return m.ID != nil && m.Method != "" }

type jsonrpcError struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
}

// lineProbe extracts routing fields for fast dispatch.
type lineProbe struct {
	Method string           `json:"method,omitzero"`
	ID     *json.RawMessage `json:"id,omitzero"`
}

// ---------- Outbound request types ----------

type initializeParams struct {
	ProtocolVersion    int                `json:"protocolVersion"`
	ClientCapabilities clientCapabilities `json:"clientCapabilities"`
	ClientInfo         clientInfo         `json:"clientInfo"`
}

type clientCapabilities struct {
	Terminal bool `json:"terminal"`
}

type clientInfo struct {
	Name    string `json:"name"`
	Title   string `json:"title"`
	Version string `json:"version"`
}

type sessionNewParams struct {
	Cwd        string `json:"cwd"`
	McpServers []any  `json:"mcpServers"`
}

type sessionLoadParams struct {
	SessionID  string `json:"sessionId"`
	Cwd        string `json:"cwd"`
	McpServers []any  `json:"mcpServers"`
}

// promptContent is a single item in the session/prompt content array.
// Flat union discriminated by Type: "text" or "image".
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
type promptContent struct {
	Type     string `json:"type"`
	Text     string `json:"text,omitzero"`
	Data     string `json:"data,omitzero"`     // Base64 image data.
	MimeType string `json:"mimeType,omitzero"` // e.g. "image/png".
}

type sessionPromptParams struct {
	SessionID string          `json:"sessionId"`
	Prompt    []promptContent `json:"prompt"`
}

type setSessionModelParams struct {
	SessionID string `json:"sessionId"`
	ModelID   string `json:"modelId"`
}

// ---------- Response types ----------

type initializeResult struct {
	ProtocolVersion   int               `json:"protocolVersion"`
	AgentCapabilities agentCapabilities `json:"agentCapabilities,omitzero"`
	AgentInfo         agentInfo         `json:"agentInfo,omitzero"`
}

type agentCapabilities struct {
	PromptCapabilities *promptCapabilities `json:"promptCapabilities,omitzero"`
	LoadSession        bool                `json:"loadSession,omitzero"`
}

type promptCapabilities struct {
	Image           bool `json:"image,omitzero"`
	EmbeddedContext bool `json:"embeddedContext,omitzero"`
}

type agentInfo struct {
	Name    string `json:"name,omitzero"`
	Version string `json:"version,omitzero"`
}

type sessionNewResult struct {
	SessionID string      `json:"sessionId"`
	Models    *modelsInfo `json:"models,omitzero"`
	Modes     *modesInfo  `json:"modes,omitzero"`
}

type modelsInfo struct {
	CurrentModelID  string      `json:"currentModelId,omitzero"`
	AvailableModels []modelInfo `json:"availableModels,omitzero"`
}

type modelInfo struct {
	ModelID string `json:"modelId"`
	Name    string `json:"name,omitzero"`
}

type modesInfo struct {
	CurrentModeID  string     `json:"currentModeId,omitzero"`
	AvailableModes []modeInfo `json:"availableModes,omitzero"`
}

type modeInfo struct {
	ID   string `json:"id"`
	Name string `json:"name,omitzero"`
}

// stopReason is the ACP stop reason returned in a session/prompt response.
type stopReason string

const (
	stopReasonEndTurn   stopReason = "end_turn"
	stopReasonMaxTokens stopReason = "max_tokens"
	stopReasonCancelled stopReason = "cancelled"
	stopReasonRefusal   stopReason = "refusal"
)

func (r stopReason) toFinishReason() genai.FinishReason {
	switch r {
	case stopReasonMaxTokens:
		return genai.FinishedLength
	case stopReasonCancelled, stopReasonRefusal:
		return genai.FinishedContentFilter
	default:
		return genai.FinishedStop
	}
}

// promptResult is the result of a session/prompt response.
type promptResult struct {
	StopReason stopReason   `json:"stopReason,omitzero"`
	Usage      *promptUsage `json:"usage,omitzero"`
}

type promptUsage struct {
	TotalTokens       int64 `json:"totalTokens,omitzero"`
	InputTokens       int64 `json:"inputTokens,omitzero"`
	OutputTokens      int64 `json:"outputTokens,omitzero"`
	ThoughtTokens     int64 `json:"thoughtTokens,omitzero"`
	CachedReadTokens  int64 `json:"cachedReadTokens,omitzero"`
	CachedWriteTokens int64 `json:"cachedWriteTokens,omitzero"`
}

// ---------- Session update types ----------

// sessionUpdateParams holds the params for session/update notifications.
type sessionUpdateParams struct {
	SessionID string          `json:"sessionId"`
	Update    json.RawMessage `json:"update"`
}

// updateProbe extracts the discriminator from a session update.
type updateProbe struct {
	SessionUpdate string `json:"sessionUpdate"`
}

// agentMessageChunkUpdate is a streaming text chunk from the agent.
type agentMessageChunkUpdate struct {
	SessionUpdate string       `json:"sessionUpdate"`
	Content       contentBlock `json:"content"`
}

// agentThoughtChunkUpdate is a streaming reasoning chunk from the agent.
type agentThoughtChunkUpdate struct {
	SessionUpdate string       `json:"sessionUpdate"`
	Content       contentBlock `json:"content"`
}

// contentBlock is a content block in message chunks. Flat union discriminated by Type.
type contentBlock struct {
	Type     string `json:"type"`
	Text     string `json:"text,omitzero"`
	Data     string `json:"data,omitzero"`
	MimeType string `json:"mimeType,omitzero"`
}

// usageUpdateUpdate is a context window / cost update.
type usageUpdateUpdate struct {
	SessionUpdate string    `json:"sessionUpdate"`
	Used          int       `json:"used"`
	Size          int       `json:"size"`
	Cost          usageCost `json:"cost,omitzero"`
}

type usageCost struct {
	Amount   float64 `json:"amount"`
	Currency string  `json:"currency"`
}

// ---------- Permission request ----------

// permissionRequestParams holds params for session/request_permission.
type permissionRequestParams struct {
	SessionID string             `json:"sessionId"`
	ToolCall  json.RawMessage    `json:"toolCall"`
	Options   []permissionOption `json:"options"`
}

// permissionOption is a single option in a permission request.
type permissionOption struct {
	OptionID string `json:"optionId"`
	Kind     string `json:"kind"` // "allow_once", "allow_always", "reject_once".
	Name     string `json:"name"`
}

// permissionResponseResult is the result sent back for a permission request.
type permissionResponseResult struct {
	OptionID string `json:"optionId"`
}
