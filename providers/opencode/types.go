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

import (
	"encoding/json"

	"github.com/maruel/genai"
)

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
	UpdateAgentMessageChunk UpdateType = "agent_message_chunk"
	UpdateAgentThoughtChunk UpdateType = "agent_thought_chunk"
	UpdateUsageUpdate       UpdateType = "usage_update"
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

// jsonrpcRequest is the envelope for all JSON-RPC 2.0 requests sent to OpenCode.
type jsonrpcRequest struct {
	JSONRPC string `json:"jsonrpc"`
	ID      int64  `json:"id,omitzero"`
	Method  Method `json:"method"`
	Params  any    `json:"params,omitzero"`
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
	JSONRPC string          `json:"jsonrpc"`
	Method  Method          `json:"method,omitzero"`
	ID      json.RawMessage `json:"id,omitzero"`
	Params  json.RawMessage `json:"params,omitzero"`
	Result  json.RawMessage `json:"result,omitzero"`
	Error   *jsonrpcError   `json:"error,omitzero"`
}

// isResponse returns true if this is a response (has ID, no method).
func (m *jsonrpcMessage) isResponse() bool { return m.ID != nil && m.Method == "" }

// isRequest returns true if this is a request from the agent (has both ID and method).
func (m *jsonrpcMessage) isRequest() bool { return m.ID != nil && m.Method != "" }

type jsonrpcError struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
}

// ---------- Routing probes ----------

// lineProbe extracts routing fields for fast dispatch without full unmarshal.
type lineProbe struct {
	Method Method          `json:"method,omitzero"`
	ID     json.RawMessage `json:"id,omitzero"`
}

// updateProbe extracts the discriminator from a session update.
type updateProbe struct {
	SessionUpdate UpdateType `json:"sessionUpdate"`
}

// ============================================================
// Input types: requests sent to OpenCode (stdin).
// ============================================================

// ---------- Handshake request params ----------

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

// ---------- Session management request params ----------

type sessionNewParams struct {
	Cwd        string      `json:"cwd"`
	McpServers []mcpServer `json:"mcpServers"`
}

type sessionLoadParams struct {
	SessionID  string      `json:"sessionId"`
	Cwd        string      `json:"cwd"`
	McpServers []mcpServer `json:"mcpServers"`
}

// mcpServer describes an MCP server to register with the session.
// ACP supports three variants (stdio, http, sse) discriminated by the Type
// field. Only stdio is used by genai (for testing).
type mcpServer struct {
	Type    string        `json:"type,omitzero"` // "http", "sse", or empty for stdio.
	Name    string        `json:"name"`
	Command string        `json:"command,omitzero"` // Stdio only.
	Args    []string      `json:"args,omitzero"`    // Stdio only.
	Env     []envVariable `json:"env,omitzero"`     // Stdio only.
	URL     string        `json:"url,omitzero"`     // HTTP/SSE only.
	Headers []httpHeader  `json:"headers,omitzero"` // HTTP/SSE only.
}

type envVariable struct {
	Name  string `json:"name"`
	Value string `json:"value"`
}

type httpHeader struct {
	Name  string `json:"name"`
	Value string `json:"value"`
}

// ---------- Prompt request params ----------

// promptContent is a single item in the session/prompt content array.
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
type promptContent struct {
	Type     ContentType     `json:"type"`
	Text     string          `json:"text,omitzero"`
	Data     string          `json:"data,omitzero"`     // Base64 image data.
	MimeType string          `json:"mimeType,omitzero"` // e.g. "image/png".
	URI      string          `json:"uri,omitzero"`
	Name     string          `json:"name,omitzero"`
	Resource json.RawMessage `json:"resource,omitzero"` // Embedded resource object.
}

type sessionPromptParams struct {
	SessionID string          `json:"sessionId"`
	Prompt    []promptContent `json:"prompt"`
}

// ---------- Model switching ----------

type setSessionModelParams struct {
	SessionID string `json:"sessionId"`
	ModelID   string `json:"modelId"`
}

// ============================================================
// Output types: notifications and responses received from OpenCode (stdout).
// ============================================================

// ---------- Session update envelope ----------

// sessionUpdateParams holds the params for session/update notifications.
type sessionUpdateParams struct {
	SessionID string          `json:"sessionId"`
	Update    json.RawMessage `json:"update"`
}

// ---------- Content types ----------

// contentBlock is a content block in message chunks. This is a flat union:
// fields are populated depending on Type.
//
//   - ContentText:         Text, Annotations
//   - ContentImage:        Data, MimeType, URI
//   - ContentResource:     Resource
//   - ContentResourceLink: URI, Name, MimeType
type contentBlock struct {
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

// agentMessageChunkUpdate is a streaming text chunk from the agent.
type agentMessageChunkUpdate struct {
	SessionUpdate UpdateType   `json:"sessionUpdate"`
	Content       contentBlock `json:"content"`
}

// agentThoughtChunkUpdate is a streaming reasoning chunk from the agent.
type agentThoughtChunkUpdate struct {
	SessionUpdate UpdateType   `json:"sessionUpdate"`
	Content       contentBlock `json:"content"`
}

// usageUpdateUpdate is a context window / cost update.
type usageUpdateUpdate struct {
	SessionUpdate UpdateType `json:"sessionUpdate"`
	Used          int        `json:"used"`
	Size          int        `json:"size"`
	Cost          usageCost  `json:"cost,omitzero"`
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

// ---------- Response types ----------

// initializeResult is the result of an initialize request.
type initializeResult struct {
	ProtocolVersion   int               `json:"protocolVersion"`
	AgentCapabilities agentCapabilities `json:"agentCapabilities,omitzero"`
	AgentInfo         agentInfo         `json:"agentInfo,omitzero"`
	AuthMethods       json.RawMessage   `json:"authMethods,omitzero"`
}

type agentCapabilities struct {
	PromptCapabilities  promptCapabilities `json:"promptCapabilities,omitzero"`
	LoadSession         bool               `json:"loadSession,omitzero"`
	McpCapabilities     json.RawMessage    `json:"mcpCapabilities,omitzero"`
	SessionCapabilities json.RawMessage    `json:"sessionCapabilities,omitzero"`
}

type promptCapabilities struct {
	Image           bool `json:"image,omitzero"`
	EmbeddedContext bool `json:"embeddedContext,omitzero"`
}

type agentInfo struct {
	Name    string `json:"name,omitzero"`
	Version string `json:"version,omitzero"`
}

// sessionNewResult is the result of a session/new request.
type sessionNewResult struct {
	SessionID string          `json:"sessionId"`
	Models    modelsInfo      `json:"models,omitzero"`
	Modes     modesInfo       `json:"modes,omitzero"`
	Meta      json.RawMessage `json:"_meta,omitzero"`
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
	StopReason stopReason  `json:"stopReason,omitzero"`
	Usage      promptUsage `json:"usage,omitzero"`
}

// promptUsage holds the token usage from a session/prompt response.
type promptUsage struct {
	TotalTokens       int64 `json:"totalTokens,omitzero"`
	InputTokens       int64 `json:"inputTokens,omitzero"`
	OutputTokens      int64 `json:"outputTokens,omitzero"`
	ThoughtTokens     int64 `json:"thoughtTokens,omitzero"`
	CachedReadTokens  int64 `json:"cachedReadTokens,omitzero"`
	CachedWriteTokens int64 `json:"cachedWriteTokens,omitzero"`
}
