// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package codex

import "encoding/json"

// JSON-RPC 2.0 wire types for the Codex CLI app-server protocol.
//
// Each line on stdin/stdout is a complete JSON object terminated by newline.
// Requests and responses carry an "id" field; notifications carry "method" only.

// ---------- Outbound types (stdin) ----------

// jsonrpcRequest is the envelope for all JSON-RPC 2.0 requests sent to codex.
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

type initializeParams struct {
	ClientInfo   clientInfo   `json:"clientInfo"`
	Capabilities capabilities `json:"capabilities"`
}

type clientInfo struct {
	Name    string `json:"name"`
	Title   string `json:"title"`
	Version string `json:"version"`
}

type capabilities struct {
	OptOutNotificationMethods []string `json:"optOutNotificationMethods"`
}

type threadStartParams struct {
	Model string `json:"model,omitzero"`
}

type threadResumeParams struct {
	ThreadID string `json:"threadId"`
}

type turnStartParams struct {
	ThreadID string      `json:"threadId"`
	Input    []turnInput `json:"input"`
	Effort   ReasoningEffort `json:"effort,omitzero"`
}

type turnInput struct {
	Type string `json:"type"`
	Text string `json:"text,omitzero"`
	URL  string `json:"url,omitzero"`
}

// ---------- Inbound envelope (stdout) ----------

// jsonrpcMessage is the JSON-RPC 2.0 envelope for all inbound messages.
type jsonrpcMessage struct {
	JSONRPC string           `json:"jsonrpc"`
	Method  string           `json:"method,omitzero"`
	ID      *json.RawMessage `json:"id,omitzero"`
	Params  json.RawMessage  `json:"params,omitzero"`
	Result  json.RawMessage  `json:"result,omitzero"`
	Error   *jsonrpcError    `json:"error,omitzero"`
}

func (m *jsonrpcMessage) isResponse() bool { return m.ID != nil }

type jsonrpcError struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
}

// lineProbe extracts routing fields for fast dispatch.
type lineProbe struct {
	Method string           `json:"method,omitzero"`
	ID     *json.RawMessage `json:"id,omitzero"`
}

// ---------- Handshake result types ----------

type threadStartResult struct {
	Thread threadStartThread `json:"thread"`
}

type threadStartThread struct {
	ID string `json:"id"`
}

type modelListResult struct {
	Data []modelInfo `json:"data"`
}

type modelInfo struct {
	ID          string `json:"id"`
	DisplayName string `json:"displayName,omitzero"`
}

// ---------- Notification params ----------

// JSON-RPC notification method constants.
const (
	methodThreadStarted             = "thread/started"
	methodTurnStarted               = "turn/started"
	methodTurnCompleted             = "turn/completed"
	methodItemStarted               = "item/started"
	methodItemCompleted             = "item/completed"
	methodItemDelta                 = "item/agentMessage/delta"
	methodTokenUsageUpdated         = "thread/tokenUsage/updated"
	methodReasoningSummaryTextDelta = "item/reasoning/summaryTextDelta"
	methodErrorNotification         = "error"
)

// threadStartedParams holds params for thread/started notifications.
type threadStartedParams struct {
	Thread threadInfo `json:"thread"`
}

type threadInfo struct {
	ID         string `json:"id"`
	CLIVersion string `json:"cliVersion,omitzero"`
	CWD        string `json:"cwd,omitzero"`
}

// turnCompletedParams holds params for turn/completed notifications.
type turnCompletedParams struct {
	ThreadID string   `json:"threadId"`
	Turn     turnInfo `json:"turn"`
}

type turnInfo struct {
	ID     string     `json:"id"`
	Status string     `json:"status"`
	Error  *turnError `json:"error,omitzero"`
}

type turnError struct {
	Message string `json:"message"`
}

// itemParams holds params for item/started and item/completed notifications.
type itemParams struct {
	Item     json.RawMessage `json:"item"`
	ThreadID string          `json:"threadId"`
	TurnID   string          `json:"turnId"`
}

type itemHeader struct {
	ID   string `json:"id"`
	Type string `json:"type"`
}

// agentMessageItem is the completed text response item.
type agentMessageItem struct {
	ID     string `json:"id"`
	Type   string `json:"type"`
	Text   string `json:"text,omitzero"`
	Status string `json:"status,omitzero"`
}

// reasoningItem is the completed reasoning/thinking item.
type reasoningItem struct {
	ID      string   `json:"id"`
	Type    string   `json:"type"`
	Summary []string `json:"summary,omitzero"`
	Status  string   `json:"status,omitzero"`
}

// itemDeltaParams holds params for item/agentMessage/delta notifications.
type itemDeltaParams struct {
	ThreadID string `json:"threadId"`
	TurnID   string `json:"turnId"`
	ItemID   string `json:"itemId"`
	Delta    string `json:"delta"`
}

// reasoningSummaryTextDeltaParams holds params for item/reasoning/summaryTextDelta.
type reasoningSummaryTextDeltaParams struct {
	ThreadID string `json:"threadId"`
	TurnID   string `json:"turnId"`
	ItemID   string `json:"itemId"`
	Delta    string `json:"delta"`
}

// tokenUsageUpdatedParams holds params for thread/tokenUsage/updated.
type tokenUsageUpdatedParams struct {
	ThreadID   string           `json:"threadId"`
	TurnID     string           `json:"turnId"`
	TokenUsage threadTokenUsage `json:"tokenUsage"`
}

type threadTokenUsage struct {
	Total tokenUsageBreakdown `json:"total"`
	Last  tokenUsageBreakdown `json:"last"`
}

type tokenUsageBreakdown struct {
	TotalTokens           int64 `json:"totalTokens"`
	InputTokens           int64 `json:"inputTokens"`
	CachedInputTokens     int64 `json:"cachedInputTokens"`
	OutputTokens          int64 `json:"outputTokens"`
	ReasoningOutputTokens int64 `json:"reasoningOutputTokens"`
}

// errorNotificationParams holds params for error notifications.
type errorNotificationParams struct {
	Error     *turnError `json:"error"`
	WillRetry bool       `json:"willRetry,omitzero"`
}
