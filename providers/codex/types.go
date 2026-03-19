// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package codex

import "encoding/json"

// JSON-RPC 2.0 wire types for the Codex CLI app-server protocol.
//
// Each line on stdin/stdout is a complete JSON object terminated by newline.
// Requests and responses carry an "id" field; notifications carry "method" only.
//
// References:
//   - https://github.com/openai/codex
//   - https://developers.openai.com/codex/cli/reference

// ---------- JSON-RPC notification method constants ----------

const (
	methodThreadStarted             = "thread/started"
	methodTurnStarted               = "turn/started"
	methodTurnCompleted             = "turn/completed"
	methodItemStarted               = "item/started"
	methodItemCompleted             = "item/completed"
	methodItemUpdated               = "item/updated"
	methodItemDelta                 = "item/agentMessage/delta"
	methodTokenUsageUpdated         = "thread/tokenUsage/updated"
	methodCommandOutputDelta        = "item/commandExecution/outputDelta"
	methodCommandTerminalInteract   = "item/commandExecution/terminalInteraction"
	methodFileChangeOutputDelta     = "item/fileChange/outputDelta"
	methodReasoningSummaryTextDelta = "item/reasoning/summaryTextDelta"
	methodReasoningSummaryPartAdded = "item/reasoning/summaryPartAdded"
	methodReasoningTextDelta        = "item/reasoning/textDelta"
	methodPlanDelta                 = "item/plan/delta"
	methodMcpToolCallProgress       = "item/mcpToolCall/progress"
	methodTurnDiffUpdated           = "turn/diff/updated"
	methodTurnPlanUpdated           = "turn/plan/updated"
	methodThreadStatusChanged       = "thread/status/changed"
	methodThreadNameUpdated         = "thread/name/updated"
	methodModelRerouted             = "model/rerouted"
	methodErrorNotification         = "error"
)

// Item type constants (camelCase as emitted by Codex v2).
const (
	itemTypeUserMessage         = "userMessage"
	itemTypeAgentMessage        = "agentMessage"
	itemTypePlan                = "plan"
	itemTypeReasoning           = "reasoning"
	itemTypeCommandExecution    = "commandExecution"
	itemTypeFileChange          = "fileChange"
	itemTypeMCPToolCall         = "mcpToolCall"
	itemTypeWebSearch           = "webSearch"
	itemTypeImageView           = "imageView"
	itemTypeContextCompaction   = "contextCompaction"
	itemTypeDynamicToolCall     = "dynamicToolCall"
	itemTypeCollabAgentToolCall = "collabAgentToolCall"
	itemTypeEnteredReviewMode   = "enteredReviewMode"
	itemTypeExitedReviewMode    = "exitedReviewMode"
)

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
	ThreadID string          `json:"threadId"`
	Input    []turnInput     `json:"input"`
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

// ---------- Thread lifecycle ----------

// threadStartedParams holds params for thread/started notifications.
type threadStartedParams struct {
	Thread threadInfo `json:"thread"`
}

// threadInfo describes a thread in thread/started params.
type threadInfo struct {
	ID            string          `json:"id"`
	CLIVersion    string          `json:"cliVersion,omitzero"`
	CreatedAt     int64           `json:"createdAt,omitzero"`
	CWD           string          `json:"cwd,omitzero"`
	Ephemeral     bool            `json:"ephemeral,omitzero"`
	GitInfo       json.RawMessage `json:"gitInfo,omitzero"`
	ModelProvider string          `json:"modelProvider,omitzero"`
	Path          string          `json:"path,omitzero"`
	Preview       string          `json:"preview,omitzero"`
	Source        string          `json:"source,omitzero"`
	UpdatedAt     int64           `json:"updatedAt,omitzero"`
	Status        threadStatus    `json:"status,omitzero"`
	Name          string          `json:"name,omitzero"`
	AgentNickname string          `json:"agentNickname,omitzero"`
	AgentRole     string          `json:"agentRole,omitzero"`
	Turns         json.RawMessage `json:"turns,omitzero"`
}

// threadStatus is a tagged union representing thread lifecycle state.
// Variants: notLoaded, idle, systemError, active (with activeFlags).
type threadStatus struct {
	Type        string   `json:"type"`
	ActiveFlags []string `json:"activeFlags,omitzero"`
}

// ---------- Turn lifecycle ----------

// turnStartedParams holds params for turn/started notifications.
type turnStartedParams struct {
	ThreadID string   `json:"threadId"`
	Turn     turnInfo `json:"turn"`
}

// turnCompletedParams holds params for turn/completed notifications.
type turnCompletedParams struct {
	ThreadID string   `json:"threadId"`
	Turn     turnInfo `json:"turn"`
}

// turnInfo describes a turn in turn/started and turn/completed params.
type turnInfo struct {
	ID     string          `json:"id"`
	Status string          `json:"status"` // success, failed, interrupted
	Error  *turnError      `json:"error,omitzero"`
	Items  json.RawMessage `json:"items,omitzero"`
}

// turnError describes a turn failure.
type turnError struct {
	Message           string          `json:"message"`
	CodexErrorInfo    json.RawMessage `json:"codexErrorInfo,omitzero"`
	AdditionalDetails string          `json:"additionalDetails,omitzero"`
}

// ---------- Item envelope ----------

// itemParams holds params for item/started, item/completed, and item/updated
// notifications. Item is raw JSON dispatched by itemHeader.Type.
type itemParams struct {
	Item     json.RawMessage `json:"item"`
	ThreadID string          `json:"threadId"`
	TurnID   string          `json:"turnId"`
}

type itemHeader struct {
	ID   string `json:"id"`
	Type string `json:"type"`
}

// ---------- Per-item-type structs ----------

// agentMessageItem is the completed text response item.
type agentMessageItem struct {
	ID     string `json:"id"`
	Type   string `json:"type"`
	Text   string `json:"text,omitzero"`
	Phase  string `json:"phase,omitzero"`
	Status string `json:"status,omitzero"`
}

// planItem is an agent plan item.
type planItem struct {
	ID     string `json:"id"`
	Type   string `json:"type"`
	Text   string `json:"text,omitzero"`
	Status string `json:"status,omitzero"`
}

// reasoningItem is the completed reasoning/thinking item.
type reasoningItem struct {
	ID      string          `json:"id"`
	Type    string          `json:"type"`
	Summary []string        `json:"summary,omitzero"`
	Content json.RawMessage `json:"content,omitzero"`
	Status  string          `json:"status,omitzero"`
}

// commandExecutionItem is a shell command execution item.
type commandExecutionItem struct {
	ID               string          `json:"id"`
	Type             string          `json:"type"`
	Command          string          `json:"command,omitzero"`
	Cwd              string          `json:"cwd,omitzero"`
	ProcessID        string          `json:"processId,omitzero"`
	Status           string          `json:"status,omitzero"`
	CommandActions   json.RawMessage `json:"commandActions,omitzero"`
	AggregatedOutput *string         `json:"aggregatedOutput,omitzero"`
	ExitCode         *int            `json:"exitCode,omitzero"`
	DurationMs       *int64          `json:"durationMs,omitzero"`
}

// fileChangeItem is a file creation/modification/deletion item.
type fileChangeItem struct {
	ID      string             `json:"id"`
	Type    string             `json:"type"`
	Changes []fileUpdateChange `json:"changes,omitzero"`
	Status  string             `json:"status,omitzero"`
}

// fileUpdateChange describes a single file change within a fileChange item.
type fileUpdateChange struct {
	Path string          `json:"path"`
	Kind patchChangeKind `json:"kind"`
	Diff string          `json:"diff,omitzero"`
}

// patchChangeKind is the discriminated kind for fileUpdateChange.
type patchChangeKind struct {
	Type     string  `json:"type"` // add, update, delete
	MovePath *string `json:"movePath,omitzero"`
}

// mcpToolCallItem is an MCP tool call item.
type mcpToolCallItem struct {
	ID         string             `json:"id"`
	Type       string             `json:"type"`
	Server     string             `json:"server,omitzero"`
	Tool       string             `json:"tool,omitzero"`
	Status     string             `json:"status,omitzero"`
	Arguments  json.RawMessage    `json:"arguments,omitzero"`
	Result     *mcpToolCallResult `json:"result,omitzero"`
	Error      *mcpToolCallError  `json:"error,omitzero"`
	DurationMs *int64             `json:"durationMs,omitzero"`
}

// mcpToolCallResult holds the result of a successful MCP tool call.
type mcpToolCallResult struct {
	Content           []json.RawMessage `json:"content"`
	StructuredContent json.RawMessage   `json:"structuredContent,omitzero"`
}

// mcpToolCallError holds the error from a failed MCP tool call.
type mcpToolCallError struct {
	Message string `json:"message"`
}

// dynamicToolCallItem is a dynamically registered tool call item.
type dynamicToolCallItem struct {
	ID           string          `json:"id"`
	Type         string          `json:"type"`
	Tool         string          `json:"tool,omitzero"`
	Arguments    json.RawMessage `json:"arguments,omitzero"`
	Status       string          `json:"status,omitzero"`
	ContentItems json.RawMessage `json:"contentItems,omitzero"`
	Success      *bool           `json:"success,omitzero"`
	DurationMs   *int64          `json:"durationMs,omitzero"`
}

// collabAgentToolCallItem is a collaborative multi-agent tool call item.
type collabAgentToolCallItem struct {
	ID                string          `json:"id"`
	Type              string          `json:"type"`
	Tool              string          `json:"tool,omitzero"`
	Status            string          `json:"status,omitzero"`
	SenderThreadID    string          `json:"senderThreadId,omitzero"`
	ReceiverThreadIDs json.RawMessage `json:"receiverThreadIds,omitzero"`
	Prompt            string          `json:"prompt,omitzero"`
	AgentsStates      json.RawMessage `json:"agentsStates,omitzero"`
}

// webSearchItem is a web search item.
type webSearchItem struct {
	ID     string           `json:"id"`
	Type   string           `json:"type"`
	Query  string           `json:"query,omitzero"`
	Action *webSearchAction `json:"action,omitzero"`
	Status string           `json:"status,omitzero"`
}

// webSearchAction is the action object within a webSearch item.
type webSearchAction struct {
	Type    string `json:"type"`
	URL     string `json:"url,omitzero"`
	Pattern string `json:"pattern,omitzero"`
}

// imageViewItem is an image viewing item.
type imageViewItem struct {
	ID     string `json:"id"`
	Type   string `json:"type"`
	Path   string `json:"path,omitzero"`
	Status string `json:"status,omitzero"`
}

// enteredReviewModeItem signals the agent entered review mode.
type enteredReviewModeItem struct {
	ID     string          `json:"id"`
	Type   string          `json:"type"`
	Review json.RawMessage `json:"review,omitzero"`
}

// exitedReviewModeItem signals the agent exited review mode.
type exitedReviewModeItem struct {
	ID     string          `json:"id"`
	Type   string          `json:"type"`
	Review json.RawMessage `json:"review,omitzero"`
}

// contextCompactionItem signals a context window compaction.
type contextCompactionItem struct {
	ID   string `json:"id"`
	Type string `json:"type"`
}

// userMessageItem is a user-submitted message item.
type userMessageItem struct {
	ID      string          `json:"id"`
	Type    string          `json:"type"`
	Content json.RawMessage `json:"content,omitzero"`
	Status  string          `json:"status,omitzero"`
}

// ---------- Delta notification params ----------

// itemDeltaParams holds params for item/agentMessage/delta notifications.
type itemDeltaParams struct {
	ThreadID string `json:"threadId"`
	TurnID   string `json:"turnId"`
	ItemID   string `json:"itemId"`
	Delta    string `json:"delta"`
}

// commandOutputDeltaParams holds params for item/commandExecution/outputDelta.
type commandOutputDeltaParams struct {
	ThreadID string `json:"threadId"`
	TurnID   string `json:"turnId"`
	ItemID   string `json:"itemId"`
	Delta    string `json:"delta"`
}

// terminalInteractionParams holds params for item/commandExecution/terminalInteraction.
type terminalInteractionParams struct {
	ThreadID  string `json:"threadId"`
	TurnID    string `json:"turnId"`
	ItemID    string `json:"itemId"`
	ProcessID string `json:"processId"`
	Stdin     string `json:"stdin"`
}

// fileChangeOutputDeltaParams holds params for item/fileChange/outputDelta.
type fileChangeOutputDeltaParams struct {
	ThreadID string `json:"threadId"`
	TurnID   string `json:"turnId"`
	ItemID   string `json:"itemId"`
	Delta    string `json:"delta"`
}

// reasoningSummaryTextDeltaParams holds params for item/reasoning/summaryTextDelta.
type reasoningSummaryTextDeltaParams struct {
	ThreadID     string `json:"threadId"`
	TurnID       string `json:"turnId"`
	ItemID       string `json:"itemId"`
	Delta        string `json:"delta"`
	SummaryIndex int    `json:"summaryIndex,omitzero"`
}

// reasoningSummaryPartAddedParams holds params for item/reasoning/summaryPartAdded.
type reasoningSummaryPartAddedParams struct {
	ThreadID     string `json:"threadId"`
	TurnID       string `json:"turnId"`
	ItemID       string `json:"itemId"`
	SummaryIndex int    `json:"summaryIndex"`
}

// reasoningTextDeltaParams holds params for item/reasoning/textDelta.
type reasoningTextDeltaParams struct {
	ThreadID     string `json:"threadId"`
	TurnID       string `json:"turnId"`
	ItemID       string `json:"itemId"`
	Delta        string `json:"delta"`
	ContentIndex int    `json:"contentIndex"`
}

// planDeltaParams holds params for item/plan/delta.
type planDeltaParams struct {
	ThreadID string `json:"threadId"`
	TurnID   string `json:"turnId"`
	ItemID   string `json:"itemId"`
	Delta    string `json:"delta"`
}

// mcpToolCallProgressParams holds params for item/mcpToolCall/progress.
type mcpToolCallProgressParams struct {
	ThreadID string `json:"threadId"`
	TurnID   string `json:"turnId"`
	ItemID   string `json:"itemId"`
	Message  string `json:"message"`
}

// ---------- Other notification params ----------

// turnDiffUpdatedParams holds params for turn/diff/updated.
type turnDiffUpdatedParams struct {
	ThreadID string          `json:"threadId"`
	TurnID   string          `json:"turnId"`
	Diff     json.RawMessage `json:"diff"`
}

// turnPlanUpdatedParams holds params for turn/plan/updated.
type turnPlanUpdatedParams struct {
	ThreadID    string          `json:"threadId"`
	TurnID      string          `json:"turnId"`
	Explanation string          `json:"explanation,omitzero"`
	Plan        json.RawMessage `json:"plan,omitzero"`
}

// threadStatusChangedParams holds params for thread/status/changed.
type threadStatusChangedParams struct {
	ThreadID string       `json:"threadId"`
	Status   threadStatus `json:"status"`
}

// threadNameUpdatedParams holds params for thread/name/updated.
type threadNameUpdatedParams struct {
	ThreadID   string `json:"threadId"`
	ThreadName string `json:"threadName"`
}

// modelReroutedParams holds params for model/rerouted.
type modelReroutedParams struct {
	ThreadID  string `json:"threadId"`
	TurnID    string `json:"turnId"`
	FromModel string `json:"fromModel"`
	ToModel   string `json:"toModel"`
	Reason    string `json:"reason,omitzero"`
}

// ---------- Token usage ----------

// tokenUsageUpdatedParams holds params for thread/tokenUsage/updated.
type tokenUsageUpdatedParams struct {
	ThreadID   string           `json:"threadId"`
	TurnID     string           `json:"turnId"`
	TokenUsage threadTokenUsage `json:"tokenUsage"`
}

type threadTokenUsage struct {
	Total              tokenUsageBreakdown `json:"total"`
	Last               tokenUsageBreakdown `json:"last"`
	ModelContextWindow *int64              `json:"modelContextWindow,omitzero"`
}

type tokenUsageBreakdown struct {
	TotalTokens           int64 `json:"totalTokens"`
	InputTokens           int64 `json:"inputTokens"`
	CachedInputTokens     int64 `json:"cachedInputTokens"`
	OutputTokens          int64 `json:"outputTokens"`
	ReasoningOutputTokens int64 `json:"reasoningOutputTokens"`
}

// ---------- Error notification ----------

// errorNotificationParams holds params for error notifications.
type errorNotificationParams struct {
	Error     *turnError `json:"error"`
	WillRetry bool       `json:"willRetry,omitzero"`
	ThreadID  string     `json:"threadId,omitzero"`
	TurnID    string     `json:"turnId,omitzero"`
}
