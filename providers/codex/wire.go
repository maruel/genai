// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Wire types for the Codex CLI app-server JSON-RPC 2.0 protocol.
//
// Type names match the upstream Rust definitions in the Codex repository:
//
//	codex-rs/app-server-protocol/src/protocol/v2.rs     — notification and item structs
//	codex-rs/app-server-protocol/src/protocol/common.rs — method string ↔ struct mapping
//
// Source: https://github.com/openai/codex
package codex

import "encoding/json"

// ============================================================
// Shared types: enums, JSON-RPC envelope, routing probes.
// ============================================================

// Method is a JSON-RPC notification method string for the codex app-server protocol.
type Method string

// JSON-RPC notification method constants for codex app-server.
const (
	MethodThreadStarted     Method = "thread/started"
	MethodTurnStarted       Method = "turn/started"
	MethodTurnCompleted     Method = "turn/completed"
	MethodItemStarted       Method = "item/started"
	MethodItemCompleted     Method = "item/completed"
	MethodItemUpdated       Method = "item/updated"
	MethodItemDelta         Method = "item/agentMessage/delta"
	MethodTokenUsageUpdated Method = "thread/tokenUsage/updated"

	MethodCommandOutputDelta        Method = "item/commandExecution/outputDelta"
	MethodCommandTerminalInteract   Method = "item/commandExecution/terminalInteraction"
	MethodFileChangeOutputDelta     Method = "item/fileChange/outputDelta"
	MethodReasoningSummaryTextDelta Method = "item/reasoning/summaryTextDelta"
	MethodReasoningSummaryPartAdded Method = "item/reasoning/summaryPartAdded"
	MethodReasoningTextDelta        Method = "item/reasoning/textDelta"
	MethodPlanDelta                 Method = "item/plan/delta"
	MethodMcpToolCallProgress       Method = "item/mcpToolCall/progress"
	MethodTurnDiffUpdated           Method = "turn/diff/updated"
	MethodTurnPlanUpdated           Method = "turn/plan/updated"
	MethodThreadStatusChanged       Method = "thread/status/changed"
	MethodThreadNameUpdated         Method = "thread/name/updated"
	MethodModelRerouted             Method = "model/rerouted"
	MethodErrorNotification         Method = "error"

	// Hook lifecycle notifications.
	MethodHookStarted   Method = "hook/started"
	MethodHookCompleted Method = "hook/completed"

	// Server-to-client approval requests (requires permission mode).
	MethodCommandRequestApproval    Method = "item/commandExecution/requestApproval"
	MethodFileChangeRequestApproval Method = "item/fileChange/requestApproval"
	MethodToolRequestUserInput      Method = "item/tool/requestUserInput"
	MethodMcpElicitationRequest     Method = "mcpServer/elicitation/request"
	MethodServerRequestResolved     Method = "serverRequest/resolved"

	// Account and configuration notifications.
	MethodAccountUpdated         Method = "account/updated"
	MethodConfigWarning          Method = "configWarning"
	MethodDeprecationNotice      Method = "deprecationNotice"
	MethodSkillsChanged          Method = "skills/changed"
	MethodMcpStartupStatusUpdate Method = "mcpServer/startupStatus/updated"
)

// ItemType is the item type discriminator for codex app-server items (camelCase).
type ItemType string

// Item type constants (camelCase as emitted by Codex v2).
const (
	ItemTypeUserMessage         ItemType = "userMessage"
	ItemTypeAgentMessage        ItemType = "agentMessage"
	ItemTypePlan                ItemType = "plan"
	ItemTypeReasoning           ItemType = "reasoning"
	ItemTypeCommandExecution    ItemType = "commandExecution"
	ItemTypeFileChange          ItemType = "fileChange"
	ItemTypeMCPToolCall         ItemType = "mcpToolCall"
	ItemTypeWebSearch           ItemType = "webSearch"
	ItemTypeImageView           ItemType = "imageView"
	ItemTypeImageGeneration     ItemType = "imageGeneration"
	ItemTypeContextCompaction   ItemType = "contextCompaction"
	ItemTypeDynamicToolCall     ItemType = "dynamicToolCall"
	ItemTypeCollabAgentToolCall ItemType = "collabAgentToolCall"
	ItemTypeHookPrompt          ItemType = "hookPrompt"
	ItemTypeEnteredReviewMode   ItemType = "enteredReviewMode"
	ItemTypeExitedReviewMode    ItemType = "exitedReviewMode"
)

// JSONRPCMessage is the JSON-RPC 2.0 envelope for codex app-server messages.
// Notifications have Method set and ID nil. Responses have ID set.
type JSONRPCMessage struct {
	JSONRPC string           `json:"jsonrpc"`
	Method  Method           `json:"method,omitzero"`
	ID      *json.RawMessage `json:"id,omitzero"`
	Params  json.RawMessage  `json:"params,omitzero"`
	Result  json.RawMessage  `json:"result,omitzero"`
	Error   *JSONRPCError    `json:"error,omitzero"`
}

// IsResponse returns true if this is a response (has an ID).
func (m *JSONRPCMessage) IsResponse() bool { return m.ID != nil }

// JSONRPCError is a JSON-RPC 2.0 error object.
type JSONRPCError struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
}

// MessageProbe extracts routing fields from a codex app-server line to
// distinguish injected JSON (has "type") from JSON-RPC (has "method"/"id").
type MessageProbe struct {
	Type   string           `json:"type,omitzero"`
	Method Method           `json:"method,omitzero"`
	ID     *json.RawMessage `json:"id,omitzero"`
}

// MethodProbe extracts the method field from a JSON-RPC message.
type MethodProbe struct {
	Method Method `json:"method,omitzero"`
}

// ============================================================
// Input types: requests sent to the codex app-server (stdin).
// ============================================================

// JSONRPCRequest is the envelope for all JSON-RPC 2.0 requests sent to codex.
type JSONRPCRequest struct {
	JSONRPC string `json:"jsonrpc"`
	ID      int64  `json:"id,omitzero"`
	Method  string `json:"method"`
	Params  any    `json:"params,omitzero"`
}

// JSONRPCNotification is a JSON-RPC 2.0 notification (no id, no response expected).
type JSONRPCNotification struct {
	JSONRPC string `json:"jsonrpc"`
	Method  string `json:"method"`
}

// Handshake request params.

// InitializeParams holds the params for the initialize request.
type InitializeParams struct {
	ClientInfo   ClientInfo   `json:"clientInfo"`
	Capabilities Capabilities `json:"capabilities"`
}

// ClientInfo identifies the client in the initialize handshake.
type ClientInfo struct {
	Name    string `json:"name"`
	Title   string `json:"title"`
	Version string `json:"version"`
}

// Capabilities declares client capabilities in the initialize handshake.
type Capabilities struct {
	OptOutNotificationMethods []Method `json:"optOutNotificationMethods"`
}

// Thread management request params.

// ThreadStartParams holds the params for thread/start.
type ThreadStartParams struct {
	Model string `json:"model,omitzero"`
}

// ThreadResumeParams holds the params for thread/resume.
type ThreadResumeParams struct {
	ThreadID string `json:"threadId"`
}

// ThreadStartResult is the result object from a thread/start JSON-RPC response.
type ThreadStartResult struct {
	Thread ThreadStartThread `json:"thread"`
}

// ThreadStartThread is the thread object inside a ThreadStartResult.
type ThreadStartThread struct {
	ID string `json:"id"`
}

// Turn request params.

// ReasoningEffort controls how much reasoning the model performs.
type ReasoningEffort string

// Reasoning effort levels, from least to most compute.
const (
	ReasoningEffortNone    ReasoningEffort = "none"
	ReasoningEffortMinimal ReasoningEffort = "minimal"
	ReasoningEffortLow     ReasoningEffort = "low"
	ReasoningEffortMedium  ReasoningEffort = "medium"
	ReasoningEffortHigh    ReasoningEffort = "high"
	ReasoningEffortXHigh   ReasoningEffort = "xhigh"
)

// TurnStartParams holds the params for turn/start.
type TurnStartParams struct {
	ThreadID string          `json:"threadId"`
	Input    []TurnInput     `json:"input"`
	Effort   ReasoningEffort `json:"effort,omitzero"`
}

// TurnInput is a single item in the turn/start input array.
// Type is "text", "image" (with URL as data URI), "localImage" (with Path),
// "skill" (with Name + Path), or "mention" (with Name + Path).
type TurnInput struct {
	Type string `json:"type"`
	Text string `json:"text,omitzero"`
	URL  string `json:"url,omitzero"`
	Path string `json:"path,omitzero"`
	Name string `json:"name,omitzero"`
}

// Future outbound request params — not yet used but documented for
// upcoming features.

// TurnInterruptParams holds the params for turn/interrupt.
type TurnInterruptParams struct {
	ThreadID string `json:"threadId"`
	TurnID   string `json:"turnId"`
}

// TurnSteerParams holds the params for turn/steer.
type TurnSteerParams struct {
	ThreadID       string      `json:"threadId"`
	Input          []TurnInput `json:"input"`
	ExpectedTurnID string      `json:"expectedTurnId"`
}

// ThreadCompactStartParams holds the params for thread/compact/start.
type ThreadCompactStartParams struct {
	ThreadID string `json:"threadId"`
}

// ThreadRollbackParams holds the params for thread/rollback.
type ThreadRollbackParams struct {
	ThreadID string `json:"threadId"`
}

// ============================================================
// Output types: notifications received from the codex app-server (stdout).
// ============================================================

// Thread lifecycle.

// ThreadStartedNotification holds the params for thread/started notifications.
type ThreadStartedNotification struct {
	Thread Thread `json:"thread"`
}

// Thread describes a thread in thread/started params.
type Thread struct {
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
	Status        ThreadStatus    `json:"status,omitzero"`
	Name          string          `json:"name,omitzero"`
	AgentNickname string          `json:"agentNickname,omitzero"`
	AgentRole     string          `json:"agentRole,omitzero"`
	Turns         json.RawMessage `json:"turns,omitzero"`
}

// ThreadStatus is a tagged union representing thread lifecycle state.
// Variants: notLoaded, idle, systemError, active (with activeFlags).
type ThreadStatus struct {
	Type        string   `json:"type"`
	ActiveFlags []string `json:"activeFlags,omitzero"`
}

// ThreadStatusChangedNotification holds params for thread/status/changed.
type ThreadStatusChangedNotification struct {
	ThreadID string       `json:"threadId"`
	Status   ThreadStatus `json:"status"`
}

// ThreadNameUpdatedNotification holds params for thread/name/updated.
type ThreadNameUpdatedNotification struct {
	ThreadID   string `json:"threadId"`
	ThreadName string `json:"threadName"`
}

// Turn lifecycle.

// TurnStartedNotification holds the params for turn/started notifications.
type TurnStartedNotification struct {
	ThreadID string `json:"threadId"`
	Turn     Turn   `json:"turn"`
}

// TurnCompletedNotification holds the params for turn/completed notifications.
type TurnCompletedNotification struct {
	ThreadID string `json:"threadId"`
	Turn     Turn   `json:"turn"`
}

// Turn describes a turn in turn/started and turn/completed params.
type Turn struct {
	ID     string          `json:"id"`
	Status string          `json:"status"`
	Error  *TurnError      `json:"error,omitzero"`
	Items  json.RawMessage `json:"items,omitzero"`
}

// TurnError describes a turn failure.
type TurnError struct {
	Message           string          `json:"message"`
	CodexErrorInfo    json.RawMessage `json:"codexErrorInfo,omitzero"`
	AdditionalDetails string          `json:"additionalDetails,omitzero"`
}

// TurnDiffUpdatedNotification holds params for turn/diff/updated.
type TurnDiffUpdatedNotification struct {
	ThreadID string          `json:"threadId"`
	TurnID   string          `json:"turnId"`
	Diff     json.RawMessage `json:"diff"`
}

// TurnPlanUpdatedNotification holds params for turn/plan/updated.
type TurnPlanUpdatedNotification struct {
	ThreadID    string          `json:"threadId"`
	TurnID      string          `json:"turnId"`
	Explanation string          `json:"explanation,omitzero"`
	Plan        json.RawMessage `json:"plan,omitzero"`
}

// Item envelope and dispatch.

// ItemNotification holds the params for item/started, item/completed, and
// item/updated notifications. Item is raw JSON dispatched by ItemHeader.Type.
type ItemNotification struct {
	Item     json.RawMessage `json:"item"`
	ThreadID string          `json:"threadId"`
	TurnID   string          `json:"turnId"`
}

// ItemHeader extracts the discriminant fields from a raw item for dispatch.
type ItemHeader struct {
	ID   string   `json:"id"`
	Type ItemType `json:"type"`
}

// AgentMessageDeltaNotification holds the params for item/agentMessage/delta notifications.
type AgentMessageDeltaNotification struct {
	ThreadID string `json:"threadId"`
	TurnID   string `json:"turnId"`
	ItemID   string `json:"itemId"`
	Delta    string `json:"delta"`
}

// Per-item-type structs.

// AgentMessageItem is an agent text response item.
type AgentMessageItem struct {
	ID             string          `json:"id"`
	Type           ItemType        `json:"type"`
	Text           string          `json:"text,omitzero"`
	Phase          string          `json:"phase,omitzero"`
	Status         string          `json:"status,omitzero"`
	MemoryCitation json.RawMessage `json:"memoryCitation,omitzero"`
}

// PlanItem is an agent plan item.
type PlanItem struct {
	ID     string   `json:"id"`
	Type   ItemType `json:"type"`
	Text   string   `json:"text,omitzero"`
	Status string   `json:"status,omitzero"`
}

// ReasoningItem is an agent reasoning/thinking item.
type ReasoningItem struct {
	ID      string          `json:"id"`
	Type    ItemType        `json:"type"`
	Summary []string        `json:"summary,omitzero"`
	Content json.RawMessage `json:"content,omitzero"`
	Status  string          `json:"status,omitzero"`
}

// CommandExecutionItem is a shell command execution item.
// Source indicates the origin: "agent" (default), "userShell",
// "unifiedExecStartup", or "unifiedExecInteraction".
type CommandExecutionItem struct {
	ID               string          `json:"id"`
	Type             ItemType        `json:"type"`
	Command          string          `json:"command,omitzero"`
	Cwd              string          `json:"cwd,omitzero"`
	ProcessID        string          `json:"processId,omitzero"`
	Source           string          `json:"source,omitzero"`
	Status           string          `json:"status,omitzero"`
	CommandActions   json.RawMessage `json:"commandActions,omitzero"`
	AggregatedOutput *string         `json:"aggregatedOutput,omitzero"`
	ExitCode         *int            `json:"exitCode,omitzero"`
	DurationMs       *int64          `json:"durationMs,omitzero"`
}

// FileChangeItem is a file creation/modification/deletion item.
type FileChangeItem struct {
	ID      string             `json:"id"`
	Type    ItemType           `json:"type"`
	Changes []FileUpdateChange `json:"changes,omitzero"`
	Status  string             `json:"status,omitzero"`
}

// McpToolCallItem is an MCP tool call item.
type McpToolCallItem struct {
	ID         string             `json:"id"`
	Type       ItemType           `json:"type"`
	Server     string             `json:"server,omitzero"`
	Tool       string             `json:"tool,omitzero"`
	Status     string             `json:"status,omitzero"`
	Arguments  json.RawMessage    `json:"arguments,omitzero"`
	Result     *McpToolCallResult `json:"result,omitzero"`
	Error      *McpToolCallError  `json:"error,omitzero"`
	DurationMs *int64             `json:"durationMs,omitzero"`
}

// DynamicToolCallItem is a dynamically registered tool call item.
type DynamicToolCallItem struct {
	ID           string          `json:"id"`
	Type         ItemType        `json:"type"`
	Tool         string          `json:"tool,omitzero"`
	Arguments    json.RawMessage `json:"arguments,omitzero"`
	Status       string          `json:"status,omitzero"`
	ContentItems json.RawMessage `json:"contentItems,omitzero"`
	Success      *bool           `json:"success,omitzero"`
	DurationMs   *int64          `json:"durationMs,omitzero"`
}

// CollabAgentToolCallItem is a collaborative multi-agent tool call item.
type CollabAgentToolCallItem struct {
	ID                string          `json:"id"`
	Type              ItemType        `json:"type"`
	Tool              string          `json:"tool,omitzero"`
	Status            string          `json:"status,omitzero"`
	SenderThreadID    string          `json:"senderThreadId,omitzero"`
	ReceiverThreadIDs json.RawMessage `json:"receiverThreadIds,omitzero"`
	Prompt            string          `json:"prompt,omitzero"`
	Model             string          `json:"model,omitzero"`
	ReasoningEffort   string          `json:"reasoningEffort,omitzero"`
	AgentsStates      json.RawMessage `json:"agentsStates,omitzero"`
}

// WebSearchItem is a web search item.
type WebSearchItem struct {
	ID     string           `json:"id"`
	Type   ItemType         `json:"type"`
	Query  string           `json:"query,omitzero"`
	Action *WebSearchAction `json:"action,omitzero"`
	Status string           `json:"status,omitzero"`
}

// ImageViewItem is an image viewing item.
type ImageViewItem struct {
	ID     string   `json:"id"`
	Type   ItemType `json:"type"`
	Path   string   `json:"path,omitzero"`
	Status string   `json:"status,omitzero"`
}

// ImageGenerationItem is an image generation item.
type ImageGenerationItem struct {
	ID            string   `json:"id"`
	Type          ItemType `json:"type"`
	Status        string   `json:"status,omitzero"`
	RevisedPrompt string   `json:"revisedPrompt,omitzero"`
	Result        string   `json:"result,omitzero"`
	SavedPath     string   `json:"savedPath,omitzero"`
}

// HookPromptItem is a hook execution prompt item.
type HookPromptItem struct {
	ID        string          `json:"id"`
	Type      ItemType        `json:"type"`
	Fragments json.RawMessage `json:"fragments,omitzero"`
}

// EnteredReviewModeItem signals the agent entered review mode.
type EnteredReviewModeItem struct {
	ID     string          `json:"id"`
	Type   ItemType        `json:"type"`
	Review json.RawMessage `json:"review,omitzero"`
}

// ExitedReviewModeItem signals the agent exited review mode.
type ExitedReviewModeItem struct {
	ID     string          `json:"id"`
	Type   ItemType        `json:"type"`
	Review json.RawMessage `json:"review,omitzero"`
}

// ContextCompactionItem signals a context window compaction.
type ContextCompactionItem struct {
	ID   string   `json:"id"`
	Type ItemType `json:"type"`
}

// UserMessageItem is a user-submitted message item.
type UserMessageItem struct {
	ID      string          `json:"id"`
	Type    ItemType        `json:"type"`
	Content json.RawMessage `json:"content,omitzero"`
	Status  string          `json:"status,omitzero"`
}

// Item field types.

// FileUpdateChange describes a single file change within a fileChange item.
type FileUpdateChange struct {
	Path string          `json:"path"`
	Kind PatchChangeKind `json:"kind"`
	Diff string          `json:"diff,omitzero"`
}

// PatchChangeKind is the discriminated kind for FileUpdateChange.
type PatchChangeKind struct {
	Type     string  `json:"type"`
	MovePath *string `json:"movePath,omitzero"`
}

// WebSearchAction is the action object within a webSearch item.
type WebSearchAction struct {
	Type    string `json:"type"`
	URL     string `json:"url,omitzero"`
	Pattern string `json:"pattern,omitzero"`
}

// McpToolCallResult holds the result of a successful MCP tool call.
type McpToolCallResult struct {
	Content           []json.RawMessage `json:"content"`
	StructuredContent json.RawMessage   `json:"structuredContent,omitzero"`
}

// McpToolCallError holds the error from a failed MCP tool call.
type McpToolCallError struct {
	Message string `json:"message"`
}

// Token usage.

// ThreadTokenUsageUpdatedNotification holds params for thread/tokenUsage/updated notifications.
type ThreadTokenUsageUpdatedNotification struct {
	ThreadID   string           `json:"threadId"`
	TurnID     string           `json:"turnId"`
	TokenUsage ThreadTokenUsage `json:"tokenUsage"`
}

// ThreadTokenUsage holds cumulative and per-turn token usage for a thread.
type ThreadTokenUsage struct {
	Total              TokenUsageBreakdown `json:"total"`
	Last               TokenUsageBreakdown `json:"last"`
	ModelContextWindow *int64              `json:"modelContextWindow,omitzero"`
}

// TokenUsageBreakdown contains a detailed breakdown of token counts.
type TokenUsageBreakdown struct {
	TotalTokens           int64 `json:"totalTokens"`
	InputTokens           int64 `json:"inputTokens"`
	CachedInputTokens     int64 `json:"cachedInputTokens"`
	OutputTokens          int64 `json:"outputTokens"`
	ReasoningOutputTokens int64 `json:"reasoningOutputTokens"`
}

// Delta notification params.

// CommandExecutionOutputDeltaNotification holds params for item/commandExecution/outputDelta.
type CommandExecutionOutputDeltaNotification struct {
	ThreadID string `json:"threadId"`
	TurnID   string `json:"turnId"`
	ItemID   string `json:"itemId"`
	Delta    string `json:"delta"`
}

// TerminalInteractionNotification holds params for item/commandExecution/terminalInteraction.
type TerminalInteractionNotification struct {
	ThreadID  string `json:"threadId"`
	TurnID    string `json:"turnId"`
	ItemID    string `json:"itemId"`
	ProcessID string `json:"processId"`
	Stdin     string `json:"stdin"`
}

// FileChangeOutputDeltaNotification holds params for item/fileChange/outputDelta.
type FileChangeOutputDeltaNotification struct {
	ThreadID string `json:"threadId"`
	TurnID   string `json:"turnId"`
	ItemID   string `json:"itemId"`
	Delta    string `json:"delta"`
}

// ReasoningSummaryTextDeltaNotification holds params for item/reasoning/summaryTextDelta.
type ReasoningSummaryTextDeltaNotification struct {
	ThreadID     string `json:"threadId"`
	TurnID       string `json:"turnId"`
	ItemID       string `json:"itemId"`
	Delta        string `json:"delta"`
	SummaryIndex int    `json:"summaryIndex"`
}

// ReasoningSummaryPartAddedNotification holds params for item/reasoning/summaryPartAdded.
type ReasoningSummaryPartAddedNotification struct {
	ThreadID     string `json:"threadId"`
	TurnID       string `json:"turnId"`
	ItemID       string `json:"itemId"`
	SummaryIndex int    `json:"summaryIndex"`
}

// ReasoningTextDeltaNotification holds params for item/reasoning/textDelta.
type ReasoningTextDeltaNotification struct {
	ThreadID     string `json:"threadId"`
	TurnID       string `json:"turnId"`
	ItemID       string `json:"itemId"`
	Delta        string `json:"delta"`
	ContentIndex int    `json:"contentIndex"`
}

// PlanDeltaNotification holds params for item/plan/delta.
type PlanDeltaNotification struct {
	ThreadID string `json:"threadId"`
	TurnID   string `json:"turnId"`
	ItemID   string `json:"itemId"`
	Delta    string `json:"delta"`
}

// McpToolCallProgressNotification holds params for item/mcpToolCall/progress.
type McpToolCallProgressNotification struct {
	ThreadID string `json:"threadId"`
	TurnID   string `json:"turnId"`
	ItemID   string `json:"itemId"`
	Message  string `json:"message"`
}

// Model rerouting.

// ModelReroutedNotification holds params for model/rerouted.
type ModelReroutedNotification struct {
	ThreadID  string `json:"threadId"`
	TurnID    string `json:"turnId"`
	FromModel string `json:"fromModel"`
	ToModel   string `json:"toModel"`
	Reason    string `json:"reason,omitzero"`
}

// Model list (handshake response).

// ModelListResult is the result of a model/list request.
type ModelListResult struct {
	Data []ModelInfo `json:"data"`
}

// ModelInfo describes a single model in a model/list result.
type ModelInfo struct {
	ID          string `json:"id"`
	DisplayName string `json:"displayName,omitzero"`
}

// Error notification.

// ErrorNotification holds params for error notifications.
type ErrorNotification struct {
	Error     *TurnError `json:"error"`
	WillRetry bool       `json:"willRetry,omitzero"`
	ThreadID  string     `json:"threadId,omitzero"`
	TurnID    string     `json:"turnId,omitzero"`
}
