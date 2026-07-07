// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Wire types for the Codex CLI app-server JSON-RPC 2.0 protocol.
//
// Type names match the upstream Rust definitions in the Codex repository:
//
//	codex-rs/app-server-protocol/src/protocol/v2/       — notification and item structs
//	codex-rs/app-server-protocol/src/protocol/common.rs — method string ↔ struct mapping
//
// Source: https://github.com/openai/codex

package codex

import (
	"encoding/json"
	"fmt"
)

// ============================================================
// Shared types: enums, JSON-RPC envelope, routing probes.
// ============================================================

// Method is a JSON-RPC notification method string for the codex app-server protocol.
type Method string

// JSON-RPC notification method constants for codex app-server.
const (
	// MethodThreadStarted reports that a thread was created.
	MethodThreadStarted Method = "thread/started"
	// MethodThreadArchived reports that a thread was archived.
	MethodThreadArchived Method = "thread/archived"
	// MethodThreadUnarchived reports that a thread was unarchived.
	MethodThreadUnarchived Method = "thread/unarchived"
	// MethodThreadClosed reports that a thread was closed.
	MethodThreadClosed Method = "thread/closed"
	// MethodTurnStarted reports that a turn started.
	MethodTurnStarted Method = "turn/started"
	// MethodTurnCompleted reports that a turn completed.
	MethodTurnCompleted Method = "turn/completed"
	// MethodItemStarted reports that an item started.
	MethodItemStarted Method = "item/started"
	// MethodItemCompleted reports that an item completed.
	MethodItemCompleted Method = "item/completed"
	// MethodItemDelta streams text deltas for an agent message item.
	MethodItemDelta Method = "item/agentMessage/delta"
	// MethodRawItemCompleted reports an internal raw response item.
	MethodRawItemCompleted Method = "rawResponseItem/completed"
	// MethodTokenUsageUpdated reports updated thread token usage.
	MethodTokenUsageUpdated Method = "thread/tokenUsage/updated"

	// MethodItemGuardianApprovalReviewStarted reports the start of an approval auto-review.
	MethodItemGuardianApprovalReviewStarted Method = "item/autoApprovalReview/started"
	// MethodItemGuardianApprovalReviewCompleted reports the completion of an approval auto-review.
	MethodItemGuardianApprovalReviewCompleted Method = "item/autoApprovalReview/completed"
	// MethodCommandExecOutputDelta streams output from a command/exec session.
	MethodCommandExecOutputDelta Method = "command/exec/outputDelta"
	// MethodProcessOutputDelta streams output from a spawned process.
	MethodProcessOutputDelta Method = "process/outputDelta"
	// MethodProcessExited reports that a spawned process exited.
	MethodProcessExited Method = "process/exited"
	// MethodCommandOutputDelta streams output for a command execution item.
	MethodCommandOutputDelta Method = "item/commandExecution/outputDelta"
	// MethodCommandTerminalInteract reports terminal input for a command execution item.
	MethodCommandTerminalInteract Method = "item/commandExecution/terminalInteraction"
	// MethodFileChangePatchUpdated reports an updated file change patch.
	MethodFileChangePatchUpdated Method = "item/fileChange/patchUpdated"
	// MethodReasoningSummaryTextDelta streams reasoning summary text.
	MethodReasoningSummaryTextDelta Method = "item/reasoning/summaryTextDelta"
	// MethodReasoningSummaryPartAdded reports a new reasoning summary part.
	MethodReasoningSummaryPartAdded Method = "item/reasoning/summaryPartAdded"
	// MethodReasoningTextDelta streams full reasoning text.
	MethodReasoningTextDelta Method = "item/reasoning/textDelta"
	// MethodPlanDelta streams text for a plan item.
	MethodPlanDelta Method = "item/plan/delta"
	// MethodMcpToolCallProgress reports progress for an MCP tool call item.
	MethodMcpToolCallProgress Method = "item/mcpToolCall/progress"
	// MethodTurnDiffUpdated reports the current turn diff.
	MethodTurnDiffUpdated Method = "turn/diff/updated"
	// MethodTurnPlanUpdated reports the current turn plan.
	MethodTurnPlanUpdated Method = "turn/plan/updated"
	// MethodThreadStatusChanged reports a thread status transition.
	MethodThreadStatusChanged Method = "thread/status/changed"
	// MethodThreadSettingsUpdated reports updated thread settings.
	MethodThreadSettingsUpdated Method = "thread/settings/updated"
	// MethodThreadNameUpdated reports an updated thread name.
	MethodThreadNameUpdated Method = "thread/name/updated"
	// MethodThreadGoalUpdated reports an updated thread goal.
	MethodThreadGoalUpdated Method = "thread/goal/updated"
	// MethodThreadGoalCleared reports that the thread goal was cleared.
	MethodThreadGoalCleared Method = "thread/goal/cleared"
	// MethodModelRerouted reports that Codex changed the model for a turn.
	MethodModelRerouted Method = "model/rerouted"
	// MethodModelVerification reports model verification requirements.
	MethodModelVerification Method = "model/verification"
	// MethodTurnModerationMetadata reports moderation metadata for a turn.
	MethodTurnModerationMetadata Method = "turn/moderationMetadata"
	// MethodWarning reports a non-fatal warning.
	MethodWarning Method = "warning"
	// MethodGuardianWarning reports a guardian warning.
	MethodGuardianWarning Method = "guardianWarning"
	// MethodErrorNotification reports an app-server error notification.
	MethodErrorNotification Method = "error"

	// Hook lifecycle notifications.
	// MethodHookStarted reports that a hook started.
	MethodHookStarted Method = "hook/started"
	// MethodHookCompleted reports that a hook completed.
	MethodHookCompleted Method = "hook/completed"

	// Server-to-client approval requests (requires permission mode).
	// MethodCommandRequestApproval requests approval for command execution.
	MethodCommandRequestApproval Method = "item/commandExecution/requestApproval"
	// MethodFileChangeRequestApproval requests approval for a file change.
	MethodFileChangeRequestApproval Method = "item/fileChange/requestApproval"
	// MethodPermissionsRequestApproval requests approval for additional permissions.
	MethodPermissionsRequestApproval Method = "item/permissions/requestApproval"
	// MethodDynamicToolCallRequest requests execution of a client-side dynamic tool.
	MethodDynamicToolCallRequest Method = "item/tool/call"
	// MethodToolRequestUserInput requests user input for a tool call.
	MethodToolRequestUserInput Method = "item/tool/requestUserInput"
	// MethodMcpElicitationRequest requests input for an MCP elicitation.
	MethodMcpElicitationRequest Method = "mcpServer/elicitation/request"
	// MethodServerRequestResolved reports that a server request was resolved.
	MethodServerRequestResolved Method = "serverRequest/resolved"
	// MethodChatgptAuthTokensRefresh requests refreshed ChatGPT auth tokens.
	MethodChatgptAuthTokensRefresh Method = "account/chatgptAuthTokens/refresh"
	// MethodAttestationGenerate requests an attestation result.
	MethodAttestationGenerate Method = "attestation/generate"

	// Account and configuration notifications.
	// MethodAccountUpdated reports updated account data.
	MethodAccountUpdated Method = "account/updated"
	// MethodAccountRateLimitsUpdated reports updated account rate limits.
	MethodAccountRateLimitsUpdated Method = "account/rateLimits/updated"
	// MethodAccountLoginCompleted reports completion of account login.
	MethodAccountLoginCompleted Method = "account/login/completed"
	// MethodAppListUpdated reports an updated MCP app list.
	MethodAppListUpdated Method = "app/list/updated"
	// MethodRemoteControlStatusChanged reports a remote-control status change.
	MethodRemoteControlStatusChanged Method = "remoteControl/status/changed"
	// MethodExternalAgentConfigImportCompleted reports completion of external agent config import.
	MethodExternalAgentConfigImportCompleted Method = "externalAgentConfig/import/completed"
	// MethodFsChanged reports watched filesystem changes.
	MethodFsChanged Method = "fs/changed"
	// MethodConfigWarning reports a configuration warning.
	MethodConfigWarning Method = "configWarning"
	// MethodDeprecationNotice reports a deprecation notice.
	MethodDeprecationNotice Method = "deprecationNotice"
	// MethodSkillsChanged reports that skills changed.
	MethodSkillsChanged Method = "skills/changed"
	// MethodMcpOauthLoginCompleted reports completion of MCP OAuth login.
	MethodMcpOauthLoginCompleted Method = "mcpServer/oauthLogin/completed"
	// MethodMcpServerStatusUpdated reports an MCP server startup status update.
	MethodMcpServerStatusUpdated Method = "mcpServer/startupStatus/updated"

	// Search and realtime notifications.
	// MethodFuzzyFileSearchSessionUpdated reports updated fuzzy file search results.
	MethodFuzzyFileSearchSessionUpdated Method = "fuzzyFileSearch/sessionUpdated"
	// MethodFuzzyFileSearchSessionCompleted reports completion of a fuzzy file search session.
	MethodFuzzyFileSearchSessionCompleted Method = "fuzzyFileSearch/sessionCompleted"
	// MethodThreadRealtimeStarted reports start of a realtime thread session.
	MethodThreadRealtimeStarted Method = "thread/realtime/started"
	// MethodThreadRealtimeItemAdded reports a new realtime item.
	MethodThreadRealtimeItemAdded Method = "thread/realtime/itemAdded"
	// MethodThreadRealtimeTranscriptDelta streams realtime transcript text.
	MethodThreadRealtimeTranscriptDelta Method = "thread/realtime/transcript/delta"
	// MethodThreadRealtimeTranscriptDone reports completion of realtime transcript text.
	MethodThreadRealtimeTranscriptDone Method = "thread/realtime/transcript/done"
	// MethodThreadRealtimeOutputAudioDelta streams realtime output audio.
	MethodThreadRealtimeOutputAudioDelta Method = "thread/realtime/outputAudio/delta"
	// MethodThreadRealtimeSdp carries realtime SDP negotiation data.
	MethodThreadRealtimeSdp Method = "thread/realtime/sdp"
	// MethodThreadRealtimeError reports a realtime session error.
	MethodThreadRealtimeError Method = "thread/realtime/error"
	// MethodThreadRealtimeClosed reports that a realtime session closed.
	MethodThreadRealtimeClosed Method = "thread/realtime/closed"
	// MethodWindowsWorldWritableWarning reports unsafe world-writable Windows paths.
	MethodWindowsWorldWritableWarning Method = "windows/worldWritableWarning"
	// MethodWindowsSandboxSetupCompleted reports completion of Windows sandbox setup.
	MethodWindowsSandboxSetupCompleted Method = "windowsSandbox/setupCompleted"
)

// ItemType is the item type discriminator for codex app-server items (camelCase).
type ItemType string

// Item type constants (camelCase as emitted by Codex v2).
const (
	// ItemTypeUserMessage identifies user-submitted message items.
	ItemTypeUserMessage ItemType = "userMessage"
	// ItemTypeAgentMessage identifies assistant text message items.
	ItemTypeAgentMessage ItemType = "agentMessage"
	// ItemTypePlan identifies plan items.
	ItemTypePlan ItemType = "plan"
	// ItemTypeReasoning identifies reasoning items.
	ItemTypeReasoning ItemType = "reasoning"
	// ItemTypeCommandExecution identifies shell command execution items.
	ItemTypeCommandExecution ItemType = "commandExecution"
	// ItemTypeFileChange identifies file change items.
	ItemTypeFileChange ItemType = "fileChange"
	// ItemTypeMCPToolCall identifies MCP tool call items.
	ItemTypeMCPToolCall ItemType = "mcpToolCall"
	// ItemTypeWebSearch identifies web search items.
	ItemTypeWebSearch ItemType = "webSearch"
	// ItemTypeImageView identifies image view items.
	ItemTypeImageView ItemType = "imageView"
	// ItemTypeImageGeneration identifies image generation items.
	ItemTypeImageGeneration ItemType = "imageGeneration"
	// ItemTypeContextCompaction identifies context compaction items.
	ItemTypeContextCompaction ItemType = "contextCompaction"
	// ItemTypeDynamicToolCall identifies dynamic tool call items.
	ItemTypeDynamicToolCall ItemType = "dynamicToolCall"
	// ItemTypeCollabAgentToolCall identifies collaborative agent tool call items.
	ItemTypeCollabAgentToolCall ItemType = "collabAgentToolCall"
	// ItemTypeHookPrompt identifies hook prompt items.
	ItemTypeHookPrompt ItemType = "hookPrompt"
	// ItemTypeEnteredReviewMode identifies review-mode entry items.
	ItemTypeEnteredReviewMode ItemType = "enteredReviewMode"
	// ItemTypeExitedReviewMode identifies review-mode exit items.
	ItemTypeExitedReviewMode ItemType = "exitedReviewMode"
)

// ThreadStartSource describes why a new thread was started.
type ThreadStartSource string

// Thread start source constants.
const (
	// ThreadStartSourceStartup indicates a thread was started during startup.
	ThreadStartSourceStartup ThreadStartSource = "startup"
	// ThreadStartSourceClear indicates a thread was started by clearing state.
	ThreadStartSourceClear ThreadStartSource = "clear"
)

// ThreadSource classifies the analytics source for a thread.
type ThreadSource string

// Thread source constants.
const (
	// ThreadSourceUser indicates a user-created thread.
	ThreadSourceUser ThreadSource = "user"
	// ThreadSourceSubagent indicates a subagent-created thread.
	ThreadSourceSubagent ThreadSource = "subagent"
	// ThreadSourceMemoryConsolidation indicates a memory consolidation thread.
	ThreadSourceMemoryConsolidation ThreadSource = "memory_consolidation"
)

// ApprovalsReviewer configures where approval requests are routed.
type ApprovalsReviewer string

// Approval reviewer constants.
const (
	// ApprovalsReviewerUser routes approval requests to the user.
	ApprovalsReviewerUser ApprovalsReviewer = "user"
	// ApprovalsReviewerAutoReview routes approval requests through auto-review.
	ApprovalsReviewerAutoReview ApprovalsReviewer = "auto_review"
	// ApprovalsReviewerGuardianSubagent routes approval requests through a guardian subagent.
	ApprovalsReviewerGuardianSubagent ApprovalsReviewer = "guardian_subagent"
)

// SandboxMode controls the sandbox mode for a thread.
type SandboxMode string

// Sandbox mode constants.
const (
	// SandboxModeReadOnly allows read-only filesystem access.
	SandboxModeReadOnly SandboxMode = "read-only"
	// SandboxModeWorkspaceWrite allows writes within the workspace.
	SandboxModeWorkspaceWrite SandboxMode = "workspace-write"
	// SandboxModeDangerFullAccess disables sandbox restrictions.
	SandboxModeDangerFullAccess SandboxMode = "danger-full-access"
)

// ReasoningSummary controls whether reasoning summaries are produced.
type ReasoningSummary string

// Reasoning summary constants.
const (
	// ReasoningSummaryAuto lets Codex choose the reasoning summary behavior.
	ReasoningSummaryAuto ReasoningSummary = "auto"
	// ReasoningSummaryConcise requests concise reasoning summaries.
	ReasoningSummaryConcise ReasoningSummary = "concise"
	// ReasoningSummaryDetailed requests detailed reasoning summaries.
	ReasoningSummaryDetailed ReasoningSummary = "detailed"
	// ReasoningSummaryNone disables reasoning summaries.
	ReasoningSummaryNone ReasoningSummary = "none"
)

// Personality controls assistant personality instructions.
type Personality string

// Personality constants.
const (
	// PersonalityNone disables additional personality instructions.
	PersonalityNone Personality = "none"
	// PersonalityFriendly requests a friendly assistant personality.
	PersonalityFriendly Personality = "friendly"
	// PersonalityPragmatic requests a pragmatic assistant personality.
	PersonalityPragmatic Personality = "pragmatic"
)

// TurnInputType is the type discriminator for turn input entries.
type TurnInputType string

// Turn input type constants.
const (
	// TurnInputTypeText identifies text input.
	TurnInputTypeText TurnInputType = "text"
	// TurnInputTypeImage identifies image input by URL or data URL.
	TurnInputTypeImage TurnInputType = "image"
	// TurnInputTypeLocalImage identifies image input by local path.
	TurnInputTypeLocalImage TurnInputType = "localImage"
	// TurnInputTypeSkill identifies skill input.
	TurnInputTypeSkill TurnInputType = "skill"
	// TurnInputTypeMention identifies mention input.
	TurnInputTypeMention TurnInputType = "mention"
)

// ImageDetail controls how much image detail to send to the model.
type ImageDetail string

// Image detail constants.
const (
	// ImageDetailAuto lets Codex choose image detail.
	ImageDetailAuto ImageDetail = "auto"
	// ImageDetailLow requests low-detail image processing.
	ImageDetailLow ImageDetail = "low"
	// ImageDetailHigh requests high-detail image processing.
	ImageDetailHigh ImageDetail = "high"
	// ImageDetailOriginal requests original image detail.
	ImageDetailOriginal ImageDetail = "original"
)

// TurnStatus is a turn lifecycle status.
type TurnStatus string

// Turn status constants.
const (
	// TurnStatusCompleted indicates the turn completed normally.
	TurnStatusCompleted TurnStatus = "completed"
	// TurnStatusInterrupted indicates the turn was interrupted.
	TurnStatusInterrupted TurnStatus = "interrupted"
	// TurnStatusFailed indicates the turn failed.
	TurnStatusFailed TurnStatus = "failed"
	// TurnStatusInProgress indicates the turn is still running.
	TurnStatusInProgress TurnStatus = "inProgress"
)

// TurnItemsView describes how much item detail is included in a turn payload.
type TurnItemsView string

// Turn items view constants.
const (
	// TurnItemsViewNotLoaded indicates turn items were not loaded.
	TurnItemsViewNotLoaded TurnItemsView = "notLoaded"
	// TurnItemsViewSummary indicates a summary view of turn items.
	TurnItemsViewSummary TurnItemsView = "summary"
	// TurnItemsViewFull indicates full turn item payloads.
	TurnItemsViewFull TurnItemsView = "full"
)

// ThreadStatusType is the type discriminator for thread status.
type ThreadStatusType string

// Thread status type constants.
const (
	// ThreadStatusTypeNotLoaded indicates the thread status was not loaded.
	ThreadStatusTypeNotLoaded ThreadStatusType = "notLoaded"
	// ThreadStatusTypeIdle indicates the thread has no active turn.
	ThreadStatusTypeIdle ThreadStatusType = "idle"
	// ThreadStatusTypeSystemError indicates a system-level thread error.
	ThreadStatusTypeSystemError ThreadStatusType = "systemError"
	// ThreadStatusTypeActive indicates the thread has active work.
	ThreadStatusTypeActive ThreadStatusType = "active"
)

// ThreadActiveFlag describes why a thread is active.
type ThreadActiveFlag string

// Thread active flag constants.
const (
	// ThreadActiveFlagWaitingOnApproval indicates work is blocked on approval.
	ThreadActiveFlagWaitingOnApproval ThreadActiveFlag = "waitingOnApproval"
	// ThreadActiveFlagWaitingOnUserInput indicates work is blocked on user input.
	ThreadActiveFlagWaitingOnUserInput ThreadActiveFlag = "waitingOnUserInput"
)

// MessagePhase classifies an assistant message.
type MessagePhase string

// Message phase constants.
const (
	// MessagePhaseCommentary identifies non-final assistant commentary.
	MessagePhaseCommentary MessagePhase = "commentary"
	// MessagePhaseFinalAnswer identifies final assistant answer text.
	MessagePhaseFinalAnswer MessagePhase = "final_answer"
)

// CommandExecutionSource describes the source of a command execution item.
type CommandExecutionSource string

// Command execution source constants.
const (
	// CommandExecutionSourceAgent indicates a command requested by the agent.
	CommandExecutionSourceAgent CommandExecutionSource = "agent"
	// CommandExecutionSourceUserShell indicates a command from the user's shell.
	CommandExecutionSourceUserShell CommandExecutionSource = "userShell"
	// CommandExecutionSourceUnifiedExecStartup indicates startup command output.
	CommandExecutionSourceUnifiedExecStartup CommandExecutionSource = "unifiedExecStartup"
	// CommandExecutionSourceUnifiedExecInteraction indicates interactive command output.
	CommandExecutionSourceUnifiedExecInteraction CommandExecutionSource = "unifiedExecInteraction"
)

// CommandExecutionStatus is a command execution lifecycle status.
type CommandExecutionStatus string

// Command execution status constants.
const (
	// CommandExecutionStatusInProgress indicates the command is still running.
	CommandExecutionStatusInProgress CommandExecutionStatus = "inProgress"
	// CommandExecutionStatusCompleted indicates the command completed.
	CommandExecutionStatusCompleted CommandExecutionStatus = "completed"
	// CommandExecutionStatusFailed indicates the command failed.
	CommandExecutionStatusFailed CommandExecutionStatus = "failed"
	// CommandExecutionStatusDeclined indicates the command was not approved.
	CommandExecutionStatusDeclined CommandExecutionStatus = "declined"
)

// PatchApplyStatus is a file change lifecycle status.
type PatchApplyStatus string

// Patch apply status constants.
const (
	// PatchApplyStatusInProgress indicates the file change is still applying.
	PatchApplyStatusInProgress PatchApplyStatus = "inProgress"
	// PatchApplyStatusCompleted indicates the file change applied.
	PatchApplyStatusCompleted PatchApplyStatus = "completed"
	// PatchApplyStatusFailed indicates the file change failed.
	PatchApplyStatusFailed PatchApplyStatus = "failed"
	// PatchApplyStatusDeclined indicates the file change was not approved.
	PatchApplyStatusDeclined PatchApplyStatus = "declined"
)

// McpToolCallStatus is an MCP tool call lifecycle status.
type McpToolCallStatus string

// MCP tool call status constants.
const (
	// McpToolCallStatusInProgress indicates the MCP tool call is still running.
	McpToolCallStatusInProgress McpToolCallStatus = "inProgress"
	// McpToolCallStatusCompleted indicates the MCP tool call completed.
	McpToolCallStatusCompleted McpToolCallStatus = "completed"
	// McpToolCallStatusFailed indicates the MCP tool call failed.
	McpToolCallStatusFailed McpToolCallStatus = "failed"
)

// DynamicToolCallStatus is a dynamic tool call lifecycle status.
type DynamicToolCallStatus string

// Dynamic tool call status constants.
const (
	// DynamicToolCallStatusInProgress indicates the dynamic tool call is still running.
	DynamicToolCallStatusInProgress DynamicToolCallStatus = "inProgress"
	// DynamicToolCallStatusCompleted indicates the dynamic tool call completed.
	DynamicToolCallStatusCompleted DynamicToolCallStatus = "completed"
	// DynamicToolCallStatusFailed indicates the dynamic tool call failed.
	DynamicToolCallStatusFailed DynamicToolCallStatus = "failed"
)

// CollabAgentTool is a collaborative agent tool name.
type CollabAgentTool string

// Collaborative agent tool constants.
const (
	// CollabAgentToolSpawnAgent starts a collaborative agent.
	CollabAgentToolSpawnAgent CollabAgentTool = "spawnAgent"
	// CollabAgentToolSendInput sends input to a collaborative agent.
	CollabAgentToolSendInput CollabAgentTool = "sendInput"
	// CollabAgentToolResumeAgent resumes a collaborative agent.
	CollabAgentToolResumeAgent CollabAgentTool = "resumeAgent"
	// CollabAgentToolWait waits for a collaborative agent.
	CollabAgentToolWait CollabAgentTool = "wait"
	// CollabAgentToolCloseAgent closes a collaborative agent.
	CollabAgentToolCloseAgent CollabAgentTool = "closeAgent"
)

// CollabAgentToolCallStatus is a collaborative agent tool call status.
type CollabAgentToolCallStatus string

// Collaborative agent tool call status constants.
const (
	// CollabAgentToolCallStatusInProgress indicates the collaborative tool call is running.
	CollabAgentToolCallStatusInProgress CollabAgentToolCallStatus = "inProgress"
	// CollabAgentToolCallStatusCompleted indicates the collaborative tool call completed.
	CollabAgentToolCallStatusCompleted CollabAgentToolCallStatus = "completed"
	// CollabAgentToolCallStatusFailed indicates the collaborative tool call failed.
	CollabAgentToolCallStatusFailed CollabAgentToolCallStatus = "failed"
)

// CollabAgentStatus is a collaborative agent lifecycle status.
type CollabAgentStatus string

// Collaborative agent status constants.
const (
	// CollabAgentStatusPendingInit indicates the agent is pending initialization.
	CollabAgentStatusPendingInit CollabAgentStatus = "pendingInit"
	// CollabAgentStatusRunning indicates the agent is running.
	CollabAgentStatusRunning CollabAgentStatus = "running"
	// CollabAgentStatusInterrupted indicates the agent was interrupted.
	CollabAgentStatusInterrupted CollabAgentStatus = "interrupted"
	// CollabAgentStatusCompleted indicates the agent completed.
	CollabAgentStatusCompleted CollabAgentStatus = "completed"
	// CollabAgentStatusErrored indicates the agent errored.
	CollabAgentStatusErrored CollabAgentStatus = "errored"
	// CollabAgentStatusShutdown indicates the agent shut down.
	CollabAgentStatusShutdown CollabAgentStatus = "shutdown"
	// CollabAgentStatusNotFound indicates the agent was not found.
	CollabAgentStatusNotFound CollabAgentStatus = "notFound"
)

// PatchChangeKindType is the type discriminator for a file change kind.
type PatchChangeKindType string

// Patch change kind type constants.
const (
	// PatchChangeKindTypeAdd identifies an added file.
	PatchChangeKindTypeAdd PatchChangeKindType = "add"
	// PatchChangeKindTypeDelete identifies a deleted file.
	PatchChangeKindTypeDelete PatchChangeKindType = "delete"
	// PatchChangeKindTypeUpdate identifies an updated file.
	PatchChangeKindTypeUpdate PatchChangeKindType = "update"
)

// WebSearchActionType is the type discriminator for a web search action.
type WebSearchActionType string

// Web search action type constants.
const (
	// WebSearchActionTypeSearch identifies a search query action.
	WebSearchActionTypeSearch WebSearchActionType = "search"
	// WebSearchActionTypeOpenPage identifies an open-page action.
	WebSearchActionTypeOpenPage WebSearchActionType = "openPage"
	// WebSearchActionTypeFindInPage identifies an in-page find action.
	WebSearchActionTypeFindInPage WebSearchActionType = "findInPage"
	// WebSearchActionTypeOther identifies an action outside the typed cases.
	WebSearchActionTypeOther WebSearchActionType = "other"
)

// ModelRerouteReason explains why a model was rerouted.
type ModelRerouteReason string

// Model reroute reason constants.
const (
	// ModelRerouteReasonHighRiskCyberActivity indicates rerouting due to high-risk cyber activity.
	ModelRerouteReasonHighRiskCyberActivity ModelRerouteReason = "high_risk_cyber_activity"
)

// ModelVerification describes a model verification requirement.
type ModelVerification string

// Model verification constants.
const (
	// ModelVerificationTrustedAccessForCyber indicates trusted access for cyber is required.
	ModelVerificationTrustedAccessForCyber ModelVerification = "trusted_access_for_cyber"
)

// InputModality is a model input modality.
type InputModality string

// Input modality constants.
const (
	// InputModalityText indicates text input support.
	InputModalityText InputModality = "text"
	// InputModalityImage indicates image input support.
	InputModalityImage InputModality = "image"
)

// AutoReviewDecisionSource identifies what produced an auto-review decision.
type AutoReviewDecisionSource string

// Auto-review decision source constants.
const (
	// AutoReviewDecisionSourceAgent indicates the agent made the auto-review decision.
	AutoReviewDecisionSourceAgent AutoReviewDecisionSource = "agent"
)

// McpServerStartupState is an MCP server startup lifecycle state.
type McpServerStartupState string

// MCP server startup state constants.
const (
	// McpServerStartupStateStarting indicates the MCP server is starting.
	McpServerStartupStateStarting McpServerStartupState = "starting"
	// McpServerStartupStateReady indicates the MCP server is ready.
	McpServerStartupStateReady McpServerStartupState = "ready"
	// McpServerStartupStateFailed indicates MCP server startup failed.
	McpServerStartupStateFailed McpServerStartupState = "failed"
	// McpServerStartupStateCancelled indicates MCP server startup was cancelled.
	McpServerStartupStateCancelled McpServerStartupState = "cancelled"
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
	JSONRPC string          `json:"jsonrpc"`
	ID      int64           `json:"id,omitzero"`
	Method  string          `json:"method"`
	Params  json.RawMessage `json:"params,omitzero"`
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
	Model                 string                     `json:"model,omitzero"`
	ModelProvider         string                     `json:"modelProvider,omitzero"`
	ServiceTier           *string                    `json:"serviceTier,omitzero"`
	Cwd                   string                     `json:"cwd,omitzero"`
	RuntimeWorkspaceRoots []string                   `json:"runtimeWorkspaceRoots,omitzero"`
	ApprovalPolicy        json.RawMessage            `json:"approvalPolicy,omitzero"`
	ApprovalsReviewer     ApprovalsReviewer          `json:"approvalsReviewer,omitzero"`
	Sandbox               SandboxMode                `json:"sandbox,omitzero"`
	Permissions           string                     `json:"permissions,omitzero"`
	Config                map[string]json.RawMessage `json:"config,omitzero"`
	ServiceName           string                     `json:"serviceName,omitzero"`
	BaseInstructions      string                     `json:"baseInstructions,omitzero"`
	DeveloperInstructions string                     `json:"developerInstructions,omitzero"`
	Personality           Personality                `json:"personality,omitzero"`
	Ephemeral             *bool                      `json:"ephemeral,omitzero"`
	SessionStartSource    ThreadStartSource          `json:"sessionStartSource,omitzero"`
	ThreadSource          ThreadSource               `json:"threadSource,omitzero"`
	Environments          []TurnEnvironmentParams    `json:"environments,omitzero"`
	DynamicTools          []DynamicToolSpec          `json:"dynamicTools,omitzero"`
	ExperimentalRawEvents bool                       `json:"experimentalRawEvents,omitzero"`
}

// ThreadResumeParams holds the params for thread/resume.
type ThreadResumeParams struct {
	ThreadID              string                     `json:"threadId"`
	History               json.RawMessage            `json:"history,omitzero"`
	Path                  string                     `json:"path,omitzero"`
	Model                 string                     `json:"model,omitzero"`
	ModelProvider         string                     `json:"modelProvider,omitzero"`
	ServiceTier           *string                    `json:"serviceTier,omitzero"`
	Cwd                   string                     `json:"cwd,omitzero"`
	RuntimeWorkspaceRoots []string                   `json:"runtimeWorkspaceRoots,omitzero"`
	ApprovalPolicy        json.RawMessage            `json:"approvalPolicy,omitzero"`
	ApprovalsReviewer     ApprovalsReviewer          `json:"approvalsReviewer,omitzero"`
	Sandbox               SandboxMode                `json:"sandbox,omitzero"`
	Permissions           string                     `json:"permissions,omitzero"`
	Config                map[string]json.RawMessage `json:"config,omitzero"`
	BaseInstructions      string                     `json:"baseInstructions,omitzero"`
	DeveloperInstructions string                     `json:"developerInstructions,omitzero"`
	Personality           Personality                `json:"personality,omitzero"`
	ExcludeTurns          bool                       `json:"excludeTurns,omitzero"`
	InitialTurnsPage      json.RawMessage            `json:"initialTurnsPage,omitzero"`
}

// ThreadStartResult is the result object from a thread/start JSON-RPC response.
type ThreadStartResult struct {
	Thread                  Thread                   `json:"thread"`
	Model                   string                   `json:"model,omitzero"`
	ModelProvider           string                   `json:"modelProvider,omitzero"`
	ServiceTier             *string                  `json:"serviceTier,omitzero"`
	Cwd                     string                   `json:"cwd,omitzero"`
	RuntimeWorkspaceRoots   []string                 `json:"runtimeWorkspaceRoots,omitzero"`
	InstructionSources      []string                 `json:"instructionSources,omitzero"`
	ApprovalPolicy          json.RawMessage          `json:"approvalPolicy,omitzero"`
	ApprovalsReviewer       ApprovalsReviewer        `json:"approvalsReviewer,omitzero"`
	Sandbox                 json.RawMessage          `json:"sandbox,omitzero"`
	ActivePermissionProfile *ActivePermissionProfile `json:"activePermissionProfile,omitzero"`
	ReasoningEffort         *ReasoningEffort         `json:"reasoningEffort,omitzero"`
	InitialTurnsPage        json.RawMessage          `json:"initialTurnsPage,omitzero"`
}

// Turn request params.

// ReasoningEffort controls how much reasoning the model performs.
type ReasoningEffort string

// Validate implements genai.ProviderOption.
func (p ReasoningEffort) Validate() error {
	switch p {
	case ReasoningEffortNone, ReasoningEffortMinimal, ReasoningEffortLow, ReasoningEffortMedium, ReasoningEffortHigh, ReasoningEffortXHigh:
		return nil
	default:
		return fmt.Errorf("invalid reasoning effort %q; use one of the ReasoningEffort* constants", string(p))
	}
}

// Reasoning effort levels, from least to most compute.
const (
	// ReasoningEffortNone disables model reasoning effort.
	ReasoningEffortNone ReasoningEffort = "none"
	// ReasoningEffortMinimal requests minimal model reasoning effort.
	ReasoningEffortMinimal ReasoningEffort = "minimal"
	// ReasoningEffortLow requests low model reasoning effort.
	ReasoningEffortLow ReasoningEffort = "low"
	// ReasoningEffortMedium requests medium model reasoning effort.
	ReasoningEffortMedium ReasoningEffort = "medium"
	// ReasoningEffortHigh requests high model reasoning effort.
	ReasoningEffortHigh ReasoningEffort = "high"
	// ReasoningEffortXHigh requests extra-high model reasoning effort.
	ReasoningEffortXHigh ReasoningEffort = "xhigh"
)

// TurnStartParams holds the params for turn/start.
type TurnStartParams struct {
	ThreadID                   string                            `json:"threadId"`
	ClientUserMessageID        string                            `json:"clientUserMessageId,omitzero"`
	Input                      []TurnInput                       `json:"input"`
	ResponsesAPIClientMetadata map[string]string                 `json:"responsesapiClientMetadata,omitzero"`
	AdditionalContext          map[string]AdditionalContextEntry `json:"additionalContext,omitzero"`
	Environments               []TurnEnvironmentParams           `json:"environments,omitzero"`
	Cwd                        string                            `json:"cwd,omitzero"`
	RuntimeWorkspaceRoots      []string                          `json:"runtimeWorkspaceRoots,omitzero"`
	ApprovalPolicy             json.RawMessage                   `json:"approvalPolicy,omitzero"`
	ApprovalsReviewer          ApprovalsReviewer                 `json:"approvalsReviewer,omitzero"`
	SandboxPolicy              json.RawMessage                   `json:"sandboxPolicy,omitzero"`
	Permissions                string                            `json:"permissions,omitzero"`
	Model                      string                            `json:"model,omitzero"`
	ServiceTier                *string                           `json:"serviceTier,omitzero"`
	Effort                     ReasoningEffort                   `json:"effort,omitzero"`
	Summary                    ReasoningSummary                  `json:"summary,omitzero"`
	Personality                Personality                       `json:"personality,omitzero"`
	OutputSchema               json.RawMessage                   `json:"outputSchema,omitzero"`
	CollaborationMode          json.RawMessage                   `json:"collaborationMode,omitzero"`
}

// TurnInput is a single item in the turn/start input array.
// Type is "text", "image" (with URL as data URI), "localImage" (with Path),
// "skill" (with Name + Path), or "mention" (with Name + Path).
type TurnInput struct {
	Type         TurnInputType `json:"type"`
	Text         string        `json:"text,omitzero"`
	TextElements []TextElement `json:"textElements,omitzero"`
	Detail       ImageDetail   `json:"detail,omitzero"`
	URL          string        `json:"url,omitzero"`
	Path         string        `json:"path,omitzero"`
	Name         string        `json:"name,omitzero"`
}

// UserInput is a single inbound user input content block in a userMessage item.
type UserInput struct {
	Type         TurnInputType `json:"type"`
	Text         string        `json:"text,omitzero"`
	TextElements []TextElement `json:"text_elements,omitzero"`
	Detail       ImageDetail   `json:"detail,omitzero"`
	URL          string        `json:"url,omitzero"`
	Path         string        `json:"path,omitzero"`
}

// TextElement is a UI-defined span within a user input text block.
type TextElement struct {
	ByteRange   ByteRange `json:"byteRange"`
	Placeholder *string   `json:"placeholder,omitzero"`
}

// ByteRange identifies a byte range in a parent text buffer.
type ByteRange struct {
	Start int `json:"start"`
	End   int `json:"end"`
}

// AdditionalContextKind identifies the trust level for turn context fragments.
type AdditionalContextKind string

// Additional context kind constants.
const (
	AdditionalContextKindUntrusted   AdditionalContextKind = "untrusted"
	AdditionalContextKindApplication AdditionalContextKind = "application"
)

// AdditionalContextEntry is one client-provided context fragment.
type AdditionalContextEntry struct {
	Value string                `json:"value"`
	Kind  AdditionalContextKind `json:"kind"`
}

// TurnEnvironmentParams selects a turn or thread environment.
type TurnEnvironmentParams struct {
	EnvironmentID string `json:"environmentId"`
	Cwd           string `json:"cwd"`
}

// DynamicToolSpec describes a dynamically registered tool.
type DynamicToolSpec struct {
	Namespace    string          `json:"namespace,omitzero"`
	Name         string          `json:"name"`
	Description  string          `json:"description"`
	InputSchema  json.RawMessage `json:"inputSchema"`
	DeferLoading bool            `json:"deferLoading,omitzero"`
}

// ActivePermissionProfile identifies the permission profile active for a thread.
type ActivePermissionProfile struct {
	ID      string  `json:"id"`
	Extends *string `json:"extends,omitzero"`
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
	ThreadID                   string                            `json:"threadId"`
	ClientUserMessageID        string                            `json:"clientUserMessageId,omitzero"`
	Input                      []TurnInput                       `json:"input"`
	ResponsesAPIClientMetadata map[string]string                 `json:"responsesapiClientMetadata,omitzero"`
	AdditionalContext          map[string]AdditionalContextEntry `json:"additionalContext,omitzero"`
	ExpectedTurnID             string                            `json:"expectedTurnId"`
}

// ThreadCompactStartParams holds the params for thread/compact/start.
type ThreadCompactStartParams struct {
	ThreadID string `json:"threadId"`
}

// ThreadRollbackParams holds the params for thread/rollback.
type ThreadRollbackParams struct {
	ThreadID string `json:"threadId"`
	NumTurns uint32 `json:"numTurns"`
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
	ID             string          `json:"id"`
	SessionID      string          `json:"sessionId,omitzero"`
	ForkedFromID   *string         `json:"forkedFromId,omitzero"`
	ParentThreadID *string         `json:"parentThreadId,omitzero"`
	CLIVersion     string          `json:"cliVersion,omitzero"`
	CreatedAt      int64           `json:"createdAt,omitzero"`
	CWD            string          `json:"cwd,omitzero"`
	Ephemeral      bool            `json:"ephemeral,omitzero"`
	GitInfo        *GitInfo        `json:"gitInfo,omitzero"`
	ModelProvider  string          `json:"modelProvider,omitzero"`
	Path           *string         `json:"path,omitzero"`
	Preview        string          `json:"preview,omitzero"`
	Source         json.RawMessage `json:"source,omitzero"`
	ThreadSource   ThreadSource    `json:"threadSource,omitzero"`
	UpdatedAt      int64           `json:"updatedAt,omitzero"`
	Status         ThreadStatus    `json:"status,omitzero"`
	Name           string          `json:"name,omitzero"`
	AgentNickname  string          `json:"agentNickname,omitzero"`
	AgentRole      string          `json:"agentRole,omitzero"`
	Turns          []Turn          `json:"turns,omitzero"`
}

// GitInfo is optional Git metadata captured for a thread.
type GitInfo struct {
	SHA       *string `json:"sha,omitzero"`
	Branch    *string `json:"branch,omitzero"`
	OriginURL *string `json:"originUrl,omitzero"`
}

// ThreadStatus is a tagged union representing thread lifecycle state.
// Variants: notLoaded, idle, systemError, active (with activeFlags).
type ThreadStatus struct {
	Type        ThreadStatusType   `json:"type"`
	ActiveFlags []ThreadActiveFlag `json:"activeFlags,omitzero"`
}

// ThreadStatusChangedNotification holds params for thread/status/changed.
type ThreadStatusChangedNotification struct {
	ThreadID string       `json:"threadId"`
	Status   ThreadStatus `json:"status"`
}

// ThreadNameUpdatedNotification holds params for thread/name/updated.
type ThreadNameUpdatedNotification struct {
	ThreadID   string  `json:"threadId"`
	ThreadName *string `json:"threadName,omitzero"`
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

// ItemStartedNotification holds params for item/started notifications.
type ItemStartedNotification struct {
	// Item is the raw upstream ThreadItem tagged union payload.
	Item json.RawMessage `json:"item"`
	// ThreadID is the thread containing the item.
	ThreadID string `json:"threadId"`
	// TurnID is the turn containing the item.
	TurnID string `json:"turnId"`
	// StartedAtMs is the Unix timestamp in milliseconds when the item started.
	StartedAtMs int64 `json:"startedAtMs"`
}

// ItemCompletedNotification holds params for item/completed notifications.
type ItemCompletedNotification struct {
	// Item is the raw upstream ThreadItem tagged union payload.
	Item json.RawMessage `json:"item"`
	// ThreadID is the thread containing the item.
	ThreadID string `json:"threadId"`
	// TurnID is the turn containing the item.
	TurnID string `json:"turnId"`
	// CompletedAtMs is the Unix timestamp in milliseconds when the item completed.
	CompletedAtMs int64 `json:"completedAtMs"`
}

// Turn describes a turn in turn/started and turn/completed params.
type Turn struct {
	ID          string            `json:"id"`
	Items       []json.RawMessage `json:"items,omitzero"`
	ItemsView   TurnItemsView     `json:"itemsView,omitzero"`
	Status      TurnStatus        `json:"status"`
	Error       *TurnError        `json:"error,omitzero"`
	StartedAt   *int64            `json:"startedAt,omitzero"`
	CompletedAt *int64            `json:"completedAt,omitzero"`
	DurationMs  *int64            `json:"durationMs,omitzero"`
}

// TurnError describes a turn failure.
type TurnError struct {
	Message           string          `json:"message"`
	CodexErrorInfo    json.RawMessage `json:"codexErrorInfo,omitzero"`
	AdditionalDetails string          `json:"additionalDetails,omitzero"`
}

// TurnDiffUpdatedNotification holds params for turn/diff/updated.
type TurnDiffUpdatedNotification struct {
	ThreadID string `json:"threadId"`
	TurnID   string `json:"turnId"`
	Diff     string `json:"diff"`
}

// TurnPlanUpdatedNotification holds params for turn/plan/updated.
type TurnPlanUpdatedNotification struct {
	ThreadID    string         `json:"threadId"`
	TurnID      string         `json:"turnId"`
	Explanation *string        `json:"explanation,omitzero"`
	Plan        []TurnPlanStep `json:"plan,omitzero"`
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
	Phase          MessagePhase    `json:"phase,omitzero"`
	MemoryCitation *MemoryCitation `json:"memoryCitation,omitzero"`
}

// PlanItem is an agent plan item.
type PlanItem struct {
	ID   string   `json:"id"`
	Type ItemType `json:"type"`
	Text string   `json:"text,omitzero"`
}

// ReasoningItem is an agent reasoning/thinking item.
type ReasoningItem struct {
	ID      string   `json:"id"`
	Type    ItemType `json:"type"`
	Summary []string `json:"summary,omitzero"`
	Content []string `json:"content,omitzero"`
}

// CommandExecutionItem is a shell command execution item.
// Source indicates the origin: "agent" (default), "userShell",
// "unifiedExecStartup", or "unifiedExecInteraction".
type CommandExecutionItem struct {
	ID               string                 `json:"id"`
	Type             ItemType               `json:"type"`
	Command          string                 `json:"command,omitzero"`
	Cwd              string                 `json:"cwd,omitzero"`
	ProcessID        string                 `json:"processId,omitzero"`
	Source           CommandExecutionSource `json:"source,omitzero"`
	Status           CommandExecutionStatus `json:"status,omitzero"`
	CommandActions   []CommandAction        `json:"commandActions,omitzero"`
	AggregatedOutput *string                `json:"aggregatedOutput,omitzero"`
	ExitCode         *int                   `json:"exitCode,omitzero"`
	DurationMs       *int64                 `json:"durationMs,omitzero"`
}

// FileChangeItem is a file creation/modification/deletion item.
type FileChangeItem struct {
	ID      string             `json:"id"`
	Type    ItemType           `json:"type"`
	Changes []FileUpdateChange `json:"changes,omitzero"`
	Status  PatchApplyStatus   `json:"status,omitzero"`
}

// McpToolCallItem is an MCP tool call item.
type McpToolCallItem struct {
	ID                string             `json:"id"`
	Type              ItemType           `json:"type"`
	Server            string             `json:"server,omitzero"`
	Tool              string             `json:"tool,omitzero"`
	Status            McpToolCallStatus  `json:"status,omitzero"`
	Arguments         json.RawMessage    `json:"arguments,omitzero"`
	McpAppResourceURI string             `json:"mcpAppResourceUri,omitzero"`
	PluginID          string             `json:"pluginId,omitzero"`
	Result            *McpToolCallResult `json:"result,omitzero"`
	Error             *McpToolCallError  `json:"error,omitzero"`
	DurationMs        *int64             `json:"durationMs,omitzero"`
}

// DynamicToolCallItem is a dynamically registered tool call item.
type DynamicToolCallItem struct {
	ID           string                             `json:"id"`
	Type         ItemType                           `json:"type"`
	Namespace    string                             `json:"namespace,omitzero"`
	Tool         string                             `json:"tool,omitzero"`
	Arguments    json.RawMessage                    `json:"arguments,omitzero"`
	Status       DynamicToolCallStatus              `json:"status,omitzero"`
	ContentItems []DynamicToolCallOutputContentItem `json:"contentItems,omitzero"`
	Success      *bool                              `json:"success,omitzero"`
	DurationMs   *int64                             `json:"durationMs,omitzero"`
}

// CollabAgentToolCallItem is a collaborative multi-agent tool call item.
type CollabAgentToolCallItem struct {
	ID                string                      `json:"id"`
	Type              ItemType                    `json:"type"`
	Tool              CollabAgentTool             `json:"tool,omitzero"`
	Status            CollabAgentToolCallStatus   `json:"status,omitzero"`
	SenderThreadID    string                      `json:"senderThreadId,omitzero"`
	ReceiverThreadIDs []string                    `json:"receiverThreadIds,omitzero"`
	Prompt            string                      `json:"prompt,omitzero"`
	Model             string                      `json:"model,omitzero"`
	ReasoningEffort   ReasoningEffort             `json:"reasoningEffort,omitzero"`
	AgentsStates      map[string]CollabAgentState `json:"agentsStates,omitzero"`
}

// CollabAgentState describes the state of a collaborative agent.
type CollabAgentState struct {
	Status  CollabAgentStatus `json:"status"`
	Message *string           `json:"message,omitzero"`
}

// MemoryCitation describes memory citations attached to an agent message.
type MemoryCitation struct {
	Entries   []MemoryCitationEntry `json:"entries"`
	ThreadIDs []string              `json:"threadIds"`
}

// MemoryCitationEntry describes one memory citation span.
type MemoryCitationEntry struct {
	Path      string `json:"path"`
	LineStart uint32 `json:"lineStart"`
	LineEnd   uint32 `json:"lineEnd"`
	Note      string `json:"note"`
}

// WebSearchItem is a web search item.
type WebSearchItem struct {
	ID     string           `json:"id"`
	Type   ItemType         `json:"type"`
	Query  string           `json:"query,omitzero"`
	Action *WebSearchAction `json:"action,omitzero"`
}

// ImageViewItem is an image viewing item.
type ImageViewItem struct {
	ID   string   `json:"id"`
	Type ItemType `json:"type"`
	Path string   `json:"path,omitzero"`
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
	ID        string               `json:"id"`
	Type      ItemType             `json:"type"`
	Fragments []HookPromptFragment `json:"fragments,omitzero"`
}

// EnteredReviewModeItem signals the agent entered review mode.
type EnteredReviewModeItem struct {
	ID     string   `json:"id"`
	Type   ItemType `json:"type"`
	Review string   `json:"review,omitzero"`
}

// ExitedReviewModeItem signals the agent exited review mode.
type ExitedReviewModeItem struct {
	ID     string   `json:"id"`
	Type   ItemType `json:"type"`
	Review string   `json:"review,omitzero"`
}

// ContextCompactionThreadItem signals a context window compaction.
type ContextCompactionThreadItem struct {
	ID   string   `json:"id"`
	Type ItemType `json:"type"`
}

// HookPromptFragment is one hook execution prompt fragment.
type HookPromptFragment struct {
	Text      string `json:"text"`
	HookRunID string `json:"hookRunId"`
}

// UserMessageItem is a user-submitted message item.
type UserMessageItem struct {
	ID       string      `json:"id"`
	Type     ItemType    `json:"type"`
	ClientID *string     `json:"clientId,omitzero"`
	Content  []UserInput `json:"content,omitzero"`
}

// Item field types.

// CommandActionType identifies a parsed command action variant.
type CommandActionType string

// Command action type constants.
const (
	CommandActionTypeRead      CommandActionType = "read"
	CommandActionTypeListFiles CommandActionType = "listFiles"
	CommandActionTypeSearch    CommandActionType = "search"
	CommandActionTypeUnknown   CommandActionType = "unknown"
)

// CommandAction describes one best-effort parsed shell command action.
type CommandAction struct {
	Type    CommandActionType `json:"type"`
	Command string            `json:"command"`
	Name    string            `json:"name,omitzero"`
	Path    string            `json:"path,omitzero"`
	Query   string            `json:"query,omitzero"`
}

// FileUpdateChange describes a single file change within a fileChange item.
type FileUpdateChange struct {
	Path string          `json:"path"`
	Kind PatchChangeKind `json:"kind"`
	Diff string          `json:"diff,omitzero"`
}

// DynamicToolCallOutputContentItemType identifies a dynamic tool output content item.
type DynamicToolCallOutputContentItemType string

// Dynamic tool output content item type constants.
const (
	DynamicToolCallOutputContentItemTypeInputText  DynamicToolCallOutputContentItemType = "inputText"
	DynamicToolCallOutputContentItemTypeInputImage DynamicToolCallOutputContentItemType = "inputImage"
)

// DynamicToolCallOutputContentItem is one dynamic tool output content item.
type DynamicToolCallOutputContentItem struct {
	Type     DynamicToolCallOutputContentItemType `json:"type"`
	Text     string                               `json:"text,omitzero"`
	ImageURL string                               `json:"imageUrl,omitzero"`
}

// PatchChangeKind is the discriminated kind for FileUpdateChange.
type PatchChangeKind struct {
	Type     PatchChangeKindType `json:"type"`
	MovePath *string             `json:"movePath,omitzero"`
}

// WebSearchAction is the action object within a webSearch item.
type WebSearchAction struct {
	Type    WebSearchActionType `json:"type"`
	Query   string              `json:"query,omitzero"`
	Queries []string            `json:"queries,omitzero"`
	URL     string              `json:"url,omitzero"`
	Pattern string              `json:"pattern,omitzero"`
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

// TurnPlanStepStatus is a plan step lifecycle state.
type TurnPlanStepStatus string

// Turn plan step status constants.
const (
	TurnPlanStepStatusPending    TurnPlanStepStatus = "pending"
	TurnPlanStepStatusInProgress TurnPlanStepStatus = "inProgress"
	TurnPlanStepStatusCompleted  TurnPlanStepStatus = "completed"
)

// TurnPlanStep is one step in a turn plan update.
type TurnPlanStep struct {
	Step   string             `json:"step"`
	Status TurnPlanStepStatus `json:"status"`
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

// FileChangePatchUpdatedNotification holds params for item/fileChange/patchUpdated.
type FileChangePatchUpdatedNotification struct {
	ThreadID string             `json:"threadId"`
	TurnID   string             `json:"turnId"`
	ItemID   string             `json:"itemId"`
	Changes  []FileUpdateChange `json:"changes"`
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

// RawResponseItemCompletedNotification holds params for rawResponseItem/completed.
type RawResponseItemCompletedNotification struct {
	ThreadID string          `json:"threadId"`
	TurnID   string          `json:"turnId"`
	Item     json.RawMessage `json:"item"`
}

// ItemGuardianApprovalReviewStartedNotification holds params for item/autoApprovalReview/started.
type ItemGuardianApprovalReviewStartedNotification struct {
	ThreadID     string                       `json:"threadId"`
	TurnID       string                       `json:"turnId"`
	StartedAtMs  int64                        `json:"startedAtMs"`
	ReviewID     string                       `json:"reviewId"`
	TargetItemID *string                      `json:"targetItemId,omitzero"`
	Review       GuardianApprovalReview       `json:"review"`
	Action       GuardianApprovalReviewAction `json:"action"`
}

// ItemGuardianApprovalReviewCompletedNotification holds params for item/autoApprovalReview/completed.
type ItemGuardianApprovalReviewCompletedNotification struct {
	ThreadID       string                       `json:"threadId"`
	TurnID         string                       `json:"turnId"`
	StartedAtMs    int64                        `json:"startedAtMs"`
	CompletedAtMs  int64                        `json:"completedAtMs"`
	ReviewID       string                       `json:"reviewId"`
	TargetItemID   *string                      `json:"targetItemId,omitzero"`
	DecisionSource AutoReviewDecisionSource     `json:"decisionSource"`
	Review         GuardianApprovalReview       `json:"review"`
	Action         GuardianApprovalReviewAction `json:"action"`
}

// GuardianApprovalReviewStatus is an approval auto-review lifecycle status.
type GuardianApprovalReviewStatus string

// Guardian approval review status constants.
const (
	GuardianApprovalReviewStatusInProgress GuardianApprovalReviewStatus = "inProgress"
	GuardianApprovalReviewStatusApproved   GuardianApprovalReviewStatus = "approved"
	GuardianApprovalReviewStatusDenied     GuardianApprovalReviewStatus = "denied"
	GuardianApprovalReviewStatusTimedOut   GuardianApprovalReviewStatus = "timedOut"
	GuardianApprovalReviewStatusAborted    GuardianApprovalReviewStatus = "aborted"
)

// GuardianRiskLevel is the risk level assigned by approval auto-review.
type GuardianRiskLevel string

// Guardian risk level constants.
const (
	GuardianRiskLevelLow      GuardianRiskLevel = "low"
	GuardianRiskLevelMedium   GuardianRiskLevel = "medium"
	GuardianRiskLevelHigh     GuardianRiskLevel = "high"
	GuardianRiskLevelCritical GuardianRiskLevel = "critical"
)

// GuardianUserAuthorization is the authorization level assigned by approval auto-review.
type GuardianUserAuthorization string

// Guardian user authorization constants.
const (
	GuardianUserAuthorizationUnknown GuardianUserAuthorization = "unknown"
	GuardianUserAuthorizationLow     GuardianUserAuthorization = "low"
	GuardianUserAuthorizationMedium  GuardianUserAuthorization = "medium"
	GuardianUserAuthorizationHigh    GuardianUserAuthorization = "high"
)

// GuardianApprovalReview is the approval auto-review payload.
type GuardianApprovalReview struct {
	Status            GuardianApprovalReviewStatus `json:"status"`
	RiskLevel         *GuardianRiskLevel           `json:"riskLevel,omitzero"`
	UserAuthorization *GuardianUserAuthorization   `json:"userAuthorization,omitzero"`
	Rationale         *string                      `json:"rationale,omitzero"`
}

// GuardianCommandSource identifies the command source under guardian review.
type GuardianCommandSource string

// Guardian command source constants.
const (
	GuardianCommandSourceShell       GuardianCommandSource = "shell"
	GuardianCommandSourceUnifiedExec GuardianCommandSource = "unifiedExec"
)

// NetworkApprovalProtocol is the protocol for network approval review actions.
type NetworkApprovalProtocol string

// Network approval protocol constants.
const (
	NetworkApprovalProtocolHTTP      NetworkApprovalProtocol = "http"
	NetworkApprovalProtocolHTTPS     NetworkApprovalProtocol = "https"
	NetworkApprovalProtocolSocks5TCP NetworkApprovalProtocol = "socks5Tcp"
	NetworkApprovalProtocolSocks5UDP NetworkApprovalProtocol = "socks5Udp"
)

// GuardianApprovalReviewAction is a flattened tagged union for guardian review actions.
type GuardianApprovalReviewAction struct {
	Type          string                  `json:"type"`
	Source        GuardianCommandSource   `json:"source,omitzero"`
	Command       string                  `json:"command,omitzero"`
	Cwd           string                  `json:"cwd,omitzero"`
	Program       string                  `json:"program,omitzero"`
	Argv          []string                `json:"argv,omitzero"`
	Files         []string                `json:"files,omitzero"`
	Target        string                  `json:"target,omitzero"`
	Host          string                  `json:"host,omitzero"`
	Protocol      NetworkApprovalProtocol `json:"protocol,omitzero"`
	Port          *uint16                 `json:"port,omitzero"`
	Server        string                  `json:"server,omitzero"`
	ToolName      string                  `json:"toolName,omitzero"`
	ConnectorID   *string                 `json:"connectorId,omitzero"`
	ConnectorName *string                 `json:"connectorName,omitzero"`
	ToolTitle     *string                 `json:"toolTitle,omitzero"`
	Reason        *string                 `json:"reason,omitzero"`
	Permissions   json.RawMessage         `json:"permissions,omitzero"`
}

// Model rerouting.

// ModelReroutedNotification holds params for model/rerouted.
type ModelReroutedNotification struct {
	ThreadID  string             `json:"threadId"`
	TurnID    string             `json:"turnId"`
	FromModel string             `json:"fromModel"`
	ToModel   string             `json:"toModel"`
	Reason    ModelRerouteReason `json:"reason,omitzero"`
}

// Model list (handshake response).

// ModelListResult is the result of a model/list request.
type ModelListResult struct {
	Data       []ModelInfo      `json:"data"`
	NextCursor *json.RawMessage `json:"nextCursor,omitzero"`
}

// ModelInfo describes a single model in a model/list result.
type ModelInfo struct {
	ID                        string              `json:"id"`
	DisplayName               string              `json:"displayName,omitzero"`
	Model                     string              `json:"model,omitzero"`
	Description               string              `json:"description,omitzero"`
	DefaultReasoningEffort    ReasoningEffort     `json:"defaultReasoningEffort,omitzero"`
	Hidden                    bool                `json:"hidden,omitzero"`
	IsDefault                 bool                `json:"isDefault,omitzero"`
	SupportsPersonality       bool                `json:"supportsPersonality,omitzero"`
	Upgrade                   *string             `json:"upgrade,omitzero"`
	UpgradeInfo               *ModelUpgradeInfo   `json:"upgradeInfo,omitzero"`
	AvailabilityNux           *json.RawMessage    `json:"availabilityNux,omitzero"`
	SupportedReasoningEfforts []ModelReasoningOpt `json:"supportedReasoningEfforts,omitzero"`
	InputModalities           []InputModality     `json:"inputModalities,omitzero"`
	ServiceTiers              []ModelServiceTier  `json:"serviceTiers,omitzero"`
	DefaultServiceTier        *string             `json:"defaultServiceTier,omitzero"`
}

// ModelUpgradeInfo holds upgrade migration information for a model.
type ModelUpgradeInfo struct {
	MigrationMarkdown string  `json:"migrationMarkdown,omitzero"`
	Model             string  `json:"model,omitzero"`
	ModelLink         *string `json:"modelLink,omitzero"`
	UpgradeCopy       *string `json:"upgradeCopy,omitzero"`
}

// ModelReasoningOpt describes a supported reasoning effort level.
type ModelReasoningOpt struct {
	ReasoningEffort ReasoningEffort `json:"reasoningEffort"`
	Description     string          `json:"description,omitzero"`
}

// ModelServiceTier describes an available model service tier.
type ModelServiceTier struct {
	ID          string `json:"id"`
	Name        string `json:"name"`
	Description string `json:"description"`
}

// Account notifications and rate-limit responses.

// AuthMode identifies the active Codex authentication mode.
type AuthMode string

// Auth mode constants.
const (
	AuthModeAPIKey              AuthMode = "apikey"
	AuthModeChatGPT             AuthMode = "chatgpt"
	AuthModeChatGPTAuthTokens   AuthMode = "chatgptAuthTokens"
	AuthModeAgentIdentity       AuthMode = "agentIdentity"
	AuthModePersonalAccessToken AuthMode = "personalAccessToken"
	AuthModeBedrockAPIKey       AuthMode = "bedrockApiKey"
)

// AccountUpdatedNotification holds params for account/updated.
type AccountUpdatedNotification struct {
	AuthMode AuthMode `json:"authMode,omitzero"`
	PlanType PlanType `json:"planType,omitzero"`
}

// AccountLoginCompletedNotification holds params for account/login/completed.
type AccountLoginCompletedNotification struct {
	LoginID string `json:"loginId,omitzero"`
	Success bool   `json:"success"`
	Error   string `json:"error,omitzero"`
}

// GetAccountRateLimitsResponse is the response to account/rateLimits/read.
type GetAccountRateLimitsResponse struct {
	RateLimits            RateLimitSnapshot             `json:"rateLimits"`
	RateLimitsByLimitID   map[string]RateLimitSnapshot  `json:"rateLimitsByLimitId,omitzero"`
	RateLimitResetCredits *RateLimitResetCreditsSummary `json:"rateLimitResetCredits,omitzero"`
}

// RateLimitResetCreditsSummary summarizes available rate-limit reset credits.
type RateLimitResetCreditsSummary struct {
	AvailableCount int64 `json:"availableCount"`
}

// ConsumeAccountRateLimitResetCreditParams holds params for account/rateLimitResetCredit/consume.
type ConsumeAccountRateLimitResetCreditParams struct {
	IdempotencyKey string `json:"idempotencyKey"`
}

// ConsumeAccountRateLimitResetCreditResponse is the response to account/rateLimitResetCredit/consume.
type ConsumeAccountRateLimitResetCreditResponse struct {
	Outcome ConsumeAccountRateLimitResetCreditOutcome `json:"outcome"`
}

// ConsumeAccountRateLimitResetCreditOutcome is the result of consuming a rate-limit reset credit.
type ConsumeAccountRateLimitResetCreditOutcome string

// Rate-limit reset credit outcome constants.
const (
	ConsumeAccountRateLimitResetCreditOutcomeReset           ConsumeAccountRateLimitResetCreditOutcome = "reset"
	ConsumeAccountRateLimitResetCreditOutcomeNothingToReset  ConsumeAccountRateLimitResetCreditOutcome = "nothingToReset"
	ConsumeAccountRateLimitResetCreditOutcomeNoCredit        ConsumeAccountRateLimitResetCreditOutcome = "noCredit"
	ConsumeAccountRateLimitResetCreditOutcomeAlreadyRedeemed ConsumeAccountRateLimitResetCreditOutcome = "alreadyRedeemed"
)

// AccountRateLimitsUpdatedNotification holds params for account/rateLimits/updated.
//
// It is a sparse rolling rate-limit update. Clients should merge available
// fields into the latest account/rateLimits/read snapshot or refetch it.
type AccountRateLimitsUpdatedNotification struct {
	RateLimits RateLimitSnapshot `json:"rateLimits"`
}

// RateLimitSnapshot is a Codex account rate-limit snapshot.
type RateLimitSnapshot struct {
	LimitID              string                     `json:"limitId,omitzero"`
	LimitName            string                     `json:"limitName,omitzero"`
	Primary              *RateLimitWindow           `json:"primary,omitzero"`
	Secondary            *RateLimitWindow           `json:"secondary,omitzero"`
	Credits              *CreditsSnapshot           `json:"credits,omitzero"`
	IndividualLimit      *SpendControlLimitSnapshot `json:"individualLimit,omitzero"`
	PlanType             PlanType                   `json:"planType,omitzero"`
	RateLimitReachedType RateLimitReachedType       `json:"rateLimitReachedType,omitzero"`
}

// PlanType identifies a Codex account plan.
type PlanType string

// Plan type constants.
const (
	PlanTypeFree                        PlanType = "free"
	PlanTypeGo                          PlanType = "go"
	PlanTypePlus                        PlanType = "plus"
	PlanTypePro                         PlanType = "pro"
	PlanTypeProLite                     PlanType = "prolite"
	PlanTypeTeam                        PlanType = "team"
	PlanTypeSelfServeBusinessUsageBased PlanType = "self_serve_business_usage_based"
	PlanTypeBusiness                    PlanType = "business"
	PlanTypeEnterpriseCbpUsageBased     PlanType = "enterprise_cbp_usage_based"
	PlanTypeEnterprise                  PlanType = "enterprise"
	PlanTypeEdu                         PlanType = "edu"
	PlanTypeUnknown                     PlanType = "unknown"
)

// RateLimitReachedType identifies which account limit was exhausted.
type RateLimitReachedType string

// Rate-limit reached type constants.
const (
	RateLimitReachedTypeRateLimitReached                 RateLimitReachedType = "rate_limit_reached"
	RateLimitReachedTypeWorkspaceOwnerCreditsDepleted    RateLimitReachedType = "workspace_owner_credits_depleted"
	RateLimitReachedTypeWorkspaceMemberCreditsDepleted   RateLimitReachedType = "workspace_member_credits_depleted"
	RateLimitReachedTypeWorkspaceOwnerUsageLimitReached  RateLimitReachedType = "workspace_owner_usage_limit_reached"
	RateLimitReachedTypeWorkspaceMemberUsageLimitReached RateLimitReachedType = "workspace_member_usage_limit_reached"
)

// RateLimitWindow is one Codex rate-limit window.
type RateLimitWindow struct {
	UsedPercent        int   `json:"usedPercent"`
	WindowDurationMins int64 `json:"windowDurationMins,omitzero"`
	ResetsAt           int64 `json:"resetsAt,omitzero"`
}

// CreditsSnapshot describes Codex account credit availability.
type CreditsSnapshot struct {
	HasCredits bool   `json:"hasCredits"`
	Unlimited  bool   `json:"unlimited"`
	Balance    string `json:"balance,omitzero"`
}

// SpendControlLimitSnapshot describes a Codex workspace spend-control limit.
type SpendControlLimitSnapshot struct {
	Limit            string `json:"limit"`
	Used             string `json:"used"`
	RemainingPercent int    `json:"remainingPercent"`
	ResetsAt         int64  `json:"resetsAt"`
}

// Error notification.

// ErrorNotification holds params for error notifications.
type ErrorNotification struct {
	Error     *TurnError `json:"error"`
	WillRetry bool       `json:"willRetry,omitzero"`
	ThreadID  string     `json:"threadId,omitzero"`
	TurnID    string     `json:"turnId,omitzero"`
}

// ThreadArchivedNotification holds params for thread/archived.
type ThreadArchivedNotification struct {
	ThreadID string `json:"threadId"`
}

// ThreadUnarchivedNotification holds params for thread/unarchived.
type ThreadUnarchivedNotification struct {
	ThreadID string `json:"threadId"`
}

// ThreadClosedNotification holds params for thread/closed.
type ThreadClosedNotification struct {
	ThreadID string `json:"threadId"`
}

// ThreadGoalUpdatedNotification holds params for thread/goal/updated.
type ThreadGoalUpdatedNotification struct {
	ThreadID string     `json:"threadId"`
	TurnID   *string    `json:"turnId,omitzero"`
	Goal     ThreadGoal `json:"goal"`
}

// ThreadGoalClearedNotification holds params for thread/goal/cleared.
type ThreadGoalClearedNotification struct {
	ThreadID string `json:"threadId"`
}

// ThreadSettingsUpdatedNotification holds params for thread/settings/updated.
type ThreadSettingsUpdatedNotification struct {
	ThreadID       string         `json:"threadId"`
	ThreadSettings ThreadSettings `json:"threadSettings"`
}

// ThreadGoalStatus is the lifecycle status of a thread goal.
type ThreadGoalStatus string

// Thread goal status constants.
const (
	ThreadGoalStatusActive        ThreadGoalStatus = "active"
	ThreadGoalStatusPaused        ThreadGoalStatus = "paused"
	ThreadGoalStatusBlocked       ThreadGoalStatus = "blocked"
	ThreadGoalStatusUsageLimited  ThreadGoalStatus = "usageLimited"
	ThreadGoalStatusBudgetLimited ThreadGoalStatus = "budgetLimited"
	ThreadGoalStatusComplete      ThreadGoalStatus = "complete"
)

// ThreadGoal is the current long-running goal state for a thread.
type ThreadGoal struct {
	ThreadID        string           `json:"threadId"`
	Objective       string           `json:"objective"`
	Status          ThreadGoalStatus `json:"status"`
	TokenBudget     *int64           `json:"tokenBudget,omitzero"`
	TokensUsed      int64            `json:"tokensUsed"`
	TimeUsedSeconds int64            `json:"timeUsedSeconds"`
	CreatedAt       int64            `json:"createdAt"`
	UpdatedAt       int64            `json:"updatedAt"`
}

// ThreadSettings contains the active settings for a thread.
type ThreadSettings struct {
	Cwd                     string                   `json:"cwd"`
	ApprovalPolicy          json.RawMessage          `json:"approvalPolicy"`
	ApprovalsReviewer       ApprovalsReviewer        `json:"approvalsReviewer"`
	SandboxPolicy           json.RawMessage          `json:"sandboxPolicy"`
	ActivePermissionProfile *ActivePermissionProfile `json:"activePermissionProfile,omitzero"`
	Model                   string                   `json:"model"`
	ModelProvider           string                   `json:"modelProvider"`
	ServiceTier             *string                  `json:"serviceTier,omitzero"`
	Effort                  *ReasoningEffort         `json:"effort,omitzero"`
	Summary                 *ReasoningSummary        `json:"summary,omitzero"`
	CollaborationMode       json.RawMessage          `json:"collaborationMode"`
	Personality             *Personality             `json:"personality,omitzero"`
}

// ModelVerificationNotification holds params for model/verification.
type ModelVerificationNotification struct {
	ThreadID      string              `json:"threadId"`
	TurnID        string              `json:"turnId"`
	Verifications []ModelVerification `json:"verifications"`
}

// TurnModerationMetadataNotification holds params for turn/moderationMetadata.
type TurnModerationMetadataNotification struct {
	ThreadID string          `json:"threadId"`
	TurnID   string          `json:"turnId"`
	Metadata json.RawMessage `json:"metadata"`
}

// WarningNotification holds params for warning.
type WarningNotification struct {
	ThreadID *string `json:"threadId,omitzero"`
	Message  string  `json:"message"`
}

// GuardianWarningNotification holds params for guardianWarning.
type GuardianWarningNotification struct {
	ThreadID string `json:"threadId"`
	Message  string `json:"message"`
}

// DeprecationNoticeNotification holds params for deprecationNotice.
type DeprecationNoticeNotification struct {
	Summary string  `json:"summary"`
	Details *string `json:"details,omitzero"`
}

// ConfigWarningNotification holds params for configWarning.
type ConfigWarningNotification struct {
	Summary string          `json:"summary"`
	Details *string         `json:"details,omitzero"`
	Path    *string         `json:"path,omitzero"`
	Range   json.RawMessage `json:"range,omitzero"`
}

// ServerRequestResolvedNotification holds params for serverRequest/resolved.
type ServerRequestResolvedNotification struct {
	ThreadID  string          `json:"threadId"`
	RequestID json.RawMessage `json:"requestId"`
}

// McpServerOauthLoginCompletedNotification holds params for mcpServer/oauthLogin/completed.
type McpServerOauthLoginCompletedNotification struct {
	Name    string  `json:"name"`
	Success bool    `json:"success"`
	Error   *string `json:"error,omitzero"`
}

// McpServerStatusUpdatedNotification holds params for mcpServer/startupStatus/updated.
type McpServerStatusUpdatedNotification struct {
	Name   string                `json:"name"`
	Status McpServerStartupState `json:"status"`
	Error  *string               `json:"error,omitzero"`
}
