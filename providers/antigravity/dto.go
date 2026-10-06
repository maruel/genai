// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Wire types for the Antigravity CLI stream-json NDJSON protocol.
//
// Type and field names mirror the Go types in the agy binary, recovered from
// its runtime type descriptors with internal/extracttypes: package `steps` for
// stdout events and package `printmode` for stdin messages. The JSON tags are
// copied verbatim. These DTOs match agy 1.2.14.
//
// Enum values are untyped strings upstream; the constants below hold the
// values observed in output.

package antigravity

import (
	"bytes"
	"encoding/json"
	"fmt"
)

// EventType is the `event` discriminator of every stream-json line, on both
// stdin and stdout.
type EventType string

// Event types.
const (
	// EventCommandResult carries the payload of a print-mode command such as `models`.
	EventCommandResult EventType = "command_result"
	// EventInit is the first stdout event of a conversation.
	EventInit EventType = "init"
	// EventResult terminates one turn.
	EventResult EventType = "result"
	// EventStepUpdate reports a trajectory step change.
	EventStepUpdate EventType = "step_update"
	// EventUser is the only accepted stdin event.
	EventUser EventType = "user"
)

// ============================================================
// Input (stdin), from package printmode.
// ============================================================

// StreamInputMessage is one user turn written to stdin.
type StreamInputMessage struct {
	Event   EventType              `json:"event"`
	Message StreamInputUserMessage `json:"message,omitzero"`
}

// StreamInputUserMessage is the message of a StreamInputMessage.
//
// Upstream, Content is a raw JSON value: either a string or an array of
// StreamInputContentBlock. There is no role field.
type StreamInputUserMessage struct {
	Content []StreamInputContentBlock `json:"content,omitzero"`
}

// UnmarshalJSON implements json.Unmarshaler to accept Content as either a
// plain string or an array of StreamInputContentBlock.
func (m *StreamInputUserMessage) UnmarshalJSON(b []byte) error {
	var raw rawStreamInputUserMessage
	if err := json.Unmarshal(b, &raw); err != nil {
		return err
	}
	if len(raw.Content) == 0 || bytes.Equal(raw.Content, []byte("null")) {
		m.Content = nil
		return nil
	}
	if raw.Content[0] == '"' {
		var s string
		if err := json.Unmarshal(raw.Content, &s); err != nil {
			return err
		}
		m.Content = []StreamInputContentBlock{{Type: "text", Text: s}}
		return nil
	}
	return json.Unmarshal(raw.Content, &m.Content)
}

type rawStreamInputUserMessage struct {
	Content json.RawMessage `json:"content"`
}

// StreamInputContentBlock is one content block. agy 1.2.14 rejects any type
// other than "text".
type StreamInputContentBlock struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

// ============================================================
// Output (stdout), from package steps.
// ============================================================

// StreamEvent is the envelope of every stdout line. Event selects the
// populated payload.
type StreamEvent struct {
	Event          EventType         `json:"event"`
	ConversationID string            `json:"conversation_id,omitzero"`
	Init           InitPayload       `json:"init,omitzero"`
	StepUpdate     StepUpdatePayload `json:"step_update,omitzero"`
	Result         JSONOutput        `json:"result,omitzero"`
	Command        JSONCommand       `json:"command,omitzero"`
}

// InitPayload describes the session configuration.
type InitPayload struct {
	Model            string            `json:"model,omitzero"`
	Cwd              string            `json:"cwd,omitzero"`
	Agent            string            `json:"agent,omitzero"`
	Tools            []string          `json:"tools,omitzero"`
	PermissionMode   string            `json:"permission_mode,omitzero"`
	JSONSchema       json.RawMessage   `json:"json_schema,omitzero"`
	ExpandedCommands []ExpandedCommand `json:"expanded_commands,omitzero"`
}

// ExpandedCommand is a slash command expanded into the prompt.
type ExpandedCommand struct {
	Name string `json:"name"`
	Type string `json:"type"`
}

// StepState is the lifecycle state of a step.
//
// It projects localharness StepUpdate.State without its "STATE_" prefix. The
// proto also declares STATE_WAITING_FOR_USER and STATE_ERROR; agy 1.2.14 has
// not been observed emitting them.
type StepState string

// Step states observed in agy 1.2.14.
const (
	StepActive StepState = "ACTIVE"
	StepDone   StepState = "DONE"
)

// StepType is the step discriminator.
//
// localharness StepUpdate holds one field per action (view_file, run_command,
// finish, error, ...); agy collapses tool actions into StepTool with ToolName.
type StepType string

// Step types observed in agy 1.2.14.
const (
	StepAgentResponse StepType = "agent_response"
	StepFinish        StepType = "finish"
	StepSystemMessage StepType = "system_message"
	StepTool          StepType = "tool"
	StepUserInput     StepType = "user_input"
)

// StepUpdatePayload reports progress on one trajectory step.
//
// An agent_response step emits incremental TextDelta values while ACTIVE.
// Usage is set once, on the DONE update of a model call. agy drops the
// localharness thinking and thinking_delta fields, so thinking text is not
// available; only JSONUsage.ThinkingTokens reports it.
type StepUpdatePayload struct {
	ConversationID  string       `json:"conversation_id"`
	StepIndex       int64        `json:"step_index"`
	State           StepState    `json:"state"`
	StepType        StepType     `json:"step_type,omitzero"`
	ToolName        string       `json:"tool_name,omitzero"`
	TextDelta       string       `json:"text_delta,omitzero"`
	DurationSeconds float64      `json:"duration_seconds,omitzero"`
	Usage           JSONUsage    `json:"usage,omitzero"`
	ToolInfo        ToolInfo     `json:"tool_info,omitzero"`
	SubagentInfo    SubagentInfo `json:"subagent_info,omitzero"`
}

// ToolInfo describes a tool call.
type ToolInfo struct {
	Name       string          `json:"name,omitzero"`
	Parameters json.RawMessage `json:"parameters,omitzero"`
	Output     string          `json:"output,omitzero"`
	Error      ToolError       `json:"error,omitzero"`
}

// ToolError is a failed tool call.
type ToolError struct {
	Type    string `json:"type"`
	Message string `json:"message"`
}

// SubagentInfo lists the subagents a step delegated to.
type SubagentInfo struct {
	Subagents []SubagentSpecInfo `json:"subagents"`
}

// SubagentSpecInfo identifies a delegated subagent conversation.
type SubagentSpecInfo struct {
	TypeName       string   `json:"type_name,omitzero"`
	Role           string   `json:"role,omitzero"`
	InitialPrompt  string   `json:"initial_prompt,omitzero"`
	ConversationID string   `json:"conversation_id,omitzero"`
	LogURI         string   `json:"log_uri,omitzero"`
	WorkspaceURIs  []string `json:"workspace_uris,omitzero"`
}

// JSONUsage is the token accounting of a model call or a whole conversation.
//
// It projects localharness UsageMetadata: ThinkingTokens is
// thoughts_token_count and CacheReadTokens is cached_content_token_count.
// OutputTokens includes ThinkingTokens.
type JSONUsage struct {
	InputTokens     int64 `json:"input_tokens"`
	OutputTokens    int64 `json:"output_tokens"`
	ThinkingTokens  int64 `json:"thinking_tokens"`
	CacheReadTokens int64 `json:"cache_read_tokens"`
	TotalTokens     int64 `json:"total_tokens"`
}

// Add accumulates u2 into u.
func (u *JSONUsage) Add(u2 *JSONUsage) {
	u.InputTokens += u2.InputTokens
	u.OutputTokens += u2.OutputTokens
	u.ThinkingTokens += u2.ThinkingTokens
	u.CacheReadTokens += u2.CacheReadTokens
	u.TotalTokens += u2.TotalTokens
}

// ResultStatus is the outcome of a turn.
//
// localharness TrajectoryStateUpdate.StopReason lists the limits that stop a
// trajectory (model calls, tool calls, tokens, quota); how agy reports them in
// a result is unknown.
type ResultStatus string

// Result statuses observed in agy 1.2.14.
const (
	StatusError   ResultStatus = "ERROR"
	StatusSuccess ResultStatus = "SUCCESS"
)

// JSONOutput is the result of one turn or command.
//
// Usage, NumTurns, and DurationSeconds are cumulative over the whole
// conversation, including turns from earlier processes that resumed it.
type JSONOutput struct {
	ConversationID   string             `json:"conversation_id"`
	Status           ResultStatus       `json:"status"`
	Response         string             `json:"response"`
	Error            string             `json:"error,omitzero"`
	DurationSeconds  float64            `json:"duration_seconds"`
	NumTurns         int64              `json:"num_turns"`
	StructuredOutput json.RawMessage    `json:"structured_output,omitzero"`
	JSONSchema       json.RawMessage    `json:"json_schema,omitzero"`
	Usage            JSONUsage          `json:"usage"`
	Command          JSONCommand        `json:"command,omitzero"`
	DeniedActions    []JSONDeniedAction `json:"denied_actions,omitzero"`
}

// AsError returns the turn failure, if any.
func (r *JSONOutput) AsError() error {
	switch r.Status {
	case StatusSuccess:
		return nil
	case StatusError:
		return fmt.Errorf("agy error: %s", r.Error)
	default:
		return fmt.Errorf("agy returned status %q: %s", r.Status, r.Error)
	}
}

// JSONDeniedAction is a tool action that print mode denied.
type JSONDeniedAction struct {
	Action  string `json:"action,omitzero"`
	Display string `json:"display_name"`
}

// JSONCommand is the payload of a print-mode command. Data depends on Name.
type JSONCommand struct {
	Name string          `json:"name"`
	Data json.RawMessage `json:"data,omitzero"`
}

// UsageData is the JSONCommand.Data of the `usage` command (upstream printmode.quotaJSON).
// Quota pools share weekly and five-hour limits across their member models.
type UsageData struct {
	Description string       `json:"description,omitzero"`
	Groups      []UsageGroup `json:"groups"`
}

// UsageGroup is a pool of models that share quota buckets.
type UsageGroup struct {
	Name        string        `json:"name"`
	Description string        `json:"description,omitzero"`
	Buckets     []UsageBucket `json:"buckets"`
}

// UsageBucket is a subscription quota window, not token accounting.
type UsageBucket struct {
	ID                string   `json:"id,omitzero"`
	Name              string   `json:"name"`
	Description       string   `json:"description,omitzero"`
	Window            string   `json:"window,omitzero"`
	Disabled          bool     `json:"disabled,omitzero"`
	RemainingFraction *float64 `json:"remaining_fraction,omitzero"`
	RemainingAmount   *int64   `json:"remaining_amount,omitzero"`
	ResetTime         string   `json:"reset_time,omitzero"`
}

// ModelsData is the JSONCommand.Data of the `models` command.
type ModelsData struct {
	Models []Model `json:"models"`
}

// Model is one model listed by `agy models` (upstream entrypoints.modelJSON).
//
// The ID encodes the reasoning effort, e.g. "gemini-3.8-flash-low".
type Model struct {
	ID    string `json:"id"`
	Label string `json:"label,omitzero"`
}

// GetID implements genai.Model.
func (m *Model) GetID() string { return m.ID }

// String implements genai.Model.
func (m *Model) String() string { return m.Label }

// Context implements genai.Model. agy does not report the context window.
func (m *Model) Context() int64 { return 0 }
