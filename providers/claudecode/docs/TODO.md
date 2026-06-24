# Future Enhancements

This document outlines how the claudecode provider could leverage the wire
protocol types (documented in `dto.go`) to expose new functionality through
the `genai` interfaces.

## Current State

The provider sends `inputUser` messages and can answer stdout
`control_request` messages through `GenOption.ControlHandler`. Slash commands,
keep-alive, and environment variable updates are unused. Calls without a
control handler still default to `--permission-mode bypassPermissions` when
tools are enabled.

## Opportunities

### 1. Rich Interactive Permission Handling

**Current state:** `GenOption.ControlHandler` can answer `can_use_tool`
requests over stdio, which supports `AskUserQuestion` and other permission
prompts without blanket bypass.

**Remaining problem:** The callback is raw and provider-specific. There is no
portable genai-level UI contract for rendering questions, dialogs, or
elicitation forms.

**Wire types:** `outputTypeControlRequest` (output) + `inputControlResponse` (input).
Claude Code sends a `controlCanUseTool` request with tool name and input; the
host responds with allow/deny via `inputControlResponse`.

**genai integration:**
- Add typed helpers for common permission response shapes.
- Consider a provider-neutral callback or channel for permission decisions.
- Extend coverage to `request_user_dialog` and `elicitation` response shapes.

### 2. Mid-Session Model Switching

**Problem:** Changing models requires starting a new subprocess.

**Wire types:** `controlReqSetModel` sent via `inputControlRequest`.

**genai integration:**
- Add a `GenOption` or method to send a model switch before the user message.
- Could enable cost optimization: start with haiku, escalate to opus when needed.

### 3. Turn Interruption

**Problem:** Cancelling a `context.Context` kills the subprocess. There's no
way to interrupt a single turn and continue the session.

**Wire types:** `controlReqInterrupt` sent via `inputControlRequest`.

**genai integration:**
- On context cancellation, send `controlReqInterrupt` instead of killing the
  process.
- Return partial results from the interrupted turn.
- Requires keeping the subprocess alive across GenSync/GenStream calls (session
  pooling).

### 4. Context Usage Monitoring

**Problem:** Users have no visibility into context window usage or when
auto-compaction will fire.

**Wire types:** `controlReqGetContextUsage` (request) and the corresponding
`outputControlResponse` (response with token breakdown).

**genai integration:**
- Expose context usage in `Result.Opaque` after each turn.
- Could enable automatic `/compact` before hitting limits.

### 5. Structured Output (JSON Mode)

**Problem:** `GenOptionText.DecodeAs` is listed as unsupported.

**Wire types:** `controlReqInitialize` accepts a `jsonSchema` field.
`outputResult.StructuredOutput` carries the validated JSON output.

**genai integration:**
- When `DecodeAs` is set, serialize the struct's JSON schema into
  `controlReqInitialize.JSONSchema`.
- Parse `StructuredOutput` from the result instead of free-text.
- Remove `DecodeAs` from the unsupported options list.

### 6. Streaming Tool Use

**Problem:** The provider streams text and thinking deltas but ignores tool
use during streaming.

**Wire types:** `contentBlockStart` (with tool name/ID) and `input_json_delta`
stream events accumulate the tool input JSON incrementally.

**genai integration:**
- Track `content_block_start` events with `type:"tool_use"`.
- Accumulate `input_json_delta` partials.
- Yield tool-use replies in the stream iterator.
- Enables callers to observe tool calls as they happen.

### 7. Keep-Alive for Long Turns

**Problem:** Long-running tool calls (multi-minute bash commands) may trigger
proxy or SSH timeouts.

**Wire types:** `inputKeepAlive`.

**genai integration:**
- Start a background goroutine that writes `inputKeepAlive` every 30s while
  waiting for output.
- Requires keeping stdin open longer (currently closed after writing the user
  message).

### 8. Subagent Progress

**Problem:** When Claude Code spawns subagents, the caller gets no visibility
until the final result.

**Wire types:** `systemTaskStarted`, `systemTaskProgress`, `systemTaskNotification`.

**genai integration:**
- Surface subagent events as `Reply` entries in the stream with a new field
  or via `Reply.Opaque`.
- Enables progress UIs for multi-agent workflows.

### 9. API Retry Visibility

**Problem:** When the API returns a retryable error, the provider blocks
silently until it succeeds or fails.

**Wire types:** `systemAPIRetry` with attempt count, max retries, delay, and
error status.

**genai integration:**
- Yield retry events in the stream so callers can display "retrying in Xs..."
- Include retry metadata in `Result.Opaque` for diagnostics.

### 10. Restore ANTHROPIC_API_KEY After Auth

**Problem:** The provider strips `ANTHROPIC_API_KEY` from the subprocess
environment so Claude Code authenticates via OAuth instead of consuming API
key credits. However, tools running inside the session (Bash scripts, MCP
servers) may need the key to call the Anthropic API.

**Wire types:** `inputUpdateEnvVars`.

**genai integration:**
- After receiving `outputInit` (auth is complete), send `inputUpdateEnvVars`
  with the original `ANTHROPIC_API_KEY` value.
- The key is then available to tools but was not used for auth.
- Requires keeping stdin open after the user message (same prerequisite as
  keep-alive and turn interruption).

## Implementation Priority

1. **Structured output** — directly unblocks a `genai` feature (`DecodeAs`).
2. **Interactive permissions** — enables safe tool use without blanket bypass.
3. **Streaming tool use** — completes the streaming story.
4. **Context usage** — low effort, high visibility.
5. **Keep-alive** — prevents silent failures on long turns.
6. **Restore ANTHROPIC_API_KEY** — do alongside keep-alive (same stdin refactor).
7. **Turn interruption** — requires session pooling, larger refactor.
8. **Model switching** — niche, useful for cost optimization.
9. **Subagent progress / API retry** — nice-to-have observability.

Items 5-7 share a prerequisite: keeping stdin open past the initial user
message so control requests can be sent mid-session.
