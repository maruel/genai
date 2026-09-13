# Pi Coding Agent Provider

Implements `genai.Provider` for the Pi coding agent CLI in RPC mode.
Translates Pi's JSONL command/event protocol over stdin/stdout into `genai.Result` / `genai.Reply`.

## Session Management

Each GenSync or GenStream call spawns a fresh `pi --mode rpc --no-session` subprocess.

## Protocol

Pi uses a custom JSONL protocol (not JSON-RPC 2.0). `type` field dispatch, optional
`id` for request/response correlation, strict LF framing.

- **No handshake**: subprocess is immediately ready.
- **Commands**: `prompt`, `steer`, `follow_up`, `abort`, `clear_queue`, session, model, thinking,
  queue-mode, compaction, retry, and bash commands. See the versioned upstream RPC
  documentation for the full command list.
- **Events**: `agent_start`, `agent_end`, `message_update`, `turn_start`, `turn_end`,
  `tool_execution_start`/`update`/`end`, `bash_execution_update`,
  `session_info_changed`, and `extension_error`.
- **Responses**: `{"type":"response", "command":"...", "success":true/false, "data":{...}}`
- **Extension UI**: may emit `extension_ui_request` requiring `extension_ui_response` on stdin.

## Upstream Source

The DTOs in `dto.go` are defined against **Pi Coding Agent v0.85.1**, commit
[`71dca871bc80b6bc97be37f0ca3189399d651fff`](https://github.com/earendil-works/pi/commit/71dca871bc80b6bc97be37f0ca3189399d651fff).

Type definitions live in https://github.com/earendil-works/pi:

- `packages/ai/src/types.ts` — `Model`, `UserMessage`, `AssistantMessage`, `ToolResultMessage`
- `packages/agent/src/types.ts` — base `AgentMessage` and `AgentEvent` types
- `packages/coding-agent/src/core/messages.ts` — coding-agent custom message variants
- `packages/coding-agent/src/core/agent-session.ts` — session event variants
- `packages/coding-agent/src/modes/json-event.ts` — stdout event projection
- `packages/coding-agent/src/modes/rpc/rpc-types.ts` — RPC command/response types

When updating wire types, clone the upstream repository at a released tag and
compare these files to find new commands, event types, or fields. Update this
baseline when the DTOs are changed.

## References

Source code:
- https://github.com/earendil-works/pi

npm package:
- https://www.npmjs.com/package/@earendil-works/pi-coding-agent

Documentation:
- [RPC protocol for v0.85.1](https://github.com/earendil-works/pi/blob/71dca871bc80b6bc97be37f0ca3189399d651fff/packages/coding-agent/docs/rpc.md)
