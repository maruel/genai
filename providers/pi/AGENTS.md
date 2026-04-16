# Pi Coding Agent Provider

Implements `genai.Provider` for the Pi coding agent CLI in RPC mode.
Translates Pi's JSONL command/event protocol over stdin/stdout into `genai.Result` / `genai.Reply`.

## Architecture

- `client.go` — Provider lifecycle, subprocess management, GenSync/GenStream
- `types.go` — JSONL type definitions: commands (input), events/responses (output)
- `docs/implementation.md` — Detailed implementation plan, type mappings, open questions

Each GenSync or GenStream call spawns a fresh `pi --mode rpc --no-session` subprocess.

## Protocol

Pi uses a custom JSONL protocol (not JSON-RPC 2.0). `type` field dispatch, optional
`id` for request/response correlation, strict LF framing.

- **No handshake**: subprocess is immediately ready.
- **Commands**: `prompt`, `set_model`, `get_available_models`, `abort`, `set_thinking_level`, `new_session`.
- **Events**: `agent_start`, `agent_end`, `message_update`, `turn_start`, `turn_end`,
  `tool_execution_start`/`update`/`end`.
- **Responses**: `{"type":"response", "command":"...", "success":true/false, "data":{...}}`
- **Extension UI**: may emit `extension_ui_request` requiring `extension_ui_response` on stdin.

## Upstream Source

Type definitions live in the monorepo https://github.com/badlogic/pi-mono:

- `packages/ai/src/types.ts` — `Model`, `UserMessage`, `AssistantMessage`, `ToolResultMessage`
- `packages/agent/src/types.ts` — `AgentMessage`, `AgentEvent`
- `packages/coding-agent/src/modes/rpc/rpc-types.ts` — RPC command/response types

When updating wire types, clone https://github.com/badlogic/pi-mono and diff
against these files to find new commands, event types, or fields.

## References

Source code:
- https://github.com/badlogic/pi-mono

npm package:
- https://www.npmjs.com/package/@mariozechner/pi-coding-agent

Documentation:
- RPC protocol: included in the npm package at `docs/rpc.md`
