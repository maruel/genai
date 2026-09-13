# Codex CLI Provider

Implements `genai.Provider` for OpenAI Codex CLI in app-server mode.
Translates Codex's JSON-RPC 2.0 wire protocol into `genai.Result` / `genai.Reply`.

## Protocol

Codex CLI runs in **app-server mode** -- a JSON-RPC 2.0 NDJSON protocol over stdin/stdout.

**Handshake sequence** (per subprocess):
1. `initialize` request -> response
2. `initialized` notification
3. `model/list` request -> response (populates model list dynamically)
4. `thread/start` or `thread/resume` -> response with thread ID

**Prompt delivery**: `turn/start` JSON-RPC request with text + optional images as data URLs.

**Streaming events**: `item/agentMessage/delta`, `item/reasoning/summaryTextDelta`,
`item/commandExecution/outputDelta`, `item/mcpToolCall/progress`.

## Session Management

Each GenSync or GenStream call spawns a fresh `codex app-server` subprocess.
Thread IDs in `Reply.Opaque["thread_id"]` enable multi-turn session resumption.

## Upstream Source

Type names in `dto.go` match the upstream Rust definitions:

- `codex-rs/app-server-protocol/src/protocol/v2/` -- notification and item structs
- `codex-rs/app-server-protocol/src/protocol/common.rs` -- method string <-> struct mapping

When updating wire types, clone https://github.com/openai/codex and diff
against these files to find new fields, item types, or notification methods.

The DTOs were last synced against `openai/codex` release `rust-v0.154.0`, commit
`6b9826e3aa83b1a5947db50f4332cb9c65f1b340`.

## References

Source code:
- https://github.com/openai/codex

Documentation:
- https://developers.openai.com/codex/cli: CLI documentation
- https://developers.openai.com/codex/cli/reference: CLI reference

## Key Design Decisions

- **Upstream naming**: Go types mirror upstream Rust struct names to simplify syncing.
- **Dynamic model list**: initial model replaced after handshake with live list from `model/list`.
- **Model pagination**: all `model/list` pages are fetched before returning the live model list.
- **Error suppression**: notifications with `willRetry=true` are silently dropped.
- **Server requests**: unsupported server-to-client requests fail the call immediately instead of waiting
  indefinitely for a response the provider cannot produce.
- **Opt-out capabilities**: handshake disables verbose notifications not needed for text generation
  (e.g., `item/fileChange/outputDelta`, `turn/diff/updated`).

## Recording Maintenance

All protocol fixtures, including recursively nested scoreboard recordings, are sanitized during recording and covered by
fixture hygiene checks. Keep recordings free of host-specific paths, server names, and installation identifiers.
