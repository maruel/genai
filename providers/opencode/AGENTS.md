# OpenCode Package

Implements `genai.Provider` for the OpenCode CLI via ACP (Agent Client Protocol):
JSON-RPC 2.0 over stdin/stdout of the `opencode acp` subprocess.

## Architecture

- `client.go` — Provider implementation: lifecycle, handshake, GenSync/GenStream
- `types.go` — ACP JSON-RPC 2.0 type definitions, organized as: shared → input → output
- `client_test.go` — Unit tests with NDJSON recording/replay
- `example_test.go` — Usage example
- `scoreboard.json` — Model capabilities and smoke test configuration

## ACP Handshake

```
→ initialize (protocolVersion:1, clientInfo:{name:"genai-opencode"})
← initialize result (agentCapabilities, agentInfo)
→ session/new (cwd, mcpServers:[]) or session/load (sessionId, cwd, mcpServers:[])
← session result (sessionId, models, modes)
→ unstable_setSessionModel (sessionId, modelId)   [if model requested]
← set model result
→ session/prompt (sessionId, prompt:[{type:"text",text:"..."}])
← session/update notifications (streaming)
← session/prompt result (stopReason, usage)
```

## Session Management

Each GenSync/GenStream call launches a fresh `opencode acp` subprocess. Session IDs
are returned in `Reply.Opaque["session_id"]` and automatically picked up from message
history for multi-turn: `session/load` is used instead of `session/new`.

## Upstream Source

Type names in `types.go` follow the upstream ACP SDK definitions:

- `packages/opencode/src/acp/agent.ts` — session update types and request/response handling

When updating wire types, clone https://github.com/anomalyco/opencode and diff
against `agent.ts` to find new session update types or fields.

## Key Design Decisions

- **Upstream naming**: Go types mirror ACP SDK naming (e.g. `agentMessageChunkUpdate`,
  `contentBlock`) to simplify syncing with the OpenCode source.
- **Typed enums**: `Method`, `UpdateType` are typed string enums for compile-time safety.
- **ACP over run mode**: `opencode run` is single-turn per process (no stdin loop).
  ACP provides long-lived JSON-RPC over stdin/stdout with multi-turn.
- **Dynamic model list**: available models come from `session/new` handshake response.
  Current model listed first.
- **Model selection**: If a model is set, sends `unstable_setSessionModel` after session
  creation. Best-effort — ignores errors from the unstable method.
- **Image support**: detected from `agentCapabilities.promptCapabilities.image`
  in the initialize response.
- **Permission auto-approve**: `session/request_permission` requests are auto-approved
  with the first "allow" option.

## References

Source code:
- https://github.com/anomalyco/opencode

Documentation:
- https://opencode.ai/docs/acp/: ACP documentation
- https://agentclientprotocol.com: ACP specification
- https://opencode.ai/docs/config/: configuration format
- https://opencode.ai/docs/providers/: provider configurations
