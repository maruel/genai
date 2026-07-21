# OpenCode Package

Implements `genai.Provider` for the OpenCode CLI via ACP (Agent Client Protocol):
JSON-RPC 2.0 over stdin/stdout of the `opencode acp` subprocess.

## ACP Handshake

```
→ initialize (protocolVersion:1, clientInfo:{name:"genai-opencode"})
← initialize result (agentCapabilities, agentInfo)
→ session/new (cwd, mcpServers:[]) or session/load (sessionId, cwd, mcpServers:[])
← session result (sessionId, configOptions)
→ session/set_config_option (sessionId, configId, value)   [if model, effort, or mode requested]
← updated configOptions
→ session/prompt (sessionId, prompt:[{type:"text",text:"..."}])
← session/update notifications (streaming)
← session/prompt result (stopReason, usage)
```

## Session Management

Each GenSync/GenStream call launches a fresh `opencode acp` subprocess. Session IDs
are returned in `Reply.Opaque["session_id"]` and automatically picked up from message
history for multi-turn: `session/load` is used instead of `session/new`.

## Upstream Source

Type names in `dto.go` follow the upstream ACP SDK definitions:

- `packages/opencode/src/acp/agent.ts` — session update types and request/response handling

When updating wire types, clone https://github.com/anomalyco/opencode and diff
against `agent.ts` to find new session update types or fields.

## Key Design Decisions

- **Upstream naming**: Go types mirror ACP SDK naming (e.g. `agentMessageChunkUpdate`,
  `contentBlock`) to simplify syncing with the OpenCode source.
- **Typed enums**: `Method`, `UpdateType` are typed string enums for compile-time safety.
- **ACP over run mode**: `opencode run` is single-turn per process (no stdin loop).
  ACP provides long-lived JSON-RPC over stdin/stdout with multi-turn.
- **Dynamic configuration**: `session/new` returns `configOptions`, including
  model, effort, and mode selectors. The model selector supplies the model list.
- **Model, effort, and mode selection**: sends `session/set_config_option` after
  session creation. Model selection remains compatible with older ACP servers
  through `session/set_model`; effort and mode require their dynamic selectors.
- **Reasoning effort**: `GenOption.Effort` is validated against the selected
  model's effort values returned by ACP, so provider-specific and custom variants
  are supported.
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
