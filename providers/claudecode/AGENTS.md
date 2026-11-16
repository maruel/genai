# Claude Code Provider

Implements `genai.Provider` backed by the Claude Code CLI subprocess.

## References

Source code:
- https://github.com/anthropics/claude-code

Current wire baseline:
- Claude Code CLI 2.1.325
- `@anthropic-ai/claude-agent-sdk` 0.3.325

Claude Code headless:
- https://code.claude.com/docs/en/headless: headless mode overview
- https://platform.claude.com/docs/en/agent-sdk/streaming-output: streaming protocol wire format
- git clone https://github.com/anthropics/claude-agent-sdk-python for SDK types (`src/claude_agent_sdk/types.py`)

## Keeping DTOs in sync

The published TypeScript SDK declaration is the authoritative released wire contract. The `claude` binary is a
Bun SEA that still embeds the Zod output schemas, including `@internal` messages and fields the SDK omits. To type
such a field seen in a log, search the binary for its key, e.g. `type:T("command_lifecycle")`, and read the
enclosing `u({...})` object.

Fetch the exact SDK version paired with the Claude Code release, then verify that every SDK message, control request,
and hook discriminator is declared in `dto.go`:

```sh
npm pack @anthropic-ai/claude-agent-sdk@0.3.325
tar -xzf anthropic-ai-claude-agent-sdk-0.3.325.tgz
./extract_schema.py package/sdk.d.ts dto.go --version 0.3.325
```
