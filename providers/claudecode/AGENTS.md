# Claude Code Provider

Implements `genai.Provider` backed by the Claude Code CLI subprocess.

## References

Source code:
- https://github.com/anthropics/claude-code

Current wire baseline:
- Claude Code CLI 2.1.330
- `@anthropic-ai/claude-agent-sdk` 0.3.330

Claude Code headless:
- https://code.claude.com/docs/en/headless: headless mode overview
- https://platform.claude.com/docs/en/agent-sdk/streaming-output: streaming protocol wire format
- git clone https://github.com/anthropics/claude-agent-sdk-python for SDK types (`src/claude_agent_sdk/types.py`)

## Keeping DTOs in sync

The published TypeScript SDK declaration is the authoritative released wire contract. Fetch the SDK version paired
with the Claude Code release above, then check that `dto.go` declares every discriminator and every field reachable
from them:

```sh
npm pack @anthropic-ai/claude-agent-sdk@<version>
tar -xzf anthropic-ai-claude-agent-sdk-<version>.tgz
./extract_schema.py package/sdk.d.ts dto.go --version <version>
```

The `claude` binary embeds the Zod output schemas, including `@internal` messages and fields the SDK omits.
`zod_schema.py` reads them: `lookup` prints the schema around a key seen in a log, and `check` lists schema fields
missing from `dto.go`. Most of those are telemetry that stream-json never sends; add one when a log shows it.

```sh
./zod_schema.py "$(readlink -f "$(command -v claude)")" lookup 'type:T("command_lifecycle")'
./zod_schema.py "$(readlink -f "$(command -v claude)")" check dto.go
```
