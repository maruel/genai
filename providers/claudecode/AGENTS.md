# Claude Code Provider

Implements `genai.Provider` backed by the Claude Code CLI subprocess.

## References

Source code:
- https://github.com/anthropics/claude-code

Claude Code headless:
- https://code.claude.com/docs/en/headless: headless mode overview
- https://platform.claude.com/docs/en/agent-sdk/streaming-output: streaming protocol wire format
- git clone https://github.com/anthropics/claude-agent-sdk-python for SDK types (`src/claude_agent_sdk/types.py`)

## Keeping DTOs in sync with the binary

The `claude` binary is a **Bun SEA** (Single Executable Application) that compiles JavaScript through
JavaScriptCore (JSC) to native x86-64 machine code. No bytecode survives.

The Zod schema property names survive as interned string constants. `extract_schema.py` leverages this:
it reads the binary, extracts complete `E.object({...})` bodies via brace-counting, and parses the
top-level field names. No dependencies needed. Run it after each claude upgrade:

```sh
./extract_schema.py [/path/to/claude-binary]
```

Compare the output against JSON struct tags in `dto.go`. Quick one-liner alternative:
```sh
strings $(which claude) | grep -o 'type:E.literal("stream_event")[^;]*'
```
