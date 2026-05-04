# Llama.cpp Provider

- **Source code**: https://github.com/ggml-org/llama.cpp

## Recording

Use `RECORD=failure_only` when re-recording tests after a server update.
This replays existing cassettes and only records new/missing ones, avoiding
unnecessary overwrites.

```
RECORD=failure_only go test ./providers/llamacpp -update-scoreboard
```

## Version Management

Run `go run ./internal/cmd/update-servers` to check for newer llama.cpp
releases. Pass `--apply` to update `BuildNumber` and re-record tests.

## Known Issues

- **Qwen3.5-2B thinking model** (`enable_thinking: true`, `--jinja`): The
  non-streaming ChatResponse may return empty `"text": ""` in
  `choices[0].message.content` while the actual response is in
  `reasoning_content`. `Content.To()` silently skips empty text content
  (instead of erroring) to handle this, but GenSync-dependent features
  (JSON, tools, seed, etc.) will still report as unsupported for this model
  unless the server starts populating `content` properly again.
