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
releases. Pass `--apply` to update `Version` and re-record tests.
