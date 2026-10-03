# OpenAI Compatible Provider

- **Official Documentation**: https://platform.openai.com/docs/api-reference
- **Go SDK**: https://github.com/openai/openai-go (supports OpenAI-compatible endpoints)

## Development

Keep compatibility assumptions beside [the client](client.go) and [the wire types](dto.go).
Add local protocol cases to [dto_test.go](dto_test.go) and HTTP round trips to
[client_test.go](client_test.go). Local protocol tests do not qualify remote model capabilities in
`scoreboard.json`; use the recording and qualification workflow in [../AGENTS.md](../AGENTS.md).
