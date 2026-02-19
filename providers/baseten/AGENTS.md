# Baseten

- **Documentation**: https://docs.baseten.co/
- **API Reference**: https://docs.baseten.co/reference/inference-api/chat-completions
- **OpenAPI Spec**: https://docs.baseten.co/reference/inference-api/llm-openapi-spec.json
- **No official Go SDK** — API is fully OpenAI-compatible

## API Notes

- Base URL: `https://inference.baseten.co/v1`
- Auth header: `Authorization: Api-Key <key>` (not `Bearer`)
- Env var: `BASETEN_API_KEY`
- Model slugs use `org/model-name` format (e.g. `deepseek-ai/DeepSeek-V3.1`)
- Supports: chat completions, streaming (SSE), tool calling, structured outputs
