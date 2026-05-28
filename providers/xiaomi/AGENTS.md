# Xiaomi MiMo

- **Official Documentation**: https://platform.xiaomimimo.com/docs/en-US
- **API Reference (OpenAI)**: https://platform.xiaomimimo.com/docs/en-US/api/chat/openai-api
- **List Models**: https://api.xiaomimimo.com/v1/models

## Implementation Notes

- Uses the OpenAI-compatible endpoint (`/v1/chat/completions`) which is strictly more feature-complete than the Anthropic-compatible endpoint.
- Supports thinking/reasoning via the `thinking` parameter and `reasoning_content` in responses.
- Supports tool calling including a built-in `web_search` tool.
- Supports multimodal input (image, audio, video) for applicable models.
- Supports TTS output for `mimo-v2.5-tts*` and `mimo-v2-tts` models.
- Supports `response_format` for structured JSON output.
- Available models: `mimo-v2.5-pro`, `mimo-v2.5`, `mimo-v2-pro`, `mimo-v2-omni`, `mimo-v2-flash`, plus TTS models.
