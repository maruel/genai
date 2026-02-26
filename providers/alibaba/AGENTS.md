# Alibaba Cloud (DashScope / Model Studio)

- **Documentation**: https://www.alibabacloud.com/help/en/model-studio/
- **OpenAI Compatibility**: https://www.alibabacloud.com/help/en/model-studio/compatibility-of-openai-with-dashscope
- **Chat Completions API**: https://www.alibabacloud.com/help/en/model-studio/qwen-api-via-openai-chat-completions
- **Function Calling**: https://www.alibabacloud.com/help/en/model-studio/qwen-function-calling
- **Models**: https://www.alibabacloud.com/help/en/model-studio/models

## Integration Notes

DashScope exposes an OpenAI-compatible endpoint at:

- International: `https://dashscope-intl.aliyuncs.com/compatible-mode/v1`
- US: `https://dashscope-us.aliyuncs.com/compatible-mode/v1`

No official Go SDK. Alibaba recommends the OpenAI Go SDK for the compatible endpoint.

API key env var: `DASHSCOPE_API_KEY` (keys are region-specific).

## Key Models

- **Text**: `qwen3-max`, `qwen3.5-plus`, `qwen3.5-flash`, `qwen-plus`, `qwen-flash`
- **Vision**: `qwen3-vl-plus`, `qwen3-vl-flash`, `qwen-vl-max`
- **Omni**: `qwen3-omni-flash` (audio/video/image/text)
- **Code**: `qwen3-coder-plus`, `qwen3-coder-flash`

## DashScope-Specific Extensions

Beyond standard OpenAI parameters:

- `enable_thinking` / `thinking_budget`: chain-of-thought mode
- `enable_search` / `search_options`: web search
- `top_k`: top-K sampling
