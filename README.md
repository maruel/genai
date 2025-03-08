# genai

Barebone generic LLM API client for Go. Has very little dependencies.

Implements support for:
- Anthropic
- Cohere
- DeepSeek
- Google's Gemini
- Groq
- Mistral
- OpenAI
- llama.cpp

Implementation is in flux.

[![Go Reference](https://pkg.go.dev/badge/github.com/maruel/genai/.svg)](https://pkg.go.dev/github.com/maruel/genai/)
[![codecov](https://codecov.io/gh/maruel/genai/graph/badge.svg?token=VLBH363B6N)](https://codecov.io/gh/maruel/genai)


## I'm poor 💸

As of March 2025, the following services offer a free tier (other limits
apply):

- [Cohere](https://docs.cohere.com/docs/rate-limits) (1000 RPCs/month)
- [Google's Gemini](https://ai.google.dev/gemini-api/docs/rate-limits) 0.25qps, 1m tokens/month
- [Groq](https://console.groq.com/docs/rate-limits) 0.5qps, 500k tokens/day
- [Mistral](https://help.mistral.ai/en/articles/225174-what-are-the-limits-of-the-free-tier) 1qps, 1B tokens/month
- Running [llama.cpp](https://github.com/ggml-org/llama.cpp) locally is free. :)


## Fun stats

As for March 2025, here's the HTTP compression supported by each provider:

| Provider | Compression (in/out) |
|----------|-------------|
| Anthropic | none/gzip |
| Cohere | none/none |
| DeepSeek | none/gzip |
| Google's Gemini | gzip/gzip |
| Groq | none/br |
| Mistral | none/br |
| OpenAI | none/br |

None support zstd. brotli is popular. Only Google support compressed HTTP POST requests!
