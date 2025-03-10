# genai

Barebone generic LLM API client for Go. Has very little dependencies.

Implements support for:
- Anthropic
- Cloudflare Workers AI
- Cohere
- DeepSeek
- Google's Gemini
- Groq
- HuggingFace
- Mistral
- OpenAI
- Perplexity
- llama.cpp

Implementation is in flux.

[![Go Reference](https://pkg.go.dev/badge/github.com/maruel/genai/.svg)](https://pkg.go.dev/github.com/maruel/genai/)
[![codecov](https://codecov.io/gh/maruel/genai/graph/badge.svg?token=VLBH363B6N)](https://codecov.io/gh/maruel/genai)


## I'm poor 💸

As of March 2025, the following services offer a free tier (other limits
apply):

- [Cloudflare Workers AI](https://developers.cloudflare.com/workers-ai/platform/pricing/) about 10k tokens/day
- [Cohere](https://docs.cohere.com/docs/rate-limits) (1000 RPCs/month)
- [Google's Gemini](https://ai.google.dev/gemini-api/docs/rate-limits) 0.25qps, 1m tokens/month
- [Groq](https://console.groq.com/docs/rate-limits) 0.5qps, 500k tokens/day
- [HuggingFace](https://huggingface.co/docs/api-inference/pricing) 10¢/month
- [Mistral](https://help.mistral.ai/en/articles/225174-what-are-the-limits-of-the-free-tier) 1qps, 1B tokens/month
- Running [llama.cpp](https://github.com/ggml-org/llama.cpp) locally is free. :)

TODO: Investigate providers at https://github.com/cheahjs/free-llm-api-resources

## Fun stats

As for March 2025, here's the HTTP POST compression supported by each provider:

| Provider | Compression accepted for POST data | Response compressed as |
|----------|-------------|-------------|
| Anthropic | none | gzip |
| Cloudflare Workers AI | none | gzip |
| Cohere | none | none |
| DeepSeek | none | gzip |
| Google's Gemini | gzip | gzip |
| Groq | none | br |
| HuggingFace | gzip&br&zstd | none |
| Mistral | none | br |
| OpenAI | none | br |
| Perplexity | none | none |

It may matter if you care about your egress bandwidth. Only HuggingFace
supports brotli and zstd as POST data but replies uncompressed (!). Google
supports gzip.


## TODO

- Schema
- Tools
- Files
    - PDF in
    - Audio in
    - Video in
- Audio out
- Video out
- Batch
- Tuning
- Embeddings
- RAG
