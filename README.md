# genai

The _high performance_ low level native Go client for LLMs.

| Provider                                                    | Country | Chat | Streaming | Vision | PDF | Audio | Video | JSON output | JSON schema | Seed | Tools | Caching |
| ----------------------------------------------------------- | ------- | ---- | --------- | ------ | --- | ----- | ----- | ----------- | ----------- | ---- | ----- | ------- |
| [Anthropic](https://console.anthropic.com/settings/billing) | 🇺🇸      | ✅   | ✅        | ✅     | ✅  | ❌    | ❌    | ❌          | ❌          | ❌   | ✅    | ⏳      |
| [Cerebras](https://cloud.cerebras.ai)                       | 🇺🇸      | ✅   | ✅        | ❌     | ❌  | ❌    | ❌    | ✅          | ✅          | ✅   | ✅    | ❌      |
| [Cloudflare Workers AI](https://dash.cloudflare.com)        | 🇺🇸      | ✅   | ✅        | ⏳     | ❌  | ⏳    | ❌    | ✅          | ✅          | ✅   | ✅    | ❌      |
| [Cohere](https://dashboard.cohere.com/billing)              | 🇨🇦      | ✅   | ✅        | ⏳     | ❌  | ❌    | ❌    | ✅          | ✅          | ✅   | ✅    | ❌      |
| [DeepSeek](https://platform.deepseek.com)                   | 🇨🇳      | ✅   | ✅        | ❌     | ❌  | ❌    | ❌    | ✅          | ❌          | ❌   | ✅    | ⏳      |
| [Google's Gemini](http://aistudio.google.com)               | 🇺🇸      | ✅   | ✅        | ✅     | ✅  | ✅    | ✅    | ✅          | ✅          | ✅   | ✅    | ✅      |
| [Groq](https://console.groq.com/dashboard/usage)            | 🇺🇸      | ✅   | ✅        | ✅     | ❌  | ❌    | ❌    | ✅          | ❌          | ✅   | ✅    | ❌      |
| [HuggingFace](https://huggingface.co/settings/billing)      | 🇺🇸      | ✅   | ✅        | ⏳     | ⏳  | ❌    | ❌    | ⏳          | ⏳          | ✅   | ✅    | ❌      |
| [llama.cpp](https://github.com/ggml-org/llama.cpp)          | N/A     | ✅   | ✅        | ⏳     | ⏳  | ⏳    | ⏳    | ⏳          | ⏳          | ✅   | ⏳    | 🔁      |
| [Mistral](https://console.mistral.ai/usage)                 | 🇫🇷      | ✅   | ✅        | ✅     | ✅  | ❌    | ❌    | ✅          | ✅          | ✅   | ✅    | ❌      |
| [Ollama](https://ollama.com/)                               | N/A     | ✅   | ✅        | ✅     | ❌  | ❌    | ❌    | ❌          | ✅          | ✅   | ✅    | 🔁      |
| [OpenAI](https://platform.openai.com/usage)                 | 🇺🇸      | ✅   | ✅        | ✅     | ✅  | ✅    | ❌    | ✅          | ✅          | ✅   | ✅    | [🔁](https://platform.openai.com/docs/guides/prompt-caching) |
| [Perplexity](https://www.perplexity.ai/settings/api)        | 🇺🇸      | ✅   | ✅        | ❌     | ❌  | ❌    | ❌    | ❌          | ⏳          | ❌   | ❌    | ❌      |
| [TogetherAI](https://api.together.ai/settings/billing)      | 🇺🇸      | ✅   | ✅        | ✅     | ❌  | ❌    | ✅    | ✅          | ✅          | ✅   | ✅    | ❌      |

- ✅ Implemented
- ⏳ To be implemented
- ❌ Not supported
- 🔁 Implicitly supported
- Streaming: chat streaming
- Vision: ability to process an image as input; most providers support PNG, JPG, WEBP and non-animated GIF
- Video: ability to process a video (e.g. MP4) as input.
- PDF: ability to process a PDF as input, possibly with OCR
- JSON output/schema: ability to output JSON in free form or with a schema
- Seed: deterministic seed for reproducibility
- Tools: tool calling

## Features

- **Full functionality**: Full access to each backend-specific functionality.
  Access the raw API if needed with full message schema as Go structs.
- **Native JSON struct serialization**: Pass a struct to tell the LLM what to
  generate, decode the reply into your struct. No need to manually fiddle with
  JSON. Supports required fields, enums, descriptions, etc.
- **Native tool calling**: Tell the LLM to call a tool directly, described a Go
  struct. No need to manually fiddle with JSON.
- **Streaming**: Streams completion reply as the output is being generated.
- **Vision**: Process images, PDFs and videos (!) as input.
- **Unit testing friendly**: record and play back API calls at HTTP level.

Implementation is in flux. :)

[![Go Reference](https://pkg.go.dev/badge/github.com/maruel/genai/.svg)](https://pkg.go.dev/github.com/maruel/genai/)
[![codecov](https://codecov.io/gh/maruel/genai/graph/badge.svg?token=VLBH363B6N)](https://codecov.io/gh/maruel/genai)

## Design

- **Safe and strict API implementation**. All you love from a statically typed
  language. Immediately fails on unknown RPC fields. Error code paths are
  properly implemented.
- **Stateless*: no global state, clients are safe to use concurrently lock-less.
- **Professional grade**: unit tested on live services.
- **Optimized for speed**: minimize memory allocations, compress data at the
  transport layer when possible.
- **Lean**: Few dependencies. No unnecessary abstraction layer.
- Easy to add new providers.


## I'm poor 💸

As of March 2025, the following services offer a free tier (other limits
apply):

- [Cerebras](https://cerebras.ai/inference) has unspecified "generous" free tier
- [Cloudflare Workers AI](https://developers.cloudflare.com/workers-ai/platform/pricing/) about 10k tokens/day
- [Cohere](https://docs.cohere.com/docs/rate-limits) (1000 RPCs/month)
- [Google's Gemini](https://ai.google.dev/gemini-api/docs/rate-limits) 0.25qps, 1m tokens/month
- [Groq](https://console.groq.com/docs/rate-limits) 0.5qps, 500k tokens/day
- [HuggingFace](https://huggingface.co/docs/api-inference/pricing) 10¢/month
- [Mistral](https://help.mistral.ai/en/articles/225174-what-are-the-limits-of-the-free-tier) 1qps, 1B tokens/month
- [Together.AI](https://api.together.ai/settings/plans) provides many models for free at 1qps
- Running [Ollama](https://ollama.com/) or [llama.cpp](https://github.com/ggml-org/llama.cpp) locally is free. :)


## HTTP transport compression

Each service provider was manually tested to see if the accept compressed POST body.

As for March 2025, here's the HTTP POST compression supported by each provider:

| Provider    | Compression accepted for POST data | Response compressed |
| ----------- | ---------------------------------- | ------------------- |
| Anthropic   | none                               | gzip                |
| Cerebras    | none                               | none                |
| Cloudflare Workers AI | none                     | gzip                |
| Cohere      | none                               | none                |
| DeepSeek    | none                               | gzip                |
| Google's Gemini | gzip                           | gzip                |
| Groq        | none                               | br                  |
| HuggingFace | none                               | none                |
| Mistral     | none                               | br                  |
| OpenAI      | none                               | br                  |
| Perplexity  | none                               | none                |

It matters if you care about your ingress/egress bandwidth. Only HuggingFace
supports brotli and zstd as POST data but replies uncompressed (!). Google
supports gzip.


## Look and feel

### Decoding answer as a typed struct

Tell the LLM to use a specific JSON schema to generate the response.

```go
package main

import (
	"context"
	"fmt"
	"log"
	"strings"

	"github.com/maruel/genai/cerebras"
	"github.com/maruel/genai/genai"
)

type Circle struct {
    Round bool `json:"round"`
}

func main() {
    c, err := cerebras.New("", "llama3.1-8b")
    if err != nil {
        log.Fatal(err)
    }
    msgs := genai.Messages{
        genai.NewTextMessage(genai.User, "Is a circle round? Reply as JSON."),
    }
    opts := genai.ChatOptions{
        Seed:        1,
        Temperature: 0.01,
        MaxTokens:   50,
        DecodeAs:    &Circle{},
    }
    resp, err := c.Chat(context.Background(), msgs, &opts)
    if err != nil {
        log.Fatal(err)
    }
    got := Circle{}
    if err := resp.Contents[0].Decode(&got); err != nil {
        log.Fatal(err)
    }
    fmt.Printf("Round: %v\n", got.Round)
}
```


## Models

Snapshot of all the supported models: [MODELS.md](MODELS.md).

Try it:

```bash
go install github.com/maruel/genai/cmd/...@latest
list-models -provider hugginface
```


## TODO

- Audio out
- Video out
- Batch
- Tuning
- Embeddings
- Handle rate limiting
- Moderation
- Thinking
- Content Blocks
- Citations
