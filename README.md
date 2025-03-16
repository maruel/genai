# genai

The _high performance_ low level native Go client for LLMs.

| Provider                                                    | Country | Chat | Streaming | Vision | JSON output | JSON schema | Seed | Tools |
| ----------------------------------------------------------- | ------- | ---- | --------- | ------ | ----------- | ----------- | ---- | ----- |
| [Anthropic](https://console.anthropic.com/settings/billing) | 🇺🇸    | ✅   | ✅        | ✅     | ❌          | ❌          | ❌   | ✅    |
| [Cerebras](https://cloud.cerebras.ai)                       | 🇺🇸    | ✅   | ✅        | ❌     | ✅          | ✅          | ✅   | ✅    |
| [Cloudflare Workers AI](https://dash.cloudflare.com)        | 🇺🇸    | ✅   | ✅        | ⏳     | ✅          | ✅          | ✅   | ✅    |
| [Cohere](https://dashboard.cohere.com/billing)              | 🇨🇦    | ✅   | ✅        | ⏳     | ✅          | ✅          | ✅   | ✅    |
| [DeepSeek](https://platform.deepseek.com)                   | 🇨🇳    | ✅   | ✅        | ❌     | ✅          | ❌          | ❌   | ✅    |
| [Google's Gemini](http://aistudio.google.com)               | 🇺🇸    | ✅   | ✅        | ✅     | ✅          | ✅          | ✅   | ✅    |
| [Groq](https://console.groq.com/dashboard/usage)            | 🇺🇸    | ✅   | ✅        | ✅     | ✅          | ❌          | ✅   | ✅    |
| [HuggingFace](https://huggingface.co/settings/billing)      | 🇺🇸    | ✅   | ✅        | ⏳     | ⏳          | ⏳          | ✅   | ✅    |
| [Mistral](https://console.mistral.ai/usage)                 | 🇫🇷    | ✅   | ✅        | ✅     | ✅          | ✅          | ✅   | ✅    |
| [OpenAI](https://platform.openai.com/usage)                 | 🇺🇸    | ✅   | ✅        | ✅     | ✅          | ✅          | ✅   | ✅    |
| [Perplexity](https://www.perplexity.ai/settings/api)        | 🇺🇸    | ✅   | ✅        | ❌     | ❌          | ⏳          | ❌   | ❌    |
| [TogetherAI](https://api.together.ai/settings/billing)      | 🇺🇸    | ✅   | ✅        | ✅     | ✅          | ✅          | ✅   | ✅    |
| [llama.cpp](https://github.com/ggml-org/llama.cpp)          | N/A     | ✅   | ✅        | ⏳     | ⏳          | ⏳          | ✅   | ⏳    |

- ✅ Implemented
- ⏳ To be implemented
- ❌ Not supported
- Streaming: chat streaming
- Vision: ability to process an image as input
- JSON output/schema: ability to output JSON in free form or with a schema
- Seed: deterministic seed for reproducibility
- Tools: tool calling

## Features

- **Safe and strict API implementation**. All you love from a statically typed
  language. Immediately fail on unknown RPC fields. Error code paths are
  properly implemented.
- **Stateless*: no global state, clients are safe to use concurrently lock-less.
- **Professional grade**: unit tested on live services.
- **Optimized for speed**: minimize memory allocations, compress data at the transport layer when possible.
- **Lean**: Very few dependencies. No unnecessary abstraction layer.
- **Full functionality**: Full access to each backend-specific functionality.
- **JSON schema**: Validate the JSON output against a Go struct you provided.
- Easy to add new providers.
- Implementation is in flux. :) For example, tool call may not work in stream mode yet.

[![Go Reference](https://pkg.go.dev/badge/github.com/maruel/genai/.svg)](https://pkg.go.dev/github.com/maruel/genai/)
[![codecov](https://codecov.io/gh/maruel/genai/graph/badge.svg?token=VLBH363B6N)](https://codecov.io/gh/maruel/genai)


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
- Running [llama.cpp](https://github.com/ggml-org/llama.cpp) locally is free. :)

TODO: Investigate providers at https://github.com/cheahjs/free-llm-api-resources

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
| HuggingFace | gzip, br or zstd                   | none                |
| Mistral     | none                               | br                  |
| OpenAI      | none                               | br                  |
| Perplexity  | none                               | none                |

It matters if you care about your ingress/egress bandwidth. Only HuggingFace
supports brotli and zstd as POST data but replies uncompressed (!). Google
supports gzip.


## Look and feel

```go
package main

import (
	"context"
	"fmt"
	"log"
	"strings"

	"github.com/invopop/jsonschema"
	"github.com/maruel/genai/cerebras"
	"github.com/maruel/genai/genaiapi"
)

func main() {
    c, err := cerebras.New("", "llama3.1-8b")
    if err != nil {
        log.Fatal(err)
    }
    msgs := []genaiapi.Message{
        {
            Role: genaiapi.User,
            Type: genaiapi.Text,
            Text: "Is a circle round? Reply as JSON.",
        },
    }
    var expected struct {
        Round bool `json:"round"`
    }
    opts := genaiapi.CompletionOptions{
        Seed:        1,
        Temperature: 0.01,
        MaxTokens:   50,
        JSONSchema: jsonschema.Reflect(expected),
    }
    resp, err := c.Completion(context.Background(), msgs, &opts)
    if err != nil {
        log.Fatal(err)
    }
    if err := resp.Decode(&expected); err != nil {
        log.Fatal(err)
    }
    fmt.Printf("Round: %v\n", expected.Round)
}
```


## TODO

- Tools
- Audio out
- Video out
- Batch
- Tuning
- Embeddings
- RAG
- Handle rate limiting
- Moderation
- Thinking
- Content Blocks
- Have the message decode the json.
- Pass tool in the message so no need to create the tool definition's JSONSchema
