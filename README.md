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

<details>
  <summary>Click here for legend of ✅ ⏳ ❌ 🔁 and columns</summary>

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

</details>


## Features

- **Full functionality**: Full access to each backend-specific functionality.
  Access the raw API if needed with full message schema as Go structs.
- **Native tool calling**: Tell the LLM to call a tool directly, described a Go
  struct. No need to manually fiddle with JSON.
- **Native JSON struct serialization**: Pass a struct to tell the LLM what to
  generate, decode the reply into your struct. No need to manually fiddle with
  JSON. Supports required fields, enums, descriptions, etc.
- **Streaming**: Streams completion reply as the output is being generated, including thinking and tool
  calling.
- **Vision**: Process images, PDFs and videos (!) as input.
- **Unit testing friendly**: record and play back API calls at HTTP level.

Implementation is in flux. :)

[![Go Reference](https://pkg.go.dev/badge/github.com/maruel/genai/.svg)](https://pkg.go.dev/github.com/maruel/genai/)
[![codecov](https://codecov.io/gh/maruel/genai/graph/badge.svg?token=VLBH363B6N)](https://codecov.io/gh/maruel/genai)

## Design

- **Safe and strict API implementation**. All you love from a statically typed
  language. Immediately fails on unknown RPC fields. Error code paths are
  properly implemented.
- **Stateless**: no global state, clients are safe to use concurrently lock-less.
- **Professional grade**: unit tested on live services.
- **Optimized for speed**: minimize memory allocations, compress data at the
  transport layer when possible. Groq, Mistral and OpenAI use brotli for HTTP compression instead of gzip.
- **Lean**: Few dependencies. No unnecessary abstraction layer.
- Easy to add new providers.


## I'm poor 💸

<details>
  <summary>Click here for a list of providers offering free quota</summary>

As of May 2025, the following services offer a free tier (other limits
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

</details>


## Look and feel


### Tool calling using predefined tool

Tool calling is the idea that a LLM can't know it call, so we provide ways for the LLM to get more knowledge
on the fly or to have side-effects. It unblocks a whole realm of possibilities. Our design enables dense
strongly typed code that favorably compares to python.

```go
package main

import (
	"context"
	"fmt"
	"log"

	"github.com/maruel/genai"
	"github.com/maruel/genai/genaitools"
	"github.com/maruel/genai/groq"
)

func main() {
	c, err := groq.New("", "llama3-8b-8192")
	if err != nil {
		log.Fatal(err)
	}
	msgs := genai.Messages{
		genai.NewTextMessage(genai.User, "What is 3214 + 5632? Leverage the tool available to you to tell me the answer. Do not explain. Be terse. Include only the answer."),
	}
	opts := genai.ChatOptions{
		Tools: []genai.ToolDef{genaitools.Arithmetic},
		// Force the LLM to do a tool call.
		ToolCallRequest: genai.ToolCallRequired,
	}
	resp, err := c.Chat(context.Background(), msgs, &opts)
	if err != nil {
		log.Fatal(err)
	}

	// Add the assistant's message to the messages list.
	msgs = append(msgs, resp.Message)

	// Process the tool call from the assistant.
	msg, err := resp.DoToolCalls(opts.Tools)
	if err != nil {
		log.Fatalf("Error calling tool: %v", err)
	}
	if msg.IsZero() {
		log.Fatal("Expected a tool call")
	}

	// Add the tool call response to the messages list.
	msgs = append(msgs, msg)

	// Follow up so the LLM can interpret the tool call response. Tell the LLM to not do a tool call this time.
	opts.ToolCallRequest = genai.ToolCallNone
	resp, err = c.Chat(context.Background(), msgs, &opts)
	if err != nil {
		log.Fatal(err)
	}

	// Print the result.
	fmt.Println(resp.AsText())
}
```

<details>
  <summary>Click to see more examples!</summary>

### Tool calling using a fully custom tool

This example provides all the details to implement a complete custom tool.

```go
package main

import (
	"context"
	"fmt"
	"log"

	"github.com/maruel/genai"
	"github.com/maruel/genai/groq"
)

func main() {
	c, err := groq.New("", "llama3-8b-8192")
	if err != nil {
		log.Fatal(err)
	}
	type math struct {
		A int `json:"a"`
		B int `json:"b"`
	}
	msgs := genai.Messages{
		genai.NewTextMessage(genai.User, "What is 3214 + 5632? Call the tool \"add\" to tell me the answer. Do not explain. Be terse. Include only the answer."),
	}
	opts := genai.ChatOptions{
		Tools: []genai.ToolDef{
			{
				Name:        "add",
				Description: "Add two numbers together and provides the result",
				Callback: func(input *math) (string, error) {
					return fmt.Sprintf("%d", input.A+input.B), nil
				},
			},
		},
		// Force the LLM to do a tool call.
		ToolCallRequest: genai.ToolCallRequired,
	}
	resp, err := c.Chat(context.Background(), msgs, &opts)
	if err != nil {
		log.Fatal(err)
	}

	// Add the assistant's message to the messages list.
	msgs = append(msgs, resp.Message)

	// Process the tool call from the assistant.
	msg, err := resp.DoToolCalls(opts.Tools)
	if err != nil {
		log.Fatalf("Error calling tool: %v", err)
	}
	if msg.IsZero() {
		log.Fatal("Expected a tool call")
	}

	// Add the tool call response to the messages list.
	msgs = append(msgs, msg)

	// Follow up so the LLM can interpret the tool call response. Tell the LLM to not do a tool call this time.
	opts.ToolCallRequest = genai.ToolCallNone
	resp, err = c.Chat(context.Background(), msgs, &opts)
	if err != nil {
		log.Fatal(err)
	}

	// Print the result.
	fmt.Println(resp.AsText())
}
```


### Decoding answer as a typed struct

Tell the LLM to use a specific JSON schema to generate the response. This is more lightweight than tool
calling. It is very useful when we want the LLM to make a choice between values, to return a number or a
boolean (true/false).

```go
package main

import (
	"context"
	"fmt"
	"log"

	"github.com/maruel/genai"
	"github.com/maruel/genai/cerebras"
)

func main() {
	c, err := cerebras.New("", "llama3.1-8b")
	if err != nil {
		log.Fatal(err)
	}
	msgs := genai.Messages{
		genai.NewTextMessage(genai.User, "Is a circle round? Reply as JSON."),
	}
	var circle struct {
		Round bool `json:"round"`
	}
	opts := genai.ChatOptions{DecodeAs: &circle}
	resp, err := c.Chat(context.Background(), msgs, &opts)
	if err != nil {
		log.Fatal(err)
	}
	if err := resp.Decode(&circle); err != nil {
		log.Fatal(err)
	}
	fmt.Printf("Round: %v\n", circle.Round)
}
```

</details>


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
- Moderation
- Citations
