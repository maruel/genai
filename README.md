# genai

The opinionated high performance professional-grade AI package for Go.

genai is _different_. Curious why it was created? See the release announcement at
[maruel.ca/post/genai-v0.1.0](https://maruel.ca/post/genai-v0.1.0).


## Features

- **Full functionality**: Full access to each backend-specific functionality.
  Access the raw API if needed with full message schema as Go structs.
- **Tool calling via reflection**: Tell the LLM to call a tool directly, described a Go
  struct. No need to manually fiddle with JSON.
- **Native JSON struct serialization**: Pass a struct to tell the LLM what to
  generate, decode the reply into your struct. No need to manually fiddle with
  JSON. Supports required fields, enums, descriptions, etc.
- **Streaming**: Streams completion reply as the output is being generated, including thinking and tool
  calling.
- **Multi-modal**: Process images, PDFs and videos (!) as input or output.
- **Unit testing friendly**: record and play back API calls at HTTP level to save 💰 and keep tests fast and
  reproducible, via the exposed HTTP transport. See [example](https://pkg.go.dev/github.com/maruel/genai/providers/anthropic#example-New-HTTP_record).
- **Rate limits and usage**: Parse the provider-specific HTTP headers and JSON response to get the tokens usage
  and remaining quota.
- Provide access to HTTP headers to enable [beta features](https://pkg.go.dev/github.com/maruel/genai#example-package-GenSyncWithToolCallLoop_with_custom_HTTP_Header).

[![Go Reference](https://pkg.go.dev/badge/github.com/maruel/genai/.svg)](https://pkg.go.dev/github.com/maruel/genai/)
[![codecov](https://codecov.io/gh/maruel/genai/graph/badge.svg?token=VLBH363B6N)](https://codecov.io/gh/maruel/genai)


## Design

- **Safe and strict API implementation**. All you love from a statically typed
  language. The library's smoke tests immediately fail on unknown RPC fields. Error code paths are properly
  implemented.
- **Stateless**. No global state, it is safe to use clients concurrently.
- **Professional grade**. smoke tested on live services with recorded traces located in `testdata/`
  directories, e.g.
  [providers/anthropic/testdata/TestClient_Scoreboard/](https://github.com/maruel/genai/tree/main/providers/anthropic/testdata/TestClient_Scoreboard/).
- **Trust, But verify**. It generates a [scoreboard](#scoreboard) based on actual behavior from each provider.
- **Optimized for speed**. Minimize memory allocations, compress data at the
  transport layer when possible. Groq, Mistral and OpenAI use brotli for HTTP compression instead of gzip,
  and POST's body to Google are gzip compressed.
- **Lean**: Few dependencies. No unnecessary abstraction layer.


## Scoreboard

| Provider                                                    | 🌐   | ➛In      | Out➛   | JSON | Schema | Chat   | Stream | Tool   | Batch | Seed | File | Cite | Think | Probs | Limits |
| ----------------------------------------------------------- | ---- | -------- | ------ | ---- | ------ | ------ | ------ | ------ | ----- | ---- | ---- | ---- | ----- | ----- | ------ |
| [anthropic](https://console.anthropic.com/settings/billing) | 🇺🇸   | 💬📄📸   | 💬     | ❌   | ❌     | ✅🤪   | ✅🤪   | ✅🧐   | ✅    | ❌   | ❌   | ✅   | ✅    | ❌    | ✅     |
| [bfl](https://dashboard.bfl.ai/)                            | 🇩🇪   | 💬       | 📸     | ❌   | ❌     | ❌     | ❌     | ❌     | ✅    | ✅   | ❌   | ❌   | ❌    | ❌    | ✅     |
| [cerebras](https://cloud.cerebras.ai)                       | 🇺🇸   | 💬       | 💬     | 🤪   | 🤪     | ✅     | ✅     | 💨🧐   | ❌    | ✅   | ❌   | ❌   | ✅    | ✅    | ✅     |
| [cloudflare](https://dash.cloudflare.com)                   | 🇺🇸   | 💬       | 💬     | 🤪   | ✅     | ✅🚩🤪 | ✅🚩🤪 | 💨     | ❌    | ✅   | ❌   | ❌   | ❌    | ❌    | ❌     |
| [cohere](https://dashboard.cohere.com/billing)              | 🇨🇦   | 💬📸     | 💬     | ✅   | ✅     | ✅     | ✅     | ✅💥   | ❌    | ✅   | ❌   | ✅   | ✅    | ✅    | ❌     |
| [deepseek](https://platform.deepseek.com)                   | 🇨🇳   | 💬       | 💬     | ✅   | ❌     | ✅     | ✅     | ✅💥   | ❌    | ❌   | ❌   | ❌   | ✅    | ✅    | ❌     |
| [gemini](http://aistudio.google.com)                        | 🇺🇸   | 🎤💬📄📸 | 💬📸   | ✅   | ✅     | ✅     | ✅     | ✅🧐   | ✅    | ✅   | ✅   | ❌   | ✅    | ✅    | ❌     |
| [groq](https://console.groq.com/dashboard/usage)            | 🇺🇸   | 💬📸     | 💬     | ✅   | ❌     | ✅     | ✅     | 💨🧐   | ❌    | ✅   | ❌   | ❌   | ✅    | ❌    | ✅     |
| [huggingface](https://huggingface.co/settings/billing)      | 🇺🇸   | 💬       | 💬     | ✅   | ❌     | ✅     | ✅     | 💨     | ❌    | ✅   | ❌   | ❌   | ✅    | ✅    | ✅     |
| [llamacpp](https://github.com/ggml-org/llama.cpp)           | 🏠   | 💬📸     | 💬     | ✅   | ✅     | ✅     | ✅     | ✅🧐   | ❌    | ✅   | ❌   | ❌   | ❌    | ✅    | ❌     |
| [mistral](https://console.mistral.ai/usage)                 | 🇫🇷   | 🎤💬📄📸 | 💬     | ✅   | ✅     | ✅     | ✅     | ✅🧐   | ❌    | ✅   | ❌   | ❌   | ❌    | ❌    | ✅     |
| [ollama](https://ollama.com/)                               | 🏠   | 💬📸     | 💬     | ✅   | ✅     | ✅     | ✅     | ✅     | ❌    | ✅   | ❌   | ❌   | ✅    | ❌    | ❌     |
| [openai](https://platform.openai.com/usage)                 | 🇺🇸   | 🎤💬📄📸 | 💬📸   | ✅   | ✅     | ✅🤪   | ✅🤪   | ✅💥🧐 | ✅    | ✅   | ✅   | ❌   | ✅    | ✅    | ✅     |
| [openairesponses](https://platform.openai.com/usage)        | 🇺🇸   | 💬📄📸   | 💬📸   | ✅   | ✅     | ✅💸🤪 | ✅💸🤪 | ✅🧐   | ❌    | ✅   | ❌   | ❌   | ✅    | ❌    | ✅     |
| [perplexity](https://www.perplexity.ai/settings/api)        | 🇺🇸   | 💬📸     | 💬     | ❌   | ✅     | ✅🤪   | ✅🤪   | ❌     | ❌    | ❌   | ❌   | ✅   | ✅    | ❌    | ❌     |
| [pollinations](https://auth.pollinations.ai/)               | 🇩🇪   | 💬📸     | 💬📸   | 🤪   | ❌     | ✅🤪   | ✅💸🤪 | ✅🧐   | ❌    | ✅   | ❌   | ❌   | ❌    | ❌    | ❌     |
| [togetherai](https://api.together.ai/settings/billing)      | 🇺🇸   | 💬📸     | 💬📸   | ✅   | ✅     | ✅🚩🤪 | ✅🚩🤪 | 💨🧐   | ❌    | ✅   | ❌   | ❌   | ❌    | ✅    | ✅     |
| openaicompatible                                            | N/A  | 💬       | 💬     | ❌   | ❌     | ✅     | ✅     | ❌     | ❌    | ❌   | ❌   | ❌   | ❌    | ❌    | ❌     |

<details>
  <summary>‼️ Click here for legend of 🏠 ✅ ❌ 💬 📄 📸 🎤 🎥 🤪 💸 🚩 💨 🧐 💥 and columns</summary>

- 🏠: Runs locally.
- ✅: Implemented and works great.
- ❌: Not supported by genai. The provider may support it, but genai does not (yet). Please send a PR to add
  it!
- 💬: Text
- 📄: PDF: process a PDF as input, possibly with OCR.
- 📸: Image
    - Input: process an image as input; most providers support PNG, JPG, WEBP and non-animated GIF
    - Output: generate images
- 🎤: Audio
- 🎥: Video: process a video (e.g. MP4) as input.
- 🤪: Partial support: no MaxTokens or StopSequences, or JSON output is flaky.
- 💸: Usage is not reported: we can't know how many tokens were used.
- 🚩: Broken FinishReason: we can't know if the request was cut off.
- 💨: Tool calling is flaky.
- 🧐: Tool calling is **not** biased towards the first value in an enum. If the provider doesn't have this, be
  mindful of the order of the values!
- 💥: Tool calling is undecided when asked a question that has no clear answer and will call both options
  instead of calling ont at random. This is good.
- 🌐: Country where the company is located.
- JSON and Schema: ability to output JSON in free form, or with a forced schema specified as a Go struct
- Chat: Buffered chat.
- Stream: Streaming output.
- Tool: Tool calling, using [genai.ToolDef](https://pkg.go.dev/github.com/maruel/genai#ToolDef)
- Batch: Process asynchronously batches during off peak hours at a discounts.
- Seed: Deterministic seed for reproducibility.
- File: Upload and store large files.
- Cite: Citation generation. Especially useful for RAG.
- Think: Supports chain-of-thought thinking process.
    - Both redacted (Anthropic, Gemini) and explicit (Deepseek R1, Qwen3, etc).
- Probs: return logprobs. Many do not support this in streaming mode.
- Limits: returns the rate limits, including the remaining quota.

</details>


## I'm poor 💸

<details>
  <summary>‼️ Click here for a list of providers offering free quota</summary>

As of May 2025, the following services offer a free tier (other limits
apply):

- [Cerebras](https://cerebras.ai/inference) has unspecified "generous" free tier
- [Cloudflare Workers AI](https://developers.cloudflare.com/workers-ai/platform/pricing/) about 10k tokens/day
- [Cohere](https://docs.cohere.com/docs/rate-limits) (1000 RPCs/month)
- [Google's Gemini](https://ai.google.dev/gemini-api/docs/rate-limits) 0.25qps, 1m tokens/month
- [Groq](https://console.groq.com/docs/rate-limits) 0.5qps, 500k tokens/day
- [HuggingFace](https://huggingface.co/docs/api-inference/pricing) 10¢/month
- [Mistral](https://help.mistral.ai/en/articles/225174-what-are-the-limits-of-the-free-tier) 1qps, 1B tokens/month
- [Pollinations.ai](https://api.together.ai/settings/plans) provides many models for free, including image
  generation
- [Together.AI](https://api.together.ai/settings/plans) provides many models for free at 1qps, including image
  generation
- Running [Ollama](https://ollama.com/) or [llama.cpp](https://github.com/ggml-org/llama.cpp) locally is free. :)

</details>


## Look and feel

### Sample usage

This selects a good default model based on Anthropic's currently published models, sends a prompt and print
the response as a string. It uses `ANTHROPIC_API_KEY` environment variable to authenticate.

```go
package main

import (
	"context"
	"fmt"
	"log"

	"github.com/maruel/genai"
	"github.com/maruel/genai/providers/anthropic"
)

func main() {
	ctx := context.Background()
	c, err := anthropic.New(ctx, &genai.ProviderOptions{}, nil)
	if err != nil {
		log.Fatal(err)
	}
	msgs := genai.Messages{
		genai.NewTextMessage("Give me a life advice that sounds good but is a bad idea in practice."),
	}
	result, err := c.GenSync(ctx, msgs)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(result.String())
}
```

This may print:

> Quit your job and follow your dreams, no matter the cost.

 💡 Click below for advanced features examples like choosing a provider at runtime, streaming, vision, image
 generation, tool calling and JSON schema.

<details>
  <summary>‼️ Click to see more examples!</summary>


### Any provider

A minimal program that will load a provider by name and send a prompt. The relevant environment variable (e.g.
`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, etc) is used automatically for authentication. Automatically selects a
models on behalf of the user. Supports [ollama](https://ollama.com/) and
[llama-server](https://github.com/ggml-org/llama.cpp) even if they run on a remote host or non-default port.

```go
package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"maps"
	"slices"
	"strings"

	"github.com/maruel/genai"
	"github.com/maruel/genai/adapters"
	"github.com/maruel/genai/providers"
)

func main() {
	ctx := context.Background()
	s := strings.Join(slices.Sorted(maps.Keys(providers.Available(ctx))), ", ")
	if s == "" {
		s = "set environment variables, e.g. `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, etc"
	}
	provider := flag.String("provider", "", "provider to use, "+s)
	model := flag.String("model", "", "model to use; "+genai.ModelCheap+", "+genai.ModelGood+" (default) or "+genai.ModelSOTA+" for automatic model selection")
	remote := flag.String("remote", "", "url to use, e.g. when using ollama or llama-server on another host")
	flag.Parse()

	query := strings.Join(flag.Args(), " ")
	if query == "" {
		log.Fatal("provide a query")
	}
	p, err := LoadProvider(ctx, *provider, &genai.ProviderOptions{Model: *model, Remote: *remote})
	if err != nil {
		log.Fatal(err)
	}
	resp, err := p.GenSync(ctx, genai.Messages{genai.NewTextMessage(query)})
	if err != nil {
		log.Fatalf("failed to use provider %q: %s", *provider, err)
	}
	fmt.Printf("%s\n", resp.String())
}

// LoadProvider loads a provider.
func LoadProvider(ctx context.Context, provider string, opts *genai.ProviderOptions) (genai.Provider, error) {
	if provider == "" {
		return nil, fmt.Errorf("no provider specified")
	}
	f := providers.All[provider]
	if f == nil {
		return nil, fmt.Errorf("unknown provider %q", provider)
	}
	c, err := f(ctx, opts, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to connect to provider %q: %w", provider, err)
	}
	// Wrap the provider with an adapter to process "<think>" tokens automatically ONLY if needed.
	return adapters.WrapThinking(c), nil
}
```


### Tool calling

A LLM can both retrieve information and act on its environment through tool calling. It unblocks a whole realm
of possibilities. Our design enables dense strongly typed code that favorably compares to python.

```go
package main

import (
	"context"
	"fmt"
	"log"

	"github.com/maruel/genai"
	"github.com/maruel/genai/adapters"
	"github.com/maruel/genai/providers/cerebras"
)

func main() {
	ctx := context.Background()
	c, err := cerebras.New(ctx, &genai.ProviderOptions{Model: "qwen-3-235b-a22b-thinking-2507"}, nil)
	if err != nil {
		log.Fatal(err)
	}
	p := adapters.WrapThinking(c)
	type math struct {
		A int `json:"a"`
		B int `json:"b"`
	}
	msgs := genai.Messages{
		genai.NewTextMessage("What is 3214 + 5632? Call the tool \"add\" to tell me the answer. Do not explain. Be terse. Include only the answer."),
	}
	opts := genai.OptionsText{
		Tools: []genai.ToolDef{
			{
				Name:        "add",
				Description: "Add two numbers together and provides the result",
				Callback: func(ctx context.Context, input *math) (string, error) {
					return fmt.Sprintf("%d", input.A+input.B), nil
				},
			},
		},
		// Force the LLM to do a tool call.
		ToolCallRequest: genai.ToolCallRequired,
	}
	resp, err := p.GenSync(ctx, msgs, &opts)
	if err != nil {
		log.Fatal(err)
	}

	// Add the assistant's message to the messages list.
	msgs = append(msgs, resp.Message)

	// Process the tool call from the assistant.
	msg, err := resp.DoToolCalls(ctx, opts.Tools)
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
	resp, err = p.GenSync(ctx, msgs, &opts)
	if err != nil {
		log.Fatal(err)
	}

	// Print the result.
	fmt.Println(resp.String())
}
```


### Generate an image (for free)

Use Together.AI's free (!) image generation albeit with low rate limit. Some providers return an URL that must be
fetched manually within a few minutes or hours, some return the data inline. This example handles both cases.

```go
package main

import (
	"context"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"

	"github.com/maruel/genai"
	"github.com/maruel/genai/providers/togetherai"
)

func main() {
	ctx := context.Background()
	c, err := togetherai.New(ctx, &genai.ProviderOptions{Model: "black-forest-labs/FLUX.1-schnell-Free"}, nil)
	if err != nil {
		log.Fatal(err)
	}
	msgs := genai.Messages{
		genai.NewTextMessage("Carton drawing of a husky playing on the beach."),
	}
	result, err := c.GenSync(ctx, msgs)
	if err != nil {
		log.Fatal(err)
	}
	for _, r := range result.Replies {
		if r.Doc.IsZero() {
			fmt.Println(r.Text)
			continue
		}
		// The image can be returned as an URL or inline, depending on the provider.
		var src io.Reader
		if r.Doc.URL != "" {
			req, err := c.HTTPClient().Get(r.Doc.URL)
			if err != nil {
				log.Fatal(err)
			} else if req.StatusCode != http.StatusOK {
				log.Fatal(req.StatusCode)
			}
			src = req.Body
			defer req.Body.Close()
		} else {
			src = r.Doc.Src
		}
		b, err := io.ReadAll(src)
		if err != nil {
			log.Fatal(err)
		}
		name := r.Doc.GetFilename()
		fmt.Printf("Wrote: %s\n", name)
		if err = os.WriteFile(name, b, 0o644); err != nil {
			log.Fatal(err)
		}
	}
}
```


### Vision and streaming reply

Leverage the content.jpg generated in the previous step to ask another provider to describe the image. The
response is streamed out the console as the reply is generated.

```go
package main

import (
	"context"
	"log"
	"os"

	"github.com/maruel/genai"
	"github.com/maruel/genai/providers/mistral"
)

func main() {
	ctx := context.Background()
	c, err := mistral.New(ctx, &genai.ProviderOptions{}, nil)
	if err != nil {
		log.Fatal(err)
	}
	f, err := os.Open("content.jpg")
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()
	msgs := genai.Messages{
		genai.Message{Requests: []genai.Request{
			{Text: "Extensively describe this image."},
			{Doc: genai.Doc{Src: f}},
		}},
	}
	fragments, finish := c.GenStream(ctx, msgs)
	for f := range fragments {
		os.Stdout.WriteString(f.TextFragment)
	}
	os.Stdout.WriteString("\n")
	if _, err := finish(); err != nil {
		log.Fatal(err)
	}
}
```


### Decoding answer as a typed struct

Tell the LLM to use a specific JSON schema to generate the response. This is much more lightweight than tool
calling! It is very useful when we want the LLM to make a choice between values, to return a number or a
boolean (true/false). Enums are supported.

```go
package main

import (
	"context"
	"fmt"
	"log"

	"github.com/maruel/genai"
	"github.com/maruel/genai/providers/openai"
)

func main() {
	ctx := context.Background()
	c, err := openai.New(ctx, &genai.ProviderOptions{}, nil)
	if err != nil {
		log.Fatal(err)
	}
	msgs := genai.Messages{
		genai.NewTextMessage("Is a circle round? Reply as JSON."),
	}
	var circle struct {
		Round bool `json:"round"`
	}
	opts := genai.OptionsText{DecodeAs: &circle}
	resp, err := c.GenSync(ctx, msgs, &opts)
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

Snapshot of all the supported models at [docs/MODELS.md](docs/MODELS.md) is updated weekly.

Try it:

```bash
go install github.com/maruel/genai/cmd/...@latest

list-models -provider huggingface
```

## Running locally

Use [cmd/llama-serve](cmd/llama-serve) to run a LLM locally, including tool calling and vision!


## TODO

PRs are appreciated for any of the following. No need to ask! Just send a PR and make it pass CI checks. ❤️

### Features

- Authentication: OAuth, service account, OIDC,
  [GITHUB_TOKEN](https://docs.github.com/en/github-models/use-github-models/integrating-ai-models-into-your-development-workflow#using-ai-models-with-github-actions).
- Server-side MCP: [Anthropic](https://docs.anthropic.com/en/docs/agents-and-tools/mcp-connector),
  [OpenAI](https://platform.openai.com/docs/guides/tools-remote-mcp)
- Real-time / Live: [Gemini](https://ai.google.dev/api/live),
  [OpenAI](https://platform.openai.com/docs/guides/realtime),
  [TogetherAI](https://docs.together.ai/docs/text-to-speech), ...
- More comprehensive file/cache abstraction
- Tokens counting: [Anthropic](https://docs.anthropic.com/en/docs/build-with-claude/token-counting),
  [Cohere](https://docs.cohere.com/reference/tokenize), [Gemini](https://ai.google.dev/api/tokens), ...
- Embeddings: [Anthropic](https://docs.anthropic.com/en/docs/build-with-claude/embeddings),
  [Cohere](https://docs.cohere.com/reference/embed), [Gemini](https://ai.google.dev/api/embeddings),
  [OpenAI](https://platform.openai.com/docs/guides/embeddings), [TogetherAI](https://docs.together.ai/docs/embeddings-overview), ...

### Providers

I'm fine with any provider being added, I'm particularly looking forward to these:

- [AWS Bedrock](https://docs.aws.amazon.com/bedrock/latest/APIReference/API_Operations_Amazon_Bedrock.html)
- [Azure AI](https://learn.microsoft.com/en-us/rest/api/aifoundry/model-inference/get-chat-completions/get-chat-completions)
- [Fireworks responses](https://fireworks.ai/docs/guides/response-api)
- [GitHub](https://docs.github.com/en/rest/models/inference) inference API, which works on GitHub Actions (!)
- [Google's Vertex AI](https://cloud.google.com/vertex-ai/docs/reference/rest)
- [Runway](https://docs.dev.runwayml.com/api-details/sdks/)
