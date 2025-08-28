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
  calling, via [go 1.23 iterators](https://go.dev/blog/range-functions).
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

| Provider                                                    | 🌐   | ➛In        | Out➛   | JSON | Schema | Chat   | Stream | Tool   | Batch | Seed | File | Cite | Think | Probs | Limits |
| ----------------------------------------------------------- | ---- | ---------- | ------ | ---- | ------ | ------ | ------ | ------ | ----- | ---- | ---- | ---- | ----- | ----- | ------ |
| [anthropic](https://console.anthropic.com/settings/billing) | 🇺🇸   | 💬📄📸     | 💬     | ❌   | ❌     | ✅🤪   | ✅🤪   | ✅🧐   | ✅    | ❌   | ❌   | ✅   | ✅    | ❌    | ✅     |
| [bfl](https://dashboard.bfl.ai/)                            | 🇩🇪   | 💬         | 📸     | ❌   | ❌     | ❌     | ❌     | ❌     | ✅    | ✅   | ❌   | ❌   | ❌    | ❌    | ✅     |
| [cerebras](https://cloud.cerebras.ai)                       | 🇺🇸   | 💬         | 💬     | 🤪   | 🤪     | ✅🤪   | ✅🤪   | ✅🧐   | ❌    | ✅   | ❌   | ❌   | ✅    | ✅    | ✅     |
| [cloudflare](https://dash.cloudflare.com)                   | 🇺🇸   | 💬         | 💬     | 🤪   | ✅     | ✅🚩🤪 | ✅🚩🤪 | ❌     | ❌    | ✅   | ❌   | ❌   | ❌    | ❌    | ❌     |
| [cohere](https://dashboard.cohere.com/billing)              | 🇨🇦   | 💬📸       | 💬     | ✅   | ✅     | ✅     | ✅     | ✅💥   | ❌    | ✅   | ❌   | ✅   | ✅    | ✅    | ❌     |
| [deepseek](https://platform.deepseek.com)                   | 🇨🇳   | 💬         | 💬     | ✅   | ❌     | ✅     | ✅     | ✅💥🧐 | ❌    | ❌   | ❌   | ❌   | ✅    | ✅    | ❌     |
| [gemini](http://aistudio.google.com)                        | 🇺🇸   | 🎤🎥💬📄📸 | 💬📸   | ✅   | ✅     | ✅     | ✅     | ✅💥🧐 | ✅    | ✅   | ✅   | ❌   | ✅    | ✅    | ❌     |
| [groq](https://console.groq.com/dashboard/usage)            | 🇺🇸   | 💬📸       | 💬     | ✅   | ❌     | ✅     | ✅     | ✅💥🧐 | ❌    | ✅   | ❌   | ❌   | ✅    | ❌    | ✅     |
| [huggingface](https://huggingface.co/settings/billing)      | 🇺🇸   | 💬         | 💬     | ✅   | ❌     | ✅     | ✅     | ❌     | ❌    | ✅   | ❌   | ❌   | ✅    | ✅    | ✅     |
| [llamacpp](https://github.com/ggml-org/llama.cpp)           | 🏠   | 💬📸       | 💬     | ✅   | ✅     | ✅     | ✅     | ✅🧐   | ❌    | ✅   | ❌   | ❌   | ❌    | ✅    | ❌     |
| [mistral](https://console.mistral.ai/usage)                 | 🇫🇷   | 🎤💬📄📸   | 💬     | ✅   | ✅     | ✅     | ✅     | ✅🧐   | ❌    | ✅   | ❌   | ❌   | ❌    | ❌    | ✅     |
| [ollama](https://ollama.com/)                               | 🏠   | 💬📸       | 💬     | ✅   | ✅     | ✅     | ✅     | 💨     | ❌    | ✅   | ❌   | ❌   | ✅    | ❌    | ❌     |
| [openai](https://platform.openai.com/usage)                 | 🇺🇸   | 🎤💬📄📸   | 💬📸   | ✅   | ✅     | ✅🤪   | ✅🤪   | ✅💥🧐 | ✅    | ✅   | ✅   | ❌   | ✅    | ✅    | ✅     |
| [openairesponses](https://platform.openai.com/usage)        | 🇺🇸   | 💬📄📸     | 💬📸   | ✅   | ✅     | ✅💸🤪 | ✅💸🤪 | ✅🧐   | ❌    | ✅   | ❌   | ❌   | ✅    | ❌    | ✅     |
| [perplexity](https://www.perplexity.ai/settings/api)        | 🇺🇸   | 💬📸       | 💬     | ❌   | ✅     | ✅🤪   | ✅🤪   | ❌     | ❌    | ❌   | ❌   | ✅   | ✅    | ❌    | ❌     |
| [pollinations](https://auth.pollinations.ai/)               | 🇩🇪   | 💬📸       | 💬📸   | 🤪   | ❌     | ✅🤪   | ✅💸🤪 | ✅💥🧐 | ❌    | ✅   | ❌   | ❌   | ❌    | ❌    | ❌     |
| [togetherai](https://api.together.ai/settings/billing)      | 🇺🇸   | 💬📸       | 💬📸   | ✅   | ✅     | ✅🚩🤪 | ✅🚩🤪 | ✅💥🧐 | ❌    | ✅   | ❌   | ❌   | ❌    | ✅    | ✅     |
| openaicompatible                                            | N/A  | 💬         | 💬     | ❌   | ❌     | ✅     | ✅     | ❌     | ❌    | ❌   | ❌   | ❌   | ❌    | ❌    | ❌     |

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


## Examples

The following examples intentionally use a variety of providers to show the extent at which you can pick and
chose.

### Text Basic ✅

[examples/txt\_to\_txt\_sync/main.go](examples/txt_to_txt_sync/main.go): This selects a good default model based
on Anthropic's currently published models, sends a prompt and prints the response as a string.

This requires [`ANTHROPIC_API_KEY`](https://console.anthropic.com/settings/keys) environment variable to
authenticate.

```go
func main() {
	ctx := context.Background()
	c, err := anthropic.New(ctx, &genai.ProviderOptions{}, nil)
	msgs := genai.Messages{
		genai.NewTextMessage("Give me a life advice that sounds good but is a bad idea in practice. Answer succinctly."),
	}
	result, err := c.GenSync(ctx, msgs)
	fmt.Println(result.String())
}
```

Try it live:

```bash
go run github.com/maruel/genai/examples/txt_to_txt_sync@latest
```

This may print:

> "Follow your passion and the money will follow."
>
> This ignores market realities, financial responsibilities, and the fact that passion alone doesn't guarantee
> income or career viability.


### Text Streaming 🏎

[examples/txt\_to\_txt\_stream/main.go](examples/txt_to_txt_stream/main.go): This is the same example as
above, with the output streamed as it replies. This leverages [go 1.23
iterators](https://go.dev/blog/range-functions). Notice how little difference there is between both.

```go
func main() {
	ctx := context.Background()
	c, err := anthropic.New(ctx, &genai.ProviderOptions{}, nil)
	msgs := genai.Messages{
		genai.NewTextMessage("Give me a life advice that sounds good but is a bad idea in practice."),
	}
	fragments, finish := c.GenStream(ctx, msgs)
	for f := range fragments {
		os.Stdout.WriteString(f.TextFragment)
	}
	_, err = finish()
}
```

Try it live:

```bash
go run github.com/maruel/genai/examples/txt_to_txt_stream@latest
```


### Text Thinking 🧠

[examples/txt\_to\_txt\_thinking/main.go](examples/txt_to_txt_thinking/main.go): genai supports for implicit
thinking (e.g. Anthropic) and explicit thinking (e.g. Deepseek). The package adapters provide logic to
automatically handle explicit Chain-of-Thoughts models, generally using `<think>` and `</think>` tokens.

This requires [`DEEPSEEK_API_KEY`](https://platform.deepseek.com/api_keys) environment variable to
authenticate.

Snippet:

```go
	c, _ := deepseek.New(ctx, &genai.ProviderOptions{Model: "deepseek-reasoner"}, nil)
	msgs := genai.Messages{
		genai.NewTextMessage("Give me a life advice that sounds good but is a bad idea in practice."),
	}
	fragments, finish := c.GenStream(ctx, msgs)
	for f := range fragments {
		if f.ThinkingFragment != "" {
			// ...
		} else if f.TextFragment != "" {
			// ...
		}
	}
```

Try it live:

```bash
go run github.com/maruel/genai/examples/txt_to_txt_thinking@latest
```


### Text Citations ✍

[examples/txt\_to\_txt\_citations/main.go](examples/txt_to_txt_citations/main.go): Send entire documents and
leverage providers which support automatic citations (Cohere, Anthropic) to leverage their functionality for a
supercharged RAG.

This requires [`COHERE_API_KEY`](https://dashboard.cohere.com/api-keys) environment variable to authenticate.

```bash
go run github.com/maruel/genai/examples/txt_to_txt_citations@latest
```

When asked _When did Darwin arrive home?_ with the introduction of _On the Origin of Species by Charles
Darwin_ passed in as a document, this may print:

> Citation: 1837
>
> Citation: returned home
>
> Citation: began to reflect on the facts he had gathered
>
> Citation: H.M.S. Beagle.
>
> Answer: 1837 was when Darwin returned home and began to reflect on the facts he had gathered during his time on H.M.S. Beagle.


### Text Websearch 🕸️

[examples/txt\_to\_txt\_websearch-sync/main.go](examples/txt_to_txt_websearch-sync/main.go): Searches the web
to answer your question.

This requires [`PERPLEXITY_API_KEY`](https://www.perplexity.ai/settings/api) environment variable to
authenticate.

Snippet:

```go
	c, _ := perplexity.New(ctx, &genai.ProviderOptions{Model: genai.ModelCheap}, nil)
	msgs := genai.Messages{{
		Requests: []genai.Request{
			{Text: "Who holds ultimate power of Canada? Answer succinctly."},
		},
	}}
	res, _ := c.GenSync(ctx, msgs)
	for _, r := range res.Replies {
		for _, ci := range r.Citations {
			fmt.Printf("Sources:\n")
			for _, src := range ci.Sources {
				if src.Type == "web" {
					fmt.Printf("- %s / %s\n", src.Title, src.URL)
				} else {
					fmt.Printf("Image: %s\n", src.URL)
				}
			}
		}
	}
	fmt.Printf("\nAnswer: %s\n", res.String())
```

Try it live:

```bash
go run github.com/maruel/genai/examples/txt_to_txt_websearch-sync@latest
```

When asked _Who holds ultimate power of Canada?_, this may print:

> Sources:
>
> - Prime Minister of Canada / https://en.wikipedia.org/wiki/Prime_Minister_of_Canada
>
> - Canadian Parliamentary System - Our Procedure /
> https://www.ourcommons.ca/procedure/our-procedure/parliamentaryFramework/c_g_parliamentaryframework-e.html
>
>
> (...)
>
> Image: https://learn.parl.ca/understanding-comprendre/images/articles/monarch-and-governor-general/house-of-commons.jpg
>
> (...)
>
> Answer: The **ultimate power in Canada** constitutionally resides with the **monarch (King Charles III)** as
> the head of state, with executive authority formally vested in him. However, (...)


### Text Websearch (streaming) 🔍️

[examples/txt\_to\_txt\_websearch-stream/main.go](examples/txt_to_txt_websearch-stream/main.go): Searches the web
to answer your question and streams the output to the console.

This requires [`PERPLEXITY_API_KEY`](https://www.perplexity.ai/settings/api) environment variable to
authenticate.

```bash
go run github.com/maruel/genai/examples/txt_to_txt_websearch-stream@latest
```

Same as above, but streaming.


### Text Tools 🧰

[examples/txt\_to\_txt\_tool-sync/main.go](examples/txt_to_txt_tool-sync/main.go): A LLM can both retrieve
information and act on its environment through tool calling. This unblocks a whole realm of possibilities. Our
design enables dense strongly typed code that favorably compares to python.

This requires [`CEREBRAS_API_KEY`](https://cloud.cerebras.ai/platform/) environment variable to authenticate.

Snippet:

```go
	type numbers struct {
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
				Callback: func(ctx context.Context, input *numbers) (string, error) {
					return fmt.Sprintf("%d", input.A+input.B), nil
				},
			},
		},
		// Force the LLM to do a tool call.
		ToolCallRequest: genai.ToolCallRequired,
	}

	// Run the loop.
	res, _, _ := adapters.GenSyncWithToolCallLoop(ctx, c, msgs, &opts)
	// Print the answer which is the last message generated.
	fmt.Println(res[len(res)-1].String())
```

Try it live:

```bash
go run github.com/maruel/genai/examples/txt_to_txt_tool-sync@latest
```

When asked _What is 3214 + 5632?_, this may print:

> 8846


### Text Tools (streaming) 🐝

[examples/txt\_to\_txt\_tool-stream/main.go](examples/txt_to_txt_tool-stream/main.go): Leverage a thinking
model to see the thinking process while trying to use tool calls to answer the user's question. This enables
keeping the user updated to see the progress.

This requires [`GROQ_API_KEY`](https://console.groq.com/keys) environment variable to authenticate.

Snippet:

```go
	fragments, finish := adapters.GenStreamWithToolCallLoop(ctx, p, msgs, &opts)
	for f := range fragments {
		if f.ThinkingFragment != "" {
			// ...
		} else if f.TextFragment != "" {
			// ...
		} else if !f.ToolCall.IsZero() {
			// ...
		}
	}
```

Try it live:

```bash
go run github.com/maruel/genai/examples/txt_to_txt_tool-stream@latest
```

When asked _What is 3214 + 5632?_, this may print:

> \# Thinking
>
> User wants result of 3214+5632 using tool "add". Must be terse, only answer, no explanation. Need to call add function with a=3214, b=5632.
>
> \# Tool call
>
> {fc\_e9b9677b-898c-46df-9deb-39122bd6c69a add {"a":3214,"b":5632} map[] {}}
>
> \# Answer
>
> 8846
>
> Tokens usage: in: 349 (cached 0), reasoning: 0, out: 58, total: 0, requests/2025-08-26 19:17:13:
> 499999/500000, tokens/2025-08-26 19:17:13: 249809/250000

In addition to the token usage, remaining quota is printed.


### Text Tools (manual)

[examples/txt\_to\_txt\_tool-manual/main.go](examples/txt_to_txt_tool-manual/main.go): Runs a manual loop and
runs tool calls directly.

This requires [`CEREBRAS_API_KEY`](https://cloud.cerebras.ai/platform/) environment variable to authenticate.

```bash
go run github.com/maruel/genai/examples/txt_to_txt_tool-manual@latest
```


### Text Decode reply as a struct ⚙

[examples/txt\_to\_txt\_decode-json/main.go](examples/txt_to_txt_decode-json/main.go): Tell the LLM to use
a specific Go struct to determine the JSON schema to generate the response. This is much more lightweight than
tool calling!

It is very useful when we want the LLM to make a choice between values, to return a number or a boolean
(true/false). Enums are supported.

This requires [`OPENAI_API_KEY`](https://platform.openai.com/settings/organization/api-keys) environment
variable to authenticate.

Snippet:

```go
	msgs := genai.Messages{
		genai.NewTextMessage("Is a circle round? Reply as JSON."),
	}
	var circle struct {
		Round bool `json:"round"`
	}
	opts := genai.OptionsText{DecodeAs: &circle}
	res, _ := c.GenSync(ctx, msgs, &opts)
	res.Decode(&circle)
	fmt.Printf("Round: %v\n", circle.Round)
```

Try it live:

```bash
go run github.com/maruel/genai/examples/txt_to_txt_decode-json@latest
```

This will print:

> Round: true
>
> Tokens usage: in: 40 (cached 0), reasoning: 64, out: 81, total: 121, requests/2025-08-26 19:13:29: 499/500,
> tokens/2025-08-26 19:13:29: 199989/200000

In addition to the token usage, remaining quota is printed.


### Text to Image 📸

[examples/txt\_to\_img/main.go](examples/txt_to_img/main.go): Use Together.AI's free (!) image generation
albeit with
low rate limit.

Some providers return an URL that must be fetched manually within a few minutes or hours, some return the data
inline. This example handles both cases.

This requires [`TOGETHER_API_KEY`](https://api.together.ai/settings/api-keys) environment variable to
authenticate.

Snippet:

```go
	msgs := genai.Messages{
		genai.NewTextMessage("Carton drawing of a husky playing on the beach."),
	}
	result, _ := c.GenSync(ctx, msgs)
	for _, r := range result.Replies {
		if r.Doc.IsZero() {
			continue
		}
		// The image can be returned as an URL or inline, depending on the provider.
		var src io.Reader
		if r.Doc.URL != "" {
			req, _ := c.HTTPClient().Get(r.Doc.URL)
			src = req.Body
			defer req.Body.Close()
		} else {
			src = r.Doc.Src
		}
		b, _ := io.ReadAll(src)
		os.WriteFile(r.Doc.GetFilename(), b, 0o644)
	}
```

Try it live:

```bash
go run github.com/maruel/genai/examples/txt_to_img@latest
```

This may generate:

> ![content.jpg](https://raw.githubusercontent.com/wiki/maruel/genai/content.jpg)

This generated picture shows a fake signature. I decided to keep this example as a reminder that the result
comes from the data harvested that was created by real humans.


### Image-Text to Video 🎥

[examples/img-txt\_to\_vid/main.go](examples/img-txt_to_vid/main.go): Leverage the content.jpg file generated in
txt\_to\_img example to ask Veo 3 from Google to generate a video based on the image.

This requires [`GEMINI_API_KEY`](https://aistudio.google.com/apikey) environment variable to authenticate.

Snippet:

```go
	// Warning: this is expensive.
	c, _ := gemini.New(ctx, &genai.ProviderOptions{Model: "veo-3.0-fast-generate-preview"}, nil)
	f, _ := os.Open("content.jpg")
	defer f.Close()
	msgs := genai.Messages{
		genai.Message{Requests: []genai.Request{
			{Text: "Carton drawing of a husky playing on the beach."},
			{Doc: genai.Doc{Src: f}},
		}},
	}
	res, _ := c.GenSync(ctx, msgs)
	// Save the file in Replies like in the previous example ...
```

Try it live:

```bash
go run github.com/maruel/genai/examples/img-txt_to_vid@latest
```

This may generate:

> ![content.avif](https://raw.githubusercontent.com/wiki/maruel/genai/content.avif)

⚠ The MP4 has been recompressed to AVIF via
[compress.sh](https://raw.githubusercontent.com/wiki/maruel/genai/compress.sh) so GitHub can render it. The
drawback is that audio is lost. View the original MP4 with audio (!) at
[content.mp4](https://raw.githubusercontent.com/wiki/maruel/genai/content.mp4).

This is very impressive, but also very expensive.


### Image-Text to Image 🖌

[examples/img-txt\_to\_img/main.go](examples/img-txt_to_img/main.go): Edit an image with a prompt. Leverage
the content.jpg file generated in txt\_to\_img example.

This requires [`BFL_API_KEY`](https://dashboard.bfl.ai/keys) environment variable to authenticate.

```bash
go run github.com/maruel/genai/examples/img-txt_to_img@latest
```

This may generate:

> ![content2.jpg](https://raw.githubusercontent.com/wiki/maruel/genai/content2.jpg)


### Image-Text to Image-Text 🍌

[examples/img-txt\_to\_img-txt/main.go](examples/img-txt_to_img-txt/main.go): Leverage the
content.jpg file generated in txt\_to\_img example to ask gemini-2.5-flash-image-preview to change the image
with a prompt and ask the model to explain what it did.

This requires [`GEMINI_API_KEY`](https://aistudio.google.com/apikey) environment variable to authenticate.

Snippet:

```go
	// Warning: This is a bit expensive.
	opts := genai.ProviderOptions{
		Model:            "gemini-2.5-flash-image-preview",
		OutputModalities: genai.Modalities{genai.ModalityImage, genai.ModalityText},
	}
	c, _ := gemini.New(ctx, &opts, nil)
	// ...
	res, _ := c.GenSync(ctx, msgs, &gemini.Options{ThinkingBudget: 0})
```

Try it live:

```bash
go run github.com/maruel/genai/examples/img-txt_to_img-txt@latest
```

This may generate:

> Of course! Here's an updated image with more animals. I added a playful dolphin jumping out of the water and
> a flock of seagulls flying overhead. I chose these animals to enhance the beach scene and create a more
> dynamic and lively atmosphere. 
>
> Wrote: content.png
>
> ![content.png](https://raw.githubusercontent.com/wiki/maruel/genai/content.png)

This is quite impressive, but also quite expensive.


### Image-Text to Text 👁

[examples/img-txt\_to\_txt/main.go](examples/img-txt_to_txt/main.go): Run vision to analyze a picture provided
as an URL (source: [wikipedia](https://en.m.wikipedia.org/wiki/File:Banana-Single.jpg)). The response is
streamed out the console as the reply is generated.

This requires [`MISTRAL_API_KEY`](https://console.mistral.ai/api-keys) environment variable to authenticate.

```bash
go run github.com/maruel/genai/examples/img-txt_to_txt@latest
```

This may generate:

> The image depicts a single ripe banana. It has a bright yellow peel with a few small brown spots, indicating
> ripeness. The banana is curved, which is typical of its natural shape, and it has a stem at the top. The
> overall appearance suggests that it is ready to be eaten.


### Image-Text to Text (local) 🏠

[examples/img-txt_to_txt_local/main.go](examples/img-txt_to_txt_local/main.go): is very similar to the
previous example!

Use [cmd/llama-serve](cmd/llama-serve) to run a LLM locally, including tool calling and vision!

Start llama-server locally either by yourself or with this utility:

```bash
go run github.com/maruel/genai/cmd/llama-serve@latest \
  -model ggml-org/gemma-3-4b-it-GGUF/gemma-3-4b-it-Q8_0.gguf#mmproj-model-f16.gguf -- \
  --temp 1.0 --top-p 0.95 --top-k 64 \
  --jinja -fa -c 0 --no-warmup
```

Run vision 100% locally on CPU with only 8GB of RAM. No GPU required!

```bash
go run github.com/maruel/genai/examples/img-txt_to_txt_local@latest
```


### Video-Text to Text 🎞️

[examples/vid-txt\_to\_txt/main.go](examples/vid-txt_to_txt/main.go): Run vision to analyze a video.

This requires [`GEMINI_API_KEY`](https://aistudio.google.com/apikey) environment variable to authenticate.

Using this video:

![video.avif](https://raw.githubusercontent.com/wiki/maruel/genai/video.avif)

Try it live:

```bash
go run github.com/maruel/genai/examples/vid-txt_to_txt@latest
```

When asked _What is the word_, this generates:

> Banana


### Audio-Text to Text 🎤

[examples/aud-txt\_to\_txt/main.go](examples/aud-txt_to_txt/main.go):
Analyze an audio [file](https://github.com/maruel/genai/raw/refs/heads/main/scoreboard/testdata/audio.mp3).

This requires [`OPENAI_API_KEY`](https://platform.openai.com/settings/organization/api-keys) environment
variable to authenticate.


Try it live:

```bash
go run github.com/maruel/genai/examples/vid-txt_to_txt@latest
```

When asked _What was the word?_, this generates:

> The word was "orange."


### Text with any provider ⁉

[examples/txt\_to\_txt\_any/main.go](examples/txt_to_txt_any/main.go): Let the user chose the provider by name.

The relevant environment variable (e.g. `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, etc) is used automatically for
authentication.

Automatically selects a models on behalf of the user. Wraps the explicit thinking tokens if needed.

Supports [ollama](https://ollama.com/) and [llama-server](https://github.com/ggml-org/llama.cpp) even if they
run on a remote host or non-default port.

Snippet:

```go
	names := strings.Join(slices.Sorted(maps.Keys(providers.Available(ctx))), ", ")
	provider := flag.String("provider", "", "provider to use, "+names)
	flag.Parse()

	f := providers.All[*provider]
	c, _ := f(ctx, &genai.ProviderOptions{}, nil)
	p := adapters.WrapThinking(c)
	res, _ := p.GenSync(...)
```


Try it live:

```bash
go run github.com/maruel/genai/examples/txt_to_txt_any@latest \
    -provider cerebras \
    "Tell a good sounding advice that is a bad idea in practice."
```


## Models 🗒

Snapshot of all the supported models at [docs/MODELS.md](docs/MODELS.md) is updated weekly.

Try it:

```bash
go install github.com/maruel/genai/cmd/...@latest

list-models -provider huggingface
```


## I'm poor 💸

As of August 2025, the following services offer a free tier (other limits
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
- Image to 3D, e.g. [github.com/Tencent-Hunyuan/Hunyuan3D-2](https://github.com/Tencent-Hunyuan/Hunyuan3D-2)

### Providers

I'd be delighted if you want to contribute any missing provider being added, I'm particularly looking forward to these:

- [Alibaba Cloud](https://www.alibabacloud.com/): Maker of Qwen models.
- [AWS Bedrock](https://docs.aws.amazon.com/bedrock/latest/APIReference/API_Operations_Amazon_Bedrock.html)
- [Azure AI](https://learn.microsoft.com/en-us/rest/api/aifoundry/model-inference/get-chat-completions/get-chat-completions)
- [Fireworks responses](https://fireworks.ai/docs/guides/response-api)
- [GitHub](https://docs.github.com/en/rest/models/inference) inference API, which works on GitHub Actions (!)
- [Google's Vertex AI](https://cloud.google.com/vertex-ai/docs/reference/rest): It supports much more
  features than Gemini API.
- [LM Studio](https://lmstudio.ai/): Easier way to run local models.
- [Open Router](https://openrouter.ai/)
- [Runway](https://docs.dev.runwayml.com/api-details/sdks/): Specialized in images and videos.
- [Synexai](https://synexa.ai): It's very cheap.
- [vLLM](https://docs.vllm.ai/): The fastest way to run local models.

Thanks in advance! 🙏

Made with ❤️ by [Marc-Antoine Ruel](https://maruel.ca)
