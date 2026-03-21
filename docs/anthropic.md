# Scoreboard

| Model                      | Mode    | ➛In    | Out➛   | Tool   | JSON | Batch | File | Cite | Text | Probs | Limits | Usage | Finish |
| -------------------------- | ------- | ------ | ------ | ------ | ---- | ----- | ---- | ---- | ---- | ----- | ------ | ----- | ------ |
| claude-opus-4-6🥇           | Sync🧠   | 💬📄📸 | 💬     | 💨🕸️   | 📐    | ✅    | ❌   | ✅   | 🛑    | ❌    | ✅     | ✅    | ✅     |
| claude-opus-4-6🥇           | Stream🧠 | 💬📄📸 | 💬     | 💨🕸️   | 📐    | ✅    | ❌   | ✅   | 🛑    | ❌    | ✅     | ✅    | ✅     |
| claude-sonnet-4-6🥈         | Sync🧠   | 💬📄📸 | 💬     | ✅🕸️   | 📐    | ✅    | ❌   | ✅   | 🛑    | ❌    | ✅     | ✅    | ✅     |
| claude-sonnet-4-6🥈         | Stream🧠 | 💬📄📸 | 💬     | ✅🕸️   | 📐    | ✅    | ❌   | ✅   | 🛑    | ❌    | ✅     | ✅    | ✅     |
| claude-haiku-4-5-20251001🥉 | Sync    | 💬📄📸 | 💬     | ✅🪨🕸️ | 📐    | ✅    | ❌   | ✅   | 📏🛑   | ❌    | ✅     | ✅    | ✅     |
| claude-haiku-4-5-20251001🥉 | Stream  | 💬📄📸 | 💬     | ✅🪨🕸️ | 📐    | ✅    | ❌   | ✅   | 📏    | ❌    | ✅     | ✅    | ✅     |
| claude-opus-4-6            | Sync    | 💬📄📸 | 💬     | ✅🪨🕸️ | 📐    | ✅    | ❌   | ✅   | 📏🛑   | ❌    | ✅     | ✅    | ✅     |
| claude-opus-4-6            | Stream  | 💬📄📸 | 💬     | ✅🪨🕸️ | 📐    | ✅    | ❌   | ✅   | 📏🛑   | ❌    | ✅     | ✅    | ✅     |
| claude-sonnet-4-5-20250929 | Sync    | 💬📄📸 | 💬     | ✅🪨🕸️ | 📐    | ✅    | ❌   | ✅   | 📏🛑   | ❌    | ✅     | ✅    | ✅     |
| claude-sonnet-4-5-20250929 | Stream  | 💬📄📸 | 💬     | ✅🪨🕸️ | 📐    | ✅    | ❌   | ✅   | 📏    | ❌    | ✅     | ✅    | ✅     |
| claude-opus-4-1-20250805   | ?       | ?      | ?      | ?      | ?    | ?     | ?    | ?    | ?    | ?     | ?      | ?     | ?      |
| claude-opus-4-20250514     | ?       | ?      | ?      | ?      | ?    | ?     | ?    | ?    | ?    | ?     | ?      | ?     | ?      |
| claude-opus-4-5-20251101   | ?       | ?      | ?      | ?      | ?    | ?     | ?    | ?    | ?    | ?     | ?      | ?     | ?      |
| claude-sonnet-4-20250514   | ?       | ?      | ?      | ?      | ?    | ?     | ?    | ?    | ?    | ?     | ?      | ?     | ?      |
| claude-sonnet-4-5-20250929 | ?       | ?      | ?      | ?      | ?    | ?     | ?    | ?    | ?    | ?     | ?      | ?     | ?      |
| claude-3-5-haiku-20241022  | ?       | ?      | ?      | ?      | ?    | ?     | ?    | ?    | ?    | ?     | ?      | ?     | ?      |
| claude-3-7-sonnet-20250219 | ?       | ?      | ?      | ?      | ?    | ?     | ?    | ?    | ?    | ?     | ?      | ?     | ?      |
| claude-3-haiku-20240307    | ?       | ?      | ?      | ?      | ?    | ?     | ?    | ?    | ?    | ?     | ?      | ?     | ?      |
| claude-3-opus-20240229     | ?       | ?      | ?      | ?      | ?    | ?     | ?    | ?    | ?    | ?     | ?      | ?     | ?      |
| claude-opus-4-1-20250805   | ?       | ?      | ?      | ?      | ?    | ?     | ?    | ?    | ?    | ?     | ?      | ?     | ?      |
| claude-opus-4-20250514     | ?       | ?      | ?      | ?      | ?    | ?     | ?    | ?    | ?    | ?     | ?      | ?     | ?      |
| claude-opus-4-5-20251101   | ?       | ?      | ?      | ?      | ?    | ?     | ?    | ?    | ?    | ?     | ?      | ?     | ?      |
| claude-sonnet-4-20250514   | ?       | ?      | ?      | ?      | ?    | ?     | ?    | ?    | ?    | ?     | ?      | ?     | ?      |
| claude-sonnet-4-6          | ?       | ?      | ?      | ?      | ?    | ?     | ?    | ?    | ?    | ?     | ?      | ?     | ?      |
<details>
<summary>‼️ Click here for the legend of columns and symbols</summary>

- 🏠: Runs locally.
- Sync:   Runs synchronously, the reply is only returned once completely generated
- Stream: Streams the reply as it is generated. Occasionally less features are supported in this mode
- 🧠: Has chain-of-thought thinking process
    - Both redacted (Anthropic, Gemini, OpenAI) and explicit (Deepseek R1, Qwen3, etc)
    - Many models can be used in both mode. In this case they will have two rows, one with thinking and one
      without. It is frequent that certain functionalities are limited in thinking mode, like tool calling.
- ✅: Implemented and works great
- ❌: Not supported by genai. The provider may support it, but genai does not (yet). Please send a PR to add
  it!
- 💬: Text
- 📄: PDF: process a PDF as input, possibly with OCR
- 📸: Image: process an image as input; most providers support PNG, JPG, WEBP and non-animated GIF, or generate images
- 🎤: Audio: process an audio file (e.g. MP3, WAV, Flac, Opus) as input, or generate audio
- 🎥: Video: process a video (e.g. MP4) as input, or generate a video (e.g. Veo 3)
- 💨: Feature is flaky (Tool calling) or inconsistent (Usage is not always reported)
- 🌐: Country where the company is located
- Tool: Tool calling, using [genai.ToolDef](https://pkg.go.dev/github.com/maruel/genai#ToolDef); best is ✅🪨🕸️
		- 🪨: Tool calling can be forced; aka you can force the model to call a tool. This is great.
		- 🕸️: Web search
- JSON: ability to output JSON in free form, or with a forced schema specified as a Go struct
    - ✅: Supports both free form and with a schema
    - ☁️ :Supports only free form
		- 📐: Supports only a schema
- Batch: Process asynchronously batches during off peak hours at a discounts
- Text: Text features
    - '🌱': Seed option for deterministic output
    - '📏': MaxTokens option to cap the amount of returned tokens
    - '🛑': Stop sequence to stop generation when a token is generated
- File: Upload and store large files via a separate API
- Cite: Citation generation from a provided document, specially useful for RAG
- Probs: Return logprobs to analyse each token probabilities
- Limits: Returns the rate limits, including the remaining quota
</details>

## Warnings

- Tool calling can't be forced when reasoning and becomes much more flaky.
- Tool calling works very well but is biased; the model is lazy and when it's unsure, it will use the tool's first argument.
- Rate limit is based on how much you spend per month: https://docs.anthropic.com/en/api/rate-limits#requirements-to-advance-tier
