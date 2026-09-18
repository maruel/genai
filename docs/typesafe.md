# Scoreboard

| Model       | Mode | ➛In   | Out➛   | Tool | JSON | Batch | File | Cite | Text | Probs | Limits | Usage | Finish |
| ----------- | ---- | ----- | ------ | ---- | ---- | ----- | ---- | ---- | ---- | ----- | ------ | ----- | ------ |
| jev-latest🥇 | Sync | 💬    | 💬     | ❌   | 📐    | ❌    | ❌   | ❌   | ❌   | ❌    | ❌     | ✅    | ❌     |
| jev-preview | ?    | ?     | ?      | ?    | ?    | ?     | ?    | ?    | ?    | ?     | ?      | ?     | ?      |
| jev-1.13.0  | ?    | ?     | ?      | ?    | ?    | ?     | ?    | ?    | ?    | ?     | ?      | ?     | ?      |
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
- 💨: Feature is flaky (Tool calling) or inconsistent (Usage or Finish reason is not always reported)
- 🌐: Country where the company is located
- Tool: Tool calling, using [genai.ToolDef](https://pkg.go.dev/github.com/maruel/genai#ToolDef); best is ✅🪨🕸️
		- 🪨: Tool calling can be forced; aka you can force the model to call a tool. This is great.
		- 🕸️: Web search
- JSON: ability to output JSON in free form, or with a forced schema specified as a Go struct
    - ✅: Supports both free form and with a schema
    - ☁️ :Supports only free form
		- 📐: Supports only a schema
		- []: Also supports an array at the root of the schema, most providers require an object
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

- TypeSafe does not generate free form text. Every request sends a state and typed questions, and returns one typed answer per question. The questions are declared with genai.GenOptionText.DecodeAs and the reply is always the JSON object of the answers keyed by question name, which the same struct decodes.
- There is no streaming API. GenStream is simulated from GenSync and returns the whole reply at once.
- The API does not report rate limits, but it enforces 1,200 requests per minute and 250,000 tokens per second for jev-1.13.0.
