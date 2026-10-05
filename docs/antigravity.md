# Scoreboard

| Model                    | Mode    | ➛In   | Out➛   | Tool | JSON | Batch | File | Cite | Text | Probs | Limits | Usage | Finish |
| ------------------------ | ------- | ----- | ------ | ---- | ---- | ----- | ---- | ---- | ---- | ----- | ------ | ----- | ------ |
| gemini-3.1-pro-high🥇     | Sync🧠   | 💬    | 💬     | ❌   | 📐    | ❌    | ❌   | ❌   | ❌   | ❌    | ❌     | ✅    | ✅     |
| gemini-3.1-pro-high🥇     | Stream🧠 | 💬    | 💬     | ❌   | 📐    | ❌    | ❌   | ❌   | ❌   | ❌    | ❌     | ✅    | ✅     |
| gemini-3.8-flash-medium🥈 | Sync🧠   | 💬    | 💬     | ❌   | 📐    | ❌    | ❌   | ❌   | ❌   | ❌    | ❌     | ✅    | ✅     |
| gemini-3.8-flash-medium🥈 | Stream🧠 | 💬    | 💬     | ❌   | 📐    | ❌    | ❌   | ❌   | ❌   | ❌    | ❌     | ✅    | ✅     |
| gemini-3.8-flash-low🥉    | Sync🧠   | 💬    | 💬     | ❌   | 📐    | ❌    | ❌   | ❌   | ❌   | ❌    | ❌     | ✅    | ✅     |
| gemini-3.8-flash-low🥉    | Stream🧠 | 💬    | 💬     | ❌   | 📐    | ❌    | ❌   | ❌   | ❌   | ❌    | ❌     | ✅    | ✅     |
| claude-opus-4-6-thinking | ?       | ?     | ?      | ?    | ?    | ?     | ?    | ?    | ?    | ?     | ?      | ?     | ?      |
| claude-sonnet-4-6        | ?       | ?     | ?      | ?    | ?    | ?     | ?    | ?    | ?    | ?     | ?      | ?     | ?      |
| gemini-3.1-pro-low       | ?       | ?     | ?      | ?    | ?    | ?     | ?    | ?    | ?    | ?     | ?      | ?     | ?      |
| gemini-3.6-flash-high    | ?       | ?     | ?      | ?    | ?    | ?     | ?    | ?    | ?    | ?     | ?      | ?     | ?      |
| gemini-3.6-flash-low     | ?       | ?     | ?      | ?    | ?    | ?     | ?    | ?    | ?    | ?     | ?      | ?     | ?      |
| gemini-3.6-flash-medium  | ?       | ?     | ?      | ?    | ?    | ?     | ?    | ?    | ?    | ?     | ?      | ?     | ?      |
| gemini-3.7-flash-high    | ?       | ?     | ?      | ?    | ?    | ?     | ?    | ?    | ?    | ?     | ?      | ?     | ?      |
| gemini-3.7-flash-low     | ?       | ?     | ?      | ?    | ?    | ?     | ?    | ?    | ?    | ?     | ?      | ?     | ?      |
| gemini-3.7-flash-medium  | ?       | ?     | ?      | ?    | ?    | ?     | ?    | ?    | ?    | ?     | ?      | ?     | ?      |
| gemini-3.8-flash-high    | ?       | ?     | ?      | ?    | ?    | ?     | ?    | ?    | ?    | ?     | ?      | ?     | ?      |
| gpt-oss-120b-medium      | ?       | ?     | ?      | ?    | ?    | ?     | ?    | ?    | ?    | ?     | ?      | ?     | ?      |
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
- 🎯: Decision: typed answers returned by SystemOne
- 📄: PDF: process a PDF as input, possibly with OCR
- 📸: Image: process an image as input; most providers support PNG, JPG, WEBP and non-animated GIF, or generate images
- 🎤: Audio: process an audio file (e.g. MP3, WAV, Flac, Opus) as input, or generate audio
- 🎥: Video: process a video (e.g. MP4) as input, or generate a video (e.g. Veo 3)
- 💨: Feature is flaky (Tool calling) or inconsistent (Usage or Finish reason is not always reported)
- 🌐: Country where the company is located
- Tool: Tool calling, using [genai.ToolDef](https://pkg.go.dev/github.com/maruel/genai#ToolDef); best is ✅🪨🕸️🔀
		- 🪨: Tool calling can be forced; aka you can force the model to call a tool. This is great.
		- 🕸️: Web search
		- 🔀: Supports messages and tool calls out of order
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
