# Scoreboard

| Model                           | Mode    | ➛In   | Out➛   | Tool | JSON | Batch | File | Cite | Text | Probs | Limits | Usage | Finish |
| ------------------------------- | ------- | ----- | ------ | ---- | ---- | ----- | ---- | ---- | ---- | ----- | ------ | ----- | ------ |
| zai-glm-4.6🥇                    | Sync🧠   | 💬    | 💬     | ✅🪨 | ✅   | ❌    | ❌   | ❌   | 🌱📏🛑 | ✅    | ❌     | ✅    | ✅     |
| zai-glm-4.6🥇                    | Stream🧠 | 💬    | 💬     | ✅🪨 | ✅   | ❌    | ❌   | ❌   | 🌱📏🛑 | ✅    | ❌     | ✅    | ✅     |
| qwen-3-235b-a22b-instruct-2507🥈 | Sync    | 💬    | 💬     | ✅🪨 | ✅   | ❌    | ❌   | ❌   | 🌱📏🛑 | ✅    | ❌     | ✅    | ✅     |
| qwen-3-235b-a22b-instruct-2507🥈 | Stream  | 💬    | 💬     | ✅🪨 | ✅   | ❌    | ❌   | ❌   | 🌱📏🛑 | ✅    | ❌     | ✅    | ✅     |
| gpt-oss-120b🥉                   | Sync🧠   | 💬    | 💬     | ✅   | ❌   | ❌    | ❌   | ❌   | 🌱📏  | ✅    | ❌     | ✅    | ✅     |
| gpt-oss-120b🥉                   | Stream🧠 | 💬    | 💬     | ✅   | ❌   | ❌    | ❌   | ❌   | 🌱📏  | ✅    | ❌     | ✅    | ✅     |
| qwen-3-32b                      | Sync🧠   | 💬    | 💬     | ✅🪨 | ✅   | ❌    | ❌   | ❌   | 🌱📏🛑 | ✅    | ❌     | ✅    | ✅     |
| qwen-3-32b                      | Stream🧠 | 💬    | 💬     | ✅🪨 | ✅   | ❌    | ❌   | ❌   | 🌱📏🛑 | ✅    | ❌     | ✅    | ✅     |
| llama-3.3-70b                   | Sync    | 💬    | 💬     | 💨🪨 | ✅   | ❌    | ❌   | ❌   | 🌱📏🛑 | ✅    | ❌     | ✅    | ✅     |
| llama-3.3-70b                   | Stream  | 💬    | 💬     | ✅🪨 | ✅   | ❌    | ❌   | ❌   | 🌱📏🛑 | ✅    | ❌     | ✅    | ✅     |
| llama3.1-8b                     | Sync    | 💬    | 💬     | ✅🪨 | ✅   | ❌    | ❌   | ❌   | 🌱📏🛑 | ✅    | ❌     | ✅    | ✅     |
| llama3.1-8b                     | Stream  | 💬    | 💬     | ✅🪨 | ✅   | ❌    | ❌   | ❌   | 🌱📏🛑 | ✅    | ❌     | ✅    | ✅     |
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

- Cerebras doesn't support images yet even if models could. https://discord.com/channels/1085960591052644463/1376887536072527982
- Most models are quantized to unspecified level: https://discord.com/channels/1085960591052644463/1085960592050896937/1372105565655928864
- Free tier has limited context: https://inference-docs.cerebras.ai/support/pricing
