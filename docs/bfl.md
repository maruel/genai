# Scoreboard

| Model               | Mode | ➛In   | Out➛   | JSON | Schema | Tool | Batch | File | Cite | Text | Probs | Limits | Usage | Finish |
| ------------------- | ---- | ----- | ------ | ---- | ------ | ---- | ----- | ---- | ---- | ---- | ----- | ------ | ----- | ------ |
| flux-dev            | Sync | 💬    | 📸     | ❌   | ❌     | ❌   | ✅    | ❌   | ❌   | 🌱   | ❌    | ✅     | ❌    | ❌     |
| flux-tools          | ?    | ?     | ?      | ?    | ?      | ?    | ?     | ?    | ?    | ?    | ?     | ?      | ?     | ?      |
| flux-pro-1.0-depth  | ?    | ?     | ?      | ?    | ?      | ?    | ?     | ?    | ?    | ?    | ?     | ?      | ?     | ?      |
| flux-pro-1.0-canny  | ?    | ?     | ?      | ?    | ?      | ?    | ?     | ?    | ?    | ?    | ?     | ?      | ?     | ?      |
| flux-pro-1.0-expand | ?    | ?     | ?      | ?    | ?      | ?    | ?     | ?    | ?    | ?    | ?     | ?      | ?     | ?      |
| flux-pro-1.0-fill   | ?    | ?     | ?      | ?    | ?      | ?    | ?     | ?    | ?    | ?    | ?     | ?      | ?     | ?      |
| flux-kontext-pro    | ?    | ?     | ?      | ?    | ?      | ?    | ?     | ?    | ?    | ?    | ?     | ?      | ?     | ?      |
| flux-kontext-max    | ?    | ?     | ?      | ?    | ?      | ?    | ?     | ?    | ?    | ?    | ?     | ?      | ?     | ?      |
| flux-pro-1.1-ultra  | ?    | ?     | ?      | ?    | ?      | ?    | ?     | ?    | ?    | ?    | ?     | ?      | ?     | ?      |
| flux-pro-1.1        | ?    | ?     | ?      | ?    | ?      | ?    | ?     | ?    | ?    | ?    | ?     | ?      | ?     | ?      |
| flux-pro            | ?    | ?     | ?      | ?    | ?      | ?    | ?     | ?    | ?    | ?    | ?     | ?      | ?     | ?      |
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
- 🧐: Tool calling is **not** biased towards the first value in an enum. If the provider doesn't have this, be
  mindful of the order of the values!
- 🌐: Country where the company is located
- JSON and Schema: ability to output JSON in free form, or with a forced schema specified as a Go struct
- Tool: Tool calling, using [genai.ToolDef](https://pkg.go.dev/github.com/maruel/genai#ToolDef)
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
