# Scoreboard

| Model               | Mode | ➛In   | Out➛   | JSON | Schema | Tool | Batch | File | Cite | Text | Probs | Limits | Usage | Finish |
| ------------------- | ---- | ----- | ------ | ---- | ------ | ---- | ----- | ---- | ---- | ---- | ----- | ------ | ----- | ------ |
| flux-dev            | 🕰️   | 💬    | 📸     | ❌   | ❌     | ❌   | ✅    | ❌   | ❌   | 🌱   | ❌    | ✅     | ❌    | ✅     |
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
<summary>‼️ Click here for legend of 🏠 ✅ ❌ 💬 📄 📸 🎤 🎥 🤪 💸 🚩 💨 🧐 💥 and columns</summary>

- 🏠: Runs locally.
- 🕰️: Runs synchronously, the reply is only returned once completely generated.
- 📡: Runs asynchronously, the reply is returned as soon as it is available.
- 🧠: Supports chain-of-thought thinking process.
    - Both redacted (Anthropic, Gemini) and explicit (Deepseek R1, Qwen3, etc).
    - Some models can be used in both mode. In this case they will have two rows, one with thinking and one
      without. It is frequent that certain functionalities are limited in thinking mode, like tool calling.
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
- 💨: Tool calling is flaky.
- 🧐: Tool calling is **not** biased towards the first value in an enum. If the provider doesn't have this, be
  mindful of the order of the values!
- 🌐: Country where the company is located.
- JSON and Schema: ability to output JSON in free form, or with a forced schema specified as a Go struct
- Chat: Buffered chat.
- Stream: Streaming output.
- Tool: Tool calling, using [genai.ToolDef](https://pkg.go.dev/github.com/maruel/genai#ToolDef)
- Batch: Process asynchronously batches during off peak hours at a discounts.
- Text: Text features:
    - '🌱': Seed option for deterministic output.
    - '📏': MaxTokens option to cap the amount of returned tokens.
    - '🛑': Stop sequence to stop generation when a token is generated.
- File: Upload and store large files.
- Cite: Citation generation. Especially useful for RAG.
- Probs: return logprobs. Many do not support this in streaming mode.
- Limits: returns the rate limits, including the remaining quota.
</details>
