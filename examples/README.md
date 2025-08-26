# Ready to run examples

The naming convention is `<input modalities>_to_<output modalities>_<particularity>`.

- txt is for text
- img is for image
- vid is for video

Other modalities supported at documents (PDF) and audio.

While nearly all providers support text to text, and most support tools, only a few support the more complex
modalities.

All these examples can be run from a local checkout, e.g.:

```bash
go run ./examples/txt_to_txt_stream
```

or directly without a local checkout, e.g.:

```bash
go run github.com/maruel/genai/examples/txt_to_txt_stream@latest
```
