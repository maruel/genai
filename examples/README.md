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

## Typed decisions

- `txt_to_decisions` asks typed questions about a ticket using TypeSafe; set `TYPESAFE_API_KEY`.
- `txt_to_decisions_local` asks the same ticket questions using a local Kev-4B server.
  It downloads the model, caches the downloads, and stops the server on exit:

  ```bash
  go run github.com/maruel/genai/examples/txt_to_decisions_local@latest
  ```

- `img-txt_to_decisions_local` asks typed questions about a local image and text context using llama.cpp.
  It downloads and starts OpenJev with its multimodal projector, caches the downloads, and stops the server
  on exit:

  ```bash
  go run github.com/maruel/genai/examples/img-txt_to_decisions_local@latest \
    -image screenshot.png -text "Review this billing error."
  ```

The image example uses a bundled sample image when `-image` is omitted. All three examples print the answers,
probability distributions, token usage, and elapsed time.
