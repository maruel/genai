# llama-serve

A simple wrapper over [llama.cpp's llama-server](https://github.com/ggml-org/llama.cpp).

## Examples

EmbeddingGemma 2 generates text and multimodal embeddings. Released nightly
[b11476](https://github.com/ggml-org/llama.cpp/releases/tag/b11476) supports it;
the default stable release v0.6.0 cannot load it. Select the nightly explicitly:

```bash
llama-serve -build 11476 -model ggml-org/embeddinggemma-2-GGUF/embeddinggemma-2-Q8_0.gguf -- \
    --embeddings --pooling mean
```

Use the client's `Embed()` method for text or documents, or `EmbedRaw()` for token inputs,
multimodal content and normalization control. See
[the Go example](../../providers/llamacpp/example_test.go) for search and document prefixes.
The wrapper uses llama-server's `-hf` and `-hff` flags to download the model
and automatically fetch its multimodal projector when available.

Kev 4B answers typed questions through `/v1/systemone`. Use
the client's `SystemOne()` method. This endpoint requires
llama.cpp v0.6.0 or newer, which is the default version.

```bash
go run github.com/maruel/genai/cmd/llama-serve@latest -model ggml-org/Kev-4B-GGUF
```

Qwen3.5 2B is a tiny but capable model, great for testing:

```bash
llama-serve -model unsloth/Qwen3.5-2B-GGUF/Qwen3.5-2B-Q4_K_M.gguf -- \
    --jinja -fa -c 0 --no-warmup
```

Gemma 3 4B with vision. llama-server automatically downloads the mmproj file.

```bash
llama-serve -model ggml-org/gemma-3-4b-it-GGUF/gemma-3-4b-it-Q8_0.gguf -- \
    --temp 1.0 --top-p 0.95 --top-k 64 \
    --jinja -fa -c 0 --no-warmup
```

Jan nano 4B is a fine tuned Qwen3 4B optimized for tool calling:

```bash
llama-serve -model Menlo/Jan-nano-gguf/jan-nano-4b-Q8_0.gguf -- \
   --temp 0.7 --top-p 0.8 --top-k 20 --min-p 0 \
   --jinja -fa -c 0 --no-warmup
```

## Frequently used flags

- `-version v0.6.0` to select a stable release, or `-version b11146` for a nightly build.
  Defaults to [Version](https://pkg.go.dev/github.com/maruel/genai/providers/llamacpp/llamacppsrv#Version).
  The existing `-build 1234` flag still selects a nightly build.
- `-http 0.0.0.0:8080` to be accessible from other machines. By default, only localhost is accessible.
- `--cache-type-k q8_0 --cache-type-v q8_0` to reduce KV cache memory usage. May negatively affect both
  performance and accuracy.
