# llama-serve

A simple wrapper over [llama.cpp's llama-server](https://github.com/ggml-org/llama.cpp).

## Examples

Kev 4B answers typed questions through `/v1/systemone`. Use
the client's `SystemOne()` method. This endpoint requires
nightly build b11361 or newer; the default stable v0.5.0 predates it.

```bash
go run github.com/maruel/genai/cmd/llama-serve@latest -version b11361 -model ggml-org/Kev-4B-GGUF
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

- `-version v0.5.0` to select a stable release, or `-version b11146` for a nightly build.
  Defaults to [Version](https://pkg.go.dev/github.com/maruel/genai/providers/llamacpp/llamacppsrv#Version).
  The existing `-build 1234` flag still selects a nightly build.
- `-http 0.0.0.0:8080` to be accessible from other machines. By default, only localhost is accessible.
- `--cache-type-k q8_0 --cache-type-v q8_0` to reduce KV cache memory usage. May negatively affect both
  performance and accuracy.
