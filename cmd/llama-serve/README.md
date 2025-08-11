# llama-server

A simple wrapper over [llama.cpp's llama-server](https://github.com/ggml-org/llama.cpp).

## Examples

Gemma 3 4B with vision:

```bash
llama-serve -model ggml-org/gemma-3-4b-it-GGUF/gemma-3-4b-it-Q8_0.gguf#mmproj-model-f16.gguf -- \
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

- `-build 6123` to use a specific build.
- `-http 0.0.0.0:8080` to be accessible from other machines.
- `--cache-type-k q8_0 --cache-type-v q8_0` to reduce KV cache memory usage.
