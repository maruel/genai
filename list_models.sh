#!/usr/bin/env bash
# Copyright 2025 Marc-Antoine Ruel. All rights reserved.
# Use of this source code is governed under the Apache License, Version 2.0
# that can be found in the LICENSE file.

set -eu

go install ./cmd/list-models
PROVIDERS=(anthropic cerebras cloudflare cohere deepseek gemini groq huggingface mistral openai)

echo "Snapshot of the models available on each provider as of $(date +%Y-%m-%d)"

for i in "${PROVIDERS[@]}"; do
	echo ""
    echo "Provider $i:"
    list-models -provider $i | sed 's/^/- /'
done
