#!/usr/bin/env bash
# Copyright 2025 Marc-Antoine Ruel. All rights reserved.
# Use of this source code is governed under the Apache License, Version 2.0
# that can be found in the LICENSE file.

set -eu

go install ./cmd/list-models
PROVIDERS=(anthropic cerebras cloudflare cohere deepseek gemini groq huggingface mistral openai togetherai)

echo "# List of models available on each provider" > MODELS.new.md
echo "" >> MODELS.new.md
echo "Snapshot of the models available on each provider as of $(date +%Y-%m-%d)" >> MODELS.new.md

for i in "${PROVIDERS[@]}"; do
	echo "" >> MODELS.new.md
    echo "## $i" >> MODELS.new.md
	echo "" >> MODELS.new.md
    list-models -provider $i | sed 's/^/- /' >> MODELS.new.md
done

mv MODELS.new.md MODELS.md
