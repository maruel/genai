#!/usr/bin/env bash
# Copyright 2025 Marc-Antoine Ruel. All rights reserved.
# Use of this source code is governed under the Apache License, Version 2.0
# that can be found in the LICENSE file.

set -eu

cd "$(dirname $0)"
cd ..

go install ./cmd/list-models ./cmd/scoreboard
PROVIDERS=(anthropic cerebras cloudflare cohere deepseek gemini groq huggingface mistral openai pollinations togetherai)

echo "# List of models available on each provider" > docs/MODELS.new.md
echo "" >> docs/MODELS.new.md
echo "Snapshot of the models available on each provider as of $(date +%Y-%m-%d)" >> docs/MODELS.new.md

for i in "${PROVIDERS[@]}"; do
	echo "- $i"
	echo "" >> docs/MODELS.new.md
    echo "## $i" >> docs/MODELS.new.md
	echo "" >> docs/MODELS.new.md
    list-models -strict -provider $i | sed 's/^/- /' >> docs/MODELS.new.md
    echo "# Scoreboard" >> docs/$i.md
	echo "" >> docs/$i.md
	scoreboard -table -provider $i >> docs/$i.md
done

mv docs/MODELS.new.md docs/MODELS.md
