#!/usr/bin/env bash
# Copyright 2025 Marc-Antoine Ruel. All rights reserved.
# Use of this source code is governed under the Apache License, Version 2.0
# that can be found in the LICENSE file.

set -euo pipefail

cd "$(dirname $0)"
cd ..

go install ./cmd/list-models ./cmd/scoreboard
# TODO: get the list automatically. Note that we want "openai", not "openaichat and "openairesponses".
PROVIDERS=(anthropic cerebras cloudflare cohere deepseek gemini groq huggingface mistral openai pollinations togetherai)

echo "# List of models available on each provider" > docs/MODELS.new.md
echo "" >> docs/MODELS.new.md
echo "Snapshot of the models available on each provider as of $(date +%Y-%m-%d)" >> docs/MODELS.new.md

for i in "${PROVIDERS[@]}"; do
	echo "- $i"
	echo "" >> docs/MODELS.new.md
    echo "## $i" >> docs/MODELS.new.md
	echo "" >> docs/MODELS.new.md
	if ! (list-models -strict -provider $i | sed 's/^/- /' >> docs/MODELS.new.md); then
		find ./providers/$i -name Warmup.yaml -delete
		go test ./providers/$i/...
		exit 1
	fi
done
mv docs/MODELS.new.md docs/MODELS.md

go generate ./...
