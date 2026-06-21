#!/usr/bin/env bash
# Copyright 2025 Marc-Antoine Ruel. All rights reserved.
# Use of this source code is governed under the Apache License, Version 2.0
# that can be found in the LICENSE file.

# Script to regenerate weekly model scores for all providers.

set -euo pipefail

cd "$(dirname "$0")"
cd ..

go install ./cmd/list-models ./cmd/scoreboard
# Providers that can't run in CI: local servers or CLI wrappers.
EXCLUDE="bfl claudecode codex llamacpp ollama openaicompatible opencode openaibase openaichat openairesponses perplexity pi"
PROVIDERS=()
for d in providers/*/; do
	name=$(basename "$d")
	if [[ " $EXCLUDE " != *" $name "* ]]; then
		PROVIDERS+=("$name")
	fi
done
# "openai" is an alias for openairesponses but list-models treats it as a distinct provider.
PROVIDERS+=("openai")
mapfile -t PROVIDERS < <(printf '%s\n' "${PROVIDERS[@]}" | sort)

MISSING=()
require_all() {
	local provider=$1
	shift
	local key
	for key in "$@"; do
		if [[ -z "${!key:-}" ]]; then
			MISSING+=("$provider: $key")
		fi
	done
}

# Fail before touching generated files or test recordings.
require_all alibaba DASHSCOPE_API_KEY_INTL DASHSCOPE_API_KEY_US
require_all anthropic ANTHROPIC_API_KEY
require_all baseten BASETEN_API_KEY
require_all cerebras CEREBRAS_API_KEY
require_all cloudflare CLOUDFLARE_ACCOUNT_ID CLOUDFLARE_API_KEY
require_all cohere COHERE_API_KEY
require_all deepseek DEEPSEEK_API_KEY
require_all gemini GEMINI_API_KEY
require_all github GITHUB_TOKEN
require_all groq GROQ_API_KEY
require_all huggingface HUGGINGFACE_API_KEY
require_all mistral MISTRAL_API_KEY
require_all openai OPENAI_API_KEY
require_all openrouter OPENROUTER_API_KEY
require_all togetherai TOGETHER_API_KEY
require_all xiaomi MIMO_API_KEY
if [[ ${#MISSING[@]} -ne 0 ]]; then
	echo "missing required environment for weekly model regeneration:" >&2
	printf '  - %s\n' "${MISSING[@]}" >&2
	exit 1
fi

echo "# List of models available on each provider" >docs/MODELS.new.md
echo "" >>docs/MODELS.new.md
echo "Snapshot of the models available on each provider as of $(date +%Y-%m-%d)" >>docs/MODELS.new.md

for i in "${PROVIDERS[@]}"; do
	echo "- $i"
	{
		echo ""
		echo "## $i"
		echo ""
	} >>docs/MODELS.new.md
	if ! list-models -strict -provider "$i" | sed 's/^/- /' >>docs/MODELS.new.md; then
		find "./providers/$i" -name Warmup.yaml -delete
		go test "./providers/$i/..."
		exit 1
	fi
done
mv docs/MODELS.new.md docs/MODELS.md

go generate ./...
