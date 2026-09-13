#!/usr/bin/env bash
# Copyright 2026 Marc-Antoine Ruel. All rights reserved.
# Use of this source code is governed under the Apache License, Version 2.0
# that can be found in the LICENSE file.

# Script to refresh provider model-list and subprocess recordings.

set -euo pipefail

if [[ $# -ne 0 ]]; then
	echo "usage: $0" >&2
	exit 2
fi

script_dir="$(CDPATH='' cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly script_dir
repo_root="$(CDPATH='' cd -- "$script_dir/.." && pwd)"
readonly repo_root
cd -- "$repo_root"

if [[ -f .env ]]; then
	set -a
	# shellcheck disable=SC1091
	source .env
	set +a
fi

# Keep this list explicit: refreshing these fixtures requires live provider
# credentials (or authenticated CLIs), and silently skipping one would leave a
# stale recording behind.
readonly PROVIDERS=(
	alibaba
	anthropic
	baseten
	cerebras
	claudecode
	cloudflare
	codex
	cohere
	deepseek
	gemini
	groq
	huggingface
	mistral
	openaichat
	openairesponses
	opencode
	openrouter
	pi
	pollinations
	togetherai
	xiaomi
)

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

# Fail before deleting any recording. The CLI providers authenticate through
# their own configuration, so only require their executables here.
require_all alibaba DASHSCOPE_API_KEY_INTL DASHSCOPE_API_KEY_US
require_all anthropic ANTHROPIC_API_KEY
require_all baseten BASETEN_API_KEY
require_all cerebras CEREBRAS_API_KEY
require_all cloudflare CLOUDFLARE_ACCOUNT_ID CLOUDFLARE_API_KEY
require_all cohere COHERE_API_KEY
require_all deepseek DEEPSEEK_API_KEY
require_all gemini GEMINI_API_KEY
require_all groq GROQ_API_KEY
require_all huggingface HUGGINGFACE_API_KEY
require_all mistral MISTRAL_API_KEY
require_all openai OPENAI_API_KEY
require_all openrouter OPENROUTER_API_KEY
require_all pollinations POLLINATIONS_API_KEY
require_all togetherai TOGETHER_API_KEY
require_all xiaomi MIMO_API_KEY
for binary in claude codex opencode pi; do
	if ! command -v "$binary" >/dev/null; then
		MISSING+=("CLI provider: $binary executable")
	fi
done
if [[ ${#MISSING[@]} -ne 0 ]]; then
	echo "missing required environment for recording refresh:" >&2
	printf '  - %s\n' "${MISSING[@]}" >&2
	exit 1
fi

refresh_provider() {
	local provider=$1
	local -a test_args=(-update-scoreboard)
	if [[ $provider == anthropic ]]; then
		test_args+=(-update-models)
	fi

	echo "Refreshing recordings for $provider"
	find "providers/$provider" -type f \( -name Warmup.yaml -o -name '*.ndjson' \) -delete
	RECORD=failure_only go test "./providers/$provider/..." -timeout=60m "${test_args[@]}"
}

for provider in "${PROVIDERS[@]}"; do
	refresh_provider "$provider"
done

go generate ./...
