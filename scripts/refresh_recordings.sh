#!/usr/bin/env bash
# Copyright 2026 Marc-Antoine Ruel. All rights reserved.
# Use of this source code is governed under the Apache License, Version 2.0
# that can be found in the LICENSE file.

# Script to refresh provider model-list and subprocess recordings.

set -euo pipefail

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

start=0
if [[ $# -eq 2 && $1 == --from ]]; then
	for i in "${!PROVIDERS[@]}"; do
		if [[ ${PROVIDERS[i]} == "$2" ]]; then
			start=$i
			break
		fi
	done
	if [[ $start -eq 0 && $2 != "${PROVIDERS[0]}" ]]; then
		echo "unknown provider: $2" >&2
		exit 2
	fi
elif [[ $# -ne 0 ]]; then
	echo "usage: $0 [--from PROVIDER]" >&2
	exit 2
fi

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
preflight_provider() {
	case $1 in
	alibaba) require_all "$1" DASHSCOPE_API_KEY_INTL DASHSCOPE_API_KEY_US ;;
	anthropic) require_all "$1" ANTHROPIC_API_KEY ;;
	baseten) require_all "$1" BASETEN_API_KEY ;;
	cerebras) require_all "$1" CEREBRAS_API_KEY ;;
	cloudflare) require_all "$1" CLOUDFLARE_ACCOUNT_ID CLOUDFLARE_API_KEY ;;
	cohere) require_all "$1" COHERE_API_KEY ;;
	deepseek) require_all "$1" DEEPSEEK_API_KEY ;;
	gemini) require_all "$1" GEMINI_API_KEY ;;
	groq) require_all "$1" GROQ_API_KEY ;;
	huggingface) require_all "$1" HUGGINGFACE_API_KEY ;;
	mistral) require_all "$1" MISTRAL_API_KEY ;;
	openaichat | openairesponses) require_all "$1" OPENAI_API_KEY ;;
	openrouter) require_all "$1" OPENROUTER_API_KEY ;;
	pollinations) require_all "$1" POLLINATIONS_API_KEY ;;
	togetherai) require_all "$1" TOGETHER_API_KEY ;;
	xiaomi) require_all "$1" MIMO_API_KEY ;;
	claudecode | codex | opencode | pi)
		local binary=$1
		if [[ $binary == claudecode ]]; then
			binary=claude
		fi
		if ! command -v "$binary" >/dev/null; then
			MISSING+=("CLI provider: $binary executable")
		fi
		;;
	esac
}

for provider in "${PROVIDERS[@]:start}"; do
	preflight_provider "$provider"
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

for provider in "${PROVIDERS[@]:start}"; do
	refresh_provider "$provider"
done

go generate ./...
