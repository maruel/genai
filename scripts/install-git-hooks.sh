#!/bin/bash
# Configures this checkout to run the repository's versioned Git hooks.

set -euo pipefail

script_dir="$(CDPATH='' cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly script_dir
repo_root="$(CDPATH='' cd -- "$script_dir/.." && pwd)"
readonly repo_root

if ! git -C "$repo_root" rev-parse --git-dir >/dev/null 2>&1; then
	exit 0
fi

git -C "$repo_root" config --local core.hooksPath scripts/hooks
