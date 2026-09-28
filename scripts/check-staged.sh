#!/bin/bash
# Copyright 2026 Marc-Antoine Ruel. All rights reserved.
# Use of this source code is governed under the Apache License, Version 2.0
# that can be found in the LICENSE file.

# Checks staged files and the AGENTS.md index before committing.

set -euo pipefail

script_dir="$(CDPATH='' cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly script_dir
repo_root="$(CDPATH='' cd -- "$script_dir/.." && pwd)"
readonly repo_root

cd -- "$repo_root"

if ! git diff --quiet; then
  printf '%s\n' 'Stage all tracked changes before committing; checks read the worktree.' >&2
  exit 1
fi

python3 scripts/lint_binaries.py

python3 scripts/update_agents_file_index.py --check

declare -a unformatted=()
while IFS= read -r -d '' file; do
  if ! formatted=$(git show ":$file" | gofmt -d); then
    printf 'Could not format staged Go file: %s\n' "$file" >&2
    exit 1
  fi
  if [ -n "$formatted" ]; then
    unformatted+=("$file")
  fi
done < <(git diff --cached --name-only --diff-filter=ACMR -z -- '*.go')

if ((${#unformatted[@]} > 0)); then
  echo '✗ Go files must be formatted before committing:'
  printf '  %s\n' "${unformatted[@]}"
  exit 1
fi
