# Build, test, lint, and format the repository.

.DEFAULT_GOAL := help
.PHONY: help build test fix verify git-hooks custom-gcl

# The custom-gcl binary is not byte-reproducible (golangci-lint custom builds
# in a random temp directory and stamps VCS metadata), so staleness is tracked
# by hashing the build inputs instead of comparing mtimes: the plugin config
# plus the pinned golangci-lint version and the Go toolchain. A branch switch
# that recreates .custom-gcl.yml with a fresh timestamp must not trigger a
# rebuild; a version or config change must.
.PHONY: custom-gcl
custom-gcl:
	@version=$$(go list -m -f '{{.Version}}' github.com/golangci/golangci-lint/v2) || exit 1; \
	want=$$({ sha256sum .custom-gcl.yml | cut -d" " -f1; echo "$$version"; go env GOVERSION; } | sha256sum | cut -d" " -f1); \
	if [ -x custom-gcl ] && [ "$$want" = "$$(cat .custom-gcl.sha 2>/dev/null)" ]; then exit 0; fi; \
	echo 'Building custom-gcl with the methodfilecheck plugin (one-off; runs when the config, golangci-lint version, or Go toolchain changes)...'; \
	go tool golangci-lint custom --version "$$version" && echo "$$want" > .custom-gcl.sha

# The one static gate. The gofmt and goimports formatters are checked by
# custom-gcl run itself (formatters section of .golangci.yml) with its warm
# analysis cache; a separate `go tool golangci-lint fmt --diff` pass would re-typecheck
# the whole tree without that cache. Caveat: when another linter fails on the
# same file, run reports the lint error only, so a formatting problem there
# surfaces on the next verify after the lint fix. fix applies the formatters
# through `go tool golangci-lint fmt`.
verify: custom-gcl
	@go run github.com/rhysd/actionlint/cmd/actionlint@v1.7.12
	@./custom-gcl run --show-stats=false ./...
	@uv run --locked ruff format --check --quiet .
	@uv run --locked ruff check --quiet .
	@files=$$(git ls-files '*.sh' 'scripts/hooks/*'); [ -z "$$files" ] || go tool shfmt -d $$files
	@go tool addlicense -ignore '.venv/**' -ignore '.ruff_cache/**' -ignore 'examples/**' -ignore '**/testdata/**' -check .
	@uv run --locked python scripts/lint_binaries.py
	@uv run --locked python scripts/update_agents_file_index.py --check
	@uv run --locked python -m unittest discover -s providers/claudecode -p 'test_*.py'

# Apply every autofix, then refresh the generated file index. Does not
# re-check; run verify for that.
fix: custom-gcl
	@./custom-gcl run --show-stats=false ./... --fix
	@go tool golangci-lint fmt
	@uv run --locked ruff check --quiet --fix .
	@uv run --locked ruff format --quiet .
	@files=$$(git ls-files '*.sh' 'scripts/hooks/*'); [ -z "$$files" ] || go tool shfmt -w $$files
	@uv run --locked python scripts/update_agents_file_index.py

build:
	@go build ./...

test:
	@go test -timeout=600s -covermode=count -coverprofile coverage.txt -bench=. -benchtime=1x ./...

git-hooks:
	@./scripts/install-git-hooks.sh

help:
	@echo 'genai - Go library for talking to LLM providers'
	@echo ''
	@echo 'Available targets:'
	@printf '  %-14s - %s\n' 'make fix' 'Apply every autofix, then refresh the file index'
	@printf '  %-14s - %s\n' 'make verify' 'Static gate: gofmt, ruff, shfmt, licenses, binaries, docs'
	@printf '  %-14s - %s\n' 'make test' 'Run Go tests with coverage and benchmark smoke checks'
	@printf '  %-14s - %s\n' 'make build' 'Build all Go packages'
	@printf '  %-14s - %s\n' 'make git-hooks' 'Install git hooks'
