# Build, test, lint, and format the repository.

.DEFAULT_GOAL := help
.PHONY: help build test fix verify git-hooks tools custom-gcl

# Tool versions. The tools target installs a tool that is missing or at another version, so
# these are the only places the versions are written down.
GOLANGCI_LINT_VERSION=v2.13.2
SHFMT_VERSION=v3.14.1
RUFF_VERSION=0.16.8

# The tools target installs into the Go and uv tool directories. Prepend them so a recipe
# that just installed a tool can run it, whatever the caller's PATH holds.
GO_BIN := $(if $(shell command -v go 2>/dev/null),$(shell go env GOPATH 2>/dev/null)/bin)
UV_BIN := $(if $(shell command -v uv 2>/dev/null),$(shell uv tool dir --bin 2>/dev/null))
export PATH := $(if $(GO_BIN),$(GO_BIN):)$(if $(UV_BIN),$(UV_BIN):)$(PATH)

tools:
	@command -v golangci-lint > /dev/null 2>&1 && golangci-lint --version 2>/dev/null | grep -Fqw "$(GOLANGCI_LINT_VERSION:v%=%)" || go install github.com/golangci/golangci-lint/v2/cmd/golangci-lint@$(GOLANGCI_LINT_VERSION)
	@command -v shfmt > /dev/null 2>&1 && shfmt --version 2>/dev/null | grep -Fqw "$(SHFMT_VERSION)" || go install mvdan.cc/sh/v3/cmd/shfmt@$(SHFMT_VERSION)
	@command -v uv > /dev/null 2>&1 || { echo 'uv is required to install the Python tools; see https://docs.astral.sh/uv/' >&2; exit 1; }
	@ruff --version 2>/dev/null | grep -Fqw "$(RUFF_VERSION)" || uv tool install --force --quiet ruff==$(RUFF_VERSION)

# The custom-gcl binary is not byte-reproducible (golangci-lint custom builds
# in a random temp directory and stamps VCS metadata), so staleness is tracked
# by hashing the build inputs instead of comparing mtimes: the plugin config
# plus the pinned golangci-lint version and the Go toolchain. A branch switch
# that recreates .custom-gcl.yml with a fresh timestamp must not trigger a
# rebuild; a version or config change must.
.PHONY: custom-gcl
custom-gcl:
	@want=$$({ sha256sum .custom-gcl.yml | cut -d" " -f1; echo "$(GOLANGCI_LINT_VERSION)"; go env GOVERSION; } | sha256sum | cut -d" " -f1); \
	if [ -x custom-gcl ] && [ "$$want" = "$$(cat .custom-gcl.sha 2>/dev/null)" ]; then exit 0; fi; \
	echo 'Building custom-gcl with the methodfilecheck plugin (one-off; runs when the config, golangci-lint version, or Go toolchain changes)...'; \
	golangci-lint custom --version $(GOLANGCI_LINT_VERSION) && echo "$$want" > .custom-gcl.sha

# The one static gate. The gofmt and goimports formatters are checked by
# custom-gcl run itself (formatters section of .golangci.yml) with its warm
# analysis cache; a separate `golangci-lint fmt --diff` pass would re-typecheck
# the whole tree without that cache. Caveat: when another linter fails on the
# same file, run reports the lint error only, so a formatting problem there
# surfaces on the next verify after the lint fix. fix applies the formatters
# through `golangci-lint fmt`.
verify: tools custom-gcl
	@./custom-gcl run --show-stats=false ./...
	@ruff format --check --quiet .
	@ruff check --quiet .
	@files=$$(git ls-files '*.sh' 'scripts/hooks/*'); [ -z "$$files" ] || { out=$$(shfmt -l $$files); [ -z "$$out" ] || { echo 'Shell files need shfmt:' >&2; echo "$$out" >&2; exit 1; }; }
	@python3 scripts/lint_binaries.py
	@python3 scripts/update_agents_file_index.py --check

# Apply every autofix, then refresh the generated file index. Does not
# re-check; run verify for that.
fix: tools custom-gcl
	@./custom-gcl run --show-stats=false ./... --fix
	@golangci-lint fmt
	@ruff check --quiet --fix .
	@ruff format --quiet .
	@files=$$(git ls-files '*.sh' 'scripts/hooks/*'); [ -z "$$files" ] || shfmt -w $$files
	@python3 scripts/update_agents_file_index.py

build:
	@go build ./...

test:
	@go test ./...

git-hooks:
	@./scripts/install-git-hooks.sh

help:
	@echo 'genai - Go library for talking to LLM providers'
	@echo ''
	@echo 'Available targets:'
	@printf '  %-14s - %s\n' 'make fix' 'Apply every autofix, then refresh the file index'
	@printf '  %-14s - %s\n' 'make verify' 'Fast static gate: gofmt, ruff, shfmt, binaries, docs (pre-push gate)'
	@printf '  %-14s - %s\n' 'make test' 'Run Go tests'
	@printf '  %-14s - %s\n' 'make build' 'Build all Go packages'
	@printf '  %-14s - %s\n' 'make git-hooks' 'Install git hooks'
