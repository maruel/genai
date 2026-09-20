# Build, test, lint, and format the repository.
.PHONY: build test lint lint-fix format format-check verify git-hooks tools

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

build:
	@go build ./...

test:
	@go test ./...

lint: tools
	@golangci-lint run --show-stats=false ./...
	@ruff check --quiet .
	@python3 scripts/lint_binaries.py
	@python3 scripts/update_agents_file_index.py --check

lint-fix: tools
	@golangci-lint run --show-stats=false ./... --fix
	@ruff check --quiet --fix .
	@ruff format --quiet .
	@python3 scripts/update_agents_file_index.py

# Apply and verify the shared formatters: gofmt and goimports through
# golangci-lint for Go, ruff format for the Python scripts, and shfmt for the
# shell scripts.
format: tools
	@golangci-lint fmt
	@ruff format --quiet .
	@files=$$(git ls-files '*.sh' 'scripts/hooks/*'); [ -z "$$files" ] || shfmt -w $$files

format-check: tools
	@out=$$(golangci-lint fmt --diff); [ -z "$$out" ] || { echo 'Go files need formatting (gofmt, goimports):' >&2; echo "$$out" >&2; exit 1; }
	@ruff format --check --quiet .
	@files=$$(git ls-files '*.sh' 'scripts/hooks/*'); [ -z "$$files" ] || { out=$$(shfmt -l $$files); [ -z "$$out" ] || { echo 'Shell files need shfmt:' >&2; echo "$$out" >&2; exit 1; }; }

verify: format-check lint

git-hooks:
	@./scripts/install-git-hooks.sh
