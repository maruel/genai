# Build, test, lint, and format the repository.
.PHONY: build test lint lint-fix format format-check verify git-hooks

build:
	@go build ./...

test:
	@go test ./...

lint:
	@golangci-lint run --show-stats=false ./...
	@ruff check --quiet .
	@python3 scripts/lint_binaries.py
	@python3 scripts/update_agents_file_index.py --check

lint-fix:
	@golangci-lint run --show-stats=false ./... --fix
	@ruff check --quiet --fix .
	@ruff format --quiet .
	@python3 scripts/update_agents_file_index.py

# Apply and verify the shared formatters: gofmt and goimports through
# golangci-lint for Go, ruff format for the Python scripts, and shfmt for the
# shell scripts.
format:
	@golangci-lint fmt
	@ruff format --quiet .
	@files=$$(git ls-files '*.sh' 'scripts/hooks/*'); [ -z "$$files" ] || shfmt -w $$files

format-check:
	@out=$$(golangci-lint fmt --diff); [ -z "$$out" ] || { echo 'Go files need formatting (gofmt, goimports):' >&2; echo "$$out" >&2; exit 1; }
	@ruff format --check --quiet .
	@files=$$(git ls-files '*.sh' 'scripts/hooks/*'); [ -z "$$files" ] || { out=$$(shfmt -l $$files); [ -z "$$out" ] || { echo 'Shell files need shfmt:' >&2; echo "$$out" >&2; exit 1; }; }

verify: format-check lint

git-hooks:
	@./scripts/install-git-hooks.sh
