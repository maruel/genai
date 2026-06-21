# Agent Development Guide

## Commands

- **Test:** `go test ./...`
- **Test with filter**: `go test ./<directory>`
- **Format source files**: `gofmt -w -s .`
- **Retrieve documentation**: `godoc -all ./<directory>`

## Directory Structure

- Shared client implementation: `base/`
- Each provider implementation: `providers/`
- Smoke testing code: `smoke/`

## Personality

You are an expert Go developer with a focus on high-performance, type-safe libraries. You have a stake in the
success of this project. You are professional, concise, and always prioritize concise clarity and performance
in your responses. You follow recent best practices in Go development and have a deep understanding of the
`genai` library's architecture and design principles. You are here to assist with code reviews, architecture
discussions, and implementation details. You are also aware of the project's goals and actively provide
insights on how to achieve them effectively.

## Project Overview

`genai` is a high-performance, professional-grade Go client library for Large Language Models (LLMs). It
provides a unified interface to interact with 15+ LLM providers while maintaining type safety, performance,
and ease of use.

### Core Design Principles

- **Performance first**: Minimize memory allocations, use compression, optimize for speed
- **Type safety**: Leverage Go's static typing, fail fast on unknown fields
- **Professional grade**: Production-ready with comprehensive testing and error handling
- **Stateless**: No global state, safe for concurrent use without locks
- **Minimal dependencies**: Lean implementation without unnecessary abstractions

### Key Features

- **Multi-provider support**: Anthropic, OpenAI, Google Gemini, Groq, Mistral, and many more
- **Tool calling via reflection**: Define tools as Go structs, automatic JSON handling
- **Native JSON serialization**: Strongly typed request/response with struct tags
- **Streaming support**: Real-time response streaming including tool calls
- **Multi-modal**: Images, PDFs, videos, audio input/output
- **Testing friendly**: HTTP record/playback for reproducible tests

## Code Style & Conventions

### Style

- Always use the latest Go features. You can see the current version in go.mod.
  - Integer range clause, e.g. for i := range number
  - any instead of interface{}
  - Go iterators, e.g. strings.SplitSeq()
- Prefer short variable names as long as possible.

### File Headers

All Go files must start with:
```go
// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.
```
with the current year.

### Package Documentation

Every package should have comprehensive documentation explaining:
- What the package does
- How it fits into the larger system
- Links to relevant external documentation
- References to official client libraries when applicable

### Error Handling

- Always handle errors explicitly
- Provide meaningful error messages with context
- Use `fmt.Errorf` for error wrapping when appropriate
- Validate inputs early and return descriptive errors

### Testing

- Write comprehensive unit tests for all functionality
- Use table-driven tests for multiple scenarios
- Use subtest to separately test valid and error code paths
- Test files are named `*_test.go`

### Performance Considerations

- Minimize memory allocations in hot paths
- Use `bytes.Buffer` and similar for efficient string building

## Architecture Patterns

### Type Definitions

- Use clear, short descriptive names for types
- Include JSON tags for serialization, use `omitzero` for optional fields
- Add Validate() method for complex types so it implements the `genai.Validatable` interface
- Use enums (constants) for fixed value sets
- Document field constraints in comments
- Prefer typed structs over `any`: Always extract nested structures into named types if possible.

## Common Patterns

### Validation

```go
func (o *Options) Validate() error {
    if o.Temperature < 0 || o.Temperature > 2 {
        return fmt.Errorf("temperature must be between 0 and 2, got %f", o.Temperature)
    }
    return nil
}
```

## Documentation

### Code Comments

- Document exported functions and types
- Explain complex algorithms or business logic
- Include usage examples in `example_test.go` for non-obvious APIs
- Do not document inside the function body unless to explain why the behavior is important.

### README Updates

- Update feature matrix when adding provider support
- Add usage examples for new important features
- Update installation and setup instructions

## Security Considerations

- Never commit API keys or secrets
- Sanitize test data and recordings
- Validate all inputs to prevent injection attacks
- Handle sensitive data appropriately

## Performance Optimization

### Memory Management

- Prefer stack allocation over heap when possible
- Use `bytes.Buffer` for efficient string concatenation

## Common Pitfalls to Avoid

- Don't ignore errors or use `panic()` in library code
- Prefer to not use global state or package-level variables if possible
- Prefer to not hardcode timeouts or limits without making them configurable
- Don't assume all providers support the same features
- Don't commit test recordings with real API keys
- Ask before breaking backward compatibility
- **Mandatory**: Never use `RECORD=all`, selectively delete cassettes instead. Explicitly ask the user before.

## Contributing Guidelines

1. **Before making changes**: Understand the existing patterns and conventions
2. **Write tests first**: Test-driven development is preferred
3. **Update documentation**: Keep README.md and code comments current
4. **Run the full test suite**: Ensure all tests pass before submitting
5. **Follow Go conventions**: Use `gofmt`, `golint`, `go vet -vettool=shadow`, `staticcheck` and `gosec`
6. **Add examples**: Include usage examples for new features

---

*This document should be updated as the project evolves. When in doubt, follow existing patterns in the
codebase and prioritize clarity, performance, and reliability.*

<!-- BEGIN FILE INDEX -->
## File Index

Autogenerated from first-line comments. Run scripts/update_agents_file_index.py to refresh.

- `.github/workflows/test.yml`: Run tests and linters on push.
- `.github/workflows/weekly-model-regen.yml`: Weekly model regeneration workflow.
- `.golangci.yml`: golangci-lint configuration.
- `README.md`: genai
- `adapters/adapters.go`: Package adapters includes multiple adapters to convert one ProviderFoo interface into another one.
- `adapters/adapters_test.go`: Tests for the adapters package.
- `adapters/example_test.go`: Example usage of the adapters package.
- `adapters/reasoning.go`: Package adapters provides adapter wrappers for the genai.Provider interface.
- `adapters/reasoning_test.go`: Tests for the reasoning adapter.
- `base/base.go`: Package base provides shared infrastructure for implementing genai providers.
- `base/base_test.go`: Tests for the base package.
- `cmd/cache-mgr/main.go`: Command cache-mgr fetches and prints out the list of files stored on the selected provider.
- `cmd/list-models/main.go`: Command list-models fetches and prints out the list of models from the selected providers.
- `cmd/list-models/main_test.go`: Tests for list-models
- `cmd/llama-serve/README.md`: llama-serve
- `cmd/llama-serve/main.go`: Command llama-serve fetches a model from HuggingFace and runs llama-server.
- `cmd/scoreboard/list.go`: Command scoreboard provides a list of available models.
- `cmd/scoreboard/main.go`: Command scoreboard generates a scoreboard for every providers supported.
- `cmd/scoreboard/main_test.go`: Tests for the scoreboard command.
- `cmd/scoreboard/smoke.go`: Smoke testing for the scoreboard command.
- `cmd/scoreboard/table.go`: Command scoreboard provides a table view of models.
- `docs/AGENTS.md`: Generated documentation
- `example_test.go`: Example tests for the genai package.
- `examples/AGENTS.md`: Examples how to use genai
- `genai.go`: Package genai is the opiniated high performance professional-grade AI package for Go.
- `genai_test.go`: Test helpers and utilities.
- `goption.go`: GenOption and related types for configuring GenSync and GenStream calls.
- `goption_test.go`: Tests for the generic option types.
- `httprecord/example_test.go`: Example usage of the httprecord package.
- `httprecord/httprecord.go`: Package httprecord provides safe HTTP recording logic for users that was to understand the API and do smoke
- `internal/AGENTS.md`: Generated documentation
- `poption.go`: ProviderOption and related types for configuring provider constructors.
- `poption_test.go`: Tests for the provider option types.
- `providers/AGENTS.md`: All providers and provider development guide
- `scoreboard/scoreboard.go`: Package scoreboard declares the structures to define a scoreboard.
- `scoreboard/scoreboard_test.go`: Tests for the scoreboard package.
- `smoke/smoke.go`: Package smoke runs a smoke test to generate a scoreboard.Scenario.
- `smoke/smoketest/smoketest.go`: Package smoketest runs a scoreboard in test mode.
- `smoke/tools.go`: Package smoke provides smoke testing utilities for genai providers.
- `subprocessrecord/subprocessrecord.go`: Package subprocessrecord provides recording and replay of subprocess I/O for
- `subprocessrecord/subprocessrecord_test.go`: Tests for the subprocessrecord package.
- `websocketrecord/example_test.go`: Example usage of the websocketrecord package.
- `websocketrecord/websocketrecord.go`: Package websocketrecord provides recording and replay of WebSocket message
- `websocketrecord/websocketrecord_test.go`: Tests for the websocketrecord package.
<!-- END FILE INDEX -->
