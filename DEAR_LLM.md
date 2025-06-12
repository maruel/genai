# Dear LLM 🤖

This document provides guidance for AI assistants working on the `genai` project.

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

### File Headers

All Go files must start with:
```go
// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.
```

### Package Documentation

Every package should have comprehensive documentation explaining:
- What the package does
- How it fits into the larger system
- Links to relevant external documentation
- References to official client libraries when applicable

### Error Handling

- Always handle errors explicitly
- Provide meaningful error messages with context
- Use `fmt.Errorf` for error wrapping
- Validate inputs early and return descriptive errors

### Testing

- Write comprehensive unit tests for all functionality
- Use table-driven tests for multiple scenarios
- Test both success and error paths
- Use HTTP record/playback for provider integration tests
- Store test data in `testdata/` directories
- Test files should be named `*_test.go`

### Performance Considerations

- Minimize memory allocations in hot paths
- Use `bytes.Buffer` and similar for efficient string building

## Architecture Patterns

### Provider Implementation

When implementing a new provider:

1. **Create package structure**:
   ```
   providers/newprovider/
   ├── client.go          # Main client implementation
   ├── client_test.go     # Unit tests
   └── testdata/          # Recorded HTTP interactions
   ```

2. **Implement required interfaces**: For a new `Client`, implement the relevant interfaces from the `genai`
   package that start with `Provider`. Use go doc to get the up to date list. Ensure that the provider
   implements all necessary methods for the interfaces it claims to support.

3. **Follow naming conventions**:
   - Constructor: `New(apiKey, model string, opts *Options) (*Client, error)`
   - Main client struct: `Client`
   - Provider-specific options: `Options`

4. **Handle provider-specific features**:
   - Map genai types to provider-specific API structures
   - Implement proper error handling for rate limits, auth, etc.
   - Support streaming if the provider offers it
   - Handle provider-specific limitations gracefully
   - Implement `Raw` suffix methods for raw API access

### Type Definitions

- Use clear, short descriptive names for types
- Include JSON tags for serialization, use `omitzero` for optional fields
- Add Validate() method for complex types so it implements the `genai.Validatable` interface
- Use enums (constants) for fixed value sets
- Document field constraints in comments

### Testing Strategy

- **Unit tests**: Test individual functions and methods
- **Smoke tests**: Test against live services with recorded traces
- **Functionality tests**: Test provider capabilities systematically

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

## Provider-Specific Notes

### Authentication

- Support environment variables for API keys
- Handle different auth mechanisms (API key, OAuth, etc.)
- Provide clear error messages for auth failures

### Feature Support

- Clearly document which features each provider supports
- Use the feature matrix in README.md as reference
- Implement graceful degradation for unsupported features

## Testing Guidelines

### Test Data Management

- Store test data in `testdata/` directories
- Use descriptive filenames for test cases
- Include both positive and negative test cases
- Keep test data minimal but comprehensive

### HTTP Recording

- Use `gopkg.in/dnaeon/go-vcr.v4` for HTTP recording
- Sanitize sensitive data (API keys, personal info)
- Update recordings when API changes
- Test both success and error scenarios

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
- Don't use global state or package-level variables
- Don't hardcode timeouts or limits without making them configurable
- Don't assume all providers support the same features
- Don't commit test recordings with real API keys
- Ask before breaking backward compatibility

## Contributing Guidelines

1. **Before making changes**: Understand the existing patterns and conventions
2. **Write tests first**: Test-driven development is preferred
3. **Update documentation**: Keep README.md and code comments current
4. **Run the full test suite**: Ensure all tests pass before submitting
5. **Follow Go conventions**: Use `gofmt`, `golint`, `go vet -vettool=shadow`, `staticcheck` and `gosec`
6. **Add examples**: Include usage examples for new features

---

*This document should be updated as the project evolves. When in doubt, follow existing patterns in the codebase and prioritize clarity, performance, and reliability.*
