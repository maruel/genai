# Dear LLM 🤖

This document provides guidance for AI assistants working on the `genai` project.

## Project Overview

`genai` is a high-performance, professional-grade Go client library for Large Language Models (LLMs). It provides a unified interface to interact with 15+ LLM providers while maintaining type safety, performance, and ease of use.

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
- Prefer HTTP compression (brotli > gzip) when supported
- Use streaming when possible to reduce latency
- Profile and benchmark performance-critical code

## Architecture Patterns

### Provider Implementation

When implementing a new provider:

1. **Create package structure**:
   ```
   providers/newprovider/
   ├── client.go          # Main client implementation
   ├── client_test.go     # Unit tests
   ├── testdata/          # Recorded HTTP interactions
   └── doc.go             # Package documentation
   ```

2. **Implement required interfaces**:
   - `genai.Provider` for basic text generation
   - `genai.ProviderImage` for image generation (if supported)
   - `genai.ProviderAudio` for audio generation (if supported)

3. **Follow naming conventions**:
   - Constructor: `New(apiKey, model string, opts *Options) (*Client, error)`
   - Main client struct: `Client`
   - Provider-specific options: `Options`

4. **Handle provider-specific features**:
   - Map genai types to provider-specific API structures
   - Implement proper error handling for rate limits, auth, etc.
   - Support streaming if the provider offers it
   - Handle provider-specific limitations gracefully

### Type Definitions

- Use clear, descriptive names for types
- Include JSON tags for serialization
- Add validation methods for complex types
- Use enums (constants) for fixed value sets
- Document field constraints in comments

### Testing Strategy

- **Unit tests**: Test individual functions and methods
- **Integration tests**: Test against recorded HTTP interactions
- **Smoke tests**: Test against live services with recorded traces
- **Functionality tests**: Test provider capabilities systematically

## Common Patterns

### HTTP Client Configuration

```go
// Configure HTTP client with compression and reasonable timeouts
client := &http.Client{
    Timeout: 30 * time.Second,
    Transport: &http.Transport{
        // Configure compression, connection pooling, etc.
    },
}
```

### Error Handling

```go
if err != nil {
    return fmt.Errorf("operation failed: %w", err)
}
```

### Validation

```go
func (o *Options) Validate() error {
    if o.Temperature < 0 || o.Temperature > 2 {
        return fmt.Errorf("temperature must be between 0 and 2, got %f", o.Temperature)
    }
    return nil
}
```

### JSON Schema Generation

```go
// Use jsonschema for automatic schema generation from Go structs
schema := jsonschema.Reflect(myStruct)
```

## Provider-Specific Notes

### Authentication

- Support environment variables for API keys
- Handle different auth mechanisms (API key, OAuth, etc.)
- Provide clear error messages for auth failures

### Rate Limiting

- Implement proper backoff strategies
- Handle rate limit responses gracefully
- Provide meaningful error messages about limits

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

### Performance Testing

- Write benchmarks for performance-critical code
- Test memory allocation patterns
- Monitor performance regression
- Use `go test -bench` and `go test -race`

## Documentation

### Code Comments

- Document exported functions and types
- Explain complex algorithms or business logic
- Include usage examples for non-obvious APIs
- Document error conditions and return values

### README Updates

- Update feature matrix when adding provider support
- Add usage examples for new features
- Update installation and setup instructions
- Keep the "I'm poor 💸" section current with free tiers

## Security Considerations

- Never commit API keys or secrets
- Sanitize test data and recordings
- Validate all inputs to prevent injection attacks
- Use secure HTTP transport (TLS)
- Handle sensitive data appropriately

## Performance Optimization

### Memory Management

- Use object pooling for frequently allocated objects
- Prefer stack allocation over heap when possible
- Use `bytes.Buffer` for efficient string concatenation
- Profile memory usage with `go tool pprof`

### Network Optimization

- Enable HTTP compression when supported
- Use connection pooling and keepalive
- Implement request batching where possible
- Stream responses to reduce memory usage

## Common Pitfalls to Avoid

- Don't ignore errors or use `panic()` in library code
- Don't use global state or package-level variables
- Don't hardcode timeouts or limits without making them configurable
- Don't assume all providers support the same features
- Don't commit test recordings with real API keys
- Don't break backward compatibility without major version bump

## Contributing Guidelines

1. **Before making changes**: Understand the existing patterns and conventions
2. **Write tests first**: Test-driven development is preferred
3. **Update documentation**: Keep README.md and code comments current
4. **Run the full test suite**: Ensure all tests pass before submitting
5. **Follow Go conventions**: Use `gofmt`, `golint`, and `go vet`
6. **Add examples**: Include usage examples for new features

---

*This document should be updated as the project evolves. When in doubt, follow existing patterns in the codebase and prioritize clarity, performance, and reliability.*