# Provider Development Guide

When investigating a feature or doing gap analysis compared to an official SDK, first do a git clone to
investigate more efficiently. Only then look at the online documentation to confirm if needed.

## Implementing a New Provider

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

3. **Handle provider-specific features**:
   - Map genai types to provider-specific API structures
   - Implement proper error handling for rate limits, auth, etc
   - Support streaming if the provider offers it
   - Handle provider-specific limitations gracefully
   - Implement `Raw` suffix methods for raw API access

### Authentication

- Support environment variables for API keys
- Handle different auth mechanisms (API key, OAuth, etc.)
- Provide clear error messages for auth failures

### Feature Support

- Clearly document which features each provider supports
- Use the feature matrix in README.md as reference
- Implement graceful degradation for unsupported features

## Testing

### Test Data Management

- Store test data in `testdata/` directories
- Use descriptive filenames for test cases
- Include both positive and negative test cases
- Keep test data minimal but comprehensive

### HTTP Recording

- Use package internal/internaltest for HTTP recording, it sanitizes sensitive data (API keys, personal info)
- Update recordings when API changes
- Test both success and error scenarios
- **NEVER** create or manually edit cassette YAML files in `testdata/`. Record them against the real API
  with `RECORD=failure_only go test` to replay existing cassettes and automatically re-record only the
  failed ones. The cassette matcher compares requests byte-for-byte (body, headers, content-length, proto)
  and hand-crafted files are fragile and unreliable.
- When tests fail due to stale or missing cassettes, re-record them with
  `RECORD=failure_only go test ./<directory>`. **Always default to `RECORD=failure_only`** (only re-records
  failed cassettes). For surgical re-recording, delete the specific cassette files that need updating, then
  run `RECORD=failure_only go test ./<directory>`. Only use `RECORD=all` when explicitly asked by the user.
  Use `-run` to scope recording to specific tests.
- When recording smoketests against live APIs, run fewer models at a time (use `-run`) to avoid exceeding
  the default 10-minute `go test` timeout. Reasoning models in particular can generate tens of thousands
  of tokens per test, making a full recording run take 20+ minutes.

### Smoke Testing with Automatic Scoreboard Updates

The smoke test framework supports automatic updating of `scoreboard.json` files when test results change.
This is particularly useful when provider capabilities evolve.

- **Unit tests**: Test individual functions and methods
- **Smoke tests**: Test against live services with recorded traces
- **Functionality tests**: Test provider capabilities systematically

### Editing scoreboard.json

You may manually edit the **metadata fields** in `scoreboard.json`: `country`, `dashboardURL`, and
`warnings`. These are not derived from tests and can be corrected directly.

You may also move a model between tested and untested scenarios. For example, move a model from the untested
bucket into a tested scenario when you know it shares capabilities with the models already there, or move it
back to untested when recordings are stale. After any such change, re-run the tests to validate.

**DO NOT manually edit the model feature data** (the `scenarios` array with model capabilities, supported
modalities, etc.) **or use `jq` to update it.** Always use `go test -update-scoreboard` instead:

```bash
go test ./providers/<provider> -update-scoreboard
```

The test framework automatically discovers models, records interactions, and updates the scoreboard correctly.

After updating any `scoreboard.json`, run `go generate ./...` to regenerate the documentation files (e.g.
`docs/*.md`).

## TODO

- When updating a provider's `Warmup.yaml` (model list), always verify that the SOTA/Good/Cheap model
  selection logic (`selectBestTextModel`, `selectBestImageModel`, etc.) resolves to the correct latest
  models. New models regularly become available and the selection logic must match.
