# Implementation Plan: Anthropic Provider Feature Parity

Plan date: 2026-02-07

Reference: [gap_analysis.md](gap_analysis.md)

## Phased Approach

Work is organized in four phases by impact and effort. Each phase can be shipped
independently. Within each phase, items are ordered by recommended implementation sequence.

---

## Phase 1: High-Impact GA Features

Goal: Close the most visible functional gaps. These are GA features that users expect.

### 1.1 Structured Outputs (`output_config.format`)

**Priority**: P0 — this is the single largest gap. Every other text provider supports
`DecodeAs`/`ReplyAsJSON` through `genai.GenOptionsText`, but Anthropic returns an error.

**API shape** (from official SDK v1.20.0):
```json
{
  "output_config": {
    "format": {
      "type": "json_schema",
      "json_schema": { ... }
    }
  }
}
```

**Implementation steps**:

1. Add types to `client.go`:
   ```go
   type OutputConfig struct {
       Format *OutputFormat `json:"format,omitzero"`
       Effort string        `json:"effort,omitzero"` // "low", "medium", "high", "max"
   }

   type OutputFormat struct {
       Type       string             `json:"type"`        // "json_schema"
       JSONSchema *jsonschema.Schema `json:"json_schema"` // the schema
   }
   ```

2. Add `OutputConfig` field to `ChatRequest`:
   ```go
   OutputConfig OutputConfig `json:"output_config,omitzero"`
   ```

3. Modify `initOptionsText()` to handle `DecodeAs` and `ReplyAsJSON`:
   - `DecodeAs != nil`: Generate JSON schema via `internal.JSONSchemaFor()`, set
     `output_config.format.type = "json_schema"` and populate the schema. The official SDK
     applies schema transformations (add `additionalProperties: false`, convert `oneOf` to
     `anyOf`, filter string formats, limit `minItems`). Replicate these transforms.
   - `ReplyAsJSON`: Set `output_config.format.type = "json_schema"` with a permissive
     schema (`{"type": "object"}` or similar), OR check if Anthropic supports a simpler
     `"json"` format type.

4. Remove the hard errors for `ReplyAsJSON` and `DecodeAs` in `initOptionsText()`.

5. Parse the structured response — the API returns content as a text block containing JSON.
   The existing `ToResult()` should work unchanged since it already extracts text content.

6. Write unit tests:
   - Table-driven test: `DecodeAs` with a struct produces correct `output_config`.
   - Table-driven test: `ReplyAsJSON` produces correct `output_config`.
   - Negative test: invalid `DecodeAs` type still fails validation.

7. Run smoke tests when possible (HTTP recording).

**Files changed**: `client.go`, `client_test.go`

**Depends on**: Nothing. Self-contained.

**Schema transformation notes**: The official SDK's `transformSchema()` does:
- Sets `additionalProperties: false` on all objects.
- Converts `oneOf` to `anyOf`.
- Filters string formats to only allowed ones.
- Limits `minItems` to at most 200.
- These transforms should be ported to ensure schema compatibility.

### 1.2 Token Counting

**Priority**: P0 — useful for cost estimation and context window management.

**API endpoint**: `POST /v1/messages/count_tokens`

**Request** (same shape as `ChatRequest` minus `stream`):
```json
{
  "model": "claude-sonnet-4-5-20250929",
  "messages": [...],
  "system": [...],
  "tools": [...],
  "tool_choice": {...},
  "thinking": {...}
}
```

**Response**:
```json
{
  "input_tokens": 1234
}
```

**Implementation steps**:

1. Add response type:
   ```go
   type CountTokensResponse struct {
       InputTokens int64 `json:"input_tokens"`
   }
   ```

2. Add methods to `Client`:
   ```go
   func (c *Client) CountTokens(ctx context.Context, msgs genai.Messages, opts ...genai.GenOptions) (int64, error)
   func (c *Client) CountTokensRaw(ctx context.Context, in *ChatRequest) (CountTokensResponse, error)
   ```

3. `CountTokens` reuses `ChatRequest.Init()` to build the request, then POSTs to
   `/v1/messages/count_tokens` instead of `/v1/messages`.

4. Consider whether to add a `genai` interface (e.g. `ProviderTokenCount`). If other
   providers (Gemini, OpenAI) also support token counting, it should be a shared interface.
   Otherwise, expose as Anthropic-specific only.

5. Write unit tests with HTTP recording.

**Files changed**: `client.go`, `client_test.go`

**Depends on**: Nothing. Self-contained.

### 1.3 Effort Level

**Priority**: P1

**Implementation steps**:

1. Add `Effort` field to Anthropic-specific `GenOptionsText`:
   ```go
   type GenOptionsText struct {
       ThinkingBudget  int64
       MessagesToCache int
       Effort          string // "low", "medium", "high", "max"; empty = default
   }
   ```

2. In `initImpl()`, when processing `*GenOptionsText`, set `c.OutputConfig.Effort = v.Effort`.

3. Add validation in `GenOptionsText.Validate()`.

4. Write unit tests.

**Files changed**: `client.go`, `client_test.go`

**Depends on**: 1.1 (OutputConfig struct must exist first).

### 1.4 Adaptive Thinking

**Priority**: P1

**Implementation steps**:

1. Add a `ThinkingMode` field to Anthropic-specific `GenOptionsText`:
   ```go
   type GenOptionsText struct {
       ThinkingBudget  int64
       ThinkingMode    string // "enabled", "disabled", "adaptive"; empty = auto-detect
       MessagesToCache int
       Effort          string
   }
   ```
   When `ThinkingMode` is empty, preserve current behavior (auto-detect from
   `ThinkingBudget`). When `ThinkingMode` is `"adaptive"`, set `Thinking.Type = "adaptive"`
   with the budget.

2. Alternatively, keep the current `ThinkingBudget`-based detection and add a sentinel value
   (e.g. `ThinkingBudget = -1` for adaptive). This is simpler but less explicit.

3. The `Thinking` struct needs to support `type: "adaptive"` as a valid value.

4. Write unit tests.

**Files changed**: `client.go`, `client_test.go`

**Depends on**: Nothing. Self-contained.

### 1.5 Claude Opus 4.6 Model

**Priority**: P1 — trivial, do early.

**Implementation steps**:

1. Add `"claude-opus-4-6"` to the `modelsMaxTokens()` switch (check official docs for max
   tokens — likely 32000 or 64000).

2. Update model alias selection if `ModelSOTA` should resolve to it.

3. Run `go test ./providers/anthropic -update-scoreboard` to update `scoreboard.json`.

**Files changed**: `client.go`, `scoreboard.json`

**Depends on**: Nothing.

---

## Phase 2: GA Completeness

Goal: Fill in remaining GA feature gaps. Lower impact but improves completeness.

### 2.1 Inference Geography

**Priority**: P2

**Steps**:
1. Add `InferenceGeo string` field to `ChatRequest` with JSON tag `json:"inference_geo,omitzero"`.
2. Add `InferenceGeo` field to `GenOptionsText`.
3. Wire in `initImpl()`.

**Effort**: Trivial. One field addition.

### 2.2 Text Editor Tool v2/v3

**Priority**: P2

**Steps**:
1. Add `"text_editor_20250429"` and `"text_editor_20250728"` to `Tool.Type` documentation
   and validation.
2. Add the corresponding `Name` mappings (both use `"str_replace_editor"`).

**Effort**: Trivial. Documentation and validation update.

### 2.3 Model Get Endpoint

**Priority**: P3

**Steps**:
1. Add `GetModel(ctx, modelID string) (Model, error)` method.
2. Add `GetModelRaw()` variant.
3. HTTP GET to `/v1/models/{model_id}`.

**Effort**: Small.

### 2.4 Batch API CRUD

**Priority**: P3

**Steps**:
1. Add `GetBatch(ctx, batchID string) (BatchResponse, error)` — GET
   `/v1/messages/batches/{id}`.
2. Add `ListBatches(ctx) ([]BatchResponse, error)` — GET `/v1/messages/batches`.
3. Add `DeleteBatch(ctx, batchID string) error` — DELETE `/v1/messages/batches/{id}`.
4. Add pagination support for `ListBatches`.

**Effort**: Small-Medium.

### 2.5 Service Tier in Usage Response

**Priority**: P3

**Steps**:
1. The `Usage` struct already has `ServiceTier string`. Decide whether to expose it in
   `genai.Usage` or leave it accessible via raw responses only.
2. If adding to `genai.Usage`, add a `ServiceTier` field there.

**Effort**: Trivial.

---

## Phase 3: High-Value Beta Features

Goal: Support beta features that are commonly used in agent workflows.

### 3.1 File Uploads

**Priority**: P2

**Steps**:
1. Add file API types: `FileMetadata`, `FileUploadRequest`, `FileListResponse`.
2. Add methods: `FileUpload()`, `FileDownload()`, `FileList()`, `FileDelete()`,
   `FileGetMetadata()`.
3. Add `FileDocumentSource` and `FileImageSource` content source types.
4. Beta header: `files-api-2025-04-14`.
5. Write tests with HTTP recording.

**Effort**: Medium. New API surface but straightforward CRUD.

### 3.2 Code Execution Tool

**Priority**: P2

**Steps**:
1. Add tool type constants: `"code_execution_20250522"`, `"code_execution_20250825"`.
2. Add response content block types for code execution results.
3. Add container configuration (`BetaContainerParams`).
4. Beta header: `code-execution-2025-05-22`.
5. Wire through Tool struct — no new fields needed beyond Type.

**Effort**: Small-Medium.

### 3.3 Web Fetch Tool

**Priority**: P2

**Steps**:
1. Add tool type constant: `"web_fetch_20250910"`.
2. Add response block type for web fetch results.
3. Handle in streaming and sync response parsing.

**Effort**: Small. Similar pattern to existing web search tool.

### 3.4 Context Management

**Priority**: P2

**Steps**:
1. Add context management config types:
   ```go
   type ContextManagement struct {
       InputTokensClearAtLeast int64 `json:"input_tokens_clear_at_least,omitzero"`
       InputTokensTrigger      int64 `json:"input_tokens_trigger,omitzero"`
       ToolUsesKeep            int64 `json:"tool_uses_keep,omitzero"`
       ToolUsesTrigger         int64 `json:"tool_uses_trigger,omitzero"`
   }
   ```
2. Add `ContextManagement` field to `ChatRequest`.
3. Add compaction response block type to content parsing.
4. Beta header: `context-management-2025-06-27`.

**Effort**: Medium.

### 3.5 Computer Use v3

**Priority**: P3

**Steps**:
1. Add `"computer_20251124"` to Tool.Type enum.
2. Add any new fields specific to v3.
3. Beta header: `computer-use-2025-01-24` (may cover v3 too).

**Effort**: Small.

### 3.6 MCP v2

**Priority**: P3

**Steps**:
1. Update beta header from `mcp-client-2025-04-04` to `mcp-client-2025-11-20` (or send
   both).
2. Review if new MCP v2 fields need to be added to `MCPServer` struct.
3. Add `MCPToolset` type if MCP v2 uses a different toolset configuration.

**Effort**: Small.

---

## Phase 4: Lower-Priority Beta Features

Goal: Comprehensive beta coverage for power users.

### 4.1 Beta Headers Passthrough

**Priority**: P3

Add a provider-specific option to pass arbitrary beta header strings. This future-proofs
against new betas without code changes.

**Steps**:
1. Add `BetaHeaders []string` field to `GenOptionsText`.
2. Append to the `anthropic-beta` header in request construction.

**Effort**: Trivial.

### 4.2 Agent Skills

**Priority**: P4

Full CRUD service for agent skills. This is a large feature that may be better suited as a
separate sub-package or may not fit the provider abstraction.

**Effort**: Medium-Large. Evaluate whether it belongs in this provider.

### 4.3 Memory Tool

**Priority**: P4

Add `"memory_20250818"` tool type. Relatively straightforward.

**Effort**: Small.

### 4.4 Search Tools (BM25/Regex)

**Priority**: P4

Add tool types for `"search_tool_bm25_20251119"` and `"search_tool_regex_20251119"`.

**Effort**: Small.

### 4.5 Remaining Beta Flags

Lower priority beta headers that can be enabled via the passthrough mechanism (4.1):
- `token-efficient-tools-2025-02-19`
- `output-128k-2025-02-19`
- `extended-cache-ttl-2025-04-11`
- `context-1m-2025-08-07`
- `interleaved-thinking-2025-05-14`
- `dev-full-thinking-2025-05-14`

**Effort**: Trivial each (if passthrough exists), Small otherwise.

---

## Existing TODOs in Code

These are pre-existing TODOs found in `client.go` that should be addressed alongside or
independently of the feature work above:

1. **Line ~186**: System prompt auto-caching — currently commented out. Enable by default or
   make configurable.

2. **Line ~715**: Support text citations in tool results and images in tool results.

3. **Line ~626**: Store MCP tool use input for multi-turn thread continuation.

4. **Line ~766**: Keep `EncryptedContent` from web search results for session continuation.

5. **Line ~787**: Support new server tool types in non-lenient mode.

6. **Line ~1272**: Batch `CustomID` is hardcoded to `"TODO"` — must generate unique IDs in
   production.

---

## Cross-Cutting Concerns

### genai Package Changes

Some features may require additions to the shared `genai` package:

| Feature            | genai Change Needed?                                     |
|--------------------|----------------------------------------------------------|
| Structured Outputs | No — `DecodeAs`/`ReplyAsJSON` already exist              |
| Token Counting     | Maybe — consider `ProviderTokenCount` interface          |
| Effort Level       | Maybe — consider shared `GenOptionsEffort` type          |
| Adaptive Thinking  | No — provider-specific option                            |
| File Uploads       | Maybe — consider shared file reference types             |
| Batch CRUD         | Maybe — consider extending `ProviderCapabilities`        |

### Testing Strategy

- **Unit tests**: All new types need validation tests. All new `Init()` paths need coverage.
- **Smoke tests**: Features that change the API request shape need HTTP recording tests.
  Run with `-update-scoreboard` for model capability changes.
- **Negative tests**: Ensure unsupported combinations (e.g. structured output + thinking)
  return clear errors.

### Backward Compatibility

- `GenOptionsText` struct gains new fields — zero values preserve existing behavior.
- `ChatRequest` gains new fields — `omitzero` tags ensure no wire format change for
  existing users.
- No existing behavior changes unless explicitly opted in.

---

## Summary

| Phase | Items | Effort   | Impact |
|-------|-------|----------|--------|
| 1     | 5     | Medium   | High   |
| 2     | 5     | Small    | Medium |
| 3     | 6     | Medium   | Medium |
| 4     | 5     | Small    | Low    |

**Recommended start**: Phase 1.5 (Opus 4.6 model) first as a quick win, then 1.1 (Structured
Outputs) as the highest-impact feature, then 1.2 (Token Counting).
