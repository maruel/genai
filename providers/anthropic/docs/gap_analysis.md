# Gap Analysis: providers/anthropic vs Official Anthropic Go SDK

Comparison date: 2026-02-07

Reference SDK: `github.com/anthropics/anthropic-sdk-go` v1.21.0 (2026-02-05)

## Current State

The `providers/anthropic` package is a single-file implementation (~1892 lines in `client.go`)
that provides a production-grade Anthropic client covering sync/stream generation, batch
processing, tool calling, web search, MCP, citations, extended thinking, and prompt caching.

It implements the `genai.Provider` interface and exposes raw API access via `*Raw` suffix
methods.

## Feature Coverage Matrix

| Feature                        | Official SDK | Local Provider | Status         |
|--------------------------------|:------------:|:--------------:|----------------|
| **GA Features**                |              |                |                |
| Messages API (sync)            | Yes          | Yes            | Complete       |
| Messages API (stream)          | Yes          | Yes            | Complete       |
| Token Counting                 | Yes          | No             | **Missing**    |
| Structured Outputs             | Yes          | Yes            | Complete       |
| Effort Level                   | Yes          | Yes            | Complete       |
| Adaptive Thinking              | Yes          | No             | **Missing**    |
| Model Listing (paginated)      | Yes          | Yes            | Complete       |
| Model Get (single)             | Yes          | No             | **Missing**    |
| Batch: Create                  | Yes          | Yes            | Complete       |
| Batch: Poll/Results            | Yes          | Yes            | Complete       |
| Batch: Cancel                  | Yes          | Yes            | Complete       |
| Batch: Get by ID               | Yes          | No             | **Missing**    |
| Batch: List                    | Yes          | No             | **Missing**    |
| Batch: Delete                  | Yes          | No             | **Missing**    |
| Tool Calling (custom)          | Yes          | Yes            | Complete       |
| Tool Choice (auto/any/tool/none) | Yes        | Yes            | Complete       |
| Web Search Tool                | Yes          | Yes            | Complete       |
| Extended Thinking (enabled)    | Yes          | Yes            | Complete       |
| Extended Thinking (disabled)   | Yes          | Yes            | Complete       |
| Extended Thinking (adaptive)   | Yes          | No             | **Missing**    |
| Citations                      | Yes          | Yes            | Complete       |
| Prompt Caching                 | Yes          | Partial        | Missing auto system prompt caching |
| Images (base64 + URL)          | Yes          | Yes            | Complete       |
| Documents/PDFs                 | Yes          | Yes            | Complete       |
| Inference Geography            | Yes          | Partial        | Response field parsed; request field not wired |
| Service Tier (request)         | Yes          | Yes            | Complete       |
| Service Tier (response/usage)  | Yes          | Partial        | Not surfaced in Usage |
| Container/Session              | Yes          | Yes            | Complete       |
| Bash Tool (GA 2025-01-24)      | Yes          | Yes            | Complete       |
| Text Editor v1 (2025-01-24)    | Yes          | Yes            | Complete       |
| Text Editor v2 (2025-04-29)    | Yes          | No             | **Missing**    |
| Text Editor v3 (2025-07-28)    | Yes          | No             | **Missing**    |
| Claude Opus 4.6 model const    | Yes          | No             | **Missing**    |
| **Beta Features**              |              |                |                |
| MCP v1 (2025-04-04)            | Yes          | Yes            | Complete       |
| MCP v2 (2025-11-20)            | Yes          | No             | **Missing**    |
| File Uploads                   | Yes          | No             | **Missing**    |
| Code Execution Tool            | Yes          | No             | **Missing**    |
| Web Fetch Tool                 | Yes          | No             | **Missing**    |
| Context Management             | Yes          | No             | **Missing**    |
| Computer Use v1 (2024-10-22)   | Yes          | Yes (type)     | Partial        |
| Computer Use v2 (2025-01-24)   | Yes          | Yes (type)     | Partial        |
| Computer Use v3 (2025-11-24)   | Yes          | No             | **Missing**    |
| Agent Skills                   | Yes          | No             | **Missing**    |
| Memory Tool                    | Yes          | No             | **Missing**    |
| Search Tools (BM25/Regex)      | Yes          | No             | **Missing**    |
| Interleaved Thinking           | Yes          | No             | **Missing**    |
| Token Efficient Tools          | Yes          | No             | **Missing**    |
| Output 128k                    | Yes          | No             | **Missing**    |
| Extended Cache TTL             | Yes          | No             | **Missing**    |
| Context 1M                     | Yes          | No             | **Missing**    |
| **Intentionally Skipped**      |              |                |                |
| Legacy Text Completions        | Yes          | No             | Deprecated     |
| Bedrock adapter                | Yes          | No             | Out of scope   |
| Vertex AI adapter              | Yes          | No             | Out of scope   |

## Detailed Gap Descriptions

### 1. Structured Outputs (GA — COMPLETE)

The Anthropic API supports native structured output via `output_config.format` (GA since
SDK v1.20.0, 2026-01-29). This accepts a JSON schema and guarantees the response conforms to
it.

**Status**: Implemented. `DecodeAs` maps to `output_config.format` with a full JSON schema
generated via `internal.JSONSchemaFor()`. `ReplyAsJSON` maps to a permissive
`{"type": "object"}` schema. The `Effort` field is not yet wired (see gap #3).

### 2. Token Counting (GA — HIGH IMPACT)

`POST /v1/messages/count_tokens` returns the input token count for a request without
executing it.

**Current behavior**: No implementation. No corresponding `genai` interface exists.

**Official SDK**: `client.Messages.CountTokens()` accepting the same parameters as a normal
message request.

**What's needed**:
- Consider adding a `ProviderTokenCount` interface in the `genai` package (or skip and expose
  via raw method only).
- Add `CountTokensRequest`/`CountTokensResponse` structs.
- Add `CountTokens()` and `CountTokensRaw()` methods on `Client`.

### 3. Effort Level (GA — COMPLETE)

Controls quality/latency tradeoff: `"low"`, `"medium"`, `"high"`, `"max"`.

**Status**: Implemented. `Effort` type with constants added to `GenOptionsText`. Wired
through `ChatRequest.OutputConfig.Effort`. Validation rejects unknown values.

### 4. Adaptive Thinking (GA — MEDIUM IMPACT)

Third thinking mode alongside `"enabled"` and `"disabled"`. Lets the model decide
autonomously whether to use extended thinking.

**Current behavior**: Only `"enabled"` and `"disabled"` are supported via
`GenOptionsText.ThinkingBudget`.

**Official SDK**: `ThinkingConfigAdaptiveParam` with `budget_tokens` (since v1.21.0).

**What's needed**:
- Extend `Thinking` struct or `GenOptionsText` to support adaptive mode.
- Consider a new field like `ThinkingMode` with `"enabled"`, `"disabled"`, `"adaptive"`.

### 5. Claude Opus 4.6 Model (GA — TRIVIAL)

Latest SOTA model `claude-opus-4-6` is not in the model constants or `maxTokens` map.

**What's needed**:
- Add to `modelsMaxTokens()`.
- Update `ModelSOTA` if applicable.
- Update `scoreboard.json` (via `go test -update-scoreboard`).

### 6. Model Get Endpoint (GA — LOW IMPACT)

`GET /v1/models/{model_id}` returns info for a single model.

**What's needed**:
- Add `GetModel()` and `GetModelRaw()` methods.

### 7. Inference Geography (GA — LOW IMPACT)

`inference_geo` request field to control where inference runs.

**Status**: The `InferenceGeo` field is parsed from the `Usage` response. The request-side
field (to control where inference runs) is not yet wired to `ChatRequest` or
`GenOptionsText`.

### 8. Batch API Completeness (GA — LOW-MEDIUM IMPACT)

Missing `Get` (single batch status by ID), `List` (all batches), and `Delete` endpoints.

**What's needed**:
- Add `GetBatch()`, `ListBatches()`, `DeleteBatch()` methods.
- Current `PokeResult` polls results; `GetBatch` polls status without downloading results.

### 9. Text Editor Tool v2/v3 (GA — LOW-MEDIUM IMPACT)

Two newer versions: `text_editor_20250429` and `text_editor_20250728`.

**What's needed**:
- Update `Tool.Type` enum documentation and add the new type strings.
- Ensure validation accepts the new types.

### 10. File Uploads (BETA — MEDIUM IMPACT)

Full file lifecycle: upload, download, list, delete, reference in messages.

**What's needed**:
- New request/response types for the files API.
- New `FileUpload()`, `FileDownload()`, `FileList()`, `FileDelete()`, `FileGetMetadata()`
  methods.
- New content source type `FileDocumentSourceParam` / `FileImageSourceParam`.
- Beta header: `files-api-2025-04-14`.

### 11. Code Execution Tool (BETA — MEDIUM IMPACT)

Sandboxed code execution tool (two versions).

**What's needed**:
- Add tool type constants for `code_execution_20250522` and `code_execution_20250825`.
- Add response block types for code execution results.
- Beta header: `code-execution-2025-05-22`.

### 12. Web Fetch Tool (BETA — MEDIUM IMPACT)

Fetches and processes web content (distinct from web search).

**What's needed**:
- Add tool type constant for `web_fetch_20250910`.
- Add response block types.
- Beta header (shared with web search or standalone).

### 13. Context Management (BETA — MEDIUM IMPACT)

Auto-compaction for long agent sessions: clear thinking, clear tool uses, compaction
triggers.

**What's needed**:
- Add context management config fields to `ChatRequest`.
- Add compaction response block type.
- Beta header: `context-management-2025-06-27`.

### 14. MCP v2 (BETA — LOW IMPACT)

Updated MCP protocol version with expanded capabilities.

**What's needed**:
- Add `mcp-client-2025-11-20` beta header when MCP servers are configured.
- Review if new fields or behaviors differ from v1.

### 15. Remaining Beta Features (LOW IMPACT each)

- **Computer Use v3**: Add `computer_20251124` tool type.
- **Agent Skills**: Full CRUD service — likely too large for provider scope.
- **Memory Tool**: Add `memory_20250818` tool type.
- **Search Tools (BM25/Regex)**: Add tool types.
- **Beta flags passthrough**: Allow users to specify arbitrary beta headers.
- **Interleaved/Full Thinking**: Beta headers for thinking turn control.
- **Token Efficient Tools**: Beta header to reduce token usage for tool schemas.
- **Output 128k**: Beta header to allow up to 128k output tokens.
- **Extended Cache TTL**: Beta header for longer cache durations.
- **Context 1M**: Beta header for 1M context window.

### 16. Service Tier in Usage Response

The API returns `service_tier` in the usage block (`"standard"`, `"priority"`, `"batch"`).

**Current behavior**: The `Usage` struct has `ServiceTier` but it's not surfaced into
`genai.Usage`.

**What's needed**:
- Consider adding `ServiceTier` to `genai.Usage` or exposing via raw response only.
