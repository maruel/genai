# Implementation Plan: Anthropic Provider Feature Parity

Plan date: 2026-02-07

Reference SDK: `github.com/anthropics/anthropic-sdk-go`

## Feature Coverage Matrix

| Feature                        | Official SDK | Local Provider | Status         |
|--------------------------------|:------------:|:--------------:|----------------|
| **GA Features**                |              |                |                |
| Messages API (sync)            | Yes          | Yes            | Complete       |
| Messages API (stream)          | Yes          | Yes            | Complete       |
| Token Counting                 | Yes          | Yes            | Complete       |
| Structured Outputs             | Yes          | Yes            | Complete       |
| Effort Level                   | Yes          | Yes            | Complete       |
| Adaptive Thinking              | Yes          | Yes            | Complete       |
| Model Listing (paginated)      | Yes          | Yes            | Complete       |
| Model Get (single)             | Yes          | Yes            | Complete       |
| Batch: Create                  | Yes          | Yes            | Complete       |
| Batch: Poll/Results            | Yes          | Yes            | Complete       |
| Batch: Cancel                  | Yes          | Yes            | Complete       |
| Batch: Get by ID               | Yes          | No             | Phase 2.4      |
| Batch: List                    | Yes          | No             | Phase 2.4      |
| Batch: Delete                  | Yes          | No             | Phase 2.4      |
| Tool Calling (custom)          | Yes          | Yes            | Complete       |
| Tool Choice (auto/any/tool/none) | Yes        | Yes            | Complete       |
| Web Search Tool                | Yes          | Yes            | Complete       |
| Extended Thinking (all modes)  | Yes          | Yes            | Complete       |
| Citations                      | Yes          | Yes            | Complete       |
| Prompt Caching                 | Yes          | Partial        | TODO #1        |
| Images (base64 + URL)          | Yes          | Yes            | Complete       |
| Documents/PDFs                 | Yes          | Yes            | Complete       |
| Inference Geography            | Yes          | Yes            | Complete       |
| Service Tier (request)         | Yes          | Yes            | Complete       |
| Service Tier (response/usage)  | Yes          | Yes            | Complete       |
| Container/Session              | Yes          | Yes            | Complete       |
| Bash Tool (GA 2025-01-24)      | Yes          | Yes            | Complete       |
| Text Editor v1 (2025-01-24)    | Yes          | Yes            | Complete       |
| Text Editor v2 (2025-04-29)    | Yes          | Yes            | Complete       |
| Text Editor v3 (2025-07-28)    | Yes          | Yes            | Complete       |
| Claude Opus 4.6 model          | Yes          | Yes            | Complete       |
| **Beta Features**              |              |                |                |
| MCP v1 (2025-04-04)            | Yes          | Yes            | Complete       |
| MCP v2 (2025-11-20)            | Yes          | No             | Phase 3.6      |
| File Uploads                   | Yes          | No             | Phase 3.1      |
| Code Execution Tool            | Yes          | No             | Phase 3.2      |
| Web Fetch Tool                 | Yes          | No             | Phase 3.3      |
| Context Management             | Yes          | No             | Phase 3.4      |
| Computer Use v1 (2024-10-22)   | Yes          | Yes (type)     | Partial        |
| Computer Use v2 (2025-01-24)   | Yes          | Yes (type)     | Partial        |
| Computer Use v3 (2025-11-24)   | Yes          | No             | Phase 3.5      |
| Agent Skills                   | Yes          | No             | Phase 4.2      |
| Memory Tool                    | Yes          | No             | Phase 4.3      |
| Search Tools (BM25/Regex)      | Yes          | No             | Phase 4.4      |
| Interleaved Thinking           | Yes          | No             | Phase 4.5      |
| Token Efficient Tools          | Yes          | No             | Phase 4.5      |
| Output 128k                    | Yes          | No             | Phase 4.5      |
| Extended Cache TTL             | Yes          | No             | Phase 4.5      |
| Context 1M                     | Yes          | No             | Phase 4.5      |
| **Intentionally Skipped**      |              |                |                |
| Legacy Text Completions        | Yes          | No             | Deprecated     |
| Bedrock adapter                | Yes          | No             | Out of scope   |
| Vertex AI adapter              | Yes          | No             | Out of scope   |

---

## Phase 2: GA Completeness

Goal: Fill in remaining GA feature gaps. Lower impact but improves completeness.

### 2.1 Inference Geography

**Priority**: P2

**Steps**:
1. Add `InferenceGeo string` field to `ChatRequest` with JSON tag `json:"inference_geo,omitzero"`.
2. Add `InferenceGeo` field to `GenOptionText`.
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

### 2.5 Service Tier in Usage Response — COMPLETE

**Priority**: P3

Added `ServiceTier string` to `genai.Usage`. Wired in Anthropic (sync, stream, batch
poll), OpenAI Chat (sync, stream), OpenAI Responses (sync, stream, poll), and Groq (sync).

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
1. Add `BetaHeaders []string` field to `GenOptionText`.
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

Pre-existing TODOs found in `client.go` that should be addressed alongside or independently
of the feature work above:

1. **Line ~284**: System prompt auto-caching — currently commented out. Enable by default or
   make configurable.

2. **Line ~816**: Support text citations in tool results and images in tool results.

3. **Line ~867**: Store MCP tool use input for multi-turn thread continuation (`ContentServerToolUse`
   value is dropped, making the next request unusable).

4. **Line ~863**: Keep `EncryptedContent` from web search results for session continuation.

5. **Line ~1374**: Batch `CustomID` is hardcoded to `"TODO"` — must generate unique IDs in
   production.

6. **Line ~1852**: `pkt.Index` matters for simultaneous content blocks but is not handled.

7. **Line ~1988**: Stream error decoding is a placeholder (`%+v` formatting).

---

## Summary

| Phase | Items | Effort   | Impact |
|-------|-------|----------|--------|
| 2     | 5     | Small    | Medium |
| 3     | 6     | Medium   | Medium |
| 4     | 5     | Small    | Low    |

**Recommended next**: Phase 2 (GA completeness), starting with 2.1 and 2.2 as trivial wins.
