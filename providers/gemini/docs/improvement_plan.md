# Gemini Provider: Improvement Plan

## Phase 1: Quick Wins (fix existing gaps in current code)

### 1.1 Complete Code Execution Tool Support

**Status**: Structs exist, but `ProcessStream` raises errors on
`ExecutableCode` and `CodeExecutionResult` parts.

**Work**:
- Map `ExecutableCode` to a genai output type (text with language annotation)
- Map `CodeExecutionResult` to a genai output type (text with outcome status)
- Add `CodeExecution` to `Tool` in request building (already present in struct)
- Add smoke tests with a model that supports code execution

**Endpoint**: Same `generateContent` / `streamGenerateContent`. Enable via:
```json
{"tools": [{"codeExecution": {}}]}
```

**Effort**: Low (~50 lines)

### 1.2 Add CountTokens

**Status**: Not implemented. Single new endpoint.

**Work**:
- Add `CountTokens(ctx, msgs, opts) (TokenCount, error)` method
- Endpoint: `POST models/{model}:countTokens`
- Request body: same `contents` + `tools` + `systemInstruction` as generation
- Response: `{ "totalTokens": int32, "cachedContentTokenCount": int32 }`
- Implement the `ProviderCountTokens` interface (or add to genai if missing)

**Effort**: Low (~40 lines)

### 1.3 Add Get Model

**Status**: `ListModels` exists, `GetModel` does not.

**Work**:
- Add `GetModel(ctx, modelID) (Model, error)` method
- Endpoint: `GET models/{model}`
- Response: same `Model` struct already defined

**Effort**: Trivial (~15 lines)

### 1.4 Handle FileData in Responses

**Status**: `ProcessStream` raises error on `FileData` parts.

**Work**:
- Map `FileData` (URI + MIME type) to `genai.Reply.Doc` with URL reference
- Commonly returned for video generation results

**Effort**: Trivial (~10 lines)

## Phase 2: Important New Features

### 2.1 File Upload API

**Status**: Not implemented. Currently only URL references are supported for
large media.

**Work**:
- Add `Upload(ctx, reader, config) (FileRef, error)` method
- Endpoint: `POST https://generativelanguage.googleapis.com/upload/v1beta/files`
  (multipart upload with resumable support)
- Add `ListFiles`, `GetFile`, `DeleteFile` methods
- Endpoints:
  - `GET files?pageSize=100`
  - `GET files/{name}`
  - `DELETE files/{name}`
- File references use `fileData.fileUri` field (already in `Part.FileData`)
- Files expire after 48 hours by default

**Effort**: Medium (~150 lines)

### 2.2 Embeddings

**Status**: Not implemented. Different use case from generation.

**Work**:
- Add `Embed(ctx, content, config) ([]float32, error)` method
- Endpoint: `POST models/{model}:embedContent`
- Request: `{ "content": Content, "taskType": string, "title": string }`
- Response: `{ "embedding": { "values": []float32 } }`
- Implement `ProviderEmbed` interface
- Models: `text-embedding-004`, `text-embedding-005`

**Effort**: Medium (~80 lines)

### 2.3 File Search Tool

**Status**: Not implemented.

**Work**:
- Add `FileSearchTool` type to tool definitions
- Requires File Search Stores service (create store, upload documents, index)
- The tool is passed in `tools` alongside function declarations:
  ```json
  {"tools": [{"fileSearch": {"storeId": "..."}}]}
  ```
- Responses include `fileSearchResults` with document chunks and scores

**Effort**: High (~200 lines, includes store management)

### 2.4 URL Context Tool

**Status**: Not implemented. Added in Gemini 3.

**Work**:
- Add `URLContext` to `Tool` struct
- Enable via: `{"tools": [{"urlContext": {}}]}`
- Model fetches and processes web pages during generation
- Responses include URL context metadata

**Effort**: Low (~30 lines for the tool; response metadata TBD)

## Phase 3: Vertex AI Provider

See [vertex_ai.md](vertex_ai.md) for full investigation.

### 3.1 Create providers/vertexai Package

**Work**:
- New package `providers/vertexai/`
- Reuse all Gemini types via type aliases or shared internal package
- Different URL construction (resource path with project/location)
- OAuth2 bearer token auth via `golang.org/x/oauth2`
- Support ADC and service account authentication
- Environment variables: `GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_LOCATION`

**Effort**: Medium (~470 lines including tests)

### 3.2 Batch Prediction API (Vertex AI only)

**Status**: TODO in code at line 2239.

**Work**:
- Implement `ProviderBatch` interface
- Endpoint: `POST projects/{P}/locations/{L}/batchPredictionJobs`
- Input/output via GCS URIs or BigQuery tables
- Async job with polling

**Effort**: Medium (~120 lines)

## Phase 4: Advanced Features (Lower Priority)

### 4.1 Live API (Real-Time Streaming)

**Status**: Not implemented. Fundamentally different protocol (WebSocket).

**Work**:
- WebSocket client for `wss://generativelanguage.googleapis.com/ws/...`
- Bidirectional audio/video streaming
- Voice activity detection
- Session management with resumption
- Ephemeral token support for client-side auth
- Model: `gemini-2.5-flash-native-audio-preview`

**Effort**: Very High (~500+ lines, new paradigm)

**Recommendation**: Defer unless there is user demand. This is a fundamentally
different interaction pattern that may not fit the library's current
request/response architecture.

### 4.2 Interactions API / Deep Research

**Status**: Not implemented. Beta API.

**Work**:
- New endpoint for stateful agentic workflows
- Background execution mode with async polling
- Deep Research agent: `deep-research-pro-preview`
- Currently better documented for Python/JS SDKs

**Effort**: High (API surface still evolving)

**Recommendation**: Wait for GA. API is in beta and may change.

### 4.3 Model Tuning

**Status**: Not implemented.

**Work**:
- Create tuning job endpoint
- Monitor tuning progress
- Use tuned model for generation

**Effort**: Medium (~100 lines)

**Recommendation**: Niche use case. Implement only if requested.

### 4.4 Google Maps Grounding

**Status**: Not implemented. Available on Gemini 2.5 Flash.

**Work**:
- Add `GoogleMaps` tool type
- Response includes location-aware grounding metadata

**Effort**: Low (~20 lines)

## Phase 5: Additional SDK Parity Items

### 5.1 Image Editing / Upscaling / Segmentation

**Status**: Only basic image generation implemented.

**Work**:
- `EditImage`: Inpaint (remove/insert), outpaint, style transfer, background
  swap, product image editing. Requires reference images with masks.
- `UpscaleImage`: Increase resolution.
- `SegmentImage`: Foreground/background segmentation.
- Endpoints: `models/{model}:predict` with different request schemas per mode.

**Effort**: High (~200 lines, many editing modes)

### 5.2 Computer Use Tool (Preview)

**Status**: Not implemented. Preview feature.

**Work**:
- Add `ComputerUse` to `Tool` struct with `Environment` field
  (`EnvironmentBrowser`)
- Model returns screenshot analysis and UI actions (click, type, scroll)
- Requires response handling for action types

**Effort**: Medium (~80 lines)

**Recommendation**: Wait for GA.

### 5.3 GenerateContentConfig Additions

**Status**: Several config fields not yet mapped.

**Work**:
- `ThinkingLevel` (LOW/MEDIUM/HIGH/MINIMAL) - alternative to numeric budget
- `MediaResolution` (LOW/MEDIUM/HIGH) - input media processing quality
- `AudioTimestamp` - include timestamps in audio responses
- `ModelSelectionConfig` - dynamic model selection
- `RoutingConfig` - multi-model routing
- `StreamFunctionCallArguments` - stream tool call args incrementally
- `FunctionDeclaration.Behavior` - BLOCKING vs NON_BLOCKING

**Effort**: Low (~40 lines for all)

### 5.4 Enterprise Web Search

**Status**: Not implemented. Vertex AI feature.

**Work**:
- Add `EnterpriseWebSearch` tool type with `ExcludeDomains` and
  `BlockingConfidence` fields
- Enterprise-grade web search with VPC-SC support

**Effort**: Trivial (~15 lines)

## Implementation Order

Recommended implementation sequence based on impact/effort ratio:

```
Phase 1 (Quick Wins)     Phase 2 (Features)      Phase 3 (Vertex AI)
  1.1 Code Execution       2.1 File Upload         3.1 vertexai package
  1.2 CountTokens           2.4 URL Context         3.2 Batch Prediction
  1.3 Get Model             2.2 Embeddings
  1.4 FileData handling     2.3 File Search

Phase 4 (Advanced)       Phase 5 (SDK Parity)
  4.4 Maps Grounding       5.3 Config additions
  4.1 Live API             5.1 Image editing
  4.2 Interactions         5.4 Enterprise Web Search
  4.3 Model Tuning         5.2 Computer Use
```

Phases 1 and 2 can proceed independently. Phase 3 depends on Phase 1 being
complete (to ensure the shared types are stable). Phases 4 and 5 are independent
and can be done in any order based on demand. Phase 5.3 (config additions) is
low effort and can be done alongside any phase.

## Current Model Support

The provider currently supports these model families:

| Model | Status | Notes |
|-------|--------|-------|
| Gemini 3 Pro/Flash | Preview | Latest generation |
| Gemini 2.5 Pro | Stable | Best reasoning |
| Gemini 2.5 Flash | Stable | Best price-performance |
| Gemini 2.5 Flash-Lite | Stable | Fastest/cheapest |
| Gemini 2.0 Flash | Deprecated | Shutdown March 31, 2026 |
| Imagen 3 | Stable | Image generation |
| Veo 2 | Stable | Video generation |
| text-embedding-004/005 | Listed | Not usable (no embed API) |
