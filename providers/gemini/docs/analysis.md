# Gemini Provider: Current State Analysis

## Overview

The `providers/gemini` package implements a client for Google's Gemini API
(`generativelanguage.googleapis.com`). It uses `base.Provider` generics for HTTP
communication, streaming, and response processing. The package is ~2,700 lines
of Go across `client.go` (request/response types, API methods) and `schema.go`
(Go struct to JSON Schema reflection).

## Implemented Features

### Text Generation
- Synchronous via `generateContent` (GenSync)
- Streaming via `streamGenerateContent?alt=sse` (GenStream)
- System instructions
- Temperature, TopP, TopK, frequency/presence penalties
- Stop sequences, max output tokens, seed
- Token usage reporting (input, output, cached, reasoning, tool-use)
- Finish reason detection and mapping

### Tool Calling
- Go struct to JSON Schema via reflection (`schema.go`)
- Function declarations auto-generated from Go types
- Tool modes: AUTO, ANY (required), VALIDATED, NONE
- Tool call parsing and response handling in streaming
- Google Search integration (`GoogleSearch` tool)
- `ToolDef.InputSchemaOverride` NOT yet supported

### JSON Output
- JSON mode via `responseMimeType: "application/json"`
- JSON Schema mode with `responseSchema` for structured responses

### Multi-Modal Input
- Text, images (JPEG, PNG, WebP), audio (AAC, FLAC, WAV)
- Video (MP4, WebM), documents/PDFs (max 10MB)
- Inline base64 and URL-referenced media

### Multi-Modal Output
- Image generation via Imagen models (`predict` endpoint)
- Video generation via Veo models (`predictLongRunning` endpoint)
- Audio/TTS generation
- Thinking/reasoning output

### Thinking / Extended Reasoning
- Dynamic thinking (model decides when to think)
- Configurable budget: 0 (disabled), -1 (dynamic), 128-32768 (fixed)
- Provider-specific `GenOptions.ThinkingBudget`

### Prompt Caching
- Create, list, extend TTL, delete cached content
- Reference in requests via `CachedContent` field
- Minimum 4096 tokens, cost 25% of regular token rate

### Async Operations (GenAsync)
- Video generation via `predictLongRunning`
- Poll job status via `PokeResult`
- 48-hour result retention

### Other
- Log probabilities (`topLogprobs`)
- Grounding metadata with citations and confidence scores
- Safety settings and ratings
- Model listing and auto-selection

## API Endpoints Used

Base URL: `https://generativelanguage.googleapis.com/v1beta/`

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `models/{model}:generateContent` | POST | Sync text generation |
| `models/{model}:streamGenerateContent?alt=sse` | POST | Streaming |
| `models/{model}:predict` | POST | Image generation (Imagen) |
| `models/{model}:predictLongRunning` | POST | Video generation (Veo) |
| `{jobId}` | GET | Poll async job |
| `models?pageSize=1000` | GET | List models |
| `cachedContents` | POST | Create cache |
| `cachedContents/{name}` | GET/PATCH/DELETE | Manage cache |
| `cachedContents?pageSize=100` | GET | List caches |

## Authentication

API key via `x-goog-api-key` header. Environment variable: `GEMINI_API_KEY`.

## genai Interfaces Implemented

- `Provider` (full interface via `var _ genai.Provider = &Client{}`)
- `GenSync`, `GenStream`
- `ListModels`
- `GenAsync`, `PokeResult` (video models only)
- `CacheAddRequest`, `CacheList`, `CacheDelete`
- `Capabilities` (reports GenAsync for video, Caching always)

## Known Limitations and TODOs in Code

1. `ToolDef.InputSchemaOverride` not implemented (line comment)
2. `GenOptionsAudio`, `GenOptionsImage`, `GenOptionsVideo` partially implemented
3. `ExecutableCode` and `CodeExecutionResult` parts trigger errors ("implement ...")
4. `FileData` parts in responses trigger errors
5. Logprobs in streaming unsupported by Gemini
6. No File Upload API (uses URLs only)
7. Batch prediction requires Vertex AI (TODO at line 2239)
8. VertexAI-only fields commented out in `ImageParameters`

## Architecture

```
Client
  +-- base.Provider[*ErrorResponse, *ChatRequest, *ChatResponse, ChatStreamChunkResponse]
  +-- base.NotImplemented (default stubs for unimplemented interfaces)
```

Request flow:
1. `genai.Message` converted to `ChatRequest.Contents` via `Content.From()`
2. Options applied via `ChatRequest.Init()`
3. Tools generated from Go struct reflection
4. JSON serialized, gzip compressed
5. Response parsed, candidates extracted, parts converted back via `Content.To()`
6. Streaming: SSE chunks accumulated via `ProcessStream()`
