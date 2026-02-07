# Official Google Go SDK vs providers/gemini: Gap Analysis

## Official SDK Overview

The official Go SDK at `google.golang.org/genai` (GitHub:
[googleapis/go-genai](https://github.com/googleapis/go-genai)) is Google's
unified SDK supporting both Gemini API and Vertex AI via a single `Client` type
with a `Backend` selector.

> The older `cloud.google.com/go/vertexai/genai` package was deprecated June 24,
> 2025 and will be removed June 24, 2026.

### Official SDK Client Structure

```go
type Client struct {
    Models            Models
    Chats             Chats
    Files             Files
    Batches           Batches
    Caches            Caches
    Tunings           Tunings
    Live              Live
    Operations        Operations
    FileSearchStores  FileSearchStores
    Documents         Documents
    Tokens            Tokens
}
```

## Feature Comparison Matrix

| Feature | Official SDK | providers/gemini | Gap |
|---------|:-----------:|:----------------:|:---:|
| **Text Generation** | | | |
| GenerateContent (sync) | Yes | Yes | - |
| GenerateContentStream | Yes | Yes | - |
| System instructions | Yes | Yes | - |
| Temperature/TopP/TopK | Yes | Yes | - |
| Stop sequences | Yes | Yes | - |
| Max output tokens | Yes | Yes | - |
| Seed | Yes | Yes | - |
| Frequency/presence penalty | Yes | Yes | - |
| Candidate count | Yes | No | **Missing** |
| Response logprobs | Yes | Partial | Sync only |
| **Structured Output** | | | |
| JSON mode (responseMimeType) | Yes | Yes | - |
| JSON Schema (responseSchema) | Yes | Yes | - |
| Enum mode | Yes | Unclear | **Verify** |
| **Tool Calling** | | | |
| Function declarations | Yes | Yes | - |
| AUTO/ANY/NONE modes | Yes | Yes | - |
| Google Search grounding | Yes | Yes | - |
| Code execution tool | Yes | Yes | - |
| File search tool | Yes | No | **Missing** |
| URL context tool | Yes | No | **Missing** |
| **Multi-Modal Input** | | | |
| Text | Yes | Yes | - |
| Images (inline/URL) | Yes | Yes | - |
| Audio (inline/URL) | Yes | Yes | - |
| Video (inline/URL) | Yes | Yes | - |
| PDF/Documents | Yes | Yes | - |
| **Multi-Modal Output** | | | |
| Text | Yes | Yes | - |
| Image generation (Imagen) | Yes | Yes | - |
| Video generation (Veo) | Yes | Yes | - |
| Audio/TTS | Yes | Yes | - |
| Native audio (Live API) | Yes | No | **Missing** |
| **Thinking** | | | |
| Thinking config | Yes | Yes | - |
| Thinking budget | Yes | Yes | - |
| Dynamic thinking | Yes | Yes | - |
| **Caching** | | | |
| Create cached content | Yes | Yes | - |
| List cached contents | Yes | Yes | - |
| Delete cached content | Yes | Yes | - |
| Update/extend TTL | Yes | Yes | - |
| **File Management** | | | |
| Upload files | Yes | No | **Missing** |
| List files | Yes | No | **Missing** |
| Get file metadata | Yes | No | **Missing** |
| Delete files | Yes | No | **Missing** |
| **Token Counting** | | | |
| CountTokens | Yes | Yes | - |
| ComputeTokens | Yes | No | **Missing** |
| **Embeddings** | | | |
| EmbedContent | Yes | No | **Missing** |
| **Batch Operations** | | | |
| Create batch job | Yes | No | **Missing** |
| List batch jobs | Yes | No | **Missing** |
| Get batch job | Yes | No | **Missing** |
| **Chat (stateful)** | | | |
| Chat sessions | Yes | N/A | By design |
| **Live API (real-time)** | | | |
| WebSocket connection | Yes | No | **Missing** |
| Bidirectional audio | Yes | No | **Missing** |
| Voice activity detection | Yes | No | **Missing** |
| **Model Tuning** | | | |
| Fine-tuning | Yes | No | **Missing** |
| List tuning jobs | Yes | No | **Missing** |
| **Model Management** | | | |
| List models | Yes | Yes | - |
| Get model | Yes | Yes | - |
| **Auth** | | | |
| API key | Yes | Yes | - |
| OAuth2/ADC | Yes | No | **Missing** |
| Ephemeral tokens | Yes | No | **Missing** |
| Service accounts | Yes | No | **Missing** |
| **Vertex AI Backend** | | | |
| Backend selector | Yes | No | **Missing** |
| Project/Location config | Yes | No | **Missing** |
| **Advanced Features** | | | |
| File search stores | Yes | No | **Missing** |
| Documents service | Yes | No | **Missing** |
| Interactions API | Beta | No | **Missing** |
| Deep Research agent | Beta | No | **Missing** |
| Safety settings | Yes | Yes | - |
| Grounding metadata | Yes | Yes | - |
| Citation metadata | Yes | Yes | - |
| Google Maps grounding | Yes | No | **Missing** |
| Computer Use tool | Preview | No | **Missing** |
| Enterprise Web Search | Yes | No | **Missing** |
| **Image Operations** | | | |
| Image generation | Yes | Yes | - |
| Image editing (inpaint/outpaint) | Yes | No | **Missing** |
| Image upscaling | Yes | No | **Missing** |
| Image segmentation | Yes | No | **Missing** |
| **Generation Config** | | | |
| SpeechConfig (TTS voice) | Yes | Partial | **Verify** |
| MediaResolution | Yes | No | **Missing** |
| AudioTimestamp | Yes | No | **Missing** |
| ResponseModalities | Yes | Yes | - |
| ThinkingLevel (LOW/MED/HIGH) | Yes | No | **Missing** |
| RoutingConfig | Yes | No | **Missing** |
| ModelSelectionConfig | Yes | No | **Missing** |
| **Tool Calling Advanced** | | | |
| VALIDATED mode | Yes | Yes | - |
| StreamFunctionCallArguments | Yes | No | **Missing** |
| FunctionDeclaration.Behavior | Yes | No | **Missing** |
| Allowed function names | Yes | Yes | - |

## Priority Gaps (High Impact)

### P0 - Core Missing Features

1. **File Upload API** - Required for large media that exceeds inline limits.
   The API supports uploading files and referencing them by URI, avoiding base64
   encoding overhead for large files.

2. ~~**CountTokens**~~ - Done. `Client.CountTokens()` implemented.

3. ~~**Code Execution Tool**~~ - Done. Full request/response handling implemented.

4. **Vertex AI Backend** - Enables enterprise features, different auth, and
   access to the batch prediction API. See `vertex_ai.md` for full analysis.

### P1 - Important Missing Features

5. **Candidate Count** - `GenerateContentConfig.CandidateCount` allows
   requesting multiple response candidates. Low effort to add.

6. ~~**Get Model**~~ - Done. `Client.GetModel()` implemented.

7. **Embeddings** - `models/{model}:embedContent` endpoint. Different use case
   from text generation but commonly needed.

8. **File Search Tool** - Server-side RAG. Google hosts the vector store and
   performs retrieval. Requires `FileSearchStores` service support.

9. **URL Context Tool** - Allows the model to fetch and process web pages during
   generation. Added in Gemini 3.

### P2 - Nice to Have

10. **Batch Operations** - Server-side batch processing. Requires Vertex AI for
    the batch prediction API.

11. **Live API** - Real-time bidirectional WebSocket streaming for voice/video.
    Requires WebSocket client, fundamentally different from HTTP request/response.

12. **Model Tuning** - Fine-tuning support. Niche use case for the library's
    scope.

13. **Interactions API** - Unified interface for agentic workflows and Deep
    Research. Currently in beta, primarily documented for Python/JS SDKs.

14. **Google Maps Grounding** - Location-aware grounding. Specialized use case.

## Non-Gaps (Intentional Differences)

- **Chat sessions**: The `genai` library is stateless by design. Chat state is
  managed by the caller via message history. This is not a gap.

- **Ephemeral tokens**: Designed for client-side (browser/mobile) use. Not
  relevant for a server-side Go library.

## Sources

- [Official Go SDK - pkg.go.dev](https://pkg.go.dev/google.golang.org/genai)
- [Official Go SDK - GitHub](https://github.com/googleapis/go-genai)
- [Gemini API Models](https://ai.google.dev/gemini-api/docs/models)
- [Gemini API Structured Output](https://ai.google.dev/gemini-api/docs/structured-output)
- [Gemini API Live](https://ai.google.dev/gemini-api/docs/live)
- [Gemini API Deep Research](https://ai.google.dev/gemini-api/docs/deep-research)
- [Gemini API Interactions](https://ai.google.dev/gemini-api/docs/interactions)
