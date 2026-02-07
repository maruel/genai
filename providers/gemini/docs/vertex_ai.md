# Vertex AI API Investigation

## Overview

Vertex AI is Google Cloud's enterprise AI platform. It provides the same Gemini
models as the Gemini Developer API but with enterprise features, different
authentication, and a different endpoint structure. The request/response JSON
schemas are **nearly identical** between Gemini API and Vertex AI.

## Key Differences from Gemini API

### Endpoints

| Aspect | Gemini API | Vertex AI |
|--------|-----------|-----------|
| Host | `generativelanguage.googleapis.com` | `{LOCATION}-aiplatform.googleapis.com` |
| API version | `/v1beta/` | `/v1/` or `/v1beta1/` |
| Model path | `models/{MODEL}` | `projects/{P}/locations/{L}/publishers/google/models/{MODEL}` |
| Global endpoint | N/A | `aiplatform.googleapis.com` (auto-routes) |

Concrete URL comparison:

```
# Gemini API
POST https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent

# Vertex AI (regional)
POST https://us-central1-aiplatform.googleapis.com/v1/projects/my-proj/locations/us-central1/publishers/google/models/gemini-2.5-flash:generateContent

# Vertex AI (global)
POST https://aiplatform.googleapis.com/v1/projects/my-proj/locations/global/publishers/google/models/gemini-2.5-flash:generateContent
```

Full endpoint mapping:

| Operation | Gemini API | Vertex AI |
|-----------|-----------|-----------|
| generateContent | `models/{M}:generateContent` | `projects/{P}/locations/{L}/publishers/google/models/{M}:generateContent` |
| streamGenerateContent | `models/{M}:streamGenerateContent?alt=sse` | Same pattern + `?alt=sse` |
| predict (image) | `models/{M}:predict` | Same pattern |
| predictLongRunning | `models/{M}:predictLongRunning` | Same pattern |
| countTokens | `models/{M}:countTokens` | Same pattern |
| embedContent | `models/{M}:embedContent` | Same pattern |
| listModels | `models?pageSize=1000` | `projects/{P}/locations/{L}/publishers/google/models` |
| cachedContents | `cachedContents` | `projects/{P}/locations/{L}/cachedContents` |
| batchPredictionJobs | N/A | `projects/{P}/locations/{L}/batchPredictionJobs` |

### Authentication

| Gemini API | Vertex AI |
|-----------|-----------|
| `x-goog-api-key: {API_KEY}` | `Authorization: Bearer {ACCESS_TOKEN}` |

Vertex AI supports:

1. **Application Default Credentials (ADC)**: Recommended. Auto-discovered in
   GCP environments. For local dev: `gcloud auth application-default login`.
2. **Service Account (2-Legged OAuth)**: For production server-to-server.
   Download JSON key file, sign JWT, exchange for access token.
3. **OAuth2 Bearer Token**: Short-lived (1 hour). For testing:
   `gcloud auth print-access-token`.

Required OAuth2 scope: `https://www.googleapis.com/auth/cloud-platform`

Required IAM role: `roles/aiplatform.user` (Vertex AI User)

### Request/Response Format

The JSON request/response schemas are **identical** with a few Vertex
AI-specific additions:

- **`labels`** (map[string]string): Billing attribution. Up to 64 per request.
- **`modelArmorConfig`**: Prompt/response sanitization. Mutually exclusive with
  `safetySettings`.

All core fields (`contents`, `generationConfig`, `tools`, `toolConfig`,
`safetySettings`, `systemInstruction`, `cachedContent`) are shared.

### Data Privacy

- Gemini API free tier: prompts may improve Google products
- Vertex AI: prompts/responses **never** used to improve products (both tiers)

## Vertex AI-Exclusive Features

1. **Batch Prediction API**: Server-side batch processing via
   `batchPredictionJobs` endpoint. Processes large volumes asynchronously with
   results written to GCS.

2. **Model Garden**: Access to 100+ third-party models (Anthropic Claude, Meta
   Llama, Mistral, etc.) through the same API surface.

3. **Provisioned Throughput**: Pre-purchase dedicated capacity (GSUs) for
   guaranteed throughput. Dynamic window enforcement for spike handling.

4. **Model Armor**: Advanced prompt/response sanitization service.

5. **Enterprise Grounding**: Ground responses in private enterprise data via
   Vertex AI Search.

6. **VPC Service Controls**: Network-level isolation for data residency.

7. **CMEK**: Customer-managed encryption keys.

8. **Custom Labels**: Per-request labels for cost tracking and billing
   attribution.

9. **Audit Logging**: Cloud Audit Logs integration.

10. **Model Tuning**: Full fine-tuning and adapter-based tuning.

11. **OpenAI-Compatible Endpoint**: Drop-in replacement at
    `projects/{P}/locations/{L}/endpoints/openapi/chat/completions`.

12. **Extended Video Generation**: Additional parameters (FPS, resolution, seed,
    Pub/Sub notifications, audio generation, last frame, compression).

## Implementation Strategy

### Recommended Approach: Vertex AI as a Thin Wrapper

The Gemini and Vertex AI APIs share ~95% of their request/response types. The
only differences are:

1. URL construction (host + resource path prefix)
2. Authentication header mechanism
3. Token refresh lifecycle
4. A few additional optional request fields

This means a Vertex AI provider can reuse nearly all Gemini provider code.

### Proposed Package Structure

```
providers/vertexai/
    client.go          # New(), Client, URL builder, auth transport
    client_test.go     # Unit tests
    testdata/          # Recorded HTTP interactions
    scoreboard.json    # Feature matrix
```

### Core Components

#### 1. URL Builder (~30 lines)

```go
type urlBuilder struct {
    host     string // e.g. "us-central1-aiplatform.googleapis.com"
    project  string
    location string
}

func (u *urlBuilder) model(model, method string) string {
    return fmt.Sprintf("https://%s/v1/projects/%s/locations/%s/publishers/google/models/%s:%s",
        u.host, url.PathEscape(u.project), url.PathEscape(u.location),
        url.PathEscape(model), method)
}

func (u *urlBuilder) cachedContents() string {
    return fmt.Sprintf("https://%s/v1/projects/%s/locations/%s/cachedContents",
        u.host, url.PathEscape(u.project), url.PathEscape(u.location))
}

func (u *urlBuilder) listModels() string {
    return fmt.Sprintf("https://%s/v1/projects/%s/locations/%s/publishers/google/models",
        u.host, url.PathEscape(u.project), url.PathEscape(u.location))
}
```

#### 2. Auth Transport (~20 lines)

```go
type tokenTransport struct {
    source oauth2.TokenSource
    base   http.RoundTripper
}

func (t *tokenTransport) RoundTrip(req *http.Request) (*http.Response, error) {
    token, err := t.source.Token()
    if err != nil {
        return nil, fmt.Errorf("failed to get access token: %w", err)
    }
    r := req.Clone(req.Context())
    r.Header.Set("Authorization", "Bearer "+token.AccessToken)
    return t.base.RoundTrip(r)
}
```

#### 3. Type Reuse Strategy

All Gemini types can be reused via type aliases:

```go
import "github.com/maruel/genai/providers/gemini"

type ChatRequest = gemini.ChatRequest
type ChatResponse = gemini.ChatResponse
type ChatStreamChunkResponse = gemini.ChatStreamChunkResponse
```

`ProcessStream()` from the gemini package is directly reusable.

For Vertex AI-specific fields, extend the request:

```go
type VertexChatRequest struct {
    gemini.ChatRequest
    Labels           map[string]string `json:"labels,omitzero"`
    ModelArmorConfig *ModelArmorConfig  `json:"modelArmorConfig,omitzero"`
}
```

#### 4. Provider Options

```go
// Environment variables:
//   GOOGLE_CLOUD_PROJECT, GOOGLE_CLOUD_LOCATION
//   GOOGLE_APPLICATION_CREDENTIALS (for service account)
//   GOOGLE_GENAI_USE_VERTEXAI=true (auto-detect)

type VertexAIConfig struct {
    Project  string // Required. GCP project ID.
    Location string // Defaults to "us-central1".
}
```

#### 5. Dependency Impact

Only one new dependency needed: `golang.org/x/oauth2` for automatic token
refresh. For ADC: `golang.org/x/oauth2/google.DefaultTokenSource()`.

Alternative zero-dependency approach: implement JWT signing via `crypto/rsa` +
`encoding/json` directly, but `golang.org/x/oauth2` is lightweight and from the
Go team.

### Estimated Effort

| Component | Lines | Effort |
|-----------|-------|--------|
| URL construction | ~30 | Trivial |
| OAuth2 transport | ~20 | Trivial |
| `New()` constructor | ~80 | Low |
| Name/ListModels/Cache with new URLs | ~100 | Low |
| Vertex AI options and extended fields | ~40 | Trivial |
| Tests | ~200 | Medium |
| **Total** | **~470** | **Low-Medium** |

### Alternative: Extract Shared Types

Instead of type aliases, extract shared types into `internal/geminicommon/`:

```
internal/geminicommon/
    types.go     # ChatRequest, ChatResponse, Content, Part, etc.
    schema.go    # Schema type and reflection
    stream.go    # ProcessStream
```

Both `providers/gemini/` and `providers/vertexai/` would import from this
internal package. Cleaner but requires a larger refactor. Recommended for a
future iteration once both providers are stable.

## Sources

- [Google Gen AI Go SDK](https://github.com/googleapis/go-genai)
- [google.golang.org/genai](https://pkg.go.dev/google.golang.org/genai)
- [Vertex AI SDK migration](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/deprecations/genai-vertexai-sdk)
- [Migrate from Gemini API to Vertex AI](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/migrate/migrate-google-ai)
- [Vertex AI generateContent REST](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/reference/rest/v1/projects.locations.publishers.models/generateContent)
- [Vertex AI Authentication](https://docs.cloud.google.com/vertex-ai/docs/authentication)
- [Service Account Auth in Go](https://pgaleone.eu/cloud/2025/06/29/vertex-ai-to-genai-sdk-service-account-auth-python-go/)
- [Vertex AI Pricing](https://cloud.google.com/vertex-ai/generative-ai/pricing)
- [Vertex AI Locations](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/learn/locations)
- [Vertex AI Quotas](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/quotas)
- [cloud.google.com/go/vertexai/genai (deprecated)](https://pkg.go.dev/cloud.google.com/go/vertexai/genai)
