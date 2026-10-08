# Gemini Provider: Improvement Plan

## Vertex AI

See [vertex_ai.md](vertex_ai.md) for the API investigation.

### Provider Package

- Add `providers/vertexai/` with project/location resource paths.
- Share native DTOs through a base package.
- Support OAuth2 bearer tokens, ADC and service account authentication.
- Read `GOOGLE_CLOUD_PROJECT` and `GOOGLE_CLOUD_LOCATION`.

### Batch Prediction

- Implement `ProviderBatch` using `batchPredictionJobs`.
- Support input/output through GCS URIs or BigQuery tables.
- Poll asynchronous jobs.

## Advanced Features

### Live API

- Add a WebSocket client with bidirectional audio/video streaming.
- Handle voice activity detection, session resumption and ephemeral tokens.
- Decide how sessions fit the library's request/response interfaces.

### Interactions and Deep Research

- Support stateful agent workflows and background execution with polling.
- Define how agent results fit the generic provider interface.

### Model Tuning

- Create and monitor tuning jobs, then use tuned models for generation.

### Google Maps Grounding

- Add the Maps tool and map location-aware grounding metadata into responses.

## SDK Parity

### Image Editing, Upscaling and Segmentation

- Add native image editing requests for masks, reference images and editing modes.
- Support upscaling and foreground/background segmentation.

### Computer Use

- Add the computer-use tool and its environment configuration.
- Handle screenshot analysis and returned UI actions.

### Generation Controls

- Add audio timestamps, model selection and routing configuration.
- Support incremental function-call arguments and declaration behavior.

### Enterprise Web Search

- Add the Vertex AI search tool with domain exclusions and blocking confidence.
