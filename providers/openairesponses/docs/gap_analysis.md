# OpenAI Responses API: Gap Analysis & Improvement Plan

## Current State Summary

The `providers/openairesponses` package implements the OpenAI Responses API
(`/v1/responses`) with support for synchronous generation, streaming, tool
calling, web search, reasoning summaries, image generation (via `/v1/images`),
video model selection, structured output (JSON schema), logprobs, and rate limit
reporting. It also defines batch request/response types but does not implement
the batch workflow.

### Implemented Interfaces

| Interface method | Status |
|---|---|
| `Provider.Name` | Done |
| `Provider.ModelID` | Done |
| `Provider.OutputModalities` | Done |
| `Provider.Capabilities` | Inherited from `base.NotImplemented` (returns empty) |
| `Provider.Scoreboard` | Done |
| `Provider.HTTPClient` | Done |
| `Provider.GenSync` | Done (text, image, video) |
| `Provider.GenStream` | Done (text; image/video simulated) |
| `Provider.ListModels` | Done |
| `Provider.GenAsync` | Done (via background mode) |
| `Provider.PokeResult` | Done (polls `GET /v1/responses/{id}`) |
| `Provider.CacheAddRequest` | Not implemented |
| `Provider.CacheList` | Not implemented |
| `Provider.CacheDelete` | Not implemented |

---

## Feature Comparison: Current Implementation vs Official OpenAI Go SDK / API

### 1. Tool Types

| Tool type | API support | Current impl | Notes |
|---|---|---|---|
| `function` | Yes | **Done** | Fully working with parallel tool calls |
| `web_search` | Yes | **Done** | Including source citations |
| `file_search` | Yes | **Missing** | Requires vector store IDs; struct fields exist but no processing |
| `code_interpreter` | Yes | **Missing** | Stream events defined but error on receipt |
| `computer_use_preview` | Yes | **Missing** | MessageType constant exists but errors on receipt |
| `image_generation` | Yes | **Partial** | Stream events defined but error on receipt; image gen uses separate `/v1/images` endpoint |
| `mcp` | Yes | **Missing** | Stream events defined but error on receipt |
| `local_shell` | Yes | **Missing** | MessageType constant exists but errors on receipt |

### 2. Input Content Types

| Input type | API support | Current impl | Notes |
|---|---|---|---|
| `input_text` | Yes | **Done** | |
| `input_image` | Yes | **Done** | Inline base64 and URL; detail level hardcoded to "auto" |
| `input_file` | Yes | **Done** | PDF and other binary files via base64 `file_data` |
| `input_audio` | Yes | **Missing** | Not handled in `FromRequest`/`FromReply` |

### 3. Output Content Types

| Output type | API support | Current impl | Notes |
|---|---|---|---|
| `output_text` | Yes | **Done** | Including annotations (url_citation) |
| `refusal` | Yes | **Done** | Surfaced as `FinishedContentFilter` with refusal text in `Reply.Text` |
| `output_audio` | No (not yet in Responses API) | N/A | |

### 4. Request Parameters

| Parameter | API support | Current impl | Notes |
|---|---|---|---|
| `model` | Yes | **Done** | |
| `input` | Yes | **Done** | |
| `instructions` | Yes | **Done** | Maps from `SystemPrompt` |
| `stream` | Yes | **Done** | |
| `background` | Yes | **Done** | Used by `GenAsync` |
| `max_output_tokens` | Yes | **Done** | |
| `max_tool_calls` | Yes | **Partial** | Field exists but not configurable via options |
| `metadata` | Yes | **Partial** | Field exists but not exposed |
| `parallel_tool_calls` | Yes | **Done** | Auto-set when tools present |
| `previous_response_id` | Yes | **Done** | Via provider-specific `GenOptionText` |
| `reasoning` | Yes | **Done** | Effort + summary = "auto" |
| `service_tier` | Yes | **Done** | Via provider-specific `GenOptionText` |
| `store` | Yes | **Partial** | Field exists but not configurable |
| `temperature` | Yes | **Done** | |
| `text.format` | Yes | **Done** | `text`, `json_schema`, `json_object` |
| `text.verbosity` | Yes | **Missing** | Field exists but never set |
| `tool_choice` | Yes | **Done** | `auto`, `required`, `none` |
| `top_p` | Yes | **Done** | |
| `top_logprobs` | Yes | **Done** | |
| `truncation` | Yes | **Done** | Via provider-specific `GenOptionText` |
| `user` | Yes | **Partial** | Field exists but not configurable |
| `include` | Yes | **Partial** | Only for web search sources |
| `prompt_cache_key` | Yes | **Missing** | Stub struct only |
| `prompt_cache_retention` | Yes | **Missing** | Stub struct only |
| `safety_identifier` | Yes | **Missing** | Stub struct only |

### 5. Streaming Events

| Event | API support | Current impl | Notes |
|---|---|---|---|
| `response.created` | Yes | **Done** | Ignored (no useful data) |
| `response.in_progress` | Yes | **Done** | Ignored |
| `response.completed` | Yes | **Done** | Extracts usage + finish reason |
| `response.failed` | Yes | **Done** | |
| `response.incomplete` | Yes | **Done** | |
| `response.queued` | Yes | **Done** | Ignored |
| `response.output_item.added` | Yes | **Done** | |
| `response.output_item.done` | Yes | **Done** | |
| `response.content_part.added` | Yes | **Done** | |
| `response.content_part.done` | Yes | **Done** | |
| `response.output_text.delta` | Yes | **Done** | |
| `response.output_text.done` | Yes | **Done** | |
| `response.output_text.annotation.added` | Yes | **Done** | url_citation only |
| `response.refusal.delta` | Yes | **Done** | Surfaced as content filter |
| `response.refusal.done` | Yes | **Done** | Surfaced as content filter |
| `response.function_call_arguments.delta` | Yes | **Done** | |
| `response.function_call_arguments.done` | Yes | **Done** | |
| `response.reasoning_summary_part.*` | Yes | **Done** | |
| `response.reasoning_summary_text.*` | Yes | **Done** | |
| `response.reasoning_text.*` | Yes | **Done** | |
| `response.web_search_call.*` | Yes | **Done** | |
| `response.file_search_call.*` | Yes | **Missing** | Errors on receipt |
| `response.image_generation_call.*` | Yes | **Missing** | Errors on receipt |
| `response.mcp_call.*` | Yes | **Missing** | Errors on receipt |
| `response.code_interpreter_call.*` | Yes | **Missing** | Errors on receipt |
| `response.custom_tool_call_input.*` | Yes | **Missing** | Errors on receipt |

### 6. Async / Batch Operations

| Feature | API support | Current impl | Notes |
|---|---|---|---|
| `background` mode | Yes | **Done** | Used by `GenAsync` |
| Batch API (`/v1/batch`) | Yes | **Missing** | Types defined (`BatchRequest`, `Batch`, etc.) but no methods |
| `GenAsync` / `PokeResult` | N/A (genai interface) | **Done** | Mapped to background mode |

### 7. Context Management

| Feature | API support | Current impl | Notes |
|---|---|---|---|
| `truncation` strategy | Yes | **Done** | Via provider-specific `GenOptionText` |
| `/v1/responses/compact` | Yes | **Missing** | Compaction endpoint not implemented |

### 8. Annotation Types

| Annotation type | API support | Current impl | Notes |
|---|---|---|---|
| `url_citation` | Yes | **Done** | |
| `file_citation` | Yes | **Missing** | Field exists in struct |
| `container_file_citation` | Yes | **Missing** | Field exists in struct |
| `file_path` | Yes | **Missing** | Field exists in struct |

### 9. Conversations API

| Feature | API support | Current impl | Notes |
|---|---|---|---|
| `conversation` parameter | Yes | **Missing** | Thread-like state management, replacement for Assistants API |
| `previous_response_id` | Yes | **Done** | Via provider-specific `GenOptionText` |

### 10. Other Missing Features

| Feature | Description | Priority |
|---|---|---|
| Audio input | `input_audio` content type | Low (API support limited) |
| ~~`previous_response_id`~~ | ~~Stateful multi-turn via server-side state~~ | **Done** |
| `conversation` parameter | Thread-based state management | Medium |
| Image detail configurability | Hardcoded to "auto"; should be option | Low |
| Stop sequences | Not supported in Responses API (only Chat Completions) | N/A |
| `text.verbosity` | Controls output verbosity ("low"/"medium"/"high") | Low |
| ~~Refusal handling~~ | ~~Currently fatal; should surface as content~~ | **Done** |
| Multiple output messages | Only first output processed fully | Low |
| `include` options | Only `web_search_call.action.sources`; missing `file_search_call.results`, `reasoning.encrypted_content`, `computer_call_output.output.image_url` | Low |
| Reasoning effort `none`, `minimal`, `xhigh` | Only `low`/`medium`/`high` defined | Low |
| Stream obfuscation | `include_obfuscation` / `starting_after` | Low |
| `presence_penalty` | -2.0 to 2.0, discourage repetition | Low |
| `prompt_cache_retention` | "24h" for extended caching | Medium |

---

## Improvement Plan

### Phase 1: High-Impact, Low-Effort

These items improve existing features with minimal structural changes.

1. ~~**Refusal handling in streaming**~~: **Done** — surfaced as `FinishedContentFilter` with refusal text in `Reply.Text`.

2. ~~**Expose `truncation` option**~~: **Done** — wired via `GenOptionText.Truncation`.

3. ~~**Expose `previous_response_id`**~~: **Done** — wired via `GenOptionText.PreviousResponseID`.

4. **Image detail configurability**: Add a field to `GenOptionText` or the
   content builder so callers can choose `"low"`, `"high"`, or `"auto"` for
   image inputs.

5. **Add missing `ReasoningEffort` values**: Add `none`, `minimal`, and `xhigh`
   constants.

### Phase 2: New Feature Coverage

6. ~~**Implement `GenAsync` / `PokeResult` via background mode**~~: **Done** — `GenAsync` sends with `background: true`, `PokeResult` polls `GET /v1/responses/{id}`.

7. **Implement file_search tool processing**: The struct fields for `file_search`
   results already exist. Wire up `Message.To()` to convert file search results
   into `genai.Reply` with citations or document references.

8. **Implement image_generation tool processing**: Handle
   `response.image_generation_call.*` streaming events. When
   `image_generation_call.completed`, extract the generated image and return as
   `genai.Reply{Doc: ...}`.

9. **Implement code_interpreter tool processing**: Handle
   `response.code_interpreter_call.*` events. Surface code execution results
   (stdout, images) as replies.

10. **Handle multiple output messages**: The current `ToResult()` iterates
    `r.Output` but the streaming path has a TODO about "multiple messages". Ensure
    all output items are processed correctly.

### Phase 3: Advanced Features

11. **Batch API implementation**: Wire up the existing `BatchRequest`, `Batch`,
    `BatchRequestInput`, `BatchRequestOutput` types:
    - Upload JSONL input file via `/v1/files`
    - Create batch via `POST /v1/batches`
    - Poll batch status via `GET /v1/batches/{id}`
    - Download results via `/v1/files/{output_file_id}/content`
    - This could also back `GenAsync` for high-throughput scenarios.

12. **Context compaction**: Implement `POST /v1/responses/compact` for long
    conversations. This could be exposed as a new method on the client or
    integrated into the message processing pipeline.

13. **MCP tool support**: Handle `mcp_call`, `mcp_list_tools`,
    `mcp_approval_request` streaming events and the corresponding input types.
    This enables server-side MCP tool invocation.

14. **Computer use tool**: Handle `computer_call` events and
    `computer_call_output` inputs. This is relevant for agent workflows.

15. **Prompt caching controls**: Implement `prompt_cache_key` and
    `prompt_cache_retention` parameters to give callers control over OpenAI's
    prompt caching behavior.

### Phase 4: Polish

16. **Stop sequences**: OpenAI Responses API may not support stop sequences
    natively (Chat Completions does). Verify and either implement or document the
    limitation clearly.

17. **Additional annotation types**: Process `file_citation`,
    `container_file_citation`, and `file_path` annotations when they appear in
    output text.

18. **`text.verbosity`**: Expose the verbosity parameter for controlling output
    length.

19. **`store` parameter**: Expose as an option so callers can opt in/out of
    response storage.

20. **`include` parameter extensibility**: Allow callers to specify additional
    include fields beyond `web_search_call.action.sources`.

---

## Priority Matrix

| Priority | Items | Rationale |
|---|---|---|
| **P0 - Critical** | ~~#1 (refusal)~~, ~~#6 (GenAsync)~~ | **All done** |
| **P1 - High** | ~~#2 (truncation)~~, ~~#3 (previous_response_id)~~, #7 (file_search), #10 (multi-output) | #2, #3 done; #7, #10 remain |
| **P2 - Medium** | #4 (image detail), #5 (reasoning effort), #8 (image_generation), #9 (code_interpreter), #11 (batch) | Extend capabilities to more tool types |
| **P3 - Low** | #12-#20 | Advanced/niche features |
