// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Wire types for the OpenAI Responses and shared OpenAI API endpoints.
//
// These types map directly to the JSON request/response structures documented at:
//   - https://platform.openai.com/docs/api-reference/responses
//   - https://platform.openai.com/docs/api-reference/models
//   - https://platform.openai.com/docs/api-reference/images
//   - https://platform.openai.com/docs/api-reference/files
//   - https://platform.openai.com/docs/api-reference/batch

package openairesponses

import (
	"github.com/invopop/jsonschema"
	"github.com/maruel/genai/base"
)

//
// Shared types (used by both OpenAI chat and responses APIs).
//

// ServiceTier is the quality of service to determine the request's priority.
type ServiceTier string

const (
	// ServiceTierAuto will utilize scale tier credits until they are exhausted if the Project is Scale tier
	// enabled, else the request will be processed using the default service tier with a lower uptime SLA and no
	// latency guarantee.
	//
	// https://openai.com/api-scale-tier/
	ServiceTierAuto ServiceTier = "auto"
	// ServiceTierDefault has the request be processed using the default service tier with a lower uptime SLA
	// and no latency guarantee.
	ServiceTierDefault ServiceTier = "default"
	// ServiceTierFlex has the request be processed with the Flex Processing service tier.
	//
	// Flex processing is in beta, and currently only available for GPT-5, o3 and o4-mini models.
	//
	// https://platform.openai.com/docs/guides/flex-processing
	ServiceTierFlex ServiceTier = "flex"
)

// ReasoningEffort is the effort the model should put into reasoning. Default is Medium.
//
// https://platform.openai.com/docs/api-reference/assistants/createAssistant#assistants-createassistant-reasoning_effort
// https://platform.openai.com/docs/guides/reasoning
type ReasoningEffort string

// Reasoning effort values.
const (
	ReasoningEffortNone    ReasoningEffort = "none"
	ReasoningEffortMinimal ReasoningEffort = "minimal"
	ReasoningEffortLow     ReasoningEffort = "low"
	ReasoningEffortMedium  ReasoningEffort = "medium"
	ReasoningEffortHigh    ReasoningEffort = "high"
	ReasoningEffortXHigh   ReasoningEffort = "xhigh"
)

// ImageRequest is documented at https://platform.openai.com/docs/api-reference/images
type ImageRequest struct {
	Prompt            string     `json:"prompt"`
	Model             string     `json:"model,omitzero"`              // Default to dall-e-2, unless a gpt-image-1 specific parameter is used.
	Background        Background `json:"background,omitzero"`         // Default "auto"
	Moderation        string     `json:"moderation,omitzero"`         // gpt-image-1: "low" or "auto"
	N                 int64      `json:"n,omitzero"`                  // Number of images to return
	OutputCompression float64    `json:"output_compression,omitzero"` // Defaults to 100. Only supported on gpt-image-1 with webp or jpeg
	OutputFormat      string     `json:"output_format,omitzero"`      // "png", "jpeg" or "webp". Defaults to png. Only supported on gpt-image-1.
	Quality           string     `json:"quality,omitzero"`            // "auto", gpt-image-1: "high", "medium", "low". dall-e-3: "hd", "standard". dall-e-2: "standard".
	ResponseFormat    string     `json:"response_format,omitzero"`    // "url" or "b64_json"; url is valid for 60 minutes; gpt-image-1 only returns b64_json
	Size              string     `json:"size,omitzero"`               // "auto", gpt-image-1: "1024x1024", "1536x1024", "1024x1536". dall-e-3: "1024x1024", "1792x1024", "1024x1792". dall-e-2: "256x256", "512x512", "1024x1024".
	Style             string     `json:"style,omitzero"`              // dall-e-3: "vivid", "natural"
	User              string     `json:"user,omitzero"`               // End-user to help monitor and detect abuse
}

// Background is only supported on gpt-image-1.
type Background string

// Background mode values.
const (
	BackgroundAuto        Background = "auto"
	BackgroundTransparent Background = "transparent"
	BackgroundOpaque      Background = "opaque"
)

// ImageResponse is the provider-specific image generation response.
type ImageResponse struct {
	Created base.Time         `json:"created"`
	Data    []ImageChoiceData `json:"data"`
	Usage   struct {
		InputTokens        int64 `json:"input_tokens"`
		OutputTokens       int64 `json:"output_tokens"`
		TotalTokens        int64 `json:"total_tokens"`
		InputTokensDetails struct {
			TextTokens  int64 `json:"text_tokens"`
			ImageTokens int64 `json:"image_tokens"`
		} `json:"input_tokens_details"`
	} `json:"usage"`
	Background   string `json:"background"`    // "opaque"
	Size         string `json:"size"`          // e.g. "1024x1024"
	Quality      string `json:"quality"`       // e.g. "medium"
	OutputFormat string `json:"output_format"` // e.g. "png"
}

// ImageChoiceData is the data for one image generation choice.
type ImageChoiceData struct {
	B64JSON       []byte `json:"b64_json"`
	RevisedPrompt string `json:"revised_prompt"` // dall-e-3 only
	URL           string `json:"url"`            // Unsupported for gpt-image-1
}

// Model is documented at https://platform.openai.com/docs/api-reference/models/object
//
// Sadly the modalities aren't reported. The only way I can think of to find it at run time is to fetch
// https://platform.openai.com/docs/models/gpt-4o-mini-realtime-preview, find the div containing
// "Modalities:", then extract the modalities from the text.
type Model struct {
	ID      string    `json:"id"`
	Object  string    `json:"object"`
	Created base.Time `json:"created"`
	OwnedBy string    `json:"owned_by"`
}

// ModelsResponse represents the response structure for OpenAI models listing.
type ModelsResponse struct {
	Object string  `json:"object"` // list
	Data   []Model `json:"data"`
}

// File is documented at https://platform.openai.com/docs/api-reference/files/object
type File struct {
	Bytes         int64     `json:"bytes"` // File size
	CreatedAt     base.Time `json:"created_at"`
	ExpiresAt     base.Time `json:"expires_at"`
	Filename      string    `json:"filename"`
	ID            string    `json:"id"`
	Object        string    `json:"object"`         // "file"
	Purpose       string    `json:"purpose"`        // One of: assistants, assistants_output, batch, batch_output, fine-tune, fine-tune-results and vision
	Status        string    `json:"status"`         // Deprecated
	StatusDetails string    `json:"status_details"` // Deprecated
}

// FileDeleteResponse is documented at https://platform.openai.com/docs/api-reference/files/delete
type FileDeleteResponse struct {
	ID      string `json:"id"`
	Object  string `json:"object"` // "file"
	Deleted bool   `json:"deleted"`
}

// FileListResponse is documented at https://platform.openai.com/docs/api-reference/files/list
type FileListResponse struct {
	Data   []File `json:"data"`
	Object string `json:"object"` // "list"
}

//
// Responses API types.
//

// Truncation controls the truncation strategy for long conversations.
type Truncation string

const (
	// TruncationDisabled means the request will fail if input exceeds the model's context.
	TruncationDisabled Truncation = "disabled"
	// TruncationAuto means the model will automatically truncate input if it exceeds the model's context.
	TruncationAuto Truncation = "auto"
)

// Response represents a request to the OpenAI Responses API.
//
// https://platform.openai.com/docs/api-reference/responses/object
type Response struct {
	Model                string            `json:"model"`
	Background           bool              `json:"background,omitzero"`
	Instructions         string            `json:"instructions,omitzero"`
	MaxOutputTokens      int64             `json:"max_output_tokens,omitzero"`
	MaxToolCalls         int64             `json:"max_tool_calls,omitzero"`
	Metadata             map[string]string `json:"metadata,omitzero"`
	ParallelToolCalls    bool              `json:"parallel_tool_calls,omitzero"`
	PreviousResponseID   string            `json:"previous_response_id,omitzero"`
	PromptCacheKey       string            `json:"prompt_cache_key,omitzero"`
	PromptCacheRetention string            `json:"prompt_cache_retention,omitzero"`
	Reasoning            ReasoningConfig   `json:"reasoning,omitzero"`
	SafetyIdentifier     string            `json:"safety_identifier,omitzero"`
	ServiceTier          ServiceTier       `json:"service_tier,omitzero"`
	Store                bool              `json:"store"`
	Temperature          float64           `json:"temperature,omitzero"`
	Text                 struct {
		Format struct {
			Type        string             `json:"type"` // "text", "json_schema", "json_object"
			Name        string             `json:"name,omitzero"`
			Description string             `json:"description,omitzero"`
			Schema      *jsonschema.Schema `json:"schema,omitzero"`
			Strict      bool               `json:"strict,omitzero"`
		} `json:"format"`
		Verbosity string `json:"verbosity,omitzero"` // "low", "medium", "high"
	} `json:"text,omitzero"`
	TopLogprobs int64    `json:"top_logprobs,omitzero"` // [0, 20]
	TopP        float64  `json:"top_p,omitzero"`
	ToolChoice  string   `json:"tool_choice,omitzero"` // "none", "auto", "required"
	Truncation  string   `json:"truncation,omitzero"`  // "disabled", "auto"
	Tools       []Tool   `json:"tools,omitzero"`
	User        string   `json:"user,omitzero"`    // Deprecated, use SafetyIdentifier and PromptCacheKey
	Include     []string `json:"include,omitzero"` // "web_search_call.action.sources"

	// Request only
	Input  []Message `json:"input,omitzero"`
	Stream bool      `json:"stream,omitzero"`

	// Response only
	ID                string            `json:"id,omitzero"`
	Object            string            `json:"object,omitzero"` // "response"
	CreatedAt         base.Time         `json:"created_at,omitzero"`
	CompletedAt       base.Time         `json:"completed_at,omitzero"`
	Status            string            `json:"status,omitzero"` // "completed"
	IncompleteDetails IncompleteDetails `json:"incomplete_details,omitzero"`
	Error             APIError          `json:"error,omitzero"`
	Output            []Message         `json:"output,omitzero"`
	Usage             Usage             `json:"usage,omitzero"`
	Billing           map[string]string `json:"billing,omitzero"` // e.g. {"payer": "openai"}
}

// ReasoningConfig represents reasoning configuration for o-series models.
type ReasoningConfig struct {
	Effort  ReasoningEffort `json:"effort,omitzero"`
	Summary string          `json:"summary,omitzero"` // "auto", "concise", "detailed"
}

// Tool represents a tool that can be called by the model.
type Tool struct {
	// "function", "file_search", "computer_use_preview", "mcp", "code_interpreter", "image_generation",
	// "local_shell", "web_search"
	Type string `json:"type,omitzero"`

	// Type == "function"
	Name        string             `json:"name,omitzero"`
	Description string             `json:"description,omitzero"`
	Parameters  *jsonschema.Schema `json:"parameters,omitzero"`
	Strict      bool               `json:"strict,omitzero"`

	// Type == "file_search"
	FileSearchVectorStoreIDs []string `json:"vector_store_ids,omitzero"`

	// Type == "web_search"
	Filters struct {
		AllowedDomains []string `json:"allowed_domains,omitzero"`
	} `json:"filters,omitzero"`
	SearchContextSize string `json:"search_context_size,omitzero"` // "low", "medium", "high"
	UserLocation      struct {
		Type     string `json:"type,omitzero"`    // "approximate"
		Country  string `json:"country,omitzero"` // "GB"
		City     string `json:"city,omitzero"`    // "London"
		Region   string `json:"region,omitzero"`  // "London"
		Timezone string `json:"timezone,omitzero"`
	} `json:"user_location,omitzero"`
}

// MessageType controls what kind of content is allowed.
//
// This means a single message cannot contain multiple kind of calls at the time time. I really don't know
// why they did this especially that they have parallel tool calling support.
type MessageType string

// Message type values for inputs and outputs.
const (
	// MessageMessage represents inputs and outputs.
	MessageMessage MessageType = "message"
	// MessageFileSearchCall represents outputs.
	MessageFileSearchCall      MessageType = "file_search_call"
	MessageComputerCall        MessageType = "computer_call"
	MessageWebSearchCall       MessageType = "web_search_call"
	MessageFunctionCall        MessageType = "function_call"
	MessageReasoning           MessageType = "reasoning"
	MessageImageGenerationCall MessageType = "image_generation_call"
	MessageCodeInterpreterCall MessageType = "code_interpreter_call"
	MessageLocalShellCall      MessageType = "local_shell_call"
	MessageMcpListTools        MessageType = "mcp_list_tools"
	MessageMcpApprovalRequest  MessageType = "mcp_approval_request"
	MessageMcpCall             MessageType = "mcp_call"
	// MessageComputerCallOutput represents inputs.
	MessageComputerCallOutput   MessageType = "computer_call_output"
	MessageFunctionCallOutput   MessageType = "function_call_output"
	MessageLocalShellCallOutput MessageType = "local_shell_call_output"
	MessageMcpApprovalResponse  MessageType = "mcp_approval_response"
	MessageItemReference        MessageType = "item_reference"
)

// Message represents a message input or output to the model.
//
// In OpenAI Responses API, Message is a mix of Message and Content because the tool call type is in the
// Message.Type.
type Message struct {
	Type MessageType `json:"type,omitzero"`

	// Type == MessageMessage
	Role    string    `json:"role,omitzero"` // "user", "assistant", "system", "developer"
	Content []Content `json:"content,omitzero"`

	// Type == MessageMessage, MessageFileSearchCall, MessageFunctionCall, MessageReasoning
	Status string `json:"status,omitzero"` // "in_progress", "completed", "incomplete", "searching", "failed"

	// Type == MessageMessage (with Role == "assistant"), MessageFileSearchCall, MessageItemReference,
	// MessageFunctionCall, MessageFunctionCallOutput, MessageReasoning
	ID string `json:"id,omitzero"` // MessageItemReference: an internal identifier for an item to reference; Others: tool call ID

	// Type == MessageFileSearchCall
	Queries []string `json:"queries,omitzero"`
	Results []struct {
		Attributes map[string]string `json:"attributes,omitzero"`
		FileID     string            `json:"file_id,omitzero"`
		Filename   string            `json:"filename,omitzero"`
		Score      float64           `json:"score,omitzero"` // [0, 1]
		Text       string            `json:"text,omitzero"`
	} `json:"results,omitzero"`

	// Type == MessageFunctionCall
	Arguments string `json:"arguments,omitzero"` // JSON
	Name      string `json:"name,omitzero"`

	// Type == MessageFunctionCall, MessageFunctionCallOutput
	CallID string `json:"call_id,omitzero"`

	// Type == MessageFunctionCallOutput
	Output string `json:"output,omitzero"` // JSON

	// Type == MessageReasoning
	EncryptedContent string             `json:"encrypted_content,omitzero"`
	Summary          []ReasoningSummary `json:"summary,omitzero"`

	// Type == MessageWebSearchCall
	Action struct {
		Type    string `json:"type,omitzero"` // "search"
		Query   string `json:"query,omitzero"`
		Sources []struct {
			Type string `json:"type,omitzero"` // "url"
			URL  string `json:"url,omitzero"`
		} `json:"sources,omitzero"`
	} `json:"action,omitzero"`
}

// ContentType defines the data being transported. It only includes actual data (text, files), no tool call nor result.
type ContentType string

// Content type values.
const (
	// ContentInputText is an input text content type.
	ContentInputText  ContentType = "input_text"
	ContentInputImage ContentType = "input_image"
	ContentInputFile  ContentType = "input_file"

	// ContentOutputText is an output text content type.
	ContentOutputText ContentType = "output_text"
	ContentRefusal    ContentType = "refusal"
)

// Content represents different types of input content.
type Content struct {
	Type ContentType `json:"type,omitzero"`

	// Type == ContentInputText, ContentOutputText
	Text string `json:"text,omitzero"`

	// Type == ContentInputImage, ContentInputFile
	FileID string `json:"file_id,omitzero"`

	// Type == ContentInputImage
	ImageURL string `json:"image_url,omitzero"` // URL or base64
	Detail   string `json:"detail,omitzero"`    // "high", "low", "auto" (default)

	// Type == ContentInputFile
	FileData string `json:"file_data,omitzero"` // TODO: confirm if base64
	Filename string `json:"filename,omitzero"`

	// Type == ContentOutputText
	Annotations []Annotation `json:"annotations,omitzero"`
	Logprobs    []Logprobs   `json:"logprobs,omitzero"`

	// Type == ContentRefusal
	Refusal string `json:"refusal,omitzero"`
}

// APIError represents an API error in the response.
type APIError struct {
	Code    string `json:"code"` // "server_error"
	Message string `json:"message"`
}

// IncompleteDetails represents details about why a response is incomplete.
type IncompleteDetails struct {
	Reason string `json:"reason"`
}

// Annotation represents annotations in output text.
type Annotation struct {
	// "file_citation", "url_citation", "container_file_citation", "file_path"
	Type string `json:"type,omitzero"`

	// Type == "file_citation", "container_file_citation", "file_path"
	FileID string `json:"file_id,omitzero"`

	// Type == "file_citation", "file_path"
	Index int64 `json:"index,omitzero"`

	// Type == "url_citation"
	URL   string `json:"url,omitzero"`
	Title string `json:"title,omitzero"`

	// Type == "url_citation", "container_file_citation"
	StartIndex int64 `json:"start_index,omitzero"`
	EndIndex   int64 `json:"end_index,omitzero"`
}

// Logprobs is the provider-specific log probabilities.
type Logprobs struct {
	Token       string  `json:"token,omitzero"`
	Bytes       []byte  `json:"bytes,omitzero"`
	Logprob     float64 `json:"logprob,omitzero"`
	TopLogprobs []struct {
		Token   string  `json:"token,omitzero"`
		Bytes   []byte  `json:"bytes,omitzero"`
		Logprob float64 `json:"logprob,omitzero"`
	} `json:"top_logprobs,omitzero"`
}

// ReasoningSummary represents reasoning summary content.
type ReasoningSummary struct {
	Type string `json:"type,omitzero"` // "summary_text"
	Text string `json:"text,omitzero"`
}

// Usage represents token usage statistics.
type Usage struct {
	InputTokens        int64 `json:"input_tokens"`
	InputTokensDetails struct {
		CachedTokens int64 `json:"cached_tokens"`
	} `json:"input_tokens_details"`
	OutputTokens        int64 `json:"output_tokens"`
	OutputTokensDetails struct {
		ReasoningTokens int64 `json:"reasoning_tokens"`
	} `json:"output_tokens_details"`
	TotalTokens int64 `json:"total_tokens"`
}

// TokenDetails provides detailed token usage breakdown.
type TokenDetails struct {
	CachedTokens    int64 `json:"cached_tokens,omitzero"`
	AudioTokens     int64 `json:"audio_tokens,omitzero"`
	ReasoningTokens int64 `json:"reasoning_tokens,omitzero"`
}

// ResponseType is one of the event at https://platform.openai.com/docs/api-reference/responses-streaming
type ResponseType string

// Response event type values.
const (
	ResponseCompleted                       ResponseType = "response.completed"
	ResponseContentPartAdded                ResponseType = "response.content_part.added"
	ResponseContentPartDone                 ResponseType = "response.content_part.done"
	ResponseCreated                         ResponseType = "response.created"
	ResponseError                           ResponseType = "error"
	ResponseFailed                          ResponseType = "response.failed"
	ResponseFileSearchCallCompleted         ResponseType = "response.file_search_call.completed"
	ResponseFileSearchCallInProgress        ResponseType = "response.file_search_call.in_progress"
	ResponseFileSearchCallSearching         ResponseType = "response.file_search_call.searching"
	ResponseFunctionCallArgumentsDelta      ResponseType = "response.function_call_arguments.delta"
	ResponseFunctionCallArgumentsDone       ResponseType = "response.function_call_arguments.done"
	ResponseImageGenerationCallCompleted    ResponseType = "response.image_generation_call.completed"
	ResponseImageGenerationCallGenerating   ResponseType = "response.image_generation_call.generating"
	ResponseImageGenerationCallInProgress   ResponseType = "response.image_generation_call.in_progress"
	ResponseImageGenerationCallPartialImage ResponseType = "response.image_generation_call.partial_image"
	ResponseInProgress                      ResponseType = "response.in_progress"
	ResponseIncomplete                      ResponseType = "response.incomplete"
	ResponseMCPCallArgumentsDelta           ResponseType = "response.mcp_call.arguments.delta"
	ResponseMCPCallArgumentsDone            ResponseType = "response.mcp_call.arguments.done"
	ResponseMCPCallCompleted                ResponseType = "response.mcp_call.completed"
	ResponseMCPCallFailed                   ResponseType = "response.mcp_call.failed"
	ResponseMCPCallInProgress               ResponseType = "response.mcp_call.in_progress"
	ResponseMCPListToolsCompleted           ResponseType = "response.mcp_list_tools.completed"
	ResponseMCPListToolsFailed              ResponseType = "response.mcp_list_tools.failed"
	ResponseMCPListToolsInProgress          ResponseType = "response.mcp_list_tools.in_progress"
	ResponseCodeInterpreterCallInterpreting ResponseType = "response.code_interpreter_call.interpreting"
	ResponseCodeInterpreterCallCompleted    ResponseType = "response.code_interpreter_call.completed"
	ResponseCustomToolCallInputDelta        ResponseType = "response.custom_tool_call_input.delta"
	ResponseCustomToolCallInputDone         ResponseType = "response.custom_tool_call_input.done"
	ResponseCodeInterpreterCallDelta        ResponseType = "response.code_interpreter_call.delta"
	ResponseCodeInterpreterCallDone         ResponseType = "response.code_interpreter_call.done"
	ResponseOutputItemAdded                 ResponseType = "response.output_item.added"
	ResponseOutputItemDone                  ResponseType = "response.output_item.done"
	ResponseOutputTextDelta                 ResponseType = "response.output_text.delta"
	ResponseOutputTextDone                  ResponseType = "response.output_text.done"
	ResponseOutputTextAnnotationAdded       ResponseType = "response.output_text.annotation.added"
	ResponseQueued                          ResponseType = "response.queued"
	ResponseReasoningSummaryPartAdded       ResponseType = "response.reasoning_summary_part.added"
	ResponseReasoningSummaryPartDone        ResponseType = "response.reasoning_summary_part.done"
	ResponseReasoningSummaryTextDelta       ResponseType = "response.reasoning_summary_text.delta"
	ResponseReasoningSummaryTextDone        ResponseType = "response.reasoning_summary_text.done"
	ResponseReasoningTextDelta              ResponseType = "response.reasoning_text.delta"
	ResponseReasoningTextDone               ResponseType = "response.reasoning_text.done"
	ResponseRefusalDelta                    ResponseType = "response.refusal.delta"
	ResponseRefusalDone                     ResponseType = "response.refusal.done"
	ResponseWebSearchCallCompleted          ResponseType = "response.web_search_call.completed"
	ResponseWebSearchCallInProgress         ResponseType = "response.web_search_call.in_progress"
	ResponseWebSearchCallSearching          ResponseType = "response.web_search_call.searching"
)

// ResponseStreamChunkResponse represents a streaming response chunk.
//
// https://platform.openai.com/docs/api-reference/responses-streaming
type ResponseStreamChunkResponse struct {
	Type           ResponseType `json:"type,omitzero"`
	SequenceNumber int64        `json:"sequence_number,omitzero"`

	// Type == ResponseCreated, ResponseInProgress, ResponseCompleted, ResponseFailed, ResponseIncomplete,
	// ResponseQueued
	Response Response `json:"response,omitzero"`

	// Type == ResponseOutputItemAdded, ResponseOutputItemDone, ResponseContentPartAdded,
	// ResponseContentPartDone, ResponseOutputTextDelta, ResponseOutputTextDone, ResponseRefusalDelta,
	// ResponseRefusalDone, ResponseFunctionCallArgumentsDelta, ResponseFunctionCallArgumentsDone,
	// ResponseReasoningSummaryPartAdded, ResponseReasoningSummaryPartDone, ResponseReasoningSummaryTextDelta,
	// ResponseReasoningSummaryTextDone, ResponseOutputTextAnnotationAdded
	OutputIndex int64 `json:"output_index,omitzero"`

	// Type == ResponseOutputItemAdded, ResponseOutputItemDone
	Item Message `json:"item,omitzero"`

	// Type == ResponseContentPartAdded, ResponseContentPartDone, ResponseOutputTextDelta,
	// ResponseOutputTextDone, ResponseRefusalDelta, ResponseRefusalDone,
	//  ResponseOutputTextAnnotationAdded
	ContentIndex int64 `json:"content_index,omitzero"`

	// Type == ResponseContentPartAdded, ResponseContentPartDone, ResponseOutputTextDelta,
	// ResponseOutputTextDone, ResponseRefusalDelta, ResponseRefusalDone, ResponseFunctionCallArgumentsDelta,
	// ResponseFunctionCallArgumentsDone, ResponseReasoningSummaryPartAdded, ResponseReasoningSummaryPartDone,
	// ResponseReasoningSummaryTextDelta, ResponseReasoningSummaryTextDone, ResponseOutputTextAnnotationAdded
	ItemID string `json:"item_id,omitzero"`

	// Type == ResponseContentPartAdded, ResponseContentPartDone, ResponseReasoningSummaryPartAdded,
	// ResponseReasoningSummaryPartDone
	Part Content `json:"part,omitzero"`

	// Type == ResponseOutputTextDelta, ResponseRefusalDelta, ResponseFunctionCallArgumentsDelta,
	// ResponseReasoningSummaryTextDelta
	Delta string `json:"delta,omitzero"`

	// Type == ResponseOutputTextDone, ResponseReasoningSummaryTextDone
	Text string `json:"text,omitzero"`

	// Type == ResponseRefusalDone
	Refusal string `json:"refusal,omitzero"`

	// Type == ResponseFunctionCallArgumentsDone
	Arguments string `json:"arguments,omitzero"`

	// Type == ResponseReasoningSummaryPartAdded, ResponseReasoningSummaryPartDone,
	// ResponseReasoningSummaryTextDelta, ResponseReasoningSummaryTextDone
	SummaryIndex int64 `json:"summary_index,omitzero"`

	// Type == ResponseOutputTextAnnotationAdded
	Annotation      Annotation `json:"annotation,omitzero"`
	AnnotationIndex int64      `json:"annotation_index,omitzero"`

	// Type == ResponseError
	ErrorResponse

	Logprobs []Logprobs `json:"logprobs,omitzero"`

	Obfuscation string `json:"obfuscation,omitzero"`

	/* TODO
	ResponseFileSearchCallCompleted
	ResponseFileSearchCallInProgress
	ResponseFileSearchCallSearching
	ResponseFunctionCallArgumentsDelta
	ResponseFunctionCallArgumentsDone
	ResponseImageGenerationCallCompleted
	ResponseImageGenerationCallGenerating
	ResponseImageGenerationCallInProgress
	ResponseImageGenerationCallPartialImage
	ResponseMCPCallArgumentsDelta
	ResponseMCPCallArgumentsDone
	ResponseMCPCallCompleted
	ResponseMCPCallFailed
	ResponseMCPCallInProgress
	ResponseMCPListToolsCompleted
	ResponseMCPListToolsFailed
	ResponseMCPListToolsInProgress
	ResponseWebSearchCallCompleted
	ResponseWebSearchCallInProgress
	ResponseWebSearchCallSearching
	*/
}

//
// Batch API types.
//

// BatchRequestInput is documented at https://platform.openai.com/docs/api-reference/batch/request-input
type BatchRequestInput struct {
	CustomID string   `json:"custom_id"`
	Method   string   `json:"method"` // "POST"
	URL      string   `json:"url"`    // "/v1/chat/completions", "/v1/embeddings", "/v1/completions", "/v1/responses"
	Body     Response `json:"body"`
}

// BatchRequestOutput is documented at https://platform.openai.com/docs/api-reference/batch/request-output
type BatchRequestOutput struct {
	CustomID string   `json:"custom_id"`
	ID       string   `json:"id"`
	Error    APIError `json:"error"`
	Response struct {
		StatusCode int      `json:"status_code"`
		RequestID  string   `json:"request_id"` // To use when contacting support
		Body       Response `json:"body"`
	} `json:"response"`
}

// BatchRequest is documented at https://platform.openai.com/docs/api-reference/batch/create
type BatchRequest struct {
	CompletionWindow string            `json:"completion_window"` // Must be "24h"
	Endpoint         string            `json:"endpoint"`          // One of /v1/responses, /v1/chat/completions, /v1/embeddings, /v1/completions
	InputFileID      string            `json:"input_file_id"`     // File must be JSONL
	Metadata         map[string]string `json:"metadata,omitzero"` // Maximum 16 keys of 64 chars, values max 512 chars
}

// Batch is documented at https://platform.openai.com/docs/api-reference/batch/object
type Batch struct {
	CancelledAt      base.Time `json:"cancelled_at"`
	CancellingAt     base.Time `json:"cancelling_at"`
	CompletedAt      base.Time `json:"completed_at"`
	CompletionWindow string    `json:"completion_window"` // "24h"
	CreatedAt        base.Time `json:"created_at"`
	Endpoint         string    `json:"endpoint"`      // Same as BatchRequest.Endpoint
	ErrorFileID      string    `json:"error_file_id"` // File ID containing the outputs of requests with errors.
	Errors           struct {
		Data []struct {
			Code    string `json:"code"`
			Line    int64  `json:"line"`
			Message string `json:"message"`
			Param   string `json:"param"`
		} `json:"data"`
	} `json:"errors"`
	ExpiredAt     base.Time         `json:"expired_at"`
	ExpiresAt     base.Time         `json:"expires_at"`
	FailedAt      base.Time         `json:"failed_at"`
	FinalizingAt  base.Time         `json:"finalizing_at"`
	ID            string            `json:"id"`
	InProgressAt  base.Time         `json:"in_progress_at"`
	InputFileID   string            `json:"input_file_id"` // Input data
	Metadata      map[string]string `json:"metadata"`
	Object        string            `json:"object"`         // "batch"
	OutputFileID  string            `json:"output_file_id"` // Output data
	RequestCounts struct {
		Completed int64 `json:"completed"`
		Failed    int64 `json:"failed"`
		Total     int64 `json:"total"`
	} `json:"request_counts"`
	Status string `json:"status"` // "completed", "in_progress", "validating", "finalizing"
}

// ErrorResponse represents an error response from the OpenAI API.
type ErrorResponse struct {
	ErrorVal struct {
		Message string `json:"message"`
		Type    string `json:"type"`
		Code    string `json:"code"`
		Param   string `json:"param"`
	} `json:"error"`
}
