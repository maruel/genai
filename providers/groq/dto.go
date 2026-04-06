// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Wire types for the Groq chat completions REST API.
//
// Reference: https://console.groq.com/docs/api-reference

package groq

import (
	"encoding/json"

	"github.com/invopop/jsonschema"
	"github.com/maruel/genai/base"
)

// ServiceTier is the quality of service to determine the request's priority.
// https://console.groq.com/docs/flex-processing
type ServiceTier string

const (
	// ServiceTierOnDemand is the default tier and the one you are used to. We have kept rate limits low in
	// order to ensure fairness and a consistent experience.
	ServiceTierOnDemand ServiceTier = "on_demand"
	// ServiceTierAuto uses on-demand rate limits, then falls back to flex tier if those limits are exceeded.
	ServiceTierAuto ServiceTier = "auto"
	// ServiceTierFlex offers on-demand processing when capacity is available, with rapid timeouts if resources
	// are constrained. This tier is perfect for workloads that prioritize fast inference and can gracefully
	// handle occasional request failures. It provides an optimal balance between performance and reliability
	// for workloads that don't require guaranteed processing.
	ServiceTierFlex ServiceTier = "flex"
)

// ReasoningFormat defines the post processing format of the reasoning done by groq for select models.
//
// See https://console.groq.com/docs/reasoning
type ReasoningFormat string

// Reasoning format values.
const (
	ReasoningFormatParsed ReasoningFormat = "parsed"
	ReasoningFormatRaw    ReasoningFormat = "raw"
	ReasoningFormatHidden ReasoningFormat = "hidden"
)

// ChatRequest is documented at https://console.groq.com/docs/api-reference#chat-create
type ChatRequest struct {
	FrequencyPenalty  float64         `json:"frequency_penalty,omitzero"` // [-2.0, 2.0]
	MaxChatTokens     int64           `json:"max_completion_tokens,omitzero"`
	Messages          []Message       `json:"messages"`
	Model             string          `json:"model"`
	ParallelToolCalls bool            `json:"parallel_tool_calls,omitzero"`
	PresencePenalty   float64         `json:"presence_penalty,omitzero"` // [-2.0, 2.0]
	ReasoningFormat   ReasoningFormat `json:"reasoning_format,omitzero"`
	ResponseFormat    struct {
		Type       string             `json:"type,omitzero"` // "json_object", "json_schema"
		JSONSchema *jsonschema.Schema `json:"json_schema,omitzero"`
	} `json:"response_format,omitzero"`
	SearchSettings struct {
		Country        string   `json:"country,omitzero"`
		ExcludeDomains []string `json:"exclude_domains,omitzero"`
		IncludeDomains []string `json:"include_domains,omitzero"`
		IncludeImages  bool     `json:"include_images,omitzero"`
	} `json:"search_settings,omitzero"`
	Seed          int64       `json:"seed,omitzero"`
	ServiceTier   ServiceTier `json:"service_tier,omitzero"`
	Stop          []string    `json:"stop,omitzero"` // keywords to stop completion
	Stream        bool        `json:"stream"`
	StreamOptions struct {
		IncludeUsage bool `json:"include_usage,omitzero"`
	} `json:"stream_options,omitzero"`
	Temperature float64 `json:"temperature,omitzero"` // [0, 2]
	Tools       []Tool  `json:"tools,omitzero"`
	// Alternative when forcing a specific function. This can probably be achieved
	// by providing a single tool and ToolChoice == "required".
	// ToolChoice struct {
	// 	Type     string `json:"type,omitzero"` // "function"
	// 	Function struct {
	// 		Name string `json:"name,omitzero"`
	// 	} `json:"function,omitzero"`
	// } `json:"tool_choice,omitzero"`
	ToolChoice string  `json:"tool_choice,omitzero"` // "none", "auto", "required", or struct {"type": "function", "function": {"name": "my_function"}}
	TopP       float64 `json:"top_p,omitzero"`       // [0, 1]
	User       string  `json:"user,omitzero"`

	// Explicitly Unsupported:
	// LogitBias           map[string]float64 `json:"logit_bias,omitzero"`
	// Logprobs            bool               `json:"logprobs,omitzero"`
	// TopLogprobs         int64                `json:"top_logprobs,omitzero"`     // [0, 20]
	// N                   int64                `json:"n,omitzero"`                // Number of choices
}

// Message is documented at https://console.groq.com/docs/api-reference#chat-create
type Message struct {
	Role       string     `json:"role"`               // "system", "assistant", "user"
	Name       string     `json:"name,omitzero"`      // An optional name for the participant. Provides the model information to differentiate between participants of the same role.
	Content    Contents   `json:"content,omitzero"`   // Content is always returned as a string.
	Reasoning  string     `json:"reasoning,omitzero"` // In replies
	Channel    string     `json:"channel,omitzero"`   // "analysis"
	ToolCalls  []ToolCall `json:"tool_calls,omitzero"`
	ToolCallID string     `json:"tool_call_id,omitzero"`

	// "browser_search"
	ExecutedTools []struct {
		Name string `json:"name,omitzero"` // "browser.search", "browser.open", "browser.find"
		// Arguments is a JSON encoded arguments for the server side tool.
		//  - "browser.search" uses BrowserSearchArguments.
		//  - "browser.open" uses BrowserOpenArguments.
		Arguments     string `json:"arguments,omitzero"`
		Index         int64  `json:"index,omitzero"`
		Type          string `json:"type,omitzero"` // "function", "python", etc.
		Output        string `json:"output,omitzero"`
		SearchResults struct {
			Results []struct {
				Title   string  `json:"title,omitzero"`
				URL     string  `json:"url,omitzero"`
				Content string  `json:"content,omitzero"`
				Score   float64 `json:"score,omitzero"`
			} `json:"results,omitzero"`
		} `json:"search_results,omitzero"`
		CodeResults []struct {
			Text string `json:"text,omitzero"`
		} `json:"code_results,omitzero"`
	} `json:"executed_tools,omitzero"`
}

// Contents exists to marshal single content text block as a string.
//
// Groq requires this for assistant messages.
type Contents []Content

// Content is a provider-specific content block.
type Content struct {
	Type ContentType `json:"type,omitzero"`

	// Type == "text"
	Text string `json:"text,omitzero"`

	// Type == "image_url"
	ImageURL struct {
		Detail string `json:"detail,omitzero"` // "auto", "low", "high"
		URL    string `json:"url,omitzero"`    // URL or base64 encoded image
	} `json:"image_url,omitzero"`
}

// ContentType is a provider-specific content type.
type ContentType string

// Content type values.
const (
	ContentText     ContentType = "text"
	ContentImageURL ContentType = "image_url"
)

// Tool is a provider-specific tool definition.
type Tool struct {
	Type     string `json:"type,omitzero"` // "function"
	Function struct {
		Name        string             `json:"name,omitzero"`
		Description string             `json:"description,omitzero"`
		Parameters  *jsonschema.Schema `json:"parameters,omitzero"`
	} `json:"function,omitzero"`
}

// ToolCall is a provider-specific tool call.
type ToolCall struct {
	Index    int64  `json:"index,omitzero"`
	Type     string `json:"type,omitzero"` // "function"
	ID       string `json:"id,omitzero"`
	Function struct {
		Name      string `json:"name,omitzero"`
		Arguments string `json:"arguments,omitzero"`
	} `json:"function,omitzero"`
}

// UsageBreakdownModel represents per-model usage information in a breakdown.
type UsageBreakdownModel struct {
	Model string `json:"model"`
	Usage Usage  `json:"usage"`
}

// UsageBreakdown contains per-model usage information.
type UsageBreakdown struct {
	Models []UsageBreakdownModel `json:"models"`
}

// ChatResponse is the provider-specific chat completion response.
type ChatResponse struct {
	Choices []struct {
		FinishReason FinishReason `json:"finish_reason"`
		Index        int64        `json:"index"`
		Message      Message      `json:"message"`
		Logprobs     struct{}     `json:"logprobs"`
	} `json:"choices"`
	Created           base.Time      `json:"created"`
	ID                string         `json:"id"`
	Model             string         `json:"model"`
	Object            string         `json:"object"` // "chat.completion"
	Usage             Usage          `json:"usage"`
	UsageBreakdown    UsageBreakdown `json:"usage_breakdown"`
	SystemFingerprint string         `json:"system_fingerprint"`
	ServiceTier       ServiceTier    `json:"service_tier"`
	Xgroq             struct {
		ID             string         `json:"id"`
		Seed           json.Number    `json:"seed,omitzero"`
		UsageBreakdown UsageBreakdown `json:"usage_breakdown,omitzero"`
	} `json:"x_groq"`
}

// FinishReason is a provider-specific finish reason.
type FinishReason string

// Finish reason values.
const (
	FinishStop          FinishReason = "stop"
	FinishLength        FinishReason = "length"
	FinishToolCalls     FinishReason = "tool_calls"
	FinishContentFilter FinishReason = "content_filter"
)

// Usage is the provider-specific token usage.
type Usage struct {
	QueueTime               float64        `json:"queue_time"`
	PromptTokens            int64          `json:"prompt_tokens"`
	PromptTime              float64        `json:"prompt_time"`
	CompletionTokens        int64          `json:"completion_tokens"`
	CompletionTime          float64        `json:"completion_time"`
	TotalTokens             int64          `json:"total_tokens"`
	TotalTime               float64        `json:"total_time"`
	PromptTokensDetails     map[string]any `json:"prompt_tokens_details,omitzero"`
	CompletionTokensDetails map[string]any `json:"completion_tokens_details,omitzero"`
}

// BrowserSearchArguments is the Argument for the "browser.search" tool.
type BrowserSearchArguments struct {
	Query  string `json:"query,omitzero"`
	TopN   int64  `json:"topn,omitzero"`
	Source string `json:"source,omitzero"`
}

// BrowserOpenArguments is the Argument for the "browser.open" tool.
type BrowserOpenArguments struct {
	Cursor int64 `json:"cursor,omitzero"`
	ID     int64 `json:"id,omitzero"`
}

// BrowserFindArguments is the Argument for the "browser.find" tool.
type BrowserFindArguments struct {
	Cursor  int64  `json:"cursor,omitzero"`
	Pattern string `json:"pattern,omitzero"`
}

// ChatStreamChunkResponse is the provider-specific streaming chat chunk.
type ChatStreamChunkResponse struct {
	ID                string    `json:"id"`
	Object            string    `json:"object"`
	Created           base.Time `json:"created"`
	Model             string    `json:"model"`
	SystemFingerprint string    `json:"system_fingerprint"`
	Choices           []struct {
		Index        int64        `json:"index"`
		Delta        Message      `json:"delta"`
		Logprobs     struct{}     `json:"logprobs"` // Groq doesn't support logprobs but still sent a null item.
		FinishReason FinishReason `json:"finish_reason"`
	} `json:"choices"`
	Usage Usage `json:"usage,omitzero"`
	Xgroq struct {
		ID             string         `json:"id"`
		Seed           json.Number    `json:"seed,omitzero"`
		Usage          Usage          `json:"usage"`
		UsageBreakdown UsageBreakdown `json:"usage_breakdown,omitzero"`
	} `json:"x_groq"`
}

// Model is the provider-specific model metadata.
type Model struct {
	ID                  string    `json:"id"`
	Object              string    `json:"object"`
	Created             base.Time `json:"created"`
	OwnedBy             string    `json:"owned_by"`
	Active              bool      `json:"active"`
	ContextWindow       int64     `json:"context_window"`
	PublicApps          []string  `json:"public_apps"`
	MaxCompletionTokens int64     `json:"max_completion_tokens"`
}

// ModelsResponse represents the response structure for Groq models listing.
type ModelsResponse struct {
	Object string  `json:"object"` // list
	Data   []Model `json:"data"`
}

// ErrorResponse is the provider-specific error response.
type ErrorResponse struct {
	ErrorVal struct {
		Message          string `json:"message"`
		Type             string `json:"type"`  // e.g. "invalid_request_error"
		Code             string `json:"code"`  // e.g. "context_length_exceeded"
		Param            string `json:"param"` // e.g. "messages"
		FailedGeneration string `json:"failed_generation"`
		StatusCode       int64  `json:"status_code"`
	} `json:"error"`
}
