// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Wire types for the GitHub Models inference and catalog REST API.
//
// Documentation: https://docs.github.com/en/rest/models

package github

import (
	"github.com/maruel/genai"
	"github.com/maruel/genai/base"
)

// ChatRequest is documented at https://docs.github.com/en/rest/models/inference
type ChatRequest struct {
	Model            string    `json:"model"`
	Messages         []Message `json:"messages"`
	MaxTokens        int64     `json:"max_tokens,omitzero"`
	Temperature      float64   `json:"temperature,omitzero"`
	TopP             float64   `json:"top_p,omitzero"`
	FrequencyPenalty float64   `json:"frequency_penalty,omitzero"`
	PresencePenalty  float64   `json:"presence_penalty,omitzero"`
	Seed             int64     `json:"seed,omitzero"`
	Stop             []string  `json:"stop,omitzero"`
	Stream           bool      `json:"stream,omitzero"`
	StreamOptions    struct {
		IncludeUsage bool `json:"include_usage,omitzero"`
	} `json:"stream_options,omitzero"`
	Tools      []Tool `json:"tools,omitzero"`
	ToolChoice string `json:"tool_choice,omitzero"` // "none", "auto", "required"
	// ResponseFormat is either {"type":"json_object"} or {"type":"json_schema","json_schema":{...}}
	ResponseFormat responseFormat `json:"response_format,omitzero"`
}

type responseFormat struct {
	Type       string           `json:"type"`
	JSONSchema genai.JSONSchema `json:"json_schema,omitzero"`
}

// Message is a provider-specific message.
type Message struct {
	Role             string     `json:"role"`
	Content          Contents   `json:"content,omitzero"`
	ReasoningContent string     `json:"reasoning_content,omitzero"` // Some models return reasoning here
	Refusal          string     `json:"refusal,omitzero"`
	Annotations      []struct{} `json:"annotations,omitzero"`
	ToolCalls        []ToolCall `json:"tool_calls,omitzero"`
	ToolCallID       string     `json:"tool_call_id,omitzero"`
	Name             string     `json:"name,omitzero"`
}

// Contents handles marshalling as a string when there is a single text content item.
type Contents []Content

// Content is a provider-specific content block.
type Content struct {
	Type ContentType `json:"type,omitzero"`

	// Type == "text"
	Text string `json:"text,omitzero"`

	// Type == "image_url"
	ImageURL ImageURL `json:"image_url,omitzero"`
}

// ImageURL is a provider-specific image URL.
type ImageURL struct {
	URL    string `json:"url"`
	Detail string `json:"detail,omitzero"` // "auto", "low", "high"
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
	Type     string   `json:"type"` // "function"
	Function Function `json:"function"`
}

// Function is a provider-specific function definition.
type Function struct {
	Name        string           `json:"name"`
	Description string           `json:"description,omitzero"`
	Parameters  genai.JSONSchema `json:"parameters,omitzero"`
	Strict      bool             `json:"strict,omitzero"`
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

// contentFilterResult is an Azure content safety filter verdict.
type contentFilterResult struct {
	Filtered bool   `json:"filtered"`
	Severity string `json:"severity,omitzero"`
	Detected bool   `json:"detected,omitzero"`
}

// contentFilterResults holds Azure content safety filter verdicts.
type contentFilterResults struct {
	Hate                  contentFilterResult `json:"hate"`
	SelfHarm              contentFilterResult `json:"self_harm"`
	Sexual                contentFilterResult `json:"sexual"`
	Violence              contentFilterResult `json:"violence"`
	Jailbreak             contentFilterResult `json:"jailbreak"`
	ProtectedMaterialCode contentFilterResult `json:"protected_material_code"`
	ProtectedMaterialText contentFilterResult `json:"protected_material_text"`
}

// ChatResponse is the provider-specific chat completion response.
type ChatResponse struct {
	ID                string    `json:"id"`
	Object            string    `json:"object"` // "chat.completion"
	Created           base.Time `json:"created"`
	Model             string    `json:"model"`
	SystemFingerprint string    `json:"system_fingerprint"`
	Choices           []struct {
		Index                int64                `json:"index"`
		FinishReason         FinishReason         `json:"finish_reason"`
		Message              Message              `json:"message"`
		Logprobs             *struct{}            `json:"logprobs"`
		ContentFilterResults contentFilterResults `json:"content_filter_results"`
	} `json:"choices"`
	Usage               Usage `json:"usage"`
	PromptFilterResults []struct {
		PromptIndex          int64                `json:"prompt_index"`
		ContentFilterResults contentFilterResults `json:"content_filter_results"`
	} `json:"prompt_filter_results"`
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
	PromptTokens        int64 `json:"prompt_tokens"`
	CompletionTokens    int64 `json:"completion_tokens"`
	TotalTokens         int64 `json:"total_tokens"`
	PromptTokensDetails struct {
		CachedTokens int64 `json:"cached_tokens"`
		AudioTokens  int64 `json:"audio_tokens"`
	} `json:"prompt_tokens_details,omitzero"`
	CompletionTokensDetails struct {
		ReasoningTokens          int64 `json:"reasoning_tokens"`
		AudioTokens              int64 `json:"audio_tokens"`
		AcceptedPredictionTokens int64 `json:"accepted_prediction_tokens"`
		RejectedPredictionTokens int64 `json:"rejected_prediction_tokens"`
	} `json:"completion_tokens_details,omitzero"`
}

// ChatStreamChunkResponse is the provider-specific streaming chat chunk.
type ChatStreamChunkResponse struct {
	ID                string    `json:"id"`
	Object            string    `json:"object"`
	Created           base.Time `json:"created"`
	Model             string    `json:"model"`
	SystemFingerprint string    `json:"system_fingerprint"`
	Obfuscation       string    `json:"obfuscation,omitzero"` // Azure-specific
	Choices           []struct {
		Index int64 `json:"index"`
		Delta struct {
			Role      string     `json:"role"`
			Content   Contents   `json:"content"`
			ToolCalls []ToolCall `json:"tool_calls"`
			Refusal   *string    `json:"refusal,omitzero"`
		} `json:"delta"`
		FinishReason         FinishReason         `json:"finish_reason"`
		Logprobs             *struct{}            `json:"logprobs"`
		ContentFilterResults contentFilterResults `json:"content_filter_results"`
	} `json:"choices"`
	Usage               Usage `json:"usage"`
	PromptFilterResults []struct {
		PromptIndex          int64                `json:"prompt_index"`
		ContentFilterResults contentFilterResults `json:"content_filter_results"`
	} `json:"prompt_filter_results"`
}

// CatalogModel is a model from the GitHub Models catalog.
type CatalogModel struct {
	ID                        string   `json:"id"`
	Name                      string   `json:"name"`
	Publisher                 string   `json:"publisher"`
	Summary                   string   `json:"summary"`
	RateLimitTier             string   `json:"rate_limit_tier"` // "low", "high"
	Capabilities              []string `json:"capabilities"`    // "chat", "tool_call", "json_output", "streaming"
	Tags                      []string `json:"tags"`
	Version                   string   `json:"version"`
	Registry                  string   `json:"registry"`
	HTMLURL                   string   `json:"html_url"`
	SupportedInputModalities  []string `json:"supported_input_modalities"`  // "text", "image"
	SupportedOutputModalities []string `json:"supported_output_modalities"` // "text"
	Limits                    struct {
		MaxInputTokens  int64 `json:"max_input_tokens"`
		MaxOutputTokens int64 `json:"max_output_tokens"`
	} `json:"limits"`
}

// ErrorResponse is the provider-specific error response.
type ErrorResponse struct {
	ErrorVal struct {
		Code    string `json:"code"`
		Message string `json:"message"`
		Type    string `json:"type"`
		Param   string `json:"param"`
		Details string `json:"details"`
	} `json:"error"`
}
