// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Package openairesponses implements a client for the OpenAI Responses API.
//
// It is described at https://platform.openai.com/docs/api-reference/responses/create
package openairesponses

// See official client at https://github.com/openai/openai-go

import (
	"fmt"
	"net/http"
	"os"

	"github.com/maruel/genai"
	"github.com/maruel/genai/base"
	"github.com/maruel/httpjson"
	"github.com/maruel/roundtrippers"
)

// Client is a client for the OpenAI Responses API.
type Client struct {
	base.ProviderGen[*ErrorResponse, *ResponseRequest, *ResponseResponse, ResponseStreamChunkResponse]
}

// ErrorResponse represents an error response from the OpenAI API.
type ErrorResponse struct {
	Error struct {
		Message string `json:"message"`
		Type    string `json:"type"`
		Code    string `json:"code"`
	} `json:"error"`
}

func (e *ErrorResponse) String() string {
	return fmt.Sprintf("openai responses error: %s (type: %s, code: %s)", e.Error.Message, e.Error.Type, e.Error.Code)
}

// New creates a new client to talk to the OpenAI Responses API.
//
// If apiKey is not provided, it tries to load it from the OPENAI_API_KEY environment variable.
// If none is found, it will still return a client coupled with an base.ErrAPIKeyRequired error.
// Get your API key at https://platform.openai.com/settings/organization/api-keys
//
// If no model is provided, only functions that do not require a model, like ListModels, will work.
// To use multiple models, create multiple clients.
// Use one of the model from https://platform.openai.com/docs/models
//
// Pass model base.PreferredCheap to use a good cheap model, base.PreferredGood for a good model or
// base.PreferredSOTA for a state-of-the-art model.
func New(apiKey, model string, wrapper func(http.RoundTripper) http.RoundTripper) (*Client, error) {
	const apiKeyURL = "https://platform.openai.com/settings/organization/api-keys"
	var err error
	if apiKey == "" {
		if apiKey = os.Getenv("OPENAI_API_KEY"); apiKey == "" {
			err = &base.ErrAPIKeyRequired{EnvVar: "OPENAI_API_KEY", URL: apiKeyURL}
		}
	}
	t := base.DefaultTransport
	if wrapper != nil {
		t = wrapper(t)
	}
	c := &Client{
		ProviderGen: base.ProviderGen[*ErrorResponse, *ResponseRequest, *ResponseResponse, ResponseStreamChunkResponse]{
			Model:                model,
			GenSyncURL:           "https://api.openai.com/v1/responses",
			ProcessStreamPackets: processStreamPackets,
			Provider: base.Provider[*ErrorResponse]{
				ProviderName: "openairesponses",
				APIKeyURL:    "", // OpenAI error message prints the api key URL already.
				ClientJSON: httpjson.Client{
					Client: &http.Client{
						Transport: &roundtrippers.Header{
							Header:    http.Header{"Authorization": {"Bearer " + apiKey}},
							Transport: &roundtrippers.RequestID{Transport: t},
						},
					},
				},
			},
		},
	}
	return c, err
}

// processStreamPackets processes stream packets for the OpenAI Responses API.
// This is a placeholder - will be implemented when GenStream is added.
func processStreamPackets(ch <-chan ResponseStreamChunkResponse, chunks chan<- genai.ContentFragment, result *genai.Result) error {
	// TODO: Implement when GenStream support is added
	return fmt.Errorf("streaming not yet implemented for OpenAI Responses API")
}

// ResponseStreamChunkResponse represents a streaming response chunk.
// This is a placeholder - will be implemented when GenStream is added.
type ResponseStreamChunkResponse struct {
	// TODO: Implement when GenStream support is added
}

// ResponseRequest represents a request to the OpenAI Responses API.
type ResponseRequest struct {
	Model              string            `json:"model"`
	Input              any               `json:"input"`
	Instructions       string            `json:"instructions,omitempty"`
	MaxOutputTokens    *int              `json:"max_output_tokens,omitempty"`
	Metadata           map[string]string `json:"metadata,omitempty"`
	ParallelToolCalls  *bool             `json:"parallel_tool_calls,omitempty"`
	PreviousResponseID string            `json:"previous_response_id,omitempty"`
	ServiceTier        string            `json:"service_tier,omitempty"`
	Store              *bool             `json:"store,omitempty"`
	Stream             bool              `json:"stream,omitempty"`
	Temperature        *float64          `json:"temperature,omitempty"`
	TopP               *float64          `json:"top_p,omitempty"`
	ToolChoice         any               `json:"tool_choice,omitempty"`
	Tools              []Tool            `json:"tools,omitempty"`
	User               string            `json:"user,omitempty"`
	Reasoning          *ReasoningConfig  `json:"reasoning,omitempty"`
}

// ReasoningConfig represents reasoning configuration for o-series models.
type ReasoningConfig struct {
	Effort  string `json:"effort,omitempty"`  // low, medium, high
	Summary string `json:"summary,omitempty"` // auto, concise, detailed
}

// Tool represents a tool that can be called by the model.
type Tool struct {
	Type                     string        `json:"type"`                       // function, file_search, etc.
	Function                 *FunctionTool `json:"function,omitempty"`         // for function tools
	FileSearchVectorStoreIDs []string      `json:"vector_store_ids,omitempty"` // for file_search tools
}

// FunctionTool represents a function tool definition.
type FunctionTool struct {
	Name        string         `json:"name"`
	Description string         `json:"description,omitempty"`
	Parameters  map[string]any `json:"parameters"`
	Strict      *bool          `json:"strict,omitempty"`
}

// InputMessage represents a message input to the model.
type InputMessage struct {
	Role    string      `json:"role"` // user, assistant, system, developer
	Content []InputItem `json:"content"`
	Type    string      `json:"type,omitempty"` // message
}

// InputItem represents different types of input content.
type InputItem struct {
	Type     string `json:"type"`                // input_text, input_image, input_file
	Text     string `json:"text,omitempty"`      // for input_text
	ImageURL string `json:"image_url,omitempty"` // for input_image
	FileID   string `json:"file_id,omitempty"`   // for input_image or input_file
	Detail   string `json:"detail,omitempty"`    // for input_image: high, low, auto
	FileData string `json:"file_data,omitempty"` // for input_file
	Filename string `json:"filename,omitempty"`  // for input_file
}

// Init implements base.InitializableRequest.
func (r *ResponseRequest) Init(msgs genai.Messages, opts genai.Options, model string) error {
	r.Model = model

	// Convert messages to the new Responses API format
	if len(msgs) == 0 {
		return fmt.Errorf("no messages provided")
	}

	// For simple text input, we can use string format
	// For complex cases with multiple messages, we need to use the message array format
	if len(msgs) == 1 && len(msgs[0].Contents) == 1 && msgs[0].Contents[0].Text != "" && msgs[0].Contents[0].Thinking == "" && len(msgs[0].Contents[0].Opaque) == 0 && msgs[0].Contents[0].Document == nil && msgs[0].Contents[0].URL == "" {
		// Simple text input
		r.Input = msgs[0].Contents[0].Text
	} else {
		// Complex input - convert to message array
		inputMsgs := make([]InputMessage, len(msgs))
		for i, msg := range msgs {
			inputMsg := InputMessage{
				Role: string(msg.Role),
				Type: "message",
			}

			// Convert contents
			inputMsg.Content = make([]InputItem, len(msg.Contents))
			for j, content := range msg.Contents {
				if content.Text != "" {
					inputMsg.Content[j] = InputItem{
						Type: "input_text",
						Text: content.Text,
					}
				} else if content.Document != nil {
					// Handle document content
					// TODO: Implement proper document/image content conversion
					inputMsg.Content[j] = InputItem{
						Type:   "input_image",
						Detail: "auto",
					}
				} else {
					return fmt.Errorf("unsupported content type")
				}
			}
			inputMsgs[i] = inputMsg
		}
		r.Input = inputMsgs
	}

	// Handle options if provided
	if opts != nil {
		switch v := opts.(type) {
		case *genai.OptionsText:
			if v.SystemPrompt != "" {
				r.Instructions = v.SystemPrompt
			}
			if v.MaxTokens > 0 {
				maxTokens := int(v.MaxTokens)
				r.MaxOutputTokens = &maxTokens
			}
			if v.Temperature > 0 {
				r.Temperature = &v.Temperature
			}
			if v.TopP > 0 {
				r.TopP = &v.TopP
			}
			// TODO: Handle tools conversion
		default:
			return fmt.Errorf("unsupported options type %T", opts)
		}
	}

	return nil
}

// SetStream implements base.InitializableRequest.
func (r *ResponseRequest) SetStream(stream bool) {
	r.Stream = stream
}

// ResponseResponse represents a response from the OpenAI Responses API.
type ResponseResponse struct {
	ID                 string             `json:"id"`
	Object             string             `json:"object"`
	CreatedAt          int64              `json:"created_at"`
	Model              string             `json:"model"`
	Background         *bool              `json:"background"`
	Error              *APIError          `json:"error"`
	IncompleteDetails  *IncompleteDetails `json:"incomplete_details"`
	Instructions       string             `json:"instructions"`
	MaxOutputTokens    *int               `json:"max_output_tokens"`
	Metadata           map[string]string  `json:"metadata"`
	Output             []OutputItem       `json:"output"`
	OutputText         string             `json:"output_text"` // SDK convenience property
	ParallelToolCalls  bool               `json:"parallel_tool_calls"`
	PreviousResponseID string             `json:"previous_response_id"`
	Reasoning          *ReasoningConfig   `json:"reasoning"`
	ServiceTier        string             `json:"service_tier"`
	Store              *bool              `json:"store"`
	Temperature        *float64           `json:"temperature"`
	TopP               *float64           `json:"top_p"`
	ToolChoice         any                `json:"tool_choice"`
	Tools              []Tool             `json:"tools"`
	Truncation         string             `json:"truncation"`
	Usage              *Usage             `json:"usage"`
	User               string             `json:"user"`
}

// APIError represents an API error in the response.
type APIError struct {
	Code    string `json:"code"`
	Message string `json:"message"`
}

// IncompleteDetails represents details about why a response is incomplete.
type IncompleteDetails struct {
	Reason string `json:"reason"`
}

// OutputItem represents different types of output content.
type OutputItem struct {
	Type    string          `json:"type"` // message, reasoning, function_call, etc.
	ID      string          `json:"id,omitempty"`
	Role    string          `json:"role,omitempty"`    // for message type
	Content []OutputContent `json:"content,omitempty"` // for message type
	Status  string          `json:"status,omitempty"`  // in_progress, completed, incomplete

	// For reasoning type
	Summary          []ReasoningSummary `json:"summary,omitempty"`
	EncryptedContent string             `json:"encrypted_content,omitempty"`

	// For function calls
	CallID string `json:"call_id,omitempty"`
	Name   string `json:"name,omitempty"`
	Output string `json:"output,omitempty"`
}

// OutputContent represents content within an output message.
type OutputContent struct {
	Type        string       `json:"type"` // output_text, refusal
	Text        string       `json:"text,omitempty"`
	Refusal     string       `json:"refusal,omitempty"`
	Annotations []Annotation `json:"annotations,omitempty"`
}

// Annotation represents annotations in output text.
type Annotation struct {
	Type       string `json:"type"` // file_citation, url_citation, etc.
	FileID     string `json:"file_id,omitempty"`
	URL        string `json:"url,omitempty"`
	Title      string `json:"title,omitempty"`
	StartIndex int    `json:"start_index,omitempty"`
	EndIndex   int    `json:"end_index,omitempty"`
}

// ReasoningSummary represents reasoning summary content.
type ReasoningSummary struct {
	Type string `json:"type"` // summary_text
	Text string `json:"text"`
}

// Usage represents token usage statistics.
type Usage struct {
	PromptTokens            int           `json:"prompt_tokens"`
	CompletionTokens        int           `json:"completion_tokens"`
	TotalTokens             int           `json:"total_tokens"`
	PromptTokensDetails     *TokenDetails `json:"prompt_tokens_details,omitempty"`
	CompletionTokensDetails *TokenDetails `json:"completion_tokens_details,omitempty"`
}

// TokenDetails provides detailed token usage breakdown.
type TokenDetails struct {
	CachedTokens    int `json:"cached_tokens,omitempty"`
	AudioTokens     int `json:"audio_tokens,omitempty"`
	ReasoningTokens int `json:"reasoning_tokens,omitempty"`
}

// ToResult implements base.ResultConverter.
func (r *ResponseResponse) ToResult() (genai.Result, error) {
	if r.Error != nil {
		return genai.Result{}, fmt.Errorf("API error: %s - %s", r.Error.Code, r.Error.Message)
	}

	result := genai.Result{}

	// Extract text content from output
	var textParts []string
	for _, output := range r.Output {
		if output.Type == "message" && output.Role == "assistant" {
			for _, content := range output.Content {
				if content.Type == "output_text" && content.Text != "" {
					textParts = append(textParts, content.Text)
				}
			}
		}
	}

	// Use convenience property if available and no text parts found
	if len(textParts) == 0 && r.OutputText != "" {
		textParts = append(textParts, r.OutputText)
	}

	if len(textParts) > 0 {
		// Join all text parts
		combinedText := textParts[0]
		for i := 1; i < len(textParts); i++ {
			combinedText += "\n" + textParts[i]
		}

		result.Contents = []genai.Content{
			{Text: combinedText},
		}
	}

	// Add usage information if available
	if r.Usage != nil {
		result.Usage = genai.Usage{
			InputTokens:  int64(r.Usage.PromptTokens),
			OutputTokens: int64(r.Usage.CompletionTokens),
		}
	}

	// Set finish reason based on status
	if r.IncompleteDetails != nil {
		result.FinishReason = genai.FinishedLength
	} else {
		result.FinishReason = genai.FinishedStop
	}

	return result, nil
}

// Scoreboard for OpenAI Responses API.
var Scoreboard = genai.Scoreboard{
	Country:      "US",
	DashboardURL: "https://platform.openai.com/usage",
	Scenarios: []genai.Scenario{
		{
			Models: []string{
				"gpt-4o",
				"gpt-4o-mini",
				"o1",
				"o1-mini",
				"o1-preview",
			},
			In: map[genai.Modality]genai.ModalCapability{
				genai.ModalityText: {Inline: true},
			},
			Out: map[genai.Modality]genai.ModalCapability{
				genai.ModalityText: {Inline: true},
			},
			GenSync: &genai.FunctionalityText{
				Thinking:   false,       // TODO: Check o-series model reasoning support
				Tools:      genai.False, // TODO: Implement tool support
				JSON:       false,       // TODO: Check JSON support
				JSONSchema: false,       // TODO: Check structured output support
				Seed:       false,       // TODO: Check if supported
			},
			GenStream: &genai.FunctionalityText{
				Thinking:   false,       // TODO: Check o-series model reasoning support
				Tools:      genai.False, // TODO: Implement tool support
				JSON:       false,       // TODO: Check JSON support
				JSONSchema: false,       // TODO: Check structured output support
				Seed:       false,       // TODO: Check if supported
			},
		},
	},
}

// Scoreboard implements genai.ProviderScoreboard.
func (c *Client) Scoreboard() genai.Scoreboard {
	return Scoreboard
}

// Interface compliance checks
var (
	_ genai.Provider           = &Client{}
	_ genai.ProviderGen        = &Client{}
	_ genai.ProviderScoreboard = &Client{}
)
