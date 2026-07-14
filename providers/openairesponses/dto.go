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
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"mime"
	"slices"
	"strings"

	"github.com/maruel/genai"
	"github.com/maruel/genai/base"
	"github.com/maruel/genai/internal"
	"github.com/maruel/genai/providers/openaibase"
)

//
// Shared types: aliases to openaibase.
//

// ServiceTier is the quality of service to determine the request's priority.
type ServiceTier = openaibase.ServiceTier

const (
	// ServiceTierAuto lets OpenAI choose the service tier.
	ServiceTierAuto = openaibase.ServiceTierAuto
	// ServiceTierDefault uses the default service tier.
	ServiceTierDefault = openaibase.ServiceTierDefault
	// ServiceTierFlex uses the flex service tier.
	ServiceTierFlex = openaibase.ServiceTierFlex
)

// ReasoningEffort is the effort the model should put into reasoning.
type ReasoningEffort = openaibase.ReasoningEffort

const (
	// ReasoningEffortNone disables reasoning effort.
	ReasoningEffortNone = openaibase.ReasoningEffortNone
	// ReasoningEffortMinimal requests minimal reasoning effort.
	ReasoningEffortMinimal = openaibase.ReasoningEffortMinimal
	// ReasoningEffortLow requests low reasoning effort.
	ReasoningEffortLow = openaibase.ReasoningEffortLow
	// ReasoningEffortMedium requests medium reasoning effort.
	ReasoningEffortMedium = openaibase.ReasoningEffortMedium
	// ReasoningEffortHigh requests high reasoning effort.
	ReasoningEffortHigh = openaibase.ReasoningEffortHigh
	// ReasoningEffortXHigh requests extra-high reasoning effort.
	ReasoningEffortXHigh = openaibase.ReasoningEffortXHigh
)

// Background is only supported on gpt-image-1.
type Background = openaibase.Background

const (
	// BackgroundAuto lets OpenAI choose the image background.
	BackgroundAuto = openaibase.BackgroundAuto
	// BackgroundTransparent requests a transparent image background.
	BackgroundTransparent = openaibase.BackgroundTransparent
	// BackgroundOpaque requests an opaque image background.
	BackgroundOpaque = openaibase.BackgroundOpaque
)

type (
	// ImageRequest is an alias to the shared OpenAI image request type.
	ImageRequest = openaibase.ImageRequest
	// ImageResponse is an alias to the shared OpenAI image response type.
	ImageResponse = openaibase.ImageResponse
	// ImageChoiceData is an alias to the shared OpenAI image choice type.
	ImageChoiceData = openaibase.ImageChoiceData
	// GenOptionImage is an alias to the shared OpenAI image generation options.
	GenOptionImage = openaibase.GenOptionImage
	// Model is an alias to the shared OpenAI model type.
	Model = openaibase.Model
	// ModelsResponse is an alias to the shared OpenAI models response type.
	ModelsResponse = openaibase.ModelsResponse
	// File is an alias to the shared OpenAI file type.
	File = openaibase.File
	// FileDeleteResponse is an alias to the shared OpenAI file deletion response type.
	FileDeleteResponse = openaibase.FileDeleteResponse
	// FileListResponse is an alias to the shared OpenAI file list response type.
	FileListResponse = openaibase.FileListResponse
	// BatchRequest is an alias to the shared OpenAI batch request type.
	BatchRequest = openaibase.BatchRequest
	// Batch is an alias to the shared OpenAI batch type.
	Batch = openaibase.Batch
	// BatchUsage is an alias to the shared OpenAI batch usage type.
	BatchUsage = openaibase.BatchUsage
	// ErrorResponse is an alias to the shared OpenAI error response type.
	ErrorResponse = openaibase.ErrorResponse
	// ErrorResponseError is an alias to the shared OpenAI error detail type.
	ErrorResponseError = openaibase.ErrorResponseError
)

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
	// Store enables server-side response storage (free, retained up to 30 days).
	// Required for previous_response_id to work. Enabled by default.
	Store            bool    `json:"store"`
	FrequencyPenalty float64 `json:"frequency_penalty,omitzero"`
	PresencePenalty  float64 `json:"presence_penalty,omitzero"`
	Temperature      float64 `json:"temperature,omitzero"`
	Text             struct {
		Format struct {
			Type        string           `json:"type"` // "text", "json_schema", "json_object"
			Name        string           `json:"name,omitzero"`
			Description string           `json:"description,omitzero"`
			Schema      genai.JSONSchema `json:"schema,omitzero"`
			Strict      bool             `json:"strict,omitzero"`
		} `json:"format"`
		Verbosity string `json:"verbosity,omitzero"` // "low", "medium", "high"
	} `json:"text,omitzero"`
	TopLogprobs int64     `json:"top_logprobs,omitzero"` // [0, 20]
	TopP        float64   `json:"top_p,omitzero"`
	ToolChoice  string    `json:"tool_choice,omitzero"` // "none", "auto", "required"
	ToolUsage   ToolUsage `json:"tool_usage,omitzero"`
	Truncation  string    `json:"truncation,omitzero"` // "disabled", "auto"
	Tools       []Tool    `json:"tools,omitzero"`
	User        string    `json:"user,omitzero"`    // Deprecated, use SafetyIdentifier and PromptCacheKey
	Include     []string  `json:"include,omitzero"` // "web_search_call.action.sources"

	// Request only
	Input  []Message `json:"input,omitzero"`
	Stream bool      `json:"stream,omitzero"`

	// Response only
	ID                string            `json:"id,omitzero"`
	Object            string            `json:"object,omitzero"` // "response"
	CreatedAt         base.TimeS        `json:"created_at,omitzero"`
	CompletedAt       base.TimeS        `json:"completed_at,omitzero"`
	Status            string            `json:"status,omitzero"` // "completed"
	IncompleteDetails IncompleteDetails `json:"incomplete_details,omitzero"`
	Error             APIError          `json:"error,omitzero"`
	Output            []Message         `json:"output,omitzero"`
	Usage             Usage             `json:"usage,omitzero"`
	Billing           map[string]string `json:"billing,omitzero"` // e.g. {"payer": "openai"}
	Moderation        string            `json:"moderation,omitzero"`
}

// Init implements base.InitializableRequest.
func (r *Response) Init(msgs genai.Messages, model string, opts ...genai.GenOption) error {
	var unsupported []string
	var errs []error
	r.Model = model
	r.Store = true
	r.Reasoning.Summary = "auto"
	for _, opt := range opts {
		if err := opt.Validate(); err != nil {
			return err
		}
		switch v := opt.(type) {
		case *GenOptionText:
			r.Reasoning.Effort = v.ReasoningEffort
			r.ServiceTier = v.ServiceTier
			r.Truncation = string(v.Truncation)
			r.PreviousResponseID = v.PreviousResponseID
		case *genai.GenOptionText:
			u, e := r.initOptionsText(v)
			unsupported = append(unsupported, u...)
			errs = append(errs, e...)
		case *genai.GenOptionTools:
			errs = append(errs, r.initOptionsTools(v)...)
		case *genai.GenOptionWeb:
			if v.Search {
				r.Tools = append(r.Tools, Tool{
					Type: "web_search",
					// SearchContextSize: "medium",
				})
				r.Include = []string{"web_search_call.action.sources"}
			}
			if v.Fetch {
				errs = append(errs, errors.New("unsupported GenOptionWeb.Fetch"))
			}
		default:
			return &base.ErrNotSupported{Options: []string{internal.TypeName(opt)}}
		}
	}
	if len(msgs) == 0 && r.PreviousResponseID == "" {
		return errors.New("no messages provided")
	}
	hasAudioVideoInput := openaibase.HasInputWithMimePrefix(msgs, "audio/") || openaibase.HasInputWithMimePrefix(msgs, "video/")
	if strings.HasPrefix(model, "gpt-5.6") && hasAudioVideoInput {
		unsupported = append(unsupported, "audio/video input")
	}

	for i := range msgs {
		// Each "Message" in OpenAI responses API is a content.
		switch {
		case len(msgs[i].ToolCallResults) > 1:
			// Handle messages with multiple tool call results by creating multiple messages
			for j := range msgs[i].ToolCallResults {
				// Create a copy of the message with only one tool call result
				msgCopy := msgs[i]
				msgCopy.ToolCallResults = []genai.ToolCallResult{msgs[i].ToolCallResults[j]}
				var newMsg Message
				if skip, err := newMsg.From(&msgCopy); err != nil {
					errs = append(errs, fmt.Errorf("message #%d: tool call results #%d: %w", i, j, err))
				} else if !skip {
					r.Input = append(r.Input, newMsg)
				}
			}
		case len(msgs[i].Replies) > 1:
			// Goddam OpenAI. Handle messages with multiple tool calls by creating multiple messages.
			var txt []genai.Reply
			for j := range msgs[i].Replies {
				if !msgs[i].Replies[j].ToolCall.IsZero() {
					msgCopy := msgs[i]
					msgCopy.Replies = []genai.Reply{msgs[i].Replies[j]}
					var newMsg Message
					if skip, err := newMsg.From(&msgCopy); err != nil {
						errs = append(errs, fmt.Errorf("message #%d: tool call #%d: %w", i, j, err))
					} else if !skip {
						r.Input = append(r.Input, newMsg)
					}
				} else {
					txt = append(txt, msgs[i].Replies[j])
				}
			}
			if len(txt) != 0 {
				// Create a copy of the message with only the non-tool call messages.
				msgCopy := msgs[i]
				msgCopy.Replies = txt
				var newMsg Message
				if skip, err := newMsg.From(&msgCopy); err != nil {
					errs = append(errs, fmt.Errorf("message #%d: %w", i, err))
				} else if !skip {
					r.Input = append(r.Input, newMsg)
				}
			}
		default:
			// It's a Request, send it as-is.
			var newMsg Message
			if skip, err := newMsg.From(&msgs[i]); err != nil {
				errs = append(errs, fmt.Errorf("message #%d: %w", i, err))
			} else if !skip {
				r.Input = append(r.Input, newMsg)
			}
		}
	}
	// If we have unsupported features but no other errors, return a structured error.
	if len(unsupported) > 0 && len(errs) == 0 {
		return &base.ErrNotSupported{Options: unsupported}
	}
	return errors.Join(errs...)
}

// SetStream implements base.InitializableRequest.
func (r *Response) SetStream(stream bool) {
	r.Stream = stream
}

// ToResult implements base.ResultConverter.
func (r *Response) ToResult() (genai.Result, error) {
	res := genai.Result{
		Usage: genai.Usage{
			InputTokens:       r.Usage.InputTokens,
			InputCachedTokens: r.Usage.InputTokensDetails.CachedTokens,
			ReasoningTokens:   r.Usage.OutputTokensDetails.ReasoningTokens,
			OutputTokens:      r.Usage.OutputTokens,
			TotalTokens:       r.Usage.TotalTokens,
			ServiceTier:       string(r.ServiceTier),
		},
	}
	for oi := range r.Output {
		if err := r.Output[oi].To(&res.Message); err != nil {
			return res, err
		}
		for i := range r.Output[oi].Content {
			for j := range r.Output[oi].Content[i].Logprobs {
				res.Logprobs = append(res.Logprobs, r.Output[oi].Content[i].Logprobs[j].To())
			}
		}
	}
	var err error
	hasRefusal := false
	for oi := range r.Output {
		for i := range r.Output[oi].Content {
			if r.Output[oi].Content[i].Type == ContentRefusal {
				hasRefusal = true
			}
		}
	}
	switch {
	case r.IncompleteDetails.Reason != "":
		if r.IncompleteDetails.Reason == "max_output_tokens" {
			res.Usage.FinishReason = genai.FinishedLength
		}
		err = errors.New(r.IncompleteDetails.Reason)
	case hasRefusal:
		res.Usage.FinishReason = genai.FinishedContentFilter
	case slices.ContainsFunc(res.Replies, func(r genai.Reply) bool { return !r.ToolCall.IsZero() }):
		res.Usage.FinishReason = genai.FinishedToolCalls
	default:
		res.Usage.FinishReason = genai.FinishedStop
	}
	return res, err
}

func (r *Response) initOptionsText(v *genai.GenOptionText) ([]string, []error) {
	var unsupported []string
	var errs []error
	r.MaxOutputTokens = v.MaxTokens
	r.Temperature = v.Temperature
	r.TopP = v.TopP
	if v.SystemPrompt != "" {
		r.Instructions = v.SystemPrompt
	}
	if v.TopK != 0 {
		unsupported = append(unsupported, "GenOptionText.TopK")
	}
	if v.TopLogprobs > 0 {
		r.TopLogprobs = v.TopLogprobs
	}
	if len(v.Stop) != 0 {
		errs = append(errs, errors.New("unsupported option Stop"))
	}
	if v.DecodeAs != nil {
		r.Text.Format.Type = "json_schema"
		// OpenAI requires a name.
		r.Text.Format.Name = "response"
		r.Text.Format.Strict = true
		s, err := v.DecodeSchema()
		if err != nil {
			errs = append(errs, err)
		} else {
			r.Text.Format.Schema = s
		}
	} else if v.ReplyAsJSON {
		r.Text.Format.Type = "json_object"
	}
	return unsupported, errs
}

func (r *Response) initOptionsTools(v *genai.GenOptionTools) []error {
	var errs []error
	if len(v.Tools) != 0 {
		r.ParallelToolCalls = true
		switch v.Force {
		case genai.ToolCallAny:
			r.ToolChoice = "auto"
		case genai.ToolCallRequired:
			r.ToolChoice = "required"
		case genai.ToolCallNone:
			r.ToolChoice = "none"
		}
		r.Tools = make([]Tool, len(v.Tools))
		for i, t := range v.Tools {
			if t.Name == "" {
				errs = append(errs, errors.New("tool name is required"))
			}
			r.Tools[i].Type = "function"
			r.Tools[i].Name = t.Name
			r.Tools[i].Description = t.Description
			s, err := t.GetInputSchema()
			if err != nil {
				errs = append(errs, err)
			}
			r.Tools[i].Parameters = s
		}
	}
	return errs
}

// ReasoningConfig represents reasoning configuration for o-series models.
type ReasoningConfig struct {
	Context string          `json:"context,omitzero"` // "current_turn"
	Effort  ReasoningEffort `json:"effort,omitzero"`
	Mode    string          `json:"mode,omitzero"`    // "standard", "pro"
	Summary string          `json:"summary,omitzero"` // "auto", "concise", "detailed"
}

// ToolUsage represents token usage for built-in tools.
type ToolUsage struct {
	ImageGen struct {
		InputTokens        int64 `json:"input_tokens"`
		InputTokensDetails struct {
			ImageTokens int64 `json:"image_tokens"`
			TextTokens  int64 `json:"text_tokens"`
		} `json:"input_tokens_details"`
		OutputTokens        int64 `json:"output_tokens"`
		OutputTokensDetails struct {
			ImageTokens int64 `json:"image_tokens"`
			TextTokens  int64 `json:"text_tokens"`
		} `json:"output_tokens_details"`
		TotalTokens int64 `json:"total_tokens"`
	} `json:"image_gen,omitzero"`
	WebSearch struct {
		NumRequests int64 `json:"num_requests"`
	} `json:"web_search,omitzero"`
}

// Tool represents a tool that can be called by the model.
type Tool struct {
	// "function", "file_search", "computer_use_preview", "mcp", "code_interpreter", "image_generation",
	// "local_shell", "web_search"
	Type string `json:"type,omitzero"`

	// Type == "function"
	Name              string           `json:"name,omitzero"`
	Description       string           `json:"description,omitzero"`
	OutputSchema      genai.JSONSchema `json:"output_schema,omitzero"`
	Parameters        genai.JSONSchema `json:"parameters,omitzero"`
	Strict            bool             `json:"strict,omitzero"`
	ReturnTokenBudget string           `json:"return_token_budget,omitzero"` // "default"

	// Type == "file_search"
	FileSearchVectorStoreIDs []string `json:"vector_store_ids,omitzero"`

	// Type == "web_search"
	Filters struct {
		AllowedDomains []string `json:"allowed_domains,omitzero"`
	} `json:"filters,omitzero"`
	SearchContentTypes []string `json:"search_content_types,omitzero"`
	SearchContextSize  string   `json:"search_context_size,omitzero"` // "low", "medium", "high"
	UserLocation       struct {
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

	// Type == MessageMessage
	Phase string `json:"phase,omitzero"` // "final_answer"

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
		Type    string   `json:"type,omitzero"` // "search"
		Queries []string `json:"queries,omitzero"`
		Query   string   `json:"query,omitzero"`
		Sources []struct {
			Type string `json:"type,omitzero"` // "url"
			URL  string `json:"url,omitzero"`
		} `json:"sources,omitzero"`
	} `json:"action,omitzero"`
}

// From must be called with at most one ToolCallResults.
func (m *Message) From(in *genai.Message) (bool, error) {
	if len(in.ToolCallResults) > 1 {
		return false, &internal.BadError{Err: errors.New("internal error")}
	}
	if len(in.ToolCallResults) != 0 {
		// Handle multiple tool call results by creating multiple messages
		// The caller (Init method) should handle this by creating separate messages
		m.Type = MessageFunctionCallOutput
		m.CallID = in.ToolCallResults[0].ID
		m.Output = in.ToolCallResults[0].Result
		return false, nil
	}
	if len(in.Requests) != 0 {
		m.Type = MessageMessage
		m.Role = "user"
		m.Content = make([]Content, len(in.Requests))
		for j := range in.Requests {
			if err := m.Content[j].FromRequest(&in.Requests[j]); err != nil {
				return false, fmt.Errorf("request #%d: %w", j, err)
			}
		}
		return len(m.Content) == 0, nil
	}
	if len(in.Replies) != 0 {
		// Handle multiple tool calls by creating multiple messages
		// The caller (Init method) should handle this by creating separate messages
		if !in.Replies[0].ToolCall.IsZero() {
			if len(in.Replies[0].ToolCall.Opaque) != 0 {
				return false, &internal.BadError{Err: errors.New("field ToolCall.Opaque not supported")}
			}
			m.Type = MessageFunctionCall
			m.CallID = in.Replies[0].ToolCall.ID
			m.Name = in.Replies[0].ToolCall.Name
			m.Arguments = in.Replies[0].ToolCall.Arguments
			return false, nil
		}
		m.Type = MessageMessage
		m.Role = "assistant"
		for j := range in.Replies {
			// TODO: should we send it back, at least the ID?
			if in.Replies[j].Reasoning != "" {
				continue
			}
			m.Content = append(m.Content, Content{})
			if err := m.Content[len(m.Content)-1].FromReply(&in.Replies[j]); err != nil {
				return false, fmt.Errorf("reply #%d: %w", j, err)
			}
		}
		return len(m.Content) == 0, nil
	}
	return false, &internal.BadError{Err: fmt.Errorf("implement message: %#v", in)}
}

// To is different here because it can be called multiple times on the same out.
//
// In the Responses API, Message is actually a mix of Message and Content.
func (m *Message) To(out *genai.Message) error {
	// We only need to implement the types that can be returned from the LLM.
	switch m.Type {
	case MessageMessage:
		for i := range m.Content {
			replies, err := m.Content[i].To()
			if err != nil {
				return fmt.Errorf("reply %d: %w", i, err)
			}
			out.Replies = append(out.Replies, replies...)
		}
	case MessageReasoning:
		for i := range m.Summary {
			if m.Summary[i].Type != "summary_text" {
				return &internal.BadError{Err: fmt.Errorf("implement summary type %q", m.Summary[i].Type)}
			}
			out.Replies = append(out.Replies, genai.Reply{Reasoning: m.Summary[i].Text})
		}
	case MessageFunctionCall:
		out.Replies = append(out.Replies, genai.Reply{ToolCall: genai.ToolCall{ID: m.CallID, Name: m.Name, Arguments: m.Arguments}})
	case MessageWebSearchCall:
		if m.Action.Type != "search" {
			return &internal.BadError{Err: fmt.Errorf("implement action type %q", m.Action.Type)}
		}
		c := genai.Citation{Sources: make([]genai.CitationSource, len(m.Action.Sources)+1)}
		c.Sources[0].Type = genai.CitationWebQuery
		c.Sources[0].Snippet = m.Action.Query
		for i, src := range m.Action.Sources {
			c.Sources[i+1].Type = genai.CitationWeb
			c.Sources[i+1].URL = src.URL
		}
		out.Replies = append(out.Replies, genai.Reply{Citation: c})
	case MessageFileSearchCall:
		for _, q := range m.Queries {
			out.Replies = append(out.Replies, genai.Reply{Citation: genai.Citation{
				Sources: []genai.CitationSource{{Type: genai.CitationWebQuery, Snippet: q}},
			}})
		}
		for _, r := range m.Results {
			out.Replies = append(out.Replies, genai.Reply{Citation: genai.Citation{
				CitedText: r.Text,
				Sources: []genai.CitationSource{{
					Type:  genai.CitationDocument,
					ID:    r.FileID,
					Title: r.Filename,
				}},
			}})
		}
	case MessageComputerCall, MessageImageGenerationCall, MessageCodeInterpreterCall, MessageLocalShellCall, MessageMcpListTools, MessageMcpApprovalRequest, MessageMcpCall, MessageComputerCallOutput, MessageFunctionCallOutput, MessageLocalShellCallOutput, MessageMcpApprovalResponse, MessageItemReference:
		return &internal.BadError{Err: fmt.Errorf("unsupported output type %q", m.Type)}
	default:
		return &internal.BadError{Err: fmt.Errorf("unsupported output type %q", m.Type)}
	}
	return nil
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

// To converts to the genai equivalent.
func (c *Content) To() ([]genai.Reply, error) {
	var out []genai.Reply
	for _, a := range c.Annotations {
		var ci genai.Citation
		switch a.Type {
		case "url_citation":
			ci = genai.Citation{
				StartIndex: a.StartIndex,
				EndIndex:   a.EndIndex,
				Sources:    []genai.CitationSource{{Type: genai.CitationWeb, URL: a.URL, Title: a.Title}},
			}
		case "file_citation", "container_file_citation":
			ci = genai.Citation{
				StartIndex: a.StartIndex,
				EndIndex:   a.EndIndex,
				Sources:    []genai.CitationSource{{Type: genai.CitationDocument, ID: a.FileID}},
			}
		case "file_path":
			ci = genai.Citation{
				Sources: []genai.CitationSource{{Type: genai.CitationDocument, ID: a.FileID}},
			}
		default:
			return out, &internal.BadError{Err: fmt.Errorf("unsupported annotation type %q", a.Type)}
		}
		out = append(out, genai.Reply{Citation: ci})
	}
	switch c.Type {
	case ContentOutputText:
		out = append(out, genai.Reply{Text: c.Text})
	case ContentRefusal:
		// Surface refusal as text so the caller can see the reason.
		out = append(out, genai.Reply{Text: c.Refusal})
	case ContentInputText, ContentInputImage, ContentInputFile:
		return out, &internal.BadError{Err: fmt.Errorf("implement content type %q", c.Type)}
	default:
		return out, &internal.BadError{Err: fmt.Errorf("implement content type %q", c.Type)}
	}
	return out, nil
}

// FromRequest converts from a genai request.
func (c *Content) FromRequest(in *genai.Request) error {
	if in.Text != "" {
		c.Type = ContentInputText
		c.Text = in.Text
		return nil
	}
	if !in.Doc.IsZero() {
		// https://platform.openai.com/docs/guides/images?api-mode=chat&format=base64-encoded#image-input-requirements
		mimeType, data, err := in.Doc.Read(10 * 1024 * 1024)
		if err != nil {
			return err
		}
		// OpenAI require a mime-type to determine if image, sound or PDF.
		if mimeType == "" {
			return fmt.Errorf("unspecified mime type for URL %q", in.Doc.URL)
		}
		switch {
		case strings.HasPrefix(mimeType, "image/"):
			c.Type = ContentInputImage
			c.Detail = "auto" // TODO: Make it configurable.
			if in.Doc.URL == "" {
				c.ImageURL = fmt.Sprintf("data:%s;base64,%s", mimeType, base64.StdEncoding.EncodeToString(data))
			} else {
				c.ImageURL = in.Doc.URL
			}
			// text/plain, text/markdown
		case strings.HasPrefix(mimeType, "text/"):
			// OpenAI responses API doesn't support text documents as attachment.
			c.Type = ContentInputText
			if in.Doc.URL != "" {
				return fmt.Errorf("%s documents must be provided inline, not as a URL", mimeType)
			}
			c.Text = string(data)
		default:
			if in.Doc.URL != "" {
				return fmt.Errorf("URL to %s file not supported", mimeType)
			}
			filename := in.Doc.GetFilename()
			if filename == "" {
				exts, err := mime.ExtensionsByType(mimeType)
				if err != nil {
					return err
				}
				if len(exts) == 0 {
					return fmt.Errorf("unknown extension for mime type %s", mimeType)
				}
				filename = "content" + exts[0]
			}
			c.Type = ContentInputFile
			c.Filename = filename
			c.FileData = fmt.Sprintf("data:%s;base64,%s", mimeType, base64.StdEncoding.EncodeToString(data))
		}
		return nil
	}
	return errors.New("unknown Request type")
}

// FromReply converts from a genai reply.
func (c *Content) FromReply(in *genai.Reply) error {
	if len(in.Opaque) != 0 {
		return &internal.BadError{Err: errors.New("field Reply.Opaque not supported")}
	}
	if in.Text != "" {
		c.Type = ContentInputText
		c.Text = in.Text
		return nil
	}
	if !in.Doc.IsZero() {
		// https://platform.openai.com/docs/guides/images?api-mode=chat&format=base64-encoded#image-input-requirements
		mimeType, data, err := in.Doc.Read(10 * 1024 * 1024)
		if err != nil {
			return err
		}
		// OpenAI require a mime-type to determine if image, sound or PDF.
		if mimeType == "" {
			return fmt.Errorf("unspecified mime type for URL %q", in.Doc.URL)
		}
		switch {
		case strings.HasPrefix(mimeType, "image/"):
			c.Type = ContentInputImage
			c.Detail = "auto" // TODO: Make it configurable.
			if in.Doc.URL == "" {
				c.ImageURL = fmt.Sprintf("data:%s;base64,%s", mimeType, base64.StdEncoding.EncodeToString(data))
			} else {
				c.ImageURL = in.Doc.URL
			}
			// text/plain, text/markdown
		case strings.HasPrefix(mimeType, "text/"):
			// OpenAI responses API doesn't support text documents as attachment.
			c.Type = ContentInputText
			if in.Doc.URL != "" {
				return fmt.Errorf("%s documents must be provided inline, not as a URL", mimeType)
			}
			c.Text = string(data)
		default:
			if in.Doc.URL != "" {
				return fmt.Errorf("URL to %s file not supported", mimeType)
			}
			filename := in.Doc.GetFilename()
			if filename == "" {
				exts, err := mime.ExtensionsByType(mimeType)
				if err != nil {
					return err
				}
				if len(exts) == 0 {
					return fmt.Errorf("unknown extension for mime type %s", mimeType)
				}
				filename = "content" + exts[0]
			}
			c.Type = ContentInputFile
			c.Filename = filename
			c.FileData = fmt.Sprintf("data:%s;base64,%s", mimeType, base64.StdEncoding.EncodeToString(data))
		}
		return nil
	}
	return &internal.BadError{Err: errors.New("unknown Reply type")}
}

// APIError represents an API error in the response.
type APIError struct {
	Code    string `json:"code"` // "server_error"
	Message string `json:"message"`
}

func (e *APIError) Error() string {
	return fmt.Sprintf("%s: %s", e.Code, e.Message)
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

// To converts to the genai equivalent.
func (l *Logprobs) To() []genai.Logprob {
	out := make([]genai.Logprob, 1, len(l.TopLogprobs)+1)
	// Intentionally discard Bytes.
	out[0] = genai.Logprob{Text: l.Token, Logprob: l.Logprob}
	for _, tlp := range l.TopLogprobs {
		out = append(out, genai.Logprob{Text: tlp.Token, Logprob: tlp.Logprob})
	}
	return out
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
		CachedTokens     int64 `json:"cached_tokens"`
		CacheWriteTokens int64 `json:"cache_write_tokens"`
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
	ResponseKeepalive                       ResponseType = "keepalive"
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
// WebSocket types.
//

// WSRequest is a WebSocket message to create a response.
//
// It adds the "type" field required by the WebSocket protocol.
// The Stream and Background fields are excluded as they are not used in WebSocket mode.
// Response-only fields (ID, Status, Output, etc.) are naturally omitted by
// omitzero since they are zero-valued on outbound requests.
//
// https://developers.openai.com/api/docs/guides/websocket-mode
type WSRequest struct {
	Type string `json:"type"` // Always "response.create"
	Response
}

// MarshalJSON implements json.Marshaler to exclude Stream and Background.
func (r *WSRequest) MarshalJSON() ([]byte, error) {
	// Use an alias to avoid infinite recursion.
	type alias Response
	return json.Marshal(struct {
		Type string `json:"type"`
		alias
		Stream     bool `json:"-"`
		Background bool `json:"-"`
	}{
		Type:  r.Type,
		alias: alias(r.Response),
	})
}

//
// Batch API types (API-specific; use Response as body).
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
