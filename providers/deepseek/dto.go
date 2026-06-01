// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Wire types for the DeepSeek chat completion API.
//
// See https://api-docs.deepseek.com/api/create-chat-completion

package deepseek

import (
	"errors"
	"fmt"
	"strings"

	"github.com/invopop/jsonschema"

	"github.com/maruel/genai"
	"github.com/maruel/genai/base"
	"github.com/maruel/genai/internal"
)

// GenOption controls thinking mode and reasoning effort.
//
// DeepSeek v4 models default to thinking enabled. Set Thinking to false to
// switch to non-thinking mode, which enables forced tool calling and avoids
// "reasoning_content must be passed back" errors across multi-turn
// conversations.
//
// See https://api-docs.deepseek.com/api/create-chat-completion
type GenOption struct {
	// Thinking controls the thinking mode. When false, thinking is disabled.
	Thinking bool
	// ReasoningEffort controls the reasoning effort of the model.
	// Ignored when Thinking is false. Default is "high".
	// For compatibility, "low" and "medium" are mapped to "high" by the API,
	// and "xhigh" is mapped to "max".
	ReasoningEffort ReasoningEffort
}

// ReasoningEffort controls the amount of effort the model puts into reasoning.
type ReasoningEffort string

const (
	// ReasoningEffortHigh is the default effort for regular requests.
	ReasoningEffortHigh ReasoningEffort = "high"
	// ReasoningEffortMax maximizes reasoning quality.
	ReasoningEffortMax ReasoningEffort = "max"
)

// Validate implements genai.Validatable.
func (g *GenOption) Validate() error {
	switch g.ReasoningEffort {
	case "", ReasoningEffortHigh, ReasoningEffortMax:
	default:
		return fmt.Errorf("invalid ReasoningEffort %q", g.ReasoningEffort)
	}
	return nil
}

// ChatRequest is documented at https://api-docs.deepseek.com/api/create-chat-completion
type ChatRequest struct {
	Model    string    `json:"model"`
	Messages []Message `json:"messages"`
	Stream   bool      `json:"stream"`
	// Thinking controls the switch between thinking and non-thinking mode.
	// Set type to "disabled" for non-thinking mode. Default is "enabled".
	Thinking struct {
		Type            string `json:"type,omitzero"`             // "enabled", "disabled"
		ReasoningEffort string `json:"reasoning_effort,omitzero"` // "high", "max"
	} `json:"thinking,omitzero"`
	Temperature    float64 `json:"temperature,omitzero"` // [0, 2]
	MaxToks        int64   `json:"max_tokens,omitzero"`  // Up to 384K for deepseek-v4-flash
	ResponseFormat struct {
		Type string `json:"type,omitzero"` // "text", "json_object"
	} `json:"response_format,omitzero"`
	Stop          []string `json:"stop,omitzero"`
	StreamOptions struct {
		IncludeUsage bool `json:"include_usage,omitzero"`
	} `json:"stream_options,omitzero"`
	TopP       float64 `json:"top_p,omitzero"`       // [0, 1]
	ToolChoice string  `json:"tool_choice,omitzero"` // "none", "auto", "required"
	Tools      []Tool  `json:"tools,omitzero"`
	Logprobs   bool    `json:"logprobs,omitzero"`
	TopLogprob int64   `json:"top_logprobs,omitzero"`
	// UserID is a custom user identifier for rate limit isolation.
	// Allowed charset: [a-zA-Z0-9\-_], max 512 chars.
	// See https://api-docs.deepseek.com/quick_start/rate_limit
	UserID string `json:"user_id,omitzero"`
}

// Init initializes the provider specific completion request with the generic completion request.
func (c *ChatRequest) Init(msgs genai.Messages, model string, opts ...genai.GenOption) error {
	c.Model = model
	if err := msgs.Validate(); err != nil {
		return err
	}
	var errs []error
	var unsupported []string
	sp := ""
	for _, opt := range opts {
		if err := opt.Validate(); err != nil {
			return err
		}
		// https://api-docs.deepseek.com/guides/reasoning_model Soon "reasoning_effort"
		switch v := opt.(type) {
		case *GenOption:
			if !v.Thinking {
				c.Thinking.Type = "disabled"
			} else if v.ReasoningEffort != "" {
				c.Thinking.ReasoningEffort = string(v.ReasoningEffort)
			}
		case *genai.GenOptionText:
			c.MaxToks = v.MaxTokens
			c.Temperature = v.Temperature
			c.TopP = v.TopP
			sp = v.SystemPrompt
			if v.TopLogprobs > 0 {
				c.TopLogprob = v.TopLogprobs
				c.Logprobs = true
			}
			if v.TopK != 0 {
				unsupported = append(unsupported, "GenOptionText.TopK")
			}
			c.Stop = v.Stop
			if v.ReplyAsJSON {
				c.ResponseFormat.Type = "json_object"
			}
			if v.DecodeAs != nil {
				errs = append(errs, errors.New("unsupported option DecodeAs"))
			}
		case *genai.GenOptionTools:
			if len(v.Tools) != 0 {
				switch v.Force {
				case genai.ToolCallAny:
					c.ToolChoice = "auto"
				case genai.ToolCallRequired:
					c.ToolChoice = "required"
				case genai.ToolCallNone:
					c.ToolChoice = "none"
				}
				c.Tools = make([]Tool, len(v.Tools))
				for i, t := range v.Tools {
					c.Tools[i].Type = "function"
					c.Tools[i].Function.Name = t.Name
					c.Tools[i].Function.Description = t.Description
					c.Tools[i].Function.Strict = true
					if c.Tools[i].Function.Parameters = t.InputSchemaOverride; c.Tools[i].Function.Parameters == nil {
						c.Tools[i].Function.Parameters = t.GetInputSchema()
					}
				}
			}
		default:
			unsupported = append(unsupported, internal.TypeName(opt))
		}
	}

	if sp != "" {
		c.Messages = append(c.Messages, Message{Role: "system", Content: sp})
	}
	for i := range msgs {
		// Split messages into multiple messages as needed.
		switch {
		case len(msgs[i].ToolCallResults) > 1:
			// Handle messages with multiple tool call results by creating multiple messages
			for j := range msgs[i].ToolCallResults {
				// Create a copy of the message with only one tool call result
				msgCopy := msgs[i]
				msgCopy.ToolCallResults = []genai.ToolCallResult{msgs[i].ToolCallResults[j]}
				var newMsg Message
				if err := newMsg.From(&msgCopy); err != nil {
					errs = append(errs, fmt.Errorf("message #%d, tool call results #%d: %w", i, j, err))
				} else {
					c.Messages = append(c.Messages, newMsg)
				}
			}
		case len(msgs[i].Requests) > 1:
			// Handle messages with multiple Request by creating multiple messages
			for j := range msgs[i].Requests {
				// Create a copy of the message with only one request
				msgCopy := msgs[i]
				msgCopy.Requests = []genai.Request{msgs[i].Requests[j]}
				var newMsg Message
				if err := newMsg.From(&msgCopy); err != nil {
					errs = append(errs, fmt.Errorf("message #%d, request #%d: %w", i, j, err))
				} else {
					c.Messages = append(c.Messages, newMsg)
				}
			}
		default:
			var newMsg Message
			if err := newMsg.From(&msgs[i]); err != nil {
				errs = append(errs, fmt.Errorf("message #%d: %w", i, err))
			} else {
				c.Messages = append(c.Messages, newMsg)
			}
		}
	}
	// If we have unsupported features but no other errors, return a structured error.
	if len(unsupported) > 0 && len(errs) == 0 {
		return &base.ErrNotSupported{Options: unsupported}
	}
	return errors.Join(errs...)
}

// SetStream sets the streaming mode.
func (c *ChatRequest) SetStream(stream bool) {
	c.Stream = stream
}

// Message is documented at https://api-docs.deepseek.com/api/create-chat-completion
type Message struct {
	Role             string     `json:"role,omitzero"` // "system", "assistant", "user"
	Name             string     `json:"name,omitzero"` // An optional name for the participant. Provides the model information to differentiate between participants of the same role.
	Content          string     `json:"content,omitzero"`
	Prefix           bool       `json:"prefix,omitzero"` // Force the model to start its answer by the content of the supplied prefix in this assistant message.
	ReasoningContent string     `json:"reasoning_content,omitzero"`
	ToolCalls        []ToolCall `json:"tool_calls,omitzero"`
	ToolCallID       string     `json:"tool_call_id,omitzero"` // Tool call that this message is responding to, with response in Content field.
}

// From must be called with at most one Request or one ToolCallResults.
func (m *Message) From(in *genai.Message) error {
	if len(in.Requests) > 1 || len(in.ToolCallResults) > 1 {
		return errors.New("internal error")
	}
	switch r := in.Role(); r {
	case "user", "assistant":
		m.Role = r
	case "computer":
		m.Role = "tool"
	default:
		return fmt.Errorf("unsupported role %q", r)
	}
	m.Name = in.User
	if len(in.Requests) == 1 {
		switch {
		case in.Requests[0].Text != "":
			m.Content += in.Requests[0].Text
		case !in.Requests[0].Doc.IsZero():
			mimeType, data, err := in.Requests[0].Doc.Read(10 * 1024 * 1024)
			if err != nil {
				return fmt.Errorf("failed to read document: %w", err)
			}
			if !strings.HasPrefix(mimeType, "text/") {
				return fmt.Errorf("deepseek only supports text documents, got %s", mimeType)
			}
			if in.Requests[0].Doc.URL != "" {
				return fmt.Errorf("%s documents must be provided inline, not as a URL", mimeType)
			}
			m.Content += string(data)
		default:
			return errors.New("unknown Request type")
		}
		return nil
	}
	for i := range in.Replies {
		if len(in.Replies[i].Opaque) != 0 {
			return fmt.Errorf("reply #%d: field Reply.Opaque not supported", i)
		}
		switch {
		case in.Replies[i].Text != "":
			m.Content += in.Replies[i].Text
		case !in.Replies[i].Doc.IsZero():
			mimeType, data, err := in.Replies[i].Doc.Read(10 * 1024 * 1024)
			if err != nil {
				return fmt.Errorf("reply #%d: failed to read document: %w", i, err)
			}
			if in.Replies[i].Doc.URL != "" {
				return fmt.Errorf("reply #%d: deepseek doesn't support document content blocks with URLs", i)
			}
			if !strings.HasPrefix(mimeType, "text/") {
				return fmt.Errorf("reply #%d: deepseek only supports text documents, got %s", 0, mimeType)
			}
			m.Content += string(data)
		case in.Replies[i].Reasoning != "":
			// DeepSeek requires reasoning_content to be passed back in
			// multi-turn conversations when thinking mode is enabled.
			// Otherwise the API rejects with "reasoning_content must
			// be passed back to the API".
			m.ReasoningContent += in.Replies[i].Reasoning
		case !in.Replies[i].ToolCall.IsZero():
			m.ToolCalls = append(m.ToolCalls, ToolCall{})
			if err := m.ToolCalls[len(m.ToolCalls)-1].From(&in.Replies[i].ToolCall); err != nil {
				return fmt.Errorf("reply #%d: %w", i, err)
			}
		default:
			return errors.New("unknown Reply type")
		}
	}
	if len(in.ToolCallResults) != 0 {
		// Process only the first tool call result in this method.
		// The Init method handles multiple tool call results by creating multiple messages.
		m.Content = in.ToolCallResults[0].Result
		m.ToolCallID = in.ToolCallResults[0].ID
	}
	return nil
}

// To converts to the genai equivalent.
func (m *Message) To(out *genai.Message) {
	// Both ReasoningContent and Content can be set on the same reply.
	if m.ReasoningContent != "" {
		out.Replies = append(out.Replies, genai.Reply{Reasoning: m.ReasoningContent})
	}
	if m.Content != "" {
		out.Replies = append(out.Replies, genai.Reply{Text: m.Content})
	}
	for i := range m.ToolCalls {
		out.Replies = append(out.Replies, genai.Reply{})
		m.ToolCalls[i].To(&out.Replies[len(out.Replies)-1].ToolCall)
	}
}

// ToolCall is a provider-specific tool call.
type ToolCall struct {
	Index    int64  `json:"index,omitzero"`
	ID       string `json:"id,omitzero"`
	Type     string `json:"type,omitzero"` // "function"
	Function struct {
		Name      string `json:"name,omitzero"`
		Arguments string `json:"arguments,omitzero"`
	} `json:"function,omitzero"`
}

// From converts from the genai equivalent.
func (t *ToolCall) From(in *genai.ToolCall) error {
	if len(in.Opaque) != 0 {
		return errors.New("field ToolCall.Opaque not supported")
	}
	t.Type = "function"
	t.ID = in.ID
	t.Function.Name = in.Name
	t.Function.Arguments = in.Arguments
	return nil
}

// To converts to the genai equivalent.
func (t *ToolCall) To(out *genai.ToolCall) {
	out.ID = t.ID
	out.Name = t.Function.Name
	out.Arguments = t.Function.Arguments
}

// Tool is a provider-specific tool definition.
type Tool struct {
	Type     string `json:"type"` // "function"
	Function struct {
		Name        string             `json:"name,omitzero"`
		Description string             `json:"description,omitzero"`
		Parameters  *jsonschema.Schema `json:"parameters,omitzero"`
		Strict      bool               `json:"strict,omitzero"` // Beta: strict JSON schema validation
	} `json:"function"`
}

// ChatResponse is the provider-specific chat completion response.
type ChatResponse struct {
	ID      string `json:"id"`
	Choices []struct {
		FinishReason FinishReason `json:"finish_reason"`
		Index        int64        `json:"index"`
		Message      Message      `json:"message"`
		Logprobs     Logprobs     `json:"logprobs"`
	} `json:"choices"`
	Created           int64  `json:"created"` // Unix timestamp
	Model             string `json:"model"`
	SystemFingerPrint string `json:"system_fingerprint"`
	Object            string `json:"object"` // chat.completion
	Usage             Usage  `json:"usage"`
}

// ToResult converts the response to a genai.Result.
func (c *ChatResponse) ToResult() (genai.Result, error) {
	out := genai.Result{
		Usage: genai.Usage{
			InputTokens:       c.Usage.PromptTokens,
			InputCachedTokens: c.Usage.PromptCacheHitTokens,
			ReasoningTokens:   c.Usage.ChatTokensDetails.ReasoningTokens,
			OutputTokens:      c.Usage.CompletionTokens,
		},
	}
	if len(c.Choices) != 1 {
		return out, fmt.Errorf("expected 1 choice, got %#v", c.Choices)
	}
	out.Usage.FinishReason = c.Choices[0].FinishReason.ToFinishReason()
	c.Choices[0].Message.To(&out.Message)
	out.Logprobs = c.Choices[0].Logprobs.To()
	return out, nil
}

// FinishReason is a provider-specific finish reason.
type FinishReason string

// Finish reason values.
const (
	FinishStop          FinishReason = "stop"
	FinishToolCalls     FinishReason = "tool_calls"
	FinishLength        FinishReason = "length"
	FinishContentFilter FinishReason = "content_filter"
	FinishInsufficient  FinishReason = "insufficient_system_resource"
)

// ToFinishReason converts to a genai.FinishReason.
func (f FinishReason) ToFinishReason() genai.FinishReason {
	switch f {
	case FinishStop:
		return genai.FinishedStop
	case FinishToolCalls:
		return genai.FinishedToolCalls
	case FinishLength:
		return genai.FinishedLength
	case FinishContentFilter:
		return genai.FinishedContentFilter
	case FinishInsufficient:
		if !internal.BeLenient {
			panic(f)
		}
		return genai.FinishReason(f)
	default:
		if !internal.BeLenient {
			panic(f)
		}
		return genai.FinishReason(f)
	}
}

// Usage is the provider-specific token usage.
type Usage struct {
	CompletionTokens      int64 `json:"completion_tokens"`
	PromptTokens          int64 `json:"prompt_tokens"`
	PromptCacheHitTokens  int64 `json:"prompt_cache_hit_tokens"`
	PromptCacheMissTokens int64 `json:"prompt_cache_miss_tokens"`
	TotalTokens           int64 `json:"total_tokens"`
	PromptTokensDetails   struct {
		CachedTokens int64 `json:"cached_tokens"`
	} `json:"prompt_tokens_details"`
	ChatTokensDetails struct {
		ReasoningTokens int64 `json:"reasoning_tokens"`
	} `json:"completion_tokens_details"`
}

// Logprobs is the provider-specific log probabilities.
type Logprobs struct {
	Content          []LogprobEntry `json:"content,omitzero"`
	ReasoningContent []LogprobEntry `json:"reasoning_content,omitzero"`
}

// LogprobEntry is a single log probability entry.
type LogprobEntry struct {
	Token       string  `json:"token"`
	Bytes       []byte  `json:"bytes"`
	Logprob     float64 `json:"logprob"`
	TopLogprobs []struct {
		Token   string  `json:"token"`
		Bytes   []byte  `json:"bytes"`
		Logprob float64 `json:"logprob"`
	} `json:"top_logprobs"`
}

// To converts to the genai equivalent.
func (l *Logprobs) To() [][]genai.Logprob {
	if len(l.Content) == 0 {
		// TODO: Handle ReasoningContent logprobs.
		return nil
	}
	out := make([][]genai.Logprob, 0, len(l.Content))
	for _, c := range l.Content {
		lp := make([]genai.Logprob, 1, len(c.TopLogprobs)+1)
		// Intentionally discard Bytes.
		lp[0] = genai.Logprob{Text: c.Token, Logprob: c.Logprob}
		for _, tlp := range c.TopLogprobs {
			lp = append(lp, genai.Logprob{Text: tlp.Token, Logprob: tlp.Logprob})
		}
		out = append(out, lp)
	}
	return out
}

// ChatStreamChunkResponse is the provider-specific streaming chat chunk.
type ChatStreamChunkResponse struct {
	ID                string `json:"id"`
	Object            string `json:"object"`  // chat.completion.chunk
	Created           int64  `json:"created"` // Unix timestamp
	Model             string `json:"model"`
	SystemFingerprint string `json:"system_fingerprint"`
	Choices           []struct {
		Index        int64        `json:"index"`
		Delta        Message      `json:"delta"`
		Logprobs     Logprobs     `json:"logprobs"`
		FinishReason FinishReason `json:"finish_reason"`
	} `json:"choices"`
	Usage Usage `json:"usage"`
}

// Model is the provider-specific model metadata.
type Model struct {
	ID      string `json:"id"`
	Object  string `json:"object"` // model
	OwnedBy string `json:"owned_by"`
}

// GetID implements genai.Model.
func (m *Model) GetID() string {
	return m.ID
}

func (m *Model) String() string {
	return m.ID
}

// Context implements genai.Model.
func (m *Model) Context() int64 {
	return 0
}

// ModelsResponse represents the response structure for DeepSeek models listing.
type ModelsResponse struct {
	Object string  `json:"object"` // list
	Data   []Model `json:"data"`
}

// ToModels converts DeepSeek models to genai.Model interfaces.
func (r *ModelsResponse) ToModels() []genai.Model {
	models := make([]genai.Model, len(r.Data))
	for i := range r.Data {
		models[i] = &r.Data[i]
	}
	return models
}

// ErrorResponse is the provider-specific error response.
type ErrorResponse struct {
	// Type  string `json:"type"`
	ErrorVal struct {
		Message string `json:"message"`
		Type    string `json:"type"`
		Param   string `json:"param"`
		Code    string `json:"code"`
	} `json:"error"`
}

func (er *ErrorResponse) Error() string {
	return fmt.Sprintf("%s: %s", er.ErrorVal.Type, er.ErrorVal.Message)
}

// IsAPIError implements base.ErrorResponseI.
func (er *ErrorResponse) IsAPIError() bool {
	return true
}
