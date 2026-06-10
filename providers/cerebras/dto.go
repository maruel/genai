// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Wire types for the Cerebras chat completion API.
//
// Reference: https://inference-docs.cerebras.ai/api-reference/chat-completions

package cerebras

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"reflect"
	"slices"
	"strings"

	"github.com/maruel/genai"
	"github.com/maruel/genai/base"
	"github.com/maruel/genai/internal"
)

// ChatRequest is documented at https://inference-docs.cerebras.ai/api-reference/chat-completions
type ChatRequest struct {
	Model               string    `json:"model"`
	Messages            []Message `json:"messages"`
	MaxCompletionTokens int64     `json:"max_completion_tokens,omitzero"` // Includes reasoning tokens
	MinCompletionTokens int64     `json:"min_completion_tokens,omitzero"`
	ResponseFormat      struct {
		// https://inference-docs.cerebras.ai/capabilities/structured-outputs
		Type       string `json:"type"` // "json_object", "json_schema"
		JSONSchema struct {
			Name   string           `json:"name"`
			Schema genai.JSONSchema `json:"schema"`
			Strict bool             `json:"strict"`
		} `json:"json_schema,omitzero"`
	} `json:"response_format,omitzero"`
	Seed          int64    `json:"seed,omitzero"`
	Stop          []string `json:"stop,omitzero"` // Up to 4 sequences
	Stream        bool     `json:"stream,omitzero"`
	StreamOptions struct {
		IncludeUsage bool `json:"include_usage,omitzero"` // Isn't necessary.
	} `json:"stream_options,omitzero"`
	Temperature       float64           `json:"temperature,omitzero"`
	TopP              float64           `json:"top_p,omitzero"`       // [0, 1.0]
	ToolChoice        string            `json:"tool_choice,omitzero"` // "none", "auto", "required" or a struct {"type": "function", "function": {"name": "my_function"}}
	Tools             []Tool            `json:"tools,omitzero"`
	ParallelToolCalls bool              `json:"parallel_tool_calls,omitzero"`
	FrequencyPenalty  float64           `json:"frequency_penalty,omitzero"`
	LogitBias         map[int64]float64 `json:"logit_bias,omitzero"`       // Token bias [-100, 100]
	N                 int64             `json:"n,omitzero"`                // Number of choices
	ServiceTier       complex128        `json:"service_tier,omitzero"`     // "auto", "default"
	PresencePenalty   float64           `json:"presence_penalty,omitzero"` // [-2, 2]
	User              string            `json:"user,omitzero"`             // End user ID to help identify abuse
	Logprobs          bool              `json:"logprobs,omitzero"`         // Whether to return log probabilities
	TopLogprobs       int64             `json:"top_logprobs,omitzero"`     // [0, 20]
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
		switch v := opt.(type) {
		case *genai.GenOptionText:
			c.MaxCompletionTokens = v.MaxTokens
			c.Temperature = v.Temperature
			c.TopP = v.TopP
			sp = v.SystemPrompt
			if v.TopLogprobs > 0 {
				c.TopLogprobs = v.TopLogprobs
				c.Logprobs = true
			}
			if v.TopK != 0 {
				unsupported = append(unsupported, "GenOptionText.TopK")
			}
			c.Stop = v.Stop
			if v.DecodeAs != nil {
				c.ResponseFormat.Type = "json_schema"
				s, err := genai.JSONSchemaFor(reflect.TypeOf(v.DecodeAs))
				if err != nil {
					errs = append(errs, err)
				} else {
					c.ResponseFormat.JSONSchema.Schema = s
				}
				c.ResponseFormat.JSONSchema.Strict = true
			} else if v.ReplyAsJSON {
				c.ResponseFormat.Type = "json_object"
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
				c.ParallelToolCalls = true
				c.Tools = make([]Tool, len(v.Tools))
				for i, t := range v.Tools {
					c.Tools[i].Type = "function"
					c.Tools[i].Function.Name = t.Name
					c.Tools[i].Function.Description = t.Description
					s, err := t.GetInputSchema()
					if err != nil {
						errs = append(errs, err)
					}
					c.Tools[i].Function.Parameters = s
				}
			}
		case genai.GenOptionSeed:
			c.Seed = int64(v)
		default:
			unsupported = append(unsupported, internal.TypeName(opt))
		}
	}

	if sp != "" {
		c.Messages = append(c.Messages, Message{Role: "system", Content: []Content{{Type: ContentText, Text: sp}}})
	}
	for i := range msgs {
		switch {
		case len(msgs[i].ToolCallResults) > 1:
			// Handle messages with multiple tool call results by creating multiple messages
			for j := range msgs[i].ToolCallResults {
				// Create a copy of the message with only one tool call result
				msgCopy := msgs[i]
				msgCopy.ToolCallResults = []genai.ToolCallResult{msgs[i].ToolCallResults[j]}
				var newMsg Message
				if err := newMsg.From(&msgCopy); err != nil {
					errs = append(errs, fmt.Errorf("message #%d: tool call results #%d: %w", i, j, err))
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
					errs = append(errs, fmt.Errorf("message #%d: request #%d: %w", i, j, err))
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

// Message is completely undocumented as of May 2025.
// https://inference-docs.cerebras.ai/api-reference/chat-completions
//
// https://discord.com/channels/1085960591052644463/1345047923339296819/1348990530956034058
type Message struct {
	Role       string     `json:"role,omitzero"` // "system", "assistant", "user"
	Content    Contents   `json:"content,omitzero"`
	Reasoning  string     `json:"reasoning,omitzero"`
	ToolCalls  []ToolCall `json:"tool_calls,omitzero"`
	ToolCallID string     `json:"tool_call_id,omitzero"`
	Name       string     `json:"name,omitzero"` // Tool call name.
}

// From must be called with at most one Request or ToolCallResults.
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
	if len(in.Requests) > 0 {
		m.Content = make([]Content, 1)
		if err := m.Content[0].FromRequest(&in.Requests[0]); err != nil {
			return err
		}
	}
	for i := range in.Replies {
		if !in.Replies[i].ToolCall.IsZero() {
			m.ToolCalls = append(m.ToolCalls, ToolCall{})
			if err := m.ToolCalls[len(m.ToolCalls)-1].From(&in.Replies[i].ToolCall); err != nil {
				return fmt.Errorf("reply #%d: %w", i, err)
			}
			continue
		}
		// Do not include thinking in the message.
		if in.Replies[i].Reasoning != "" {
			continue
		}
		m.Content = append(m.Content, Content{})
		if err := m.Content[len(m.Content)-1].FromReply(&in.Replies[i]); err != nil {
			return fmt.Errorf("reply #%d: %w", i, err)
		}
	}
	if len(in.ToolCallResults) != 0 {
		// Process only the first tool call result in this method.
		// The Init method handles multiple tool call results by creating multiple messages.
		m.Content = Contents{{Type: ContentText, Text: in.ToolCallResults[0].Result}}
		m.ToolCallID = in.ToolCallResults[0].ID
		m.Name = in.ToolCallResults[0].Name
	}
	return nil
}

// To converts to the genai equivalent.
func (m *Message) To(out *genai.Message) error {
	out.Replies = make([]genai.Reply, 0, len(m.Content)+len(m.ToolCalls))
	if m.Reasoning != "" {
		out.Replies = append(out.Replies, genai.Reply{Reasoning: m.Reasoning})
	}
	for _, content := range m.Content {
		switch content.Type {
		case ContentText:
			out.Replies = append(out.Replies, genai.Reply{Text: content.Text})
		default:
			return &internal.BadError{Err: fmt.Errorf("implement content type %q", content.Type)}
		}
	}
	for i := range m.ToolCalls {
		out.Replies = append(out.Replies, genai.Reply{})
		m.ToolCalls[i].To(&out.Replies[len(out.Replies)-1].ToolCall)
	}
	if len(out.Replies) == 0 {
		// This happens with gpt-oss-120b with Stop.
		return errors.New("model sent no reply")
	}
	return nil
}

// Content is a provider-specific content block.
type Content struct {
	Type ContentType `json:"type,omitzero"`
	Text string      `json:"text,omitzero"`
}

// FromRequest converts from a genai request.
func (c *Content) FromRequest(in *genai.Request) error {
	if in.Text != "" {
		c.Type = ContentText
		c.Text = in.Text
		return nil
	}
	if !in.Doc.IsZero() {
		// Check if this is a text document
		mimeType, data, err := in.Doc.Read(10 * 1024 * 1024)
		if err != nil {
			return fmt.Errorf("failed to read document: %w", err)
		}
		if !strings.HasPrefix(mimeType, "text/") {
			return fmt.Errorf("cerebras only supports text documents, got %s", mimeType)
		}
		if in.Doc.URL != "" {
			return fmt.Errorf("%s documents must be provided inline, not as a URL", mimeType)
		}
		c.Type = ContentText
		c.Text = string(data)
		return nil
	}
	return errors.New("unknown Request type")
}

// FromReply converts from a genai reply.
func (c *Content) FromReply(in *genai.Reply) error {
	if len(in.Opaque) != 0 {
		return &internal.BadError{Err: errors.New("field Reply.Opaque not supported")}
	}
	switch {
	case in.Text != "":
		c.Type = ContentText
		c.Text = in.Text
	case in.Reasoning != "":
		// Ignore
	case !in.Doc.IsZero():
		// Check if this is a text document
		mimeType, data, err := in.Doc.Read(10 * 1024 * 1024)
		if err != nil {
			return fmt.Errorf("failed to read document: %w", err)
		}
		if !strings.HasPrefix(mimeType, "text/") {
			return fmt.Errorf("cerebras only supports text documents, got %s", mimeType)
		}
		if in.Doc.URL != "" {
			return fmt.Errorf("%s documents must be provided inline, not as a URL", mimeType)
		}
		c.Type = ContentText
		c.Text = string(data)
	default:
		// Cerebras doesn't support other document types.
		return &internal.BadError{Err: errors.New("internal error: unknown Reply type")}
	}
	return nil
}

// ContentType is a provider-specific content type.
type ContentType string

// Content type values.
const (
	ContentText ContentType = "text"
)

// Contents represents a slice of Content with custom unmarshalling to handle
// both string and Content struct types.
type Contents []Content

// MarshalJSON implements json.Marshaler.
func (c *Contents) MarshalJSON() ([]byte, error) {
	// If there's only one content and it's a string, marshal as a string.
	if len(*c) == 1 && (*c)[0].Type == ContentText {
		return json.Marshal((*c)[0].Text)
	}
	// If there's many contents, marshal as an array of Content.
	return json.Marshal([]Content(*c))
}

// UnmarshalJSON implements custom unmarshalling for Contents type
// to handle cases where content could be a string or Content struct.
func (c *Contents) UnmarshalJSON(b []byte) error {
	if bytes.Equal(b, []byte("null")) {
		*c = nil
		return nil
	}
	d := json.NewDecoder(bytes.NewReader(b))
	if !internal.BeLenient {
		d.DisallowUnknownFields()
	}
	if err := d.Decode((*[]Content)(c)); err == nil {
		return nil
	}

	v := Content{}
	d = json.NewDecoder(bytes.NewReader(b))
	if !internal.BeLenient {
		d.DisallowUnknownFields()
	}
	if err := d.Decode(&v); err == nil {
		*c = Contents{v}
		return nil
	}

	s := ""
	if err := json.Unmarshal(b, &s); err != nil {
		return err
	}
	*c = Contents{{Type: ContentText, Text: s}}
	return nil
}

// Tool is a provider-specific tool definition.
type Tool struct {
	Type     string `json:"type"` // "function"
	Function struct {
		Name        string           `json:"name"`
		Description string           `json:"description"`
		Parameters  genai.JSONSchema `json:"parameters"`
	} `json:"function"`
}

// ToolCall is a provider-specific tool call.
type ToolCall struct {
	Type     string `json:"type,omitzero"` // "function"
	ID       string `json:"id,omitzero"`
	Index    int64  `json:"index,omitzero"`
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

// ChatResponse is the provider-specific chat completion response.
type ChatResponse struct {
	ID                string    `json:"id"`
	Model             string    `json:"model"`
	Object            string    `json:"object"` // "chat.completion"
	SystemFingerprint string    `json:"system_fingerprint"`
	Created           base.Time `json:"created"`
	Choices           []struct {
		Index        int64        `json:"index"`
		FinishReason FinishReason `json:"finish_reason"`
		Message      Message      `json:"message"`
		Logprobs     Logprobs     `json:"logprobs"`
	} `json:"choices"`
	Usage    Usage `json:"usage"`
	TimeInfo struct {
		QueueTime  float64   `json:"queue_time"`      // In seconds
		PromptTime float64   `json:"prompt_time"`     // In seconds
		ChatTime   float64   `json:"completion_time"` // In seconds
		TotalTime  float64   `json:"total_time"`      // In seconds
		Created    base.Time `json:"created"`
	} `json:"time_info"`
}

// ToResult converts the response to a genai.Result.
func (c *ChatResponse) ToResult() (genai.Result, error) {
	out := genai.Result{
		// At the moment, Cerebras doesn't support cached tokens.
		Usage: genai.Usage{
			InputTokens:       c.Usage.PromptTokens,
			InputCachedTokens: c.Usage.PromptTokensDetails.CachedTokens,
			OutputTokens:      c.Usage.CompletionTokens,
			TotalTokens:       c.Usage.TotalTokens,
		},
	}
	if len(c.Choices) != 1 {
		return out, fmt.Errorf("expected 1 choice, got %#v", c.Choices)
	}
	out.Usage.FinishReason = c.Choices[0].FinishReason.ToFinishReason()
	err := c.Choices[0].Message.To(&out.Message)
	if out.Usage.FinishReason == genai.FinishedStop && slices.ContainsFunc(out.Replies, func(r genai.Reply) bool { return !r.ToolCall.IsZero() }) {
		// Lie for the benefit of everyone.
		out.Usage.FinishReason = genai.FinishedToolCalls
	}
	out.Logprobs = c.Choices[0].Logprobs.To()
	return out, err
}

// FinishReason is a provider-specific finish reason.
type FinishReason string

// Finish reason values.
const (
	FinishStop          FinishReason = "stop"
	FinishToolCalls     FinishReason = "tool_calls"
	FinishLength        FinishReason = "length"
	FinishContentFilter FinishReason = "content_filter"
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
	default:
		if !internal.BeLenient {
			panic(f)
		}
		return genai.FinishReason(f)
	}
}

// ChatStreamChunkResponse is the provider-specific streaming chat chunk.
type ChatStreamChunkResponse struct {
	ID                string    `json:"id"`
	Model             string    `json:"model"`
	Object            string    `json:"object"`
	SystemFingerprint string    `json:"system_fingerprint"`
	Created           base.Time `json:"created"`
	Choices           []struct {
		Delta struct {
			Role      string     `json:"role"`
			Content   Contents   `json:"content"`
			Reasoning string     `json:"reasoning"`
			ToolCalls []ToolCall `json:"tool_calls"`
		} `json:"delta"`
		Index        int64        `json:"index"`
		FinishReason FinishReason `json:"finish_reason"`
		Logprobs     Logprobs     `json:"logprobs"`
	} `json:"choices"`
	Usage    Usage `json:"usage"`
	TimeInfo struct {
		QueueTime  float64   `json:"queue_time"`
		PromptTime float64   `json:"prompt_time"`
		ChatTime   float64   `json:"completion_time"`
		TotalTime  float64   `json:"total_time"`
		Created    base.Time `json:"created"`
	} `json:"time_info"`
}

// Logprobs is the provider-specific log probabilities.
type Logprobs struct {
	Content []struct {
		Token       string  `json:"token"`
		Bytes       []byte  `json:"bytes"`
		Logprob     float64 `json:"logprob"`
		TopLogprobs []struct {
			Token   string  `json:"token"`
			Bytes   []byte  `json:"bytes"`
			Logprob float64 `json:"logprob"`
		} `json:"top_logprobs"`
	} `json:"content"`
}

// To converts to the genai equivalent.
func (l *Logprobs) To() [][]genai.Logprob {
	if len(l.Content) == 0 {
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

// Usage is the provider-specific token usage.
type Usage struct {
	PromptTokens        int64 `json:"prompt_tokens"`
	CompletionTokens    int64 `json:"completion_tokens"`
	TotalTokens         int64 `json:"total_tokens"`
	PromptTokensDetails struct {
		CachedTokens int64 `json:"cached_tokens"`
	} `json:"prompt_tokens_details"`
}

// Model is the provider-specific model metadata.
type Model struct {
	ID      string    `json:"id"`
	Object  string    `json:"object"`
	Created base.Time `json:"created"`
	OwnedBy string    `json:"owned_by"`
}

// GetID implements genai.Model.
func (m *Model) GetID() string {
	return m.ID
}

func (m *Model) String() string {
	if m.Created < 100 {
		return m.ID
	}
	return fmt.Sprintf("%s (%s)", m.ID, m.Created.AsTime().Format("2006-01-02"))
}

// Context implements genai.Model.
func (m *Model) Context() int64 {
	return 0
}

// ModelsResponse represents the response structure for Cerebras models listing.
type ModelsResponse struct {
	Object string  `json:"object"`
	Data   []Model `json:"data"`
}

// ToModels converts Cerebras models to genai.Model interfaces.
func (r *ModelsResponse) ToModels() []genai.Model {
	models := make([]genai.Model, len(r.Data))
	for i := range r.Data {
		models[i] = &r.Data[i]
	}
	return models
}

// ErrorResponse is the provider-specific error response.
type ErrorResponse struct {
	// Either this
	Detail string `json:"detail"`

	// Or this (tool call)
	StatusCode int64 `json:"status_code"`
	ErrorVal   struct {
		Message          string `json:"message"`
		Type             string `json:"type"`
		Param            string `json:"param"`
		Code             string `json:"code"`
		FailedGeneration string `json:"failed_generation"`
	} `json:"error"`

	// Or this
	Message          string `json:"message"`
	Type             string `json:"type"`
	Param            string `json:"param"`
	Code             string `json:"code"`
	FailedGeneration string `json:"failed_generation"`
}

func (er *ErrorResponse) Error() string {
	if er.Detail != "" {
		return er.Detail
	}
	if er.StatusCode != 0 {
		return fmt.Sprintf("%s/%s/%s: %s while generating %q", er.ErrorVal.Type, er.ErrorVal.Param, er.ErrorVal.Code, er.ErrorVal.Message, er.ErrorVal.FailedGeneration)
	}
	if er.FailedGeneration != "" {
		return fmt.Sprintf("%s/%s/%s: %s while generating %q", er.Type, er.Param, er.Code, er.Message, er.FailedGeneration)
	}
	// Check if this is actually an empty error response (all fields are empty)
	if er.Type == "" && er.Param == "" && er.Code == "" && er.Message == "" {
		return "unknown error (empty error response from API)"
	}
	return fmt.Sprintf("%s/%s/%s: %s", er.Type, er.Param, er.Code, er.Message)
}

// IsAPIError implements base.ErrorResponseI.
func (er *ErrorResponse) IsAPIError() bool {
	return true
}
