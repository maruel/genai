// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Wire types for the OpenAI-compatible chat completion API.

package openaicompatible

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	"github.com/maruel/genai"
	"github.com/maruel/genai/base"
	"github.com/maruel/genai/internal"
)

// ChatRequest is the provider-specific chat completion request.
//
// Strict schema enforcement and parallel_tool_calls are omitted because they are
// not common requirements across providers. ToolChoice modes depend on the remote
// model; for example, Xiaomi's dedicated client only sends "auto".
type ChatRequest struct {
	Model            string    `json:"model,omitzero"`
	Messages         []Message `json:"messages"`
	MaxTokens        int64     `json:"max_tokens,omitzero"`
	Stop             []string  `json:"stop,omitzero"`
	Stream           bool      `json:"stream,omitzero"`
	Temperature      float64   `json:"temperature,omitzero"`
	TopP             float64   `json:"top_p,omitzero"` // [0, 1.0]
	FrequencyPenalty float64   `json:"frequency_penalty,omitzero"`
	PresencePenalty  float64   `json:"presence_penalty,omitzero"` // [-2, 2]
	ToolChoice       string    `json:"tool_choice,omitzero"`      // "auto", "none", "required"
	Tools            []Tool    `json:"tools,omitzero"`
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
			c.MaxTokens = v.MaxTokens
			c.Temperature = v.Temperature
			c.TopP = v.TopP
			sp = v.SystemPrompt
			if v.TopK != 0 {
				unsupported = append(unsupported, "GenOptionText.TopK")
			}
			if v.TopLogprobs > 0 {
				unsupported = append(unsupported, "GenOptionText.TopLogprobs")
			}
			c.Stop = v.Stop
			if v.ReplyAsJSON {
				errs = append(errs, errors.New("unsupported option ReplyAsJSON"))
			}
			if v.DecodeAs != nil {
				errs = append(errs, errors.New("unsupported option DecodeAs"))
			}
		case *genai.GenOptionTools:
			if len(v.Tools) != 0 {
				switch v.Force {
				case genai.ToolCallAny:
					c.ToolChoice = "auto"
				case genai.ToolCallNone:
					c.ToolChoice = "none"
				case genai.ToolCallRequired:
					c.ToolChoice = "required"
				default:
					return fmt.Errorf("unsupported tool choice %d", v.Force)
				}
				c.Tools = make([]Tool, len(v.Tools))
				for i := range v.Tools {
					t := &v.Tools[i]
					s, err := t.GetInputSchema()
					if err != nil {
						errs = append(errs, fmt.Errorf("tool #%d: %w", i, err))
						continue
					}
					c.Tools[i] = Tool{Type: "function", Function: ToolFunction{Name: t.Name, Description: t.Description, Parameters: s}}
				}
			}
		default:
			unsupported = append(unsupported, internal.TypeName(opt))
		}
	}

	c.Messages = make([]Message, 0, len(msgs)+1)
	if sp != "" {
		c.Messages = append(c.Messages, Message{Role: "system", Content: []Content{{Type: ContentText, Text: sp}}})
	}
	for i := range msgs {
		if len(msgs[i].ToolCallResults) > 1 {
			for j := range msgs[i].ToolCallResults {
				msg := msgs[i]
				msg.ToolCallResults = msgs[i].ToolCallResults[j : j+1]
				var m Message
				if err := m.From(&msg); err != nil {
					errs = append(errs, fmt.Errorf("message #%d, result #%d: %w", i, j, err))
				} else {
					c.Messages = append(c.Messages, m)
				}
			}
			continue
		}
		var m Message
		if err := m.From(&msgs[i]); err != nil {
			errs = append(errs, fmt.Errorf("message #%d: %w", i, err))
		} else {
			c.Messages = append(c.Messages, m)
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

// Tool is a function tool definition.
type Tool struct {
	Type     string       `json:"type"`
	Function ToolFunction `json:"function"`
}

// ToolFunction describes a function and its input schema.
type ToolFunction struct {
	Name        string           `json:"name"`
	Description string           `json:"description,omitzero"`
	Parameters  genai.JSONSchema `json:"parameters"`
}

// ToolCall is a function call or a streamed function call delta.
type ToolCall struct {
	ID       string           `json:"id,omitzero"`
	Type     string           `json:"type,omitzero"`
	Function ToolCallFunction `json:"function,omitzero"`
	// Index identifies parallel stream deltas. Nil means the provider omitted it.
	Index *int64 `json:"index,omitzero"`
}

// To converts a complete call to genai and validates its arguments.
// Empty arguments represent a no-argument call and become an empty JSON object.
func (t *ToolCall) To(out *genai.ToolCall) error {
	args := t.Function.Arguments
	if args == "" {
		args = "{}"
	}
	if !json.Valid([]byte(args)) {
		return fmt.Errorf("invalid tool call arguments %q", args)
	}
	*out = genai.ToolCall{ID: t.ID, Name: t.Function.Name, Arguments: args}
	return out.Validate()
}

// ToolCallFunction contains a function name and JSON arguments or argument fragments.
type ToolCallFunction struct {
	Name      string `json:"name,omitzero"`
	Arguments string `json:"arguments,omitzero"`
}

// Message accepts the common Chat Completions message fields and content variants.
type Message struct {
	Role       string     `json:"role,omitzero"` // "system", "assistant", "user", "tool"
	Content    Contents   `json:"content,omitzero"`
	ToolCalls  []ToolCall `json:"tool_calls,omitzero"`
	ToolCallID string     `json:"tool_call_id,omitzero"`
}

// IsZero reports whether the value is zero.
func (m *Message) IsZero() bool {
	return m.Role == "" && len(m.Content) == 0 && len(m.ToolCalls) == 0 && m.ToolCallID == ""
}

// From converts from a genai.Message to a Message. It accepts at most one tool result.
func (m *Message) From(in *genai.Message) error {
	if len(in.ToolCallResults) > 1 {
		return errors.New("expected at most one tool result")
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
		m.Content = make([]Content, 0, len(in.Requests))
		for i := range in.Requests {
			switch {
			case in.Requests[i].Text != "":
				m.Content = append(m.Content, Content{Type: ContentText, Text: in.Requests[i].Text})
			case !in.Requests[i].Doc.IsZero():
				// Check if this is a text document
				mimeType, data, err := in.Requests[i].Doc.Read(10 * 1024 * 1024)
				if err != nil {
					return fmt.Errorf("request #%d: failed to read document: %w", i, err)
				}
				// text/plain, text/markdown
				if !strings.HasPrefix(mimeType, "text/") {
					return fmt.Errorf("request #%d: openaicompatible only supports text documents, got %s", i, mimeType)
				}
				if in.Requests[i].Doc.URL != "" {
					return fmt.Errorf("request #%d: %s documents must be provided inline, not as a URL", i, mimeType)
				}
				m.Content = append(m.Content, Content{Type: ContentText, Text: string(data)})
			default:
				return fmt.Errorf("request #%d: unknown Request type", 0)
			}
		}
	}
	if len(in.Replies) > 0 {
		for i := range in.Replies {
			if len(in.Replies[i].Opaque) != 0 {
				return &internal.BadError{Err: fmt.Errorf("reply #%d: field Reply.Opaque not supported", i)}
			}
			switch {
			case in.Replies[i].Text != "":
				m.Content = append(m.Content, Content{Type: ContentText, Text: in.Replies[i].Text})
			case !in.Replies[i].ToolCall.IsZero():
				t := &in.Replies[i].ToolCall
				if len(t.Opaque) != 0 {
					return errors.New("field ToolCall.Opaque not supported")
				}
				m.ToolCalls = append(m.ToolCalls, ToolCall{ID: t.ID, Type: "function", Function: ToolCallFunction{Name: t.Name, Arguments: t.Arguments}})
			case in.Replies[i].Reasoning != "":
				// Ignore
			case !in.Replies[i].Doc.IsZero():
				// Check if this is a text document
				mimeType, data, err := in.Replies[i].Doc.Read(10 * 1024 * 1024)
				if err != nil {
					return fmt.Errorf("reply #%d: failed to read document: %w", i, err)
				}
				// text/plain, text/markdown
				if !strings.HasPrefix(mimeType, "text/") {
					return fmt.Errorf("reply #%d: openaicompatible only supports text documents, got %s", i, mimeType)
				}
				if in.Replies[i].Doc.URL != "" {
					return fmt.Errorf("reply #%d: %s documents must be provided inline, not as a URL", i, mimeType)
				}
				m.Content = append(m.Content, Content{Type: ContentText, Text: string(data)})
			default:
				return &internal.BadError{Err: fmt.Errorf("reply #%d: unknown Reply type", i)}
			}
		}
	}
	if len(in.ToolCallResults) != 0 {
		m.Content = Contents{{Type: ContentText, Text: in.ToolCallResults[0].Result}}
		m.ToolCallID = in.ToolCallResults[0].ID
	}
	return nil
}

// To converts to the genai equivalent.
func (m *Message) To(out *genai.Message) error {
	out.Replies = make([]genai.Reply, 0, len(m.Content)+len(m.ToolCalls))
	for _, content := range m.Content {
		if content.Type != "" && content.Type != ContentText {
			return fmt.Errorf("unsupported content type %q", content.Type)
		}
		if content.Text != "" {
			out.Replies = append(out.Replies, genai.Reply{Text: content.Text})
		}
	}
	for i := range m.ToolCalls {
		var r genai.Reply
		if err := m.ToolCalls[i].To(&r.ToolCall); err != nil {
			return fmt.Errorf("tool call #%d: %w", i, err)
		}
		out.Replies = append(out.Replies, r)
	}
	return nil
}

// Content is a provider-specific content block.
type Content struct {
	Type ContentType `json:"type,omitzero"`
	Text string      `json:"text,omitzero"`
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
	if len(*c) == 1 && (*c)[0].Type == ContentText {
		return json.Marshal((*c)[0].Text)
	}
	return json.Marshal([]Content(*c))
}

// UnmarshalJSON implements custom unmarshalling for Contents type
// to handle cases where content could be a string or Content struct.
func (c *Contents) UnmarshalJSON(b []byte) error {
	if bytes.Equal(b, []byte("null")) {
		*c = nil
		return nil
	}
	if err := json.Unmarshal(b, (*[]Content)(c)); err == nil {
		return nil
	}

	v := Content{}
	if err := json.Unmarshal(b, &v); err == nil {
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

// ChatResponse captures all the different ways providers can reply.
type ChatResponse struct {
	Message
	Message2     Message      `json:"message"`
	FinishReason FinishReason `json:"finish_reason"`
	Choices      []struct {
		FinishReason FinishReason `json:"finish_reason"`
		Message      Message      `json:"message"`
	} `json:"choices"`
	Usage Usage `json:"usage"`
}

// ToResult converts the response to a genai.Result.
func (c *ChatResponse) ToResult() (genai.Result, error) {
	out := genai.Result{
		Usage: genai.Usage{
			InputTokens:  c.Usage.PromptTokens,
			OutputTokens: c.Usage.CompletionTokens,
			TotalTokens:  c.Usage.TotalTokens,
		},
	}
	if len(c.Choices) > 1 {
		return out, fmt.Errorf("expected 1 choice, got %#v", c)
	}
	if len(c.Choices) == 1 {
		out.Usage.FinishReason = c.Choices[0].FinishReason.ToFinishReason()
		err := c.Choices[0].Message.To(&out.Message)
		normalizeToolFinish(&out.Usage, len(c.Choices[0].Message.ToolCalls) != 0)
		return out, err
	}
	m := c.Message2
	if m.IsZero() {
		m = c.Message
	}
	if m.IsZero() {
		return out, fmt.Errorf("expected 1 choice, got %#v", c)
	}
	if err := m.To(&out.Message); err != nil {
		return out, err
	}
	out.Usage.FinishReason = c.FinishReason.ToFinishReason()
	normalizeToolFinish(&out.Usage, len(m.ToolCalls) != 0)
	return out, nil
}

// FinishReason is a provider-specific finish reason.
type FinishReason string

// ToFinishReason converts to a genai.FinishReason.
func (f FinishReason) ToFinishReason() genai.FinishReason {
	switch f {
	case FinishStop:
		return genai.FinishedStop
	case FinishLength:
		return genai.FinishedLength
	case FinishToolCalls, FinishFunctionCall:
		return genai.FinishedToolCalls
	default:
		return genai.FinishReason(f)
	}
}

// Finish reason values.
const (
	FinishFunctionCall FinishReason = "function_call"
	FinishLength       FinishReason = "length"
	FinishStop         FinishReason = "stop"
	FinishToolCalls    FinishReason = "tool_calls"
)

// ChatStreamChunkResponse is the provider-specific streaming chat chunk.
type ChatStreamChunkResponse struct {
	Delta struct {
		Text    string  `json:"text"`
		Message Message `json:"message"`
	} `json:"delta"`
	FinishReason FinishReason `json:"finish_reason"`
	Choices      []struct {
		Delta struct {
			Message
		} `json:"delta"`
		FinishReason FinishReason `json:"finish_reason"`
		ToolCalls    []ToolCall   `json:"tool_calls"` // Some providers send whole calls outside delta.
	} `json:"choices"`
	Usage Usage `json:"usage"`
}

// normalizeToolFinish handles providers that report stop or omit the finish reason after tool calls.
// Other finish reasons, including truncation and provider extensions, stay unchanged.
func normalizeToolFinish(u *genai.Usage, hasTools bool) {
	if hasTools && (u.FinishReason == "" || u.FinishReason == genai.FinishedStop) {
		u.FinishReason = genai.FinishedToolCalls
	}
}

// Usage is the provider-specific token usage.
type Usage struct {
	PromptTokens     int64 `json:"prompt_tokens"`
	CompletionTokens int64 `json:"completion_tokens"`
	TotalTokens      int64 `json:"total_tokens"`
}

//

// ErrorResponse is as generic as possible since error responses are highly non-standard.
type ErrorResponse map[string]json.RawMessage

func (er *ErrorResponse) Error() string {
	return fmt.Sprintf("%s", *er)
}

// IsAPIError implements base.ErrorResponseI.
func (er *ErrorResponse) IsAPIError() bool {
	return true
}
