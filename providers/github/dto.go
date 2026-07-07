// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Wire types for the GitHub Models inference and catalog REST API.
//
// Documentation: https://docs.github.com/en/rest/models

package github

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	"github.com/maruel/genai"
	"github.com/maruel/genai/base"
	"github.com/maruel/genai/internal"
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
			if v.TopLogprobs != 0 {
				unsupported = append(unsupported, "GenOptionText.TopLogprobs")
			}
			c.Stop = v.Stop
			if v.DecodeAs != nil {
				s, err := v.DecodeSchema()
				if err != nil {
					errs = append(errs, err)
				} else {
					raw := append(s[:len(s)-1], `,"name":"response"}`...)
					c.ResponseFormat = responseFormat{Type: "json_schema", JSONSchema: raw}
				}
			} else if v.ReplyAsJSON {
				c.ResponseFormat = responseFormat{Type: "json_object"}
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
		c.Messages = append(c.Messages, Message{Role: "system", Content: Contents{{Type: ContentText, Text: sp}}})
	}
	for i := range msgs {
		if len(msgs[i].ToolCallResults) > 1 {
			for j := range msgs[i].ToolCallResults {
				msgCopy := msgs[i]
				msgCopy.ToolCallResults = []genai.ToolCallResult{msgs[i].ToolCallResults[j]}
				var newMsg Message
				if err := newMsg.From(&msgCopy); err != nil {
					errs = append(errs, fmt.Errorf("message #%d: tool call results #%d: %w", i, j, err))
				} else {
					c.Messages = append(c.Messages, newMsg)
				}
			}
		} else {
			var newMsg Message
			if err := newMsg.From(&msgs[i]); err != nil {
				errs = append(errs, fmt.Errorf("message #%d: %w", i, err))
			} else {
				c.Messages = append(c.Messages, newMsg)
			}
		}
	}
	if len(unsupported) > 0 && len(errs) == 0 {
		return &base.ErrNotSupported{Options: unsupported}
	}
	return errors.Join(errs...)
}

// SetStream sets the streaming mode.
func (c *ChatRequest) SetStream(stream bool) {
	c.Stream = stream
	if stream {
		c.StreamOptions.IncludeUsage = true
	}
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

// From must be called with at most one ToolCallResults.
func (m *Message) From(in *genai.Message) error {
	if len(in.ToolCallResults) > 1 {
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
	if len(in.Requests) != 0 {
		m.Content = make(Contents, len(in.Requests))
		for i := range in.Requests {
			if err := m.Content[i].FromRequest(&in.Requests[i]); err != nil {
				return fmt.Errorf("request #%d: %w", i, err)
			}
		}
	}
	if len(in.Replies) != 0 {
		m.Content = make(Contents, 0, len(in.Replies))
		for i := range in.Replies {
			if in.Replies[i].Reasoning != "" {
				continue
			}
			if !in.Replies[i].ToolCall.IsZero() {
				m.ToolCalls = append(m.ToolCalls, ToolCall{})
				if err := m.ToolCalls[len(m.ToolCalls)-1].From(&in.Replies[i].ToolCall); err != nil {
					return fmt.Errorf("reply #%d: %w", i, err)
				}
				continue
			}
			m.Content = append(m.Content, Content{})
			if err := m.Content[len(m.Content)-1].FromReply(&in.Replies[i]); err != nil {
				return fmt.Errorf("reply #%d: %w", i, err)
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
	for _, c := range m.Content {
		switch c.Type {
		case ContentText:
			out.Replies = append(out.Replies, genai.Reply{Text: c.Text})
		default:
			return &internal.BadError{Err: fmt.Errorf("implement content type %q", c.Type)}
		}
	}
	for i := range m.ToolCalls {
		out.Replies = append(out.Replies, genai.Reply{})
		m.ToolCalls[i].To(&out.Replies[len(out.Replies)-1].ToolCall)
	}
	if len(out.Replies) == 0 {
		return errors.New("model sent no reply")
	}
	return nil
}

// Contents handles marshalling as a string when there is a single text content item.
type Contents []Content

// IsZero reports whether the value is zero.
func (c *Contents) IsZero() bool {
	return len(*c) == 0
}

// MarshalJSON implements json.Marshaler.
func (c *Contents) MarshalJSON() ([]byte, error) {
	if len(*c) == 0 {
		return []byte("null"), nil
	}
	if len(*c) == 1 && (*c)[0].Type == ContentText {
		return json.Marshal((*c)[0].Text)
	}
	return json.Marshal([]Content(*c))
}

// UnmarshalJSON implements json.Unmarshaler.
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
	if s != "" {
		*c = Contents{{Type: ContentText, Text: s}}
	} else {
		*c = nil
	}
	return nil
}

// Content is a provider-specific content block.
type Content struct {
	Type ContentType `json:"type,omitzero"`

	// Type == "text"
	Text string `json:"text,omitzero"`

	// Type == "image_url"
	ImageURL ImageURL `json:"image_url,omitzero"`
}

// FromRequest converts from a genai request.
func (c *Content) FromRequest(in *genai.Request) error {
	if in.Text != "" {
		c.Type = ContentText
		c.Text = in.Text
		return nil
	}
	if !in.Doc.IsZero() {
		mimeType, data, err := in.Doc.Read(10 * 1024 * 1024)
		if err != nil {
			return err
		}
		switch {
		case (in.Doc.URL != "" && mimeType == "") || strings.HasPrefix(mimeType, "image/"):
			c.Type = ContentImageURL
			if in.Doc.URL == "" {
				c.ImageURL = ImageURL{URL: fmt.Sprintf("data:%s;base64,%s", mimeType, base64.StdEncoding.EncodeToString(data))}
			} else {
				c.ImageURL = ImageURL{URL: in.Doc.URL}
			}
		case strings.HasPrefix(mimeType, "text/"):
			c.Type = ContentText
			if in.Doc.URL != "" {
				return fmt.Errorf("%s documents must be provided inline, not as a URL", mimeType)
			}
			c.Text = string(data)
		default:
			return fmt.Errorf("unsupported mime type %s", mimeType)
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
		c.Type = ContentText
		c.Text = in.Text
		return nil
	}
	if !in.Doc.IsZero() {
		mimeType, data, err := in.Doc.Read(10 * 1024 * 1024)
		if err != nil {
			return err
		}
		switch {
		case (in.Doc.URL != "" && mimeType == "") || strings.HasPrefix(mimeType, "image/"):
			c.Type = ContentImageURL
			if in.Doc.URL == "" {
				c.ImageURL = ImageURL{URL: fmt.Sprintf("data:%s;base64,%s", mimeType, base64.StdEncoding.EncodeToString(data))}
			} else {
				c.ImageURL = ImageURL{URL: in.Doc.URL}
			}
		case strings.HasPrefix(mimeType, "text/"):
			c.Type = ContentText
			if in.Doc.URL != "" {
				return fmt.Errorf("%s documents must be provided inline, not as a URL", mimeType)
			}
			c.Text = string(data)
		default:
			return &internal.BadError{Err: fmt.Errorf("unsupported mime type %s", mimeType)}
		}
		return nil
	}
	return &internal.BadError{Err: errors.New("unknown Reply type")}
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

// ToResult converts the response to a genai.Result.
func (c *ChatResponse) ToResult() (genai.Result, error) {
	total := c.Usage.TotalTokens
	if total == 0 {
		total = c.Usage.PromptTokens + c.Usage.CompletionTokens
	}
	out := genai.Result{
		Usage: genai.Usage{
			InputTokens:       c.Usage.PromptTokens,
			InputCachedTokens: c.Usage.PromptTokensDetails.CachedTokens,
			ReasoningTokens:   c.Usage.CompletionTokensDetails.ReasoningTokens,
			OutputTokens:      c.Usage.CompletionTokens,
			TotalTokens:       total,
		},
	}
	if len(c.Choices) != 1 {
		return out, &internal.BadError{Err: fmt.Errorf("server returned an unexpected number of choices, expected 1, got %d", len(c.Choices))}
	}
	out.Usage.FinishReason = c.Choices[0].FinishReason.ToFinishReason()
	err := c.Choices[0].Message.To(&out.Message)
	return out, err
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
	case FinishToolCalls:
		return genai.FinishedToolCalls
	case FinishContentFilter:
		return genai.FinishedContentFilter
	default:
		if !internal.BeLenient {
			panic(f)
		}
		return genai.FinishReason(f)
	}
}

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

// GetID implements genai.Model.
func (m *CatalogModel) GetID() string {
	return m.ID
}

// String implements fmt.Stringer.
func (m *CatalogModel) String() string {
	s := fmt.Sprintf("%s (%s)", m.ID, m.Publisher)
	if m.Limits.MaxInputTokens > 0 {
		s += fmt.Sprintf(" Context: %d/%d", m.Limits.MaxInputTokens, m.Limits.MaxOutputTokens)
	}
	return s
}

// Context implements genai.Model.
func (m *CatalogModel) Context() int64 {
	return m.Limits.MaxInputTokens
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

func (er *ErrorResponse) Error() string {
	suffix := ""
	if er.ErrorVal.Param != "" {
		suffix = fmt.Sprintf(" (param: %q)", er.ErrorVal.Param)
	}
	return fmt.Sprintf("%s (%s): %s%s", er.ErrorVal.Code, er.ErrorVal.Type, er.ErrorVal.Message, suffix)
}

// IsAPIError implements base.ErrorResponseI.
func (er *ErrorResponse) IsAPIError() bool {
	return true
}
