// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Wire types for the Xiaomi MiMo OpenAI-compatible chat completion API.
//
// See https://platform.xiaomimimo.com/docs/en-US/api/chat/openai-api

package xiaomi

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	"github.com/invopop/jsonschema"
	"github.com/maruel/genai"
	"github.com/maruel/genai/base"
	"github.com/maruel/genai/internal"
)

// GenOption controls thinking mode.
//
// MiMo v2.5 models default to thinking enabled. Set Thinking to false to
// switch to non-thinking mode.
type GenOption struct {
	// Thinking controls the thinking mode. When false, thinking is disabled.
	Thinking bool
}

// Validate implements genai.Validatable.
func (g *GenOption) Validate() error {
	return nil
}

// ChatRequest is documented at https://platform.xiaomimimo.com/docs/en-US/api/chat/openai-api
type ChatRequest struct {
	Model    string    `json:"model"`
	Messages []Message `json:"messages"`
	Stream   bool      `json:"stream"`
	Thinking struct {
		Type string `json:"type,omitzero"` // "enabled", "disabled"
	} `json:"thinking,omitzero"`
	Temperature float64 `json:"temperature,omitzero"`
	MaxTokens   int64   `json:"max_completion_tokens,omitzero"`
	ResponseFormat struct {
		Type string `json:"type,omitzero"` // "text", "json_object"
	} `json:"response_format,omitzero"`
	Stop          []string `json:"stop,omitzero"`
	StreamOptions struct {
		IncludeUsage bool `json:"include_usage,omitzero"`
	} `json:"stream_options,omitzero"`
	TopP       float64 `json:"top_p,omitzero"`
	FreqPenalty float64 `json:"frequency_penalty,omitzero"`
	PresPenalty float64 `json:"presence_penalty,omitzero"`
	ToolChoice  string  `json:"tool_choice,omitzero"` // "auto"
	Tools       []Tool  `json:"tools,omitzero"`
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
		case *GenOption:
			if !v.Thinking {
				c.Thinking.Type = "disabled"
			}
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
					c.ToolChoice = "auto" // MiMo only supports "auto"
				case genai.ToolCallNone:
					c.ToolChoice = "auto"
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
		c.Messages = append(c.Messages, Message{Role: "system", Content: Contents{{Type: ContentText, Text: sp}}})
	}
	for i := range msgs {
		// Split messages into multiple messages as needed.
		switch {
		case len(msgs[i].ToolCallResults) > 1:
			for j := range msgs[i].ToolCallResults {
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
			for j := range msgs[i].Requests {
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
	if len(unsupported) > 0 && len(errs) == 0 {
		return &base.ErrNotSupported{Options: unsupported}
	}
	return errors.Join(errs...)
}

// SetStream sets the streaming mode.
func (c *ChatRequest) SetStream(stream bool) {
	c.Stream = stream
}

// Contents is a collection of content blocks with custom marshaling.
//
// When marshaling, if the contents is a single text block, it's marshaled as a string.
// When unmarshaling, it handles both string and array formats.
type Contents []Content

// MarshalJSON implements json.Marshaler.
func (c Contents) MarshalJSON() ([]byte, error) {
	if len(c) == 1 && c[0].Type == ContentText {
		return json.Marshal(c[0].Text)
	}
	return json.Marshal([]Content(c))
}

// UnmarshalJSON implements json.Unmarshaler.
//
// OpenAI replies with content as a string.
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

	s := ""
	if err := json.Unmarshal(b, &s); err != nil {
		return err
	}
	*c = Contents{{Type: ContentText, Text: s}}
	return nil
}

// Message is documented at https://platform.xiaomimimo.com/docs/en-US/api/chat/openai-api
type Message struct {
	Role             string     `json:"role,omitzero"` // "system", "assistant", "user"
	Name             string     `json:"name,omitzero"`
	Content          Contents   `json:"content,omitzero"`
	ReasoningContent string     `json:"reasoning_content,omitzero"`
	ToolCalls        []ToolCall `json:"tool_calls,omitzero"`
	ToolCallID       string     `json:"tool_call_id,omitzero"`
}

// Content is a provider-specific content block.
type Content struct {
	Type ContentType `json:"type,omitzero"`

	// Type == "text"
	Text string `json:"text,omitzero"`

	// Type == "image_url"
	ImageURL struct {
		URL string `json:"url,omitzero"`
	} `json:"image_url,omitzero"`

	// Type == "input_audio"
	InputAudio struct {
		Data   string `json:"data,omitzero"` // base64 encoded or URL
		Format string `json:"format,omitzero"` // "mp3", "wav"
	} `json:"input_audio,omitzero"`

	// Type == "video_url"
	VideoURL struct {
		URL             string  `json:"url,omitzero"`
		FPS             float64 `json:"fps,omitzero"`
		MediaResolution string  `json:"media_resolution,omitzero"` // "default", "max"
	} `json:"video_url,omitzero"`
}

// ContentType is a provider-specific content type.
type ContentType string

// Content type values.
const (
	ContentText       ContentType = "text"
	ContentImageURL   ContentType = "image_url"
	ContentInputAudio ContentType = "input_audio"
	ContentVideoURL   ContentType = "video_url"
)

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
		if in.Requests[0].Text != "" {
			m.Content = Contents{{Type: ContentText, Text: in.Requests[0].Text}}
			return nil
		}
		if !in.Requests[0].Doc.IsZero() {
			return m.fromDoc(&in.Requests[0].Doc)
		}
		return errors.New("unknown Request type")
	}
	for i := range in.Replies {
		if len(in.Replies[i].Opaque) != 0 {
			return fmt.Errorf("reply #%d: field Reply.Opaque not supported", i)
		}
		switch {
		case in.Replies[i].Text != "":
			m.Content = append(m.Content, Content{Type: ContentText, Text: in.Replies[i].Text})
		case !in.Replies[i].Doc.IsZero():
			mimeType, data, err := in.Replies[i].Doc.Read(10 * 1024 * 1024)
			if err != nil {
				return fmt.Errorf("reply #%d: failed to read document: %w", i, err)
			}
			if in.Replies[i].Doc.URL != "" {
				return fmt.Errorf("reply #%d: xiaomi doesn't support document content blocks with URLs", i)
			}
			if !strings.HasPrefix(mimeType, "text/") {
				return fmt.Errorf("reply #%d: xiaomi only supports text documents, got %s", i, mimeType)
			}
			m.Content = append(m.Content, Content{Type: ContentText, Text: string(data)})
		case in.Replies[i].Reasoning != "":
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
		m.Content = Contents{{Type: ContentText, Text: in.ToolCallResults[0].Result}}
		m.ToolCallID = in.ToolCallResults[0].ID
	}
	return nil
}

// fromDoc converts a genai.Doc to the appropriate MiMo content format.
func (m *Message) fromDoc(doc *genai.Doc) error {
	mimeType, data, err := doc.Read(10 * 1024 * 1024)
	if err != nil {
		return fmt.Errorf("failed to read document: %w", err)
	}
	if mimeType == "" {
		return fmt.Errorf("unspecified mime type for URL %q", doc.URL)
	}
	switch {
	case strings.HasPrefix(mimeType, "image/"):
		c := Content{Type: ContentImageURL}
		if doc.URL == "" {
			c.ImageURL.URL = fmt.Sprintf("data:%s;base64,%s", mimeType, base64.StdEncoding.EncodeToString(data))
		} else {
			c.ImageURL.URL = doc.URL
		}
		m.Content = Contents{c}
	case strings.HasPrefix(mimeType, "audio/"):
		if doc.URL != "" {
			return errors.New("URL to audio file not supported")
		}
		c := Content{Type: ContentInputAudio}
		c.InputAudio.Data = base64.StdEncoding.EncodeToString(data)
		switch mimeType {
		case "audio/mpeg":
			c.InputAudio.Format = "mp3"
		case "audio/wav":
			c.InputAudio.Format = "wav"
		default:
			c.InputAudio.Format = "mp3"
		}
		m.Content = Contents{c}
	case strings.HasPrefix(mimeType, "video/"):
		c := Content{Type: ContentVideoURL}
		if doc.URL == "" {
			c.VideoURL.URL = fmt.Sprintf("data:%s;base64,%s", mimeType, base64.StdEncoding.EncodeToString(data))
		} else {
			c.VideoURL.URL = doc.URL
		}
		m.Content = Contents{c}
	case strings.HasPrefix(mimeType, "text/"):
		if doc.URL != "" {
			return fmt.Errorf("%s documents must be provided inline, not as a URL", mimeType)
		}
		m.Content = Contents{{Type: ContentText, Text: string(data)}}
	default:
		return fmt.Errorf("unsupported mime type: %s", mimeType)
	}
	return nil
}

// To converts to the genai equivalent.
func (m *Message) To(out *genai.Message) {
	if m.ReasoningContent != "" {
		out.Replies = append(out.Replies, genai.Reply{Reasoning: m.ReasoningContent})
	}
	for _, c := range m.Content {
		if c.Type == ContentText && c.Text != "" {
			out.Replies = append(out.Replies, genai.Reply{Text: c.Text})
		}
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
		Strict      bool               `json:"strict,omitzero"`
	} `json:"function"`
}

// ChatResponse is the provider-specific chat completion response.
type ChatResponse struct {
	ID      string `json:"id"`
	Choices []struct {
		FinishReason FinishReason `json:"finish_reason"`
		Index        int64        `json:"index"`
		Message      Message      `json:"message"`
	} `json:"choices"`
	Created int64  `json:"created"`
	Model   string `json:"model"`
	Object  string `json:"object"`
	Usage   Usage  `json:"usage"`
}

// ToResult converts the response to a genai.Result.
func (c *ChatResponse) ToResult() (genai.Result, error) {
	out := genai.Result{
		Usage: genai.Usage{
			InputTokens:       c.Usage.PromptTokens,
			InputCachedTokens: c.Usage.PromptTokensDetails.CachedTokens,
			ReasoningTokens:   c.Usage.CompletionTokensDetails.ReasoningTokens,
			OutputTokens:      c.Usage.CompletionTokens,
		},
	}
	if len(c.Choices) != 1 {
		return out, fmt.Errorf("expected 1 choice, got %#v", c.Choices)
	}
	out.Usage.FinishReason = c.Choices[0].FinishReason.ToFinishReason()
	c.Choices[0].Message.To(&out.Message)
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
	FinishRepetition    FinishReason = "repetition_truncation"
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
	case FinishRepetition:
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
	CompletionTokens int64 `json:"completion_tokens"`
	PromptTokens     int64 `json:"prompt_tokens"`
	TotalTokens      int64 `json:"total_tokens"`
	CompletionTokensDetails struct {
		ReasoningTokens int64 `json:"reasoning_tokens"`
	} `json:"completion_tokens_details"`
	PromptTokensDetails struct {
		CachedTokens int64 `json:"cached_tokens"`
		ImageTokens  int64 `json:"image_tokens"`
		AudioTokens  int64 `json:"audio_tokens"`
		VideoTokens  int64 `json:"video_tokens"`
	} `json:"prompt_tokens_details"`
}

// ChatStreamChunkResponse is the provider-specific streaming chat chunk.
type ChatStreamChunkResponse struct {
	ID                string `json:"id"`
	Object            string `json:"object"`
	Created           int64  `json:"created"`
	Model             string `json:"model"`
	SystemFingerprint string `json:"system_fingerprint"`
	Choices           []struct {
		Index        int64        `json:"index"`
		Delta        Message      `json:"delta"`
		FinishReason FinishReason `json:"finish_reason"`
	} `json:"choices"`
	Usage Usage `json:"usage"`
}

// Model is the provider-specific model metadata.
type Model struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
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

// ModelsResponse represents the response structure for MiMo models listing.
type ModelsResponse struct {
	Object string  `json:"object"`
	Data   []Model `json:"data"`
}

// ToModels converts MiMo models to genai.Model interfaces.
func (r *ModelsResponse) ToModels() []genai.Model {
	models := make([]genai.Model, len(r.Data))
	for i := range r.Data {
		models[i] = &r.Data[i]
	}
	return models
}

// ErrorResponse is the provider-specific error response.
type ErrorResponse struct {
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
