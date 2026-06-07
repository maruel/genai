// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Wire types for the Mistral chat completions API.
//
// Documented at https://docs.mistral.ai/api/#tag/chat/operation/chat_completion_v1_chat_completions_post

package mistral

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"reflect"
	"strings"

	"github.com/invopop/jsonschema"
	"github.com/maruel/genai"
	"github.com/maruel/genai/base"
	"github.com/maruel/genai/internal"
)

// ChatRequest is documented at https://docs.mistral.ai/api/#tag/chat/operation/chat_completion_v1_chat_completions_post
type ChatRequest struct {
	Model          string    `json:"model"`
	Temperature    float64   `json:"temperature,omitzero"` // [0, 2]
	TopP           float64   `json:"top_p,omitzero"`       // [0, 1]
	MaxTokens      int64     `json:"max_tokens,omitzero"`
	Stream         bool      `json:"stream"`
	Stop           []string  `json:"stop,omitzero"` // keywords to stop completion
	RandomSeed     int64     `json:"random_seed,omitzero"`
	Messages       []Message `json:"messages"`
	ResponseFormat struct {
		Type       string `json:"type,omitzero"` // "text", "json_object", "json_schema"
		JSONSchema struct {
			Name        string             `json:"name,omitzero"`
			Description string             `json:"description,omitzero"`
			Strict      bool               `json:"strict,omitzero"`
			Schema      *jsonschema.Schema `json:"schema,omitzero"`
		} `json:"json_schema,omitzero"`
	} `json:"response_format,omitzero"`
	Tools []Tool `json:"tools,omitzero"`
	// Alternative when forcing a specific function. This can probably be achieved
	// by providing a single tool and ToolChoice == "required".
	// ToolChoice struct {
	// 	Type     string `json:"type,omitzero"` // "function"
	// 	Function struct {
	// 		Name string `json:"name,omitzero"`
	// 	} `json:"function,omitzero"`
	// } `json:"tool_choice,omitzero"`
	ToolChoice       string  `json:"tool_choice,omitzero"`       // "auto", "none", "any", "required"
	PresencePenalty  float64 `json:"presence_penalty,omitzero"`  // [-2.0, 2.0]
	FrequencyPenalty float64 `json:"frequency_penalty,omitzero"` // [-2.0, 2.0]
	N                int64   `json:"n,omitzero"`                 // Number of choices; input tokens are only billed once
	Prediction       struct {
		// Enable users to specify expected results, optimizing response times by
		// leveraging known or predictable content. This approach is especially
		// effective for updating text documents or code files with minimal
		// changes, reducing latency while maintaining high-quality results.
		Type    string `json:"type,omitzero"` // "content"
		Content string `json:"content,omitzero"`
	} `json:"prediction,omitzero"`
	ParallelToolCalls bool   `json:"parallel_tool_calls,omitzero"` // defaults to true anyway
	PromptMode        string `json:"prompt_mode,omitzero"`         // "reasoning"
	SafePrompt        bool   `json:"safe_prompt,omitzero"`         // Injects a safety prompt

	// See https://docs.mistral.ai/capabilities/document/
	DocumentImageLimit int64 `json:"document_image_limit,omitzero"`
	DocumentPageLimit  int64 `json:"document_page_limit,omitzero"`
	IncludeImageBase64 bool  `json:"include_image_base64,omitzero"`
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
			if v.DecodeAs != nil {
				c.ResponseFormat.Type = "json_schema"
				// Mistral requires a name.
				c.ResponseFormat.JSONSchema.Name = "response"
				c.ResponseFormat.JSONSchema.Strict = true
				c.ResponseFormat.JSONSchema.Schema = internal.JSONSchemaFor(reflect.TypeOf(v.DecodeAs))
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
				c.Tools = make([]Tool, len(v.Tools))
				for i, t := range v.Tools {
					c.Tools[i].Type = "function"
					c.Tools[i].Function.Name = t.Name
					c.Tools[i].Function.Description = t.Description
					if c.Tools[i].Function.Parameters = t.InputSchemaOverride; c.Tools[i].Function.Parameters == nil {
						c.Tools[i].Function.Parameters = t.GetInputSchema()
					}
					// This costs a lot more.
					c.Tools[i].Function.Strict = true
				}
			}
		// GenOptionWeb is not supported. Web search is only available via the Agents API, not chat completions.
		// https://docs.mistral.ai/agents/tools/built-in/websearch
		case genai.GenOptionSeed:
			c.RandomSeed = int64(v)
		default:
			unsupported = append(unsupported, internal.TypeName(opt))
		}
	}

	if sp != "" {
		c.Messages = append(c.Messages, Message{Role: "system", Content: []Content{{Type: ContentText, Text: sp}}})
	}
	for i := range msgs {
		if len(msgs[i].ToolCallResults) > 1 {
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
		} else {
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

// Message is documented at https://docs.mistral.ai/api/#tag/chat/operation/chat_completion_v1_chat_completions_post
//
// See the python implementation at
// https://github.com/mistralai/mistral-common/blob/main/src/mistral_common/protocol/instruct/messages.py
type Message struct {
	Role       string     `json:"role"`             // "system", "assistant", "user", "tool"
	Content    []Content  `json:"content,omitzero"` // For system and assistant, must be at most a single string.
	Prefix     bool       `json:"prefix,omitzero"`  // Whether the message is a prefix
	ToolCalls  []ToolCall `json:"tool_calls,omitzero"`
	ToolCallID string     `json:"tool_call_id,omitzero"`
	Name       string     `json:"name,omitzero"`
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
		m.Content = make([]Content, len(in.Requests))
		for i := range in.Requests {
			if err := m.Content[i].FromRequest(&in.Requests[i]); err != nil {
				return fmt.Errorf("request #%d: %w", i, err)
			}
		}
	}
	if len(in.Replies) != 0 {
		for i := range in.Replies {
			if !in.Replies[i].ToolCall.IsZero() {
				m.ToolCalls = append(m.ToolCalls, ToolCall{})
				if err := m.ToolCalls[i].From(&in.Replies[i].ToolCall); err != nil {
					return fmt.Errorf("reply #%d: %w", i, err)
				}
				continue
			}
			if err := m.Content[i].FromReply(&in.Replies[i]); err != nil {
				return fmt.Errorf("reply #%d: %w", i, err)
			}
		}
	}
	if len(in.ToolCallResults) != 0 {
		// Process only the first tool call result in this method.
		// The Init method handles multiple tool call results by creating multiple messages.
		m.ToolCallID = in.ToolCallResults[0].ID
		m.Name = in.ToolCallResults[0].Name
		// Mistral supports images urls!!
		m.Content = []Content{{Type: ContentText, Text: in.ToolCallResults[0].Result}}
	}
	return nil
}

// Content is a piece of information sent back and forth.
//
// Only the user can send non-textual information.
type Content struct {
	Type ContentType `json:"type"`

	// Type == ContentText
	Text string `json:"text,omitzero"`

	// Type == ContentImageURL
	ImageURL struct {
		URL    string `json:"url,omitzero"`    // Can be inline.
		Detail string `json:"detail,omitzero"` // undocumented, likely "auto" like OpenAI
	} `json:"image_url,omitzero"`
	ModelConfig struct{} `json:"model_config,omitzero"`

	// Type == ContentDocumentURL
	DocumentURL  string `json:"document_url,omitzero"`
	DocumentName string `json:"document_name,omitzero"`

	// Type == ContentReference
	ReferenceIDs []int64 `json:"reference_ids,omitzero"`

	// Type == ContentInputAudio
	InputAudio []byte `json:"input_audio,omitzero"`
}

// FromRequest converts a genai.Request to a Content.
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
		case strings.HasPrefix(mimeType, "audio/"):
			if in.Doc.URL != "" {
				return errors.New("unsupported URL audio reference")
			}
			c.Type = ContentInputAudio
			c.InputAudio = data
		case strings.HasPrefix(mimeType, "image/"):
			c.Type = ContentImageURL
			if in.Doc.URL == "" {
				c.ImageURL.URL = fmt.Sprintf("data:%s;base64,%s", mimeType, base64.StdEncoding.EncodeToString(data))
			} else {
				c.ImageURL.URL = in.Doc.URL
			}
		case mimeType == "application/pdf":
			c.Type = ContentDocumentURL
			if in.Doc.URL == "" {
				// Inexplicably, Mistral supports inline images but not PDF.
				return errors.New("unsupported inline document")
			}
			c.DocumentName = in.Doc.GetFilename()
			c.DocumentURL = in.Doc.URL
			// text/plain, text/markdown
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

// FromReply converts a genai.Reply to a Content.
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
		case strings.HasPrefix(mimeType, "audio/"):
			if in.Doc.URL != "" {
				return errors.New("unsupported URL audio reference")
			}
			c.Type = ContentInputAudio
			c.InputAudio = data
		case strings.HasPrefix(mimeType, "image/"):
			c.Type = ContentImageURL
			if in.Doc.URL == "" {
				c.ImageURL.URL = fmt.Sprintf("data:%s;base64,%s", mimeType, base64.StdEncoding.EncodeToString(data))
			} else {
				c.ImageURL.URL = in.Doc.URL
			}
		case mimeType == "application/pdf":
			c.Type = ContentDocumentURL
			if in.Doc.URL == "" {
				// Inexplicably, Mistral supports inline images but not PDF.
				return errors.New("unsupported inline document")
			}
			c.DocumentName = in.Doc.GetFilename()
			c.DocumentURL = in.Doc.URL
			// text/plain, text/markdown
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

// ContentType is the type of a content block in a message.
//
// I got the whole list from an error message by sending a bad content type.
type ContentType string

// Content type values.
const (
	ContentText        ContentType = "text"
	ContentImageURL    ContentType = "image_url"
	ContentDocumentURL ContentType = "document_url"
	ContentReference   ContentType = "reference"
	ContentBBox        ContentType = "bbox"
	ContentFileURL     ContentType = "file_url"
	ContentInputAudio  ContentType = "input_audio"
	ContentFile        ContentType = "file"
	ContentThinking    ContentType = "thinking"
	ContentAudioURL    ContentType = "audio_url"
)

// Tool represents a tool definition.
type Tool struct {
	Type     string `json:"type,omitzero"` // "function"
	Function struct {
		Name        string             `json:"name,omitzero"`
		Description string             `json:"description,omitzero"`
		Strict      bool               `json:"strict,omitzero"`
		Parameters  *jsonschema.Schema `json:"parameters,omitzero"`
	} `json:"function,omitzero"`
}

// ChatResponse is the response from the chat API.
type ChatResponse struct {
	ID      string    `json:"id"`
	Object  string    `json:"object"` // "chat.completion"
	Model   string    `json:"model"`
	Created base.Time `json:"created"`
	Choices []struct {
		FinishReason FinishReason    `json:"finish_reason"`
		Index        int64           `json:"index"`
		Message      MessageResponse `json:"message"`
		Logprobs     struct{}        `json:"logprobs"`
	} `json:"choices"`
	Usage Usage `json:"usage"`
}

// ToResult converts the ChatResponse to a genai.Result.
func (c *ChatResponse) ToResult() (genai.Result, error) {
	out := genai.Result{
		// At the moment, Mistral doesn't support cached tokens.
		Usage: genai.Usage{
			InputTokens:  c.Usage.PromptTokens,
			OutputTokens: c.Usage.CompletionTokens,
			TotalTokens:  c.Usage.TotalTokens,
		},
	}
	if len(c.Choices) != 1 {
		return out, fmt.Errorf("server returned an unexpected number of choices, expected 1, got %d", len(c.Choices))
	}
	out.Usage.FinishReason = c.Choices[0].FinishReason.ToFinishReason()
	err := c.Choices[0].Message.To(&out.Message)
	return out, err
}

// FinishReason represents the reason a generation finished.
type FinishReason string

// Finish reason values.
const (
	FinishStop          FinishReason = "stop"
	FinishLength        FinishReason = "length"
	FinishToolCalls     FinishReason = "tool_calls"
	FinishContentFilter FinishReason = "content_filter"
)

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

// Usage represents token usage information.
type Usage struct {
	PromptTokens       int64 `json:"prompt_tokens"`
	CompletionTokens   int64 `json:"completion_tokens"`
	TotalTokens        int64 `json:"total_tokens"`
	PromptAudioSeconds int64 `json:"prompt_audio_seconds"`
}

// MessageResponse represents a message in the API response.
type MessageResponse struct {
	Role      string     `json:"role"`
	Content   string     `json:"content"`
	Prefix    bool       `json:"prefix"`
	ToolCalls []ToolCall `json:"tool_calls"`
}

// To converts a MessageResponse to a genai.Message.
func (m *MessageResponse) To(out *genai.Message) error {
	if m.Content != "" {
		out.Replies = []genai.Reply{{Text: m.Content}}
	}
	for i := range m.ToolCalls {
		out.Replies = append(out.Replies, genai.Reply{})
		m.ToolCalls[i].To(&out.Replies[len(out.Replies)-1].ToolCall)
	}
	return nil
}

// ToolCall represents a tool call.
type ToolCall struct {
	ID       string `json:"id,omitzero"`
	Type     string `json:"type,omitzero"`
	Function struct {
		Name      string `json:"name,omitzero"`
		Arguments string `json:"arguments,omitzero"`
	} `json:"function,omitzero"`
	Index int64 `json:"index,omitzero"`
}

// From converts a genai.ToolCall to a ToolCall.
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

// To converts a ToolCall to a genai.ToolCall.
func (t *ToolCall) To(out *genai.ToolCall) {
	out.ID = t.ID
	out.Name = t.Function.Name
	out.Arguments = t.Function.Arguments
}

// ChatStreamChunkResponse represents a streaming chunk from the chat API.
type ChatStreamChunkResponse struct {
	ID      string    `json:"id"`
	Object  string    `json:"object"` // "chat.completion.chunk"
	Created base.Time `json:"created"`
	Model   string    `json:"model"`
	Choices []struct {
		Index int64 `json:"index"`
		Delta struct {
			Role      string     `json:"role"`
			Content   string     `json:"content"`
			ToolCalls []ToolCall `json:"tool_calls"`
		} `json:"delta"`
		FinishReason FinishReason `json:"finish_reason"`
		Logprobs     struct{}     `json:"logprobs"`
	} `json:"choices"`
	Usage Usage  `json:"usage"`
	P     string `json:"p"` // "abcdefghijklmnopqrstu" WTF?
}

// Model is documented at https://docs.mistral.ai/api/#tag/models/operation/retrieve_model_v1_models__model_id__get
type Model struct {
	ID           string    `json:"id"`
	Object       string    `json:"object"`
	Created      base.Time `json:"created"`
	OwnedBy      string    `json:"owned_by"`
	Capabilities struct {
		Audio                      bool `json:"audio"`
		AudioSpeech                bool `json:"audio_speech"`
		AudioTranscription         bool `json:"audio_transcription"`
		AudioTranscriptionRealtime bool `json:"audio_transcription_realtime"`
		Classification             bool `json:"classification"`
		CompletionChat             bool `json:"completion_chat"`
		CompletionFim              bool `json:"completion_fim"`
		FineTuning                 bool `json:"fine_tuning"`
		FunctionCalling            bool `json:"function_calling"`
		Moderation                 bool `json:"moderation"`
		OCR                        bool `json:"ocr"`
		Reasoning                  bool `json:"reasoning"`
		Vision                     bool `json:"vision"`
	} `json:"capabilities"`
	Name                        string   `json:"name"`
	Description                 string   `json:"description"`
	MaxContextLength            int64    `json:"max_context_length"`
	Aliases                     []string `json:"aliases"`
	Deprecation                 string   `json:"deprecation"`
	DeprecationReplacementModel string   `json:"deprecation_replacement_model,omitzero"`
	DefaultModelTemperature     float64  `json:"default_model_temperature"`
	Type                        string   `json:"type"` // "base"
}

// GetID returns the model ID.
func (m *Model) GetID() string {
	return m.ID
}

func (m *Model) String() string {
	var caps []string
	if m.Capabilities.CompletionChat {
		caps = append(caps, "chat")
	}
	if m.Capabilities.CompletionFim {
		caps = append(caps, "fim")
	}
	if m.Capabilities.FunctionCalling {
		caps = append(caps, "function")
	}
	if m.Capabilities.FineTuning {
		caps = append(caps, "fine-tuning")
	}
	if m.Capabilities.Vision {
		caps = append(caps, "vision")
	}
	suffix := ""
	if m.Deprecation != "" {
		suffix += " (deprecated)"
	}
	prefix := m.ID
	if m.ID != m.Name {
		prefix += " (" + m.Name + ")"
	}
	// Not including Created and Description because Created is not set and Description is not useful.
	return fmt.Sprintf("%s: %s Context: %d%s", prefix, strings.Join(caps, "/"), m.MaxContextLength, suffix)
}

// Context returns the context window size.
func (m *Model) Context() int64 {
	return m.MaxContextLength
}

// ModelsResponse represents the response structure for Mistral models listing.
type ModelsResponse struct {
	Object string  `json:"object"` // list
	Data   []Model `json:"data"`
}

// ToModels converts Mistral models to genai.Model interfaces.
func (r *ModelsResponse) ToModels() []genai.Model {
	// As of 2025-08, Mistral returns duplicate models voxtral-mini-latest and voxtral-mini-2507. Filter them
	// out so the client is not confused.
	seen := make(map[string]struct{}, len(r.Data))
	models := make([]genai.Model, 0, len(r.Data))
	for i := range r.Data {
		id := r.Data[i].ID
		if _, ok := seen[id]; ok {
			continue
		}
		seen[id] = struct{}{}
		models = append(models, &r.Data[i])
	}
	return models
}

//

// ErrorResponse is the most goddam unstructured way to process errors. Basically what happens is that any
// point in the Mistral stack can return an error and each python library generates a different structure.
type ErrorResponse struct {
	// When simple issue like auth failure.
	// Message   string `json:"message"`
	RequestID string `json:"request_id"`

	// First error type
	Object string `json:"object"` // "error"
	// Message string `json:"message"`
	Type  string      `json:"type"`
	Param string      `json:"param"`
	Code  json.Number `json:"code"` // Sometimes a string, sometimes a int64.

	// Second error type
	Detail ErrorDetails `json:"detail"`

	// Third error type
	// Object  string `json:"object"` // error
	Message ErrorMessage `json:"message"`
	// Type  string `json:"type"`
	// Param string `json:"param"`
	// Code  int64  `json:"code"`
}

func (er *ErrorResponse) Error() string {
	out := er.Type
	if out != "" {
		out += ": "
	}
	if s := er.Detail.String(); s != "" {
		return out + s
	}
	return out + er.Message.Detail.String()
}

// IsAPIError returns true.
func (er *ErrorResponse) IsAPIError() bool {
	return true
}

// ErrorDetail can be either a struct or a string. When a string, it decodes into Msg.
type ErrorDetail struct {
	Type string            `json:"type"` // "string_type", "missing"
	Msg  string            `json:"msg"`
	Loc  []json.RawMessage `json:"loc"` // to be joined, a mix of string and number
	// Input is either a list or an instance of struct { Type string `json:"type"` }.
	Input json.RawMessage `json:"input"`
	Ctx   json.RawMessage `json:"ctx"`
	URL   string          `json:"url"`
}

func (ed *ErrorDetail) String() string {
	if ed.Type == "" && len(ed.Loc) == 0 {
		// This was actually a string
		return ed.Msg
	}
	return fmt.Sprintf("%s: %s at %s", ed.Type, ed.Msg, formatErrorLoc(ed.Loc))
}

func formatErrorLoc(loc []json.RawMessage) string {
	parts := make([]string, len(loc))
	for i, raw := range loc {
		var s string
		if err := json.Unmarshal(raw, &s); err == nil {
			parts[i] = s
		} else {
			parts[i] = string(raw)
		}
	}
	return "[" + strings.Join(parts, " ") + "]"
}

// ErrorDetails represents a collection of error details.
type ErrorDetails []ErrorDetail

func (ed *ErrorDetails) String() string {
	var out strings.Builder
	for _, e := range *ed {
		out.WriteString(e.String())
	}
	return out.String()
}

// UnmarshalJSON implements json.Unmarshaler.
func (ed *ErrorDetails) UnmarshalJSON(b []byte) error {
	d := json.NewDecoder(bytes.NewReader(b))
	if !internal.BeLenient {
		d.DisallowUnknownFields()
	}
	if err := d.Decode((*[]ErrorDetail)(ed)); err == nil {
		return nil
	}
	s := ""
	if err := json.Unmarshal(b, &s); err != nil {
		return err
	}
	*ed = []ErrorDetail{{Msg: s}}
	return nil
}

// ErrorMessage represents an error message that can be a string or object.
type ErrorMessage struct {
	Detail ErrorDetails `json:"detail"`
}

// UnmarshalJSON implements json.Unmarshaler.
func (er *ErrorMessage) UnmarshalJSON(b []byte) error {
	s := ""
	if err := json.Unmarshal(b, &s); err == nil {
		er.Detail = ErrorDetails{{Msg: s}}
		return nil
	}
	var x struct {
		Detail ErrorDetails `json:"detail"`
	}
	d := json.NewDecoder(bytes.NewReader(b))
	if !internal.BeLenient {
		d.DisallowUnknownFields()
	}
	if err := d.Decode(&x); err != nil {
		return err
	}
	er.Detail = x.Detail
	return nil
}
