// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Package anthropic implements a client for the Anthropic API, to use Claude.
//
// It is described at
// https://docs.anthropic.com/en/api/
package anthropic

// See official client at https://github.com/anthropics/anthropic-sdk-go

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/invopop/jsonschema"
	"github.com/maruel/genai"
	"github.com/maruel/genai/internal"
	"github.com/maruel/httpjson"
	"github.com/maruel/roundtrippers"
)

// Scoreboard for Anthropic.
//
// # Warnings
//
//   - No Anthropic models support structured output, you have to use tool calling instead.
//   - Thinking is set to false because it doesn't happen systematically and the smoke tests do not trigger the
//     condition. This is a bug in the smoke test.
var Scoreboard = genai.Scoreboard{
	Scenarios: []genai.Scenario{
		{
			In:     []genai.Modality{genai.ModalityText},
			Out:    []genai.Modality{genai.ModalityText},
			Models: []string{"claude-3-haiku-20240307", "claude-2.0", "claude-2.1", "claude-3-opus-20240229", "claude-3-sonnet-20240229"},
			Chat: genai.Functionality{
				Inline:             true,
				URL:                false,
				Thinking:           false,
				ReportTokenUsage:   true,
				ReportFinishReason: true,
				MaxTokens:          true,
				StopSequence:       true,
				Tools:              false,
				JSON:               false,
				JSONSchema:         false,
			},
			ChatStream: genai.Functionality{
				Inline:             true,
				URL:                false,
				Thinking:           false,
				ReportTokenUsage:   true,
				ReportFinishReason: true,
				MaxTokens:          true,
				StopSequence:       true,
				Tools:              false,
				JSON:               false,
				JSONSchema:         false,
			},
		},
		{
			In:     []genai.Modality{genai.ModalityText, genai.ModalityImage, genai.ModalityPDF},
			Out:    []genai.Modality{genai.ModalityText},
			Models: []string{"claude-3-5-haiku-20241022", "claude-3-5-sonnet-20240620", "claude-3-5-sonnet-20241022"},
			Chat: genai.Functionality{
				Inline:             true,
				URL:                true,
				Thinking:           false,
				ReportTokenUsage:   true,
				ReportFinishReason: true,
				MaxTokens:          true,
				StopSequence:       true,
				Tools:              true,
				JSON:               false,
				JSONSchema:         false,
			},
			ChatStream: genai.Functionality{
				Inline:             true,
				URL:                true,
				Thinking:           false,
				ReportTokenUsage:   true,
				ReportFinishReason: true,
				MaxTokens:          true,
				StopSequence:       true,
				Tools:              true,
				JSON:               false,
				JSONSchema:         false,
			},
		},
		{
			In:     []genai.Modality{genai.ModalityText, genai.ModalityImage, genai.ModalityPDF},
			Out:    []genai.Modality{genai.ModalityText},
			Models: []string{"claude-3-7-sonnet-20250219", "claude-opus-4-20250514", "claude-sonnet-4-20250514"},
			Chat: genai.Functionality{
				Inline:             true,
				URL:                true,
				Thinking:           false,
				ReportTokenUsage:   true,
				ReportFinishReason: true,
				MaxTokens:          true,
				StopSequence:       true,
				Tools:              true,
				JSON:               false,
				JSONSchema:         false,
			},
			ChatStream: genai.Functionality{
				Inline:             true,
				URL:                true,
				Thinking:           false,
				ReportTokenUsage:   true,
				ReportFinishReason: true,
				MaxTokens:          true,
				StopSequence:       true,
				Tools:              true,
				JSON:               false,
				JSONSchema:         false,
			},
		},
	},
}

// ChatOptions includes Anthropic specific options.
type ChatOptions struct {
	genai.ChatOptions

	// ThinkingBudget is the maximum number of tokens the LLM can use to think about the answer. When 0,
	// thinking is disabled. It generally must be above 1024 and below MaxTokens.
	ThinkingBudget int64
	// MessagesToCache specify the number of messages to cache in the request.
	//
	// By default, the system prompt and tools will be cached.
	//
	// https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching
	MessagesToCache int
}

// https://docs.anthropic.com/en/api/messages
type ChatRequest struct {
	Model    string    `json:"model,omitzero"`
	MaxToks  int64     `json:"max_tokens"`
	Messages []Message `json:"messages"`
	Metadata struct {
		UserID string `json:"user_id,omitzero"`
	} `json:"metadata,omitzero"`
	StopSequences []string        `json:"stop_sequences,omitzero"`
	Stream        bool            `json:"stream,omitzero"`
	System        []SystemMessage `json:"system,omitzero"`      // Must be type "text"
	Temperature   float64         `json:"temperature,omitzero"` // [0, 1]
	Thinking      Thinking        `json:"thinking,omitzero"`
	ToolChoice    ToolChoice      `json:"tool_choice,omitzero"`
	Tools         []Tool          `json:"tools,omitzero"`
	TopK          int64           `json:"top_k,omitzero"` // [1, ]
	TopP          float64         `json:"top_p,omitzero"` // [0, 1]
}

// Init initializes the provider specific completion request with the generic completion request.
func (c *ChatRequest) Init(msgs genai.Messages, opts genai.Validatable, model string) error {
	c.Model = model
	var errs []error
	var unsupported []string
	msgToCache := 0
	// Default to disabled thinking.
	c.Thinking.Type = "disabled"
	if opts != nil {
		if err := opts.Validate(); err != nil {
			errs = append(errs, err)
		} else {
			switch v := opts.(type) {
			case *ChatOptions:
				unsupported, errs = c.initOptions(&v.ChatOptions)
				msgToCache = v.MessagesToCache
				if v.ThinkingBudget > 0 {
					if v.ThinkingBudget >= v.MaxTokens {
						errs = append(errs, fmt.Errorf("invalid ThinkingBudget(%d) >= MaxTokens(%d)", v.ThinkingBudget, v.MaxTokens))
					}
					// https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking
					// Thinking isn’t compatible with temperature, top_p, or top_k modifications as well as forced tool use.
					c.Thinking.BudgetTokens = v.ThinkingBudget
					c.Thinking.Type = "enabled"
				}
			case *genai.ChatOptions:
				unsupported, errs = c.initOptions(v)
			default:
				errs = append(errs, fmt.Errorf("unsupported options type %T", opts))
			}
		}
	}
	if c.MaxToks == 0 {
		// TODO: Query the model. Anthropic requires a value! This is quite annoying.
		if strings.HasPrefix(model, "claude-3-opus-") || strings.HasPrefix(model, "claude-3-haiku-") {
			c.MaxToks = 4096
		} else if strings.HasPrefix(model, "claude-3-5-") {
			c.MaxToks = 8192
		} else if strings.HasPrefix(model, "claude-3-7-") {
			c.MaxToks = 64000
		} else if strings.HasPrefix(model, "claude-4-sonnet-") {
			c.MaxToks = 64000
		} else if strings.HasPrefix(model, "claude-4-opus-") {
			c.MaxToks = 32000
		} else {
			// Default value for new models.
			c.MaxToks = 32000
		}
	}

	if err := msgs.Validate(); err != nil {
		errs = append(errs, err)
	} else {
		c.Messages = make([]Message, 0, len(msgs))
		for i := range msgs {
			switch msgs[i].Role {
			case genai.User, genai.Assistant:
				c.Messages = append(c.Messages, Message{})
				if err := c.Messages[len(c.Messages)-1].From(&msgs[i]); err != nil {
					errs = append(errs, fmt.Errorf("message %d: %w", i, err))
				}
				if i == msgToCache-1 {
					c.Messages[i].CacheControl.Type = "ephemeral"
				}
			default:
				errs = append(errs, fmt.Errorf("message %d: unsupported role %q", i, msgs[i].Role))
				continue
			}
		}
	}
	if len(unsupported) > 0 {
		// If we have unsupported features but no other errors, return a continuable error
		if len(errs) == 0 {
			return &genai.UnsupportedContinuableError{Unsupported: unsupported}
		}
		// Otherwise, add the unsupported features to the error list
		errs = append(errs, &genai.UnsupportedContinuableError{Unsupported: unsupported})
	}
	return errors.Join(errs...)
}

func (c *ChatRequest) SetStream(stream bool) {
	c.Stream = stream
}

func (c *ChatRequest) initOptions(v *genai.ChatOptions) ([]string, []error) {
	var unsupported []string
	var errs []error
	c.MaxToks = v.MaxTokens
	c.Temperature = v.Temperature
	if v.SystemPrompt != "" {
		c.System = []SystemMessage{
			{
				Type: "text",
				Text: v.SystemPrompt,
			},
		}
		// TODO: Add automatic caching.
		// c.System[0].CacheControl.Type = "ephemeral"
	}
	c.TopP = v.TopP
	if v.Seed != 0 {
		unsupported = append(unsupported, "Seed")
	}
	c.TopK = v.TopK
	c.StopSequences = v.Stop
	if v.ReplyAsJSON {
		errs = append(errs, errors.New("unsupported option ReplyAsJSON"))
	}
	if v.DecodeAs != nil {
		errs = append(errs, errors.New("unsupported option DecodeAs"))
	}
	if len(v.Tools) != 0 {
		// We need to discard claude 2 and 3. This is a bit annoying to have to hardcode this.
		if strings.HasPrefix(c.Model, "claude-2") || strings.HasPrefix(c.Model, "claude-3-haiku") ||
			strings.HasPrefix(c.Model, "claude-3-sonnet") || strings.HasPrefix(c.Model, "claude-3-opus") {
			errs = append(errs, errors.New("unsupported option Tools"))
		}
		switch v.ToolCallRequest {
		case genai.ToolCallAny:
			c.ToolChoice.Type = ToolChoiceAuto
		case genai.ToolCallRequired:
			c.ToolChoice.Type = ToolChoiceAny
		case genai.ToolCallNone:
			c.ToolChoice.Type = ToolChoiceNone
		}
		c.Tools = make([]Tool, len(v.Tools))
		for i, t := range v.Tools {
			// Weirdly enough, we must not set the type. See example at
			// https://docs.anthropic.com/en/docs/build-with-claude/tool-use/overview
			// c.Tools[i].Type = "custom"
			c.Tools[i].Name = t.Name
			c.Tools[i].Description = t.Description
			if c.Tools[i].InputSchema = t.InputSchemaOverride; c.Tools[i].InputSchema == nil {
				c.Tools[i].InputSchema = t.GetInputSchema()
			}
		}
	}
	return unsupported, errs
}

// SystemMessage is used in the system prompt.
type SystemMessage struct {
	Type         string `json:"type,omitzero"` // "text"
	Text         string `json:"text,omitzero"`
	CacheControl struct {
		Type string `json:"type,omitzero"` // "ephemeral"
	} `json:"cache_control,omitzero"`
	Citations []Citation `json:"citations,omitzero"`
}

// https://docs.anthropic.com/en/api/messages
type Message struct {
	Type string `json:"type,omitzero"` // "message"
	Role string `json:"role"`          // "assistant", "user"
	// Anthropic's Content doesn't distinguish between actual content (text,
	// documents) and tool use.
	Content      []Content `json:"content"`
	CacheControl struct {
		Type string `json:"type,omitzero"` // "ephemeral"
	} `json:"cache_control,omitzero"`
}

func (m *Message) From(in *genai.Message) error {
	switch role := in.Role; role {
	case genai.Assistant, genai.User:
		m.Role = string(in.Role)
	default:
		return fmt.Errorf("unsupported role %q", role)
	}
	m.Content = make([]Content, len(in.Contents)+len(in.ToolCalls)+len(in.ToolCallResults))
	for i := range in.Contents {
		if err := m.Content[i].FromContent(&in.Contents[i]); err != nil {
			return fmt.Errorf("block %d: %w", i, err)
		}
	}
	offset := len(in.Contents)
	for i := range in.ToolCalls {
		if err := m.Content[offset+i].FromToolCall(&in.ToolCalls[i]); err != nil {
			return fmt.Errorf("block %d: %w", offset+i, err)
		}
	}
	offset += len(in.ToolCalls)
	for i := range in.ToolCallResults {
		if err := m.Content[offset+i].FromToolCallResult(&in.ToolCallResults[i]); err != nil {
			return fmt.Errorf("block %d: %w", offset+i, err)
		}
	}
	return nil
}

func (m *Message) To(out *genai.Message) error {
	switch role := m.Role; role {
	case "assistant", "user":
		out.Role = genai.Role(role)
	default:
		return fmt.Errorf("unsupported role %q", role)
	}
	// We need to split actual content and tool calls.
	for i := range m.Content {
		switch m.Content[i].Type {
		case ContentText, ContentThinking, ContentRedactedThinking:
			out.Contents = append(out.Contents, genai.Content{})
			if err := m.Content[i].ToContent(&out.Contents[len(out.Contents)-1]); err != nil {
				return fmt.Errorf("block %d: %w", i, err)
			}
		case ContentToolUse:
			out.ToolCalls = append(out.ToolCalls, genai.ToolCall{})
			if err := m.Content[i].ToToolCall(&out.ToolCalls[len(out.ToolCalls)-1]); err != nil {
				return fmt.Errorf("block %d: %w", i, err)
			}
			// ContentImage, ContentDocument, ContentToolResult
		default:
			return fmt.Errorf("unsupported content type %q", m.Content[i].Type)
		}
	}
	return nil
}

type Content struct {
	Type ContentType `json:"type"`
	// Type == "text"
	Text string `json:"text,omitzero"`

	// Type == "thinking"
	Thinking  string `json:"thinking,omitzero"`
	Signature []byte `json:"signature,omitzero"`

	// Type == "redacted_thinking"
	Data string `json:"data,omitzero"`

	// Type == "text", "image", "tool_use", "tool_result", "document"
	CacheControl struct {
		Type string `json:"type,omitzero"` // "ephemeral"
	} `json:"cache_control,omitzero"`

	// Type == "text", "document"
	Citations []Citation `json:"citations,omitzero"`

	// Type == "image", "document"
	Source struct {
		// Content.Type == "image": "base64", "url"
		// Content.Type == "document": "base64, "url", "text", "content"
		Type string `json:"type,omitzero"`

		// Type == "base64", "url", "text"
		// Content.Type == "image": "image/jpeg", "image/png", "image/gif", "image/webp"
		// Content.Type == "document": "application/pdf"
		MediaType string `json:"media_type,omitzero"`

		// Type == "base64", "text"
		Data []byte `json:"data,omitzero"` // base64 encoded if base64, else as is.

		// Type == "url"
		URL string `json:"url,omitzero"`

		// Type == "content"
		// Only "text" and "image" are allowed.
		Content []Content `json:"content,omitzero"`
	} `json:"source,omitzero"`

	// Type == "tool_use"
	ID    string `json:"id,omitzero"`
	Input any    `json:"input,omitzero"`
	Name  string `json:"name,omitzero"`

	// Type == "tool_result"
	ToolUseID string `json:"tool_use_id,omitzero"`
	IsError   bool   `json:"is_error,omitzero"`
	// Only "text" and "image" are allowed.
	Content []Content `json:"content,omitzero"`

	// "document"
	Context string `json:"context,omitzero"`
	Title   string `json:"title,omitzero"`
}

func (c *Content) FromContent(in *genai.Content) error {
	if in.Text != "" {
		c.Type = ContentText
		c.Text = in.Text
		return nil
	}
	if in.Thinking != "" {
		c.Type = ContentThinking
		c.Thinking = in.Thinking
		if in.Opaque != nil {
			if b, ok := in.Opaque["signature"].([]byte); ok {
				c.Signature = b
			}
		}
		return nil
	}
	if in.Opaque != nil {
		if s, ok := in.Opaque["redacted_thinking"].(string); ok {
			c.Type = ContentRedactedThinking
			c.Data = s
			return nil
		}
		return fmt.Errorf("unexpected Opaque %v", in.Opaque)
	}

	mimeType, data, err := in.ReadDocument(10 * 1024 * 1024)
	if err != nil {
		return err
	}
	// Anthropic require a mime-type to determine if image or PDF.
	if mimeType == "" {
		return fmt.Errorf("unspecified mime type for URL %q", in.URL)
	}
	c.CacheControl.Type = "ephemeral"
	switch {
	case strings.HasPrefix(mimeType, "image/"):
		c.Type = ContentImage
		if in.URL != "" {
			c.Source.Type = "url"
			c.Source.URL = in.URL
		} else {
			c.Source.MediaType = mimeType
			c.Source.Type = "base64"
			c.Source.Data = data
		}
	case mimeType == "application/pdf":
		c.Type = ContentDocument
		if in.URL != "" {
			c.Source.Type = "url"
			c.Source.URL = in.URL
		} else {
			c.Source.MediaType = mimeType
			c.Source.Type = "base64"
			c.Source.Data = data
		}
	default:
		return fmt.Errorf("unsupported content mime-type %s", mimeType)
	}
	return nil
}

func (c *Content) FromToolCall(in *genai.ToolCall) error {
	c.Type = ContentToolUse
	c.ID = in.ID
	c.Name = in.Name
	if err := json.Unmarshal([]byte(in.Arguments), &c.Input); err != nil {
		return fmt.Errorf("failed to marshal input: %w; for tool call: %#v", err, in)
	}
	return nil
}

func (c *Content) FromToolCallResult(in *genai.ToolCallResult) error {
	// TODO: Support text citation.
	// TODO: Support image.
	c.Type = ContentToolResult
	c.ToolUseID = in.ID
	c.IsError = false // Interesting!
	c.Content = []Content{{Type: ContentText, Text: in.Result}}
	return nil
}

func (c *Content) ToContent(out *genai.Content) error {
	switch c.Type {
	case ContentText:
		out.Text = c.Text
	case ContentThinking:
		out.Thinking = c.Thinking
		out.Opaque = map[string]any{"signature": c.Signature}
	case ContentRedactedThinking:
		out.Opaque = map[string]any{"redacted_thinking": c.Signature}
	default:
		return fmt.Errorf("unsupported content type %q", c.Type)
	}
	return nil
}

func (c *Content) ToToolCall(out *genai.ToolCall) error {
	switch c.Type {
	case ContentToolUse:
		out.ID = c.ID
		out.Name = c.Name
		raw, err := json.Marshal(c.Input)
		if err != nil {
			return fmt.Errorf("failed to marshal input: %w; for tool call: %#v", err, c)
		}
		out.Arguments = string(raw)
	default:
		return fmt.Errorf("unsupported content type %q", c.Type)
	}
	return nil
}

type ContentType string

const (
	ContentText             ContentType = "text"
	ContentImage            ContentType = "image"
	ContentToolUse          ContentType = "tool_use"
	ContentToolResult       ContentType = "tool_result"
	ContentDocument         ContentType = "document"
	ContentThinking         ContentType = "thinking"
	ContentRedactedThinking ContentType = "redacted_thinking"
)

// Citation is used both for system message and user message.
//
// https://docs.anthropic.com/en/api/messages#body-messages-content-citations
// https://docs.anthropic.com/en/api/messages#body-system-citations
type Citation struct {
	Type string `json:"type,omitzero"` // "char_location", "page_location", "content_block_location", "web_search_result_location"

	// Content.Type == "text"
	CitedText     string `json:"cited_text,omitzero"`
	DocumentIndex int64  `json:"document_index,omitzero"`
	DocumentTitle string `json:"document_title,omitzero"`
	// Type == "char_location"
	EndCharIndex   int64 `json:"end_char_index,omitzero"`
	StartCharIndex int64 `json:"start_char_index,omitzero"`
	// Type == "page_location"
	EndPageNumber   int64 `json:"end_page_number,omitzero"`
	StartPageNumber int64 `json:"start_page_number,omitzero"`
	// Type == "content_block_location"
	EndBlockIndex   int64 `json:"end_block_index,omitzero"`
	StartBlockIndex int64 `json:"start_block_index,omitzero"`
	// Type == "web_search_result_location"
	EncryptedIndex string `json:"encrypted_index,omitzero"`
	Title          string `json:"title,omitzero"`
	URL            string `json:"url,omitzero"`

	// Content.Type == "document"
	Enabled bool `json:"enabled,omitzero"`
}

type Thinking struct {
	BudgetTokens int64  `json:"budget_tokens,omitzero"` // >1024 and less than max_tokens
	Type         string `json:"type,omitzero"`          // "enabled", "disabled"
}

// https://docs.anthropic.com/en/api/messages#body-tool-choice
type ToolChoiceType string

const (
	// ToolChoiceAuto tells the LLM it is free to use a tool if desired.
	ToolChoiceAuto ToolChoiceType = "auto"
	// ToolChoiceAny tells the LLM must use a tool.
	ToolChoiceAny ToolChoiceType = "any"
	// ToolChoiceTool tells the LLM it must use the tool named in ToolChoice.Name.
	ToolChoiceTool ToolChoiceType = "tool"
	// ToolChoiceNone tells the LLM no tool must be used.
	ToolChoiceNone ToolChoiceType = "none"
)

type ToolChoice struct {
	Type ToolChoiceType `json:"type,omitzero"`

	// Type == "auto", "any", "tool"
	// Defaults to allow multiple tool calls simultaneously.
	DisableParallelToolUse bool `json:"disable_parallel_tool_use,omitzero"`

	// Type == "tool"
	Name string `json:"name,omitzero"`
}

// https://docs.anthropic.com/en/api/messages#body-tools
type Tool struct {
	Type string `json:"type,omitzero"` // "custom", "computer_20241022", "computer_20250124", "bash_20241022", "bash_20250124", "text_editor_20241022", "text_editor_20250124"
	// Type == "custom"
	Description string             `json:"description,omitzero"`
	InputSchema *jsonschema.Schema `json:"input_schema,omitzero"`

	// Type == "custom": tool name
	// Type == "computer_20241022", "computer_20250124": "computer"
	// Type == "bash_20241022", "bash_20250124": "bash"
	// Type == "text_editor_20241022", "text_editor_20250124": "str_replace_editor"
	Name string `json:"name,omitzero"`

	// Type == "custom", "computer_20241022", "computer_20250124", "bash_20241022", "bash_20250124", "text_editor_20241022", "text_editor_20250124"
	CacheControl struct {
		Type string `json:"type,omitzero"` // "ephemeral"
	} `json:"cache_control,omitzero"`

	// Type == "computer_20241022", "computer_20250124"
	DisplayNumber   int64 `json:"display_number,omitzero"`
	DisplayHeightPX int64 `json:"display_height_px,omitzero"`
	DisplayWidthPX  int64 `json:"display_width_px,omitzero"`
}

type ChatResponse struct {
	Message                 // Role is always "assistant"
	ID           string     `json:"id"`
	Model        string     `json:"model"`
	StopReason   StopReason `json:"stop_reason"`
	StopSequence string     `json:"stop_sequence"`
	Type         string     `json:"type"` // "message"
	Usage        Usage      `json:"usage"`
}

func (c *ChatResponse) ToResult() (genai.ChatResult, error) {
	out := genai.ChatResult{
		Usage: genai.Usage{
			InputTokens:       c.Usage.InputTokens,
			InputCachedTokens: c.Usage.CacheReadInputTokens,
			OutputTokens:      c.Usage.OutputTokens,
			FinishReason:      c.StopReason.ToFinishReason(),
		},
	}
	err := c.To(&out.Message)
	return out, err
}

type StopReason string

const (
	StopEndTurn   StopReason = "end_turn"
	StopToolUse   StopReason = "tool_use"
	StopSequence  StopReason = "stop_sequence"
	StopMaxTokens StopReason = "max_tokens"
)

func (s StopReason) ToFinishReason() genai.FinishReason {
	switch s {
	case StopEndTurn:
		return genai.FinishedStop
	case StopToolUse:
		return genai.FinishedToolCalls
	case StopSequence:
		return genai.FinishedStopSequence
	case StopMaxTokens:
		return genai.FinishedLength
	default:
		return genai.FinishReason(s)
	}
}

// https://docs.anthropic.com/en/api/messages-streaming
// Each stream uses the following event flow:
//   - message_start: contains a Message object with empty content.
//   - A series of content blocks, each of which have a content_block_start, one
//     or more content_block_delta events, and a content_block_stop event. Each
//     content block will have an index that corresponds to its index in the final
//     Message content array.
//   - One or more message_delta events, indicating top-level changes to the
//     final Message object.
//   - A final message_stop event.
type ChatStreamChunkResponse struct {
	Type ChunkType `json:"type"`

	// Type == "message_start"
	Message struct {
		ID           string     `json:"id"`
		Type         string     `json:"type"` // "message", "thinking"
		Role         string     `json:"role"`
		Model        string     `json:"model"`
		Content      []string   `json:"content"`
		StopReason   StopReason `json:"stop_reason"`
		StopSequence string     `json:"stop_sequence"`
		Usage        Usage      `json:"usage"`
	} `json:"message"`

	Index int64 `json:"index"`

	// Type == "content_block_start"
	ContentBlock struct {
		Type ContentType `json:"type"`

		// Type == "text"
		Text string `json:"text"`

		// Type == "thinking"
		Thinking  string `json:"thinking"`
		Signature []byte `json:"signature"` // Never actually filed but present on content_block_start.

		// Type == "tool_use"
		ID    string `json:"id"`
		Name  string `json:"name"`
		Input any    `json:"input"`
	} `json:"content_block"`

	// Type == "content_block_delta"
	Delta struct {
		Type DeltaType `json:"type"`

		// Type == "text_delta"
		Text string `json:"text"`

		// Type == "input_json_delta"
		PartialJSON string `json:"partial_json"`

		// Type == "thinking_delta"
		Thinking string `json:"thinking"`

		// Type == "signature_delta"
		Signature []byte `json:"signature"`

		// Type == ""
		StopReason   StopReason `json:"stop_reason"`
		StopSequence string     `json:"stop_sequence"`
	} `json:"delta"`
	Usage Usage `json:"usage"`
}

type ChunkType string

const (
	ChunkMessageStart      ChunkType = "message_start"
	ChunkMessageDelta      ChunkType = "message_delta"
	ChunkMessageStop       ChunkType = "message_stop"
	ChunkContentBlockStart ChunkType = "content_block_start"
	ChunkContentBlockDelta ChunkType = "content_block_delta"
	ChunkContentBlockStop  ChunkType = "content_block_stop"
	ChunkPing              ChunkType = "ping"
)

type DeltaType string

const (
	DeltaText      DeltaType = "text_delta"
	DeltaInputJSON DeltaType = "input_json_delta"
	DeltaThinking  DeltaType = "thinking_delta"
	DeltaSignature DeltaType = "signature_delta"
)

type Usage struct {
	InputTokens              int64  `json:"input_tokens"`
	OutputTokens             int64  `json:"output_tokens"`
	CacheCreationInputTokens int64  `json:"cache_creation_input_tokens"`
	CacheReadInputTokens     int64  `json:"cache_read_input_tokens"`
	ServiceTier              string `json:"service_tier"` // "standard"
}

type Model struct {
	CreatedAt   time.Time `json:"created_at"`
	DisplayName string    `json:"display_name"`
	ID          string    `json:"id"`
	Type        string    `json:"type"`
}

func (m *Model) GetID() string {
	return m.ID
}

func (m *Model) String() string {
	return fmt.Sprintf("%s: %s (%s)", m.ID, m.DisplayName, m.CreatedAt.Format("2006-01-02"))
}

func (m *Model) Context() int64 {
	return 0
}

// ModelsResponse represents the response structure for Anthropic models listing
type ModelsResponse struct {
	Data    []Model `json:"data"`
	FirstID string  `json:"first_id"`
	HasMore bool    `json:"has_more"`
	LastID  string  `json:"last_id"`
}

// ToModels converts Anthropic models to genai.Model interfaces
func (r *ModelsResponse) ToModels() []genai.Model {
	models := make([]genai.Model, len(r.Data))
	for i := range r.Data {
		models[i] = &r.Data[i]
	}
	return models
}

//

type ErrorResponse struct {
	Type  string `json:"type"` // "error"
	Error struct {
		Type    string `json:"type"` // e.g. "invalid_request_error"
		Message string `json:"message"`
	} `json:"error"`
}

func (er *ErrorResponse) String() string {
	return fmt.Sprintf("error %s: %s", er.Error.Type, er.Error.Message)
}

// Client implements genai.ChatProvider and genai.ModelProvider.
type Client struct {
	internal.ClientChat[*ErrorResponse, *ChatRequest, *ChatResponse, ChatStreamChunkResponse]
}

// New creates a new client to talk to the Anthropic platform API.
//
// If apiKey is not provided, it tries to load it from the ANTHROPIC_API_KEY environment variable.
// If none is found, it returns an error.
// Get an API key at https://console.anthropic.com/settings/keys
//
// If no model is provided, only functions that do not require a model, like ListModels, will work.
// To use multiple models, create multiple clients.
// Use one of the model from https://docs.anthropic.com/en/docs/about-claude/models/all-models
//
// r can be used to throttle outgoing requests, record calls, etc. It defaults to http.DefaultTransport.
func New(apiKey, model string, r http.RoundTripper) (*Client, error) {
	const apiKeyURL = "https://console.anthropic.com/settings/keys"
	if apiKey == "" {
		if apiKey = os.Getenv("ANTHROPIC_API_KEY"); apiKey == "" {
			return nil, errors.New("anthropic API key is required; get one at " + apiKeyURL)
		}
	}
	if r == nil {
		r = http.DefaultTransport
	}
	// Anthropic allows Opaque fields for thinking signatures
	return &Client{
		ClientChat: internal.ClientChat[*ErrorResponse, *ChatRequest, *ChatResponse, ChatStreamChunkResponse]{
			Model:                model,
			ChatURL:              "https://api.anthropic.com/v1/messages",
			AllowOpaqueFields:    true,
			ProcessStreamPackets: processStreamPackets,
			ClientBase: internal.ClientBase[*ErrorResponse]{
				ClientJSON: httpjson.Client{
					Client: &http.Client{Transport: &roundtrippers.Header{
						Transport: &roundtrippers.Retry{
							Transport: &roundtrippers.RequestID{
								Transport: r,
							},
						},
						Header: http.Header{"x-api-key": {apiKey}, "anthropic-version": {"2023-06-01"}},
					}},
					Lenient: internal.BeLenient,
				},
				APIKeyURL: apiKeyURL,
			},
		},
	}, nil
}

func (c *Client) Scoreboard() genai.Scoreboard {
	return Scoreboard
}

func (c *Client) ListModels(ctx context.Context) ([]genai.Model, error) {
	// https://docs.anthropic.com/en/api/models-list
	return internal.ListModels[*ErrorResponse, *ModelsResponse](ctx, &c.ClientBase, "https://api.anthropic.com/v1/models?limit=1000")
}

func processStreamPackets(ch <-chan ChatStreamChunkResponse, chunks chan<- genai.MessageFragment, result *genai.ChatResult) error {
	defer func() {
		// We need to empty the channel to avoid blocking the goroutine.
		for range ch {
		}
	}()
	pendingCall := genai.ToolCall{}
	for pkt := range ch {
		f := genai.MessageFragment{}
		// See testdata/TestClient_Chat_thinking/ChatStream.yaml as a great example.
		// TODO: pkt.Index matters here, as the LLM may fill multiple content blocks simultaneously.
		switch pkt.Type {
		case ChunkMessageStart:
			switch pkt.Message.Role {
			case "assistant":
			default:
				return fmt.Errorf("unexpected role %q", pkt.Message.Role)
			}
			result.InputTokens = pkt.Message.Usage.InputTokens
			result.InputCachedTokens = pkt.Message.Usage.CacheReadInputTokens
			// There's some tokens listed there. Still save it in case it breaks midway.
			result.OutputTokens = pkt.Message.Usage.OutputTokens
			continue
		case ChunkContentBlockStart:
			switch pkt.ContentBlock.Type {
			case ContentText:
				f.TextFragment = pkt.ContentBlock.Text
			case ContentThinking:
				f.ThinkingFragment = pkt.ContentBlock.Thinking
			case ContentToolUse:
				pendingCall.ID = pkt.ContentBlock.ID
				pendingCall.Name = pkt.ContentBlock.Name
				pendingCall.Arguments = ""
				// TODO: Is there anything to do with Input? pendingCall.Arguments = pkt.ContentBlock.Input
			default:
				return fmt.Errorf("missing implementation for content block %q", pkt.ContentBlock.Type)
			}
		case ChunkContentBlockDelta:
			switch pkt.Delta.Type {
			case DeltaText:
				f.TextFragment = pkt.Delta.Text
			case DeltaThinking:
				f.ThinkingFragment = pkt.Delta.Thinking
			case DeltaSignature:
				f.Opaque = map[string]any{"signature": pkt.Delta.Signature}
			case DeltaInputJSON:
				pendingCall.Arguments += pkt.Delta.PartialJSON
			default:
				return fmt.Errorf("missing implementation for content block delta %q", pkt.Delta.Type)
			}
		case ChunkContentBlockStop:
			// Marks a closure of the block pkt.Index. Nothing to do.
			if pendingCall.ID != "" {
				f.ToolCall = pendingCall
				pendingCall = genai.ToolCall{}
			}
		case ChunkMessageDelta:
			// Includes finish reason and output tokens usage (but not input tokens!)
			result.FinishReason = pkt.Delta.StopReason.ToFinishReason()
			result.OutputTokens = pkt.Usage.OutputTokens
		case ChunkMessageStop:
			// Doesn't contain anything.
			continue
		case ChunkPing:
			// Doesn't contain anything.
			continue
		default:
			return fmt.Errorf("unknown stream block %q", pkt.Type)
		}
		if !f.IsZero() {
			if err := result.Accumulate(f); err != nil {
				return err
			}
			chunks <- f
		}
	}
	return nil
}

var (
	_ genai.ChatProvider  = &Client{}
	_ genai.ModelProvider = &Client{}
)
