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
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"iter"
	"net/http"
	"net/url"
	"os"
	"slices"
	"strconv"
	"strings"
	"time"

	"github.com/invopop/jsonschema"
	"github.com/maruel/genai"
	"github.com/maruel/genai/base"
	"github.com/maruel/genai/internal"
	"github.com/maruel/genai/scoreboard"
	"github.com/maruel/httpjson"
	"github.com/maruel/roundtrippers"
)

// Scoreboard for Anthropic.
var Scoreboard = scoreboard.Score{
	Warnings: []string{
		"No Anthropic model support structured output, you have to use tool calling instead.",
		"Tool calling works very well but is biased; the model is lazy and when it's unsure, it will use the tool's first argument.",
		"Rate limit is based on how much you spend per month: https://docs.anthropic.com/en/api/rate-limits#requirements-to-advance-tier",
	},
	Country:      "US",
	DashboardURL: "https://console.anthropic.com/settings/billing",
	Scenarios: []scoreboard.Scenario{
		{
			Models: []string{"claude-3-5-haiku-20241022"},
			In: map[genai.Modality]scoreboard.ModalCapability{
				genai.ModalityText: {Inline: true},
				genai.ModalityImage: {
					Inline:           true,
					URL:              true,
					SupportedFormats: []string{"image/gif", "image/jpeg", "image/png", "image/webp"},
				},
				genai.ModalityDocument: {
					Inline:           true,
					URL:              true,
					SupportedFormats: []string{"application/pdf"},
				},
			},
			Out: map[genai.Modality]scoreboard.ModalCapability{genai.ModalityText: {Inline: true}},
			GenSync: &scoreboard.FunctionalityText{
				ReportRateLimits: true,
				Tools:            scoreboard.True,
				BiasedTool:       scoreboard.True,
				Citations:        true,
			},
			GenStream: &scoreboard.FunctionalityText{
				ReportRateLimits: true,
				Tools:            scoreboard.True,
				BiasedTool:       scoreboard.True,
				Citations:        true,
			},
		},
		{
			Models: []string{"claude-sonnet-4-20250514"},
			In: map[genai.Modality]scoreboard.ModalCapability{
				genai.ModalityText: {Inline: true},
				genai.ModalityImage: {
					Inline:           true,
					URL:              true,
					SupportedFormats: []string{"image/gif", "image/jpeg", "image/png", "image/webp"},
				},
				genai.ModalityDocument: {
					Inline:           true,
					URL:              true,
					SupportedFormats: []string{"application/pdf"},
				},
			},
			Out: map[genai.Modality]scoreboard.ModalCapability{genai.ModalityText: {Inline: true}},
			GenSync: &scoreboard.FunctionalityText{
				ReportRateLimits: true,
				Tools:            scoreboard.True,
				BiasedTool:       scoreboard.True,
				Citations:        true,
			},
			GenStream: &scoreboard.FunctionalityText{
				ReportRateLimits: true,
				Tools:            scoreboard.True,
				BiasedTool:       scoreboard.True,
				Citations:        true,
			},
		},
		{
			Models:   []string{"claude-sonnet-4-20250514"},
			Thinking: true,
			In: map[genai.Modality]scoreboard.ModalCapability{
				genai.ModalityText: {Inline: true},
				genai.ModalityImage: {
					Inline:           true,
					URL:              true,
					SupportedFormats: []string{"image/gif", "image/jpeg", "image/png", "image/webp"},
				},
				genai.ModalityDocument: {
					Inline:           true,
					URL:              true,
					SupportedFormats: []string{"application/pdf"},
				},
			},
			Out: map[genai.Modality]scoreboard.ModalCapability{genai.ModalityText: {Inline: true}},
			GenSync: &scoreboard.FunctionalityText{
				ReportRateLimits: true,
				Tools:            scoreboard.True,
				NoMaxTokens:      true,
				Citations:        true,
			},
			GenStream: &scoreboard.FunctionalityText{
				ReportRateLimits: true,
				Tools:            scoreboard.True,
				NoMaxTokens:      true,
				Citations:        true,
			},
		},
		{
			Comments: "They take more than 10 minutes to run the test, which causes it to timeout. And they cost a lot.",
			Models: []string{
				"claude-opus-4-20250514",
				"claude-opus-4-1-20250805",
			},
		},
		{
			Comments: "Old models",
			Models: []string{
				"claude-3-5-sonnet-20240620",
				"claude-3-5-sonnet-20241022",
				"claude-3-7-sonnet-20250219",
				"claude-3-haiku-20240307",
				"claude-3-opus-20240229",
			},
		},
	},
}

// OptionsText defines Anthropic specific options.
type OptionsText struct {
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

func (o *OptionsText) Validate() error {
	return nil
}

// ChatRequest is documented at https://docs.anthropic.com/en/api/messages
type ChatRequest struct {
	Model      string      `json:"model,omitzero"`
	MaxTokens  int64       `json:"max_tokens"`
	Messages   []Message   `json:"messages"`
	Container  string      `json:"container,omitzero"` // identifier for reuse across requests
	MCPServers []MCPServer `json:"mcp_servers,omitzero"`
	Metadata   struct {
		UserID string `json:"user_id,omitzero"` // Should be a hash or UUID, opaque, to detect abuse, no PII
	} `json:"metadata,omitzero"`
	ServiceTier   string          `json:"service_tier,omitzero"` // "auto", "standard_only"
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
func (c *ChatRequest) Init(msgs genai.Messages, model string, opts ...genai.Options) error {
	return c.initImpl(msgs, model, true, opts...)
}

func (c *ChatRequest) initImpl(msgs genai.Messages, model string, cache bool, opts ...genai.Options) error {
	c.Model = model
	if err := msgs.Validate(); err != nil {
		return err
	}
	var errs []error
	var unsupported []string
	msgToCache := 0
	// Default to disabled thinking.
	c.Thinking.Type = "disabled"
	// Anthropic requires a value! And their models listing API doesn't provide the model's acceptable values! This is quite annoying.
	c.MaxTokens = modelsMaxTokens(model)
	for _, opt := range opts {
		if err := opt.Validate(); err != nil {
			return err
		}
		switch v := opt.(type) {
		case *OptionsText:
			if cache {
				msgToCache = v.MessagesToCache
			} else if v.MessagesToCache != 0 {
				unsupported = append(unsupported, "MessagesToCache")
			}
			if v.ThinkingBudget > 0 {
				if v.ThinkingBudget >= c.MaxTokens {
					errs = append(errs, fmt.Errorf("invalid ThinkingBudget(%d) >= MaxTokens(%d)", v.ThinkingBudget, c.MaxTokens))
				}
				// https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking
				// Thinking isn’t compatible with temperature, top_p, or top_k modifications as well as forced tool use.
				c.Thinking.BudgetTokens = v.ThinkingBudget
				c.Thinking.Type = "enabled"
			}
		case *genai.OptionsText:
			unsupported, errs = c.initOptions(v)
		default:
			errs = append(errs, fmt.Errorf("unsupported options type %T", opts))
		}
	}

	c.Messages = make([]Message, 0, len(msgs))
	for i := range msgs {
		c.Messages = append(c.Messages, Message{})
		if err := c.Messages[len(c.Messages)-1].From(&msgs[i]); err != nil {
			errs = append(errs, fmt.Errorf("message #%d: %w", i, err))
		}
		if i == msgToCache-1 {
			c.Messages[i].CacheControl.Type = "ephemeral"
		}
		if err := c.Messages[len(c.Messages)-1].Validate(); err != nil {
			errs = append(errs, fmt.Errorf("message #%d: %w", i, err))
		}
	}
	// If we have unsupported features but no other errors, return a continuable error
	if len(unsupported) > 0 && len(errs) == 0 {
		return &genai.UnsupportedContinuableError{Unsupported: unsupported}
	}
	return errors.Join(errs...)
}

func (c *ChatRequest) SetStream(stream bool) {
	c.Stream = stream
}

func (c *ChatRequest) initOptions(v *genai.OptionsText) ([]string, []error) {
	var unsupported []string
	var errs []error
	if v.TopLogprobs > 0 {
		unsupported = append(unsupported, "TopLogprobs")
	}
	if v.MaxTokens != 0 {
		c.MaxTokens = v.MaxTokens
	}
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

func modelsMaxTokens(model string) int64 {
	// Anthropic requires a value! And their models listing API doesn't provide the model's acceptable values! This is quite annoying.
	if strings.HasPrefix(model, "claude-3-opus-") || strings.HasPrefix(model, "claude-3-haiku-") {
		return 4096
	} else if strings.HasPrefix(model, "claude-3-5-") {
		return 8192
	} else if strings.HasPrefix(model, "claude-3-7-") {
		return 64000
	} else if strings.HasPrefix(model, "claude-4-sonnet-") {
		return 64000
	} else if strings.HasPrefix(model, "claude-4-opus-") {
		return 32000
	}
	// Default value for new models.
	return 32000
}

// MCPServer is documented at  https://docs.anthropic.com/en/api/messages#body-mcp-servers
type MCPServer struct {
	Name               string `json:"name"` //
	Type               string `json:"type"` // "url"
	URL                string `json:"url"`
	AuthorizationToken string `json:"authorization_token,omitzero"`
	ToolConfiguration  struct {
		AllowedTools []string `json:"allowed_tools,omitzero"`
		Enabled      bool     `json:"enabled,omitzero"`
	} `json:"tool_configuration,omitzero"`
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

// Message is documented at https://docs.anthropic.com/en/api/messages
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

func (m *Message) Validate() error {
	if m.Type != "" && m.Type != "message" {
		// Allow empty type, it is not required.
		return fmt.Errorf("unsupported message type %q", m.Type)
	}
	switch m.Role {
	case "assistant", "user":
		// Valid.
	case "":
		return errors.New("message doesn't have role defined")
	default:
		return fmt.Errorf("unsupported role %q", m.Role)
	}
	if len(m.Content) == 0 {
		return errors.New("message doesn't have content defined")
	}
	for i := range m.Content {
		if err := m.Content[i].Validate(); err != nil {
			return fmt.Errorf("content block %d: %w", i, err)
		}
	}
	return nil
}

func (m *Message) From(in *genai.Message) error {
	switch r := in.Role(); r {
	case "assistant", "user":
		m.Role = r
	case "computer":
		m.Role = "user"
	default:
		return fmt.Errorf("unsupported role %q", r)
	}
	m.Content = make([]Content, len(in.Requests)+len(in.Replies)+len(in.ToolCallResults))
	for i := range in.Requests {
		if err := m.Content[i].FromRequest(&in.Requests[i]); err != nil {
			return fmt.Errorf("request #%d: %w", i, err)
		}
	}
	offset := len(in.Requests)
	for i := range in.Replies {
		if err := m.Content[i+offset].FromReply(&in.Replies[i]); err != nil {
			return fmt.Errorf("reply #%d: %w", i, err)
		}
	}
	offset += len(in.Replies)
	for i := range in.ToolCallResults {
		if err := m.Content[offset+i].FromToolCallResult(&in.ToolCallResults[i]); err != nil {
			return fmt.Errorf("tool call results #%d: %w", offset+i, err)
		}
	}
	return nil
}

func (m *Message) To(out *genai.Message) error {
	// Make sure the message was initialized properly before converting.
	if err := m.Validate(); err != nil {
		return err
	}
	// We need to split actual content and tool calls.
	for i := range m.Content {
		switch m.Content[i].Type {
		case ContentText, ContentThinking, ContentRedactedThinking, ContentToolUse:
			c := genai.Reply{}
			if skip, err := m.Content[i].To(&c); err != nil {
				return fmt.Errorf("reply #%d: %w", i, err)
			} else if !skip {
				out.Replies = append(out.Replies, c)
			}
		case ContentImage, ContentDocument, ContentToolResult:
			fallthrough
		default:
			return fmt.Errorf("unsupported content type %q", m.Content[i].Type)
		}
	}
	return nil
}

type Content struct {
	Type ContentType `json:"type"`
	// Valid with Type == ContentText.
	Text string `json:"text,omitzero"`

	// Valid with Type == ContentThinking.
	Thinking  string `json:"thinking,omitzero"`
	Signature []byte `json:"signature,omitzero"`

	// Valid with Type == ContentRedactedThinking.
	Data string `json:"data,omitzero"`

	// Valid with Type == ContentText, ContentImage, ContentDocument, ContentToolUse, ContentToolResult.
	CacheControl struct {
		Type string `json:"type,omitzero"` // "ephemeral"
	} `json:"cache_control,omitzero"`

	// Valid with Type == ContentText, ContentDocument.
	Citations Citations `json:"citations,omitzero"`

	// Valid with Type == ContentImage, ContentDocument.
	Source struct {
		Type SourceType `json:"type,omitzero"`

		// Valid with Source.Type == SourceBase64, SourceURL, SourceText
		// Content.Type == ContentImage: "image/gif", "image/jpeg", "image/png", "image/webp"
		// Content.Type == ContentDocument: "application/pdf", "text/plain"
		MediaType string `json:"media_type,omitzero"`

		// Valid with Source.Type == SourceBase64, SourceURL, SourceText
		Data string `json:"data,omitzero"` // base64 encoded if base64, else as is, e.g. text plain data.

		// Valid with Source.Type == SourceURL
		URL string `json:"url,omitzero"`

		// Valid with Source.Type == SourceContent
		// Only ContentText and ContentImage are allowed.
		Content []Content `json:"content,omitzero"`

		// Valid with Source.Type == SourceFileID
		FileID string `json:"file_id,omitzero"` // File ID for the content, used for file uploads.
	} `json:"source,omitzero"`

	// Valid with Type == ContentToolUse
	ID    string `json:"id,omitzero"`
	Input any    `json:"input,omitzero"`
	Name  string `json:"name,omitzero"`

	// Valid with Type == ContentToolResult
	ToolUseID string `json:"tool_use_id,omitzero"`
	IsError   bool   `json:"is_error,omitzero"`
	// Only ContentText and ContentImage are allowed.
	Content []Content `json:"content,omitzero"`

	// Valid with Type == ContentDocument
	Context string `json:"context,omitzero"` // Context about the document that will not be cited from
	Title   string `json:"title,omitzero"`   // Document title when using Source
}

func (c *Content) Validate() error {
	switch c.Type {
	case ContentText:
		// It happens with citations.
		// if c.Text == "" {
		// 	return errors.New("ContentText: Text must be set")
		// }
		if c.Thinking != "" || len(c.Signature) > 0 || c.Data != "" || c.ID != "" || c.Name != "" || c.Input != nil ||
			c.ToolUseID != "" || c.IsError || len(c.Content) > 0 || c.Context != "" || c.Title != "" {
			return errors.New("ContentText: unexpected fields set")
		}
	case ContentThinking:
		if c.Thinking == "" {
			return errors.New("ContentThinking: Thinking must be set")
		}
		if c.Text != "" || len(c.Signature) > 0 && c.Signature == nil || c.Data != "" || c.ID != "" || c.Name != "" || c.Input != nil ||
			c.ToolUseID != "" || c.IsError || len(c.Content) > 0 || c.Context != "" || c.Title != "" {
			return errors.New("ContentThinking: unexpected fields set")
		}
	case ContentRedactedThinking:
		if c.Data == "" {
			return errors.New("ContentRedactedThinking: Data must be set")
		}
		if c.Text != "" || c.Thinking != "" || len(c.Signature) > 0 || c.ID != "" || c.Name != "" || c.Input != nil ||
			c.ToolUseID != "" || c.IsError || len(c.Content) > 0 || c.Context != "" || c.Title != "" {
			return errors.New("ContentRedactedThinking: unexpected fields set")
		}
	case ContentImage, ContentDocument:
		if c.Source.Type == "" && c.Source.URL == "" && c.Source.Data == "" {
			return fmt.Errorf("%s: Source must be set", c.Type)
		}
		if c.Text != "" || c.Thinking != "" || len(c.Signature) > 0 || c.Data != "" || c.ID != "" || c.Name != "" || c.Input != nil ||
			c.ToolUseID != "" || c.IsError || len(c.Content) > 0 {
			return fmt.Errorf("%s: unexpected fields set", c.Type)
		}
	case ContentToolUse:
		if c.ID == "" || c.Name == "" {
			return errors.New("ContentToolUse: ID and Name must be set")
		}
		if c.Text != "" || c.Thinking != "" || len(c.Signature) > 0 || c.Data != "" ||
			c.ToolUseID != "" || c.IsError || len(c.Content) > 0 || c.Context != "" || c.Title != "" {
			return errors.New("ContentToolUse: unexpected fields set")
		}
	case ContentToolResult:
		if c.ToolUseID == "" {
			return errors.New("ContentToolResult: ToolUseID must be set")
		}
		if c.Text != "" || c.Thinking != "" || len(c.Signature) > 0 || c.Data != "" || c.ID != "" || c.Name != "" || c.Input != nil ||
			c.Context != "" || c.Title != "" {
			return errors.New("ContentToolResult: unexpected fields set")
		}
	default:
		return fmt.Errorf("unknown ContentType %q", c.Type)
	}
	return nil
}

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
		// Anthropic require a mime-type to determine if image or PDF.
		if mimeType == "" {
			return fmt.Errorf("unspecified mime type for URL %q", in.Doc.URL)
		}
		c.CacheControl.Type = "ephemeral"
		switch {
		case strings.HasPrefix(mimeType, "image/"):
			c.Type = ContentImage
			if in.Doc.URL != "" {
				c.Source.Type = SourceURL
				c.Source.URL = in.Doc.URL
			} else {
				c.Source.MediaType = mimeType
				c.Source.Type = SourceBase64
				c.Source.Data = base64.StdEncoding.EncodeToString(data)
			}
		case mimeType == "application/pdf":
			c.Type = ContentDocument
			if in.Doc.URL != "" {
				c.Source.Type = SourceURL
				c.Source.URL = in.Doc.URL
			} else {
				c.Source.MediaType = mimeType
				c.Source.Type = SourceBase64
				c.Source.Data = base64.StdEncoding.EncodeToString(data)
			}
		case strings.HasPrefix(mimeType, "text/plain"):
			c.Type = ContentDocument
			if in.Doc.URL != "" {
				return errors.New("text/plain documents must be provided inline, not as a URL")
			}
			// In particular, the API refuses "text/plain; charset=utf-8". WTF.
			c.Source.MediaType = "text/plain"
			c.Source.Type = SourceText
			c.Source.Data = string(data)
			// Enable citations for text/plain documents
			c.Citations = Citations{Enabled: true}
		default:
			return fmt.Errorf("unsupported content mime-type %s", mimeType)
		}
		return nil
	}
	return errors.New("unknown Request type")
}

func (c *Content) FromReply(in *genai.Reply) error {
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
	if !in.ToolCall.IsZero() {
		if len(in.ToolCall.Opaque) != 0 {
			return errors.New("field ToolCall.Opaque not supported")
		}
		c.Type = ContentToolUse
		c.ID = in.ToolCall.ID
		c.Name = in.ToolCall.Name
		if err := json.Unmarshal([]byte(in.ToolCall.Arguments), &c.Input); err != nil {
			return fmt.Errorf("failed to marshal input: %w; for tool call: %#v", err, in)
		}
		return nil
	}
	if !in.Doc.IsZero() {
		mimeType, data, err := in.Doc.Read(10 * 1024 * 1024)
		if err != nil {
			return err
		}
		// Anthropic require a mime-type to determine if image or PDF.
		if mimeType == "" {
			return fmt.Errorf("unspecified mime type for URL %q", in.Doc.URL)
		}
		c.CacheControl.Type = "ephemeral"
		switch {
		case strings.HasPrefix(mimeType, "image/"):
			c.Type = ContentImage
			if in.Doc.URL != "" {
				c.Source.Type = SourceURL
				c.Source.URL = in.Doc.URL
			} else {
				c.Source.MediaType = mimeType
				c.Source.Type = SourceBase64
				c.Source.Data = base64.StdEncoding.EncodeToString(data)
			}
		case mimeType == "application/pdf":
			c.Type = ContentDocument
			if in.Doc.URL != "" {
				c.Source.Type = SourceURL
				c.Source.URL = in.Doc.URL
			} else {
				c.Source.MediaType = mimeType
				c.Source.Type = SourceBase64
				c.Source.Data = base64.StdEncoding.EncodeToString(data)
			}
		case strings.HasPrefix(mimeType, "text/plain"):
			c.Type = ContentDocument
			if in.Doc.URL != "" {
				return errors.New("text/plain documents must be provided inline, not as a URL")
			}
			// In particular, the API refuses "text/plain; charset=utf-8". WTF.
			c.Source.MediaType = "text/plain"
			c.Source.Type = SourceText
			c.Source.Data = string(data)
			// Enable citations for text/plain documents
			c.Citations = Citations{Enabled: true}
		default:
			return fmt.Errorf("unsupported content mime-type %s", mimeType)
		}
		return nil
	}
	return errors.New("unknown Reply type")
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

func (c *Content) To(out *genai.Reply) (bool, error) {
	switch c.Type {
	case ContentText:
		out.Text = c.Text
		if len(c.Citations.Citations) > 0 {
			out.Citations = make([]genai.Citation, len(c.Citations.Citations))
			for i := range c.Citations.Citations {
				if err := c.Citations.Citations[i].To(&out.Citations[i]); err != nil {
					return false, fmt.Errorf("citation %d: %w", i, err)
				}
			}
		} else if len(out.Text) == 0 {
			// This happens with citations with claude 4 sonnet with thinking enabled where an empty text packet is
			// sent first.
			return true, nil
		}
	case ContentThinking:
		out.Thinking = c.Thinking
		out.Opaque = map[string]any{"signature": c.Signature}
	case ContentRedactedThinking:
		out.Opaque = map[string]any{"redacted_thinking": c.Signature}
	case ContentToolUse:
		out.ToolCall.ID = c.ID
		out.ToolCall.Name = c.Name
		raw, err := json.Marshal(c.Input)
		if err != nil {
			return false, fmt.Errorf("failed to marshal input: %w; for tool call: %#v", err, c)
		}
		out.ToolCall.Arguments = string(raw)
		return false, nil
	case ContentImage, ContentDocument, ContentToolResult:
		fallthrough
	default:
		return false, fmt.Errorf("unsupported content type %q", c.Type)
	}
	return false, nil
}

// Citations is a mess.
//
// It can be a configuration. It is described in messages.content object[], type Document, citations.
// It is explained at https://docs.anthropic.com/en/api/messages#body-messages-content-citations-enabled
//
// It can be actual citations. It is described in messages.content object[], type Text, citations []object.
// https://docs.anthropic.com/en/api/messages#body-messages-content-citations
//
// https://docs.anthropic.com/en/docs/build-with-claude/citations
type Citations struct {
	Citations []Citation
	Enabled   bool
}

type citationsObject struct {
	Enabled bool `json:"enabled"`
}

// SourceType is described at https://docs.anthropic.com/en/api/messages#body-messages-content-source
type SourceType string

const (
	SourceBase64  SourceType = "base64"
	SourceURL     SourceType = "url"
	SourceText    SourceType = "text"
	SourceContent SourceType = "content"
)

// UnmarshalJSON implements json.Unmarshaler for Citations.
// It attempts to unmarshal the input as either a slice of Citations or a struct with an Enabled field.
func (c *Citations) UnmarshalJSON(b []byte) error {
	if bytes.Equal(b, []byte("null")) {
		c.Citations = nil
		c.Enabled = false
		return nil
	}
	d := json.NewDecoder(bytes.NewReader(b))
	if !internal.BeLenient {
		d.DisallowUnknownFields()
	}
	var cc []Citation
	if err := d.Decode(&cc); err == nil {
		c.Citations = cc
		c.Enabled = false
		return nil
	}

	o := citationsObject{}
	d = json.NewDecoder(bytes.NewReader(b))
	if !internal.BeLenient {
		d.DisallowUnknownFields()
	}
	if err := d.Decode(&o); err != nil {
		return err
	}
	c.Enabled = o.Enabled
	c.Citations = nil
	return nil
}

// MarshalJSON implements json.Marshaler for Citations.
// It marshals the struct as a slice if Citations field is not empty, otherwise as a struct with the Enabled field.
func (c Citations) MarshalJSON() ([]byte, error) {
	if len(c.Citations) > 0 {
		return json.Marshal(c.Citations)
	}
	if c.Enabled {
		objectCitations := citationsObject{Enabled: c.Enabled}
		return json.Marshal(objectCitations)
	}
	return []byte("null"), nil
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

func (c *Citation) To(dst *genai.Citation) error {
	dst.Text = c.CitedText
	dst.Type = c.Type
	switch c.Type {
	case "char_location":
		dst.StartIndex = c.StartCharIndex
		dst.EndIndex = c.EndCharIndex
	case "page_location":
		// For page location, we'll store the page info in Location
		dst.Location = map[string]any{
			"start_page": c.StartPageNumber,
			"end_page":   c.EndPageNumber,
		}
	case "content_block_location":
		// For block location, we'll store the block info in Location
		dst.Location = map[string]any{
			"start_block": c.StartBlockIndex,
			"end_block":   c.EndBlockIndex,
		}
	case "web_search_result_location":
		// For web search results, create a source with URL and title
		dst.Sources = []genai.CitationSource{{
			Type:  "web",
			URL:   c.URL,
			Title: c.Title,
			Metadata: map[string]any{
				"encrypted_index": c.EncryptedIndex,
			},
		}}
	}
	// Add document information as a source if available
	if c.DocumentIndex > 0 || c.DocumentTitle != "" {
		docSource := genai.CitationSource{
			Type:  "document",
			Title: c.DocumentTitle,
			Metadata: map[string]any{
				"document_index": c.DocumentIndex,
			},
		}
		// For web search results, we already have sources, so append
		if c.Type == "web_search_result_location" {
			dst.Sources = append(dst.Sources, docSource)
		} else {
			dst.Sources = []genai.CitationSource{docSource}
		}
	}
	return nil
}

type Thinking struct {
	BudgetTokens int64  `json:"budget_tokens,omitzero"` // >1024 and less than max_tokens
	Type         string `json:"type,omitzero"`          // "enabled", "disabled"
}

// ToolChoiceType is documented at  https://docs.anthropic.com/en/api/messages#body-tool-choice
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

// Tool is documented at  https://docs.anthropic.com/en/api/messages#body-tools
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
	Container    struct {
		ExpiresAt time.Time `json:"expires_at"`
		ID        string    `json:"id"`
	} `json:"container"`
}

func (c *ChatResponse) ToResult() (genai.Result, error) {
	out := genai.Result{
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

// StopReason is documented at https://docs.anthropic.com/en/api/messages#response-stop-reason
type StopReason string

const (
	StopEndTurn   StopReason = "end_turn"
	StopToolUse   StopReason = "tool_use"
	StopSequence  StopReason = "stop_sequence"
	StopMaxTokens StopReason = "max_tokens"
	StopPauseTurn StopReason = "pause_turn" //  We paused a long-running turn. You may provide the response back as-is in a subsequent request to let the model continue.
	StopRefusal   StopReason = "refusal"
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
	case StopRefusal:
		return genai.FinishedContentFilter
	case StopPauseTurn:
		return genai.Pending
	default:
		if !internal.BeLenient {
			panic(s)
		}
		return genai.FinishReason(s)
	}
}

// ChatStreamChunkResponse is documented at https://docs.anthropic.com/en/api/messages-streaming
//
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

	// Type == ChunkContentBlockStart
	ContentBlock struct {
		Type ContentType `json:"type"`

		// Type == ContentText
		Text string `json:"text"`

		// Type == ContentThinking
		Thinking  string `json:"thinking"`
		Signature []byte `json:"signature"` // Never actually filed but present on content_block_start.

		// Type == ContentToolUse
		ID    string `json:"id"`
		Name  string `json:"name"`
		Input any    `json:"input"`

		Citations []struct{} `json:"citations"` // Empty, not used in the API.
	} `json:"content_block"`

	// Type == ChunkContentBlockDelta
	Delta struct {
		Type DeltaType `json:"type"`

		// Type == DeltaText
		Text string `json:"text"`

		// Type == DeltaInputJSON
		PartialJSON string `json:"partial_json"`

		// Type == DeltaThinking
		Thinking string `json:"thinking"`

		// Type == DeltaSignature
		Signature []byte `json:"signature"`

		// Type == DeltaCitations
		Citation Citation `json:"citation"`

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
	DeltaCitations DeltaType = "citations_delta"
)

type Usage struct {
	InputTokens              int64  `json:"input_tokens"`
	OutputTokens             int64  `json:"output_tokens"`
	CacheCreationInputTokens int64  `json:"cache_creation_input_tokens"`
	CacheReadInputTokens     int64  `json:"cache_read_input_tokens"`
	ServiceTier              string `json:"service_tier"` // "standard", "batch"
}

//

// BatchRequest is documented at https://docs.anthropic.com/en/api/creating-message-batches
type BatchRequest struct {
	Requests []BatchRequestItem `json:"requests"`
}

// BatchRequestItem is documented at https://docs.anthropic.com/en/api/creating-message-batches
type BatchRequestItem struct {
	CustomID string      `json:"custom_id"`
	Params   ChatRequest `json:"params"`
}

func (b *BatchRequestItem) Init(msgs genai.Messages, model string, opts ...genai.Options) error {
	// TODO: We need to make this unique, the field is required. The problem is that this breaks HTTP recording.
	// var bytes [12]byte
	// _, _ = rand.Read(bytes[:])
	// b.CustomID = base64.RawURLEncoding.EncodeToString(bytes[:])
	b.CustomID = "TODO"
	return b.Params.initImpl(msgs, model, false, opts...)
}

// BatchResponse is documented at https://docs.anthropic.com/en/api/creating-message-batches
type BatchResponse struct {
	ID                string    `json:"id"`   // Starts with "msgbatch_"
	Type              string    `json:"type"` // "message_batch"
	ArchivedAt        time.Time `json:"archived_at"`
	CancelInitiatedAt time.Time `json:"cancel_initiated_at"`
	CreatedAt         time.Time `json:"created_at"`
	EndedAt           time.Time `json:"ended_at"`
	ExpiresAt         time.Time `json:"expires_at"`
	ProcessingStatus  string    `json:"processing_status"` // "in_progress", "canceling", "ended"
	RequestCounts     struct {
		Canceled   int64 `json:"canceled"`
		Errored    int64 `json:"errored"`
		Expired    int64 `json:"expired"`
		Processing int64 `json:"processing"`
		Succeeded  int64 `json:"succeeded"`
	} `json:"request_counts"`
	ResultsURL string `json:"results_url"`
}

// BatchQueryResponse is documented at https://docs.anthropic.com/en/api/retrieving-message-batch-results
type BatchQueryResponse struct {
	CustomID string `json:"custom_id"`
	Result   struct {
		// Adding synthetic "not_found" result.
		Type string `json:"type"` // "succeeded", "canceled", "expired", "errored"

		// Type == "succeeded"
		// Message is not a standard message, it's closer to streaming's version.
		Message struct {
			Type         string     `json:"type"` // "message"
			Role         string     `json:"role"` // "assistant"
			Content      []Content  `json:"content"`
			ID           string     `json:"id"`
			Model        string     `json:"model"`
			StopReason   StopReason `json:"stop_reason"`
			StopSequence string     `json:"stop_sequence"`
			Usage        Usage      `json:"usage"`
		} `json:"message"`

		// Type == "errored"
		Error struct {
			Type  string `json:"type"` // "error"
			Error struct {
				Type    string   `json:"type"`    // "invalid_request_error"
				Message string   `json:"message"` // e.g. "metadata.thinking: Extra inputs are not permitted"
				Details struct{} `json:"details"`
			} `json:"error"`
		} `json:"error"`
	} `json:"result"`
}

func (b *BatchQueryResponse) To(out *genai.Message) error {
	// We need to split actual content and tool calls.
	for i := range b.Result.Message.Content {
		switch b.Result.Message.Content[i].Type {
		case ContentText, ContentThinking, ContentRedactedThinking, ContentToolUse:
			c := genai.Reply{}
			if skip, err := b.Result.Message.Content[i].To(&c); err != nil {
				return fmt.Errorf("block %d: %w", i, err)
			} else if !skip {
				out.Replies = append(out.Replies, c)
			}
		case ContentImage, ContentDocument, ContentToolResult:
			fallthrough
		default:
			return fmt.Errorf("unsupported content type %q", b.Result.Message.Content[i].Type)
		}
	}
	return nil
}

//

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

// ErrorResponse is documented at https://docs.anthropic.com/en/api/messages#response-error
type ErrorResponse struct {
	Type     string `json:"type"` // "error"
	ErrorVal struct {
		// Type is one of "invalid_request_error", "authentication_error", "billing_error", "permission_error", "not_found_error", "rate_limit_error", "timeout_error", "api_error", "overloaded_error"
		Type    string `json:"type"`
		Message string `json:"message"`
	} `json:"error"`
	RequestID string `json:"request_id"`
}

func (er *ErrorResponse) Error() string {
	return fmt.Sprintf("%s: %s", er.ErrorVal.Type, er.ErrorVal.Message)
}

func (er *ErrorResponse) IsAPIError() bool {
	return true
}

// Client implements genai.Provider.
type Client struct {
	impl base.Provider[*ErrorResponse, *ChatRequest, *ChatResponse, ChatStreamChunkResponse]
}

// New creates a new client to talk to the Anthropic platform API.
//
// If opts.APIKey is not provided, it tries to load it from the ANTHROPIC_API_KEY environment variable.
// If none is found, it will still return a client coupled with an base.ErrAPIKeyRequired error.
// Get an API key at https://console.anthropic.com/settings/keys
//
// To use multiple models, create multiple clients.
// Use one of the model from https://docs.anthropic.com/en/docs/about-claude/models/all-models
//
// wrapper optionally wraps the HTTP transport. Useful for HTTP recording and playback, or to tweak HTTP
// retries, or to throttle outgoing requests.
func New(ctx context.Context, opts *genai.ProviderOptions, wrapper func(http.RoundTripper) http.RoundTripper) (*Client, error) {
	if opts.AccountID != "" {
		return nil, errors.New("unexpected option AccountID")
	}
	if opts.Remote != "" {
		return nil, errors.New("unexpected option Remote")
	}
	const apiKeyURL = "https://console.anthropic.com/settings/keys"
	var err error
	apiKey := opts.APIKey
	if apiKey == "" {
		if apiKey = os.Getenv("ANTHROPIC_API_KEY"); apiKey == "" {
			err = &base.ErrAPIKeyRequired{EnvVar: "ANTHROPIC_API_KEY", URL: apiKeyURL}
		}
	}
	mod := genai.Modalities{genai.ModalityText}
	if len(opts.OutputModalities) != 0 && !slices.Equal(opts.OutputModalities, mod) {
		return nil, fmt.Errorf("unexpected option Modalities %s, only text is supported", mod)
	}
	t := base.DefaultTransport
	if wrapper != nil {
		t = wrapper(t)
	}
	// Anthropic allows Opaque fields for thinking signatures
	c := &Client{
		impl: base.Provider[*ErrorResponse, *ChatRequest, *ChatResponse, ChatStreamChunkResponse]{
			GenSyncURL:           "https://api.anthropic.com/v1/messages",
			ProcessStreamPackets: processStreamPackets,
			PreloadedModels:      opts.PreloadedModels,
			ProcessHeaders:       processHeaders,
			ProviderBase: base.ProviderBase[*ErrorResponse]{
				APIKeyURL: apiKeyURL,
				Lenient:   internal.BeLenient,
				Client: http.Client{
					Transport: &roundtrippers.Header{
						Header:    http.Header{"x-api-key": {apiKey}, "anthropic-version": {"2023-06-01"}},
						Transport: &roundtrippers.RequestID{Transport: t},
					},
				},
			},
		},
	}
	if err == nil {
		switch opts.Model {
		case genai.ModelNone:
		case genai.ModelCheap, genai.ModelGood, genai.ModelSOTA, "":
			if c.impl.Model, err = c.selectBestTextModel(ctx, opts.Model); err != nil {
				return nil, err
			}
			c.impl.OutputModalities = mod
		default:
			c.impl.Model = opts.Model
			c.impl.OutputModalities = mod
		}
	}
	return c, err
}

// selectBestTextModel selects the most recent model based on the preference (cheap, good, or SOTA).
//
// We may want to make this function overridable in the future by the client since this is going to break one
// day or another.
func (c *Client) selectBestTextModel(ctx context.Context, preference string) (string, error) {
	mdls, err := c.ListModels(ctx)
	if err != nil {
		return "", fmt.Errorf("failed to automatically select the model: %w", err)
	}
	cheap := preference == genai.ModelCheap
	good := preference == genai.ModelGood || preference == ""
	selectedModel := ""
	var date time.Time
	for _, mdl := range mdls {
		m := mdl.(*Model)
		// Always select the most recent model.
		if !date.IsZero() && m.CreatedAt.Before(date) {
			continue
		}
		if cheap {
			if strings.Contains(m.ID, "-haiku-") {
				date = m.CreatedAt
				selectedModel = m.ID
			}
		} else if good {
			if strings.Contains(m.ID, "-sonnet-") {
				date = m.CreatedAt
				selectedModel = m.ID
			}
		} else {
			if strings.Contains(m.ID, "-opus-") {
				date = m.CreatedAt
				selectedModel = m.ID
			}
		}
	}
	if selectedModel == "" {
		return "", errors.New("failed to find a model automatically")
	}
	return selectedModel, nil
}

// GenAsync implements genai.ProviderGenAsync.
//
// It requests the providers' batch API and returns the job ID. It can take up to 24 hours to complete.
func (c *Client) GenAsync(ctx context.Context, msgs genai.Messages, opts ...genai.Options) (genai.Job, error) {
	if err := c.impl.Validate(); err != nil {
		return "", err
	}
	// https://docs.anthropic.com/en/docs/build-with-claude/batch-processing
	// https://docs.anthropic.com/en/api/creating-message-batches
	// Anthropic supports creating multiple processing requests as part on one HTTP post. I'm not sure the value
	// of doing that, as it increases the risks of partial failure. So for now do not expose the functionality
	// of creating multiple requests at once unless we realize there's a specific use case.
	b := BatchRequest{Requests: []BatchRequestItem{{}}}
	if err := b.Requests[0].Init(msgs, c.impl.Model, opts...); err != nil {
		return "", err
	}
	resp, err := c.GenAsyncRaw(ctx, b)
	return genai.Job(resp.ID), err
}

// GenAsyncRaw provides access to the raw API structure.
func (c *Client) GenAsyncRaw(ctx context.Context, b BatchRequest) (BatchResponse, error) {
	resp := BatchResponse{}
	u := "https://api.anthropic.com/v1/messages/batches"
	err := c.impl.DoRequest(ctx, "POST", u, &b, &resp)
	return resp, err
}

// PokeResult implements genai.ProviderGenAsync.
//
// It retrieves the result for a job ID.
func (c *Client) PokeResult(ctx context.Context, id genai.Job) (genai.Result, error) {
	res := genai.Result{}
	resp, err := c.PokeResultRaw(ctx, id)
	if err != nil && resp.Result.Type == "not_found_error" {
		res.Usage.FinishReason = genai.Pending
		return res, nil
	}
	if resp.Result.Type == "errored" {
		return res, fmt.Errorf("error %s: %s", resp.Result.Error.Error.Type, resp.Result.Error.Error.Message)
	}
	err = resp.To(&res.Message)
	res.Usage.InputTokens = resp.Result.Message.Usage.InputTokens
	res.Usage.InputCachedTokens = resp.Result.Message.Usage.CacheReadInputTokens
	res.Usage.OutputTokens = resp.Result.Message.Usage.OutputTokens
	res.Usage.TotalTokens = res.Usage.InputTokens + res.Usage.InputCachedTokens + res.Usage.OutputTokens
	res.Usage.FinishReason = resp.Result.Message.StopReason.ToFinishReason()
	if err == nil {
		err = res.Validate()
	}
	return res, err
}

// PokeResultRaw provides access to the raw API structure.
func (c *Client) PokeResultRaw(ctx context.Context, id genai.Job) (BatchQueryResponse, error) {
	resp := BatchQueryResponse{}
	// Warning: The documentation at https://docs.anthropic.com/en/api/retrieving-message-batch-results states
	// that the URL may change in the future.
	u := "https://api.anthropic.com/v1/messages/batches/" + url.PathEscape(string(id)) + "/results"
	err := c.impl.DoRequest(ctx, "GET", u, nil, &resp)
	if err != nil {
		var herr *httpjson.Error
		if errors.As(err, &herr) && herr.StatusCode == 404 {
			er := ErrorResponse{}
			if json.Unmarshal(herr.ResponseBody, &er) == nil {
				if er.ErrorVal.Type == "not_found_error" {
					resp.Result.Type = er.ErrorVal.Type
				}
			}
		}
	}
	return resp, err
}

func (c *Client) Cancel(ctx context.Context, id genai.Job) error {
	_, err := c.CancelRaw(ctx, id)
	return err
}

func (c *Client) CancelRaw(ctx context.Context, id genai.Job) (BatchResponse, error) {
	u := "https://api.anthropic.com/v1/messages/batches/" + url.PathEscape(string(id)) + "/cancel"
	resp := BatchResponse{}
	err := c.impl.DoRequest(ctx, "POST", u, &struct{}{}, &resp)
	return resp, err
}

// Name implements genai.Provider.
//
// It returns the name of the provider.
func (c *Client) Name() string {
	return "anthropic"
}

// ModelID implements genai.Provider.
//
// It returns the selected model ID.
func (c *Client) ModelID() string {
	return c.impl.Model
}

// OutputModalities implements genai.Provider.
//
// It returns the output modalities, i.e. what kind of output the model will generate (text, audio, image,
// video, etc).
func (c *Client) OutputModalities() genai.Modalities {
	return c.impl.OutputModalities
}

// Scoreboard implements scoreboard.ProviderScore.
func (c *Client) Scoreboard() scoreboard.Score {
	return Scoreboard
}

// GenSync implements genai.Provider.
func (c *Client) GenSync(ctx context.Context, msgs genai.Messages, opts ...genai.Options) (genai.Result, error) {
	return c.impl.GenSync(ctx, msgs, opts...)
}

// GenSyncRaw provides access to the raw API.
func (c *Client) GenSyncRaw(ctx context.Context, in *ChatRequest, out *ChatResponse) error {
	return c.impl.GenSyncRaw(ctx, in, out)
}

// GenStream implements genai.Provider.
func (c *Client) GenStream(ctx context.Context, msgs genai.Messages, opts ...genai.Options) (iter.Seq[genai.ReplyFragment], func() (genai.Result, error)) {
	return c.impl.GenStream(ctx, msgs, opts...)
}

// GenStreamRaw provides access to the raw API.
func (c *Client) GenStreamRaw(ctx context.Context, in *ChatRequest, out chan<- ChatStreamChunkResponse) error {
	return c.impl.GenStreamRaw(ctx, in, out)
}

// ListModels implements genai.Provider.
func (c *Client) ListModels(ctx context.Context) ([]genai.Model, error) {
	if c.impl.PreloadedModels != nil {
		return c.impl.PreloadedModels, nil
	}
	// https://docs.anthropic.com/en/api/models-list
	var resp ModelsResponse
	if err := c.impl.DoRequest(ctx, "GET", "https://api.anthropic.com/v1/models?limit=1000", nil, &resp); err != nil {
		return nil, err
	}
	return resp.ToModels(), nil
}

func processStreamPackets(ch <-chan ChatStreamChunkResponse, chunks chan<- genai.ReplyFragment, result *genai.Result) error {
	defer func() {
		// We need to empty the channel to avoid blocking the goroutine.
		for range ch {
		}
	}()
	pendingCall := genai.ToolCall{}
	for pkt := range ch {
		f := genai.ReplyFragment{}
		// See testdata/TestClient_Chat_thinking/ChatStream.yaml as a great example.
		// TODO: pkt.Index matters here, as the LLM may fill multiple content blocks simultaneously.
		switch pkt.Type {
		case ChunkMessageStart:
			switch pkt.Message.Role {
			case "assistant":
			default:
				return fmt.Errorf("unexpected role %q", pkt.Message.Role)
			}
			result.Usage.InputTokens = pkt.Message.Usage.InputTokens
			result.Usage.InputCachedTokens = pkt.Message.Usage.CacheReadInputTokens
			// There's some tokens listed there. Still save it in case it breaks midway.
			result.Usage.OutputTokens = pkt.Message.Usage.OutputTokens
			result.Usage.TotalTokens = result.Usage.InputTokens + result.Usage.InputCachedTokens + result.Usage.OutputTokens
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
			case ContentRedactedThinking:
				f.Opaque = map[string]any{"redacted_thinking": pkt.ContentBlock.Signature}
			case ContentImage, ContentDocument, ContentToolResult:
				fallthrough
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
			case DeltaCitations:
				if err := pkt.Delta.Citation.To(&f.Citation); err != nil {
					return fmt.Errorf("failed to parse citation: %w", err)
				}
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
			result.Usage.FinishReason = pkt.Delta.StopReason.ToFinishReason()
			result.Usage.OutputTokens = pkt.Usage.OutputTokens
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

func processHeaders(h http.Header) []genai.RateLimit {
	requestsLimit, _ := strconv.ParseInt(h.Get("Anthropic-Ratelimit-Requests-Limit"), 10, 64)
	requestsRemaining, _ := strconv.ParseInt(h.Get("Anthropic-Ratelimit-Requests-Remaining"), 10, 64)
	requestsReset, _ := time.Parse(time.RFC3339, h.Get("Anthropic-Ratelimit-Requests-Reset"))

	tokensLimit, _ := strconv.ParseInt(h.Get("Anthropic-Ratelimit-Tokens-Limit"), 10, 64)
	tokensRemaining, _ := strconv.ParseInt(h.Get("Anthropic-Ratelimit-Tokens-Remaining"), 10, 64)
	tokensReset, _ := time.Parse(time.RFC3339, h.Get("Anthropic-Ratelimit-Tokens-Reset"))

	var limits []genai.RateLimit
	if requestsLimit > 0 {
		limits = append(limits, genai.RateLimit{
			Type:      genai.Requests,
			Period:    genai.PerOther,
			Limit:     requestsLimit,
			Remaining: requestsRemaining,
			Reset:     requestsReset.Round(10 * time.Millisecond),
		})
	}
	if tokensLimit > 0 {
		limits = append(limits, genai.RateLimit{
			Type:      genai.Tokens,
			Period:    genai.PerOther,
			Limit:     tokensLimit,
			Remaining: tokensRemaining,
			Reset:     tokensReset.Round(10 * time.Millisecond),
		})
	}
	return limits
}

var (
	_ genai.Validatable        = &Message{}
	_ genai.Validatable        = &Content{}
	_ genai.Provider           = &Client{}
	_ scoreboard.ProviderScore = &Client{}
)
