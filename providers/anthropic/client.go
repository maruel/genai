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
	_ "embed"
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

//go:embed scoreboard.json
var scoreboardJSON []byte

// Scoreboard for Anthropic.
func Scoreboard() scoreboard.Score {
	var s scoreboard.Score
	d := json.NewDecoder(bytes.NewReader(scoreboardJSON))
	d.DisallowUnknownFields()
	if err := d.Decode(&s); err != nil {
		panic(fmt.Errorf("failed to unmarshal scoreboard.json: %w", err))
	}
	return s
}

// GenOptionsText defines Anthropic specific options.
type GenOptionsText struct {
	// ThinkingBudget is the maximum number of tokens the LLM can use to reason about the answer. When 0,
	// reasoning is disabled. It generally must be above 1024 and below MaxTokens.
	ThinkingBudget int64
	// MessagesToCache specify the number of messages to cache in the request.
	//
	// By default, the system prompt and tools will be cached.
	//
	// https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching
	MessagesToCache int
}

func (o *GenOptionsText) Validate() error {
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
func (c *ChatRequest) Init(msgs genai.Messages, model string, opts ...genai.GenOptions) error {
	return c.initImpl(msgs, model, true, opts...)
}

func (c *ChatRequest) initImpl(msgs genai.Messages, model string, cache bool, opts ...genai.GenOptions) error {
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
		case *GenOptionsText:
			if cache {
				msgToCache = v.MessagesToCache
			} else if v.MessagesToCache != 0 {
				unsupported = append(unsupported, "GenOptionsText.MessagesToCache")
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
		case *genai.GenOptionsText:
			u, e := c.initOptionsText(v)
			unsupported = append(unsupported, u...)
			errs = append(errs, e...)
		case *genai.GenOptionsTools:
			u, e := c.initOptionsTools(v)
			unsupported = append(unsupported, u...)
			errs = append(errs, e...)
		default:
			unsupported = append(unsupported, internal.TypeName(opt))
		}
	}

	// Post process to take into account limitations by the provider.
	if c.Model == "claude-sonnet-4-20250514" && c.Thinking.Type == "enabled" && c.ToolChoice.Type == ToolChoiceAny {
		unsupported = append(unsupported, "GenOptionsTools.Force")
		c.ToolChoice.Type = ToolChoiceAuto
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
		if len(errs) == 0 {
			if err := c.Messages[len(c.Messages)-1].Validate(); err != nil {
				errs = append(errs, fmt.Errorf("message #%d: %w", i, err))
			}
		}
	}
	// If we have unsupported features but no other errors, return a structured error.
	if len(unsupported) > 0 && len(errs) == 0 {
		return &base.ErrNotSupported{Options: unsupported}
	}
	return errors.Join(errs...)
}

func (c *ChatRequest) SetStream(stream bool) {
	c.Stream = stream
}

func (c *ChatRequest) initOptionsText(v *genai.GenOptionsText) ([]string, []error) {
	var unsupported []string
	var errs []error
	if v.TopLogprobs > 0 {
		unsupported = append(unsupported, "GenOptionsText.TopLogprobs")
	}
	if v.MaxTokens != 0 {
		c.MaxTokens = v.MaxTokens
	}
	c.Temperature = v.Temperature
	if v.SystemPrompt != "" {
		c.System = []SystemMessage{{Type: "text", Text: v.SystemPrompt}}
		// TODO: Add automatic caching.
		// c.System[0].CacheControl.Type = "ephemeral"
	}
	c.TopP = v.TopP
	c.TopK = v.TopK
	c.StopSequences = v.Stop
	if v.ReplyAsJSON {
		errs = append(errs, errors.New("unsupported option ReplyAsJSON"))
	}
	if v.DecodeAs != nil {
		errs = append(errs, errors.New("unsupported option DecodeAs"))
	}
	return unsupported, errs
}

func (c *ChatRequest) initOptionsTools(v *genai.GenOptionsTools) ([]string, []error) {
	var unsupported []string
	var errs []error
	if len(v.Tools) != 0 {
		switch v.Force {
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
	if v.WebSearch {
		// https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/web-search-tool
		c.Tools = append(c.Tools, Tool{
			Type: "web_search_20250305",
			Name: "web_search",
			// MaxUses: 10,
		})
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

// MCPServer is documented at https://docs.anthropic.com/en/api/messages#body-mcp-servers
//
// https://docs.anthropic.com/en/docs/agents-and-tools/mcp-connector
// states that HTTP header "anthropic-beta": "mcp-client-2025-04-04"
// required.
type MCPServer struct {
	Name               string            `json:"name"` //
	Type               string            `json:"type"` // "url"
	URL                string            `json:"url"`
	AuthorizationToken string            `json:"authorization_token,omitzero"`
	ToolConfiguration  ToolConfiguration `json:"tool_configuration,omitzero"`
}

type ToolConfiguration struct {
	AllowedTools []string `json:"allowed_tools,omitzero"`
	Enabled      bool     `json:"enabled,omitzero"`
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
	return m.validate(true)
}

func (m *Message) validate(strict bool) error {
	if m.Type != "" && m.Type != "message" {
		// Allow empty type, it is not required.
		return &internal.BadError{Err: fmt.Errorf("implement message type %q", m.Type)}
	}
	switch m.Role {
	case "assistant", "user":
		// Valid.
	case "":
		return &internal.BadError{Err: errors.New("message doesn't have role defined")}
	default:
		return &internal.BadError{Err: fmt.Errorf("unsupported role %q", m.Role)}
	}
	if len(m.Content) == 0 {
		return &internal.BadError{Err: errors.New("message doesn't have content defined")}
	}
	for i := range m.Content {
		if err := m.Content[i].validate(strict); err != nil {
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
		return &internal.BadError{Err: fmt.Errorf("unsupported role %q", r)}
	}
	m.Content = make([]Content, 0, len(in.Requests)+len(in.Replies)+len(in.ToolCallResults))
	for i := range in.Requests {
		m.Content = append(m.Content, Content{})
		if err := m.Content[len(m.Content)-1].FromRequest(&in.Requests[i]); err != nil {
			return fmt.Errorf("request #%d: %w", i, err)
		}
	}
	for i := range in.Replies {
		c := Content{}
		if skip, err := c.FromReply(&in.Replies[i]); err != nil {
			return fmt.Errorf("reply #%d: %w", i, err)
		} else if !skip {
			m.Content = append(m.Content, c)
		}
	}
	for i := range in.ToolCallResults {
		m.Content = append(m.Content, Content{})
		m.Content[len(m.Content)-1].FromToolCallResult(&in.ToolCallResults[i])
	}
	return nil
}

func (m *Message) To(out *genai.Message) error {
	// Make sure the message was initialized properly before converting.
	// Ask for a "non-strict" check because sometimes the server replies with an invalid message (!)
	if err := m.validate(false); err != nil {
		return err
	}
	// We need to split actual content and tool calls.
	for i := range m.Content {
		replies, err := m.Content[i].To()
		if err != nil {
			return fmt.Errorf("reply #%d: %w", i, err)
		}
		out.Replies = append(out.Replies, replies...)
	}
	return nil
}

// Content is a recursive data type of content.
type Content struct {
	// Type is what this content is, which defines which fields are valid. Some types are only valid at the root
	// level and some types are only valid in embedded content.
	Type ContentType `json:"type"`
	// Type == ContentText.
	Text string `json:"text,omitzero"`

	// Type == ContentThinking.
	Thinking  string `json:"thinking,omitzero"`
	Signature []byte `json:"signature,omitzero"`

	// Type == ContentRedactedThinking.
	Data string `json:"data,omitzero"`

	// Type == ContentText, ContentImage, ContentDocument, ContentToolUse, ContentToolResult.
	CacheControl struct {
		Type string `json:"type,omitzero"` // "ephemeral"
	} `json:"cache_control,omitzero"`

	// Type == ContentText, ContentDocument.
	Citations Citations `json:"citations,omitzero"`

	// Type == ContentImage, ContentDocument.
	Source Source `json:"source,omitzero"`

	// Type == ContentToolUse, ContentServerToolUse, ContentMCPToolUse
	ID string `json:"id,omitzero"`
	// For ContentServerToolUse, it will be {"query": "..."}. For ContentMCPToolUse, it depends on the MCP
	// server.
	Input any `json:"input,omitzero"` // To reorder, I need to redo HTTP recordings.
	// Type == ContentToolUse, ContentServerToolUse, ContentMCPToolUse
	Name string `json:"name,omitzero"`

	// Type == ContentToolResult, ContentMCPToolResult
	ToolUseID string `json:"tool_use_id,omitzero"`
	IsError   bool   `json:"is_error,omitzero"`

	// Type == ContentToolResult, ContentWebSearchToolResult, ContentMCPToolResult
	// - ContentToolResult: Only ContentText and ContentImage are allowed.
	// - ContentWebSearchToolResult: Only ContentWebSearchResult is allowed.
	Content []Content `json:"content,omitzero"`

	// Type == ContentMCPToolUse
	ServerName string `json:"server_name,omitzero"`

	// Type == ContentDocument
	Context string `json:"context,omitzero"` // Context about the document that will not be cited from

	// Type == ContentWebSearchResult
	URL              string `json:"url,omitzero"`
	EncryptedContent string `json:"encrypted_content,omitzero"`
	PageAge          string `json:"page_age,omitzero"` // "12 hours ago", "4 days ago", "1 week ago", "April 29, 2025", null

	// Type == ContentDocument, ContentWebSearchResult
	Title string `json:"title,omitzero"` // Document title when using Source, web page title
}

// Validate checks that the expected fields are set.
func (c *Content) Validate() error {
	return c.validate(true)
}

// validate checks that the expected fields are set.
func (c *Content) validate(strict bool) error {
	switch c.Type {
	case ContentText:
		// It happens with citations.
		if c.Text == "" && len(c.Citations.Citations) == 0 {
			// The server sometimes return invalid messages with no text or citations (!). This is present in
			// testdata/TestClient/Scoreboard/claude-sonnet-4-20250514_thinking/GenSync-Citations-text-plain.yaml
			if strict {
				return &internal.BadError{Err: fmt.Errorf("%s: fields Text or Citations must be set", c.Type)}
			}
		}
		if c.Thinking != "" || len(c.Signature) > 0 || c.Data != "" || !c.Source.IsZero() || c.ID != "" || c.Name != "" || c.Input != nil ||
			c.ToolUseID != "" || c.IsError || len(c.Content) > 0 || c.ServerName != "" || c.Context != "" || c.URL != "" || c.EncryptedContent != "" || c.PageAge != "" || c.Title != "" {
			return &internal.BadError{Err: fmt.Errorf("%s: unexpected fields set", c.Type)}
		}
	case ContentThinking:
		if c.Thinking == "" {
			// c.Signature is optional.
			return &internal.BadError{Err: fmt.Errorf("%s: fields Thinking must be set", c.Type)}
		}
		if c.Text != "" || c.Data != "" || len(c.Citations.Citations) > 0 || !c.Source.IsZero() || c.ID != "" || c.Name != "" || c.Input != nil ||
			c.ToolUseID != "" || c.IsError || len(c.Content) > 0 || c.ServerName != "" || c.Context != "" || c.URL != "" || c.EncryptedContent != "" || c.PageAge != "" || c.Title != "" {
			return &internal.BadError{Err: fmt.Errorf("%s: unexpected fields set", c.Type)}
		}
	case ContentRedactedThinking:
		if c.Data == "" {
			return &internal.BadError{Err: fmt.Errorf("%s: fields Data must be set", c.Type)}
		}
		if c.Text != "" || c.Thinking != "" || len(c.Signature) > 0 || len(c.Citations.Citations) > 0 || !c.Source.IsZero() || c.ID != "" || c.Name != "" || c.Input != nil ||
			c.ToolUseID != "" || c.IsError || len(c.Content) > 0 || c.ServerName != "" || c.Context != "" || c.URL != "" || c.EncryptedContent != "" || c.PageAge != "" || c.Title != "" {
			return &internal.BadError{Err: fmt.Errorf("%s: unexpected fields set", c.Type)}
		}
	case ContentImage, ContentDocument:
		if c.Source.Type == "" && c.Source.URL == "" && c.Source.Data == "" {
			return &internal.BadError{Err: fmt.Errorf("%s: fields Source must be set", c.Type)}
		}
		if c.Text != "" || c.Thinking != "" || len(c.Signature) > 0 || c.Data != "" || len(c.Citations.Citations) > 0 || c.ID != "" || c.Name != "" || c.Input != nil ||
			c.ToolUseID != "" || c.IsError || len(c.Content) > 0 || c.ServerName != "" || c.Context != "" || c.URL != "" || c.EncryptedContent != "" || c.PageAge != "" || c.Title != "" {
			return &internal.BadError{Err: fmt.Errorf("%s: unexpected fields set", c.Type)}
		}
	case ContentToolUse:
		if c.ID == "" || c.Name == "" || c.Input == nil {
			return &internal.BadError{Err: fmt.Errorf("%s: fields ID, Name, Input must be set", c.Type)}
		}
		if c.Text != "" || c.Thinking != "" || len(c.Signature) > 0 || c.Data != "" || len(c.Citations.Citations) > 0 || !c.Source.IsZero() ||
			c.ToolUseID != "" || c.IsError || len(c.Content) > 0 || c.ServerName != "" || c.Context != "" || c.URL != "" || c.EncryptedContent != "" || c.PageAge != "" || c.Title != "" {
			return &internal.BadError{Err: fmt.Errorf("%s: unexpected fields set", c.Type)}
		}
	case ContentToolResult:
		if c.ToolUseID == "" || len(c.Content) == 0 {
			return &internal.BadError{Err: fmt.Errorf("%s: fields ToolUseID, Content, must be set", c.Type)}
		}
		if c.Text != "" || c.Thinking != "" || len(c.Signature) > 0 || c.Data != "" || len(c.Citations.Citations) > 0 || !c.Source.IsZero() || c.ID != "" || c.Name != "" || c.Input != nil ||
			c.IsError || c.ServerName != "" || c.Context != "" || c.URL != "" || c.EncryptedContent != "" || c.PageAge != "" || c.Title != "" {
			return &internal.BadError{Err: fmt.Errorf("%s: unexpected fields set", c.Type)}
		}
	case ContentMCPToolUse:
		if c.ID == "" || c.Name == "" || c.Input == nil || c.ServerName == "" {
			return &internal.BadError{Err: fmt.Errorf("%s: fields ID, Name, Input, ServerName must be set", c.Type)}
		}
		if c.Text != "" || c.Thinking != "" || len(c.Signature) > 0 || c.Data != "" || len(c.Citations.Citations) > 0 || !c.Source.IsZero() ||
			c.ToolUseID != "" || c.IsError || len(c.Content) > 0 || c.Context != "" || c.URL != "" || c.EncryptedContent != "" || c.PageAge != "" || c.Title != "" {
			return &internal.BadError{Err: fmt.Errorf("%s: unexpected fields set", c.Type)}
		}
	case ContentMCPToolResult:
		if c.ToolUseID == "" || len(c.Content) == 0 {
			return &internal.BadError{Err: fmt.Errorf("%s: fields ToolUseID, Content must be set", c.Type)}
		}
		if c.Text != "" || c.Thinking != "" || len(c.Signature) > 0 || c.Data != "" || len(c.Citations.Citations) > 0 || !c.Source.IsZero() || c.ID != "" || c.Name != "" || c.Input != nil ||
			c.ServerName != "" || c.Context != "" || c.URL != "" || c.EncryptedContent != "" || c.PageAge != "" || c.Title != "" {
			return &internal.BadError{Err: fmt.Errorf("%s: unexpected fields set", c.Type)}
		}
	case ContentServerToolUse:
		if c.ID == "" || c.Name == "" || c.Input == nil {
			return &internal.BadError{Err: fmt.Errorf("%s: fields ID, Name, Input must be set", c.Type)}
		}
		if c.Text != "" || c.Thinking != "" || len(c.Signature) > 0 || c.Data != "" || len(c.Citations.Citations) > 0 || !c.Source.IsZero() ||
			c.ToolUseID != "" || c.IsError || len(c.Content) > 0 || c.ServerName != "" || c.Context != "" || c.URL != "" || c.EncryptedContent != "" || c.PageAge != "" || c.Title != "" {
			return &internal.BadError{Err: fmt.Errorf("%s: unexpected fields set", c.Type)}
		}
	case ContentWebSearchToolResult:
		if c.ToolUseID == "" || len(c.Content) == 0 {
			return &internal.BadError{Err: fmt.Errorf("%s: fields ToolUseID, Content must be set", c.Type)}
		}
		if c.Text != "" || c.Thinking != "" || len(c.Signature) > 0 || c.Data != "" || len(c.Citations.Citations) > 0 || !c.Source.IsZero() || c.ID != "" || c.Name != "" || c.Input != nil ||
			c.ServerName != "" || c.Context != "" || c.URL != "" || c.EncryptedContent != "" || c.PageAge != "" || c.Title != "" {
			return &internal.BadError{Err: fmt.Errorf("%s: unexpected fields set", c.Type)}
		}
	case ContentWebSearchResult:
		if c.URL == "" {
			return &internal.BadError{Err: fmt.Errorf("%s: fields URL must be set", c.Type)}
		}
		if c.Text != "" || c.Thinking != "" || len(c.Signature) > 0 || c.Data != "" || len(c.Citations.Citations) > 0 || !c.Source.IsZero() || c.ID != "" || c.Name != "" || c.Input != nil ||
			c.ToolUseID != "" || c.IsError || len(c.Content) > 0 || c.ServerName != "" || c.Context != "" || c.EncryptedContent != "" || c.PageAge != "" || c.Title != "" {
			return &internal.BadError{Err: fmt.Errorf("%s: unexpected fields set", c.Type)}
		}
	default:
		return &internal.BadError{Err: fmt.Errorf("implement ContentType %q", c.Type)}
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
			// text/plain, text/markdown
		case strings.HasPrefix(mimeType, "text/"):
			c.Type = ContentDocument
			if in.Doc.URL != "" {
				return fmt.Errorf("%s documents must be provided inline, not as a URL", mimeType)
			}
			// In particular, the API refuses "text/plain; charset=utf-8" or "text/markdown". WTF.
			c.Source.MediaType = "text/plain"
			c.Source.Type = SourceText
			c.Source.Data = string(data)
			// Enable citations for text documents
			c.Citations = Citations{Enabled: true}
		default:
			return fmt.Errorf("unsupported content mime-type %s", mimeType)
		}
		return nil
	}
	return errors.New("unknown Request type")
}

func (c *Content) FromReply(in *genai.Reply) (bool, error) {
	if in.Text != "" {
		c.Type = ContentText
		c.Text = in.Text
		return false, nil
	}
	if in.Reasoning != "" {
		c.Type = ContentThinking
		c.Thinking = in.Reasoning
		if in.Opaque != nil {
			if b, ok := in.Opaque["signature"].([]byte); ok {
				c.Signature = b
			}
		}
		return false, nil
	}
	if in.Opaque != nil {
		if s, ok := in.Opaque["redacted_thinking"].(string); ok {
			c.Type = ContentRedactedThinking
			c.Data = s
			return false, nil
		}
		if v, ok := in.Opaque["mcp_tool_use"].(map[string]any); ok {
			c.Type = ContentMCPToolUse
			ok := false
			if c.ID, ok = v["id"].(string); !ok {
				return false, errors.New("field Opaque.mcp_tool_use.id not found")
			}
			if c.Name, ok = v["name"].(string); !ok {
				return false, errors.New("field Opaque.mcp_tool_use.name not found")
			}
			if c.Input, ok = v["input"]; !ok {
				return false, errors.New("field Opaque.mcp_tool_use.input not found")
			}
			return false, nil
		}
		if v, ok := in.Opaque["mcp_tool_result"].(map[string]any); ok {
			c.Type = ContentMCPToolResult
			ok := false
			if c.ToolUseID, ok = v["tool_use_id"].(string); !ok {
				return false, errors.New("field Opaque.mcp_tool_result.tool_use_id not found")
			}
			if c.IsError, ok = v["is_error"].(bool); !ok {
				return false, errors.New("field Opaque.mcp_tool_result.is_error not found")
			}
			if c.Content, ok = v["content"].([]Content); !ok {
				return false, errors.New("field Opaque.mcp_tool_result.content not found")
			}
			if c.ServerName, ok = v["tool_use_id"].(string); !ok {
				return false, errors.New("field Opaque.mcp_tool_result.server_name not found")
			}
			return false, nil
		}
		return false, fmt.Errorf("unexpected Opaque %v", in.Opaque)
	}
	if !in.ToolCall.IsZero() {
		if len(in.ToolCall.Opaque) != 0 {
			return false, errors.New("field ToolCall.Opaque not supported")
		}
		c.Type = ContentToolUse
		c.ID = in.ToolCall.ID
		c.Name = in.ToolCall.Name
		if err := json.Unmarshal([]byte(in.ToolCall.Arguments), &c.Input); err != nil {
			return false, fmt.Errorf("failed to marshal input: %w; for tool call: %#v", err, in)
		}
		return false, nil
	}
	if !in.Doc.IsZero() {
		mimeType, data, err := in.Doc.Read(10 * 1024 * 1024)
		if err != nil {
			return false, err
		}
		// Anthropic require a mime-type to determine if image or PDF.
		if mimeType == "" {
			return false, fmt.Errorf("unspecified mime type for URL %q", in.Doc.URL)
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
			// text/plain, text/markdown
		case strings.HasPrefix(mimeType, "text/"):
			c.Type = ContentDocument
			if in.Doc.URL != "" {
				return false, fmt.Errorf("%s documents must be provided inline, not as a URL", mimeType)
			}
			// In particular, the API refuses "text/plain; charset=utf-8" or "text/markdown". WTF.
			c.Source.MediaType = "text/plain"
			c.Source.Type = SourceText
			c.Source.Data = string(data)
			// Enable citations for text documents
			c.Citations = Citations{Enabled: true}
		default:
			return false, fmt.Errorf("unsupported content mime-type %s", mimeType)
		}
		return false, nil
	}
	if !in.Citation.IsZero() {
		// Skip.
		return true, nil
	}
	return false, &internal.BadError{Err: errors.New("unknown Reply type")}
}

func (c *Content) FromToolCallResult(in *genai.ToolCallResult) {
	// TODO: Support text citation.
	// TODO: Support image.
	c.Type = ContentToolResult
	c.ToolUseID = in.ID
	c.IsError = false
	c.Content = []Content{{Type: ContentText, Text: in.Result}}
}

func (c *Content) To() ([]genai.Reply, error) {
	var out []genai.Reply
	switch c.Type {
	case ContentText:
		if c.Text != "" {
			r := genai.Reply{Text: c.Text}
			if len(c.Citations.Citations) > 0 {
				// Signal that if we ever send the citations back, we want to merge them with the text.
				r.Opaque = map[string]any{"merge": true}
			}
			out = append(out, r)
		}
		for i := range c.Citations.Citations {
			r := genai.Reply{}
			if err := c.Citations.Citations[i].To(&r.Citation); err != nil {
				return out, fmt.Errorf("citation %d: %w", i, err)
			}
			out = append(out, r)
		}
	case ContentThinking:
		out = append(out, genai.Reply{Reasoning: c.Thinking, Opaque: map[string]any{"signature": c.Signature}})
	case ContentRedactedThinking:
		out = append(out, genai.Reply{Opaque: map[string]any{"redacted_thinking": c.Signature}})
	case ContentToolUse:
		raw, err := json.Marshal(c.Input)
		if err != nil {
			return out, &internal.BadError{Err: fmt.Errorf("failed to marshal input: %w; for tool call: %#v", err, c)}
		}
		out = append(out, genai.Reply{ToolCall: genai.ToolCall{ID: c.ID, Name: c.Name, Arguments: string(raw)}})
	case ContentWebSearchToolResult:
		src := make([]genai.CitationSource, len(c.Content))
		for i, cc := range c.Content {
			if cc.Type != ContentWebSearchResult {
				return out, &internal.BadError{Err: fmt.Errorf("implement content type %q while processing %q", cc.Type, c.Type)}
			}
			src[i].Type = genai.CitationWeb
			src[i].URL = cc.URL
			src[i].Title = cc.Title
			src[i].Date = cc.PageAge
			// TODO: Keep cc.EncryptedContent to be able to continue the thread.
		}
		out = append(out, genai.Reply{Citation: genai.Citation{Sources: src}})
	case ContentServerToolUse:
		// TODO: We drop the value, which makes the next request unusable.
		switch c.Name {
		case "web_search":
			q := WebSearch{}
			// We marshal then unmarshal to normalize the input.
			b, err := json.Marshal(c.Input)
			if err != nil {
				return out, &internal.BadError{Err: fmt.Errorf("failed to marshal server tool call %s: %w", c.Name, err)}
			}
			d := json.NewDecoder(bytes.NewReader(b))
			if !internal.BeLenient {
				d.DisallowUnknownFields()
			}
			if err := d.Decode(&q); err != nil {
				return out, &internal.BadError{Err: fmt.Errorf("failed to decode server tool call %s: %w", c.Name, err)}
			}
			out = append(out, genai.Reply{
				Citation: genai.Citation{Sources: []genai.CitationSource{{Type: genai.CitationWebQuery, Snippet: q.Query}}},
			})
		default:
			// Oops, more work to do!
			if !internal.BeLenient {
				return out, &internal.BadError{Err: fmt.Errorf("implement server tool call %q", c.Name)}
			}
		}
	case ContentMCPToolUse:
		opaque := map[string]any{"mcp_tool_use": map[string]any{
			"id":    c.ID,
			"input": c.Input,
			"name":  c.Name,
		}}
		out = append(out, genai.Reply{Opaque: opaque})
	case ContentMCPToolResult:
		opaque := map[string]any{"mcp_tool_result": map[string]any{
			"tool_use_id": c.ToolUseID,
			"is_error":    c.IsError,
			"content":     c.Content,
			"server_name": c.ServerName,
		}}
		out = append(out, genai.Reply{Opaque: opaque})
	case ContentWebSearchResult, ContentImage, ContentDocument, ContentToolResult:
		fallthrough
	default:
		return out, &internal.BadError{Err: fmt.Errorf("implement content type %q", c.Type)}
	}
	return out, nil
}

// WebSearch is the server tool use.
type WebSearch struct {
	Query string `json:"query"`
}

// SourceType is described at https://docs.anthropic.com/en/api/messages#body-messages-content-source
type SourceType string

const (
	SourceBase64  SourceType = "base64"
	SourceURL     SourceType = "url"
	SourceText    SourceType = "text"
	SourceContent SourceType = "content"
)

type Source struct {
	Type SourceType `json:"type,omitzero"`

	// Type == SourceBase64, SourceURL, SourceText
	//
	// Content.Type == ContentImage: "image/gif", "image/jpeg", "image/png", "image/webp"
	// Content.Type == ContentDocument: "application/pdf", "text/plain"
	MediaType string `json:"media_type,omitzero"`

	// Type == SourceBase64, SourceURL, SourceText
	Data string `json:"data,omitzero"` // base64 encoded if base64, else as is, e.g. text plain data.

	// Type == SourceURL
	URL string `json:"url,omitzero"`

	// Type == SourceContent
	// Only ContentText and ContentImage are allowed.
	Content []Content `json:"content,omitzero"`

	// Type == SourceFileID
	FileID string `json:"file_id,omitzero"` // File ID for the content, used for file uploads.
}

func (s *Source) IsZero() bool {
	return s.Type == "" && s.MediaType == "" && s.Data == "" && s.URL == "" && len(s.Content) == 0 && s.FileID == ""
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
	ContentText                ContentType = "text"
	ContentImage               ContentType = "image"
	ContentToolUse             ContentType = "tool_use"
	ContentToolResult          ContentType = "tool_result"
	ContentMCPToolUse          ContentType = "mcp_tool_use"
	ContentMCPToolResult       ContentType = "mcp_tool_result"
	ContentDocument            ContentType = "document"
	ContentThinking            ContentType = "thinking"
	ContentRedactedThinking    ContentType = "redacted_thinking"
	ContentServerToolUse       ContentType = "server_tool_use"
	ContentWebSearchResult     ContentType = "web_search_result"
	ContentWebSearchToolResult ContentType = "web_search_tool_result"
)

type CitationType string

const (
	CitationText                    CitationType = "text"
	CitationCharLocation            CitationType = "char_location"
	CitationPageLocation            CitationType = "page_location"
	CitationContentBlockLocation    CitationType = "content_block_location"
	CitationWebSearchResultLocation CitationType = "web_search_result_location"
)

// Citation is used both for system message and user message.
//
// https://docs.anthropic.com/en/api/messages#body-messages-content-citations
// https://docs.anthropic.com/en/api/messages#body-system-citations
type Citation struct {
	Type CitationType `json:"type,omitzero"`

	// Content.Type == CitationText, CitationCharLocation, CitationWebSearchResultLocation
	CitedText string `json:"cited_text,omitzero"`

	// Content.Type == CitationText
	DocumentIndex int64  `json:"document_index,omitzero"`
	DocumentTitle string `json:"document_title,omitzero"`

	// Type == CitationCharLocation
	EndCharIndex   int64 `json:"end_char_index,omitzero"`
	StartCharIndex int64 `json:"start_char_index,omitzero"`

	// Type == CitationPageLocation
	EndPageNumber   int64 `json:"end_page_number,omitzero"`
	StartPageNumber int64 `json:"start_page_number,omitzero"`

	// Type == CitationContentBlockLocation
	EndBlockIndex   int64 `json:"end_block_index,omitzero"`
	StartBlockIndex int64 `json:"start_block_index,omitzero"`

	// Type == CitationWebSearchResultLocation
	EncryptedIndex string `json:"encrypted_index,omitzero"`
	Title          string `json:"title,omitzero"`
	URL            string `json:"url,omitzero"`

	// Content.Type == "document"
	Enabled bool `json:"enabled,omitzero"`
}

func (c *Citation) To(dst *genai.Citation) error {
	switch c.Type {
	case CitationText, CitationCharLocation, CitationPageLocation, CitationContentBlockLocation:
		// TODO: Trigger CitationPageLocation, CitationContentBlockLocation in the smoke test.
		dst.Sources = []genai.CitationSource{{
			Type:            genai.CitationDocument,
			Title:           c.DocumentTitle,
			ID:              strconv.FormatInt(c.DocumentIndex, 10),
			Snippet:         c.CitedText,
			StartCharIndex:  c.StartCharIndex,
			EndCharIndex:    c.EndCharIndex,
			StartPageNumber: c.StartPageNumber,
			EndPageNumber:   c.EndPageNumber,
			StartBlockIndex: c.StartBlockIndex,
			EndBlockIndex:   c.EndBlockIndex,
		}}
	case CitationWebSearchResultLocation:
		// This is only emitted by claude-sonnet-4-20250514 and claude-opus-4-1-20250805 and not
		// claude-3-5-haiku-20241022.
		dst.Sources = []genai.CitationSource{{
			Type:    genai.CitationWeb,
			URL:     c.URL,
			Title:   c.Title,
			Snippet: c.CitedText,
		}}
	default:
		return &internal.BadError{Err: fmt.Errorf("implement handling citation type: %s", c.Type)}
	}
	return nil
}

type Thinking struct {
	BudgetTokens int64  `json:"budget_tokens,omitzero"` // >1024 and less than max_tokens
	Type         string `json:"type,omitzero"`          // "enabled", "disabled"
}

// ToolChoiceType is documented at https://docs.anthropic.com/en/api/messages#body-tool-choice
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

// Tool is documented at https://docs.anthropic.com/en/api/messages#body-tools
type Tool struct {
	Type string `json:"type,omitzero"` // "custom", "computer_20241022", "computer_20250124", "bash_20241022", "bash_20250124", "text_editor_20241022", "text_editor_20250124", "web_search_20250305"
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

	// Type == "web_search_20250305"
	AllowedDomains []string `json:"allowed_domains,omitzero"`
	BlockedDomains []string `json:"blocked_domains,omitzero"`
	UserLocation   struct {
		Type     string `json:"type,omitzero"`     // "approximate"
		City     string `json:"city,omitzero"`     // "San Francisco"
		Region   string `json:"region,omitzero"`   // "California"
		Country  string `json:"country,omitzero"`  // "US"
		Timezone string `json:"timezone,omitzero"` // "America/Los_Angeles"; https://en.wikipedia.org/wiki/List_of_tz_database_time_zones
	} `json:"user_location,omitzero"`
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

		// Type == ContentToolUse, ContentMCPToolUse
		ID    string `json:"id"`
		Name  string `json:"name"`
		Input any    `json:"input"`

		Citations []struct{} `json:"citations"` // Empty, not used in the API.

		// Type == ContentWebSearchToolResult, ContentMCPToolResult
		ToolUseID string    `json:"tool_use_id"`
		Content   []Content `json:"content"`

		// Type == ContentMCPToolResult
		ServerName string `json:"server_name,omitzero"`
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
	ChunkError             ChunkType = "error"
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
	InputTokens              int64 `json:"input_tokens"`
	OutputTokens             int64 `json:"output_tokens"`
	CacheCreationInputTokens int64 `json:"cache_creation_input_tokens"`
	CacheReadInputTokens     int64 `json:"cache_read_input_tokens"`
	CacheCreation            struct {
		Ephemeral1hInputTokens int64 `json:"ephemeral_1h_input_tokens"`
		Ephemeral5mInputTokens int64 `json:"ephemeral_5m_input_tokens"`
	} `json:"cache_creation"`
	ServiceTier   string `json:"service_tier"` // "standard", "batch"
	ServerToolUse struct {
		WebSearchRequests int64 `json:"web_search_requests"`
		WebFetchRequests  int64 `json:"web_fetch_requests"`
	} `json:"server_tool_use"`
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

func (b *BatchRequestItem) Init(msgs genai.Messages, model string, opts ...genai.GenOptions) error {
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
		replies, err := b.Result.Message.Content[i].To()
		if err != nil {
			return fmt.Errorf("block %d: %w", i, err)
		}
		out.Replies = append(out.Replies, replies...)
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
		Details struct{} `json:"details"`
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
	base.NotImplemented
	impl base.Provider[*ErrorResponse, *ChatRequest, *ChatResponse, ChatStreamChunkResponse]
}

// New creates a new client to talk to the Anthropic platform API.
//
// If ProviderOptionAPIKey is not provided, it tries to load it from the ANTHROPIC_API_KEY environment variable.
// If none is found, it will still return a client coupled with an base.ErrAPIKeyRequired error.
// Get an API key at https://console.anthropic.com/settings/keys
//
// To use multiple models, create multiple clients.
// Use one of the model from https://docs.anthropic.com/en/docs/about-claude/models/all-models
func New(ctx context.Context, opts ...genai.ProviderOption) (*Client, error) {
	var apiKey, model string
	var modalities genai.Modalities
	var preloadedModels []genai.Model
	var wrapper func(http.RoundTripper) http.RoundTripper
	for _, opt := range opts {
		if err := opt.Validate(); err != nil {
			return nil, err
		}
		switch v := opt.(type) {
		case genai.ProviderOptionAPIKey:
			apiKey = string(v)
		case genai.ProviderOptionModel:
			model = string(v)
		case genai.ProviderOptionModalities:
			modalities = genai.Modalities(v)
		case genai.ProviderOptionPreloadedModels:
			preloadedModels = []genai.Model(v)
		case genai.ProviderOptionTransportWrapper:
			wrapper = v
		default:
			return nil, fmt.Errorf("unsupported option type %T", opt)
		}
	}
	const apiKeyURL = "https://console.anthropic.com/settings/keys"
	var err error
	if apiKey == "" {
		if apiKey = os.Getenv("ANTHROPIC_API_KEY"); apiKey == "" {
			err = &base.ErrAPIKeyRequired{EnvVar: "ANTHROPIC_API_KEY", URL: apiKeyURL}
		}
	}
	mod := genai.Modalities{genai.ModalityText}
	if len(modalities) != 0 && !slices.Equal(modalities, mod) {
		return nil, fmt.Errorf("unexpected option Modalities %s, only text is supported", mod)
	}
	t := base.DefaultTransport
	if wrapper != nil {
		t = wrapper(t)
	}
	// Anthropic allows Opaque fields for thinking signatures
	c := &Client{
		impl: base.Provider[*ErrorResponse, *ChatRequest, *ChatResponse, ChatStreamChunkResponse]{
			GenSyncURL:      "https://api.anthropic.com/v1/messages",
			ProcessStream:   ProcessStream,
			PreloadedModels: preloadedModels,
			ProcessHeaders:  processHeaders,
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
		switch model {
		case "":
		case genai.ModelCheap, genai.ModelGood, genai.ModelSOTA:
			if c.impl.Model, err = c.selectBestTextModel(ctx, model); err != nil {
				return nil, err
			}
			c.impl.OutputModalities = mod
		default:
			c.impl.Model = model
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
func (c *Client) GenAsync(ctx context.Context, msgs genai.Messages, opts ...genai.GenOptions) (genai.Job, error) {
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

// Scoreboard implements genai.Provider.
func (c *Client) Scoreboard() scoreboard.Score {
	return Scoreboard()
}

// HTTPClient returns the HTTP client to fetch results (e.g. videos) generated by the provider.
func (c *Client) HTTPClient() *http.Client {
	return &c.impl.Client
}

// GenSync implements genai.Provider.
func (c *Client) GenSync(ctx context.Context, msgs genai.Messages, opts ...genai.GenOptions) (genai.Result, error) {
	return c.impl.GenSync(ctx, msgs, opts...)
}

// GenSyncRaw provides access to the raw API.
func (c *Client) GenSyncRaw(ctx context.Context, in *ChatRequest, out *ChatResponse) error {
	return c.impl.GenSyncRaw(ctx, in, out)
}

// GenStream implements genai.Provider.
func (c *Client) GenStream(ctx context.Context, msgs genai.Messages, opts ...genai.GenOptions) (iter.Seq[genai.Reply], func() (genai.Result, error)) {
	return c.impl.GenStream(ctx, msgs, opts...)
}

// GenStreamRaw provides access to the raw API.
func (c *Client) GenStreamRaw(ctx context.Context, in *ChatRequest) (iter.Seq[ChatStreamChunkResponse], func() error) {
	return c.impl.GenStreamRaw(ctx, in)
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

// ProcessStream converts the raw packets from the streaming API into Reply fragments.
func ProcessStream(chunks iter.Seq[ChatStreamChunkResponse]) (iter.Seq[genai.Reply], func() (genai.Usage, [][]genai.Logprob, error)) {
	var finalErr error
	var u genai.Usage

	return func(yield func(genai.Reply) bool) {
			// At the moment, only supported for server_tool_use / web_search.
			pendingServerCall := ""
			pendingJSON := ""
			pendingToolCall := genai.ToolCall{}
			for pkt := range chunks {
				f := genai.Reply{}
				// See testdata/TestClient_Chat_thinking/ChatStream.yaml as a great example.
				// TODO: pkt.Index matters here, as the LLM may fill multiple content blocks simultaneously.
				switch pkt.Type {
				case ChunkMessageStart:
					switch pkt.Message.Role {
					case "assistant":
					default:
						finalErr = &internal.BadError{Err: fmt.Errorf("unexpected role %q", pkt.Message.Role)}
						return
					}
					u.InputTokens = pkt.Message.Usage.InputTokens
					u.InputCachedTokens = pkt.Message.Usage.CacheReadInputTokens
					// There's some tokens listed there. Still save it in case it breaks midway.
					u.OutputTokens = pkt.Message.Usage.OutputTokens
					u.TotalTokens = u.InputTokens + u.InputCachedTokens + u.OutputTokens
					continue
				case ChunkContentBlockStart:
					switch pkt.ContentBlock.Type {
					case ContentText:
						f.Text = pkt.ContentBlock.Text
					case ContentThinking:
						f.Reasoning = pkt.ContentBlock.Thinking
					case ContentToolUse:
						pendingToolCall.ID = pkt.ContentBlock.ID
						pendingToolCall.Name = pkt.ContentBlock.Name
						pendingToolCall.Arguments = ""
						// TODO: Is there anything to do with Input? pendingCall.Arguments = pkt.ContentBlock.Input
					case ContentRedactedThinking:
						f.Opaque = map[string]any{"redacted_thinking": pkt.ContentBlock.Signature}
					case ContentServerToolUse:
						// Discard the data for now. It may be necessary in the future to keep in in Opaque.
						pendingServerCall = pkt.ContentBlock.Name
						switch pendingServerCall {
						case "web_search":
							// This is great!
						default:
							// Oops, more work to do!
							if !internal.BeLenient {
								finalErr = &internal.BadError{Err: fmt.Errorf("implement server tool call %q", pendingServerCall)}
								return
							}
						}
					case ContentWebSearchToolResult:
						f.Citation.Sources = make([]genai.CitationSource, len(pkt.ContentBlock.Content))
						for i, cc := range pkt.ContentBlock.Content {
							if cc.Type != ContentWebSearchResult {
								finalErr = &internal.BadError{Err: fmt.Errorf("implement content type %q while processing %q", cc.Type, pkt.ContentBlock.Type)}
								return
							}
							f.Citation.Sources[i].Type = genai.CitationWeb
							f.Citation.Sources[i].URL = cc.URL
							f.Citation.Sources[i].Title = cc.Title
							f.Citation.Sources[i].Date = cc.PageAge
							// EncryptedContent is not really useful?
						}
					case ContentMCPToolUse:
						f.Opaque = map[string]any{"mcp_tool_use": map[string]any{
							"id":    pkt.ContentBlock.ID,
							"input": pkt.ContentBlock.Input,
							"name":  pkt.ContentBlock.Name,
						}}
					case ContentMCPToolResult:
						f.Opaque = map[string]any{"mcp_tool_result": map[string]any{
							"tool_use_id": pkt.ContentBlock.ToolUseID,
							//"is_error":    pkt.ContentBlock.IsError,
							"content":     pkt.ContentBlock.Content,
							"server_name": pkt.ContentBlock.ServerName,
						}}
					case ContentWebSearchResult, ContentImage, ContentDocument, ContentToolResult:
						fallthrough
					default:
						finalErr = &internal.BadError{Err: fmt.Errorf("implement content block %q", pkt.ContentBlock.Type)}
						return
					}
				case ChunkContentBlockDelta:
					switch pkt.Delta.Type {
					case DeltaText:
						f.Text = pkt.Delta.Text
					case DeltaThinking:
						f.Reasoning = pkt.Delta.Thinking
					case DeltaSignature:
						f.Opaque = map[string]any{"signature": pkt.Delta.Signature}
					case DeltaInputJSON:
						pendingJSON += pkt.Delta.PartialJSON
					case DeltaCitations:
						if err := pkt.Delta.Citation.To(&f.Citation); err != nil {
							finalErr = &internal.BadError{Err: fmt.Errorf("failed to parse citation: %w", err)}
							return
						}
					default:
						finalErr = &internal.BadError{Err: fmt.Errorf("implement content block delta %q", pkt.Delta.Type)}
						return
					}
				case ChunkContentBlockStop:
					// Marks a closure of the block pkt.Index. Flush accumulated JSON if appropriate.
					if pendingToolCall.ID != "" {
						pendingToolCall.Arguments = pendingJSON
						f.ToolCall = pendingToolCall
						pendingToolCall = genai.ToolCall{}
					}
					// Why not web_search_20250305 ??
					switch pendingServerCall {
					case "web_search":
						q := WebSearch{}
						d := json.NewDecoder(strings.NewReader(pendingJSON))
						if !internal.BeLenient {
							d.DisallowUnknownFields()
						}
						if err := d.Decode(&q); err != nil {
							finalErr = &internal.BadError{Err: fmt.Errorf("failed to decode pending server tool call %s: %w", pendingServerCall, err)}
							return
						}
						f.Citation.Sources = []genai.CitationSource{{
							Type:    genai.CitationWebQuery,
							Snippet: q.Query,
						}}
						pendingServerCall = ""
					case "":
					default:
						// Oops, more work to do!
						if !internal.BeLenient {
							finalErr = &internal.BadError{Err: fmt.Errorf("implement server tool call %q", pendingServerCall)}
							return
						}
					}
					pendingJSON = ""
				case ChunkMessageDelta:
					// Includes finish reason and output tokens usage (but not input tokens!)
					u.FinishReason = pkt.Delta.StopReason.ToFinishReason()
					u.OutputTokens = pkt.Usage.OutputTokens
				case ChunkMessageStop:
					// Doesn't contain anything.
					continue
				case ChunkPing:
					// Doesn't contain anything.
					continue
				case ChunkError:
					// TODO: See it in the field to decode properly.
					finalErr = fmt.Errorf("got error: %+v", pkt)
					return
				default:
					finalErr = &internal.BadError{Err: fmt.Errorf("implement stream block %q", pkt.Type)}
					return
				}
				if !yield(f) {
					return
				}
			}
		}, func() (genai.Usage, [][]genai.Logprob, error) {
			return u, nil, finalErr
		}
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

func (c *Client) Capabilities() genai.ProviderCapabilities {
	return genai.ProviderCapabilities{
		GenAsync: true,
	}
}

var (
	_ internal.Validatable = &Message{}
	_ internal.Validatable = &Content{}
	_ genai.Provider       = &Client{}
)
