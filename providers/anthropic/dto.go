// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Wire types for the Anthropic Messages API.
//
// Type names match the upstream API documentation:
//
//	https://docs.anthropic.com/en/api/messages
//	https://docs.anthropic.com/en/api/messages-streaming
//	https://docs.anthropic.com/en/api/creating-message-batches
//	https://docs.anthropic.com/en/api/files
//	https://docs.anthropic.com/en/api/models-list

package anthropic

import (
	"bytes"
	_ "embed"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"reflect"
	"strconv"
	"strings"
	"time"

	"github.com/invopop/jsonschema"

	"github.com/maruel/genai"
	"github.com/maruel/genai/base"
	"github.com/maruel/genai/internal"
)

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
	InferenceGeo  string          `json:"inference_geo,omitzero"`
	ServiceTier   string          `json:"service_tier,omitzero"` // "auto", "standard_only"
	StopSequences []string        `json:"stop_sequences,omitzero"`
	Stream        bool            `json:"stream,omitzero"`
	System        []SystemMessage `json:"system,omitzero"`      // Must be type "text"
	Temperature   float64         `json:"temperature,omitzero"` // [0, 1]
	Thinking      Thinking        `json:"thinking,omitzero"`
	ToolChoice    ToolChoice      `json:"tool_choice,omitzero"`
	Tools         []Tool          `json:"tools,omitzero"`
	TopK          int64           `json:"top_k,omitzero"` // [1, ]
	TopP          float64         `json:"top_p,omitzero"` // [0, 1]
	OutputConfig  OutputConfig    `json:"output_config,omitzero"`
}

// OutputConfig is documented at https://docs.anthropic.com/en/api/messages#body-output-config
type OutputConfig struct {
	Effort Effort       `json:"effort,omitzero"`
	Format OutputFormat `json:"format,omitzero"`
}

// OutputFormat is documented at https://docs.anthropic.com/en/api/messages#body-output-config-format
type OutputFormat struct {
	Type   string             `json:"type,omitzero"`
	Schema *jsonschema.Schema `json:"schema,omitzero"`
}

// Init initializes the provider specific completion request with the generic completion request.
func (c *ChatRequest) Init(msgs genai.Messages, model string, opts ...genai.GenOption) error {
	return c.initImpl(msgs, model, true, opts...)
}

func (c *ChatRequest) initImpl(msgs genai.Messages, model string, cache bool, opts ...genai.GenOption) error {
	c.Model = model
	if err := msgs.Validate(); err != nil {
		return err
	}
	var errs []error
	var unsupported []string
	msgToCache := 0
	// Default thinking: adaptive for models that support it, disabled otherwise.
	if _, ok := modelsAdaptiveBlocked.Load(model); ok {
		c.Thinking.Type = "disabled"
	} else {
		c.Thinking.Type = "adaptive"
		c.Thinking.Display = "summarized"
	}
	// Anthropic requires a non-zero max_tokens.
	if v, ok := modelsMaxTokens.Load(model); ok {
		if i, ok := v.(int); ok {
			c.MaxTokens = int64(i)
		} else {
			return fmt.Errorf("internal error: invalid cached MaxTokens %v", v)
		}
	}
	for _, opt := range opts {
		if err := opt.Validate(); err != nil {
			return err
		}
		switch v := opt.(type) {
		case *GenOptionText:
			if cache {
				msgToCache = v.MessagesToCache
			} else if v.MessagesToCache != 0 {
				unsupported = append(unsupported, "GenOptionText.MessagesToCache")
			}
			c.OutputConfig.Effort = v.Effort
			c.InferenceGeo = v.InferenceGeo
			switch v.Thinking {
			case ThinkingAdaptive:
				c.Thinking.Type = string(ThinkingAdaptive)
				// Default to summarized so reasoning text is returned.
				// opus-4-8/4-7 default to "omitted" which hides thinking text.
				c.Thinking.Display = "summarized"
			case ThinkingDisabled:
				c.Thinking.Type = "disabled"
				c.Thinking.Display = ""
			case ThinkingEnabled:
				if v.ThinkingBudget >= c.MaxTokens {
					errs = append(errs, fmt.Errorf("invalid ThinkingBudget(%d) >= MaxTokens(%d)", v.ThinkingBudget, c.MaxTokens))
				}
				c.Thinking.BudgetTokens = v.ThinkingBudget
				c.Thinking.Type = string(ThinkingEnabled)
			default:
				// Auto-detect from ThinkingBudget for backward compatibility.
				if v.ThinkingBudget > 0 {
					if v.ThinkingBudget >= c.MaxTokens {
						errs = append(errs, fmt.Errorf("invalid ThinkingBudget(%d) >= MaxTokens(%d)", v.ThinkingBudget, c.MaxTokens))
					}
					// https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking
					// Thinking isn't compatible with temperature, top_p, or top_k modifications as well as forced tool use.
					c.Thinking.BudgetTokens = v.ThinkingBudget
					c.Thinking.Type = string(ThinkingEnabled)
				}
			}
			if v.ThinkingDisplay != "" {
				c.Thinking.Display = v.ThinkingDisplay
			}
		case *genai.GenOptionText:
			unsupported = append(unsupported, c.initOptionsText(v)...)
		case *genai.GenOptionTools:
			c.initOptionsTools(v)
		case *genai.GenOptionWeb:
			c.initOptionsWeb(v)
		default:
			unsupported = append(unsupported, internal.TypeName(opt))
		}
	}

	// Post process to take into account limitations by the provider.
	// Forced tool use is incompatible with thinking.
	// https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking
	if c.Thinking.Type != "disabled" && c.ToolChoice.Type == ToolChoiceAny {
		unsupported = append(unsupported, "GenOptionTools.Force")
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

// SetStream sets the streaming mode.
func (c *ChatRequest) SetStream(stream bool) {
	c.Stream = stream
}

func (c *ChatRequest) initOptionsText(v *genai.GenOptionText) []string {
	var unsupported []string
	if v.TopLogprobs > 0 {
		unsupported = append(unsupported, "GenOptionText.TopLogprobs")
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
	if v.DecodeAs != nil {
		c.OutputConfig.Format = OutputFormat{
			Type:   "json_schema",
			Schema: internal.JSONSchemaFor(reflect.TypeOf(v.DecodeAs)),
		}
	} else if v.ReplyAsJSON {
		c.OutputConfig.Format = OutputFormat{
			Type:   "json_schema",
			Schema: &jsonschema.Schema{Type: "object"},
		}
	}
	return unsupported
}

func (c *ChatRequest) initOptionsTools(v *genai.GenOptionTools) {
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
}

func (c *ChatRequest) initOptionsWeb(v *genai.GenOptionWeb) {
	if v.Search {
		// https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/web-search-tool
		c.Tools = append(c.Tools, Tool{
			Type: "web_search_20250305",
			Name: "web_search",
			// MaxUses: 10,
		})
	}
	if v.Fetch {
		// https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/web-fetch-tool
		c.Tools = append(c.Tools, Tool{
			Type: "web_fetch_20250910",
			Name: "web_fetch",
		})
	}
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

// ToolConfiguration configures tool calling behavior.
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

// Validate implements genai.Validatable.
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

// From converts from the genai equivalent.
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

// To converts to the genai equivalent.
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

// Contents is a []Content with custom JSON unmarshaling that accepts both a single object and an array.
//
// The web_fetch_tool_result API returns "content": {...} (single object) unlike web_search_tool_result which
// returns "content": [...] (array).
type Contents []Content

// UnmarshalJSON handles both single object and array JSON content.
func (c *Contents) UnmarshalJSON(data []byte) error {
	data = bytes.TrimSpace(data)
	if len(data) == 0 {
		return nil
	}
	if data[0] == '[' {
		return json.Unmarshal(data, (*[]Content)(c))
	}
	var single Content
	if err := json.Unmarshal(data, &single); err != nil {
		return err
	}
	*c = []Content{single}
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
	// Type == ContentToolUse; indicates what invoked the tool call.
	Caller Caller `json:"caller,omitzero"`

	// Type == ContentToolResult, ContentMCPToolResult
	ToolUseID string `json:"tool_use_id,omitzero"`
	IsError   bool   `json:"is_error,omitzero"`

	// Type == ContentToolResult, ContentWebSearchToolResult, ContentWebFetchToolResult, ContentMCPToolResult
	// - ContentToolResult: Only ContentText and ContentImage are allowed.
	// - ContentWebSearchToolResult: Only ContentWebSearchResult is allowed.
	// - ContentWebFetchToolResult: Only ContentWebFetchResult and ContentWebFetchToolError are allowed.
	Content Contents `json:"content,omitzero"`

	// Type == ContentMCPToolUse
	ServerName string `json:"server_name,omitzero"`

	// Type == ContentDocument
	Context string `json:"context,omitzero"` // Context about the document that will not be cited from

	// Type == ContentWebSearchResult
	URL              string `json:"url,omitzero"`
	EncryptedContent string `json:"encrypted_content,omitzero"`
	PageAge          string `json:"page_age,omitzero"` // "12 hours ago", "4 days ago", "1 week ago", "April 29, 2025", null

	// Type == ContentWebFetchResult
	RetrievedAt string `json:"retrieved_at,omitzero"`

	// Type == ContentWebFetchToolError
	ErrorCode string `json:"error_code,omitzero"`

	// Type == ContentDocument, ContentWebSearchResult, ContentWebFetchResult
	Title string `json:"title,omitzero"` // Document title when using Source, web page title
}

// Caller indicates what invoked a tool call.
type Caller struct {
	// Type is "direct" or "code_execution_20250825" or "code_execution_20260120".
	Type string `json:"type"`
	// ToolID is set when the tool was invoked by another tool.
	ToolID string `json:"tool_id,omitzero"`
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
			c.ToolUseID != "" || c.IsError || len(c.Content) > 0 || c.ServerName != "" || c.Context != "" || c.URL != "" || c.EncryptedContent != "" || c.PageAge != "" || c.Title != "" ||
			c.RetrievedAt != "" || c.ErrorCode != "" {
			return &internal.BadError{Err: fmt.Errorf("%s: unexpected fields set", c.Type)}
		}
	case ContentThinking:
		if c.Thinking == "" {
			// c.Signature is optional.
			return &internal.BadError{Err: fmt.Errorf("%s: fields Thinking must be set", c.Type)}
		}
		if c.Text != "" || c.Data != "" || len(c.Citations.Citations) > 0 || !c.Source.IsZero() || c.ID != "" || c.Name != "" || c.Input != nil ||
			c.ToolUseID != "" || c.IsError || len(c.Content) > 0 || c.ServerName != "" || c.Context != "" || c.URL != "" || c.EncryptedContent != "" || c.PageAge != "" || c.Title != "" ||
			c.RetrievedAt != "" || c.ErrorCode != "" {
			return &internal.BadError{Err: fmt.Errorf("%s: unexpected fields set", c.Type)}
		}
	case ContentRedactedThinking:
		if c.Data == "" {
			return &internal.BadError{Err: fmt.Errorf("%s: fields Data must be set", c.Type)}
		}
		if c.Text != "" || c.Thinking != "" || len(c.Signature) > 0 || len(c.Citations.Citations) > 0 || !c.Source.IsZero() || c.ID != "" || c.Name != "" || c.Input != nil ||
			c.ToolUseID != "" || c.IsError || len(c.Content) > 0 || c.ServerName != "" || c.Context != "" || c.URL != "" || c.EncryptedContent != "" || c.PageAge != "" || c.Title != "" ||
			c.RetrievedAt != "" || c.ErrorCode != "" {
			return &internal.BadError{Err: fmt.Errorf("%s: unexpected fields set", c.Type)}
		}
	case ContentImage, ContentDocument:
		if c.Source.Type == "" && c.Source.URL == "" && c.Source.Data == "" {
			return &internal.BadError{Err: fmt.Errorf("%s: fields Source must be set", c.Type)}
		}
		if c.Text != "" || c.Thinking != "" || len(c.Signature) > 0 || c.Data != "" || len(c.Citations.Citations) > 0 || c.ID != "" || c.Name != "" || c.Input != nil ||
			c.ToolUseID != "" || c.IsError || len(c.Content) > 0 || c.ServerName != "" || c.Context != "" || c.URL != "" || c.EncryptedContent != "" || c.PageAge != "" || c.Title != "" ||
			c.RetrievedAt != "" || c.ErrorCode != "" {
			return &internal.BadError{Err: fmt.Errorf("%s: unexpected fields set", c.Type)}
		}
	case ContentToolUse:
		if c.ID == "" || c.Name == "" || c.Input == nil {
			return &internal.BadError{Err: fmt.Errorf("%s: fields ID, Name, Input must be set", c.Type)}
		}
		if c.Text != "" || c.Thinking != "" || len(c.Signature) > 0 || c.Data != "" || len(c.Citations.Citations) > 0 || !c.Source.IsZero() ||
			c.ToolUseID != "" || c.IsError || len(c.Content) > 0 || c.ServerName != "" || c.Context != "" || c.URL != "" || c.EncryptedContent != "" || c.PageAge != "" || c.Title != "" ||
			c.RetrievedAt != "" || c.ErrorCode != "" {
			return &internal.BadError{Err: fmt.Errorf("%s: unexpected fields set", c.Type)}
		}
	case ContentToolResult:
		if c.ToolUseID == "" || len(c.Content) == 0 {
			return &internal.BadError{Err: fmt.Errorf("%s: fields ToolUseID, Content, must be set", c.Type)}
		}
		if c.Text != "" || c.Thinking != "" || len(c.Signature) > 0 || c.Data != "" || len(c.Citations.Citations) > 0 || !c.Source.IsZero() || c.ID != "" || c.Name != "" || c.Input != nil ||
			c.IsError || c.ServerName != "" || c.Context != "" || c.URL != "" || c.EncryptedContent != "" || c.PageAge != "" || c.Title != "" ||
			c.RetrievedAt != "" || c.ErrorCode != "" {
			return &internal.BadError{Err: fmt.Errorf("%s: unexpected fields set", c.Type)}
		}
	case ContentMCPToolUse:
		if c.ID == "" || c.Name == "" || c.Input == nil || c.ServerName == "" {
			return &internal.BadError{Err: fmt.Errorf("%s: fields ID, Name, Input, ServerName must be set", c.Type)}
		}
		if c.Text != "" || c.Thinking != "" || len(c.Signature) > 0 || c.Data != "" || len(c.Citations.Citations) > 0 || !c.Source.IsZero() ||
			c.ToolUseID != "" || c.IsError || len(c.Content) > 0 || c.Context != "" || c.URL != "" || c.EncryptedContent != "" || c.PageAge != "" || c.Title != "" ||
			c.RetrievedAt != "" || c.ErrorCode != "" {
			return &internal.BadError{Err: fmt.Errorf("%s: unexpected fields set", c.Type)}
		}
	case ContentMCPToolResult:
		if c.ToolUseID == "" || len(c.Content) == 0 {
			return &internal.BadError{Err: fmt.Errorf("%s: fields ToolUseID, Content must be set", c.Type)}
		}
		if c.Text != "" || c.Thinking != "" || len(c.Signature) > 0 || c.Data != "" || len(c.Citations.Citations) > 0 || !c.Source.IsZero() || c.ID != "" || c.Name != "" || c.Input != nil ||
			c.ServerName != "" || c.Context != "" || c.URL != "" || c.EncryptedContent != "" || c.PageAge != "" || c.Title != "" ||
			c.RetrievedAt != "" || c.ErrorCode != "" {
			return &internal.BadError{Err: fmt.Errorf("%s: unexpected fields set", c.Type)}
		}
	case ContentServerToolUse:
		if c.ID == "" || c.Name == "" || c.Input == nil {
			return &internal.BadError{Err: fmt.Errorf("%s: fields ID, Name, Input must be set", c.Type)}
		}
		if c.Text != "" || c.Thinking != "" || len(c.Signature) > 0 || c.Data != "" || len(c.Citations.Citations) > 0 || !c.Source.IsZero() ||
			c.ToolUseID != "" || c.IsError || len(c.Content) > 0 || c.ServerName != "" || c.Context != "" || c.URL != "" || c.EncryptedContent != "" || c.PageAge != "" || c.Title != "" ||
			c.RetrievedAt != "" || c.ErrorCode != "" {
			return &internal.BadError{Err: fmt.Errorf("%s: unexpected fields set", c.Type)}
		}
	case ContentWebSearchToolResult:
		if c.ToolUseID == "" || len(c.Content) == 0 {
			return &internal.BadError{Err: fmt.Errorf("%s: fields ToolUseID, Content must be set", c.Type)}
		}
		if c.Text != "" || c.Thinking != "" || len(c.Signature) > 0 || c.Data != "" || len(c.Citations.Citations) > 0 || !c.Source.IsZero() || c.ID != "" || c.Name != "" || c.Input != nil ||
			c.ServerName != "" || c.Context != "" || c.URL != "" || c.EncryptedContent != "" || c.PageAge != "" || c.Title != "" ||
			c.RetrievedAt != "" || c.ErrorCode != "" {
			return &internal.BadError{Err: fmt.Errorf("%s: unexpected fields set", c.Type)}
		}
	case ContentWebSearchResult:
		if c.URL == "" {
			return &internal.BadError{Err: fmt.Errorf("%s: fields URL must be set", c.Type)}
		}
		if c.Text != "" || c.Thinking != "" || len(c.Signature) > 0 || c.Data != "" || len(c.Citations.Citations) > 0 || !c.Source.IsZero() || c.ID != "" || c.Name != "" || c.Input != nil ||
			c.ToolUseID != "" || c.IsError || len(c.Content) > 0 || c.ServerName != "" || c.Context != "" || c.EncryptedContent != "" || c.PageAge != "" || c.Title != "" ||
			c.RetrievedAt != "" || c.ErrorCode != "" {
			return &internal.BadError{Err: fmt.Errorf("%s: unexpected fields set", c.Type)}
		}
	case ContentWebFetchToolResult:
		if c.ToolUseID == "" || len(c.Content) == 0 {
			return &internal.BadError{Err: fmt.Errorf("%s: fields ToolUseID, Content must be set", c.Type)}
		}
		if c.Text != "" || c.Thinking != "" || len(c.Signature) > 0 || c.Data != "" || len(c.Citations.Citations) > 0 || !c.Source.IsZero() || c.ID != "" || c.Name != "" || c.Input != nil ||
			c.ServerName != "" || c.Context != "" || c.URL != "" || c.EncryptedContent != "" || c.PageAge != "" || c.Title != "" ||
			c.RetrievedAt != "" || c.ErrorCode != "" {
			return &internal.BadError{Err: fmt.Errorf("%s: unexpected fields set", c.Type)}
		}
	case ContentWebFetchResult:
		if c.URL == "" {
			return &internal.BadError{Err: fmt.Errorf("%s: fields URL must be set", c.Type)}
		}
		// web_fetch_result has: url, content (nested document), retrieved_at, title (optional).
		if c.Thinking != "" || len(c.Signature) > 0 || c.Data != "" || len(c.Citations.Citations) > 0 || !c.Source.IsZero() || c.ID != "" || c.Name != "" || c.Input != nil ||
			c.ToolUseID != "" || c.IsError || c.ServerName != "" || c.Context != "" || c.EncryptedContent != "" || c.PageAge != "" ||
			c.ErrorCode != "" {
			return &internal.BadError{Err: fmt.Errorf("%s: unexpected fields set", c.Type)}
		}
	case ContentWebFetchToolError:
		if c.ErrorCode == "" {
			return &internal.BadError{Err: fmt.Errorf("%s: fields ErrorCode must be set", c.Type)}
		}
		if c.Text != "" || c.Thinking != "" || len(c.Signature) > 0 || c.Data != "" || len(c.Citations.Citations) > 0 || !c.Source.IsZero() || c.ID != "" || c.Name != "" || c.Input != nil ||
			c.ToolUseID != "" || c.IsError || len(c.Content) > 0 || c.ServerName != "" || c.Context != "" || c.EncryptedContent != "" || c.PageAge != "" || c.Title != "" ||
			c.RetrievedAt != "" || c.URL != "" {
			return &internal.BadError{Err: fmt.Errorf("%s: unexpected fields set", c.Type)}
		}
	default:
		return &internal.BadError{Err: fmt.Errorf("implement ContentType %q", c.Type)}
	}
	return nil
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

// FromReply converts from a genai reply.
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

// FromToolCallResult converts from a genai tool call result.
func (c *Content) FromToolCallResult(in *genai.ToolCallResult) {
	// TODO: Support text citation.
	// TODO: Support image.
	c.Type = ContentToolResult
	c.ToolUseID = in.ID
	c.IsError = false
	c.Content = []Content{{Type: ContentText, Text: in.Result}}
}

// To converts to the genai equivalent.
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
		for i := range c.Content {
			cc := &c.Content[i]
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
		case "web_fetch":
			q := WebFetch{}
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
				Citation: genai.Citation{Sources: []genai.CitationSource{{Type: genai.CitationWeb, URL: q.URL}}},
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
	case ContentWebFetchToolResult:
		for i := range c.Content {
			cc := &c.Content[i]
			switch cc.Type {
			case ContentWebFetchResult:
				title := cc.Title
				if title == "" && len(cc.Content) > 0 {
					title = cc.Content[0].Title
				}
				out = append(out, genai.Reply{
					Citation: genai.Citation{Sources: []genai.CitationSource{{
						Type:  genai.CitationWeb,
						URL:   cc.URL,
						Title: title,
					}}},
				})
			case ContentWebFetchToolError:
				out = append(out, genai.Reply{
					Opaque: map[string]any{"web_fetch_error": cc.ErrorCode},
				})
			default:
				return out, &internal.BadError{Err: fmt.Errorf("implement content type %q while processing %q", cc.Type, c.Type)}
			}
		}
	case ContentWebSearchResult, ContentWebFetchResult, ContentWebFetchToolError, ContentImage, ContentDocument, ContentToolResult:
		return out, &internal.BadError{Err: fmt.Errorf("implement content type %q", c.Type)}
	default:
		return out, &internal.BadError{Err: fmt.Errorf("implement content type %q", c.Type)}
	}
	return out, nil
}

// WebSearch is the server tool use input for web_search.
type WebSearch struct {
	Query string `json:"query"`
}

// WebFetch is the server tool use input for web_fetch.
type WebFetch struct {
	URL string `json:"url"`
}

// SourceType is described at https://docs.anthropic.com/en/api/messages#body-messages-content-source
type SourceType string

// Source type values.
const (
	SourceBase64  SourceType = "base64"
	SourceURL     SourceType = "url"
	SourceText    SourceType = "text"
	SourceContent SourceType = "content"
	SourceFileID  SourceType = "file"
)

// Source is a provider-specific content source.
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

// IsZero reports whether the value is zero.
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

// ContentType is a provider-specific content type.
type ContentType string

// Content type values.
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
	ContentWebFetchResult      ContentType = "web_fetch_result"
	ContentWebFetchToolResult  ContentType = "web_fetch_tool_result"
	ContentWebFetchToolError   ContentType = "web_fetch_tool_error"
)

// CitationType is a provider-specific citation type.
type CitationType string

// Citation type values.
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

// To converts to the genai equivalent.
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
		// To confirm.
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

// Thinking is a provider-specific thinking block.
type Thinking struct {
	BudgetTokens int64  `json:"budget_tokens,omitzero"` // >1024 and less than max_tokens; unused for adaptive
	Display      string `json:"display,omitzero"`       // "summarized", "omitted"; omitted is default on opus-4-8/4-7
	Type         string `json:"type,omitzero"`          // "enabled", "disabled", "adaptive"
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

// ToolChoice configures tool selection behavior.
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
	Type string `json:"type,omitzero"` // "custom", "computer_20241022", "computer_20250124", "bash_20241022", "bash_20250124", "text_editor_20241022", "text_editor_20250124", "text_editor_20250429", "text_editor_20250728", "web_search_20250305", "web_fetch_20250910"
	// Type == "custom"
	Description string             `json:"description,omitzero"`
	InputSchema *jsonschema.Schema `json:"input_schema,omitzero"`

	// Type == "custom": tool name
	// Type == "computer_20241022", "computer_20250124": "computer"
	// Type == "bash_20241022", "bash_20250124": "bash"
	// Type == "text_editor_20241022", "text_editor_20250124", "text_editor_20250429", "text_editor_20250728": "str_replace_editor"
	Name string `json:"name,omitzero"`

	// Type == "custom", "computer_20241022", "computer_20250124", "bash_20241022", "bash_20250124", "text_editor_20241022", "text_editor_20250124", "text_editor_20250429", "text_editor_20250728"
	CacheControl struct {
		Type string `json:"type,omitzero"` // "ephemeral"
	} `json:"cache_control,omitzero"`

	// Type == "text_editor_20250728"
	MaxCharacters int64 `json:"max_characters,omitzero"`

	// Type == "computer_20241022", "computer_20250124"
	DisplayNumber   int64 `json:"display_number,omitzero"`
	DisplayHeightPX int64 `json:"display_height_px,omitzero"`
	DisplayWidthPX  int64 `json:"display_width_px,omitzero"`

	// Type == "web_search_20250305", "web_fetch_20250910"
	AllowedDomains []string `json:"allowed_domains,omitzero"`
	BlockedDomains []string `json:"blocked_domains,omitzero"`
	UserLocation   struct {
		Type     string `json:"type,omitzero"`     // "approximate"
		City     string `json:"city,omitzero"`     // "San Francisco"
		Region   string `json:"region,omitzero"`   // "California"
		Country  string `json:"country,omitzero"`  // "US"
		Timezone string `json:"timezone,omitzero"` // "America/Los_Angeles"; https://en.wikipedia.org/wiki/List_of_tz_database_time_zones
	} `json:"user_location,omitzero"`

	// Type == "web_fetch_20250910"
	MaxContentTokens int64 `json:"max_content_tokens,omitzero"` // Max tokens of fetched content to return. Default is 10000.
	MaxUses          int64 `json:"max_uses,omitzero"`           // Max number of fetches per request. Default is 5.
}

// ChatResponse is the provider-specific chat completion response.
type ChatResponse struct {
	Message                         // Role is always "assistant"
	ID           string             `json:"id"`
	Model        string             `json:"model"`
	StopReason   StopReason         `json:"stop_reason"`
	StopSequence string             `json:"stop_sequence"`
	StopDetails  RefusalStopDetails `json:"stop_details"`
	Type         string             `json:"type"` // "message"
	Usage        Usage              `json:"usage"`
	Container    struct {
		ExpiresAt time.Time `json:"expires_at"`
		ID        string    `json:"id"`
	} `json:"container"`
}

// ToResult converts the response to a genai.Result.
func (c *ChatResponse) ToResult() (genai.Result, error) {
	out := genai.Result{
		Usage: genai.Usage{
			InputTokens:       c.Usage.InputTokens,
			InputCachedTokens: c.Usage.CacheReadInputTokens,
			OutputTokens:      c.Usage.OutputTokens,
			FinishReason:      c.StopReason.ToFinishReason(),
			ServiceTier:       c.Usage.ServiceTier,
		},
	}
	err := c.To(&out.Message)
	return out, err
}

// StopReason is documented at https://docs.anthropic.com/en/api/messages#response-stop-reason
type StopReason string

// Stop reason values.
const (
	StopEndTurn   StopReason = "end_turn"
	StopToolUse   StopReason = "tool_use"
	StopSequence  StopReason = "stop_sequence"
	StopMaxTokens StopReason = "max_tokens"
	StopPauseTurn StopReason = "pause_turn" //  We paused a long-running turn. You may provide the response back as-is in a subsequent request to let the model continue.
	StopRefusal   StopReason = "refusal"
)

// ToFinishReason converts to a genai.FinishReason.
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

// RefusalStopDetails is documented at https://docs.anthropic.com/en/docs/build-with-claude/handling-stop-reasons
type RefusalStopDetails struct {
	Type        string `json:"type"`        // Always "refusal"
	Category    string `json:"category"`    // "cyber", "bio", or null
	Explanation string `json:"explanation"` // Human-readable or null
}

// UnmarshalJSON implements json.Unmarshaler to handle null stop_details.
func (r *RefusalStopDetails) UnmarshalJSON(b []byte) error {
	if string(b) == "null" {
		return nil
	}
	type alias RefusalStopDetails
	return json.Unmarshal(b, (*alias)(r))
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
		ID           string             `json:"id"`
		Type         string             `json:"type"` // "message", "thinking"
		Role         string             `json:"role"`
		Model        string             `json:"model"`
		Content      []string           `json:"content"`
		StopReason   StopReason         `json:"stop_reason"`
		StopSequence string             `json:"stop_sequence"`
		StopDetails  RefusalStopDetails `json:"stop_details"`
		Usage        Usage              `json:"usage"`
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
		ID     string `json:"id"`
		Name   string `json:"name"`
		Input  any    `json:"input"`
		Caller Caller `json:"caller"`

		// Always empty on content_block_start; actual citations arrive as citations_delta in subsequent deltas.
		Citations Citations `json:"citations"`

		// Type == ContentWebSearchToolResult, ContentWebFetchToolResult, ContentMCPToolResult
		ToolUseID string   `json:"tool_use_id"`
		Content   Contents `json:"content"`

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
		StopReason   StopReason         `json:"stop_reason"`
		StopSequence string             `json:"stop_sequence"`
		StopDetails  RefusalStopDetails `json:"stop_details"`
	} `json:"delta"`

	Usage Usage `json:"usage"`
}

// ChunkType is the type of a streaming chunk.
type ChunkType string

// Chunk type values.
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

// DeltaType is the type of a streaming delta.
type DeltaType string

// Delta type values.
const (
	DeltaText      DeltaType = "text_delta"
	DeltaInputJSON DeltaType = "input_json_delta"
	DeltaThinking  DeltaType = "thinking_delta"
	DeltaSignature DeltaType = "signature_delta"
	DeltaCitations DeltaType = "citations_delta"
)

// Usage is the provider-specific token usage.
type Usage struct {
	InputTokens              int64 `json:"input_tokens"`
	OutputTokens             int64 `json:"output_tokens"`
	CacheCreationInputTokens int64 `json:"cache_creation_input_tokens"`
	CacheReadInputTokens     int64 `json:"cache_read_input_tokens"`
	CacheCreation            struct {
		Ephemeral1hInputTokens int64 `json:"ephemeral_1h_input_tokens"`
		Ephemeral5mInputTokens int64 `json:"ephemeral_5m_input_tokens"`
	} `json:"cache_creation"`
	ServiceTier   string `json:"service_tier"`  // "standard", "batch"
	InferenceGeo  string `json:"inference_geo"` // "not_available", "us", "eu"
	ServerToolUse struct {
		WebSearchRequests int64 `json:"web_search_requests"`
		WebFetchRequests  int64 `json:"web_fetch_requests"`
	} `json:"server_tool_use"`
	OutputTokensDetails struct {
		ThinkingTokens int64 `json:"thinking_tokens"`
	} `json:"output_tokens_details"`
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

// Init initializes the request from the given parameters.
func (b *BatchRequestItem) Init(msgs genai.Messages, model string, opts ...genai.GenOption) error {
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

// To converts to the genai equivalent.
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

// BatchListResponse is documented at https://docs.anthropic.com/en/api/listing-message-batches
type BatchListResponse struct {
	Data    []BatchResponse `json:"data"`
	FirstID string          `json:"first_id"`
	HasMore bool            `json:"has_more"`
	LastID  string          `json:"last_id"`
}

// BatchDeleteResponse is documented at https://docs.anthropic.com/en/api/deleting-message-batches
type BatchDeleteResponse struct {
	ID   string `json:"id"`
	Type string `json:"type"` // "message_batch_deleted"
}

// FileMetadata is described at https://docs.anthropic.com/en/api/files
type FileMetadata struct {
	ID           string    `json:"id"`
	CreatedAt    time.Time `json:"created_at"`
	Filename     string    `json:"filename"`
	MimeType     string    `json:"mime_type"`
	SizeBytes    int64     `json:"size_bytes"`
	Type         string    `json:"type"` // "file"
	Downloadable bool      `json:"downloadable"`
}

// FileListResponse is the response for listing files.
//
// https://docs.anthropic.com/en/api/files
type FileListResponse struct {
	Data    []FileMetadata `json:"data"`
	FirstID string         `json:"first_id"`
	HasMore bool           `json:"has_more"`
	LastID  string         `json:"last_id"`
}

// FileDeleteResponse is the response for deleting a file.
//
// https://docs.anthropic.com/en/api/files
type FileDeleteResponse struct {
	ID   string `json:"id"`
	Type string `json:"type"` // "file_deleted"
}

//

// CapabilitySupport indicates whether a capability is supported.
type CapabilitySupport struct {
	Supported bool `json:"supported"`
}

// ThinkingTypes describes supported thinking type configurations.
type ThinkingTypes struct {
	Adaptive CapabilitySupport `json:"adaptive"`
	Enabled  CapabilitySupport `json:"enabled"`
}

// ThinkingCapability describes thinking support and type configurations.
type ThinkingCapability struct {
	Supported bool          `json:"supported"`
	Types     ThinkingTypes `json:"types"`
}

// ContextManagementCapability describes context management support and available strategies.
type ContextManagementCapability struct {
	ClearThinking20251015 CapabilitySupport `json:"clear_thinking_20251015"`
	ClearToolUses20250919 CapabilitySupport `json:"clear_tool_uses_20250919"`
	Compact20260112       CapabilitySupport `json:"compact_20260112"`
	Supported             bool              `json:"supported"`
}

// EffortCapability describes effort (reasoning_effort) support and available levels.
type EffortCapability struct {
	High      CapabilitySupport `json:"high"`
	Low       CapabilitySupport `json:"low"`
	Max       CapabilitySupport `json:"max"`
	Medium    CapabilitySupport `json:"medium"`
	Supported bool              `json:"supported"`
	XHigh     CapabilitySupport `json:"xhigh"`
}

// ModelCapabilities describes model capability information.
type ModelCapabilities struct {
	Batch             CapabilitySupport           `json:"batch"`
	Citations         CapabilitySupport           `json:"citations"`
	CodeExecution     CapabilitySupport           `json:"code_execution"`
	ContextManagement ContextManagementCapability `json:"context_management"`
	Effort            EffortCapability            `json:"effort"`
	ImageInput        CapabilitySupport           `json:"image_input"`
	PDFInput          CapabilitySupport           `json:"pdf_input"`
	StructuredOutputs CapabilitySupport           `json:"structured_outputs"`
	Thinking          ThinkingCapability          `json:"thinking"`
}

// Model is the provider-specific model metadata.
type Model struct {
	Capabilities   ModelCapabilities `json:"capabilities"`
	CreatedAt      time.Time         `json:"created_at"`
	DisplayName    string            `json:"display_name"`
	ID             string            `json:"id"`
	MaxInputTokens int64             `json:"max_input_tokens"`
	MaxTokens      int64             `json:"max_tokens"`
	Type           string            `json:"type"`
}

// GetID implements genai.Model.
func (m *Model) GetID() string {
	return m.ID
}

func (m *Model) String() string {
	return fmt.Sprintf("%s: %s (%s)", m.ID, m.DisplayName, m.CreatedAt.Format("2006-01-02"))
}

// Context implements genai.Model.
func (m *Model) Context() int64 {
	return m.MaxInputTokens
}

// ModelsResponse represents the response structure for Anthropic models listing.
type ModelsResponse struct {
	Data    []Model `json:"data"`
	FirstID string  `json:"first_id"`
	HasMore bool    `json:"has_more"`
	LastID  string  `json:"last_id"`
}

// ToModels converts Anthropic models to genai.Model interfaces.
func (r *ModelsResponse) ToModels() []genai.Model {
	models := make([]genai.Model, len(r.Data))
	for i := range r.Data {
		models[i] = &r.Data[i]
	}
	return models
}

// CountTokensRequest is the request to the count_tokens API. It contains only the fields accepted by the
// endpoint, which is a subset of ChatRequest.
//
// https://docs.anthropic.com/en/api/messages-count-tokens
type CountTokensRequest struct {
	Model        string          `json:"model,omitzero"`
	Messages     []Message       `json:"messages"`
	System       []SystemMessage `json:"system,omitzero"`
	Thinking     Thinking        `json:"thinking,omitzero"`
	ToolChoice   ToolChoice      `json:"tool_choice,omitzero"`
	Tools        []Tool          `json:"tools,omitzero"`
	OutputConfig OutputConfig    `json:"output_config,omitzero"`
}

// CountTokensResponse is the response from the count_tokens API.
//
// https://docs.anthropic.com/en/api/messages-count-tokens
type CountTokensResponse struct {
	InputTokens int64 `json:"input_tokens"`
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

// IsAPIError implements base.ErrorResponseI.
func (er *ErrorResponse) IsAPIError() bool {
	return true
}
