// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Wire types for the Groq chat completions REST API.
//
// Reference: https://console.groq.com/docs/api-reference

package groq

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

// ServiceTier is the quality of service to determine the request's priority.
// https://console.groq.com/docs/flex-processing
type ServiceTier string

const (
	// ServiceTierOnDemand is the default tier and the one you are used to. We have kept rate limits low in
	// order to ensure fairness and a consistent experience.
	ServiceTierOnDemand ServiceTier = "on_demand"
	// ServiceTierAuto uses on-demand rate limits, then falls back to flex tier if those limits are exceeded.
	ServiceTierAuto ServiceTier = "auto"
	// ServiceTierFlex offers on-demand processing when capacity is available, with rapid timeouts if resources
	// are constrained. This tier is perfect for workloads that prioritize fast inference and can gracefully
	// handle occasional request failures. It provides an optimal balance between performance and reliability
	// for workloads that don't require guaranteed processing.
	ServiceTierFlex ServiceTier = "flex"
)

// ReasoningFormat defines the post processing format of the reasoning done by groq for select models.
//
// See https://console.groq.com/docs/reasoning
type ReasoningFormat string

// Reasoning format values.
const (
	ReasoningFormatParsed ReasoningFormat = "parsed"
	ReasoningFormatRaw    ReasoningFormat = "raw"
	ReasoningFormatHidden ReasoningFormat = "hidden"
)

// ChatRequest is documented at https://console.groq.com/docs/api-reference#chat-create
type ChatRequest struct {
	FrequencyPenalty  float64         `json:"frequency_penalty,omitzero"` // [-2.0, 2.0]
	MaxChatTokens     int64           `json:"max_completion_tokens,omitzero"`
	Messages          []Message       `json:"messages"`
	Model             string          `json:"model"`
	ParallelToolCalls bool            `json:"parallel_tool_calls,omitzero"`
	PresencePenalty   float64         `json:"presence_penalty,omitzero"` // [-2.0, 2.0]
	ReasoningFormat   ReasoningFormat `json:"reasoning_format,omitzero"`
	ResponseFormat    struct {
		Type       string           `json:"type,omitzero"` // "json_object", "json_schema"
		JSONSchema genai.JSONSchema `json:"json_schema,omitzero"`
	} `json:"response_format,omitzero"`
	SearchSettings struct {
		Country        string   `json:"country,omitzero"`
		ExcludeDomains []string `json:"exclude_domains,omitzero"`
		IncludeDomains []string `json:"include_domains,omitzero"`
		IncludeImages  bool     `json:"include_images,omitzero"`
	} `json:"search_settings,omitzero"`
	Seed          int64       `json:"seed,omitzero"`
	ServiceTier   ServiceTier `json:"service_tier,omitzero"`
	Stop          []string    `json:"stop,omitzero"` // keywords to stop completion
	Stream        bool        `json:"stream"`
	StreamOptions struct {
		IncludeUsage bool `json:"include_usage,omitzero"`
	} `json:"stream_options,omitzero"`
	Temperature float64 `json:"temperature,omitzero"` // [0, 2]
	Tools       []Tool  `json:"tools,omitzero"`
	// Alternative when forcing a specific function. This can probably be achieved
	// by providing a single tool and ToolChoice == "required".
	// ToolChoice struct {
	// 	Type     string `json:"type,omitzero"` // "function"
	// 	Function struct {
	// 		Name string `json:"name,omitzero"`
	// 	} `json:"function,omitzero"`
	// } `json:"tool_choice,omitzero"`
	ToolChoice string  `json:"tool_choice,omitzero"` // "none", "auto", "required", or struct {"type": "function", "function": {"name": "my_function"}}
	TopP       float64 `json:"top_p,omitzero"`       // [0, 1]
	User       string  `json:"user,omitzero"`

	// Explicitly Unsupported:
	// LogitBias           map[string]float64 `json:"logit_bias,omitzero"`
	// Logprobs            bool               `json:"logprobs,omitzero"`
	// TopLogprobs         int64                `json:"top_logprobs,omitzero"`     // [0, 20]
	// N                   int64                `json:"n,omitzero"`                // Number of choices
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
			c.ServiceTier = v.ServiceTier
			c.ReasoningFormat = v.ReasoningFormat
		case *genai.GenOptionText:
			u, err := c.initOptionsText(v)
			unsupported = append(unsupported, u...)
			if err != nil {
				errs = append(errs, err)
			}
			sp = v.SystemPrompt
		case *genai.GenOptionTools:
			if err := c.initOptionsTools(v); err != nil {
				errs = append(errs, err)
			}
		case *genai.GenOptionWeb:
			if v.Search {
				// https://console.groq.com/docs/browser-search
				// TODO: Country and domains
				c.SearchSettings.IncludeImages = true
				c.Tools = append(c.Tools, Tool{Type: "browser_search"})
			}
			// Fetch (visit_website) is only available on compound models, not chat completions.
			// https://console.groq.com/docs/agentic-tooling
			if v.Fetch {
				unsupported = append(unsupported, "GenOptionWeb.Fetch")
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

func (c *ChatRequest) initOptionsText(v *genai.GenOptionText) ([]string, error) {
	var unsupported []string
	c.MaxChatTokens = v.MaxTokens
	c.Temperature = v.Temperature
	c.TopP = v.TopP
	if v.TopK != 0 {
		unsupported = append(unsupported, "GenOptionText.TopK")
	}
	if v.TopLogprobs != 0 {
		unsupported = append(unsupported, "GenOptionText.TopLogprobs")
	}
	c.Stop = v.Stop
	if v.DecodeAs != nil {
		c.ResponseFormat.Type = "json_schema"
		s, err := v.DecodeSchema()
		if err != nil {
			return unsupported, err
		}
		c.ResponseFormat.JSONSchema = append(s[:len(s)-1], `,"name":"response"}`...)
	} else if v.ReplyAsJSON {
		c.ResponseFormat.Type = "json_object"
	}
	return unsupported, nil
}

func (c *ChatRequest) initOptionsTools(v *genai.GenOptionTools) error {
	if len(v.Tools) != 0 {
		switch v.Force {
		case genai.ToolCallAny:
			c.ToolChoice = "auto"
		case genai.ToolCallRequired:
			c.ToolChoice = "required"
		case genai.ToolCallNone:
			c.ToolChoice = "none"
		}
		// Documentation states max is 128 tools.
		c.Tools = make([]Tool, len(v.Tools))
		for i, t := range v.Tools {
			c.Tools[i].Type = "function"
			c.Tools[i].Function.Name = t.Name
			c.Tools[i].Function.Description = t.Description
			s, err := t.GetInputSchema()
			if err != nil {
				return err
			}
			c.Tools[i].Function.Parameters = s
		}
	}
	return nil
}

// Message is documented at https://console.groq.com/docs/api-reference#chat-create
type Message struct {
	Role       string     `json:"role"`               // "system", "assistant", "user"
	Name       string     `json:"name,omitzero"`      // An optional name for the participant. Provides the model information to differentiate between participants of the same role.
	Content    Contents   `json:"content,omitzero"`   // Content is always returned as a string.
	Reasoning  string     `json:"reasoning,omitzero"` // In replies
	Channel    string     `json:"channel,omitzero"`   // "analysis"
	ToolCalls  []ToolCall `json:"tool_calls,omitzero"`
	ToolCallID string     `json:"tool_call_id,omitzero"`

	// "browser_search"
	ExecutedTools []struct {
		Name string `json:"name,omitzero"` // "browser.search", "browser.open", "browser.find"
		// Arguments is a JSON encoded arguments for the server side tool.
		//  - "browser.search" uses BrowserSearchArguments.
		//  - "browser.open" uses BrowserOpenArguments.
		Arguments     string `json:"arguments,omitzero"`
		Index         int64  `json:"index,omitzero"`
		Type          string `json:"type,omitzero"` // "function", "python", etc.
		Output        string `json:"output,omitzero"`
		SearchResults struct {
			Results []struct {
				Title   string  `json:"title,omitzero"`
				URL     string  `json:"url,omitzero"`
				Content string  `json:"content,omitzero"`
				Score   float64 `json:"score,omitzero"`
			} `json:"results,omitzero"`
		} `json:"search_results,omitzero"`
		CodeResults []struct {
			Text string `json:"text,omitzero"`
		} `json:"code_results,omitzero"`
	} `json:"executed_tools,omitzero"`
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
	m.Name = in.User
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
				// DeepSeek and Qwen recommend against passing reasoning back.
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
		// Process only the first tool call result in this method.
		// The Init method handles multiple tool call results by creating multiple messages.
		m.Content = Contents{{Type: ContentText, Text: in.ToolCallResults[0].Result}}
		m.ToolCallID = in.ToolCallResults[0].ID
	}
	return nil
}

// To converts to the genai equivalent.
func (m *Message) To(out *genai.Message) error {
	if m.Reasoning != "" {
		out.Replies = append(out.Replies, genai.Reply{Reasoning: m.Reasoning})
	}
	for _, c := range m.Content {
		switch c.Type {
		case ContentText:
			if c.Text == "" {
				return &internal.BadError{Err: errors.New("empty content text")}
			}
			out.Replies = append(out.Replies, genai.Reply{Text: c.Text})
		case ContentImageURL:
			return &internal.BadError{Err: fmt.Errorf("implement content type %q", c.Type)}
		default:
			return &internal.BadError{Err: fmt.Errorf("implement content type %q", c.Type)}
		}
	}
	for i := range m.ToolCalls {
		out.Replies = append(out.Replies, genai.Reply{})
		m.ToolCalls[i].To(&out.Replies[len(out.Replies)-1].ToolCall)
	}
	for _, t := range m.ExecutedTools {
		// If Name is empty, use Type instead
		toolName := t.Name
		if toolName == "" {
			toolName = t.Type
		}
		switch toolName {
		case "browser.search", "search":
			d := json.NewDecoder(strings.NewReader(t.Arguments))
			d.DisallowUnknownFields()
			args := BrowserSearchArguments{}
			if err := d.Decode(&args); err != nil {
				return &internal.BadError{Err: fmt.Errorf("failed to unmarshal arguments for executed tool %q: %w", toolName, err)}
			}
			c := genai.Citation{
				Sources: make([]genai.CitationSource, 0, len(t.SearchResults.Results)+1),
			}
			c.Sources = append(c.Sources, genai.CitationSource{
				Type:    genai.CitationWebQuery,
				Snippet: args.Query,
			})
			for _, r := range t.SearchResults.Results {
				c.Sources = append(c.Sources, genai.CitationSource{
					Type: genai.CitationWeb, Title: r.Title, URL: r.URL, Snippet: r.Content,
				})
			}
			out.Replies = append(out.Replies, genai.Reply{Citation: c})
		case "browser.open", "browser.find", "visit":
			// Ignore, it's really useless.
		case "python":
			// Ignore python execution tool results from model reasoning
		default:
			return &internal.BadError{Err: fmt.Errorf("implement executed tool %q", toolName)}
		}
	}
	return nil
}

// Contents exists to marshal single content text block as a string.
//
// Groq requires this for assistant messages.
type Contents []Content

// IsZero reports whether the value is zero.
func (c *Contents) IsZero() bool {
	return len(*c) == 0
}

// MarshalJSON implements json.Marshaler.
func (c *Contents) MarshalJSON() ([]byte, error) {
	if len(*c) == 0 {
		// It's important otherwise Qwen3 fails with:
		// ('messages.2.content' : value must be a string) OR ('messages.2.content' : minimum number of items is 1)
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
		// e.g. tool calls.
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
		// Decode empty string as nil.
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
	ImageURL struct {
		Detail string `json:"detail,omitzero"` // "auto", "low", "high"
		URL    string `json:"url,omitzero"`    // URL or base64 encoded image
	} `json:"image_url,omitzero"`
}

// FromRequest converts from a genai request.
func (c *Content) FromRequest(in *genai.Request) error {
	// DeepSeek and Qwen recommend against passing reasoning back to the model.
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
				c.ImageURL.URL = fmt.Sprintf("data:%s;base64,%s", mimeType, base64.StdEncoding.EncodeToString(data))
			} else {
				c.ImageURL.URL = in.Doc.URL
			}
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

// FromReply converts from a genai reply.
func (c *Content) FromReply(in *genai.Reply) error {
	if len(in.Opaque) != 0 {
		return &internal.BadError{Err: errors.New("field Reply.Opaque not supported")}
	}
	// DeepSeek and Qwen recommend against passing reasoning back to the model.
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
				c.ImageURL.URL = fmt.Sprintf("data:%s;base64,%s", mimeType, base64.StdEncoding.EncodeToString(data))
			} else {
				c.ImageURL.URL = in.Doc.URL
			}
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
	// TODO: ExecutedTools.
	return &internal.BadError{Err: errors.New("unknown Reply type")}
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
	Type     string `json:"type,omitzero"` // "function"
	Function struct {
		Name        string           `json:"name,omitzero"`
		Description string           `json:"description,omitzero"`
		Parameters  genai.JSONSchema `json:"parameters,omitzero"`
	} `json:"function,omitzero"`
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

// UsageBreakdownModel represents per-model usage information in a breakdown.
type UsageBreakdownModel struct {
	Model string `json:"model"`
	Usage Usage  `json:"usage"`
}

// UsageBreakdown contains per-model usage information.
type UsageBreakdown struct {
	Models []UsageBreakdownModel `json:"models"`
}

// ChatResponse is the provider-specific chat completion response.
type ChatResponse struct {
	Choices []struct {
		FinishReason FinishReason `json:"finish_reason"`
		Index        int64        `json:"index"`
		Message      Message      `json:"message"`
		Logprobs     struct{}     `json:"logprobs"`
	} `json:"choices"`
	Created           base.TimeS     `json:"created"`
	ID                string         `json:"id"`
	Model             string         `json:"model"`
	Object            string         `json:"object"` // "chat.completion"
	Usage             Usage          `json:"usage"`
	UsageBreakdown    UsageBreakdown `json:"usage_breakdown"`
	SystemFingerprint string         `json:"system_fingerprint"`
	ServiceTier       ServiceTier    `json:"service_tier"`
	Xgroq             struct {
		ID             string         `json:"id"`
		Seed           json.Number    `json:"seed,omitzero"`
		UsageBreakdown UsageBreakdown `json:"usage_breakdown,omitzero"`
	} `json:"x_groq"`
}

// ToResult converts the response to a genai.Result.
func (c *ChatResponse) ToResult() (genai.Result, error) {
	out := genai.Result{
		// At the moment, Groq does not support cached tokens.
		Usage: genai.Usage{
			InputTokens:  c.Usage.PromptTokens,
			OutputTokens: c.Usage.CompletionTokens,
			TotalTokens:  c.Usage.TotalTokens,
			ServiceTier:  string(c.ServiceTier),
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
	QueueTime               float64                    `json:"queue_time"`
	PromptTokens            int64                      `json:"prompt_tokens"`
	PromptTime              float64                    `json:"prompt_time"`
	CompletionTokens        int64                      `json:"completion_tokens"`
	CompletionTime          float64                    `json:"completion_time"`
	TotalTokens             int64                      `json:"total_tokens"`
	TotalTime               float64                    `json:"total_time"`
	PromptTokensDetails     map[string]json.RawMessage `json:"prompt_tokens_details,omitzero"`
	CompletionTokensDetails map[string]json.RawMessage `json:"completion_tokens_details,omitzero"`
}

// BrowserSearchArguments is the Argument for the "browser.search" tool.
type BrowserSearchArguments struct {
	Query  string `json:"query,omitzero"`
	TopN   int64  `json:"topn,omitzero"`
	Source string `json:"source,omitzero"`
}

// BrowserOpenArguments is the Argument for the "browser.open" tool.
type BrowserOpenArguments struct {
	Cursor int64 `json:"cursor,omitzero"`
	ID     int64 `json:"id,omitzero"`
}

// BrowserFindArguments is the Argument for the "browser.find" tool.
type BrowserFindArguments struct {
	Cursor  int64  `json:"cursor,omitzero"`
	Pattern string `json:"pattern,omitzero"`
}

// ChatStreamChunkResponse is the provider-specific streaming chat chunk.
type ChatStreamChunkResponse struct {
	ID                string     `json:"id"`
	Object            string     `json:"object"`
	Created           base.TimeS `json:"created"`
	Model             string     `json:"model"`
	SystemFingerprint string     `json:"system_fingerprint"`
	Choices           []struct {
		Index        int64        `json:"index"`
		Delta        Message      `json:"delta"`
		Logprobs     struct{}     `json:"logprobs"` // Groq doesn't support logprobs but still sent a null item.
		FinishReason FinishReason `json:"finish_reason"`
	} `json:"choices"`
	Usage Usage `json:"usage,omitzero"`
	Xgroq struct {
		ID             string         `json:"id"`
		Seed           json.Number    `json:"seed,omitzero"`
		Usage          Usage          `json:"usage"`
		UsageBreakdown UsageBreakdown `json:"usage_breakdown,omitzero"`
	} `json:"x_groq"`
}

// Model is the provider-specific model metadata.
type Model struct {
	ID                          string            `json:"id"`
	Object                      string            `json:"object"`
	Created                     base.TimeS        `json:"created"`
	OwnedBy                     string            `json:"owned_by"`
	Name                        string            `json:"name"`
	Active                      bool              `json:"active"`
	ContextWindow               int64             `json:"context_window"`
	ContextLength               int64             `json:"context_length"`
	PublicApps                  []string          `json:"public_apps"`
	InputModalities             []string          `json:"input_modalities"`
	OutputModalities            []string          `json:"output_modalities"`
	SupportedFeatures           []string          `json:"supported_features"`
	SupportedSamplingParameters []string          `json:"supported_sampling_parameters"`
	HuggingFaceID               string            `json:"hugging_face_id"`
	Pricing                     map[string]string `json:"pricing"`
	MaxCompletionTokens         int64             `json:"max_completion_tokens"`
	MaxOutputLength             int64             `json:"max_output_length"`
}

// GetID implements genai.Model.
func (m *Model) GetID() string {
	return m.ID
}

func (m *Model) String() string {
	suffix := ""
	if !m.Active {
		suffix = " (inactive)"
	}
	return fmt.Sprintf("%s (%s) Context: %d/%d%s", m.ID, m.Created.AsTime().Format("2006-01-02"), m.Context(), m.MaxOutputLength, suffix)
}

// Context implements genai.Model.
func (m *Model) Context() int64 {
	return max(m.ContextLength, m.ContextWindow)
}

// ModelsResponse represents the response structure for Groq models listing.
type ModelsResponse struct {
	Object string  `json:"object"` // list
	Data   []Model `json:"data"`
}

// ToModels converts Groq models to genai.Model interfaces.
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
		Message          string `json:"message"`
		Type             string `json:"type"`  // e.g. "invalid_request_error"
		Code             string `json:"code"`  // e.g. "context_length_exceeded"
		Param            string `json:"param"` // e.g. "messages"
		FailedGeneration string `json:"failed_generation"`
		StatusCode       int64  `json:"status_code"`
	} `json:"error"`
}

func (er *ErrorResponse) Error() string {
	suffix := ""
	if er.ErrorVal.Param != "" {
		suffix += fmt.Sprintf(" (param: %q)", er.ErrorVal.Param)
	}
	if er.ErrorVal.FailedGeneration != "" {
		suffix += fmt.Sprintf("failed generation: %q", er.ErrorVal.FailedGeneration)
	}
	return fmt.Sprintf("%s (%s): %s%s", er.ErrorVal.Code, er.ErrorVal.Type, er.ErrorVal.Message, suffix)
}

// IsAPIError implements base.ErrorResponseI.
func (er *ErrorResponse) IsAPIError() bool {
	return true
}
