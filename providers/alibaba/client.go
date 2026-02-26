// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Package alibaba implements a client for the Alibaba Cloud DashScope API (Model Studio).
//
// It uses the OpenAI-compatible endpoint documented at
// https://www.alibabacloud.com/help/en/model-studio/compatibility-of-openai-with-dashscope
//
// Regional endpoints:
//   - International: https://dashscope-intl.aliyuncs.com/compatible-mode/v1
//   - US: https://dashscope-us.aliyuncs.com/compatible-mode/v1
//   - China: https://dashscope.aliyuncs.com/compatible-mode/v1
package alibaba

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
	"os"
	"slices"
	"strings"

	"github.com/invopop/jsonschema"
	"github.com/maruel/genai"
	"github.com/maruel/genai/base"
	"github.com/maruel/genai/internal"
	"github.com/maruel/genai/scoreboard"
	"github.com/maruel/roundtrippers"
)

//go:embed scoreboard.json
var scoreboardJSON []byte

// Scoreboard for Alibaba Cloud.
func Scoreboard() scoreboard.Score {
	var s scoreboard.Score
	d := json.NewDecoder(bytes.NewReader(scoreboardJSON))
	d.DisallowUnknownFields()
	if err := d.Decode(&s); err != nil {
		panic(fmt.Errorf("failed to unmarshal scoreboard.json: %w", err))
	}
	return s
}

// ChatRequest is the OpenAI-compatible chat completion request with DashScope extensions.
type ChatRequest struct {
	Model          string    `json:"model"`
	Messages       []Message `json:"messages"`
	Stream         bool      `json:"stream"`
	Temperature    float64   `json:"temperature,omitzero"` // [0, 2]
	TopP           float64   `json:"top_p,omitzero"`       // [0, 1]
	TopK           int64     `json:"top_k,omitzero"`       // DashScope extension
	MaxToks        int64     `json:"max_tokens,omitzero"`  // Max output tokens
	Stop           []string  `json:"stop,omitzero"`        // Stop sequences
	ResponseFormat struct {
		Type string `json:"type,omitzero"` // "text", "json_object"
	} `json:"response_format,omitzero"`
	StreamOptions struct {
		IncludeUsage bool `json:"include_usage,omitzero"`
	} `json:"stream_options,omitzero"`
	ToolChoice string `json:"tool_choice,omitzero"` // "none", "auto", "required"
	Tools      []Tool `json:"tools,omitzero"`
	Seed       int64  `json:"seed,omitzero"`
	// DashScope extension: enable web search.
	EnableSearch bool `json:"enable_search,omitzero"`
}

// Init initializes the request from genai types.
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
			c.MaxToks = v.MaxTokens
			c.Temperature = v.Temperature
			c.TopP = v.TopP
			c.TopK = int64(v.TopK)
			sp = v.SystemPrompt
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
				}
			}
		case *genai.GenOptionWeb:
			if v.Search {
				c.EnableSearch = true
			}
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
		c.Messages = append(c.Messages, Message{Role: "system", Content: Contents{{Type: ContentText, Text: sp}}})
	}
	for i := range msgs {
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
	c.StreamOptions.IncludeUsage = stream
}

// Message is an OpenAI-compatible message with DashScope extensions.
type Message struct {
	Role             string     `json:"role,omitzero"` // "system", "assistant", "user"
	Name             string     `json:"name,omitzero"`
	Content          Contents   `json:"content,omitzero"`
	ReasoningContent string     `json:"reasoning_content,omitzero"` // Qwen3 thinking mode
	ToolCalls        []ToolCall `json:"tool_calls,omitzero"`
	ToolCallID       string     `json:"tool_call_id,omitzero"`
}

// From converts a genai.Message. Must be called with at most one ToolCallResults.
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
				// Qwen recommends against passing reasoning back.
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
	if m.ReasoningContent != "" {
		out.Replies = append(out.Replies, genai.Reply{Reasoning: m.ReasoningContent})
	}
	for _, c := range m.Content {
		switch c.Type {
		case ContentText:
			if c.Text == "" {
				return &internal.BadError{Err: errors.New("empty content text")}
			}
			out.Replies = append(out.Replies, genai.Reply{Text: c.Text})
		case ContentImageURL:
			return &internal.BadError{Err: fmt.Errorf("unsupported output content type %q", c.Type)}
		default:
			return &internal.BadError{Err: fmt.Errorf("unsupported output content type %q", c.Type)}
		}
	}
	for i := range m.ToolCalls {
		out.Replies = append(out.Replies, genai.Reply{})
		m.ToolCalls[i].To(&out.Replies[len(out.Replies)-1].ToolCall)
	}
	return nil
}

// Contents marshals single text blocks as a string for compatibility.
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

// Content is a content block supporting text and image_url types.
type Content struct {
	Type ContentType `json:"type,omitzero"`

	// Type == "text"
	Text string `json:"text,omitzero"`

	// Type == "image_url"
	ImageURL struct {
		URL    string `json:"url,omitzero"`
		Detail string `json:"detail,omitzero"` // "auto", "low", "high"
	} `json:"image_url,omitzero"`
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
				c.ImageURL.URL = fmt.Sprintf("data:%s;base64,%s", mimeType, base64.StdEncoding.EncodeToString(data))
			} else {
				c.ImageURL.URL = in.Doc.URL
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
				c.ImageURL.URL = fmt.Sprintf("data:%s;base64,%s", mimeType, base64.StdEncoding.EncodeToString(data))
			} else {
				c.ImageURL.URL = in.Doc.URL
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

// ContentType is a content type discriminator.
type ContentType string

// Content type values.
const (
	ContentText     ContentType = "text"
	ContentImageURL ContentType = "image_url"
)

// Tool is an OpenAI-compatible tool definition.
type Tool struct {
	Type     string `json:"type"` // "function"
	Function struct {
		Name        string             `json:"name,omitzero"`
		Description string             `json:"description,omitzero"`
		Parameters  *jsonschema.Schema `json:"parameters,omitzero"`
	} `json:"function"`
}

// ToolCall is a tool call in a response.
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

// ChatResponse is the chat completion response.
type ChatResponse struct {
	ID                string `json:"id"`
	SystemFingerprint string `json:"system_fingerprint"`
	RequestID         string `json:"request_id"` // DashScope-specific
	Choices           []struct {
		FinishReason FinishReason    `json:"finish_reason"`
		Index        int64           `json:"index"`
		Message      Message         `json:"message"`
		Logprobs     json.RawMessage `json:"logprobs"`
	} `json:"choices"`
	Created int64  `json:"created"`
	Model   string `json:"model"`
	Object  string `json:"object"` // "chat.completion"
	Usage   Usage  `json:"usage"`
}

// ToResult converts to a genai.Result.
func (c *ChatResponse) ToResult() (genai.Result, error) {
	out := genai.Result{
		Usage: genai.Usage{
			InputTokens:       c.Usage.PromptTokens,
			InputCachedTokens: c.Usage.PromptTokensDetails.CachedTokens,
			OutputTokens:      c.Usage.CompletionTokens,
		},
	}
	if len(c.Choices) != 1 {
		return out, fmt.Errorf("expected 1 choice, got %d", len(c.Choices))
	}
	out.Usage.FinishReason = c.Choices[0].FinishReason.ToFinishReason()
	err := c.Choices[0].Message.To(&out.Message)
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
	case "":
		return ""
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

// Usage is the token usage in a response.
type Usage struct {
	CompletionTokens    int64 `json:"completion_tokens"`
	PromptTokens        int64 `json:"prompt_tokens"`
	TotalTokens         int64 `json:"total_tokens"`
	PromptTokensDetails struct {
		CachedTokens int64 `json:"cached_tokens"`
		TextTokens   int64 `json:"text_tokens"`  // VL models
		ImageTokens  int64 `json:"image_tokens"` // VL models
	} `json:"prompt_tokens_details"`
	CompletionTokensDetails struct {
		ReasoningTokens int64 `json:"reasoning_tokens"`
		TextTokens      int64 `json:"text_tokens"` // VL models
	} `json:"completion_tokens_details"`
}

// ChatStreamChunkResponse is a streaming chat chunk.
type ChatStreamChunkResponse struct {
	ID                string `json:"id"`
	SystemFingerprint string `json:"system_fingerprint"`
	RequestID         string `json:"request_id"` // DashScope-specific
	Object            string `json:"object"`     // "chat.completion.chunk"
	Created           int64  `json:"created"`    // Unix timestamp
	Model             string `json:"model"`
	Choices           []struct {
		Index        int64           `json:"index"`
		Delta        Message         `json:"delta"`
		Logprobs     json.RawMessage `json:"logprobs"`
		FinishReason FinishReason    `json:"finish_reason"`
	} `json:"choices"`
	Usage Usage `json:"usage"`
}

// Model is a model returned by the DashScope models API.
type Model struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Created int64  `json:"created"`
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
	// DashScope models API does not return context window size.
	return 0
}

// ModelsResponse is the response from the DashScope models listing endpoint.
type ModelsResponse struct {
	Object  string  `json:"object"` // "list"
	Data    []Model `json:"data"`
	FirstID string  `json:"first_id"`
	LastID  string  `json:"last_id"`
	HasMore bool    `json:"has_more"`
}

// ToModels converts to genai.Model interfaces.
func (r *ModelsResponse) ToModels() []genai.Model {
	models := make([]genai.Model, len(r.Data))
	for i := range r.Data {
		models[i] = &r.Data[i]
	}
	return models
}

// ErrorResponse is the DashScope error response.
type ErrorResponse struct {
	ID        string `json:"id"`         // DashScope includes id in error responses
	RequestID string `json:"request_id"` // DashScope-specific
	ErrorVal  struct {
		Message string `json:"message"`
		Type    string `json:"type"`
		Param   string `json:"param"`
		Code    string `json:"code"`
	} `json:"error"`
}

func (er *ErrorResponse) Error() string {
	return fmt.Sprintf("%s: %s", er.ErrorVal.Type, er.ErrorVal.Message)
}

// IsAPIError implements base.ErrAPI.
func (er *ErrorResponse) IsAPIError() bool {
	return true
}

// Client implements genai.Provider for Alibaba Cloud DashScope.
type Client struct {
	base.NotImplemented
	impl    base.Provider[*ErrorResponse, *ChatRequest, *ChatResponse, ChatStreamChunkResponse]
	baseURL string
}

// ProviderOptionBackend selects a DashScope regional endpoint.
type ProviderOptionBackend string

const (
	// BackendIntl is the international (Singapore) endpoint (default).
	BackendIntl ProviderOptionBackend = "dashscope-intl"
	// BackendUS is the US (Virginia) endpoint.
	BackendUS ProviderOptionBackend = "dashscope-us"
	// BackendCN is the China (Beijing) endpoint.
	BackendCN ProviderOptionBackend = "dashscope"
)

// Validate implements genai.ProviderOption.
func (p ProviderOptionBackend) Validate() error {
	switch p {
	case BackendIntl, BackendUS, BackendCN:
		return nil
	default:
		return fmt.Errorf("unknown backend %q, use BackendIntl, BackendUS, or BackendCN", string(p))
	}
}

// New creates a new client for the Alibaba Cloud DashScope API.
//
// If ProviderOptionAPIKey is not provided, it tries DASHSCOPE_API_KEY_INTL,
// DASHSCOPE_API_KEY_US, DASHSCOPE_API_KEY_CN (auto-selecting the backend),
// then DASHSCOPE_API_KEY.
//
// ProviderOptionBackend selects a named regional endpoint (e.g. BackendUS).
// When set, the matching DASHSCOPE_API_KEY_<region> is tried first.
// ProviderOptionRemote overrides all other endpoint selection with a full URL.
func New(ctx context.Context, opts ...genai.ProviderOption) (*Client, error) {
	var apiKey, model, remote string
	var backend ProviderOptionBackend
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
		case genai.ProviderOptionRemote:
			remote = string(v)
		case ProviderOptionBackend:
			backend = v
		default:
			return nil, fmt.Errorf("unsupported option type %T", opt)
		}
	}
	const apiKeyURL = "https://modelstudio.console.alibabacloud.com/"
	var err error
	if apiKey == "" {
		// When backend is set, try its matching region key first.
		switch backend {
		case BackendIntl:
			apiKey = os.Getenv("DASHSCOPE_API_KEY_INTL")
		case BackendUS:
			apiKey = os.Getenv("DASHSCOPE_API_KEY_US")
		case BackendCN:
			apiKey = os.Getenv("DASHSCOPE_API_KEY_CN")
		default:
			// Auto-detect: first region key found sets the backend.
			if v := os.Getenv("DASHSCOPE_API_KEY_INTL"); v != "" {
				apiKey = v
				backend = BackendIntl
			} else if v := os.Getenv("DASHSCOPE_API_KEY_US"); v != "" {
				apiKey = v
				backend = BackendUS
			} else if v := os.Getenv("DASHSCOPE_API_KEY_CN"); v != "" {
				apiKey = v
				backend = BackendCN
			}
		}
		if apiKey == "" {
			if apiKey = os.Getenv("DASHSCOPE_API_KEY"); apiKey == "" {
				err = &base.ErrAPIKeyRequired{EnvVar: "DASHSCOPE_API_KEY", URL: apiKeyURL}
			}
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
	if remote == "" {
		switch backend {
		case BackendUS:
			remote = "https://dashscope-us.aliyuncs.com/compatible-mode/v1"
		case BackendCN:
			remote = "https://dashscope.aliyuncs.com/compatible-mode/v1"
		default:
			remote = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
		}
	}
	c := &Client{
		baseURL: remote,
		impl: base.Provider[*ErrorResponse, *ChatRequest, *ChatResponse, ChatStreamChunkResponse]{
			GenSyncURL:      remote + "/chat/completions",
			ProcessStream:   ProcessStream,
			PreloadedModels: preloadedModels,
			ProviderBase: base.ProviderBase[*ErrorResponse]{
				APIKeyURL: apiKeyURL,
				Lenient:   internal.BeLenient,
				Client: http.Client{
					Transport: &roundtrippers.Header{
						Header:    http.Header{"Authorization": {"Bearer " + apiKey}},
						Transport: &roundtrippers.RequestID{Transport: t},
					},
				},
			},
		},
	}
	if err == nil {
		switch model {
		case "":
		case string(genai.ModelCheap):
			c.impl.Model = "qwen3-30b-a3b"
			c.impl.OutputModalities = mod
		case string(genai.ModelGood):
			c.impl.Model = "qwen3-235b-a22b"
			c.impl.OutputModalities = mod
		case string(genai.ModelSOTA):
			c.impl.Model = "qwen3-max"
			c.impl.OutputModalities = mod
		default:
			c.impl.Model = model
			c.impl.OutputModalities = mod
		}
	}
	return c, err
}

// Name implements genai.Provider.
func (c *Client) Name() string {
	return "alibaba"
}

// ModelID implements genai.Provider.
func (c *Client) ModelID() string {
	return c.impl.Model
}

// OutputModalities implements genai.Provider.
func (c *Client) OutputModalities() genai.Modalities {
	return c.impl.OutputModalities
}

// Scoreboard implements genai.Provider.
func (c *Client) Scoreboard() scoreboard.Score {
	return Scoreboard()
}

// HTTPClient returns the HTTP client.
func (c *Client) HTTPClient() *http.Client {
	return &c.impl.Client
}

// GenSync implements genai.Provider.
func (c *Client) GenSync(ctx context.Context, msgs genai.Messages, opts ...genai.GenOption) (genai.Result, error) {
	return c.impl.GenSync(ctx, msgs, opts...)
}

// GenSyncRaw provides access to the raw API.
func (c *Client) GenSyncRaw(ctx context.Context, in *ChatRequest, out *ChatResponse) error {
	return c.impl.GenSyncRaw(ctx, in, out)
}

// GenStream implements genai.Provider.
func (c *Client) GenStream(ctx context.Context, msgs genai.Messages, opts ...genai.GenOption) (iter.Seq[genai.Reply], func() (genai.Result, error)) {
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
	var resp ModelsResponse
	if err := c.impl.DoRequest(ctx, "GET", c.baseURL+"/models", nil, &resp); err != nil {
		return nil, err
	}
	return resp.ToModels(), nil
}

// ProcessStream converts raw stream chunks to genai.Reply fragments.
func ProcessStream(chunks iter.Seq[ChatStreamChunkResponse]) (iter.Seq[genai.Reply], func() (genai.Usage, [][]genai.Logprob, error)) {
	var finalErr error
	u := genai.Usage{}

	return func(yield func(genai.Reply) bool) {
			pendingToolCall := ToolCall{}
			for pkt := range chunks {
				if len(pkt.Choices) != 1 {
					continue
				}
				if pkt.Usage.CompletionTokens != 0 {
					u.InputTokens = pkt.Usage.PromptTokens
					u.InputCachedTokens = pkt.Usage.PromptTokensDetails.CachedTokens
					u.OutputTokens = pkt.Usage.CompletionTokens
					u.FinishReason = pkt.Choices[0].FinishReason.ToFinishReason()
				}
				if len(pkt.Choices[0].Delta.ToolCalls) > 1 {
					finalErr = &internal.BadError{Err: fmt.Errorf("implement multiple tool calls: %#v", pkt)}
					return
				}
				switch role := pkt.Choices[0].Delta.Role; role {
				case "assistant", "":
				default:
					finalErr = &internal.BadError{Err: fmt.Errorf("unexpected role %q", role)}
					return
				}
				f := genai.Reply{}
				if s := pkt.Choices[0].Delta.ReasoningContent; s != "" {
					f.Reasoning = s
				}
				for _, c := range pkt.Choices[0].Delta.Content {
					switch c.Type {
					case ContentText:
						f.Text += c.Text
					default:
						finalErr = &internal.BadError{Err: fmt.Errorf("unsupported stream content type %q", c.Type)}
						return
					}
				}
				// Buffer tool call arguments to send as a whole.
				if len(pkt.Choices[0].Delta.ToolCalls) == 1 {
					if t := pkt.Choices[0].Delta.ToolCalls[0]; t.ID != "" {
						if pendingToolCall.ID == "" {
							pendingToolCall = t
							if !f.IsZero() {
								finalErr = &internal.BadError{Err: fmt.Errorf("implement tool call with metadata: %#v", pkt)}
								return
							}
							continue
						}
						// Flush previous.
						pendingToolCall.To(&f.ToolCall)
						pendingToolCall = t
					} else if pendingToolCall.ID != "" {
						pendingToolCall.Function.Arguments += t.Function.Arguments
						if !f.IsZero() {
							finalErr = &internal.BadError{Err: fmt.Errorf("implement tool call with metadata: %#v", pkt)}
							return
						}
						continue
					}
				} else if pendingToolCall.ID != "" {
					pendingToolCall.To(&f.ToolCall)
					pendingToolCall = ToolCall{}
				}
				if !yield(f) {
					return
				}
			}
		}, func() (genai.Usage, [][]genai.Logprob, error) {
			return u, nil, finalErr
		}
}

var _ genai.Provider = &Client{}
