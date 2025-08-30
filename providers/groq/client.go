// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Package groq implements a client for the Groq API.
//
// It is described at https://console.groq.com/docs/api-reference
package groq

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
	"reflect"
	"slices"
	"strconv"
	"strings"
	"time"

	"github.com/invopop/jsonschema"
	"github.com/maruel/genai"
	"github.com/maruel/genai/base"
	"github.com/maruel/genai/internal"
	"github.com/maruel/genai/scoreboard"
	"github.com/maruel/roundtrippers"
)

//go:embed scoreboard.json
var scoreboardJSON []byte

// Scoreboard for Groq.
func Scoreboard() scoreboard.Score {
	var s scoreboard.Score
	d := json.NewDecoder(bytes.NewReader(scoreboardJSON))
	d.DisallowUnknownFields()
	if err := d.Decode(&s); err != nil {
		panic(fmt.Errorf("failed to unmarshal scoreboard.json: %w", err))
	}
	return s
}

// TODO: Split in two.

// Options is the Groq-specific options.
type Options struct {
	// ReasoningFormat requests Groq to process the stream on our behalf. It must only be used on reasoning
	// models. It is required for reasoning models to enable JSON structured output or tool calling.
	ReasoningFormat ReasoningFormat
	// ServiceTier specify the priority.
	ServiceTier ServiceTier
}

func (o *Options) Validate() error {
	// TODO: validate ReasoningFormat and ServiceTier.
	return nil
}

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
		Type       string             `json:"type,omitzero"` // "json_object", "json_schema"
		JSONSchema *jsonschema.Schema `json:"json_schema,omitzero"`
	} `json:"response_format,omitzero"`
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
func (c *ChatRequest) Init(msgs genai.Messages, model string, opts ...genai.Options) error {
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
		case *Options:
			c.ServiceTier = v.ServiceTier
			c.ReasoningFormat = v.ReasoningFormat
		case *genai.OptionsText:
			u, e := c.initOptionsText(v)
			unsupported = append(unsupported, u...)
			errs = append(errs, e...)
			sp = v.SystemPrompt
		case *genai.OptionsTools:
			u, e := c.initOptionsTools(v)
			unsupported = append(unsupported, u...)
			errs = append(errs, e...)
		default:
			errs = append(errs, fmt.Errorf("unsupported options type %T", opt))
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
	// If we have unsupported features but no other errors, return a continuable error
	if len(unsupported) > 0 && len(errs) == 0 {
		return &genai.UnsupportedContinuableError{Unsupported: unsupported}
	}
	return errors.Join(errs...)
}

func (c *ChatRequest) SetStream(stream bool) {
	c.Stream = stream
}

func (c *ChatRequest) initOptionsText(v *genai.OptionsText) ([]string, []error) {
	var errs []error
	var unsupported []string
	c.MaxChatTokens = v.MaxTokens
	c.Temperature = v.Temperature
	c.TopP = v.TopP
	c.Seed = v.Seed
	if v.TopK != 0 {
		unsupported = append(unsupported, "OptionsText.TopK")
	}
	if v.TopLogprobs != 0 {
		unsupported = append(unsupported, "OptionsText.TopLogprobs")
	}
	c.Stop = v.Stop
	if v.DecodeAs != nil {
		c.ResponseFormat.Type = "json_schema"
		c.ResponseFormat.JSONSchema = internal.JSONSchemaFor(reflect.TypeOf(v.DecodeAs))
		c.ResponseFormat.JSONSchema.Extras = map[string]any{"name": "response"}
	} else if v.ReplyAsJSON {
		c.ResponseFormat.Type = "json_object"
	}
	return unsupported, errs
}

func (c *ChatRequest) initOptionsTools(v *genai.OptionsTools) ([]string, []error) {
	var errs []error
	var unsupported []string
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
			if c.Tools[i].Function.Parameters = t.InputSchemaOverride; c.Tools[i].Function.Parameters == nil {
				c.Tools[i].Function.Parameters = t.GetInputSchema()
			}
		}
	}
	if v.WebSearch {
		unsupported = append(unsupported, "OptionsTools.WebSearch")
	}
	return unsupported, errs
}

// ReasoningFormat defines the post processing format of the reasoning done by groq for select models.
//
// See https://console.groq.com/docs/reasoning
type ReasoningFormat string

const (
	ReasoningFormatParsed ReasoningFormat = "parsed"
	ReasoningFormatRaw    ReasoningFormat = "raw"
	ReasoningFormatHidden ReasoningFormat = "hidden"
)

// Message is documented at https://console.groq.com/docs/api-reference#chat-create
type Message struct {
	Role       string     `json:"role"`          // "system", "assistant", "user"
	Name       string     `json:"name,omitzero"` // An optional name for the participant. Provides the model information to differentiate between participants of the same role.
	Content    Contents   `json:"content,omitzero"`
	ToolCalls  []ToolCall `json:"tool_calls,omitzero"`
	ToolCallID string     `json:"tool_call_id,omitzero"`
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

// Contents exists to marshal single content text block as a string.
//
// Groq requires this for assistant messages.
type Contents []Content

func (c *Contents) IsZero() bool {
	return len(*c) == 0
}

func (c *Contents) MarshalJSON() ([]byte, error) {
	if len(*c) == 0 {
		// It's important otherwise Qwen3 fails with:
		// ('messages.2.content' : value must be a string) OR ('messages.2.content' : minimum number of items is 1)
		return []byte("null"), nil
	}
	if len(*c) == 1 && (*c)[0].Type == ContentText {
		return json.Marshal((*c)[0].Text)
	}
	return json.Marshal(([]Content)(*c))
}

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
		case strings.HasPrefix(mimeType, "text/plain"):
			c.Type = ContentText
			if in.Doc.URL != "" {
				return errors.New("text/plain documents must be provided inline, not as a URL")
			}
			c.Text = string(data)
		default:
			return fmt.Errorf("unsupported mime type %s", mimeType)
		}
		return nil
	}
	return errors.New("unknown Request type")
}

func (c *Content) FromReply(in *genai.Reply) error {
	if len(in.Opaque) != 0 {
		return errors.New("field Reply.Opaque not supported")
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
		case strings.HasPrefix(mimeType, "text/plain"):
			c.Type = ContentText
			if in.Doc.URL != "" {
				return errors.New("text/plain documents must be provided inline, not as a URL")
			}
			c.Text = string(data)
		default:
			return fmt.Errorf("unsupported mime type %s", mimeType)
		}
		return nil
	}
	return errors.New("unknown Reply type")
}

type ContentType string

const (
	ContentText     ContentType = "text"
	ContentImageURL ContentType = "image_url"
)

type Tool struct {
	Type     string `json:"type,omitzero"` // "function"
	Function struct {
		Name        string             `json:"name,omitzero"`
		Description string             `json:"description,omitzero"`
		Parameters  *jsonschema.Schema `json:"parameters,omitzero"`
	} `json:"function,omitzero"`
}

type ToolCall struct {
	Index    int64  `json:"index,omitzero"`
	Type     string `json:"type,omitzero"` // "function"
	ID       string `json:"id,omitzero"`
	Function struct {
		Name      string `json:"name,omitzero"`
		Arguments string `json:"arguments,omitzero"`
	} `json:"function,omitzero"`
}

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

func (t *ToolCall) To(out *genai.ToolCall) {
	out.ID = t.ID
	out.Name = t.Function.Name
	out.Arguments = t.Function.Arguments
}

type ChatResponse struct {
	Choices []struct {
		FinishReason FinishReason    `json:"finish_reason"`
		Index        int64           `json:"index"`
		Message      MessageResponse `json:"message"`
		Logprobs     struct{}        `json:"logprobs"`
	} `json:"choices"`
	Created        base.Time `json:"created"`
	ID             string    `json:"id"`
	Model          string    `json:"model"`
	Object         string    `json:"object"` // "chat.completion"
	Usage          Usage     `json:"usage"`
	UsageBreakdown struct {
		Models []struct {
			Model string `json:"model"`
			Usage struct {
				QueueTime        float64 `json:"queue_time"`
				PromptTokens     int64   `json:"prompt_tokens"`
				PromptTime       float64 `json:"prompt_time"`
				CompletionTokens int64   `json:"completion_tokens"`
				CompletionTime   float64 `json:"completion_time"`
				TotalTokens      int64   `json:"total_tokens"`
				TotalTime        float64 `json:"total_time"`
			} `json:"usage"`
		} `json:"models"`
	} `json:"usage_breakdown"`
	SystemFingerprint string      `json:"system_fingerprint"`
	ServiceTier       ServiceTier `json:"service_tier"`
	Xgroq             struct {
		ID string `json:"id"`
	} `json:"x_groq"`
}

func (c *ChatResponse) ToResult() (genai.Result, error) {
	out := genai.Result{
		// At the moment, Groq does not support cached tokens.
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

type FinishReason string

const (
	FinishStop          FinishReason = "stop"
	FinishLength        FinishReason = "length"
	FinishToolCalls     FinishReason = "tool_calls"
	FinishContentFilter FinishReason = "content_filter"
)

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

type Usage struct {
	QueueTime        float64 `json:"queue_time"`
	PromptTokens     int64   `json:"prompt_tokens"`
	PromptTime       float64 `json:"prompt_time"`
	CompletionTokens int64   `json:"completion_tokens"`
	CompletionTime   float64 `json:"completion_time"`
	TotalTokens      int64   `json:"total_tokens"`
	TotalTime        float64 `json:"total_time"`
}

type MessageResponse struct {
	Role      string     `json:"role"`
	Reasoning string     `json:"reasoning"`
	Content   string     `json:"content"`
	ToolCalls []ToolCall `json:"tool_calls"`
}

func (m *MessageResponse) To(out *genai.Message) error {
	if m.Reasoning != "" {
		out.Replies = append(out.Replies, genai.Reply{Reasoning: m.Reasoning})
	}
	if m.Content != "" {
		out.Replies = append(out.Replies, genai.Reply{Text: m.Content})
	}
	for i := range m.ToolCalls {
		out.Replies = append(out.Replies, genai.Reply{})
		m.ToolCalls[i].To(&out.Replies[len(out.Replies)-1].ToolCall)
	}
	return nil
}

type ChatStreamChunkResponse struct {
	ID                string    `json:"id"`
	Object            string    `json:"object"`
	Created           base.Time `json:"created"`
	Model             string    `json:"model"`
	SystemFingerprint string    `json:"system_fingerprint"`
	Choices           []struct {
		Index int64 `json:"index"`
		Delta struct {
			Role      string     `json:"role"`
			Content   string     `json:"content"`
			Reasoning string     `json:"reasoning"`
			Channel   string     `json:"channel"` // "analysis"
			ToolCalls []ToolCall `json:"tool_calls"`
		} `json:"delta"`
		Logprobs     struct{}     `json:"logprobs"`
		FinishReason FinishReason `json:"finish_reason"`
	} `json:"choices"`
	Xgroq struct {
		ID    string `json:"id"`
		Usage Usage  `json:"usage"`
	} `json:"x_groq"`
}

type Model struct {
	ID                  string    `json:"id"`
	Object              string    `json:"object"`
	Created             base.Time `json:"created"`
	OwnedBy             string    `json:"owned_by"`
	Active              bool      `json:"active"`
	ContextWindow       int64     `json:"context_window"`
	PublicApps          []string  `json:"public_apps"`
	MaxCompletionTokens int64     `json:"max_completion_tokens"`
}

func (m *Model) GetID() string {
	return m.ID
}

func (m *Model) String() string {
	suffix := ""
	if !m.Active {
		suffix = " (inactive)"
	}
	return fmt.Sprintf("%s (%s) Context: %d/%d%s", m.ID, m.Created.AsTime().Format("2006-01-02"), m.ContextWindow, m.MaxCompletionTokens, suffix)
}

func (m *Model) Context() int64 {
	return m.ContextWindow
}

// ModelsResponse represents the response structure for Groq models listing
type ModelsResponse struct {
	Object string  `json:"object"` // list
	Data   []Model `json:"data"`
}

// ToModels converts Groq models to genai.Model interfaces
func (r *ModelsResponse) ToModels() []genai.Model {
	models := make([]genai.Model, len(r.Data))
	for i := range r.Data {
		models[i] = &r.Data[i]
	}
	return models
}

//

type ErrorResponse struct {
	ErrorVal struct {
		Message          string `json:"message"`
		Type             string `json:"type"`
		Code             string `json:"code"`
		FailedGeneration string `json:"failed_generation"`
		StatusCode       int64  `json:"status_code"`
	} `json:"error"`
}

func (er *ErrorResponse) Error() string {
	suffix := ""
	if er.ErrorVal.FailedGeneration != "" {
		suffix = fmt.Sprintf("failed generation: %q", er.ErrorVal.FailedGeneration)
	}
	return fmt.Sprintf("%s (%s): %s%s", er.ErrorVal.Code, er.ErrorVal.Type, er.ErrorVal.Message, suffix)
}

func (er *ErrorResponse) IsAPIError() bool {
	return true
}

// Client implements genai.Provider.
type Client struct {
	impl base.Provider[*ErrorResponse, *ChatRequest, *ChatResponse, ChatStreamChunkResponse]
}

// New creates a new client to talk to the Groq platform API.
//
// If opts.APIKey is not provided, it tries to load it from the GROQ_API_KEY environment variable.
// If none is found, it will still return a client coupled with an base.ErrAPIKeyRequired error.
// Get your API key at https://console.groq.com/keys
//
// To use multiple models, create multiple clients.
// Use one of the model from https://console.groq.com/dashboard/limits or https://console.groq.com/docs/models
//
// wrapper optionally wraps the HTTP transport. Useful for HTTP recording and playback, or to tweak HTTP
// retries, or to throttle outgoing requests.
//
// Tool use requires the use of a model that supports it.
// https://console.groq.com/docs/tool-use
func New(ctx context.Context, opts *genai.ProviderOptions, wrapper func(http.RoundTripper) http.RoundTripper) (*Client, error) {
	if err := opts.Validate(); err != nil {
		return nil, err
	}
	if opts.AccountID != "" {
		return nil, errors.New("unexpected option AccountID")
	}
	if opts.Remote != "" {
		return nil, errors.New("unexpected option Remote")
	}
	apiKey := opts.APIKey
	const apiKeyURL = "https://console.groq.com/keys"
	var err error
	if apiKey == "" {
		if apiKey = os.Getenv("GROQ_API_KEY"); apiKey == "" {
			err = &base.ErrAPIKeyRequired{EnvVar: "GROQ_API_KEY", URL: apiKeyURL}
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
	c := &Client{
		impl: base.Provider[*ErrorResponse, *ChatRequest, *ChatResponse, ChatStreamChunkResponse]{
			GenSyncURL:           "https://api.groq.com/openai/v1/chat/completions",
			ProcessStreamPackets: processStreamPackets,
			PreloadedModels:      opts.PreloadedModels,
			ProcessHeaders:       processHeaders,
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

// selectBestTextModel selects the most appropriate model based on the preference (cheap, good, or SOTA).
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
	for _, mdl := range mdls {
		m := mdl.(*Model)
		// This is meh.
		if cheap {
			if strings.HasSuffix(m.ID, "instant") {
				selectedModel = m.ID
			}
		} else if good {
			if strings.Contains(m.ID, "maverick") {
				selectedModel = m.ID
			}
		} else {
			if strings.HasPrefix(m.ID, "qwen") {
				selectedModel = m.ID
			}
		}
	}
	if selectedModel == "" {
		return "", errors.New("failed to find a model automatically")
	}
	return selectedModel, nil
}

// Name implements genai.Provider.
//
// It returns the name of the provider.
func (c *Client) Name() string {
	return "groq"
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
	// https://console.groq.com/docs/api-reference#models-list
	var resp ModelsResponse
	if err := c.impl.DoRequest(ctx, "GET", "https://api.groq.com/openai/v1/models", nil, &resp); err != nil {
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
	for pkt := range ch {
		if len(pkt.Choices) != 1 {
			continue
		}
		if pkt.Xgroq.Usage.TotalTokens != 0 {
			result.Usage.InputTokens = pkt.Xgroq.Usage.PromptTokens
			result.Usage.OutputTokens = pkt.Xgroq.Usage.CompletionTokens
			result.Usage.TotalTokens = pkt.Xgroq.Usage.TotalTokens
			result.Usage.FinishReason = pkt.Choices[0].FinishReason.ToFinishReason()
		}
		switch role := pkt.Choices[0].Delta.Role; role {
		case "assistant", "":
		default:
			return fmt.Errorf("unexpected role %q", role)
		}
		if len(pkt.Choices[0].Delta.ToolCalls) > 1 {
			return errors.New("implement multiple tool calls")
		}
		f := genai.ReplyFragment{
			TextFragment:      pkt.Choices[0].Delta.Content,
			ReasoningFragment: pkt.Choices[0].Delta.Reasoning,
		}
		if len(pkt.Choices[0].Delta.ToolCalls) == 1 {
			pkt.Choices[0].Delta.ToolCalls[0].To(&f.ToolCall)
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
	var limits []genai.RateLimit
	requestsLimit, _ := strconv.ParseInt(h.Get("X-Ratelimit-Limit-Requests"), 10, 64)
	requestsRemaining, _ := strconv.ParseInt(h.Get("X-Ratelimit-Remaining-Requests"), 10, 64)
	requestsReset, _ := time.ParseDuration(h.Get("X-Ratelimit-Reset-Requests"))

	tokensLimit, _ := strconv.ParseInt(h.Get("X-Ratelimit-Limit-Tokens"), 10, 64)
	tokensRemaining, _ := strconv.ParseInt(h.Get("X-Ratelimit-Remaining-Tokens"), 10, 64)
	tokensReset, _ := time.ParseDuration(h.Get("X-Ratelimit-Reset-Tokens"))

	if requestsLimit > 0 {
		limits = append(limits, genai.RateLimit{
			Type:      genai.Requests,
			Period:    genai.PerOther,
			Limit:     requestsLimit,
			Remaining: requestsRemaining,
			Reset:     time.Now().Add(requestsReset).Round(10 * time.Millisecond),
		})
	}
	if tokensLimit > 0 {
		limits = append(limits, genai.RateLimit{
			Type:      genai.Tokens,
			Period:    genai.PerOther,
			Limit:     tokensLimit,
			Remaining: tokensRemaining,
			Reset:     time.Now().Add(tokensReset).Round(10 * time.Millisecond),
		})
	}
	return limits
}

var _ genai.Provider = &Client{}
