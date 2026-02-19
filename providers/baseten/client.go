// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Package baseten implements a client for the Baseten inference API.
//
// It is described at https://docs.baseten.co/reference/inference-api/chat-completions
//
// Baseten offers an OpenAI-compatible chat completions endpoint hosting a limited
// set of high-performance models (DeepSeek, GLM, Kimi, MiniMax, gpt-oss).
package baseten

import (
	"bytes"
	"cmp"
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

	"github.com/invopop/jsonschema"
	"github.com/maruel/genai"
	"github.com/maruel/genai/base"
	"github.com/maruel/genai/internal"
	"github.com/maruel/genai/scoreboard"
	"github.com/maruel/roundtrippers"
)

//go:embed scoreboard.json
var scoreboardJSON []byte

// Scoreboard for Baseten.
func Scoreboard() scoreboard.Score {
	var s scoreboard.Score
	d := json.NewDecoder(bytes.NewReader(scoreboardJSON))
	d.DisallowUnknownFields()
	if err := d.Decode(&s); err != nil {
		panic(fmt.Errorf("failed to unmarshal scoreboard.json: %w", err))
	}
	return s
}

// ChatRequest is documented at https://docs.baseten.co/api-reference/openai-compatible
type ChatRequest struct {
	Model               string    `json:"model"`
	Messages            []Message `json:"messages"`
	MaxCompletionTokens int64     `json:"max_completion_tokens,omitzero"`
	FrequencyPenalty    float64   `json:"frequency_penalty,omitzero"`
	PresencePenalty     float64   `json:"presence_penalty,omitzero"`
	Logprobs            bool      `json:"logprobs,omitzero"`
	TopLogprobs         int64     `json:"top_logprobs,omitzero"`
	ResponseFormat      struct {
		Type       string `json:"type"` // "json_object", "json_schema"
		JSONSchema struct {
			Name   string             `json:"name"`
			Schema *jsonschema.Schema `json:"schema"`
			Strict bool               `json:"strict"`
		} `json:"json_schema,omitzero"`
	} `json:"response_format,omitzero"`
	Seed          int64    `json:"seed,omitzero"`
	Stop          []string `json:"stop,omitzero"`
	Stream        bool     `json:"stream,omitzero"`
	StreamOptions struct {
		IncludeUsage bool `json:"include_usage,omitzero"`
	} `json:"stream_options,omitzero"`
	Temperature       float64 `json:"temperature,omitzero"`
	TopP              float64 `json:"top_p,omitzero"`
	TopK              int64   `json:"top_k,omitzero"`
	ToolChoice        string  `json:"tool_choice,omitzero"` // "none", "auto", "required"
	Tools             []Tool  `json:"tools,omitzero"`
	ParallelToolCalls bool    `json:"parallel_tool_calls,omitzero"`
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
			c.MaxCompletionTokens = v.MaxTokens
			c.Temperature = v.Temperature
			c.TopP = v.TopP
			c.TopK = v.TopK
			sp = v.SystemPrompt
			if v.TopLogprobs > 0 {
				c.TopLogprobs = v.TopLogprobs
				c.Logprobs = true
			}
			c.Stop = v.Stop
			if v.DecodeAs != nil {
				c.ResponseFormat.Type = "json_schema"
				c.ResponseFormat.JSONSchema.Schema = internal.JSONSchemaFor(reflect.TypeOf(v.DecodeAs))
				c.ResponseFormat.JSONSchema.Strict = true
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
				c.ParallelToolCalls = true
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
		switch {
		case len(msgs[i].ToolCallResults) > 1:
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
		case len(msgs[i].Requests) > 1:
			for j := range msgs[i].Requests {
				msgCopy := msgs[i]
				msgCopy.Requests = []genai.Request{msgs[i].Requests[j]}
				var newMsg Message
				if err := newMsg.From(&msgCopy); err != nil {
					errs = append(errs, fmt.Errorf("message #%d: request #%d: %w", i, j, err))
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

// Message is a provider-specific message.
type Message struct {
	Role             string     `json:"role,omitzero"` // "system", "assistant", "user"
	Content          Contents   `json:"content,omitzero"`
	Reasoning        string     `json:"reasoning,omitzero"`
	ReasoningContent string     `json:"reasoning_content,omitzero"`
	ToolCalls        []ToolCall `json:"tool_calls,omitzero"`
	ToolCallID       string     `json:"tool_call_id,omitzero"`
	Name             string     `json:"name,omitzero"`
}

// From must be called with at most one Request or ToolCallResults.
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
	if len(in.Requests) > 0 {
		m.Content = make([]Content, 1)
		if err := m.Content[0].FromRequest(&in.Requests[0]); err != nil {
			return err
		}
	}
	for i := range in.Replies {
		if !in.Replies[i].ToolCall.IsZero() {
			m.ToolCalls = append(m.ToolCalls, ToolCall{})
			if err := m.ToolCalls[len(m.ToolCalls)-1].From(&in.Replies[i].ToolCall); err != nil {
				return fmt.Errorf("reply #%d: %w", i, err)
			}
			continue
		}
		if in.Replies[i].Reasoning != "" {
			continue
		}
		m.Content = append(m.Content, Content{})
		if err := m.Content[len(m.Content)-1].FromReply(&in.Replies[i]); err != nil {
			return fmt.Errorf("reply #%d: %w", i, err)
		}
	}
	if len(in.ToolCallResults) != 0 {
		m.Content = Contents{{Type: ContentText, Text: in.ToolCallResults[0].Result}}
		m.ToolCallID = in.ToolCallResults[0].ID
		m.Name = in.ToolCallResults[0].Name
	}
	return nil
}

// To converts to the genai equivalent.
func (m *Message) To(out *genai.Message) error {
	out.Replies = make([]genai.Reply, 0, len(m.Content)+len(m.ToolCalls))
	// Some models use "reasoning", others (e.g. GLM) use "reasoning_content".
	if r := cmp.Or(m.Reasoning, m.ReasoningContent); r != "" {
		out.Replies = append(out.Replies, genai.Reply{Reasoning: r})
	}
	for _, content := range m.Content {
		switch content.Type {
		case ContentText:
			out.Replies = append(out.Replies, genai.Reply{Text: content.Text})
		default:
			return &internal.BadError{Err: fmt.Errorf("implement content type %q", content.Type)}
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

// Content is a provider-specific content block.
type Content struct {
	Type ContentType `json:"type,omitzero"`
	Text string      `json:"text,omitzero"`

	// Type == "image_url"
	ImageURL struct {
		Detail string `json:"detail,omitzero"` // "auto", "low", "high"
		URL    string `json:"url,omitzero"`    // URL or base64 encoded image
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
	switch {
	case in.Text != "":
		c.Type = ContentText
		c.Text = in.Text
	case !in.Doc.IsZero():
		mimeType, data, err := in.Doc.Read(10 * 1024 * 1024)
		if err != nil {
			return fmt.Errorf("failed to read document: %w", err)
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
	default:
		return &internal.BadError{Err: errors.New("internal error: unknown Reply type")}
	}
	return nil
}

// ContentType is a provider-specific content type.
type ContentType string

// Content type values.
const (
	ContentText     ContentType = "text"
	ContentImageURL ContentType = "image_url"
)

// Contents represents a slice of Content with custom unmarshalling to handle
// both string and Content struct types.
type Contents []Content

// MarshalJSON implements json.Marshaler.
func (c *Contents) MarshalJSON() ([]byte, error) {
	if len(*c) == 1 && (*c)[0].Type == ContentText {
		return json.Marshal((*c)[0].Text)
	}
	return json.Marshal([]Content(*c))
}

// UnmarshalJSON implements custom unmarshalling for Contents type.
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

	v := Content{}
	d = json.NewDecoder(bytes.NewReader(b))
	if !internal.BeLenient {
		d.DisallowUnknownFields()
	}
	if err := d.Decode(&v); err == nil {
		*c = Contents{v}
		return nil
	}

	s := ""
	if err := json.Unmarshal(b, &s); err != nil {
		return err
	}
	*c = Contents{{Type: ContentText, Text: s}}
	return nil
}

// Tool is a provider-specific tool definition.
type Tool struct {
	Type     string `json:"type"` // "function"
	Function struct {
		Name        string             `json:"name"`
		Description string             `json:"description"`
		Parameters  *jsonschema.Schema `json:"parameters"`
	} `json:"function"`
}

// ToolCall is a provider-specific tool call.
type ToolCall struct {
	Type     string `json:"type,omitzero"` // "function"
	ID       string `json:"id,omitzero"`
	Index    int64  `json:"index,omitzero"`
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

// ChatResponse is the provider-specific chat completion response.
type ChatResponse struct {
	ID                string    `json:"id"`
	Model             string    `json:"model"`
	Object            string    `json:"object"` // "chat.completion"
	SystemFingerprint string    `json:"system_fingerprint"`
	Created           base.Time `json:"created"`
	Choices           []struct {
		Index        int64        `json:"index"`
		FinishReason FinishReason `json:"finish_reason"`
		Message      Message      `json:"message"`
		Logprobs     Logprobs     `json:"logprobs"`
	} `json:"choices"`
	Usage Usage `json:"usage"`
}

// ToResult converts the response to a genai.Result.
func (c *ChatResponse) ToResult() (genai.Result, error) {
	out := genai.Result{
		Usage: genai.Usage{
			InputTokens:       c.Usage.PromptTokens,
			InputCachedTokens: c.Usage.PromptTokensDetails.CachedTokens,
			OutputTokens:      c.Usage.CompletionTokens,
			ReasoningTokens:   c.Usage.CompletionTokensDetails.ReasoningTokens,
			TotalTokens:       c.Usage.TotalTokens,
		},
	}
	if len(c.Choices) != 1 {
		return out, fmt.Errorf("expected 1 choice, got %d", len(c.Choices))
	}
	out.Usage.FinishReason = c.Choices[0].FinishReason.ToFinishReason()
	err := c.Choices[0].Message.To(&out.Message)
	if out.Usage.FinishReason == genai.FinishedStop && slices.ContainsFunc(out.Replies, func(r genai.Reply) bool { return !r.ToolCall.IsZero() }) {
		out.Usage.FinishReason = genai.FinishedToolCalls
	}
	out.Logprobs = c.Choices[0].Logprobs.To()
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

// ChatStreamChunkResponse is the provider-specific streaming chat chunk.
type ChatStreamChunkResponse struct {
	ID                string    `json:"id"`
	Model             string    `json:"model"`
	Object            string    `json:"object"`
	ServiceTier       string    `json:"service_tier"`
	SystemFingerprint string    `json:"system_fingerprint"`
	Created           base.Time `json:"created"`
	Choices           []struct {
		Delta struct {
			Role             string          `json:"role"`
			Content          Contents        `json:"content"`
			Reasoning        string          `json:"reasoning"`
			ReasoningContent string          `json:"reasoning_content"`
			FunctionCall     json.RawMessage `json:"function_call"`
			Refusal          json.RawMessage `json:"refusal"`
			ToolCalls        []ToolCall      `json:"tool_calls"`
		} `json:"delta"`
		Index        int64           `json:"index"`
		FinishReason FinishReason    `json:"finish_reason"`
		StopReason   json.RawMessage `json:"stop_reason"`
		Logprobs     Logprobs        `json:"logprobs"`
	} `json:"choices"`
	Usage Usage `json:"usage"`
}

// Logprobs is the provider-specific log probabilities.
type Logprobs struct {
	Content []struct {
		Token       string  `json:"token"`
		Bytes       []byte  `json:"bytes"`
		Logprob     float64 `json:"logprob"`
		TopLogprobs []struct {
			Token   string  `json:"token"`
			Bytes   []byte  `json:"bytes"`
			Logprob float64 `json:"logprob"`
		} `json:"top_logprobs"`
	} `json:"content"`
}

// To converts to the genai equivalent.
func (l *Logprobs) To() [][]genai.Logprob {
	if len(l.Content) == 0 {
		return nil
	}
	out := make([][]genai.Logprob, 0, len(l.Content))
	for _, c := range l.Content {
		lp := make([]genai.Logprob, 1, len(c.TopLogprobs)+1)
		lp[0] = genai.Logprob{Text: c.Token, Logprob: c.Logprob}
		for _, tlp := range c.TopLogprobs {
			lp = append(lp, genai.Logprob{Text: tlp.Token, Logprob: tlp.Logprob})
		}
		out = append(out, lp)
	}
	return out
}

// Usage is the provider-specific token usage.
type Usage struct {
	PromptTokens        int64 `json:"prompt_tokens"`
	CompletionTokens    int64 `json:"completion_tokens"`
	TotalTokens         int64 `json:"total_tokens"`
	PromptTokensDetails struct {
		AudioTokens  int64 `json:"audio_tokens"`
		CachedTokens int64 `json:"cached_tokens"`
	} `json:"prompt_tokens_details"`
	CompletionTokensDetails struct {
		AcceptedPredictionTokens int64 `json:"accepted_prediction_tokens"`
		AudioTokens              int64 `json:"audio_tokens"`
		ReasoningTokens          int64 `json:"reasoning_tokens"`
		RejectedPredictionTokens int64 `json:"rejected_prediction_tokens"`
	} `json:"completion_tokens_details"`
}

// Model is the provider-specific model metadata.
type Model struct {
	ID                          string   `json:"id"`
	Object                      string   `json:"object,omitzero"`
	OwnedBy                     string   `json:"owned_by,omitzero"`
	Name                        string   `json:"name,omitzero"`
	Description                 string   `json:"description,omitzero"`
	ContextLength               int64    `json:"context_length,omitzero"`
	MaxCompletionTokens         int64    `json:"max_completion_tokens,omitzero"`
	Quantization                string   `json:"quantization,omitzero"`
	Created                     int64    `json:"created,omitzero"`
	SupportedFeatures           []string `json:"supported_features,omitzero"`
	SupportedSamplingParameters []string `json:"supported_sampling_parameters,omitzero"`

	// Pricing per token.
	Pricing map[string]string `json:"pricing,omitzero"`
}

// GetID implements genai.Model.
func (m *Model) GetID() string {
	return m.ID
}

func (m *Model) String() string {
	return fmt.Sprintf("%s (context: %d)", m.ID, m.ContextLength)
}

// Context implements genai.Model.
func (m *Model) Context() int64 {
	return m.ContextLength
}

// ErrorResponse is the provider-specific error response.
//
// Baseten can return three error formats:
//   - {"error": "string message"}
//   - {"error": {"message": ..., "type": ..., ...}}
//   - {"message": ..., "type": ..., "code": ...} (e.g. image_url not supported)
type ErrorResponse struct {
	// ErrorString is set when "error" is a plain string.
	ErrorString string
	// ErrorVal is set when "error" is an object or top-level message/type/code fields.
	ErrorVal struct {
		Message string      `json:"message"`
		Type    string      `json:"type"`
		Param   string      `json:"param"`
		Code    json.Number `json:"code"`
	}
	Detail string `json:"detail"`
}

// UnmarshalJSON implements json.Unmarshaler.
func (er *ErrorResponse) UnmarshalJSON(b []byte) error {
	var raw struct {
		Error   json.RawMessage `json:"error"`
		Detail  string          `json:"detail"`
		Message string          `json:"message"`
		Type    string          `json:"type"`
		Code    json.Number     `json:"code"`
	}
	if err := json.Unmarshal(b, &raw); err != nil {
		return err
	}
	er.Detail = raw.Detail
	if len(raw.Error) > 0 {
		// Try string first.
		var s string
		if err := json.Unmarshal(raw.Error, &s); err == nil {
			er.ErrorString = s
			return nil
		}
		// Try struct.
		return json.Unmarshal(raw.Error, &er.ErrorVal)
	}
	// Handle top-level message/type/code format.
	if raw.Message != "" {
		er.ErrorVal.Message = raw.Message
		er.ErrorVal.Type = raw.Type
		er.ErrorVal.Code = raw.Code
	}
	return nil
}

func (er *ErrorResponse) Error() string {
	if er.ErrorString != "" {
		return er.ErrorString
	}
	if er.Detail != "" {
		return er.Detail
	}
	if er.ErrorVal.Message != "" {
		return fmt.Sprintf("%s/%s/%s: %s", er.ErrorVal.Type, er.ErrorVal.Param, er.ErrorVal.Code, er.ErrorVal.Message)
	}
	return "unknown error (empty error response from API)"
}

// IsAPIError implements base.ErrorResponseI.
func (er *ErrorResponse) IsAPIError() bool {
	return true
}

// ModelsResponse represents the response from the /v1/models endpoint.
type ModelsResponse struct {
	Object string  `json:"object"` // list
	Data   []Model `json:"data"`
}

// ToModels converts models to genai.Model interfaces.
func (r *ModelsResponse) ToModels() []genai.Model {
	models := make([]genai.Model, len(r.Data))
	for i := range r.Data {
		models[i] = &r.Data[i]
	}
	return models
}

// Client implements genai.Provider.
type Client struct {
	base.NotImplemented
	impl base.Provider[*ErrorResponse, *ChatRequest, *ChatResponse, ChatStreamChunkResponse]
}

// New creates a new client to talk to the Baseten inference API.
//
// If apiKey is not provided via ProviderOptionAPIKey, it tries to load it from the BASETEN_API_KEY environment
// variable. If none is found, it will still return a client coupled with a base.ErrAPIKeyRequired error.
// Get an API key at https://app.baseten.co/settings/account/api_keys
//
// To use multiple models, create multiple clients.
// Use one of the models from https://docs.baseten.co/development/model-apis/overview
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
	const apiKeyURL = "https://app.baseten.co/settings/account/api_keys"
	var err error
	if apiKey == "" {
		if apiKey = os.Getenv("BASETEN_API_KEY"); apiKey == "" {
			err = &base.ErrAPIKeyRequired{EnvVar: "BASETEN_API_KEY", URL: apiKeyURL}
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
	c := &Client{
		impl: base.Provider[*ErrorResponse, *ChatRequest, *ChatResponse, ChatStreamChunkResponse]{
			GenSyncURL:      "https://inference.baseten.co/v1/chat/completions",
			ProcessStream:   ProcessStream,
			ProcessHeaders:  processHeaders,
			PreloadedModels: preloadedModels,
			LieToolCalls:    true,
			ProviderBase: base.ProviderBase[*ErrorResponse]{
				APIKeyURL: apiKeyURL,
				Lenient:   internal.BeLenient,
				Client: http.Client{
					// Baseten uses "Api-Key" prefix instead of "Bearer".
					Transport: &roundtrippers.Header{
						Header:    http.Header{"Authorization": {"Api-Key " + apiKey}},
						Transport: &roundtrippers.RequestID{Transport: t},
					},
				},
			},
		},
	}
	if err == nil {
		switch model {
		case "":
		case string(genai.ModelCheap), string(genai.ModelGood), string(genai.ModelSOTA):
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

// selectBestTextModel selects the most appropriate model based on the preference (cheap, good, or SOTA).
//
// It reads the scoreboard to find the reference model for the preference, extracts its family prefix,
// then picks the model with the highest version number from the available models.
func (c *Client) selectBestTextModel(ctx context.Context, preference string) (string, error) {
	mdls, err := c.ListModels(ctx)
	if err != nil {
		return "", fmt.Errorf("failed to automatically select the model: %w", err)
	}
	// Find the reference model from the scoreboard.
	s := Scoreboard()
	var ref string
	for _, sc := range s.Scenarios {
		if len(sc.Models) == 0 {
			continue
		}
		switch preference {
		case string(genai.ModelCheap):
			if sc.Cheap {
				ref = sc.Models[0]
			}
		case string(genai.ModelGood):
			if sc.Good {
				ref = sc.Models[0]
			}
		default:
			if sc.SOTA {
				ref = sc.Models[0]
			}
		}
		if ref != "" {
			break
		}
	}
	if ref == "" {
		return "", errors.New("no reference model found in scoreboard")
	}
	// Extract the model family prefix and find the highest version.
	prefix := modelFamilyPrefix(ref)
	selected := ""
	highestVer := -1.0
	for _, mdl := range mdls {
		id := mdl.GetID()
		if !strings.HasPrefix(id, prefix) {
			continue
		}
		if v := parseLeadingFloat(id[len(prefix):]); v > highestVer {
			highestVer = v
			selected = id
		}
	}
	if selected == "" {
		return "", errors.New("failed to find a model automatically")
	}
	return selected, nil
}

// modelFamilyPrefix extracts the model family prefix by stripping the trailing version number.
//
// E.g., "zai-org/GLM-5" -> "zai-org/GLM-", "MiniMaxAI/MiniMax-M2.5" -> "MiniMaxAI/MiniMax-M".
func modelFamilyPrefix(id string) string {
	i := len(id)
	// Skip trailing non-digit suffix (e.g., 'b' in "120b").
	for i > 0 && id[i-1] != '.' && (id[i-1] < '0' || id[i-1] > '9') {
		i--
	}
	// Skip trailing digits and dots (the version number).
	for i > 0 && (id[i-1] >= '0' && id[i-1] <= '9' || id[i-1] == '.') {
		i--
	}
	if i == 0 {
		return id
	}
	return id[:i]
}

// parseLeadingFloat parses a leading float from the string (e.g., "5" -> 5, "4.7" -> 4.7, "120b" -> 120).
func parseLeadingFloat(s string) float64 {
	end := 0
	for end < len(s) && (s[end] >= '0' && s[end] <= '9' || s[end] == '.') {
		end++
	}
	v, _ := strconv.ParseFloat(s[:end], 64)
	return v
}

// Name implements genai.Provider.
func (c *Client) Name() string {
	return "baseten"
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

// HTTPClient returns the HTTP client to fetch results generated by the provider.
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
	// https://docs.baseten.co/reference/inference-api/models
	var resp ModelsResponse
	if err := c.impl.DoRequest(ctx, "GET", "https://inference.baseten.co/v1/models", nil, &resp); err != nil {
		return nil, err
	}
	return resp.ToModels(), nil
}

// ProcessStream converts the raw packets from the streaming API into Reply fragments.
func ProcessStream(chunks iter.Seq[ChatStreamChunkResponse]) (iter.Seq[genai.Reply], func() (genai.Usage, [][]genai.Logprob, error)) {
	var finalErr error
	u := genai.Usage{}
	var l [][]genai.Logprob

	return func(yield func(genai.Reply) bool) {
			pendingToolCall := ToolCall{}
			for pkt := range chunks {
				if len(pkt.Choices) != 1 {
					continue
				}
				switch role := pkt.Choices[0].Delta.Role; role {
				case "", "assistant":
				default:
					finalErr = &internal.BadError{Err: fmt.Errorf("unexpected role %q", role)}
					return
				}
				if pkt.Usage.TotalTokens != 0 {
					u.InputTokens = pkt.Usage.PromptTokens
					u.InputCachedTokens = pkt.Usage.PromptTokensDetails.CachedTokens
					u.OutputTokens = pkt.Usage.CompletionTokens
					u.ReasoningTokens = pkt.Usage.CompletionTokensDetails.ReasoningTokens
					u.TotalTokens = pkt.Usage.TotalTokens
				}
				if fr := pkt.Choices[0].FinishReason; fr != "" {
					u.FinishReason = fr.ToFinishReason()
				}

				for _, nt := range pkt.Choices[0].Delta.ToolCalls {
					if pendingToolCall.ID != "" {
						// Detect continuation: empty ID (OpenAI), same ID (Kimi),
						// or empty name with different ID (MiniMax).
						isContinuation := nt.ID == "" || nt.ID == pendingToolCall.ID || nt.Function.Name == ""
						if isContinuation {
							pendingToolCall.Function.Arguments += nt.Function.Arguments
							if nt.Function.Name != "" {
								pendingToolCall.Function.Name = nt.Function.Name
							}
						} else {
							f := genai.Reply{}
							pendingToolCall.To(&f.ToolCall)
							if !yield(f) {
								return
							}
							pendingToolCall = nt
						}
					} else {
						pendingToolCall = nt
					}
				}
				if r := cmp.Or(pkt.Choices[0].Delta.Reasoning, pkt.Choices[0].Delta.ReasoningContent); r != "" {
					if !yield(genai.Reply{Reasoning: r}) {
						return
					}
				}
				for _, content := range pkt.Choices[0].Delta.Content {
					switch content.Type {
					case ContentText:
						if !yield(genai.Reply{Text: content.Text}) {
							return
						}
					default:
						finalErr = &internal.BadError{Err: fmt.Errorf("implement content type %q", content.Type)}
						return
					}
				}
				if len(pkt.Choices[0].Logprobs.Content) != 0 {
					l = append(l, pkt.Choices[0].Logprobs.To()...)
				}
			}
			if pendingToolCall.ID != "" {
				f := genai.Reply{}
				pendingToolCall.To(&f.ToolCall)
				if !yield(f) {
					return
				}
			}
		}, func() (genai.Usage, [][]genai.Logprob, error) {
			return u, l, finalErr
		}
}

func processHeaders(h http.Header) []genai.RateLimit {
	return nil
}

var _ genai.Provider = &Client{}
