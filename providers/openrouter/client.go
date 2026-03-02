// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Package openrouter implements a client for the OpenRouter API.
//
// OpenRouter is a unified interface to 300+ LLM models from 60+ providers. It provides
// an OpenAI-compatible API with additional features like provider routing, model fallbacks,
// and transparent pricing.
//
// It is described at https://openrouter.ai/docs/api/reference/overview
package openrouter

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

// Scoreboard for OpenRouter.
func Scoreboard() scoreboard.Score {
	var s scoreboard.Score
	d := json.NewDecoder(bytes.NewReader(scoreboardJSON))
	d.DisallowUnknownFields()
	if err := d.Decode(&s); err != nil {
		panic(fmt.Errorf("failed to unmarshal scoreboard.json: %w", err))
	}
	return s
}

// GenOption is the OpenRouter-specific options.
type GenOption struct {
	// Models specifies fallback models to try if the primary model fails.
	// OpenRouter will try each model in order on failure (rate limit, content moderation, etc).
	Models []string
	// Route specifies the routing strategy. Use "fallback" to try models in order.
	Route Route
	// Provider configures provider routing preferences.
	Provider *ProviderPreferences
}

// Validate implements genai.Validatable.
func (o *GenOption) Validate() error {
	return nil
}

// Route is the routing strategy for multi-model requests.
type Route string

const (
	// RouteFallback tries providers in order, falling back on failure.
	RouteFallback Route = "fallback"
)

// ProviderPreferences configures how OpenRouter selects providers.
type ProviderPreferences struct {
	// Sort prioritizes providers by "price", "throughput", or "latency".
	Sort string `json:"sort,omitzero"`
	// AllowFallbacks enables automatic failover to other providers.
	AllowFallbacks *bool `json:"allow_fallbacks,omitzero"`
	// RequireParameters routes only to providers supporting all requested parameters.
	RequireParameters bool `json:"require_parameters,omitzero"`
	// DataCollection controls data retention. Use "deny" to restrict to non-collecting providers.
	DataCollection string `json:"data_collection,omitzero"`
	// Order specifies an ordered list of provider names to try.
	Order []string `json:"order,omitzero"`
	// Only restricts requests to these providers only.
	Only []string `json:"only,omitzero"`
	// Ignore blacklists these providers.
	Ignore []string `json:"ignore,omitzero"`
	// Quantizations filters by quantization level (int4, int8, fp8, fp16, bf16, fp32).
	Quantizations []string `json:"quantizations,omitzero"`
	// ZDR enables Zero Data Retention mode.
	ZDR bool `json:"zdr,omitzero"`
	// EnforceDistillableText requires text output compatible with distillation.
	EnforceDistillableText bool `json:"enforce_distillable_text,omitzero"`
}

// Reasoning configures reasoning/thinking behavior for supported models.
type Reasoning struct {
	// Effort controls how much effort the model spends on reasoning ("low", "medium", "high").
	Effort string `json:"effort,omitzero"`
	// Summary controls whether to include a summary of reasoning ("auto", "concise", "detailed").
	Summary string `json:"summary,omitzero"`
}

// ChatRequest is documented at https://openrouter.ai/docs/api/api-reference/chat/send-chat-completion-request
type ChatRequest struct {
	FrequencyPenalty    float64   `json:"frequency_penalty,omitzero"` // [-2.0, 2.0]
	Logprobs            bool      `json:"logprobs,omitzero"`
	TopLogprobs         int64     `json:"top_logprobs,omitzero"` // [0, 20]
	MaxTokens           int64     `json:"max_tokens,omitzero"`
	MaxCompletionTokens int64     `json:"max_completion_tokens,omitzero"`
	Messages            []Message `json:"messages"`
	Model               string    `json:"model"`
	Models              []string  `json:"models,omitzero"`
	Modalities          []string  `json:"modalities,omitzero"`
	PresencePenalty     float64   `json:"presence_penalty,omitzero"` // [-2.0, 2.0]
	ResponseFormat      struct {
		Type       string             `json:"type,omitzero"` // "json_object", "json_schema"
		JSONSchema *jsonschema.Schema `json:"json_schema,omitzero"`
	} `json:"response_format,omitzero"`
	Reasoning     *Reasoning           `json:"reasoning,omitzero"`
	Route         Route                `json:"route,omitzero"`
	Provider      *ProviderPreferences `json:"provider,omitzero"`
	Seed          int64                `json:"seed,omitzero"`
	Stop          []string             `json:"stop,omitzero"`
	Stream        bool                 `json:"stream"`
	StreamOptions struct {
		IncludeUsage bool `json:"include_usage,omitzero"`
	} `json:"stream_options,omitzero"`
	Temperature       float64            `json:"temperature,omitzero"` // [0, 2]
	Tools             []Tool             `json:"tools,omitzero"`
	ToolChoice        string             `json:"tool_choice,omitzero"` // "none", "auto", "required"
	ParallelToolCalls *bool              `json:"parallel_tool_calls,omitzero"`
	TopP              float64            `json:"top_p,omitzero"` // [0, 1]
	User              string             `json:"user,omitzero"`
	Metadata          map[string]string  `json:"metadata,omitzero"`
	LogitBias         map[string]float64 `json:"logit_bias,omitzero"`
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
			c.Models = v.Models
			c.Route = v.Route
			c.Provider = v.Provider
		case *genai.GenOptionText:
			unsupported = append(unsupported, c.initOptionsText(v)...)
			sp = v.SystemPrompt
		case *genai.GenOptionTools:
			c.initOptionsTools(v)
		case *genai.GenOptionWeb:
			unsupported = append(unsupported, internal.TypeName(opt))
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
		} else {
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

func (c *ChatRequest) initOptionsText(v *genai.GenOptionText) []string {
	var unsupported []string
	c.MaxTokens = v.MaxTokens
	c.Temperature = v.Temperature
	c.TopP = v.TopP
	if v.TopK != 0 {
		unsupported = append(unsupported, "GenOptionText.TopK")
	}
	if v.TopLogprobs != 0 {
		c.Logprobs = true
		c.TopLogprobs = v.TopLogprobs
	}
	c.Stop = v.Stop
	if v.DecodeAs != nil {
		c.ResponseFormat.Type = "json_schema"
		c.ResponseFormat.JSONSchema = internal.JSONSchemaFor(reflect.TypeOf(v.DecodeAs))
		c.ResponseFormat.JSONSchema.Extras = map[string]any{"name": "response"}
	} else if v.ReplyAsJSON {
		c.ResponseFormat.Type = "json_object"
	}
	return unsupported
}

func (c *ChatRequest) initOptionsTools(v *genai.GenOptionTools) {
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
}

// ReasoningDetail is a structured reasoning step from models that expose chain-of-thought.
//
// Different upstream models use different schemas: OpenAI uses Type+Content,
// Qwen uses Format+Index+Text.
type ReasoningDetail struct {
	Type    string `json:"type,omitzero"`    // "text", "summary", etc. (OpenAI format)
	Content string `json:"content,omitzero"` // The reasoning text (OpenAI format).
	Format  string `json:"format,omitzero"`  // "unknown", etc. (Qwen format)
	Index   int64  `json:"index,omitzero"`   // Segment index (Qwen format).
	Text    string `json:"text,omitzero"`    // The reasoning text (Qwen format).
}

// Annotation is a provider-specific annotation attached to a message.
type Annotation struct {
	Type        string `json:"type,omitzero"` // "url_citation"
	URLCitation struct {
		StartIndex int64  `json:"start_index,omitzero"`
		EndIndex   int64  `json:"end_index,omitzero"`
		Title      string `json:"title,omitzero"`
		URL        string `json:"url,omitzero"`
	} `json:"url_citation,omitzero"`
}

// Message is the OpenRouter message format, compatible with OpenAI.
type Message struct {
	Role             string            `json:"role"`             // "system", "assistant", "user"
	Name             string            `json:"name,omitzero"`    // Optional sender name.
	Content          Contents          `json:"content,omitzero"` // Content is always returned as a string.
	ToolCalls        []ToolCall        `json:"tool_calls,omitzero"`
	ToolCallID       string            `json:"tool_call_id,omitzero"`
	Refusal          *string           `json:"refusal,omitzero"`
	Reasoning        *string           `json:"reasoning,omitzero"`
	ReasoningDetails []ReasoningDetail `json:"reasoning_details,omitzero"`
	Annotations      []Annotation      `json:"annotations,omitzero"`
}

// From converts from a genai.Message to the OpenRouter message format.
//
// Must be called with at most one ToolCallResults.
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

// To converts the OpenRouter message to a genai.Message.
func (m *Message) To(out *genai.Message) error {
	if m.Reasoning != nil && *m.Reasoning != "" {
		out.Replies = append(out.Replies, genai.Reply{Reasoning: *m.Reasoning})
	}
	for _, c := range m.Content {
		switch c.Type {
		case ContentText:
			if c.Text == "" {
				return &internal.BadError{Err: errors.New("empty content text")}
			}
			out.Replies = append(out.Replies, genai.Reply{Text: c.Text})
		default:
			return &internal.BadError{Err: fmt.Errorf("implement content type %q", c.Type)}
		}
	}
	for i := range m.ToolCalls {
		out.Replies = append(out.Replies, genai.Reply{})
		m.ToolCalls[i].To(&out.Replies[len(out.Replies)-1].ToolCall)
	}
	return nil
}

// Contents exists to marshal single content text block as a string.
//
// OpenRouter (OpenAI-compatible) requires this for assistant messages.
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
		Name        string             `json:"name,omitzero"`
		Description string             `json:"description,omitzero"`
		Parameters  *jsonschema.Schema `json:"parameters,omitzero"`
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

// CompletionTokensDetails breaks down completion token usage.
type CompletionTokensDetails struct {
	ReasoningTokens          int64 `json:"reasoning_tokens,omitzero"`
	AudioTokens              int64 `json:"audio_tokens,omitzero"`
	AcceptedPredictionTokens int64 `json:"accepted_prediction_tokens,omitzero"`
	RejectedPredictionTokens int64 `json:"rejected_prediction_tokens,omitzero"`
	ImageTokens              int64 `json:"image_tokens,omitzero"`
}

// PromptTokensDetails breaks down prompt token usage.
type PromptTokensDetails struct {
	CachedTokens     int64 `json:"cached_tokens,omitzero"`
	CacheWriteTokens int64 `json:"cache_write_tokens,omitzero"`
	AudioTokens      int64 `json:"audio_tokens,omitzero"`
	VideoTokens      int64 `json:"video_tokens,omitzero"`
}

// CostDetails breaks down cost components.
type CostDetails struct {
	UpstreamInferenceCost            float64 `json:"upstream_inference_cost,omitzero"`
	UpstreamInferencePromptCost      float64 `json:"upstream_inference_prompt_cost,omitzero"`
	UpstreamInferenceCompletionsCost float64 `json:"upstream_inference_completions_cost,omitzero"`
}

// Usage is the provider-specific token usage.
type Usage struct {
	PromptTokens            int64                   `json:"prompt_tokens"`
	CompletionTokens        int64                   `json:"completion_tokens"`
	TotalTokens             int64                   `json:"total_tokens"`
	Cost                    float64                 `json:"cost,omitzero"` // OpenRouter-specific: actual cost in USD.
	CostDetails             CostDetails             `json:"cost_details,omitzero"`
	CompletionTokensDetails CompletionTokensDetails `json:"completion_tokens_details,omitzero"`
	PromptTokensDetails     PromptTokensDetails     `json:"prompt_tokens_details,omitzero"`
	IsByok                  bool                    `json:"is_byok,omitzero"`
}

// Logprobs is the provider-specific log probabilities.
type Logprobs struct {
	Content []TokenLogprob `json:"content"`
	Refusal []TokenLogprob `json:"refusal,omitzero"`
}

// To converts to the genai equivalent.
func (l *Logprobs) To() [][]genai.Logprob {
	if len(l.Content) == 0 {
		return nil
	}
	out := make([][]genai.Logprob, 0, len(l.Content))
	for _, p := range l.Content {
		lp := make([]genai.Logprob, 1, len(p.TopLogprobs)+1)
		lp[0] = genai.Logprob{Text: p.Token, Logprob: p.Logprob}
		for _, tlp := range p.TopLogprobs {
			lp = append(lp, genai.Logprob{Text: tlp.Token, Logprob: tlp.Logprob})
		}
		out = append(out, lp)
	}
	return out
}

// TokenLogprob is a single token's log probability with alternatives.
type TokenLogprob struct {
	Token       string       `json:"token"`
	Logprob     float64      `json:"logprob"`
	Bytes       []int        `json:"bytes"`
	TopLogprobs []TopLogprob `json:"top_logprobs"`
}

// TopLogprob is an alternative token's log probability.
type TopLogprob struct {
	Token   string  `json:"token"`
	Logprob float64 `json:"logprob"`
	Bytes   []int   `json:"bytes"`
}

// InlineError is an upstream error embedded in an otherwise successful response.
type InlineError struct {
	Message  string         `json:"message,omitzero"`
	Code     any            `json:"code,omitzero"` // Can be string or number.
	Metadata map[string]any `json:"metadata,omitzero"`
}

// ChatResponse is the provider-specific chat completion response.
type ChatResponse struct {
	Choices []struct {
		FinishReason       FinishReason `json:"finish_reason"`
		NativeFinishReason string       `json:"native_finish_reason,omitzero"`
		Index              int64        `json:"index"`
		Message            Message      `json:"message"`
		Logprobs           *Logprobs    `json:"logprobs,omitzero"`
	} `json:"choices"`
	Created           base.Time    `json:"created"`
	ID                string       `json:"id"`
	Model             string       `json:"model"`    // The model that actually served the request.
	Object            string       `json:"object"`   // "chat.completion"
	Provider          string       `json:"provider"` // The provider that served the request.
	Usage             Usage        `json:"usage"`
	SystemFingerprint string       `json:"system_fingerprint"`
	Error             *InlineError `json:"error,omitzero"` // Upstream error on partial failure.
	UserID            string       `json:"user_id,omitzero"`
}

// ToResult converts the response to a genai.Result.
func (c *ChatResponse) ToResult() (genai.Result, error) {
	out := genai.Result{
		Usage: genai.Usage{
			InputTokens:       c.Usage.PromptTokens,
			InputCachedTokens: c.Usage.PromptTokensDetails.CachedTokens,
			ReasoningTokens:   c.Usage.CompletionTokensDetails.ReasoningTokens,
			OutputTokens:      c.Usage.CompletionTokens,
			TotalTokens:       c.Usage.TotalTokens,
		},
	}
	if c.Error != nil {
		return out, fmt.Errorf("upstream error: %s", c.Error.Message)
	}
	if len(c.Choices) != 1 {
		return out, &internal.BadError{Err: fmt.Errorf("server returned an unexpected number of choices, expected 1, got %d", len(c.Choices))}
	}
	out.Usage.FinishReason = c.Choices[0].FinishReason.ToFinishReason()
	if c.Choices[0].Logprobs != nil {
		out.Logprobs = c.Choices[0].Logprobs.To()
	}
	err := c.Choices[0].Message.To(&out.Message)
	return out, err
}

// FinishReason is a provider-specific finish reason.
type FinishReason string

// Finish reason values.
const (
	FinishStop          FinishReason = "stop"
	FinishLength        FinishReason = "length"
	FinishToolCalls     FinishReason = "tool_calls"
	FinishContentFilter FinishReason = "content_filter"
	FinishError         FinishReason = "error"
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
	case FinishError:
		return genai.FinishReason("error")
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
	Object            string    `json:"object"`
	Created           base.Time `json:"created"`
	Model             string    `json:"model"`
	Provider          string    `json:"provider,omitzero"`
	SystemFingerprint string    `json:"system_fingerprint"`
	Choices           []struct {
		Index              int64        `json:"index"`
		Delta              Message      `json:"delta"`
		FinishReason       FinishReason `json:"finish_reason"`
		NativeFinishReason string       `json:"native_finish_reason,omitzero"`
		Logprobs           *Logprobs    `json:"logprobs,omitzero"`
	} `json:"choices"`
	Usage  Usage        `json:"usage,omitzero"`
	Error  *InlineError `json:"error,omitzero"` // Upstream error on partial failure.
	UserID string       `json:"user_id,omitzero"`
}

// Model is the provider-specific model metadata.
type Model struct {
	ID            string       `json:"id"`
	Name          string       `json:"name"`
	Created       base.Time    `json:"created"`
	Description   string       `json:"description"`
	ContextLength int64        `json:"context_length"`
	Pricing       ModelPricing `json:"pricing"`
	Architecture  struct {
		Tokenizer        string   `json:"tokenizer"`
		InstructType     string   `json:"instruct_type"`
		Modality         string   `json:"modality"` // "text->text", "text+image->text", etc.
		InputModalities  []string `json:"input_modalities"`
		OutputModalities []string `json:"output_modalities"`
	} `json:"architecture"`
	TopProvider struct {
		ContextLength       int64 `json:"context_length"`
		MaxCompletionTokens int64 `json:"max_completion_tokens"`
		IsModerated         bool  `json:"is_moderated"`
	} `json:"top_provider"`
	HuggingFaceID       *string        `json:"hugging_face_id"`
	PerRequestLimits    map[string]any `json:"per_request_limits"`
	SupportedParameters []string       `json:"supported_parameters"`
	DefaultParameters   map[string]any `json:"default_parameters"`
	CanonicalSlug       string         `json:"canonical_slug"`
	ExpirationDate      *string        `json:"expiration_date"`
}

// ModelPricing contains the per-token pricing information.
type ModelPricing struct {
	Prompt            string `json:"prompt"`                      // USD per token.
	Completion        string `json:"completion"`                  // USD per token.
	Image             string `json:"image"`                       // USD per image.
	Request           string `json:"request"`                     // USD per request.
	Audio             string `json:"audio,omitzero"`              // USD per audio token.
	InternalReasoning string `json:"internal_reasoning,omitzero"` // USD per reasoning token.
	InputCacheRead    string `json:"input_cache_read,omitzero"`   // USD per cached input token read.
	InputCacheWrite   string `json:"input_cache_write,omitzero"`  // USD per cached input token write.
	WebSearch         string `json:"web_search,omitzero"`         // USD per web search.
}

// GetID implements genai.Model.
func (m *Model) GetID() string {
	return m.ID
}

func (m *Model) String() string {
	return fmt.Sprintf("%s (%s) Context: %d", m.ID, m.Name, m.ContextLength)
}

// Context implements genai.Model.
func (m *Model) Context() int64 {
	return m.ContextLength
}

// ModelsResponse represents the response structure for OpenRouter models listing.
type ModelsResponse struct {
	Data []Model `json:"data"`
}

// ToModels converts OpenRouter models to genai.Model interfaces.
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
		Message  string         `json:"message"`
		Type     string         `json:"type"`
		Code     any            `json:"code"` // Can be string or number.
		Metadata map[string]any `json:"metadata,omitzero"`
	} `json:"error"`
	UserID string `json:"user_id,omitzero"`
}

func (er *ErrorResponse) Error() string {
	return fmt.Sprintf("%v (%s): %s", er.ErrorVal.Code, er.ErrorVal.Type, er.ErrorVal.Message)
}

// IsAPIError implements base.ErrorResponseI.
func (er *ErrorResponse) IsAPIError() bool {
	return true
}

// Client implements genai.Provider for OpenRouter.
type Client struct {
	base.NotImplemented
	impl base.Provider[*ErrorResponse, *ChatRequest, *ChatResponse, ChatStreamChunkResponse]
}

// New creates a new client to talk to the OpenRouter API.
//
// If apiKey is not provided, it tries to load it from the OPENROUTER_API_KEY environment variable.
// If none is found, it will still return a client coupled with a base.ErrAPIKeyRequired error.
// Get your API key at https://openrouter.ai/settings/keys
//
// OpenRouter model IDs use the format "provider/model", e.g. "anthropic/claude-3.5-sonnet".
// See https://openrouter.ai/models for available models.
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
	const apiKeyURL = "https://openrouter.ai/settings/keys"
	var err error
	if apiKey == "" {
		if apiKey = os.Getenv("OPENROUTER_API_KEY"); apiKey == "" {
			err = &base.ErrAPIKeyRequired{EnvVar: "OPENROUTER_API_KEY", URL: apiKeyURL}
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
			GenSyncURL:      "https://openrouter.ai/api/v1/chat/completions",
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
		case string(genai.ModelCheap), string(genai.ModelGood), string(genai.ModelSOTA):
			c.impl.Model = c.selectBestTextModel(model)
			c.impl.OutputModalities = mod
		default:
			c.impl.Model = model
			c.impl.OutputModalities = mod
		}
	}
	return c, err
}

// selectBestTextModel selects the most appropriate model based on the preference.
func (c *Client) selectBestTextModel(preference string) string {
	switch preference {
	case string(genai.ModelCheap):
		return "qwen/qwen3.5-35b-a3b"
	case string(genai.ModelGood):
		return "qwen/qwen3.5-122b-a10b"
	default:
		// SOTA
		return "qwen/qwen3.5-397b-a17b"
	}
}

// Name implements genai.Provider.
func (c *Client) Name() string {
	return "openrouter"
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
	if err := c.impl.DoRequest(ctx, "GET", "https://openrouter.ai/api/v1/models", nil, &resp); err != nil {
		return nil, err
	}
	return resp.ToModels(), nil
}

// ProcessStream converts the raw packets from the streaming API into Reply fragments.
func ProcessStream(chunks iter.Seq[ChatStreamChunkResponse]) (iter.Seq[genai.Reply], func() (genai.Usage, [][]genai.Logprob, error)) {
	var finalErr error
	var logprobs [][]genai.Logprob
	u := genai.Usage{}

	return func(yield func(genai.Reply) bool) {
			pendingToolCall := ToolCall{}
			for pkt := range chunks {
				if len(pkt.Choices) != 1 {
					continue
				}
				if pkt.Usage.TotalTokens != 0 {
					u.InputTokens = pkt.Usage.PromptTokens
					u.InputCachedTokens = pkt.Usage.PromptTokensDetails.CachedTokens
					u.ReasoningTokens = pkt.Usage.CompletionTokensDetails.ReasoningTokens
					u.OutputTokens = pkt.Usage.CompletionTokens
					u.TotalTokens = pkt.Usage.TotalTokens
				}
				if pkt.Choices[0].FinishReason != "" {
					u.FinishReason = pkt.Choices[0].FinishReason.ToFinishReason()
				}
				if pkt.Choices[0].Logprobs != nil {
					logprobs = append(logprobs, pkt.Choices[0].Logprobs.To()...)
				}
				switch role := pkt.Choices[0].Delta.Role; role {
				case "assistant", "":
				default:
					finalErr = &internal.BadError{Err: fmt.Errorf("unexpected role %q", role)}
					return
				}
				f := genai.Reply{}
				for _, c := range pkt.Choices[0].Delta.Content {
					switch c.Type {
					case ContentText:
						f.Text += c.Text
					default:
						finalErr = &internal.BadError{Err: fmt.Errorf("implement content type %q", c.Type)}
						return
					}
				}
				// OpenRouter streams tool call arguments incrementally. Buffer them and
				// yield the complete tool call once a new call starts or the stream ends.
				if len(pkt.Choices[0].Delta.ToolCalls) > 1 {
					finalErr = &internal.BadError{Err: fmt.Errorf("implement multiple tool calls: %#v", pkt)}
					return
				}
				if len(pkt.Choices[0].Delta.ToolCalls) == 1 {
					t := pkt.Choices[0].Delta.ToolCalls[0]
					if t.ID != "" {
						// A new call.
						if pendingToolCall.ID == "" {
							pendingToolCall = t
							if !f.IsZero() {
								if !yield(f) {
									return
								}
							}
							continue
						}
						// Flush previous.
						pendingToolCall.To(&f.ToolCall)
						pendingToolCall = t
					} else if pendingToolCall.ID != "" {
						// Continuation: accumulate arguments.
						pendingToolCall.Function.Arguments += t.Function.Arguments
						if !f.IsZero() {
							finalErr = &internal.BadError{Err: fmt.Errorf("implement tool call with metadata: %#v", pkt)}
							return
						}
						continue
					}
				} else if pendingToolCall.ID != "" {
					// Flush pending tool call.
					pendingToolCall.To(&f.ToolCall)
					pendingToolCall = ToolCall{}
				}
				if !f.IsZero() {
					if !yield(f) {
						return
					}
				}
			}
		}, func() (genai.Usage, [][]genai.Logprob, error) {
			return u, logprobs, finalErr
		}
}

var _ genai.Provider = &Client{}
