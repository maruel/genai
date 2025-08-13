// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Package cerebras implements a client for the Cerebras API.
//
// It is described at https://inference-docs.cerebras.ai/api-reference/
package cerebras

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"os"
	"reflect"
	"strconv"
	"strings"
	"time"

	"github.com/invopop/jsonschema"
	"github.com/maruel/genai"
	"github.com/maruel/genai/base"
	"github.com/maruel/genai/internal"
	"github.com/maruel/httpjson"
	"github.com/maruel/roundtrippers"
)

// Scoreboard for Cerebras.
//
// # Warnings
//
//   - qwen-3-32b has broken FinishReason when streaming, and doesn't support tool calling with streaming.
//   - Cerebras doesn't support images yet even if models could.
//     https://discord.com/channels/1085960591052644463/1376887536072527982/1376887536072527982
//   - Tool calling is flaky with all models.
//   - Most models are quantized to unspecified level: https://discord.com/channels/1085960591052644463/1085960592050896937/1372105565655928864
//   - qwen-3-32b is not quantized: https://discord.com/channels/1085960591052644463/1085960592050896937/1374399258890997830
//   - Free tier has limited context: https://inference-docs.cerebras.ai/support/pricing
var Scoreboard = genai.Scoreboard{
	Country:      "US",
	DashboardURL: "https://cloud.cerebras.ai",
	Scenarios: []genai.Scenario{
		{
			// "llama-3.1-8b" works too but the ListModels() API returns the malformed string.
			Models: []string{"llama3.1-8b", "llama-3.3-70b"},
			In:     map[genai.Modality]genai.ModalCapability{genai.ModalityText: {Inline: true}},
			Out:    map[genai.Modality]genai.ModalCapability{genai.ModalityText: {Inline: true}},
			GenSync: &genai.FunctionalityText{
				ReportRateLimits: true,
				Tools:            genai.Flaky,
				BiasedTool:       genai.Flaky,
				IndecisiveTool:   genai.Flaky,
				JSON:             true,
				JSONSchema:       true,
				Seed:             true,
				TopLogprobs:      true,
			},
			GenStream: &genai.FunctionalityText{
				ReportRateLimits: true,
				Tools:            genai.Flaky,
				BiasedTool:       genai.Flaky,
				IndecisiveTool:   genai.Flaky,
				JSON:             true,
				JSONSchema:       true,
				Seed:             true,
				TopLogprobs:      true,
			},
		},
		{
			Models:             []string{"qwen-3-32b"},
			Thinking:           true,
			ThinkingTokenStart: "<think>",
			ThinkingTokenEnd:   "\n</think>\n",
			In:                 map[genai.Modality]genai.ModalCapability{genai.ModalityText: {Inline: true}},
			Out:                map[genai.Modality]genai.ModalCapability{genai.ModalityText: {Inline: true}},
			GenSync: &genai.FunctionalityText{
				ReportRateLimits: true,
				Tools:            genai.Flaky,
				BiasedTool:       genai.True,
				JSON:             true,
				JSONSchema:       true,
				Seed:             true,
				TopLogprobs:      true,
			},
			GenStream: &genai.FunctionalityText{
				ReportRateLimits: true,
				Seed:             true,
				TopLogprobs:      true,
			},
		},
		{
			// Llama-4 scout supports genai.ModalityImage but Cerebras doesn't support this yet.
			// This may change in the future.
			Models: []string{"llama-4-scout-17b-16e-instruct"},
			In:     map[genai.Modality]genai.ModalCapability{genai.ModalityText: {Inline: true}},
			Out:    map[genai.Modality]genai.ModalCapability{genai.ModalityText: {Inline: true}},
			GenSync: &genai.FunctionalityText{
				ReportRateLimits: true,
				Tools:            genai.Flaky,
				BiasedTool:       genai.True,
				JSON:             true,
				JSONSchema:       true,
				Seed:             true,
				TopLogprobs:      true,
			},
			GenStream: &genai.FunctionalityText{
				ReportRateLimits: true,
				Tools:            genai.Flaky,
				BiasedTool:       genai.True,
				JSON:             true,
				JSONSchema:       true,
				Seed:             true,
				TopLogprobs:      true,
			},
		},
		{
			Models: []string{
				"deepseek-r1-distill-llama-70b",
				"gpt-oss-120b",
				"llama-4-maverick-17b-128e-instruct",
				"qwen-3-235b-a22b-instruct-2507",
				"qwen-3-coder-480b",
			},
		},
		{
			Models: []string{
				"qwen-3-235b-a22b-thinking-2507",
			},
			Thinking: true,
		},
	},
}

// Official python client: https://github.com/Cerebras/cerebras-cloud-sdk-python
//
// CompletionsResource.create() at
// https://github.com/Cerebras/cerebras-cloud-sdk-python/blob/main/src/cerebras/cloud/sdk/resources/chat/completions.py

// ChatRequest is documented at https://inference-docs.cerebras.ai/api-reference/chat-completions
type ChatRequest struct {
	Model               string    `json:"model"`
	Messages            []Message `json:"messages"`
	MaxCompletionTokens int64     `json:"max_completion_tokens,omitzero"` // Includes reasoning tokens
	MinCompletionTokens int64     `json:"min_completion_tokens,omitzero"`
	ResponseFormat      struct {
		// https://inference-docs.cerebras.ai/capabilities/structured-outputs
		Type       string `json:"type"` // "json_object", "json_schema"
		JSONSchema struct {
			Name   string             `json:"name"`
			Schema *jsonschema.Schema `json:"schema"`
			Strict bool               `json:"strict"`
		} `json:"json_schema,omitzero"`
	} `json:"response_format,omitzero"`
	Seed          int64    `json:"seed,omitzero"`
	Stop          []string `json:"stop,omitzero"` // Up to 4 sequences
	Stream        bool     `json:"stream,omitzero"`
	StreamOptions struct {
		IncludeUsage bool `json:"include_usage,omitzero"` // Isn't necessary.
	} `json:"stream_options,omitzero"`
	Temperature       float64           `json:"temperature,omitzero"`
	TopP              float64           `json:"top_p,omitzero"`       // [0, 1.0]
	ToolChoice        string            `json:"tool_choice,omitzero"` // "none", "auto", "required" or a struct {"type": "function", "function": {"name": "my_function"}}
	Tools             []Tool            `json:"tools,omitzero"`
	ParallelToolCalls bool              `json:"parallel_tool_calls,omitzero"`
	FrequencyPenalty  float64           `json:"frequency_penalty,omitzero"`
	LogitBias         map[int64]float64 `json:"logit_bias,omitzero"`       // Token bias [-100, 100]
	N                 int64             `json:"n,omitzero"`                // Number of choices
	ServiceTier       complex128        `json:"service_tier,omitzero"`     // "auto", "default"
	PresencePenalty   float64           `json:"presence_penalty,omitzero"` // [-2, 2]
	User              string            `json:"user,omitzero"`             // End user ID to help identify abuse
	Logprobs          bool              `json:"logprobs,omitzero"`         // Whether to return log probabilities
	TopLogprobs       int64             `json:"top_logprobs,omitzero"`     // [0, 20]
}

// Init initializes the provider specific completion request with the generic completion request.
func (c *ChatRequest) Init(msgs genai.Messages, opts genai.Options, model string) error {
	c.Model = model
	var errs []error
	var unsupported []string
	sp := ""
	if opts != nil {
		switch v := opts.(type) {
		case *genai.OptionsText:
			c.MaxCompletionTokens = v.MaxTokens
			c.Temperature = v.Temperature
			c.TopP = v.TopP
			sp = v.SystemPrompt
			c.Seed = v.Seed
			if v.TopLogprobs > 0 {
				c.TopLogprobs = v.TopLogprobs
				c.Logprobs = true
			}
			if v.TopK != 0 {
				unsupported = append(unsupported, "TopK")
			}
			c.Stop = v.Stop
			if v.DecodeAs != nil {
				c.ResponseFormat.Type = "json_schema"
				c.ResponseFormat.JSONSchema.Schema = internal.JSONSchemaFor(reflect.TypeOf(v.DecodeAs))
				c.ResponseFormat.JSONSchema.Strict = true
			} else if v.ReplyAsJSON {
				c.ResponseFormat.Type = "json_object"
			}
			if len(v.Tools) != 0 {
				switch v.ToolCallRequest {
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
		default:
			errs = append(errs, fmt.Errorf("unsupported options type %T", opts))
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
					errs = append(errs, fmt.Errorf("message %d, tool result %d: %w", i, j, err))
				} else {
					c.Messages = append(c.Messages, newMsg)
				}
			}
		} else {
			var newMsg Message
			if err := newMsg.From(&msgs[i]); err != nil {
				errs = append(errs, fmt.Errorf("message %d: %w", i, err))
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

// Message is completely undocumented as of May 2025.
// https://inference-docs.cerebras.ai/api-reference/chat-completions
//
// https://discord.com/channels/1085960591052644463/1345047923339296819/1348990530956034058
type Message struct {
	Role       string     `json:"role,omitzero"` // "system", "assistant", "user"
	Content    Contents   `json:"content,omitzero"`
	ToolCalls  []ToolCall `json:"tool_calls,omitzero"`
	ToolCallID string     `json:"tool_call_id,omitzero"`
	Name       string     `json:"name,omitzero"` // Tool call name.
}

type Content struct {
	Type ContentType `json:"type,omitzero"`
	Text string      `json:"text,omitzero"`
}

type ContentType string

const (
	ContentText ContentType = "text"
)

// Contents represents a slice of Content with custom unmarshalling to handle
// both string and Content struct types.
type Contents []Content

func (c *Contents) MarshalJSON() ([]byte, error) {
	// If there's only one content and it's a string, marshal as a string.
	if len(*c) == 1 && (*c)[0].Type == ContentText {
		return json.Marshal((*c)[0].Text)
	}
	onlyText := true
	for i := range *c {
		if (*c)[i].Type != ContentText {
			onlyText = false
		}
	}
	if onlyText {
		// Merge everything together as a string. This is only needed for a few models like qwen-3-32b and
		// qwen-3-235b-a22b.
		b := strings.Builder{}
		for i := range *c {
			b.WriteString((*c)[i].Text)
			b.WriteString("\n")
		}
		return json.Marshal(b.String())
	}
	// If there's many contents, marshal as an array of Content.
	return json.Marshal(([]Content)(*c))
}

// UnmarshalJSON implements custom unmarshalling for Contents type
// to handle cases where content could be a string or Content struct.
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

// From converts from a genai.Message to a Message.
func (m *Message) From(in *genai.Message) error {
	switch in.Role {
	case genai.User, genai.Assistant:
		m.Role = string(in.Role)
	case genai.Computer:
		fallthrough
	default:
		return fmt.Errorf("unsupported role %q", in.Role)
	}
	if len(in.Contents) > 0 {
		m.Content = make([]Content, 0, len(in.Contents))
		for i := range in.Contents {
			if in.Contents[i].Text != "" {
				m.Content = append(m.Content, Content{
					Type: ContentText,
					Text: in.Contents[i].Text,
				})
			} else if in.Contents[i].Thinking != "" {
				// Ignore
			} else if in.Contents[i].Document != nil {
				// Check if this is a text/plain document
				mimeType, data, err := in.Contents[i].ReadDocument(10 * 1024 * 1024)
				if err != nil {
					return fmt.Errorf("failed to read document: %w", err)
				}
				if strings.HasPrefix(mimeType, "text/plain") {
					if in.Contents[i].URL != "" {
						return errors.New("text/plain documents must be provided inline, not as a URL")
					}
					m.Content = append(m.Content, Content{
						Type: ContentText,
						Text: string(data),
					})
				} else {
					return fmt.Errorf("cerebras only supports text/plain documents, got %s", mimeType)
				}
			} else {
				// Cerebras doesn't support other document types.
				return fmt.Errorf("unsupported content type %#v", in.Contents[i])
			}
		}
	}
	if len(in.ToolCalls) != 0 {
		m.ToolCalls = make([]ToolCall, len(in.ToolCalls))
		for i := range in.ToolCalls {
			m.ToolCalls[i].From(&in.ToolCalls[i])
		}
	}
	if len(in.ToolCallResults) != 0 {
		if len(in.Contents) != 0 || len(in.ToolCalls) != 0 {
			// This could be worked around.
			return fmt.Errorf("can't have tool call result along content or tool calls")
		}
		if len(in.ToolCallResults) != 1 {
			// This should not happen since ChatRequest.Init() works around this.
			return fmt.Errorf("can't have more than one tool call result at a time")
		}
		// Process only the first tool call result in this method.
		// The Init method handles multiple tool call results by creating multiple messages.
		m.Role = "tool"
		m.Content = Contents{{Type: ContentText, Text: in.ToolCallResults[0].Result}}
		m.ToolCallID = in.ToolCallResults[0].ID
		m.Name = in.ToolCallResults[0].Name
	}
	return nil
}

func (m *Message) To(out *genai.Message) error {
	switch role := m.Role; role {
	case "user", "assistant":
		out.Role = genai.Role(role)
	default:
		return fmt.Errorf("unsupported role %q", role)
	}
	if len(m.ToolCalls) != 0 {
		out.ToolCalls = make([]genai.ToolCall, len(m.ToolCalls))
		for i := range m.ToolCalls {
			m.ToolCalls[i].To(&out.ToolCalls[i])
		}
	}
	if len(m.Content) != 0 {
		out.Contents = make([]genai.Content, len(m.Content))
		for i, content := range m.Content {
			if content.Type == ContentText {
				out.Contents[i] = genai.Content{Text: content.Text}
			}
		}
	}
	return nil
}

type Tool struct {
	Type     string `json:"type"` // "function"
	Function struct {
		Name        string             `json:"name"`
		Description string             `json:"description"`
		Parameters  *jsonschema.Schema `json:"parameters"`
	} `json:"function"`
}

type ToolCall struct {
	Type     string `json:"type,omitzero"` // "function"
	ID       string `json:"id,omitzero"`
	Index    int64  `json:"index,omitzero"`
	Function struct {
		Name      string `json:"name,omitzero"`
		Arguments string `json:"arguments,omitzero"`
	} `json:"function,omitzero"`
}

func (t *ToolCall) From(in *genai.ToolCall) {
	t.Type = "function"
	t.ID = in.ID
	t.Index = 0 // Unsure.
	t.Function.Name = in.Name
	t.Function.Arguments = in.Arguments
}

func (t *ToolCall) To(out *genai.ToolCall) {
	out.ID = t.ID
	out.Name = t.Function.Name
	out.Arguments = t.Function.Arguments
}

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
	Usage    Usage `json:"usage"`
	TimeInfo struct {
		QueueTime  float64   `json:"queue_time"`      // In seconds
		PromptTime float64   `json:"prompt_time"`     // In seconds
		ChatTime   float64   `json:"completion_time"` // In seconds
		TotalTime  float64   `json:"total_time"`      // In seconds
		Created    base.Time `json:"created"`
	} `json:"time_info"`
}

func (c *ChatResponse) ToResult() (genai.Result, error) {
	out := genai.Result{
		// At the moment, Cerebras doesn't support cached tokens.
		Usage: genai.Usage{
			InputTokens:       c.Usage.PromptTokens,
			InputCachedTokens: c.Usage.PromptTokensDetails.CachedTokens,
			OutputTokens:      c.Usage.CompletionTokens,
			TotalTokens:       c.Usage.TotalTokens,
		},
	}
	if len(c.Choices) != 1 {
		return out, fmt.Errorf("expected 1 choice, got %#v", c.Choices)
	}
	out.FinishReason = c.Choices[0].FinishReason.ToFinishReason()
	err := c.Choices[0].Message.To(&out.Message)
	if len(out.ToolCalls) != 0 && out.FinishReason == genai.FinishedStop {
		// Lie for the benefit of everyone.
		out.FinishReason = genai.FinishedToolCalls
	}
	out.Logprobs = c.Choices[0].Logprobs.To()
	return out, err
}

type FinishReason string

const (
	FinishStop          FinishReason = "stop"
	FinishToolCalls     FinishReason = "tool_calls"
	FinishLength        FinishReason = "length"
	FinishContentFilter FinishReason = "content_filter"
)

func (f FinishReason) ToFinishReason() genai.FinishReason {
	switch f {
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

type ChatStreamChunkResponse struct {
	ID                string    `json:"id"`
	Model             string    `json:"model"`
	Object            string    `json:"object"`
	SystemFingerprint string    `json:"system_fingerprint"`
	Created           base.Time `json:"created"`
	Choices           []struct {
		Delta struct {
			Role      string     `json:"role"`
			Content   Contents   `json:"content"`
			ToolCalls []ToolCall `json:"tool_calls"`
		} `json:"delta"`
		Index        int64        `json:"index"`
		FinishReason FinishReason `json:"finish_reason"`
		Logprobs     Logprobs     `json:"logprobs"`
	} `json:"choices"`
	Usage    Usage `json:"usage"`
	TimeInfo struct {
		QueueTime  float64   `json:"queue_time"`
		PromptTime float64   `json:"prompt_time"`
		ChatTime   float64   `json:"completion_time"`
		TotalTime  float64   `json:"total_time"`
		Created    base.Time `json:"created"`
	} `json:"time_info"`
}

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

func (l *Logprobs) To() []genai.Logprobs {
	if len(l.Content) == 0 {
		return nil
	}
	out := make([]genai.Logprobs, 0, len(l.Content))
	for i, c := range l.Content {
		out = append(out, genai.Logprobs{Text: c.Token, Bytes: c.Bytes, Logprob: c.Logprob, TopLogprobs: make([]genai.TopLogprob, 0, len(c.TopLogprobs))})
		for _, tlp := range c.TopLogprobs {
			out[i].TopLogprobs = append(out[i].TopLogprobs, genai.TopLogprob{Text: tlp.Token, Bytes: tlp.Bytes, Logprob: tlp.Logprob})
		}
	}
	return out
}

type Usage struct {
	PromptTokens        int64 `json:"prompt_tokens"`
	CompletionTokens    int64 `json:"completion_tokens"`
	TotalTokens         int64 `json:"total_tokens"`
	PromptTokensDetails struct {
		CachedTokens int64 `json:"cached_tokens"`
	} `json:"prompt_tokens_details"`
}

type Model struct {
	ID      string    `json:"id"`
	Object  string    `json:"object"`
	Created base.Time `json:"created"`
	OwnedBy string    `json:"owned_by"`
}

func (m *Model) GetID() string {
	return m.ID
}

func (m *Model) String() string {
	return fmt.Sprintf("%s (%s)", m.ID, m.Created.AsTime().Format("2006-01-02"))
}

func (m *Model) Context() int64 {
	return 0
}

// ModelsResponse represents the response structure for Cerebras models listing
type ModelsResponse struct {
	Object string  `json:"object"`
	Data   []Model `json:"data"`
}

// ToModels converts Cerebras models to genai.Model interfaces
func (r *ModelsResponse) ToModels() []genai.Model {
	models := make([]genai.Model, len(r.Data))
	for i := range r.Data {
		models[i] = &r.Data[i]
	}
	return models
}

//

type ErrorResponse struct {
	// Either this
	Detail string `json:"detail"`

	// Or this (tool call)
	StatusCode int64 `json:"status_code"`
	ErrorVal   struct {
		Message          string `json:"message"`
		Type             string `json:"type"`
		Param            string `json:"param"`
		Code             string `json:"code"`
		FailedGeneration string `json:"failed_generation"`
	} `json:"error"`

	// Or this
	Message          string `json:"message"`
	Type             string `json:"type"`
	Param            string `json:"param"`
	Code             string `json:"code"`
	FailedGeneration string `json:"failed_generation"`
}

func (er *ErrorResponse) Error() string {
	if er.Detail != "" {
		return er.Detail
	}
	if er.StatusCode != 0 {
		return fmt.Sprintf("%s/%s/%s: %s while generating %q", er.ErrorVal.Type, er.ErrorVal.Param, er.ErrorVal.Code, er.ErrorVal.Message, er.ErrorVal.FailedGeneration)
	}
	if er.FailedGeneration != "" {
		return fmt.Sprintf("%s/%s/%s: %s while generating %q", er.Type, er.Param, er.Code, er.Message, er.FailedGeneration)
	}
	return fmt.Sprintf("%s/%s/%s: %s", er.Type, er.Param, er.Code, er.Message)
}

func (er *ErrorResponse) IsAPIError() bool {
	return true
}

// Client implements genai.ProviderGen and genai.ProviderModel.
type Client struct {
	base.ProviderGen[*ErrorResponse, *ChatRequest, *ChatResponse, ChatStreamChunkResponse]
}

// New creates a new client to talk to the Cerebras platform API.
//
// If opts.APIKey is not provided, it tries to load it from the CEREBRAS_API_KEY environment variable.
// If none is found, it will still return a client coupled with an base.ErrAPIKeyRequired error.
// Get an API key at http://cloud.cerebras.ai/
//
// To use multiple models, create multiple clients.
// Use one of the model from https://cerebras.ai/inference
//
// wrapper optionally wraps the HTTP transport. Useful for HTTP recording and playback, or to tweak HTTP
// retries, or to throttle outgoing requests.
func New(opts *genai.OptionsProvider, wrapper func(http.RoundTripper) http.RoundTripper) (*Client, error) {
	if opts.AccountID != "" {
		return nil, errors.New("unexpected option AccountID")
	}
	if opts.Remote != "" {
		return nil, errors.New("unexpected option Remote")
	}
	apiKey := opts.APIKey
	const apiKeyURL = "https://cloud.cerebras.ai/platform/"
	var err error
	if apiKey == "" {
		if apiKey = os.Getenv("CEREBRAS_API_KEY"); apiKey == "" {
			err = &base.ErrAPIKeyRequired{EnvVar: "CEREBRAS_API_KEY", URL: apiKeyURL}
		}
	}
	model := opts.Model
	if model == "" {
		model = base.PreferredGood
	}
	t := base.DefaultTransport
	if wrapper != nil {
		t = wrapper(t)
	}
	c := &Client{
		ProviderGen: base.ProviderGen[*ErrorResponse, *ChatRequest, *ChatResponse, ChatStreamChunkResponse]{
			Model:                model,
			GenSyncURL:           "https://api.cerebras.ai/v1/chat/completions",
			ProcessStreamPackets: processStreamPackets,
			ProcessHeaders:       processHeaders,
			LieToolCalls:         true,
			Provider: base.Provider[*ErrorResponse]{
				ProviderName: "cerebras",
				APIKeyURL:    apiKeyURL,
				ClientJSON: httpjson.Client{
					Lenient: internal.BeLenient,
					Client: &http.Client{
						Transport: &roundtrippers.Header{
							Header:    http.Header{"Authorization": {"Bearer " + apiKey}},
							Transport: &roundtrippers.RequestID{Transport: t},
						},
					},
				},
			},
		},
	}
	switch model {
	case base.NoModel:
		c.Model = ""
	case base.PreferredCheap, base.PreferredGood, base.PreferredSOTA:
		if err == nil {
			if c.Model, err = c.selectBestModel(context.Background(), model); err != nil {
				return nil, err
			}
		}
	}
	return c, err
}

// selectBestModel selects the most appropriate model based on the preference (cheap, good, or SOTA).
func (c *Client) selectBestModel(ctx context.Context, preference string) (string, error) {
	mdls, err := c.ListModels(ctx)
	if err != nil {
		return "", err
	}
	cheap := preference == base.PreferredCheap
	good := preference == base.PreferredGood
	selectedModel := ""
	var created base.Time
	for _, mdl := range mdls {
		// WARNING: This is fragile and will break in the future.
		m := mdl.(*Model)
		if cheap {
			if strings.HasPrefix(m.ID, "llama3") && (created == 0 || m.Created < created) {
				// For the cheapest, we want the oldest model as it is generally cheaper.
				created = m.Created
				selectedModel = m.ID
			}
		} else if good {
			if strings.HasPrefix(m.ID, "llama-4") && (created == 0 || m.Created > created) {
				// For the greatest, we want the newest model as it is generally better.
				created = m.Created
				selectedModel = m.ID
			}
		} else {
			if strings.HasPrefix(m.ID, "qwen-") && (created == 0 || m.Created > created) {
				// For the greatest, we want the newest model as it is generally better.
				created = m.Created
				selectedModel = m.ID
			}
		}
	}
	if selectedModel == "" {
		return "", errors.New("failed to find a model automatically")
	}
	return selectedModel, nil
}

func (c *Client) Scoreboard() genai.Scoreboard {
	return Scoreboard
}

func (c *Client) ListModels(ctx context.Context) ([]genai.Model, error) {
	// https://inference-docs.cerebras.ai/api-reference/models
	return base.ListModels[*ErrorResponse, *ModelsResponse](ctx, &c.Provider, "https://api.cerebras.ai/v1/models")
}

func processStreamPackets(ch <-chan ChatStreamChunkResponse, chunks chan<- genai.ContentFragment, result *genai.Result) error {
	defer func() {
		// We need to empty the channel to avoid blocking the goroutine.
		for range ch {
		}
	}()
	for pkt := range ch {
		if len(pkt.Choices) != 1 {
			continue
		}
		switch role := pkt.Choices[0].Delta.Role; role {
		case "", "assistant":
		default:
			return fmt.Errorf("unexpected role %q", role)
		}
		if pkt.Usage.TotalTokens != 0 {
			result.InputTokens = pkt.Usage.PromptTokens
			result.InputCachedTokens = pkt.Usage.PromptTokensDetails.CachedTokens
			result.OutputTokens = pkt.Usage.CompletionTokens
			result.TotalTokens = pkt.Usage.TotalTokens
			result.FinishReason = pkt.Choices[0].FinishReason.ToFinishReason()
		}

		for _, nt := range pkt.Choices[0].Delta.ToolCalls {
			f := genai.ContentFragment{ToolCall: genai.ToolCall{
				ID:        nt.ID,
				Name:      nt.Function.Name,
				Arguments: nt.Function.Arguments,
			}}
			if err := result.Accumulate(f); err != nil {
				return err
			}
			chunks <- f
		}
		for _, content := range pkt.Choices[0].Delta.Content {
			switch content.Type {
			case ContentText:
				f := genai.ContentFragment{TextFragment: content.Text}
				if !f.IsZero() {
					if err := result.Accumulate(f); err != nil {
						return err
					}
					chunks <- f
				}
			default:
				return fmt.Errorf("unexpected content type %q", content.Type)
			}
		}
		if len(pkt.Choices[0].Logprobs.Content) != 0 {
			result.Logprobs = append(result.Logprobs, pkt.Choices[0].Logprobs.To()...)
		}
	}
	return nil
}

func processHeaders(h http.Header) []genai.RateLimit {
	requestsLimit, _ := strconv.ParseInt(h.Get("X-Ratelimit-Limit-Requests-Day"), 10, 64)
	requestsRemaining, _ := strconv.ParseInt(h.Get("X-Ratelimit-Remaining-Requests-Day"), 10, 64)
	requestsReset, _ := time.ParseDuration(h.Get("X-Ratelimit-Reset-Requests-Day") + "s")

	tokensLimit, _ := strconv.ParseInt(h.Get("X-Ratelimit-Limit-Tokens-Minute"), 10, 64)
	tokensRemaining, _ := strconv.ParseInt(h.Get("X-Ratelimit-Remaining-Tokens-Minute"), 10, 64)
	tokensReset, _ := time.ParseDuration(h.Get("X-Ratelimit-Reset-Tokens-Minute") + "s")

	var limits []genai.RateLimit
	if requestsLimit > 0 {
		limits = append(limits, genai.RateLimit{
			Type:      genai.Requests,
			Period:    genai.PerDay,
			Limit:     requestsLimit,
			Remaining: requestsRemaining,
			Reset:     time.Now().Add(requestsReset).Round(10 * time.Millisecond),
		})
	}
	if tokensLimit > 0 {
		limits = append(limits, genai.RateLimit{
			Type:      genai.Tokens,
			Period:    genai.PerMinute,
			Limit:     tokensLimit,
			Remaining: tokensRemaining,
			Reset:     time.Now().Add(tokensReset).Round(10 * time.Millisecond),
		})
	}
	return limits
}

var (
	_ genai.Provider           = &Client{}
	_ genai.ProviderGen        = &Client{}
	_ genai.ProviderModel      = &Client{}
	_ genai.ProviderScoreboard = &Client{}
)
