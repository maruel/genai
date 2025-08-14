// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Package groq implements a client for the Groq API.
//
// It is described at https://console.groq.com/docs/api-reference
package groq

import (
	"context"
	"encoding/base64"
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

// Scoreboard for Groq.
//
// # Warnings
//
//   - Thinking models like qwen/qwen3-32b fails with tool calling when streaming. Currently disabled even not
//     streaming in the client code.
//   - No models has consistent tool calling.
var Scoreboard = genai.Scoreboard{
	Country:      "US",
	DashboardURL: "https://console.groq.com/dashboard/usage",
	Scenarios: []genai.Scenario{
		{
			Models: []string{
				// llama-3.1-8b-instant will be both indecisive and biased given its size and quantization. 70b is a
				// bit better but not perfect.
				"llama-3.1-8b-instant",
				"llama-3.3-70b-versatile",
			},
			In:  map[genai.Modality]genai.ModalCapability{genai.ModalityText: {Inline: true}},
			Out: map[genai.Modality]genai.ModalCapability{genai.ModalityText: {Inline: true}},
			GenSync: &genai.FunctionalityText{
				ReportRateLimits: true,
				Tools:            genai.Flaky,
				BiasedTool:       genai.Flaky,
				IndecisiveTool:   genai.Flaky,
				JSON:             true,
				Seed:             true,
			},
			GenStream: &genai.FunctionalityText{
				ReportRateLimits: true,
				Tools:            genai.Flaky,
				BiasedTool:       genai.Flaky,
				IndecisiveTool:   genai.Flaky,
				JSON:             true,
				Seed:             true,
			},
		},
		{
			Models:             []string{"deepseek-r1-distill-llama-70b"},
			Thinking:           true,
			ThinkingTokenStart: "<think>",
			ThinkingTokenEnd:   "\n</think>\n",
			In:                 map[genai.Modality]genai.ModalCapability{genai.ModalityText: {Inline: true}},
			Out:                map[genai.Modality]genai.ModalCapability{genai.ModalityText: {Inline: true}},
			GenSync: &genai.FunctionalityText{
				ReportRateLimits: true,
				Tools:            genai.Flaky,
				BiasedTool:       genai.True,
				IndecisiveTool:   genai.Flaky,
				JSON:             true, // Only when using ReasoningFormat: ReasoningFormatParsed
				Seed:             true,
			},
			GenStream: &genai.FunctionalityText{
				ReportRateLimits: true,
				Tools:            genai.Flaky,
				BiasedTool:       genai.True,
				IndecisiveTool:   genai.Flaky,
				JSON:             true, // Only when using ReasoningFormat: ReasoningFormatParsed
				Seed:             true,
			},
		},
		{
			Models: []string{
				"meta-llama/llama-4-scout-17b-16e-instruct",
				"meta-llama/llama-4-maverick-17b-128e-instruct",
			},
			In: map[genai.Modality]genai.ModalCapability{
				genai.ModalityImage: {
					Inline:           true,
					URL:              true,
					SupportedFormats: []string{"image/gif", "image/jpeg", "image/png", "image/webp"},
				},
				genai.ModalityText: {Inline: true},
			},
			Out: map[genai.Modality]genai.ModalCapability{genai.ModalityText: {Inline: true}},
			GenSync: &genai.FunctionalityText{
				ReportRateLimits: true,
				Tools:            genai.Flaky,
				BiasedTool:       genai.Flaky, // Sometimes tool calling fails.
				IndecisiveTool:   genai.Flaky, // Sometimes tool calling fails.
				JSON:             true,
				Seed:             true,
			},
			GenStream: &genai.FunctionalityText{
				ReportRateLimits: true,
				Tools:            genai.Flaky,
				BiasedTool:       genai.Flaky, // Sometimes tool calling fails.
				IndecisiveTool:   genai.Flaky, // Sometimes tool calling fails.
				JSON:             true,
				Seed:             true,
			},
		},
		{
			Models:             []string{"qwen/qwen3-32b"},
			Thinking:           true,
			ThinkingTokenStart: "<think>",
			ThinkingTokenEnd:   "\n</think>\n",
			In:                 map[genai.Modality]genai.ModalCapability{genai.ModalityText: {Inline: true}},
			Out:                map[genai.Modality]genai.ModalCapability{genai.ModalityText: {Inline: true}},
			GenSync: &genai.FunctionalityText{
				ReportRateLimits: true,
				Tools:            genai.Flaky,
				IndecisiveTool:   genai.Flaky,
				BiasedTool:       genai.Flaky,
				JSON:             true, // Only when using ReasoningFormat: ReasoningFormatParsed
				Seed:             true,
			},
			GenStream: &genai.FunctionalityText{
				ReportRateLimits: true,
				Tools:            genai.Flaky,
				IndecisiveTool:   genai.Flaky,
				BiasedTool:       genai.Flaky,
				JSON:             true, // Only when using ReasoningFormat: ReasoningFormatParsed
				Seed:             true,
			},
		},
		{
			Models: []string{"moonshotai/kimi-k2-instruct"},
			In:     map[genai.Modality]genai.ModalCapability{genai.ModalityText: {Inline: true}},
			Out:    map[genai.Modality]genai.ModalCapability{genai.ModalityText: {Inline: true}},
			GenSync: &genai.FunctionalityText{
				ReportRateLimits: true,
				Tools:            genai.Flaky,
				BiasedTool:       genai.Flaky, // Mostly true but tool calling itself is flaky.
				JSON:             true,
				Seed:             true,
			},
			GenStream: &genai.FunctionalityText{
				ReportRateLimits: true,
				Tools:            genai.Flaky,
				BiasedTool:       genai.Flaky, // Mostly true but tool calling itself is flaky.
				JSON:             true,
				Seed:             true,
			},
		},
		// Deprecated models.
		{
			Models: []string{
				"gemma2-9b-it",
				"llama3-70b-8192",
				"llama3-8b-8192",
			},
		},
		// Unsupported models.
		{
			Models: []string{
				"allam-2-7b",
				"compound-beta-mini",
				"compound-beta",
				"distil-whisper-large-v3-en",
				"meta-llama/llama-guard-4-12b",
				"meta-llama/llama-prompt-guard-2-22m",
				"meta-llama/llama-prompt-guard-2-86m",
				"openai/gpt-oss-20b",
				"openai/gpt-oss-120b",
				"playai-tts-arabic",
				"playai-tts",
				"whisper-large-v3-turbo",
				"whisper-large-v3",
			},
		},
	},
}

// OptionsText is the Groq-specific options.
type OptionsText struct {
	genai.OptionsText

	// ReasoningFormat requests Groq to process the stream on our behalf. It must only be used on thinking
	// models. It is required for thinking models to enable JSON structured output or tool calling.
	ReasoningFormat ReasoningFormat
	// ServiceTier specify the priority.
	ServiceTier ServiceTier
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
func (c *ChatRequest) Init(msgs genai.Messages, opts genai.Options, model string) error {
	c.Model = model
	var errs []error
	var unsupported []string
	sp := ""
	if opts != nil {
		switch v := opts.(type) {
		case *OptionsText:
			unsupported, errs = c.initOptions(&v.OptionsText, model)
			sp = v.SystemPrompt
			c.ServiceTier = v.ServiceTier
			c.ReasoningFormat = v.ReasoningFormat
		case *genai.OptionsText:
			unsupported, errs = c.initOptions(v, model)
			sp = v.SystemPrompt
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

func (c *ChatRequest) initOptions(v *genai.OptionsText, model string) ([]string, []error) {
	var errs []error
	var unsupported []string
	c.MaxChatTokens = v.MaxTokens
	c.Temperature = v.Temperature
	c.TopP = v.TopP
	c.Seed = v.Seed
	if v.TopK != 0 {
		unsupported = append(unsupported, "TopK")
	}
	if v.TopLogprobs != 0 {
		unsupported = append(unsupported, "TopLogprobs")
	}
	c.Stop = v.Stop
	if v.DecodeAs != nil {
		c.ResponseFormat.Type = "json_schema"
		c.ResponseFormat.JSONSchema = internal.JSONSchemaFor(reflect.TypeOf(v.DecodeAs))
		c.ResponseFormat.JSONSchema.Extras = map[string]any{"name": "response"}
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
	switch r := in.Role(); r {
	case "user", "assistant":
		m.Role = r
	case "computer":
		m.Role = "tool"
	default:
		return fmt.Errorf("unsupported role %q", r)
	}
	m.Name = in.User
	if len(in.Request) != 0 {
		m.Content = make(Contents, 0, len(in.Request))
		for i := range in.Request {
			m.Content = append(m.Content, Content{})
			if err := m.Content[len(m.Content)-1].FromRequest(&in.Request[i]); err != nil {
				return fmt.Errorf("request %d: %w", i, err)
			}
		}
	}
	if len(in.Reply) != 0 {
		m.Content = make(Contents, 0, len(in.Reply))
		for i := range in.Reply {
			if in.Reply[i].Thinking != "" {
				// DeepSeek and Qwen recommend against passing reasoning back.
				continue
			}
			m.Content = append(m.Content, Content{})
			if err := m.Content[len(m.Content)-1].FromReply(&in.Reply[i]); err != nil {
				return fmt.Errorf("reply %d: %w", i, err)
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
		if len(in.Request) != 0 || len(in.ToolCalls) != 0 {
			// This could be worked around.
			return fmt.Errorf("can't have tool call result along content or tool calls")
		}
		if len(in.ToolCallResults) != 1 {
			// This should not happen since ChatRequest.Init() works around this.
			return fmt.Errorf("can't have more than one tool call result at a time")
		}
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

func (t *ToolCall) From(in *genai.ToolCall) {
	t.Type = "function"
	t.ID = in.ID
	t.Function.Name = in.Name
	t.Function.Arguments = in.Arguments
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
	out.FinishReason = c.Choices[0].FinishReason.ToFinishReason()
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
	if len(m.ToolCalls) != 0 {
		out.ToolCalls = make([]genai.ToolCall, len(m.ToolCalls))
		for i := range m.ToolCalls {
			m.ToolCalls[i].To(&out.ToolCalls[i])
		}
	}
	if m.Reasoning != "" {
		out.Reply = append(out.Reply, genai.Reply{Thinking: m.Reasoning})
	}
	if m.Content != "" {
		out.Reply = append(out.Reply, genai.Reply{Text: m.Content})
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

// Client implements genai.ProviderGen and genai.ProviderModel.
type Client struct {
	base.ProviderGen[*ErrorResponse, *ChatRequest, *ChatResponse, ChatStreamChunkResponse]
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
func New(opts *genai.OptionsProvider, wrapper func(http.RoundTripper) http.RoundTripper) (*Client, error) {
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
			GenSyncURL:           "https://api.groq.com/openai/v1/chat/completions",
			ProcessStreamPackets: processStreamPackets,
			ProcessHeaders:       processHeaders,
			Provider: base.Provider[*ErrorResponse]{
				ProviderName: "groq",
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

func (c *Client) Scoreboard() genai.Scoreboard {
	return Scoreboard
}

func (c *Client) ListModels(ctx context.Context) ([]genai.Model, error) {
	// https://console.groq.com/docs/api-reference#models-list
	return base.ListModels[*ErrorResponse, *ModelsResponse](ctx, &c.Provider, "https://api.groq.com/openai/v1/models")
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
			result.InputTokens = pkt.Xgroq.Usage.PromptTokens
			result.OutputTokens = pkt.Xgroq.Usage.CompletionTokens
			result.TotalTokens = pkt.Xgroq.Usage.TotalTokens
			result.FinishReason = pkt.Choices[0].FinishReason.ToFinishReason()
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
			TextFragment:     pkt.Choices[0].Delta.Content,
			ThinkingFragment: pkt.Choices[0].Delta.Reasoning,
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

var (
	_ genai.Provider           = &Client{}
	_ genai.ProviderGen        = &Client{}
	_ genai.ProviderModel      = &Client{}
	_ genai.ProviderScoreboard = &Client{}
)
