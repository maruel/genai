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
	"strings"
	"time"

	"github.com/invopop/jsonschema"
	"github.com/maruel/genai"
	"github.com/maruel/genai/internal"
	"github.com/maruel/httpjson"
	"github.com/maruel/roundtrippers"
)

// Scoreboard for Groq.
//
// # Warnings
//
//   - qwen-qwq-32b fails with tool calling when streaming. Currently disabled even not streaming in the
//     client code.
var Scoreboard = genai.Scoreboard{
	Scenarios: []genai.Scenario{
		{
			In:  []genai.Modality{genai.ModalityText},
			Out: []genai.Modality{genai.ModalityText},

			Models: []string{
				"llama3-8b-8192",
				"gemma2-9b-it",
				"llama-3.1-8b-instant",
				"llama-3.3-70b-versatile",
				"llama3-70b-8192",
				"mistral-saba-24b",
			},
			Chat: genai.Functionality{
				Inline:             true,
				URL:                false,
				Thinking:           false,
				ReportTokenUsage:   true,
				ReportFinishReason: true,
				MaxTokens:          true,
				StopSequence:       true,
				Tools:              true,
				JSON:               true,
				JSONSchema:         false,
			},
			ChatStream: genai.Functionality{
				Inline:             true,
				URL:                false,
				Thinking:           false,
				ReportTokenUsage:   true,
				ReportFinishReason: true,
				MaxTokens:          true,
				StopSequence:       true,
				Tools:              true,
				JSON:               true,
				JSONSchema:         false,
			},
		},
		{
			In:  []genai.Modality{genai.ModalityText},
			Out: []genai.Modality{genai.ModalityText},
			Models: []string{
				"qwen-qwq-32b",
				"deepseek-r1-distill-llama-70b",
			},
			Chat: genai.Functionality{
				Inline:             true,
				URL:                false,
				Thinking:           true,
				ReportTokenUsage:   true,
				ReportFinishReason: true,
				MaxTokens:          true,
				StopSequence:       true,
				Tools:              false,
				JSON:               true,
				JSONSchema:         false,
			},
			ChatStream: genai.Functionality{
				Inline:             true,
				URL:                false,
				Thinking:           true,
				ReportTokenUsage:   true,
				ReportFinishReason: true,
				MaxTokens:          true,
				StopSequence:       true,
				Tools:              false,
				JSON:               true,
				JSONSchema:         false,
			},
		},
		{
			In:  []genai.Modality{genai.ModalityImage, genai.ModalityText},
			Out: []genai.Modality{genai.ModalityText},
			Models: []string{
				"meta-llama/llama-4-scout-17b-16e-instruct",
				"meta-llama/llama-4-maverick-17b-128e-instruct",
			},
			Chat: genai.Functionality{
				Inline:             true,
				URL:                true,
				Thinking:           false,
				ReportTokenUsage:   true,
				ReportFinishReason: true,
				MaxTokens:          true,
				StopSequence:       true,
				Tools:              false,
				JSON:               true,
				JSONSchema:         false,
			},
			ChatStream: genai.Functionality{
				Inline:             true,
				URL:                true,
				Thinking:           false,
				ReportTokenUsage:   true,
				ReportFinishReason: true,
				MaxTokens:          true,
				StopSequence:       true,
				Tools:              true,
				JSON:               true,
				JSONSchema:         false,
			},
		},
	},
}

// TODO: Expose option c.ReasoningFormat through ChatOptions.
// Ref: https://console.groq.com/docs/reasoning/

// ChatOptions is the Groq-specific options.
type ChatOptions struct {
	genai.ChatOptions

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

// https://console.groq.com/docs/api-reference#chat-create
type ChatRequest struct {
	FrequencyPenalty  float64         `json:"frequency_penalty,omitzero"` // [-2.0, 2.0]
	MaxChatTokens     int64           `json:"max_completion_tokens,omitzero"`
	Messages          []Message       `json:"messages"`
	Model             string          `json:"model"`
	ParallelToolCalls bool            `json:"parallel_tool_calls,omitzero"`
	PresencePenalty   float64         `json:"presence_penalty,omitzero"` // [-2.0, 2.0]
	ReasoningFormat   ReasoningFormat `json:"reasoning_format,omitzero"`
	ResponseFormat    struct {
		Type       string         `json:"type,omitzero"` // "json_object", "json_schema"
		JSONSchema map[string]any `json:"json_schema,omitzero"`
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
func (c *ChatRequest) Init(msgs genai.Messages, opts genai.Validatable, model string) error {
	c.Model = model
	var errs []error
	var unsupported []string
	sp := ""
	if opts != nil {
		if err := opts.Validate(); err != nil {
			errs = append(errs, err)
		} else {
			switch v := opts.(type) {
			case *ChatOptions:
				unsupported, errs = c.initOptions(&v.ChatOptions, model)
				sp = v.SystemPrompt
				c.ServiceTier = v.ServiceTier
			case *genai.ChatOptions:
				unsupported, errs = c.initOptions(v, model)
				sp = v.SystemPrompt
			default:
				errs = append(errs, fmt.Errorf("unsupported options type %T", opts))
			}
		}
	}

	if err := msgs.Validate(); err != nil {
		errs = append(errs, err)
	} else {
		offset := 0
		if sp != "" {
			offset = 1
		}
		c.Messages = make([]Message, len(msgs)+offset)
		if sp != "" {
			c.Messages[0].Role = "system"
			c.Messages[0].Content = []Content{{Type: ContentText, Text: sp}}
		}
		for i := range msgs {
			if err := c.Messages[i+offset].From(&msgs[i]); err != nil {
				errs = append(errs, fmt.Errorf("message %d: %w", i, err))
			}
		}
	}
	if len(unsupported) > 0 {
		// If we have unsupported features but no other errors, return a continuable error
		if len(errs) == 0 {
			return &genai.UnsupportedContinuableError{Unsupported: unsupported}
		}
		// Otherwise, add the unsupported features to the error list
		errs = append(errs, &genai.UnsupportedContinuableError{Unsupported: unsupported})
	}
	return errors.Join(errs...)
}

func (c *ChatRequest) SetStream(stream bool) {
	c.Stream = stream
}

func (c *ChatRequest) initOptions(v *genai.ChatOptions, model string) ([]string, []error) {
	var errs []error
	var unsupported []string
	c.MaxChatTokens = v.MaxTokens
	c.Temperature = v.Temperature
	c.TopP = v.TopP
	c.Seed = v.Seed
	if v.TopK != 0 {
		unsupported = append(unsupported, "TopK")
	}
	c.Stop = v.Stop
	if v.DecodeAs != nil {
		// Groq seems to require a "name" property. Hack by encoding, decoding, changing.
		b, err := json.Marshal(jsonschema.Reflect(v.DecodeAs))
		if err != nil {
			errs = append(errs, err)
		} else {
			m := map[string]any{}
			if err = json.Unmarshal(b, &m); err != nil {
				errs = append(errs, err)
			} else {
				c.ResponseFormat.Type = "json_schema"
				m["name"] = "response"
				c.ResponseFormat.JSONSchema = m
			}
		}
	} else if v.ReplyAsJSON {
		c.ResponseFormat.Type = "json_object"
	}
	if len(v.Tools) != 0 {
		// This is annoying to hardcode here but it still "succeeds" while returning nothing.
		if model == "qwen-qwq-32b" {
			// TODO: It works when not streaming.
			errs = append(errs, errors.New("unsupported option Tools"))
		} else {
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

// https://console.groq.com/docs/api-reference#chat-create
type Message struct {
	Role       string     `json:"role"`          // "system", "assistant", "user"
	Name       string     `json:"name,omitzero"` // An optional name for the participant. Provides the model information to differentiate between participants of the same role.
	Content    Contents   `json:"content,omitzero"`
	ToolCalls  []ToolCall `json:"tool_calls,omitzero"`
	ToolCallID string     `json:"tool_call_id,omitzero"`
}

func (m *Message) From(in *genai.Message) error {
	switch in.Role {
	case genai.User, genai.Assistant:
		m.Role = string(in.Role)
	default:
		return fmt.Errorf("unsupported role %q", in.Role)
	}
	m.Name = in.User
	if len(in.Contents) != 0 {
		m.Content = make(Contents, 0, len(in.Contents))
		for i := range in.Contents {
			if in.Contents[i].Thinking != "" {
				// DeepSeek and Qwen recommend against passing reasoning back.
				continue
			}
			m.Content = append(m.Content, Content{})
			if err := m.Content[len(m.Content)-1].From(&in.Contents[i]); err != nil {
				return fmt.Errorf("block %d: %w", i, err)
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
			// This could be worked around.
			return fmt.Errorf("can't have more than one tool call result at a time")
		}
		m.Role = "tool"
		m.Content = Contents{{Type: ContentText, Text: in.ToolCallResults[0].Result}}
		m.ToolCallID = in.ToolCallResults[0].ID
	}
	return nil
}

// Contents exists to marshal single content text block as a string.
//
// Groq requires this for assistant messages.
type Contents []Content

func (c *Contents) MarshalJSON() ([]byte, error) {
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

func (c *Content) From(in *genai.Content) error {
	// DeepSeek and Qwen recommend against passing reasoning back to the model.
	if in.Text != "" {
		c.Type = ContentText
		c.Text = in.Text
		return nil
	}
	mimeType, data, err := in.ReadDocument(10 * 1024 * 1024)
	if err != nil {
		return err
	}
	switch {
	case (in.URL != "" && mimeType == "") || strings.HasPrefix(mimeType, "image/"):
		c.Type = ContentImageURL
		if in.URL == "" {
			c.ImageURL.URL = fmt.Sprintf("data:%s;base64,%s", mimeType, base64.StdEncoding.EncodeToString(data))
		} else {
			c.ImageURL.URL = in.URL
		}
	default:
		return fmt.Errorf("unsupported mime type %s", mimeType)
	}
	return nil
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
	Created        Time   `json:"created"`
	ID             string `json:"id"`
	Model          string `json:"model"`
	Object         string `json:"object"` // "chat.completion"
	Usage          Usage  `json:"usage"`
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
	SystemFingerprint string `json:"system_fingerprint"`
	Xgroq             struct {
		ID string `json:"id"`
	} `json:"x_groq"`
}

func (c *ChatResponse) ToResult() (genai.ChatResult, error) {
	out := genai.ChatResult{
		// At the moment, Groq does not support cached tokens.
		Usage: genai.Usage{
			InputTokens:  c.Usage.PromptTokens,
			OutputTokens: c.Usage.CompletionTokens,
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
	Role      genai.Role `json:"role"`
	Reasoning string     `json:"reasoning"`
	Content   string     `json:"content"`
	ToolCalls []ToolCall `json:"tool_calls"`
}

func (m *MessageResponse) To(out *genai.Message) error {
	switch role := m.Role; role {
	case "assistant", "user":
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
	if m.Reasoning != "" {
		out.Contents = append(out.Contents, genai.Content{Thinking: m.Reasoning})
	}
	if m.Content != "" {
		out.Contents = append(out.Contents, genai.Content{Text: m.Content})
	}
	return nil
}

type ChatStreamChunkResponse struct {
	ID                string `json:"id"`
	Object            string `json:"object"`
	Created           Time   `json:"created"`
	Model             string `json:"model"`
	SystemFingerprint string `json:"system_fingerprint"`
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

// Time is a JSON encoded unix timestamp.
type Time int64

func (t *Time) AsTime() time.Time {
	return time.Unix(int64(*t), 0)
}

type Model struct {
	ID                  string   `json:"id"`
	Object              string   `json:"object"`
	Created             Time     `json:"created"`
	OwnedBy             string   `json:"owned_by"`
	Active              bool     `json:"active"`
	ContextWindow       int64    `json:"context_window"`
	PublicApps          []string `json:"public_apps"`
	MaxCompletionTokens int64    `json:"max_completion_tokens"`
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
	Error struct {
		Message          string `json:"message"`
		Type             string `json:"type"`
		Code             string `json:"code"`
		FailedGeneration string `json:"failed_generation"`
		StatusCode       int64  `json:"status_code"`
	} `json:"error"`
}

func (er *ErrorResponse) String() string {
	suffix := ""
	if er.Error.FailedGeneration != "" {
		suffix = fmt.Sprintf("error Failed generation: %q", er.Error.FailedGeneration)
	}
	return fmt.Sprintf("error %s (%s): %s%s", er.Error.Code, er.Error.Type, er.Error.Message, suffix)
}

// Client implements genai.ProviderChat and genai.ProviderModel.
type Client struct {
	internal.ClientChat[*ErrorResponse, *ChatRequest, *ChatResponse, ChatStreamChunkResponse]
}

// New creates a new client to talk to the Groq platform API.
//
// If apiKey is not provided, it tries to load it from the GROQ_API_KEY environment variable.
// If none is found, it returns an error.
// Get your API key at https://console.groq.com/keys
// If no model is provided, only functions that do not require a model, like ListModels, will work.
// To use multiple models, create multiple clients.
// Use one of the model from https://console.groq.com/dashboard/limits or https://console.groq.com/docs/models
//
// r can be used to throttle outgoing requests, record calls, etc. It defaults to http.DefaultTransport.
//
// Tool use requires the use of a model that supports it.
// https://console.groq.com/docs/tool-use
func New(apiKey, model string, r http.RoundTripper) (*Client, error) {
	const apiKeyURL = "https://console.groq.com/keys"
	if apiKey == "" {
		if apiKey = os.Getenv("GROQ_API_KEY"); apiKey == "" {
			return nil, errors.New("groq API key is required; get one at " + apiKeyURL)
		}
	}
	if r == nil {
		r = http.DefaultTransport
	}
	return &Client{
		ClientChat: internal.ClientChat[*ErrorResponse, *ChatRequest, *ChatResponse, ChatStreamChunkResponse]{
			Model:                model,
			ChatURL:              "https://api.groq.com/openai/v1/chat/completions",
			ProcessStreamPackets: processStreamPackets,
			ClientBase: internal.ClientBase[*ErrorResponse]{
				ClientJSON: httpjson.Client{
					Client: &http.Client{Transport: &roundtrippers.Header{
						Transport: &roundtrippers.Retry{
							Transport: &roundtrippers.RequestID{
								Transport: r,
							},
						},
						Header: http.Header{"Authorization": {"Bearer " + apiKey}},
					}},
					Lenient: internal.BeLenient,
				},
				APIKeyURL: apiKeyURL,
			},
		},
	}, nil
}

func (c *Client) Name() string {
	return "groq"
}

func (c *Client) Scoreboard() genai.Scoreboard {
	return Scoreboard
}

func (c *Client) ListModels(ctx context.Context) ([]genai.Model, error) {
	// https://console.groq.com/docs/api-reference#models-list
	return internal.ListModels[*ErrorResponse, *ModelsResponse](ctx, &c.ClientBase, "https://api.groq.com/openai/v1/models")
}

func processStreamPackets(ch <-chan ChatStreamChunkResponse, chunks chan<- genai.MessageFragment, result *genai.ChatResult) error {
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
		f := genai.MessageFragment{
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

var (
	_ genai.ProviderChat  = &Client{}
	_ genai.ProviderModel = &Client{}
)
