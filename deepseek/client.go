// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Package deepseek implements a client for the DeepSeek API.
//
// It is described at https://api-docs.deepseek.com/
package deepseek

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"os"

	"github.com/invopop/jsonschema"
	"github.com/maruel/genai"
	"github.com/maruel/genai/internal"
	"github.com/maruel/httpjson"
	"github.com/maruel/roundtrippers"
)

// Scoreboard for DeepSeek.
//
// # Warnings
//
//   - JSON schema decoding is announced to be added later in the doc.
//   - Tool calling works very well but is biased; the model is lazy and when it's unsure, it will use the
//     tool's first argument.
//   - Tool calling is not supported with deepseek-reasoner.
//   - DeepSeek doesn't do rate limiting: https://api-docs.deepseek.com/quick_start/rate_limit
var Scoreboard = genai.Scoreboard{
	Scenarios: []genai.Scenario{
		{
			In:     []genai.Modality{genai.ModalityText},
			Out:    []genai.Modality{genai.ModalityText},
			Models: []string{"deepseek-chat"},
			Chat: genai.Functionality{
				Inline:             true,
				URL:                false,
				Thinking:           false,
				ReportTokenUsage:   true,
				ReportFinishReason: true,
				MaxTokens:          true,
				StopSequence:       true,
				Tools:              genai.True,
				UnbiasedTool:       false,
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
				Tools:              genai.True,
				UnbiasedTool:       false,
				JSON:               true,
				JSONSchema:         false,
			},
		},
		{
			In:     []genai.Modality{genai.ModalityText},
			Out:    []genai.Modality{genai.ModalityText},
			Models: []string{"deepseek-reasoner"},
			Chat: genai.Functionality{
				Inline:             true,
				URL:                false,
				Thinking:           true,
				ReportTokenUsage:   true,
				ReportFinishReason: true,
				MaxTokens:          true,
				StopSequence:       true,
				Tools:              genai.False,
				UnbiasedTool:       false,
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
				Tools:              genai.False,
				UnbiasedTool:       false,
				JSON:               true,
				JSONSchema:         false,
			},
		},
	},
}

// https://api-docs.deepseek.com/api/create-chat-completion
type ChatRequest struct {
	Model            string    `json:"model"`
	Messages         []Message `json:"messages"`
	Stream           bool      `json:"stream"`
	Temperature      float64   `json:"temperature,omitzero"`       // [0, 2]
	FrequencyPenalty float64   `json:"frequency_penalty,omitzero"` // [-2, 2]
	MaxToks          int64     `json:"max_tokens,omitzero"`        // [1, 8192]
	PresencePenalty  float64   `json:"presence_penalty,omitzero"`  // [-2, 2]
	ResponseFormat   struct {
		Type string `json:"type,omitzero"` // "text", "json_object"
	} `json:"response_format,omitzero"`
	Stop          []string `json:"stop,omitzero"`
	StreamOptions struct {
		IncludeUsage bool `json:"include_usage,omitzero"`
	} `json:"stream_options,omitzero"`
	TopP float64 `json:"top_p,omitzero"` // [0, 1]
	// Alternative when forcing a specific function. This can probably be achieved
	// by providing a single tool and ToolChoice == "required".
	// ToolChoice struct {
	// 	Type     string `json:"type,omitzero"` // "function"
	// 	Function struct {
	// 		Name string `json:"name,omitzero"`
	// 	} `json:"function,omitzero"`
	// } `json:"tool_choice,omitzero"`
	ToolChoice string `json:"tool_choice,omitzero"` // "none", "auto", "required"
	Tools      []Tool `json:"tools,omitzero"`
	Logprobs   bool   `json:"logprobs,omitzero"`
	TopLogprob int64  `json:"top_logprobs,omitzero"`
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
			// https://api-docs.deepseek.com/guides/reasoning_model Soon "reasoning_effort"
			switch v := opts.(type) {
			case *genai.ChatOptions:
				c.MaxToks = v.MaxTokens
				c.Temperature = v.Temperature
				c.TopP = v.TopP
				sp = v.SystemPrompt
				if v.Seed != 0 {
					unsupported = append(unsupported, "Seed")
				}
				if v.TopK != 0 {
					unsupported = append(unsupported, "TopK")
				}
				c.Stop = v.Stop
				if v.ReplyAsJSON {
					c.ResponseFormat.Type = "json_object"
				}
				if v.DecodeAs != nil {
					errs = append(errs, errors.New("unsupported option DecodeAs"))
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
			c.Messages[0].Content = sp
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

// https://api-docs.deepseek.com/api/create-chat-completion
type Message struct {
	Role             string     `json:"role,omitzero"` // "system", "assistant", "user"
	Name             string     `json:"name,omitzero"` // An optional name for the participant. Provides the model information to differentiate between participants of the same role.
	Content          string     `json:"content,omitzero"`
	Prefix           bool       `json:"prefix,omitzero"` // Force the model to start its answer by the content of the supplied prefix in this assistant message.
	ReasoningContent string     `json:"reasoning_content,omitzero"`
	ToolCalls        []ToolCall `json:"tool_calls,omitzero"`
	ToolCallID       string     `json:"tool_call_id,omitzero"` // Tool call that this message is responding to, with response in Content field.
}

func (m *Message) From(in *genai.Message) error {
	switch in.Role {
	case genai.User, genai.Assistant:
		m.Role = string(in.Role)
	default:
		return fmt.Errorf("unsupported role %q", in.Role)
	}
	m.Name = in.User
	for i := range in.Contents {
		// Thinking content should not be returned to the model.
		m.Content += in.Contents[i].Text
		if in.Contents[i].Filename != "" || in.Contents[i].Document != nil || in.Contents[i].URL != "" {
			return errors.New("deepseek doesn't support document content blocks")
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
		m.Content = in.ToolCallResults[0].Result
		m.ToolCallID = in.ToolCallResults[0].ID
	}
	return nil
}

func (m *Message) To(out *genai.Message) error {
	switch role := m.Role; role {
	case "user":
		out.Role = genai.Role(role)
	case "assistant", "model":
		out.Role = genai.Assistant
	default:
		return fmt.Errorf("unsupported role %q", role)
	}
	if len(m.ToolCalls) != 0 {
		out.ToolCalls = make([]genai.ToolCall, len(m.ToolCalls))
		for i := range m.ToolCalls {
			m.ToolCalls[i].To(&out.ToolCalls[i])
		}
	}
	// Both ReasoningContent and Content can be set on the same reply.
	if m.ReasoningContent != "" {
		out.Contents = []genai.Content{{Thinking: m.Content}}
	}
	if m.Content != "" {
		out.Contents = append(out.Contents, genai.Content{Text: m.Content})
	}
	return nil
}

type ToolCall struct {
	Index    int64  `json:"index,omitzero"`
	ID       string `json:"id,omitzero"`
	Type     string `json:"type,omitzero"` // "function"
	Function struct {
		Name      string `json:"name,omitzero"`
		Arguments string `json:"arguments,omitzero"`
	} `json:"function,omitzero"`
}

func (t *ToolCall) From(in *genai.ToolCall) {
	t.Type = "function"
	t.Index = 0 // Unsure
	t.ID = in.ID
	t.Function.Name = in.Name
	t.Function.Arguments = in.Arguments
}

func (t *ToolCall) To(out *genai.ToolCall) {
	out.ID = t.ID
	out.Name = t.Function.Name
	out.Arguments = t.Function.Arguments
}

type Tool struct {
	Type     string `json:"type"` // "function"
	Function struct {
		Name        string             `json:"name,omitzero"`
		Description string             `json:"description,omitzero"`
		Parameters  *jsonschema.Schema `json:"parameters,omitzero"`
	} `json:"function"`
}

type ChatResponse struct {
	ID      string `json:"id"`
	Choices []struct {
		FinishReason FinishReason `json:"finish_reason"`
		Index        int64        `json:"index"`
		Message      Message      `json:"message"`
		Logprobs     Logprobs     `json:"logprobs"`
	} `json:"choices"`
	Created           int64  `json:"created"` // Unix timestamp
	Model             string `json:"model"`
	SystemFingerPrint string `json:"system_fingerprint"`
	Object            string `json:"object"` // chat.completion
	Usage             Usage  `json:"usage"`
}

func (c *ChatResponse) ToResult() (genai.ChatResult, error) {
	out := genai.ChatResult{
		Usage: genai.Usage{
			InputTokens:       c.Usage.PromptTokens,
			InputCachedTokens: c.Usage.PromptCacheHitTokens,
			OutputTokens:      c.Usage.CompletionTokens,
		},
	}
	if len(c.Choices) != 1 {
		return out, fmt.Errorf("expected 1 choice, got %#v", c.Choices)
	}
	out.FinishReason = c.Choices[0].FinishReason.ToFinishReason()
	err := c.Choices[0].Message.To(&out.Message)
	return out, err
}

type FinishReason string

const (
	FinishStop          FinishReason = "stop"
	FinishToolCalls     FinishReason = "tool_calls"
	FinishLength        FinishReason = "length"
	FinishContentFilter FinishReason = "content_filter"
	FinishInsufficient  FinishReason = "insufficient_system_resource"
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
		return genai.FinishReason(f)
	}
}

type Usage struct {
	CompletionTokens      int64 `json:"completion_tokens"`
	PromptTokens          int64 `json:"prompt_tokens"`
	PromptCacheHitTokens  int64 `json:"prompt_cache_hit_tokens"`
	PromptCacheMissTokens int64 `json:"prompt_cache_miss_tokens"`
	TotalTokens           int64 `json:"total_tokens"`
	PromptTokensDetails   struct {
		CachedTokens int64 `json:"cached_tokens"`
	} `json:"prompt_tokens_details"`
	ChatTokensDetails struct {
		ReasoningTokens int64 `json:"reasoning_tokens"`
	} `json:"completion_tokens_details"`
}

type Logprobs struct {
	Content []struct {
		Token       string  `json:"token"`
		Logprob     float64 `json:"logprob"`
		Bytes       []int64 `json:"bytes"`
		TopLogprobs []struct {
			Token   string  `json:"token"`
			Logprob float64 `json:"logprob"`
			Bytes   []int64 `json:"bytes"`
		} `json:"top_logprobs"`
	} `json:"content"`
}

type ChatStreamChunkResponse struct {
	ID                string `json:"id"`
	Object            string `json:"object"`  // chat.completion.chunk
	Created           int64  `json:"created"` // Unix timestamp
	Model             string `json:"model"`
	SystemFingerprint string `json:"system_fingerprint"`
	Choices           []struct {
		Index        int64        `json:"index"`
		Delta        Message      `json:"delta"`
		Logprobs     Logprobs     `json:"logprobs"`
		FinishReason FinishReason `json:"finish_reason"`
	} `json:"choices"`
	Usage Usage `json:"usage"`
}

type Model struct {
	ID      string `json:"id"`
	Object  string `json:"object"` // model
	OwnedBy string `json:"owned_by"`
}

func (m *Model) GetID() string {
	return m.ID
}

func (m *Model) String() string {
	return m.ID
}

func (m *Model) Context() int64 {
	return 0
}

// ModelsResponse represents the response structure for DeepSeek models listing
type ModelsResponse struct {
	Object string  `json:"object"` // list
	Data   []Model `json:"data"`
}

// ToModels converts DeepSeek models to genai.Model interfaces
func (r *ModelsResponse) ToModels() []genai.Model {
	models := make([]genai.Model, len(r.Data))
	for i := range r.Data {
		models[i] = &r.Data[i]
	}
	return models
}

//

type ErrorResponse struct {
	// Type  string `json:"type"`
	Error struct {
		Message string `json:"message"`
		Type    string `json:"type"`
		Param   string `json:"param"`
		Code    string `json:"code"`
	} `json:"error"`
}

func (er *ErrorResponse) String() string {
	return fmt.Sprintf("error %s: %s", er.Error.Type, er.Error.Message)
}

// Client implements genai.ProviderChat and genai.ProviderModel.
type Client struct {
	internal.ClientChat[*ErrorResponse, *ChatRequest, *ChatResponse, ChatStreamChunkResponse]
}

// New creates a new client to talk to the DeepSeek platform API in China.
//
// If apiKey is not provided, it tries to load it from the DEEPSEEK_API_KEY environment variable.
// If none is found, it returns an error.
// Get your API key at https://platform.deepseek.com/api_keys
// If no model is provided, only functions that do not require a model, like ListModels, will work.
// To use multiple models, create multiple clients.
// Use one of the model from https://api-docs.deepseek.com/quick_start/pricing
//
// r can be used to throttle outgoing requests, record calls, etc. It defaults to http.DefaultTransport.
func New(apiKey, model string, r http.RoundTripper) (*Client, error) {
	const apiKeyURL = "https://platform.deepseek.com/api_keys"
	if apiKey == "" {
		if apiKey = os.Getenv("DEEPSEEK_API_KEY"); apiKey == "" {
			return nil, errors.New("deepseek API key is required; get one at " + apiKeyURL)
		}
	}
	if r == nil {
		r = http.DefaultTransport
	}
	return &Client{
		ClientChat: internal.ClientChat[*ErrorResponse, *ChatRequest, *ChatResponse, ChatStreamChunkResponse]{
			Model:                model,
			ChatURL:              "https://api.deepseek.com/chat/completions",
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
	return "deepseek"
}

func (c *Client) Scoreboard() genai.Scoreboard {
	return Scoreboard
}

func (c *Client) ListModels(ctx context.Context) ([]genai.Model, error) {
	// https://api-docs.deepseek.com/api/list-models
	return internal.ListModels[*ErrorResponse, *ModelsResponse](ctx, &c.ClientBase, "https://api.deepseek.com/models")
}

// TODO: Caching: https://api-docs.deepseek.com/guides/kv_cache

func processStreamPackets(ch <-chan ChatStreamChunkResponse, chunks chan<- genai.MessageFragment, result *genai.ChatResult) error {
	defer func() {
		// We need to empty the channel to avoid blocking the goroutine.
		for range ch {
		}
	}()
	pendingCall := ToolCall{}
	for pkt := range ch {
		if len(pkt.Choices) != 1 {
			continue
		}
		if pkt.Usage.CompletionTokens != 0 {
			result.InputTokens = pkt.Usage.PromptTokens
			result.InputCachedTokens = pkt.Usage.PromptCacheHitTokens
			result.OutputTokens = pkt.Usage.CompletionTokens
			result.FinishReason = pkt.Choices[0].FinishReason.ToFinishReason()
		}
		if len(pkt.Choices[0].Delta.ToolCalls) > 1 {
			return fmt.Errorf("implement multiple tool calls: %#v", pkt)
		}
		switch role := pkt.Choices[0].Delta.Role; role {
		case "assistant", "":
		default:
			return fmt.Errorf("unexpected role %q", role)
		}
		f := genai.MessageFragment{
			TextFragment:     pkt.Choices[0].Delta.Content,
			ThinkingFragment: pkt.Choices[0].Delta.ReasoningContent,
		}
		// DeepSeek streams the arguments. Buffer the arguments to send the fragment as a whole tool call.
		if len(pkt.Choices[0].Delta.ToolCalls) == 1 {
			if t := pkt.Choices[0].Delta.ToolCalls[0]; t.ID != "" {
				// A new call.
				if pendingCall.ID == "" {
					pendingCall = t
					if !f.IsZero() {
						return fmt.Errorf("implement tool call with metadata: %#v", pkt)
					}
					continue
				}
				// Flush.
				pendingCall.To(&f.ToolCall)
				pendingCall = t
			} else if pendingCall.ID != "" {
				// Continuation.
				pendingCall.Function.Arguments += t.Function.Arguments
				if !f.IsZero() {
					return fmt.Errorf("implement tool call with metadata: %#v", pkt)
				}
				continue
			}
		} else if pendingCall.ID != "" {
			// Flush.
			pendingCall.To(&f.ToolCall)
			pendingCall = ToolCall{}
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
	_ genai.ProviderChat       = &Client{}
	_ genai.ProviderModel      = &Client{}
	_ genai.ProviderScoreboard = &Client{}
)
