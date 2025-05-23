// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Package cerebras implements a client for the Cerebras API.
//
// It is described at https://inference-docs.cerebras.ai/api-reference/
package cerebras

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"os"
	"time"

	"github.com/invopop/jsonschema"
	"github.com/maruel/genai"
	"github.com/maruel/genai/internal"
	"github.com/maruel/genai/internal/sse"
	"github.com/maruel/httpjson"
	"github.com/maruel/roundtrippers"
)

// Official python client: https://github.com/Cerebras/cerebras-cloud-sdk-python
//
// CompletionsResource.create() at
// https://github.com/Cerebras/cerebras-cloud-sdk-python/blob/main/src/cerebras/cloud/sdk/resources/chat/completions.py

// https://inference-docs.cerebras.ai/api-reference/chat-completions
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
			case *genai.ChatOptions:
				c.MaxCompletionTokens = v.MaxTokens
				c.Temperature = v.Temperature
				c.TopP = v.TopP
				sp = v.SystemPrompt
				c.Seed = v.Seed
				if v.TopK != 0 {
					unsupported = append(unsupported, "TopK")
				}
				c.Stop = v.Stop
				if v.ReplyAsJSON {
					c.ResponseFormat.Type = "json_object"
				}
				if v.DecodeAs != nil {
					c.ResponseFormat.Type = "json_schema"
					c.ResponseFormat.JSONSchema.Schema = jsonschema.Reflect(v.DecodeAs)
					c.ResponseFormat.JSONSchema.Strict = true
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
						c.Tools[i].Function.Parameters = t.InputSchema()
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
			c.Messages[0].Content = []Content{{
				Type: ContentText,
				Text: sp,
			}}
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
	if len(*c) == 1 && (*c)[0].Type == ContentText {
		return json.Marshal((*c)[0].Text)
	}
	return json.Marshal(([]Content)(*c))
}

// UnmarshalJSON implements custom unmarshalling for Contents type
// to handle cases where content could be a string or Content struct.
func (c *Contents) UnmarshalJSON(data []byte) error {
	// Try unmarshalling as a string first
	var contentStr string
	if err := json.Unmarshal(data, &contentStr); err == nil {
		// If it worked, create a single content with the string
		*c = Contents{{
			Type: ContentText,
			Text: contentStr,
		}}
		return nil
	}

	// If that failed, try as array of Content
	var contents []Content
	if err := json.Unmarshal(data, &contents); err != nil {
		return err
	}
	*c = contents
	return nil
}

// From converts from a genai.Message to a Message.
func (m *Message) From(in *genai.Message) error {
	switch in.Role {
	case genai.User, genai.Assistant:
		m.Role = string(in.Role)
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
			} else {
				// Cerebras doesn't support documents yet.
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
			// This could be worked around.
			return fmt.Errorf("can't have more than one tool call result at a time")
		}
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
	ID                string `json:"id"`
	Model             string `json:"model"`
	Object            string `json:"object"` // "chat.completion"
	SystemFingerprint string `json:"system_fingerprint"`
	Created           Time   `json:"created"`
	Choices           []struct {
		Index        int64        `json:"index"`
		FinishReason FinishReason `json:"finish_reason"`
		Message      Message      `json:"message"`
	} `json:"choices"`
	Usage    Usage `json:"usage"`
	TimeInfo struct {
		QueueTime  float64 `json:"queue_time"`      // In seconds
		PromptTime float64 `json:"prompt_time"`     // In seconds
		ChatTime   float64 `json:"completion_time"` // In seconds
		TotalTime  float64 `json:"total_time"`      // In seconds
		Created    Time    `json:"created"`
	} `json:"time_info"`
}

func (c *ChatResponse) ToResult() (genai.ChatResult, error) {
	out := genai.ChatResult{
		// At the moment, Cerebras doesn't support cached tokens.
		Usage: genai.Usage{
			InputTokens:  c.Usage.PromptTokens,
			OutputTokens: c.Usage.CompletionTokens,
			FinishReason: c.Choices[0].FinishReason.ToFinishReason(),
		},
	}
	if len(c.Choices) != 1 {
		return out, fmt.Errorf("expected 1 choice, got %#v", c.Choices)
	}
	err := c.Choices[0].Message.To(&out.Message)
	return out, err
}

type FinishReason string

const (
	FinishStop          FinishReason = "stop"
	FinishToolCalls     FinishReason = "tool_calls"
	FinishLength        FinishReason = "length"
	FinishContentFilter FinishReason = "content_filter"
)

func (f FinishReason) ToFinishReason() string {
	return string(f)
}

type ChatStreamChunkResponse struct {
	ID                string `json:"id"`
	Model             string `json:"model"`
	Object            string `json:"object"`
	SystemFingerprint string `json:"system_fingerprint"`
	Created           Time   `json:"created"`
	Choices           []struct {
		Delta struct {
			Role      string     `json:"role"`
			Content   Contents   `json:"content"`
			ToolCalls []ToolCall `json:"tool_calls"`
		} `json:"delta"`
		Index        int64        `json:"index"`
		FinishReason FinishReason `json:"finish_reason"`
	} `json:"choices"`
	Usage    Usage `json:"usage"`
	TimeInfo struct {
		QueueTime  float64 `json:"queue_time"`
		PromptTime float64 `json:"prompt_time"`
		ChatTime   float64 `json:"completion_time"`
		TotalTime  float64 `json:"total_time"`
		Created    Time    `json:"created"`
	} `json:"time_info"`
}

type Usage struct {
	PromptTokens     int64 `json:"prompt_tokens"`
	CompletionTokens int64 `json:"completion_tokens"`
	TotalTokens      int64 `json:"total_tokens"`
}

type Time int64

func (t *Time) AsTime() time.Time {
	return time.Unix(int64(*t), 0)
}

//

type errorResponse struct {
	// Either this
	Detail string `json:"detail"`

	// Or this
	Message string `json:"message"`
	Type    string `json:"type"`
	Param   string `json:"param"`
	Code    string `json:"code"`
}

func (er *errorResponse) String() string {
	if er.Detail != "" {
		return fmt.Sprintf("error %s", er.Detail)
	}
	return fmt.Sprintf("error %s/%s/%s: %s", er.Type, er.Param, er.Code, er.Message)
}

// Client implements the REST JSON based API.
type Client struct {
	internal.ClientBase[errorResponse]

	model   string
	chatURL string
}

// New creates a new client to talk to the Cerebras platform API.
//
// If apiKey is not provided, it tries to load it from the CEREBRAS_API_KEY environment variable.
// If none is found, it returns an error.
// Get an API key at http://cloud.cerebras.ai/
// If no model is provided, only functions that do not require a model, like ListModels, will work.
// To use multiple models, create multiple clients.
// Use one of the model from https://cerebras.ai/inference
func New(apiKey, model string) (*Client, error) {
	const apiKeyURL = "https://cloud.cerebras.ai/platform/"
	if apiKey == "" {
		if apiKey = os.Getenv("CEREBRAS_API_KEY"); apiKey == "" {
			return nil, errors.New("cerebras API key is required; get one at " + apiKeyURL)
		}
	}
	return &Client{
		model:   model,
		chatURL: "https://api.cerebras.ai/v1/chat/completions",
		ClientBase: internal.ClientBase[errorResponse]{
			ClientJSON: httpjson.Client{
				Client: &http.Client{Transport: &roundtrippers.Header{
					Transport: &roundtrippers.Retry{
						Transport: &roundtrippers.RequestID{
							Transport: http.DefaultTransport,
						},
					},
					Header: http.Header{"Authorization": {"Bearer " + apiKey}},
				}},
				Lenient: internal.BeLenient,
			},
			APIKeyURL: apiKeyURL,
		},
	}, nil
}

func (c *Client) Chat(ctx context.Context, msgs genai.Messages, opts genai.Validatable) (genai.ChatResult, error) {
	// https://inference-docs.cerebras.ai/api-reference/chat-completions
	for i, msg := range msgs {
		for j, content := range msg.Contents {
			if len(content.Opaque) != 0 {
				return genai.ChatResult{}, fmt.Errorf("message #%d content #%d: field Opaque not supported", i, j)
			}
		}
	}
	rpcin := ChatRequest{}
	var continuableErr error
	if err := rpcin.Init(msgs, opts, c.model); err != nil {
		// If it's an UnsupportedContinuableError, we can continue
		if uce, ok := err.(*genai.UnsupportedContinuableError); ok {
			// Store the error to return later if no other error occurs
			continuableErr = uce
			// Otherwise log the error but continue
		} else {
			return genai.ChatResult{}, err
		}
	}
	rpcout := ChatResponse{}
	if err := c.ChatRaw(ctx, &rpcin, &rpcout); err != nil {
		return genai.ChatResult{}, err
	}
	result, err := rpcout.ToResult()
	if err != nil {
		return result, err
	}
	// Return the continuable error if no other error occurred
	if continuableErr != nil {
		return result, continuableErr
	}
	return result, nil
}

func (c *Client) ChatRaw(ctx context.Context, in *ChatRequest, out *ChatResponse) error {
	if err := c.validate(); err != nil {
		return err
	}
	in.Stream = false
	return c.DoRequest(ctx, "POST", c.chatURL, in, out)
}

func (c *Client) ChatStream(ctx context.Context, msgs genai.Messages, opts genai.Validatable, chunks chan<- genai.MessageFragment) (genai.ChatResult, error) {
	return internal.ChatStream(ctx, msgs, opts, chunks, c.model, c.ChatStreamRaw, processStreamPackets, false)
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
		switch role := pkt.Choices[0].Delta.Role; role {
		case "", "assistant":
		default:
			return fmt.Errorf("unexpected role %q", role)
		}
		if pkt.Usage.TotalTokens != 0 {
			result.InputTokens = pkt.Usage.PromptTokens
			result.OutputTokens = pkt.Usage.CompletionTokens
			result.FinishReason = pkt.Choices[0].FinishReason.ToFinishReason()
		}

		for _, nt := range pkt.Choices[0].Delta.ToolCalls {
			f := genai.MessageFragment{ToolCall: genai.ToolCall{
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
				f := genai.MessageFragment{TextFragment: content.Text}
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
	}
	return nil
}

func (c *Client) ChatStreamRaw(ctx context.Context, in *ChatRequest, out chan<- ChatStreamChunkResponse) error {
	if err := c.validate(); err != nil {
		return err
	}
	in.Stream = true
	resp, err := c.ClientJSON.Request(ctx, "POST", c.chatURL, nil, in)
	if err != nil {
		return fmt.Errorf("failed to get server response: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		return c.DecodeError(ctx, c.chatURL, resp)
	}
	return sse.Process(resp.Body, out, &errorResponse{})
}

func (c *Client) validate() error {
	if c.model == "" {
		return errors.New("a model is required")
	}
	return nil
}

type Model struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Created Time   `json:"created"`
	OwnedBy string `json:"owned_by"`
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

func (c *Client) ListModels(ctx context.Context) ([]genai.Model, error) {
	// https://inference-docs.cerebras.ai/api-reference/models
	var out struct {
		Object string  `json:"object"`
		Data   []Model `json:"data"`
	}
	if err := c.DoRequest(ctx, "GET", "https://api.cerebras.ai/v1/models", nil, &out); err != nil {
		return nil, err
	}
	models := make([]genai.Model, len(out.Data))
	for i := range out.Data {
		models[i] = &out.Data[i]
	}
	return models, nil
}

var (
	_ genai.ChatProvider  = &Client{}
	_ genai.ModelProvider = &Client{}
)
