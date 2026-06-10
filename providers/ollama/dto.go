// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Wire types for the Ollama chat API.
//
// Types map to the Ollama Go SDK definitions:
//
//	https://pkg.go.dev/github.com/ollama/ollama/api
//
// API documentation:
//
//	https://github.com/ollama/ollama/blob/main/docs/api.md

package ollama

import (
	"encoding/json"
	"errors"
	"fmt"
	"reflect"
	"slices"
	"strings"
	"time"

	"github.com/maruel/genai"
	"github.com/maruel/genai/base"
	"github.com/maruel/genai/internal"
	"github.com/maruel/genai/internal/bb"
)

// ChatRequest is somewhat documented at
// https://github.com/ollama/ollama/blob/main/docs/api.md#generate-a-chat-completion
//
// The source of truth is at
// https://pkg.go.dev/github.com/ollama/ollama/api#ChatRequest
//
// The implementation is at https://pkg.go.dev/github.com/ollama/ollama/server#Server.ChatHandler
//
// We do not refer to the original struct for two reasons:
// - It uses pointers, which is not ergonomic and is unnecessary.
// - It would fetch a ton of junk.
// - There's two raw JSON fields (completely unnecessary!).
type ChatRequest struct {
	Model       string             `json:"model"`
	Messages    []Message          `json:"messages"`
	Stream      bool               `json:"stream"`
	Format      ChatRequestFormat  `json:"format,omitzero"`
	KeepAlive   string             `json:"keep_alive,omitzero"` // Default "5m"
	Tools       []Tool             `json:"tools,omitzero"`
	Options     ChatRequestOptions `json:"options,omitzero"`
	Think       ReasoningEffort    `json:"think,omitzero"`
	Logprobs    bool               `json:"logprobs,omitzero"`
	TopLogprobs int64              `json:"top_logprobs,omitzero"`
}

// ChatRequestFormat is not documented.
//
// See fromChatRequest() in https://github.com/ollama/ollama/blob/main/openai/openai.go for the actual
// expected format. I think that using llama-server's actual format may be saner.
//
// See llmServer.Completion() in https://github.com/ollama/ollama/blob/main/llm/server.go for the use. Ollama
// doesn't use llama-server's native JSON support at all.
type ChatRequestFormat struct {
	Type   string
	Schema genai.JSONSchema
}

// MarshalJSON implements json.Marshaler.
func (c *ChatRequestFormat) MarshalJSON() ([]byte, error) {
	if c.Type != "" {
		return json.Marshal(c.Type)
	}
	return json.Marshal(c.Schema)
}

// ReasoningEffort controls the amount of thinking effort.
//
// It maps to the "think" field in the Ollama API, which accepts boolean
// (true/false) or string ("low", "medium", "high") values.
type ReasoningEffort string

// Reasoning effort values.
const (
	// ReasoningEffortOff disables thinking. Serialized as JSON false.
	ReasoningEffortOff    ReasoningEffort = "off"
	ReasoningEffortLow    ReasoningEffort = "low"
	ReasoningEffortMedium ReasoningEffort = "medium"
	ReasoningEffortHigh   ReasoningEffort = "high"
)

// MarshalJSON implements json.Marshaler.
//
// "off" is serialized as boolean false to match the Ollama API.
func (r ReasoningEffort) MarshalJSON() ([]byte, error) {
	if r == ReasoningEffortOff {
		return []byte("false"), nil
	}
	return json.Marshal(string(r))
}

// UnmarshalJSON implements json.Unmarshaler.
func (r *ReasoningEffort) UnmarshalJSON(b []byte) error {
	if string(b) == "false" {
		*r = ReasoningEffortOff
		return nil
	}
	if string(b) == "true" {
		// Boolean true means default thinking; map to empty (server default).
		*r = ""
		return nil
	}
	var s string
	if err := json.Unmarshal(b, &s); err != nil {
		return err
	}
	*r = ReasoningEffort(s)
	return r.Validate()
}

// Validate implements genai.Validatable.
func (r ReasoningEffort) Validate() error {
	switch r {
	case "", ReasoningEffortOff, ReasoningEffortLow, ReasoningEffortMedium, ReasoningEffortHigh:
		return nil
	default:
		return fmt.Errorf("invalid reasoning effort %q", r)
	}
}

// GenOptionText defines Ollama specific options.
type GenOptionText struct {
	// ReasoningEffort controls the thinking effort level ("off", "low", "medium", "high").
	ReasoningEffort ReasoningEffort
}

// Validate implements genai.Validatable.
func (o *GenOptionText) Validate() error {
	return o.ReasoningEffort.Validate()
}

// ChatRequestOptions is named Options in ollama.
//
// It is documented at: https://github.com/ollama/ollama/blob/main/docs/modelfile.md#valid-parameters-and-values
type ChatRequestOptions struct {
	// Fields from: https://pkg.go.dev/github.com/ollama/ollama/api#Runner
	NumCtx    int64 `json:"num_ctx,omitzero"` // Context Window, default 4096
	NumBatch  int64 `json:"num_batch,omitzero"`
	NumGPU    int64 `json:"num_gpu,omitzero"`
	MainGPU   int64 `json:"main_gpu,omitzero"`
	UseMMap   bool  `json:"use_mmap,omitzero"`
	NumThread int64 `json:"num_thread,omitzero"`

	// Fields from: https://pkg.go.dev/github.com/ollama/ollama/api#Options
	NumKeep          int64    `json:"num_keep,omitzero"`          //
	Seed             int64    `json:"seed,omitzero"`              //
	NumPredict       int64    `json:"num_predict,omitzero"`       // MaxTokens
	TopK             int64    `json:"top_k,omitzero"`             // Default: 40
	TopP             float64  `json:"top_p,omitzero"`             // Default: 0.9
	MinP             float64  `json:"min_p,omitzero"`             // Default: 0.0
	TypicalP         float64  `json:"typical_p,omitzero"`         //
	RepeatLastN      int64    `json:"repeat_last_n,omitzero"`     // Lookback for repeated tokens, default 64
	Temperature      float64  `json:"temperature,omitzero"`       // default 0.7 or 0.8?
	RepeatPenalty    float64  `json:"repeat_penalty,omitzero"`    // default 1.1
	PresencePenalty  float64  `json:"presence_penalty,omitzero"`  //
	FrequencyPenalty float64  `json:"frequency_penalty,omitzero"` //
	Stop             []string `json:"stop,omitzero"`              //
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
			c.Options.NumPredict = v.MaxTokens
			c.Options.Temperature = v.Temperature
			c.Options.TopP = v.TopP
			sp = v.SystemPrompt
			c.Options.TopK = v.TopK
			c.Options.Stop = v.Stop
			if v.TopLogprobs > 0 {
				c.TopLogprobs = v.TopLogprobs
				c.Logprobs = true
			}
			if v.DecodeAs != nil {
				s, err := genai.JSONSchemaFor(reflect.TypeOf(v.DecodeAs))
				if err != nil {
					errs = append(errs, err)
				} else {
					c.Format.Schema = s
				}
			} else if v.ReplyAsJSON {
				c.Format.Type = "json"
			}
		case *genai.GenOptionTools:
			if len(v.Tools) != 0 {
				switch v.Force {
				case genai.ToolCallAny:
				case genai.ToolCallRequired, genai.ToolCallNone:
					// Don't fail.
					unsupported = append(unsupported, "GenOptionTools.Force")
				}
				c.Tools = make([]Tool, len(v.Tools))
				for i, t := range v.Tools {
					c.Tools[i].Type = "function"
					c.Tools[i].Function.Name = t.Name
					c.Tools[i].Function.Description = t.Description
					s, err := t.GetInputSchema()
					if err != nil {
						errs = append(errs, err)
					}
					c.Tools[i].Function.Parameters = s
				}
			}
		case *GenOptionText:
			if v.ReasoningEffort != "" {
				c.Think = v.ReasoningEffort
			}
		case genai.GenOptionSeed:
			c.Options.Seed = int64(v)
		default:
			unsupported = append(unsupported, internal.TypeName(opt))
		}
	}

	if sp != "" {
		c.Messages = append(c.Messages, Message{Role: "system", Content: sp})
	}
	for i := range msgs {
		switch {
		case len(msgs[i].ToolCallResults) > 1:
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
		case len(msgs[i].Requests) > 1:
			// Handle messages with multiple messages by creating multiple messages
			// TODO: For multi-modal requests we could put them together. It's only a problem for text-only
			// messages.
			for j := range msgs[i].Requests {
				// Create a copy of the message with only one request
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
	// If we have unsupported features but no other errors, return a structured error.
	if len(unsupported) > 0 && len(errs) == 0 {
		return &base.ErrNotSupported{Options: unsupported}
	}
	return errors.Join(errs...)
}

// Message is described at https://github.com/ollama/ollama/blob/main/docs/api.md#parameters-1
//
// The source of truth is at https://pkg.go.dev/github.com/ollama/ollama/api#Message
type Message struct {
	Role       string     `json:"role,omitzero"` // "system", "assistant", "user"
	Content    string     `json:"content,omitzero"`
	Thinking   string     `json:"thinking,omitzero"`
	Images     [][]byte   `json:"images,omitzero"` // List of images as base64 encoded strings.
	ToolCalls  []ToolCall `json:"tool_calls,omitzero"`
	ToolCallID string     `json:"tool_call_id,omitzero"`
	ToolName   string     `json:"tool_name,omitzero"`
}

// From must be called with at most one Request or one ToolCallResults.
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
	// Ollama only supports one text content per message but multiple images. We need to validate first. We may
	// implement that later.
	if len(in.Requests) == 1 {
		switch {
		case in.Requests[0].Text != "":
			m.Content = in.Requests[0].Text
		case !in.Requests[0].Doc.IsZero():
			mimeType, data, err := in.Requests[0].Doc.Read(10 * 1024 * 1024)
			if err != nil {
				return err
			}
			switch {
			case strings.HasPrefix(mimeType, "image/"):
				if in.Requests[0].Doc.URL != "" {
					return errors.New("url are not supported for images")
				}
				m.Images = append(m.Images, data)
				// text/plain, text/markdown
			case strings.HasPrefix(mimeType, "text/"):
				if in.Requests[0].Doc.URL != "" {
					return fmt.Errorf("%s documents must be provided inline, not as a URL", mimeType)
				}
				// Append text/plain document content to the message content
				if m.Content != "" {
					m.Content += "\n" + string(data)
				} else {
					m.Content = string(data)
				}
			default:
				return fmt.Errorf("ollama unsupported content type %q", mimeType)
			}
		default:
			return errors.New("unknown Request type")
		}
	}
	for i := range in.Replies {
		if len(in.Replies[i].Opaque) != 0 {
			return &internal.BadError{Err: fmt.Errorf("reply #%d: field Reply.Opaque not supported", i)}
		}
		switch {
		case in.Replies[i].Text != "":
			m.Content = in.Replies[i].Text
		case !in.Replies[i].Doc.IsZero():
			mimeType, data, err := in.Replies[i].Doc.Read(10 * 1024 * 1024)
			if err != nil {
				return err
			}
			switch {
			case strings.HasPrefix(mimeType, "image/"):
				if in.Replies[i].Doc.URL != "" {
					return fmt.Errorf("reply #%d: url are not supported for images", i)
				}
				m.Images = append(m.Images, data)
				// text/plain, text/markdown
			case strings.HasPrefix(mimeType, "text/"):
				if in.Replies[i].Doc.URL != "" {
					return fmt.Errorf("reply #%d: %s documents must be provided inline, not as a URL", i, mimeType)
				}
				// Append text document content to the message content
				if m.Content != "" {
					m.Content += "\n" + string(data)
				} else {
					m.Content = string(data)
				}
			default:
				return &internal.BadError{Err: fmt.Errorf("reply #%d: ollama unsupported content type %q", i, mimeType)}
			}
		case !in.Replies[i].ToolCall.IsZero():
			m.ToolCalls = append(m.ToolCalls, ToolCall{})
			if err := m.ToolCalls[len(m.ToolCalls)-1].From(&in.Replies[i].ToolCall); err != nil {
				return fmt.Errorf("reply #%d: %w", i, err)
			}
		case in.Replies[i].Reasoning != "":
			// Don't send reasoning back.
		default:
			return &internal.BadError{Err: fmt.Errorf("reply #%d: unknown Reply type: %v", i, in.Replies)}
		}
	}
	if len(in.ToolCallResults) != 0 {
		// Process only the first tool call result in this method.
		// The Init method handles multiple tool call results by creating multiple messages.
		m.Content = in.ToolCallResults[0].Result
		m.ToolCallID = in.ToolCallResults[0].ID
		m.ToolName = in.ToolCallResults[0].Name
	}
	return nil
}

// To converts the provider Message to a genai.Message.
func (m *Message) To(out *genai.Message) error {
	if m.Thinking != "" {
		out.Replies = append(out.Replies, genai.Reply{Reasoning: m.Thinking})
	}
	if m.Content != "" {
		out.Replies = append(out.Replies, genai.Reply{Text: m.Content})
	}
	for i := range m.Images {
		out.Replies = append(out.Replies, genai.Reply{
			Doc: genai.Doc{Filename: "image.jpg", Src: &bb.BytesBuffer{D: m.Images[i]}},
		})
	}
	for i := range m.ToolCalls {
		out.Replies = append(out.Replies, genai.Reply{})
		if err := m.ToolCalls[i].To(&out.Replies[len(out.Replies)-1].ToolCall); err != nil {
			return fmt.Errorf("tool call %d: %w", i, err)
		}
	}
	return nil
}

// ToolCall is somewhat documented at https://github.com/ollama/ollama/blob/main/docs/api.md#response-16
// https://pkg.go.dev/github.com/ollama/ollama/api#ToolCall
type ToolCall struct {
	ID       string `json:"id,omitzero"`
	Function struct {
		Index     int64           `json:"index,omitzero"`
		Name      string          `json:"name"`
		Arguments json.RawMessage `json:"arguments"`
	} `json:"function"`
}

// From converts a genai.ToolCall to the provider ToolCall.
func (t *ToolCall) From(in *genai.ToolCall) error {
	if len(in.Opaque) != 0 {
		return &internal.BadError{Err: errors.New("field ToolCall.Opaque not supported")}
	}
	t.ID = in.ID
	t.Function.Name = in.Name
	return json.Unmarshal([]byte(in.Arguments), &t.Function.Arguments)
}

// To converts the provider ToolCall to a genai.ToolCall.
func (t *ToolCall) To(out *genai.ToolCall) error {
	out.ID = t.ID
	out.Name = t.Function.Name
	b, err := json.Marshal(t.Function.Arguments)
	out.Arguments = string(b)
	return err
}

// Tool is somewhat documented at https://github.com/ollama/ollama/blob/main/docs/api.md#chat-request-with-tools
// https://pkg.go.dev/github.com/ollama/ollama/api#Tool
type Tool struct {
	Type     string `json:"type,omitzero"` // "function"
	Function struct {
		Description string           `json:"description,omitzero"`
		Name        string           `json:"name,omitzero"`
		Parameters  genai.JSONSchema `json:"parameters,omitzero"`
	} `json:"function,omitzero"`
}

// Logprob represents per-token log probability information.
//
// See https://pkg.go.dev/github.com/ollama/ollama/api#Logprob
type Logprob struct {
	Token       string         `json:"token"`
	Logprob     float64        `json:"logprob"`
	Bytes       []byte         `json:"bytes,omitzero"`
	TopLogprobs []TokenLogprob `json:"top_logprobs"`
}

// TokenLogprob represents a single token's log probability.
//
// See https://pkg.go.dev/github.com/ollama/ollama/api#TokenLogprob
type TokenLogprob struct {
	Token   string  `json:"token"`
	Logprob float64 `json:"logprob"`
	Bytes   []byte  `json:"bytes,omitzero"`
}

// ToGenaiLogprobs converts a slice of Logprob to the genai equivalent.
func ToGenaiLogprobs(logprobs []Logprob) [][]genai.Logprob {
	if len(logprobs) == 0 {
		return nil
	}
	out := make([][]genai.Logprob, 0, len(logprobs))
	for _, p := range logprobs {
		lp := make([]genai.Logprob, 1, len(p.TopLogprobs)+1)
		lp[0] = genai.Logprob{Text: p.Token, Logprob: p.Logprob}
		for _, tlp := range p.TopLogprobs {
			lp = append(lp, genai.Logprob{Text: tlp.Token, Logprob: tlp.Logprob})
		}
		out = append(out, lp)
	}
	return out
}

// ChatResponse is somewhat documented at https://github.com/ollama/ollama/blob/main/docs/api.md#response-10
// https://pkg.go.dev/github.com/ollama/ollama/api#ChatResponse
type ChatResponse struct {
	Model      string     `json:"model"`
	CreatedAt  time.Time  `json:"created_at"`
	Message    Message    `json:"message"`
	DoneReason DoneReason `json:"done_reason"`
	Done       bool       `json:"done"`
	Logprobs   []Logprob  `json:"logprobs,omitzero"`

	// Remote model info, set when using a remote/proxy model.
	RemoteModel string `json:"remote_model,omitzero"`
	RemoteHost  string `json:"remote_host,omitzero"`

	// https://pkg.go.dev/github.com/ollama/ollama/api#Metrics
	TotalDuration      time.Duration `json:"total_duration"`
	LoadDuration       time.Duration `json:"load_duration"`
	PromptEvalCount    int64         `json:"prompt_eval_count"`
	PromptEvalDuration time.Duration `json:"prompt_eval_duration"`
	EvalCount          int64         `json:"eval_count"`
	EvalDuration       time.Duration `json:"eval_duration"`
	PeakMemory         uint64        `json:"peak_memory,omitzero"`
}

// ToResult converts the ChatResponse to a genai.Result.
func (c *ChatResponse) ToResult() (genai.Result, error) {
	out := genai.Result{
		// TODO: llama-server supports caching and we should report it.
		Usage: genai.Usage{
			InputTokens:  c.PromptEvalCount,
			OutputTokens: c.EvalCount,
			FinishReason: c.DoneReason.ToFinishReason(),
		},
		Logprobs: ToGenaiLogprobs(c.Logprobs),
	}
	err := c.Message.To(&out.Message)
	if out.Usage.FinishReason == genai.FinishedStop && slices.ContainsFunc(out.Replies, func(r genai.Reply) bool { return !r.ToolCall.IsZero() }) {
		// Lie for the benefit of everyone.
		out.Usage.FinishReason = genai.FinishedToolCalls
	}
	return out, err
}

// DoneReason is not documented.
type DoneReason string

// DoneReason values for completion status.
const (
	// DoneStop means the model finished generating normally.
	DoneStop DoneReason = "stop"
	// DoneLength means the model hit the token limit.
	DoneLength DoneReason = "length"
	// DoneLoad means the model was loaded.
	//
	// See https://pkg.go.dev/github.com/ollama/ollama/server#Server.ChatHandler
	DoneLoad DoneReason = "load"
	// DoneUnload means the model was unloaded.
	DoneUnload DoneReason = "unload"
)

// ToFinishReason converts the DoneReason to a genai.FinishReason.
func (d DoneReason) ToFinishReason() genai.FinishReason {
	switch d {
	case DoneStop:
		return genai.FinishedStop
	case DoneLength:
		return genai.FinishedLength
	case DoneLoad, DoneUnload:
		return genai.FinishReason(d)
	default:
		if !internal.BeLenient {
			panic(d)
		}
		return genai.FinishReason(d)
	}
}

// ChatStreamChunkResponse is a streaming chunk from the chat API.
type ChatStreamChunkResponse ChatResponse

// Model is somewhat documented at https://pkg.go.dev/github.com/ollama/ollama/api#ListModelResponse
type Model struct {
	Name       string    `json:"name"`
	Model      string    `json:"model"`
	ModifiedAt time.Time `json:"modified_at"`
	Size       int64     `json:"size"`
	Digest     string    `json:"digest"`
	// https://pkg.go.dev/github.com/ollama/ollama/api#ModelDetails
	Details struct {
		ParentModel       string   `json:"parent_model"`
		Format            string   `json:"format"`
		Family            string   `json:"family"`
		Families          []string `json:"families"`
		ParameterSize     string   `json:"parameter_size"`
		QuantizationLevel string   `json:"quantization_level"`
	} `json:"details"`
}

// GetID implements genai.Model.
func (m *Model) GetID() string {
	return m.Name
}

func (m *Model) String() string {
	return fmt.Sprintf("%s (%s)", m.Name, m.Details.QuantizationLevel)
}

// Context implements genai.Model.
func (m *Model) Context() int64 {
	return 0
}

// ModelsResponse is the response from the list models API.
type ModelsResponse struct {
	Models []Model `json:"models"`
}

// ToModels converts the ModelsResponse to a slice of genai.Model.
func (r *ModelsResponse) ToModels() []genai.Model {
	models := make([]genai.Model, len(r.Models))
	for i := range r.Models {
		models[i] = &r.Models[i]
	}
	return models
}

type pullModelRequest struct {
	Model    string `json:"model"`
	Insecure bool   `json:"insecure"`
	Stream   bool   `json:"stream"`
}

type pullModelResponse struct {
	Status    string `json:"status"`
	Digest    string `json:"digest"`
	Total     int64  `json:"total"`
	Completed int64  `json:"completed"`
}

// Version is the response from the version API.
type Version struct {
	Version string `json:"version"`
}

// ErrorResponse is an error returned by the Ollama API.
type ErrorResponse struct {
	ErrorVal string `json:"error"`
}

func (er *ErrorResponse) Error() string {
	return er.ErrorVal
}

// IsAPIError implements base.APIError.
func (er *ErrorResponse) IsAPIError() bool {
	return true
}
