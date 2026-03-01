// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Package ollama implements a client for the Ollama API.
//
// It is described at https://github.com/ollama/ollama/blob/main/docs/api.md
// and https://pkg.go.dev/github.com/ollama/ollama/api
package ollama

import (
	"bufio"
	"bytes"
	"context"
	_ "embed"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"iter"
	"net/http"
	"reflect"
	"slices"
	"strings"
	"time"

	"github.com/invopop/jsonschema"
	"github.com/maruel/genai"
	"github.com/maruel/genai/base"
	"github.com/maruel/genai/internal"
	"github.com/maruel/genai/internal/bb"
	"github.com/maruel/genai/scoreboard"
	"github.com/maruel/roundtrippers"
	"golang.org/x/sync/errgroup"
)

//go:embed scoreboard.json
var scoreboardJSON []byte

// Scoreboard for Ollama.
func Scoreboard() scoreboard.Score {
	var s scoreboard.Score
	d := json.NewDecoder(bytes.NewReader(scoreboardJSON))
	d.DisallowUnknownFields()
	if err := d.Decode(&s); err != nil {
		panic(fmt.Errorf("failed to unmarshal scoreboard.json: %w", err))
	}
	return s
}

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
	Schema *jsonschema.Schema
}

// MarshalJSON implements json.Marshaler.
func (c *ChatRequestFormat) MarshalJSON() ([]byte, error) {
	if c.Type != "" {
		return json.Marshal(c.Type)
	}
	return json.Marshal(c.Schema)
}

// ReasoningEffort controls the amount of thinking effort.
type ReasoningEffort string

// Reasoning effort values.
const (
	ReasoningEffortLow    ReasoningEffort = "low"
	ReasoningEffortMedium ReasoningEffort = "medium"
	ReasoningEffortHigh   ReasoningEffort = "high"
)

// Validate implements genai.Validatable.
func (r ReasoningEffort) Validate() error {
	switch r {
	case "", ReasoningEffortLow, ReasoningEffortMedium, ReasoningEffortHigh:
		return nil
	default:
		return fmt.Errorf("invalid reasoning effort %q", r)
	}
}

// GenOptionText defines Ollama specific options.
type GenOptionText struct {
	// ReasoningEffort controls the thinking effort level ("low", "medium", "high").
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
				c.Format.Schema = internal.JSONSchemaFor(reflect.TypeOf(v.DecodeAs))
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
					if c.Tools[i].Function.Parameters = t.InputSchemaOverride; c.Tools[i].Function.Parameters == nil {
						c.Tools[i].Function.Parameters = t.GetInputSchema()
					}
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
		Index     int64  `json:"index,omitzero"`
		Name      string `json:"name"`
		Arguments any    `json:"arguments"`
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
		Description string             `json:"description,omitzero"`
		Name        string             `json:"name,omitzero"`
		Parameters  *jsonschema.Schema `json:"parameters,omitzero"`
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

// ToGenai converts a slice of Logprob to the genai equivalent.
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

//

// We cannot use ClientChat because GenSync and GenStream try to pull on first failure, and GenStream receives
// line separated JSON instead of SSE.

// Client implements genai.Provider.
type Client struct {
	base.NotImplemented
	impl            base.ProviderBase[*ErrorResponse]
	preloadedModels []genai.Model
	baseURL         string
	chatURL         string
}

// New creates a new client to talk to the Ollama API.
//
// ProviderOptionRemote defaults to "http://localhost:11434".
//
// Ollama doesn't have any mean of authentication so ProviderOptionAPIKey is not supported.
//
// To use multiple models, create multiple clients.
// Use one of the model from https://ollama.com/library
//
// Automatic model selection via ModelCheap, ModelGood, ModelSOTA is using hardcoded models. Before using an
// hardcoded model ID, it will ask ollama to determine if a model is already loaded and it will use that
// instead.
func New(ctx context.Context, opts ...genai.ProviderOption) (*Client, error) {
	var baseURL, model string
	var modalities genai.Modalities
	var preloadedModels []genai.Model
	var wrapper func(http.RoundTripper) http.RoundTripper
	for _, opt := range opts {
		if err := opt.Validate(); err != nil {
			return nil, err
		}
		switch v := opt.(type) {
		case genai.ProviderOptionRemote:
			baseURL = string(v)
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
	if baseURL == "" {
		baseURL = "http://localhost:11434"
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
		impl: base.ProviderBase[*ErrorResponse]{
			Lenient: internal.BeLenient,
			Client: http.Client{
				Transport: &roundtrippers.RequestID{Transport: t},
			},
		},
		preloadedModels: preloadedModels,
		baseURL:         baseURL,
		chatURL:         baseURL + "/api/chat",
	}
	switch model {
	case "":
	case string(genai.ModelCheap), string(genai.ModelGood), string(genai.ModelSOTA):
		c.impl.Model = c.selectBestTextModel(ctx, model)
		c.impl.OutputModalities = mod
	default:
		c.impl.Model = model
		c.impl.OutputModalities = mod
	}
	return c, nil
}

// selectBestTextModel selects the most appropriate model based on the preference (cheap, good, or SOTA).
//
// We may want to make this function overridable in the future by the client since this is going to break one
// day or another.
func (c *Client) selectBestTextModel(ctx context.Context, preference string) string {
	// There's no way to list what's the current best models and no way to list the models in the library:
	// https://github.com/ollama/ollama/issues/8241

	// Figure out the model loaded if any. Ignore the error.
	m, _ := c.ListModels(ctx)
	if len(m) > 0 {
		return m[0].GetID()
	}
	// Hard code some popular models, it's more useful than failing hard. The model is not immediately pulled,
	// it will be pulled upon first use.
	switch preference {
	case string(genai.ModelCheap):
		return "gemma3:1b"
	case string(genai.ModelSOTA):
		return "qwen3:32b"
	case string(genai.ModelGood), "":
		return "qwen3:4b"
	default:
		return "qwen3:4b"
	}
}

// Name implements genai.Provider.
//
// It returns the name of the provider.
func (c *Client) Name() string {
	return "ollama"
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
func (c *Client) GenSync(ctx context.Context, msgs genai.Messages, opts ...genai.GenOption) (genai.Result, error) {
	res := genai.Result{}
	in := ChatRequest{}
	if err := in.Init(msgs, c.impl.Model, opts...); err != nil {
		return res, err
	}
	var out ChatResponse
	if err := c.GenSyncRaw(ctx, &in, &out); err != nil {
		return res, err
	}
	res, err := out.ToResult()
	if err != nil {
		return res, err
	}
	if err = res.Validate(); err != nil {
		return res, &internal.BadError{Err: err}
	}
	return res, nil
}

// GenSyncRaw provides access to the raw API.
func (c *Client) GenSyncRaw(ctx context.Context, in *ChatRequest, out *ChatResponse) error {
	if err := c.Validate(); err != nil {
		return err
	}
	in.Stream = false
	err := c.impl.DoRequest(ctx, "POST", c.chatURL, in, out)
	if err != nil {
		// TODO: Cheezy.
		if strings.Contains(err.Error(), "not found") {
			if err := c.PullModel(ctx, c.impl.Model); err != nil {
				return err
			}
			// Retry.
			err = c.impl.DoRequest(ctx, "POST", c.chatURL, in, out)
		}
	}
	return err
}

// GenStream implements genai.Provider.
func (c *Client) GenStream(ctx context.Context, msgs genai.Messages, opts ...genai.GenOption) (iter.Seq[genai.Reply], func() (genai.Result, error)) {
	res := genai.Result{}
	var finalErr error

	fnFragments := func(yield func(genai.Reply) bool) {
		in := ChatRequest{}
		if err := in.Init(msgs, c.impl.Model, opts...); err != nil {
			finalErr = err
			return
		}
		chunks, finish1 := c.GenStreamRaw(ctx, &in)
		fragments, finish2 := ProcessStream(chunks)
		for f := range fragments {
			if f.IsZero() {
				continue
			}
			if err := f.Validate(); err != nil {
				// Catch provider implementation bugs.
				finalErr = &internal.BadError{Err: err}
				break
			}
			if err := res.Accumulate(&f); err != nil {
				finalErr = &internal.BadError{Err: err}
				return
			}
			if !yield(f) {
				break
			}
		}
		if err := finish1(); finalErr == nil {
			finalErr = err
		}
		var err error
		res.Usage, res.Logprobs, err = finish2()
		if finalErr == nil {
			finalErr = err
		}
	}
	fnFinish := func() (genai.Result, error) {
		if res.Usage.FinishReason == genai.FinishedStop && slices.ContainsFunc(res.Replies, func(r genai.Reply) bool { return !r.ToolCall.IsZero() }) {
			// Lie for the benefit of everyone.
			res.Usage.FinishReason = genai.FinishedToolCalls
		}
		return res, finalErr
	}
	return fnFragments, fnFinish
}

// GenStreamRaw provides access to the raw API.
func (c *Client) GenStreamRaw(ctx context.Context, in *ChatRequest) (iter.Seq[ChatStreamChunkResponse], func() error) {
	var finalError error
	finish := func() error {
		return finalError
	}
	if finalError = c.Validate(); finalError != nil {
		finalError = &internal.BadError{Err: finalError}
		return yieldNothing[ChatStreamChunkResponse], finish
	}
	in.Stream = true
	// Try first, if it immediately errors out requesting to pull, pull then try again.
	resp, err1 := c.impl.JSONRequest(ctx, "POST", c.chatURL, in)
	if err1 != nil {
		finalError = &internal.BadError{Err: fmt.Errorf("failed to get server response: %w", err1)}
		return yieldNothing[ChatStreamChunkResponse], finish
	}

	// Process the stream in a separate goroutine to make sure that when the client iterate, there is already a
	// packet waiting for it. This reduces the overall latency.
	out := make(chan ChatStreamChunkResponse, 16)
	eg := errgroup.Group{}
	eg.Go(func() error {
		defer close(out)
		// Ollama doesn't use SSE.
		err2 := processJSONStream(resp.Body, out, c.impl.Lenient)
		_ = resp.Body.Close()
		if err2 == nil || !strings.Contains(err2.Error(), "not found") {
			return err2
		}
		// Model was not present. Try to pull then rerun again.
		if err2 = c.PullModel(ctx, c.impl.Model); err2 != nil {
			return &internal.BadError{Err: err2}
		}
		// Try a second time now that the model was pulled successfully.
		if resp, err2 = c.impl.JSONRequest(ctx, "POST", c.chatURL, in); err2 != nil {
			return &internal.BadError{Err: fmt.Errorf("failed to get server response: %w", err2)}
		}
		defer func() { _ = resp.Body.Close() }()
		if resp.StatusCode != http.StatusOK {
			return c.impl.DecodeError(c.chatURL, resp)
		}
		// Ollama doesn't use SSE.
		return processJSONStream(resp.Body, out, c.impl.Lenient)
	})

	return func(yield func(ChatStreamChunkResponse) bool) {
		for pkt := range out {
			if !yield(pkt) {
				break
			}
		}
		// Drain remaining messages to unblock the producer goroutine so
		// eg.Wait() doesn't deadlock.
		for range out {
		}
	}, eg.Wait
}

// ListModels implements genai.Provider.
func (c *Client) ListModels(ctx context.Context) ([]genai.Model, error) {
	if c.preloadedModels != nil {
		return c.preloadedModels, nil
	}
	// https://github.com/ollama/ollama/blob/main/docs/api.md#list-local-models
	var resp ModelsResponse
	if err := c.impl.DoRequest(ctx, "GET", c.baseURL+"/api/tags", nil, &resp); err != nil {
		return nil, err
	}
	return resp.ToModels(), nil
}

// PullModel is the equivalent of "ollama pull".
//
// Files are cached under $HOME/.ollama/models/manifests/registry.ollama.ai/library/ or $OLLAMA_MODELS.
func (c *Client) PullModel(ctx context.Context, model string) error {
	in := pullModelRequest{Model: model}
	// TODO: Stream updates instead of hanging for several minutes.
	out := pullModelResponse{}
	if err := c.impl.DoRequest(ctx, "POST", c.baseURL+"/api/pull", &in, &out); err != nil {
		return fmt.Errorf("pull failed: %w", err)
	} else if out.Status != "success" {
		return fmt.Errorf("pull failed: %s", out.Status)
	}
	return nil
}

// Version returns the Ollama server version.
func (c *Client) Version(ctx context.Context) (string, error) {
	v := Version{}
	if err := c.impl.DoRequest(ctx, "GET", c.baseURL+"/api/version", nil, &v); err != nil {
		return v.Version, fmt.Errorf("failed to get version: %w", err)
	}
	return v.Version, nil
}

// Ping checks that the Ollama server is reachable.
func (c *Client) Ping(ctx context.Context) error {
	_, err := c.Version(ctx)
	return err
}

// Validate returns an error if the client is not properly configured.
func (c *Client) Validate() error {
	if c.impl.Model == "" {
		return errors.New("a model is required")
	}
	return nil
}

// processJSONStream processes a \n separated JSON stream. This is different from other backends which use
// SSE.
func processJSONStream(body io.Reader, out chan<- ChatStreamChunkResponse, lenient bool) error {
	for r := bufio.NewReader(body); ; {
		line, err := r.ReadBytes('\n')
		if line = bytes.TrimSpace(line); err == io.EOF {
			if len(line) == 0 {
				return nil
			}
		} else if err != nil {
			return &internal.BadError{Err: fmt.Errorf("failed to get server response: %w", err)}
		}
		if len(line) == 0 {
			continue
		}
		d := json.NewDecoder(bytes.NewReader(line))
		if !lenient {
			d.DisallowUnknownFields()
		}
		d.UseNumber()
		msg := ChatStreamChunkResponse{}
		if err := d.Decode(&msg); err != nil {
			d := json.NewDecoder(bytes.NewReader(line))
			if !lenient {
				d.DisallowUnknownFields()
			}
			d.UseNumber()
			er := ErrorResponse{}
			if err := d.Decode(&er); err != nil {
				return &internal.BadError{Err: fmt.Errorf("failed to decode server response %q: %w", string(line), err)}
			}
			return &er
		}
		out <- msg
	}
}

// ProcessStream converts the raw packets from the streaming API into Reply fragments.
func ProcessStream(chunks iter.Seq[ChatStreamChunkResponse]) (iter.Seq[genai.Reply], func() (genai.Usage, [][]genai.Logprob, error)) {
	var finalErr error
	u := genai.Usage{}
	var l [][]genai.Logprob

	return func(yield func(genai.Reply) bool) {
			for pkt := range chunks {
				if pkt.EvalCount != 0 {
					u.InputTokens = pkt.PromptEvalCount
					u.OutputTokens = pkt.EvalCount
					u.FinishReason = pkt.DoneReason.ToFinishReason()
				}
				l = append(l, ToGenaiLogprobs(pkt.Logprobs)...)
				switch role := pkt.Message.Role; role {
				case "", "assistant":
				default:
					finalErr = &internal.BadError{Err: fmt.Errorf("unexpected role %q", role)}
					return
				}
				if pkt.Message.Thinking != "" {
					if !yield(genai.Reply{Reasoning: pkt.Message.Thinking}) {
						return
					}
				}
				for i := range pkt.Message.ToolCalls {
					f := genai.Reply{}
					if err := pkt.Message.ToolCalls[i].To(&f.ToolCall); err != nil {
						finalErr = &internal.BadError{Err: err}
						return
					}
					if !yield(f) {
						return
					}
				}
				if pkt.Message.Content != "" {
					if !yield(genai.Reply{Text: pkt.Message.Content}) {
						return
					}
				}
			}
		}, func() (genai.Usage, [][]genai.Logprob, error) {
			return u, l, finalErr
		}
}

func yieldNothing[T any](yield func(T) bool) {
}

var _ genai.Provider = &Client{}
