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

// Scoreboard for Ollama.
//
// # Warnings
//
//   - https://ollama.com/search?c=tool is not a reliable source of truth to determine which model has tool
//     support enabled. See https://github.com/ollama/ollama/issues/9680.
//   - Figure out tools as streaming support recently got added to llama.cpp.
//   - Ollama supports more than what the client supports.
var Scoreboard = scoreboard.Score{
	Country:      "Local",
	DashboardURL: "https://ollama.com/",
	Scenarios: []scoreboard.Scenario{
		{
			Models: []string{"gemma3:4b"},
			In: map[genai.Modality]scoreboard.ModalCapability{
				genai.ModalityImage: {
					Inline:           true,
					SupportedFormats: []string{"image/jpeg", "image/png", "image/webp"},
				},
				genai.ModalityText: {Inline: true},
			},
			Out: map[genai.Modality]scoreboard.ModalCapability{genai.ModalityText: {Inline: true}},
			GenSync: &scoreboard.FunctionalityText{
				JSON:       true,
				JSONSchema: true,
				Seed:       true,
			},
			GenStream: &scoreboard.FunctionalityText{
				JSON:       true,
				JSONSchema: true,
				Seed:       true,
			},
		},
		{
			Models:             []string{"qwen3:4b"},
			Thinking:           true,
			ThinkingTokenStart: "<think>",
			ThinkingTokenEnd:   "\n</think>\n",
			In: map[genai.Modality]scoreboard.ModalCapability{
				genai.ModalityText: {Inline: true},
			},
			Out: map[genai.Modality]scoreboard.ModalCapability{genai.ModalityText: {Inline: true}},
			GenSync: &scoreboard.FunctionalityText{
				Tools:      scoreboard.True,
				JSON:       true,
				JSONSchema: true,
				Seed:       true,
			},
			GenStream: &scoreboard.FunctionalityText{
				Tools:      scoreboard.True,
				JSON:       true,
				JSONSchema: true,
				Seed:       true,
			},
		},
	},
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
	Model     string             `json:"model"`
	Messages  []Message          `json:"messages"`
	Stream    bool               `json:"stream"`
	Format    ChatRequestFormat  `json:"format,omitzero"`
	KeepAlive string             `json:"keep_alive,omitzero"` // Default "5m"
	Tools     []Tool             `json:"tools,omitzero"`
	Options   ChatRequestOptions `json:"options,omitzero"`
	Think     bool               `json:"think,omitzero"`
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

func (c *ChatRequestFormat) MarshalJSON() ([]byte, error) {
	if c.Type != "" {
		return json.Marshal(c.Type)
	}
	return json.Marshal(c.Schema)
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
		case *genai.OptionsText:
			c.Options.NumPredict = v.MaxTokens
			c.Options.Temperature = v.Temperature
			c.Options.TopP = v.TopP
			sp = v.SystemPrompt
			c.Options.Seed = v.Seed
			c.Options.TopK = v.TopK
			c.Options.Stop = v.Stop
			if v.TopLogprobs > 0 {
				unsupported = append(unsupported, "TopLogprobs")
			}
			if v.DecodeAs != nil {
				c.Format.Schema = internal.JSONSchemaFor(reflect.TypeOf(v.DecodeAs))
			} else if v.ReplyAsJSON {
				c.Format.Type = "json"
			}
			if len(v.Tools) != 0 {
				switch v.ToolCallRequest {
				case genai.ToolCallAny:
				case genai.ToolCallRequired:
					// Don't fail.
					unsupported = append(unsupported, "ToolCallRequest")
				case genai.ToolCallNone:
					unsupported = append(unsupported, "ToolCallRequest")
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

	if sp != "" {
		c.Messages = append(c.Messages, Message{Role: "system", Content: sp})
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
		} else if len(msgs[i].Requests) > 1 {
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

// Message is described at https://github.com/ollama/ollama/blob/main/docs/api.md#parameters-1
//
// The source of truth is at https://pkg.go.dev/github.com/ollama/ollama/api#Message
type Message struct {
	Role      string     `json:"role,omitzero"` // "system", "assistant", "user"
	Content   string     `json:"content,omitzero"`
	Thinking  string     `json:"thinking,omitzero"`
	Images    [][]byte   `json:"images,omitzero"` // List of images as base64 encoded strings.
	ToolCalls []ToolCall `json:"tool_calls,omitzero"`
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
		if in.Requests[0].Text != "" {
			m.Content = in.Requests[0].Text
		} else if !in.Requests[0].Doc.IsZero() {
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
			case strings.HasPrefix(mimeType, "text/plain"):
				if in.Requests[0].Doc.URL != "" {
					return errors.New("text/plain documents must be provided inline, not as a URL")
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
		} else {
			return errors.New("unknown Request type")
		}
	}
	for i := range in.Replies {
		if len(in.Replies[i].Opaque) != 0 {
			return fmt.Errorf("reply #%d: field Reply.Opaque not supported", i)
		}
		if in.Replies[i].Text != "" {
			m.Content = in.Replies[i].Text
		} else if !in.Replies[i].Doc.IsZero() {
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
			case strings.HasPrefix(mimeType, "text/plain"):
				if in.Replies[i].Doc.URL != "" {
					return fmt.Errorf("reply #%d: text/plain documents must be provided inline, not as a URL", i)
				}
				// Append text/plain document content to the message content
				if m.Content != "" {
					m.Content += "\n" + string(data)
				} else {
					m.Content = string(data)
				}
			default:
				return fmt.Errorf("reply #%d: ollama unsupported content type %q", i, mimeType)
			}
		} else if !in.Replies[i].ToolCall.IsZero() {
			m.ToolCalls = append(m.ToolCalls, ToolCall{})
			if err := m.ToolCalls[len(m.ToolCalls)-1].From(&in.Replies[i].ToolCall); err != nil {
				return fmt.Errorf("reply #%d: %w", i, err)
			}
		} else {
			return fmt.Errorf("reply #%d: unknown Reply type", i)
		}
	}
	if len(in.ToolCallResults) != 0 {
		// Ollama doesn't use tool ID nor name in the result, hence only one tool can be called at a time.
		// Process only the first tool call result in this method.
		// The Init method handles multiple tool call results by creating multiple messages.
		m.Content = in.ToolCallResults[0].Result
	}
	return nil
}

func (m *Message) To(out *genai.Message) error {
	if m.Content != "" {
		out.Replies = []genai.Reply{{Text: m.Content}}
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
	Function struct {
		Index     int64  `json:"index,omitzero"`
		Name      string `json:"name"`
		Arguments any    `json:"arguments"`
	} `json:"function"`
}

func (t *ToolCall) From(in *genai.ToolCall) error {
	if len(in.Opaque) != 0 {
		return errors.New("field ToolCall.Opaque not supported")
	}
	t.Function.Name = in.Name
	return json.Unmarshal([]byte(in.Arguments), &t.Function.Arguments)
}

func (t *ToolCall) To(out *genai.ToolCall) error {
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

// ChatResponse is somewhat documented at https://github.com/ollama/ollama/blob/main/docs/api.md#response-10
// https://pkg.go.dev/github.com/ollama/ollama/api#ChatResponse
type ChatResponse struct {
	Model      string     `json:"model"`
	CreatedAt  time.Time  `json:"created_at"`
	Message    Message    `json:"message"`
	DoneReason DoneReason `json:"done_reason"`
	Done       bool       `json:"done"`

	// 	https://pkg.go.dev/github.com/ollama/ollama/api#Metrics
	TotalDuration      time.Duration `json:"total_duration"`
	LoadDuration       time.Duration `json:"load_duration"`
	PromptEvalCount    int64         `json:"prompt_eval_count"`
	PromptEvalDuration time.Duration `json:"prompt_eval_duration"`
	EvalCount          int64         `json:"eval_count"`
	EvalDuration       time.Duration `json:"eval_duration"`
}

func (c *ChatResponse) ToResult() (genai.Result, error) {
	out := genai.Result{
		// TODO: llama-server supports caching and we should report it.
		Usage: genai.Usage{
			InputTokens:  c.PromptEvalCount,
			OutputTokens: c.EvalCount,
			FinishReason: c.DoneReason.ToFinishReason(),
		},
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

const (
	DoneStop   DoneReason = "stop"
	DoneLength DoneReason = "length"
	// See https://pkg.go.dev/github.com/ollama/ollama/server#Server.ChatHandler
	DoneLoad   DoneReason = "load"
	DoneUnload DoneReason = "unload"
)

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

func (m *Model) GetID() string {
	return m.Name
}

func (m *Model) String() string {
	return fmt.Sprintf("%s (%s)", m.Name, m.Details.QuantizationLevel)
}

func (m *Model) Context() int64 {
	return 0
}

type ModelsResponse struct {
	Models []Model `json:"models"`
}

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

type Version struct {
	Version string `json:"version"`
}

//

type ErrorResponse struct {
	ErrorVal string `json:"error"`
}

func (er *ErrorResponse) Error() string {
	return er.ErrorVal
}

func (er *ErrorResponse) IsAPIError() bool {
	return true
}

//

// We cannot use ClientChat because GenSync and GenStream try to pull on first failure, and GenStream receives
// line separated JSON instead of SSE.

// Client implements genai.Provider.
type Client struct {
	impl    base.ProviderBase[*ErrorResponse]
	baseURL string
	chatURL string
}

// New creates a new client to talk to the Ollama API.
//
// Options Remote defaults to "http://localhost:11434".
//
// Ollama doesn't have any mean of authentication so options APIKey is not supported.
//
// To use multiple models, create multiple clients.
// Use one of the model from https://ollama.com/library
//
// Automatic model selection via ModelCheap, ModelGood, ModelSOTA is using hardcoded models. Before using an
// hardcoded model ID, it will ask ollama to determine if a model is already loaded and it will use that
// instead.
//
// wrapper optionally wraps the HTTP transport. Useful for HTTP recording and playback, or to tweak HTTP
// retries, or to throttle outgoing requests.
func New(ctx context.Context, opts *genai.ProviderOptions, wrapper func(http.RoundTripper) http.RoundTripper) (*Client, error) {
	if opts.APIKey != "" {
		return nil, errors.New("unexpected option APIKey")
	}
	if opts.AccountID != "" {
		return nil, errors.New("unexpected option AccountID")
	}
	baseURL := opts.Remote
	if baseURL == "" {
		baseURL = "http://localhost:11434"
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
		impl: base.ProviderBase[*ErrorResponse]{
			Lenient: internal.BeLenient,
			Client: http.Client{
				Transport: &roundtrippers.RequestID{Transport: t},
			},
		},
		baseURL: baseURL,
		chatURL: baseURL + "/api/chat",
	}
	switch opts.Model {
	case genai.ModelNone:
	case genai.ModelCheap, genai.ModelGood, genai.ModelSOTA, "":
		c.impl.Model = c.selectBestTextModel(ctx, opts.Model)
		c.impl.OutputModalities = mod
	default:
		c.impl.Model = opts.Model
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
	case genai.ModelCheap:
		return "gemma3:1b"
	default:
		fallthrough
	case genai.ModelGood, "":
		return "qwen3:30b"
	case genai.ModelSOTA:
		return "qwen3:32b"
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

// Scoreboard implements scoreboard.ProviderScore.
func (c *Client) Scoreboard() scoreboard.Score {
	return Scoreboard
}

// GenSync implements genai.Provider.
func (c *Client) GenSync(ctx context.Context, msgs genai.Messages, opts ...genai.Options) (genai.Result, error) {
	result := genai.Result{}
	in := ChatRequest{}
	var continuableErr error
	if err := in.Init(msgs, c.impl.Model, opts...); err != nil {
		if uce, ok := err.(*genai.UnsupportedContinuableError); ok {
			continuableErr = uce
		} else {
			return result, err
		}
	}
	var out ChatResponse
	if err := c.GenSyncRaw(ctx, &in, &out); err != nil {
		return result, err
	}
	result, err := out.ToResult()
	if err != nil {
		return result, err
	}
	if err = result.Validate(); err != nil {
		return result, err
	}
	return result, continuableErr
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
		if strings.Contains(err.Error(), "not found, try pulling it first") {
			if err = c.PullModel(ctx, c.impl.Model); err != nil {
				return err
			}
			// Retry.
			err = c.impl.DoRequest(ctx, "POST", c.chatURL, in, out)
		}
	}
	return err
}

// GenStream implements genai.Provider.
func (c *Client) GenStream(ctx context.Context, msgs genai.Messages, opts ...genai.Options) (iter.Seq[genai.ReplyFragment], func() (genai.Result, error)) {
	res := genai.Result{}
	var continuableErr error
	var finalErr error

	fnFragments := func(yield func(genai.ReplyFragment) bool) {
		in := ChatRequest{}
		if err := in.Init(msgs, c.impl.Model, opts...); err != nil {
			if uce, ok := err.(*genai.UnsupportedContinuableError); ok {
				continuableErr = uce
			} else {
				finalErr = err
				return
			}
		}
		chunks := make(chan ChatStreamChunkResponse, 32)
		// TODO: Replace with an iterator.
		fragments := make(chan genai.ReplyFragment)
		eg, ctx2 := errgroup.WithContext(ctx)
		eg.Go(func() error {
			err := processStreamPackets(chunks, fragments, &res)
			close(fragments)
			return err
		})
		eg.Go(func() error {
			err := c.GenStreamRaw(ctx2, &in, chunks)
			close(chunks)
			return err
		})
		for f := range fragments {
			if !yield(f) {
				break
			}
		}
		// Make sure the channel is emptied.
		for range fragments {
		}
		if err2 := eg.Wait(); err2 != nil {
			finalErr = err2
		}
	}
	fnFinish := func() (genai.Result, error) {
		if res.Usage.FinishReason == genai.FinishedStop && slices.ContainsFunc(res.Replies, func(r genai.Reply) bool { return !r.ToolCall.IsZero() }) {
			// Lie for the benefit of everyone.
			res.Usage.FinishReason = genai.FinishedToolCalls
		}
		if finalErr != nil {
			return res, finalErr
		}
		return res, continuableErr
	}
	return fnFragments, fnFinish
}

// GenStreamRaw provides access to the raw API.
func (c *Client) GenStreamRaw(ctx context.Context, in *ChatRequest, out chan<- ChatStreamChunkResponse) error {
	if err := c.Validate(); err != nil {
		return err
	}
	in.Stream = true
	// Try first, if it immediately errors out requesting to pull, pull then try again.
	resp, err := c.impl.JSONRequest(ctx, "POST", c.chatURL, in)
	if err != nil {
		return fmt.Errorf("failed to get server response: %w", err)
	}
	err = processJSONStream(resp.Body, out, c.impl.Lenient)
	_ = resp.Body.Close()
	if err == nil || !strings.Contains(err.Error(), "not found, try pulling it first") {
		return err
	}
	// Model was not present. Try to pull then rerun again.
	if err = c.PullModel(ctx, c.impl.Model); err != nil {
		return err
	}
	if resp, err = c.impl.JSONRequest(ctx, "POST", c.chatURL, in); err != nil {
		return fmt.Errorf("failed to get server response: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		return c.impl.DecodeError(c.chatURL, resp)
	}
	return processJSONStream(resp.Body, out, c.impl.Lenient)
}

// ListModels implements genai.Provider.
func (c *Client) ListModels(ctx context.Context) ([]genai.Model, error) {
	// https://github.com/ollama/ollama/blob/main/docs/api.md#list-local-models
	var resp ModelsResponse
	if err := c.impl.DoRequest(ctx, "GET", c.baseURL+"/api/tags", nil, &resp); err != nil {
		return nil, err
	}
	return resp.ToModels(), nil
}

// PullModel is the equivalent of "ollama pull".
//
// Files are cached under $HOME/.ollama/models/manifests/registry.ollama.ai/library/ or $OLLAMA_MODELS
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

func (c *Client) Version(ctx context.Context) (string, error) {
	v := Version{}
	if err := c.impl.DoRequest(ctx, "GET", c.baseURL+"/api/version", nil, &v); err != nil {
		return v.Version, fmt.Errorf("failed to get version: %w", err)
	}
	return v.Version, nil
}

func (c *Client) Ping(ctx context.Context) error {
	_, err := c.Version(ctx)
	return err
}

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
			return fmt.Errorf("failed to get server response: %w", err)
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
				return fmt.Errorf("failed to decode server response %q: %w", string(line), err)
			}
			return &er
		}
		out <- msg
	}
}

func processStreamPackets(ch <-chan ChatStreamChunkResponse, chunks chan<- genai.ReplyFragment, result *genai.Result) error {
	defer func() {
		// We need to empty the channel to avoid blocking the goroutine.
		for range ch {
		}
	}()
	for pkt := range ch {
		if pkt.EvalCount != 0 {
			result.Usage.InputTokens = pkt.PromptEvalCount
			result.Usage.OutputTokens = pkt.EvalCount
			result.Usage.FinishReason = pkt.DoneReason.ToFinishReason()
		}
		switch role := pkt.Message.Role; role {
		case "", "assistant":
		default:
			return fmt.Errorf("unexpected role %q", role)
		}
		for i := range pkt.Message.ToolCalls {
			f := genai.ReplyFragment{}
			if err := pkt.Message.ToolCalls[i].To(&f.ToolCall); err != nil {
				return err
			}
			if err := result.Accumulate(f); err != nil {
				return err
			}
			chunks <- f
		}
		f := genai.ReplyFragment{TextFragment: pkt.Message.Content}
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
	_ genai.Provider           = &Client{}
	_ scoreboard.ProviderScore = &Client{}
)
