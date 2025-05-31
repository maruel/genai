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
	"net/http"
	"strings"
	"time"

	"github.com/invopop/jsonschema"
	"github.com/maruel/genai"
	"github.com/maruel/genai/internal"
	"github.com/maruel/httpjson"
	"github.com/maruel/roundtrippers"
	"golang.org/x/sync/errgroup"
)

// Scoreboard for Ollama.
//
// # Warnings
//
//   - Figure out tools as streaming support recently got added to llama.cpp.
//   - Ollama supports more than what the client supports.
var Scoreboard = genai.Scoreboard{
	Scenarios: []genai.Scenario{
		{
			In:     []genai.Modality{genai.ModalityImage, genai.ModalityText},
			Out:    []genai.Modality{genai.ModalityText},
			Models: []string{"gemma3:4b"},
			Chat: genai.Functionality{
				Inline:             true,
				URL:                false,
				Thinking:           false,
				ReportTokenUsage:   true,
				ReportFinishReason: true,
				MaxTokens:          true,
				StopSequence:       true,
				Tools:              false,
				JSON:               true,
				JSONSchema:         true,
			},
			ChatStream: genai.Functionality{
				Inline:             true,
				URL:                false,
				Thinking:           false,
				ReportTokenUsage:   true,
				ReportFinishReason: true,
				MaxTokens:          true,
				StopSequence:       true,
				Tools:              false,
				JSON:               true,
				JSONSchema:         true,
			},
		},
	},
}

// https://github.com/ollama/ollama/blob/main/docs/api.md#generate-a-chat-completion
// https://pkg.go.dev/github.com/ollama/ollama/api#ChatRequest
type ChatRequest struct {
	Model    string    `json:"model"`
	Stream   bool      `json:"stream"`
	Messages []Message `json:"messages"`
	Tools    []Tool    `json:"tools,omitzero"`
	Format   any       `json:"format,omitzero"` // Either *jsonschema.Schema or string
	// https://github.com/ollama/ollama/blob/main/docs/modelfile.md#valid-parameters-and-values
	// https://pkg.go.dev/github.com/ollama/ollama/api#Options
	// https://pkg.go.dev/github.com/ollama/ollama/api#Runner
	Options struct {
		Mirostat      int64    `json:"mirostat,omitzero"` // [0, 1, 2]
		MirostatEta   float64  `json:"mirostat_eta,omitzero"`
		MirostatTau   float64  `json:"mirostat_tau,omitzero"`
		NumCtx        int64    `json:"num_ctx,omitzero"`        // Context Window, default 2048
		RepeatLastN   int64    `json:"repeat_last_n,omitzero"`  // Lookback for repeated tokens, default 64
		RepeatPenalty float64  `json:"repeat_penalty,omitzero"` // default 1.1
		Temperature   float64  `json:"temperature,omitzero"`    // default 0.8
		Seed          int64    `json:"seed,omitzero"`
		Stop          []string `json:"stop,omitzero"`        // keywords to stop completion
		NumPredict    int64    `json:"num_predict,omitzero"` // Max tokens
		TopK          int64    `json:"top_k,omitzero"`       // Default: 40
		TopP          float64  `json:"top_p,omitzero"`       // Default: 0.9
		MinP          float64  `json:"min_p,omitzero"`       // Default: 0.0
	} `json:"options,omitzero"`
	KeepAlive string `json:"keep_alive,omitzero"` // Default "5m"
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
				c.Options.NumPredict = v.MaxTokens
				c.Options.Temperature = v.Temperature
				c.Options.TopP = v.TopP
				sp = v.SystemPrompt
				c.Options.Seed = v.Seed
				c.Options.TopK = v.TopK
				c.Options.Stop = v.Stop
				if v.DecodeAs != nil {
					c.Format = jsonschema.Reflect(v.DecodeAs)
				} else if v.ReplyAsJSON {
					c.Format = "json"
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

// https://github.com/ollama/ollama/blob/main/docs/api.md#parameters-1
//
// https://pkg.go.dev/github.com/ollama/ollama/api#Message
type Message struct {
	Role      string     `json:"role,omitzero"` // "system", "assistant", "user"
	Content   string     `json:"content,omitzero"`
	Images    [][]byte   `json:"images,omitzero"` // List of images as base64 encoded strings.
	ToolCalls []ToolCall `json:"tool_calls,omitzero"`
}

func (m *Message) From(in *genai.Message) error {
	switch in.Role {
	case genai.User, genai.Assistant:
		m.Role = string(in.Role)
	default:
		return fmt.Errorf("unsupported role %q", in.Role)
	}
	// Ollama only supports one text content per message but multiple images. We need to validate first.
	txt := 0
	for i := range in.Contents {
		if in.Contents[i].Text != "" {
			txt++
		}
	}
	if txt > 1 {
		return errors.New("ollama only supports one text content per message; todo to implement this transparently")
	}
	for i := range in.Contents {
		if in.Contents[i].Text != "" {
			m.Content = in.Contents[i].Text
		} else {
			mimeType, data, err := in.Contents[i].ReadDocument(10 * 1024 * 1024)
			if err != nil {
				return err
			}
			// Only support images.
			if !strings.HasPrefix(mimeType, "image/") {
				return fmt.Errorf("ollama unsupported content type %q", mimeType)
			}
			m.Images = append(m.Images, data)
		}
	}
	if len(in.ToolCalls) != 0 {
		m.ToolCalls = make([]ToolCall, len(in.ToolCalls))
		for i := range in.ToolCalls {
			if err := m.ToolCalls[i].From(&in.ToolCalls[i]); err != nil {
				return fmt.Errorf("tool call %d: %w", i, err)
			}
		}
	}
	if len(in.ToolCallResults) != 0 {
		if len(in.Contents) != 0 || len(in.ToolCalls) != 0 {
			// This could be worked around.
			return fmt.Errorf("can't have tool call result along content or tool calls")
		}
		// Ollama doesn't use tool ID nor name in the result, hence only one tool can be called at a time.
		if len(in.ToolCallResults) != 1 {
			return fmt.Errorf("can't have more than one tool call result at a time")
		}
		m.Role = "tool"
		m.Content = in.ToolCallResults[0].Result
	}
	return nil
}

func (m *Message) To(out *genai.Message) error {
	switch m.Role {
	case "assistant", "user":
		out.Role = genai.Role(m.Role)
	default:
		return fmt.Errorf("unsupported role %q", m.Role)
	}
	if m.Content != "" {
		out.Contents = []genai.Content{{Text: m.Content}}
	}
	for i := range m.Images {
		out.Contents = append(out.Contents, genai.Content{Filename: "image.jpg", Document: bytes.NewReader(m.Images[i])})
	}
	if len(m.ToolCalls) != 0 {
		out.ToolCalls = make([]genai.ToolCall, len(m.ToolCalls))
		for i := range m.ToolCalls {
			if err := m.ToolCalls[i].To(&out.ToolCalls[i]); err != nil {
				return fmt.Errorf("tool call %d: %w", i, err)
			}
		}
	}
	return nil
}

// https://github.com/ollama/ollama/blob/main/docs/api.md#response-16
// https://pkg.go.dev/github.com/ollama/ollama/api#ToolCall
type ToolCall struct {
	Function struct {
		Index     int64  `json:"index,omitzero"`
		Name      string `json:"name"`
		Arguments any    `json:"arguments"`
	} `json:"function"`
}

func (t *ToolCall) From(in *genai.ToolCall) error {
	t.Function.Name = in.Name
	return json.Unmarshal([]byte(in.Arguments), &t.Function.Arguments)
}

func (t *ToolCall) To(out *genai.ToolCall) error {
	out.Name = t.Function.Name
	b, err := json.Marshal(t.Function.Arguments)
	out.Arguments = string(b)
	return err
}

// https://github.com/ollama/ollama/blob/main/docs/api.md#chat-request-with-tools
// https://pkg.go.dev/github.com/ollama/ollama/api#Tool
type Tool struct {
	Type     string `json:"type,omitzero"` // "function"
	Function struct {
		Description string             `json:"description,omitzero"`
		Name        string             `json:"name,omitzero"`
		Parameters  *jsonschema.Schema `json:"parameters,omitzero"`
	} `json:"function,omitzero"`
}

// https://github.com/ollama/ollama/blob/main/docs/api.md#response-10
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

func (c *ChatResponse) ToResult() (genai.ChatResult, error) {
	out := genai.ChatResult{
		// TODO: llama-server supports caching and we should report it.
		Usage: genai.Usage{
			InputTokens:  c.PromptEvalCount,
			OutputTokens: c.EvalCount,
			FinishReason: c.DoneReason.ToFinishReason(),
		},
	}
	err := c.Message.To(&out.Message)
	if len(out.ToolCalls) != 0 && out.FinishReason == genai.FinishedStop {
		// Lie for the benefit of everyone.
		out.FinishReason = genai.FinishedToolCalls
	}
	return out, err
}

type DoneReason string

const (
	DoneStop DoneReason = "stop"
)

func (d DoneReason) ToFinishReason() genai.FinishReason {
	return genai.FinishReason(d)
}

type ChatStreamChunkResponse ChatResponse

// https://pkg.go.dev/github.com/ollama/ollama/api#ListModelResponse
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

//

type ErrorResponse struct {
	Error string `json:"error"`
}

func (er *ErrorResponse) String() string {
	return "error " + er.Error
}

//

// We cannot use ClientChat because Chat and ChatStream try to pull on first failure, and ChatStream receives
// line separated JSON instead of SSE.

// Client implements genai.ProviderChat.
type Client struct {
	internal.ClientBase[*ErrorResponse]

	model   string
	baseURL string
	chatURL string
}

// New creates a new client to talk to the Ollama API.
//
// To use multiple models, create multiple clients.
// Use one of the model from https://ollama.com/library
//
// r can be used to throttle outgoing requests, record calls, etc. It defaults to http.DefaultTransport.
func New(baseURL, model string, r http.RoundTripper) (*Client, error) {
	if r == nil {
		r = http.DefaultTransport
	}
	return &Client{
		ClientBase: internal.ClientBase[*ErrorResponse]{
			ClientJSON: httpjson.Client{
				Client: &http.Client{
					Transport: &roundtrippers.Retry{
						Transport: &roundtrippers.RequestID{
							Transport: r,
						},
					},
				},
				Lenient: internal.BeLenient,
			},
		},
		baseURL: baseURL,
		chatURL: baseURL + "/api/chat",
		model:   model,
	}, nil
}

func (c *Client) Name() string {
	return "ollama"
}

func (c *Client) Scoreboard() genai.Scoreboard {
	return Scoreboard
}

func (c *Client) Chat(ctx context.Context, msgs genai.Messages, opts genai.Validatable) (genai.ChatResult, error) {
	result := genai.ChatResult{}
	for i, msg := range msgs {
		for j, content := range msg.Contents {
			if len(content.Opaque) != 0 {
				return result, fmt.Errorf("message #%d content #%d: field Opaque not supported", i, j)
			}
		}
	}

	var in ChatRequest
	var continuableErr error
	if err := in.Init(msgs, opts, c.model); err != nil {
		if uce, ok := err.(*genai.UnsupportedContinuableError); ok {
			continuableErr = uce
		} else {
			return result, err
		}
	}
	var out ChatResponse
	if err := c.ChatRaw(ctx, &in, &out); err != nil {
		return result, err
	}
	result, err := out.ToResult()
	if err != nil {
		return result, err
	}
	return result, continuableErr
}

func (c *Client) ChatRaw(ctx context.Context, in *ChatRequest, out *ChatResponse) error {
	if err := c.validate(); err != nil {
		return err
	}
	in.Stream = false
	err := c.DoRequest(ctx, "POST", c.chatURL, in, out)
	if err != nil {
		// TODO: Cheezy.
		if strings.Contains(err.Error(), "not found, try pulling it first") {
			if err = c.PullModel(ctx, c.model); err != nil {
				return err
			}
			// Retry.
			err = c.DoRequest(ctx, "POST", c.chatURL, in, out)
		}
	}
	return err
}

func (c *Client) ChatStream(ctx context.Context, msgs genai.Messages, opts genai.Validatable, chunks chan<- genai.MessageFragment) (genai.ChatResult, error) {
	result := genai.ChatResult{}
	for i, msg := range msgs {
		for j, content := range msg.Contents {
			if len(content.Opaque) != 0 {
				return result, fmt.Errorf("message #%d content #%d: field Opaque not supported", i, j)
			}
		}
	}

	in := ChatRequest{}
	var continuableErr error
	if err := in.Init(msgs, opts, c.model); err != nil {
		if uce, ok := err.(*genai.UnsupportedContinuableError); ok {
			continuableErr = uce
		} else {
			return result, err
		}
	}
	ch := make(chan ChatStreamChunkResponse)
	eg, ctx := errgroup.WithContext(ctx)
	eg.Go(func() error {
		return processStreamPackets(ch, chunks, &result)
	})
	err := c.ChatStreamRaw(ctx, &in, ch)
	close(ch)
	if err2 := eg.Wait(); err2 != nil {
		err = err2
	}
	if len(result.ToolCalls) != 0 && result.FinishReason == genai.FinishedStop {
		// Lie for the benefit of everyone.
		result.FinishReason = genai.FinishedToolCalls
	}
	if err != nil {
		return result, err
	}
	return result, continuableErr
}

func (c *Client) ChatStreamRaw(ctx context.Context, in *ChatRequest, out chan<- ChatStreamChunkResponse) error {
	if err := c.validate(); err != nil {
		return err
	}
	in.Stream = true
	// Try first, if it immediately errors out requesting to pull, pull then try again.
	resp, err := c.ClientJSON.Request(ctx, "POST", c.chatURL, nil, in)
	if err != nil {
		return fmt.Errorf("failed to get server response: %w", err)
	}
	err = processJSONStream(resp.Body, out, c.ClientJSON.Lenient)
	_ = resp.Body.Close()
	if err == nil || !strings.Contains(err.Error(), "not found, try pulling it first") {
		return err
	}
	// Model was not present. Try to pull then rerun again.
	if err = c.PullModel(ctx, c.model); err != nil {
		return err
	}
	if resp, err = c.ClientJSON.Request(ctx, "POST", c.chatURL, nil, in); err != nil {
		return fmt.Errorf("failed to get server response: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		return c.DecodeError(ctx, c.chatURL, resp)
	}
	return processJSONStream(resp.Body, out, c.ClientJSON.Lenient)
}

func (c *Client) ListModels(ctx context.Context) ([]genai.Model, error) {
	// https://github.com/ollama/ollama/blob/main/docs/api.md#list-local-models
	return internal.ListModels[*ErrorResponse, *ModelsResponse](ctx, &c.ClientBase, c.baseURL+"/api/tags")
}

func (c *Client) ModelID() string {
	return c.model
}

// PullModel is the equivalent of "ollama pull".
//
// Files are cached under $HOME/.ollama/models/manifests/registry.ollama.ai/library/ or $OLLAMA_MODELS
func (c *Client) PullModel(ctx context.Context, model string) error {
	in := pullModelRequest{Model: model}
	// TODO: Stream updates instead of hanging for several minutes.
	out := pullModelResponse{}
	if err := c.DoRequest(ctx, "POST", c.baseURL+"/api/pull", &in, &out); err != nil {
		return fmt.Errorf("pull failed: %w", err)
	} else if out.Status != "success" {
		return fmt.Errorf("pull failed: %s", out.Status)
	}
	return nil
}

func (c *Client) validate() error {
	if c.model == "" {
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
			return errors.New(er.String())
		}
		out <- msg
	}
}

func processStreamPackets(ch <-chan ChatStreamChunkResponse, chunks chan<- genai.MessageFragment, result *genai.ChatResult) error {
	defer func() {
		// We need to empty the channel to avoid blocking the goroutine.
		for range ch {
		}
	}()
	for pkt := range ch {
		if pkt.EvalCount != 0 {
			result.InputTokens = pkt.PromptEvalCount
			result.OutputTokens = pkt.EvalCount
			result.FinishReason = pkt.DoneReason.ToFinishReason()
		}
		switch role := pkt.Message.Role; role {
		case "", "assistant":
		default:
			return fmt.Errorf("unexpected role %q", role)
		}
		for i := range pkt.Message.ToolCalls {
			f := genai.MessageFragment{}
			if err := pkt.Message.ToolCalls[i].To(&f.ToolCall); err != nil {
				return err
			}
			if err := result.Accumulate(f); err != nil {
				return err
			}
			chunks <- f
		}
		f := genai.MessageFragment{TextFragment: pkt.Message.Content}
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
