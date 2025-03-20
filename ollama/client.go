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
	"log/slog"
	"net/http"
	"strings"
	"time"

	"github.com/invopop/jsonschema"
	"github.com/maruel/genai"
	"github.com/maruel/httpjson"
	"golang.org/x/sync/errgroup"
)

// https://github.com/ollama/ollama/blob/main/docs/api.md#generate-a-chat-completion
// https://pkg.go.dev/github.com/ollama/ollama/api#ChatRequest
type CompletionRequest struct {
	Model    string             `json:"model"`
	Stream   bool               `json:"stream"`
	Messages []Message          `json:"messages"`
	Tools    []Tool             `json:"tools,omitzero"`
	Format   *jsonschema.Schema `json:"format,omitzero"`
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
func (c *CompletionRequest) Init(msgs genai.Messages, opts genai.Validatable) error {
	var errs []error
	sp := ""
	if opts != nil {
		if err := opts.Validate(); err != nil {
			errs = append(errs, err)
		} else {
			switch v := opts.(type) {
			case *genai.CompletionOptions:
				c.Options.NumPredict = v.MaxTokens
				c.Options.Temperature = v.Temperature
				c.Options.TopP = v.TopP
				sp = v.SystemPrompt
				c.Options.Seed = v.Seed
				c.Options.TopK = v.TopK
				c.Options.Stop = v.Stop
				if v.ReplyAsJSON && v.DecodeAs == nil {
					return errors.New("ollama only supports structured JSON response. Use DecodeAs")
				}
				if v.DecodeAs != nil {
					c.Format = jsonschema.Reflect(v.DecodeAs)
				}
				if len(v.Tools) != 0 {
					c.Tools = make([]Tool, len(v.Tools))
					for i, t := range v.Tools {
						c.Tools[i].Type = "function"
						c.Tools[i].Function.Name = t.Name
						c.Tools[i].Function.Description = t.Description
						if t.InputsAs != nil {
							c.Tools[i].Function.Parameters = jsonschema.Reflect(t.InputsAs)
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
	return errors.Join(errs...)
}

// https://github.com/ollama/ollama/blob/main/docs/api.md#parameters-1
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
type CompletionResponse struct {
	Model      string    `json:"model"`
	CreatedAt  time.Time `json:"created_at"`
	Message    Message   `json:"message"`
	DoneReason string    `json:"done_reason"`
	Done       bool      `json:"done"`

	// 	https://pkg.go.dev/github.com/ollama/ollama/api#Metrics
	TotalDuration      time.Duration `json:"total_duration"`
	LoadDuration       time.Duration `json:"load_duration"`
	PromptEvalCount    int64         `json:"prompt_eval_count"`
	PromptEvalDuration time.Duration `json:"prompt_eval_duration"`
	EvalCount          int64         `json:"eval_count"`
	EvalDuration       time.Duration `json:"eval_duration"`
}

func (c *CompletionResponse) ToResult() (genai.CompletionResult, error) {
	out := genai.CompletionResult{
		Usage: genai.Usage{
			InputTokens:  c.PromptEvalCount,
			OutputTokens: c.EvalCount,
		},
	}
	err := c.Message.To(&out.Message)
	return out, err
}

type CompletionStreamChunkResponse CompletionResponse

//

type errorResponse struct {
	Error string `json:"error"`
}

//

// Client implements the REST JSON based API.
type Client struct {
	// Client is exported for testing replay purposes.
	Client httpjson.Client

	model   string
	baseURL string
}

// New creates a new client to talk to the Ollama API.
//
// To use multiple models, create multiple clients.
// Use one of the model from https://ollama.com/library
func New(baseURL, model string) (*Client, error) {
	return &Client{baseURL: baseURL, model: model}, nil
}

func (c *Client) Completion(ctx context.Context, msgs genai.Messages, opts genai.Validatable) (genai.CompletionResult, error) {
	rpcin := CompletionRequest{Model: c.model}
	if err := rpcin.Init(msgs, opts); err != nil {
		return genai.CompletionResult{}, err
	}
	rpcout := CompletionResponse{}
	if err := c.CompletionRaw(ctx, &rpcin, &rpcout); err != nil {
		return genai.CompletionResult{}, fmt.Errorf("failed to get chat response: %w", err)
	}
	return rpcout.ToResult()
}

func (c *Client) CompletionRaw(ctx context.Context, in *CompletionRequest, out *CompletionResponse) error {
	if err := c.validate(); err != nil {
		return err
	}
	in.Stream = false
	err := c.post(ctx, c.baseURL+"/api/chat", in, out)
	if err != nil {
		// TODO: Cheezy.
		if strings.Contains(err.Error(), "not found, try pulling it first") {
			if err = c.PullModel(ctx, c.model); err != nil {
				return fmt.Errorf("failed to pull model: %w", err)
			}
			// Retry.
			err = c.post(ctx, c.baseURL+"/api/chat", in, out)
		}
	}
	return err
}

func (c *Client) CompletionStream(ctx context.Context, msgs genai.Messages, opts genai.Validatable, chunks chan<- genai.MessageFragment) error {
	in := CompletionRequest{Model: c.model}
	if err := in.Init(msgs, opts); err != nil {
		return err
	}
	ch := make(chan CompletionStreamChunkResponse)
	eg, ctx := errgroup.WithContext(ctx)
	eg.Go(func() error {
		return processStreamPackets(ch, chunks)
	})
	err := c.CompletionStreamRaw(ctx, &in, ch)
	close(ch)
	if err2 := eg.Wait(); err2 != nil {
		err = err2
	}
	return err
}

func processStreamPackets(ch <-chan CompletionStreamChunkResponse, chunks chan<- genai.MessageFragment) error {
	defer func() {
		// We need to empty the channel to avoid blocking the goroutine.
		for range ch {
		}
	}()
	for pkt := range ch {
		switch role := pkt.Message.Role; role {
		case "", "assistant":
		default:
			return fmt.Errorf("unexpected role %q", role)
		}
		if word := pkt.Message.Content; word != "" {
			chunks <- genai.MessageFragment{TextFragment: word}
		}
	}
	return nil
}

func (c *Client) CompletionStreamRaw(ctx context.Context, in *CompletionRequest, out chan<- CompletionStreamChunkResponse) error {
	if err := c.validate(); err != nil {
		return err
	}
	in.Stream = true
	// Try first, if it immediately errors out requesting to pull, pull then try again.
	var err error
	for range 2 {
		var resp *http.Response
		if resp, err = c.Client.PostRequest(ctx, c.baseURL+"/api/chat", nil, in); err != nil {
			return fmt.Errorf("failed to get server response: %w", err)
		}
		sent := false
		for r := bufio.NewReader(resp.Body); ; {
			var line []byte
			line, err = r.ReadBytes('\n')
			if line = bytes.TrimSpace(line); err == io.EOF {
				if len(line) == 0 {
					_ = resp.Body.Close()
					return nil
				}
			} else if err != nil {
				_ = resp.Body.Close()
				return fmt.Errorf("failed to get server response: %w", err)
			}
			if len(line) == 0 {
				continue
			}
			if err = parseStreamLine(line, out); err != nil {
				_ = resp.Body.Close()
				// Missing local model error.
				if !sent && strings.Contains(err.Error(), "not found, try pulling it first") {
					if err = c.PullModel(ctx, c.model); err != nil {
						return fmt.Errorf("failed to pull model: %w", err)
					}
					// Breaak of the inner for loop to retry the HTTP request.
					break
				}
				// Generic error.
				return err
			}
			// A chunk was sent. We can't retry past that point.
			sent = true
		}
	}
	return err
}

func parseStreamLine(line []byte, out chan<- CompletionStreamChunkResponse) error {
	d := json.NewDecoder(bytes.NewReader(line))
	d.DisallowUnknownFields()
	d.UseNumber()
	msg := CompletionStreamChunkResponse{}
	if err := d.Decode(&msg); err != nil {
		er := errorResponse{}
		d := json.NewDecoder(bytes.NewReader(line))
		d.DisallowUnknownFields()
		d.UseNumber()
		if err := d.Decode(&er); err != nil {
			return fmt.Errorf("failed to decode server response %q: %w", string(line), err)
		}
		return fmt.Errorf("server error: %s", er.Error)
	}
	out <- msg
	return nil
}

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

func (c *Client) ListModels(ctx context.Context) ([]genai.Model, error) {
	// https://github.com/ollama/ollama/blob/main/docs/api.md#list-local-models
	var out struct {
		Models []Model `json:"models"`
	}
	if err := c.Client.Get(ctx, c.baseURL+"/api/tags", nil, &out); err != nil {
		return nil, err
	}
	models := make([]genai.Model, len(out.Models))
	for i := range out.Models {
		models[i] = &out.Models[i]
	}
	return models, nil
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

// PullModel is the equivalent of "ollama pull".
func (c *Client) PullModel(ctx context.Context, model string) error {
	in := pullModelRequest{Model: model}
	// TODO: Stream updates instead of hanging for several minutes.
	out := pullModelResponse{}
	if err := c.post(ctx, c.baseURL+"/api/pull", &in, &out); err != nil {
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

func (c *Client) post(ctx context.Context, url string, in, out any) error {
	resp, err := c.Client.PostRequest(ctx, url, nil, in)
	if err != nil {
		return err
	}
	er := errorResponse{}
	switch i, err := httpjson.DecodeResponse(resp, out, &er); i {
	case 0:
		return nil
	case 1:
		// OpenAI error message prints the api key URL already.
		var herr *httpjson.Error
		if errors.As(err, &herr) {
			return fmt.Errorf("%w: error: %s", herr, er.Error)
		}
		return fmt.Errorf("%w: error: %s", herr, er.Error)
	default:
		var herr *httpjson.Error
		if errors.As(err, &herr) {
			slog.WarnContext(ctx, "ollama", "url", url, "err", err, "response", string(herr.ResponseBody), "status", herr.StatusCode)
		} else {
			slog.WarnContext(ctx, "ollama", "url", url, "err", err)
		}
		return err
	}
}

var (
	_ genai.CompletionProvider = &Client{}
	_ genai.ModelProvider      = &Client{}
)
