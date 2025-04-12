// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Package mistral implements a client for the Mistral API.
//
// It is described at https://docs.mistral.ai/api/
package mistral

import (
	"bufio"
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/invopop/jsonschema"
	"github.com/maruel/genai"
	"github.com/maruel/genai/internal"
	"github.com/maruel/httpjson"
	"github.com/maruel/roundtrippers"
	"golang.org/x/sync/errgroup"
)

// https://docs.mistral.ai/api/#tag/chat/operation/chat_completion_v1_chat_completions_post
type ChatRequest struct {
	Model          string    `json:"model"`
	Temperature    float64   `json:"temperature,omitzero"` // [0, 2]
	TopP           float64   `json:"top_p,omitzero"`       // [0, 1]
	MaxTokens      int64     `json:"max_tokens,omitzero"`
	Stream         bool      `json:"stream"`
	Stop           []string  `json:"stop,omitzero"` // keywords to stop completion
	RandomSeed     int64     `json:"random_seed,omitzero"`
	Messages       []Message `json:"messages"`
	ResponseFormat struct {
		Type       string `json:"type,omitzero"` // "text", "json_object", "json_schema"
		JSONSchema struct {
			Name        string             `json:"name,omitzero"`
			Description string             `json:"description,omitzero"`
			Strict      bool               `json:"strict,omitzero"`
			Schema      *jsonschema.Schema `json:"schema,omitzero"`
		} `json:"json_schema,omitzero"`
	} `json:"response_format,omitzero"`
	Tools []Tool `json:"tools,omitzero"`
	// Alternative when forcing a specific function. This can probably be achieved
	// by providing a single tool and ToolChoice == "required".
	// ToolChoice struct {
	// 	Type     string `json:"type,omitzero"` // "function"
	// 	Function struct {
	// 		Name string `json:"name,omitzero"`
	// 	} `json:"function,omitzero"`
	// } `json:"tool_choice,omitzero"`
	ToolChoice       string  `json:"tool_choice,omitzero"`       // "auto", "none", "any", "required"
	PresencePenalty  float64 `json:"presence_penalty,omitzero"`  // [-2.0, 2.0]
	FrequencyPenalty float64 `json:"frequency_penalty,omitzero"` // [-2.0, 2.0]
	N                int64   `json:"n,omitzero"`                 // Number of choices
	Prediction       struct {
		// Enable users to specify expected results, optimizing response times by
		// leveraging known or predictable content. This approach is especially
		// effective for updating text documents or code files with minimal
		// changes, reducing latency while maintaining high-quality results.
		Type    string `json:"type,omitzero"` // "content"
		Content string `json:"content,omitzero"`
	} `json:"prediction,omitzero"`
	SafePrompt bool `json:"safe_prompt,omitzero"`

	// See https://docs.mistral.ai/capabilities/document/
	DocumentImageLimit int64 `json:"document_image_limit,omitzero"`
	DocumentPageLimit  int64 `json:"document_page_limit,omitzero"`
	IncludeImageBase64 bool  `json:"include_image_base64,omitzero"`
}

// Init initializes the provider specific completion request with the generic completion request.
func (c *ChatRequest) Init(msgs genai.Messages, opts genai.Validatable) error {
	var errs []error
	sp := ""
	if opts != nil {
		if err := opts.Validate(); err != nil {
			errs = append(errs, err)
		} else {
			switch v := opts.(type) {
			case *genai.ChatOptions:
				c.MaxTokens = v.MaxTokens
				c.Temperature = v.Temperature
				c.TopP = v.TopP
				sp = v.SystemPrompt
				c.RandomSeed = v.Seed
				if v.TopK != 0 {
					errs = append(errs, errors.New("mistral does not support TopK"))
				}
				c.Stop = v.Stop
				if v.ReplyAsJSON {
					c.ResponseFormat.Type = "json_object"
				}
				if v.DecodeAs != nil {
					c.ResponseFormat.Type = "json_schema"
					// Mistral requires a name.
					c.ResponseFormat.JSONSchema.Name = "response"
					c.ResponseFormat.JSONSchema.Strict = true
					c.ResponseFormat.JSONSchema.Schema = jsonschema.Reflect(v.DecodeAs)
				}
				if len(v.Tools) != 0 {
					// Let's assume if the user provides tools, they want to use them.
					c.ToolChoice = "required"
					c.Tools = make([]Tool, len(v.Tools))
					for i, t := range v.Tools {
						c.Tools[i].Type = "function"
						c.Tools[i].Function.Name = t.Name
						c.Tools[i].Function.Description = t.Description
						if t.InputsAs != nil {
							c.Tools[i].Function.Parameters = jsonschema.Reflect(t.InputsAs)
						}
						// This costs a lot more.
						c.Tools[i].Function.Strict = true
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
			c.Messages[0].Content = []Content{{Type: "text", Text: sp}}
		}
		for i := range msgs {
			if err := c.Messages[i+offset].From(&msgs[i]); err != nil {
				errs = append(errs, fmt.Errorf("message %d: %w", i, err))
			}
		}
	}
	return errors.Join(errs...)
}

// https://docs.mistral.ai/api/#tag/chat/operation/chat_completion_v1_chat_completions_post
type Message struct {
	Role    string    `json:"role"` // "system", "assistant", "user"
	Content []Content `json:"content"`
}

func (m *Message) From(in *genai.Message) error {
	switch in.Role {
	case genai.User, genai.Assistant:
		m.Role = string(in.Role)
	default:
		return fmt.Errorf("unsupported role %q", in.Role)
	}
	if len(in.Contents) != 0 {
		m.Content = make([]Content, len(in.Contents))
		for i := range in.Contents {
			if err := m.Content[i].From(&in.Contents[i]); err != nil {
				return fmt.Errorf("block %d: %w", i, err)
			}
		}
	}
	if len(in.ToolCalls) != 0 {
		return errors.New("mistral supports tool calls, we need to implement it")
	}
	return nil
}

type Content struct {
	Type string `json:"type"` // "text", "reference", "document_url", "image_url"

	// Type == "text"
	Text string `json:"text,omitzero"`

	// Type == "reference"
	ReferenceIDs []int64 `json:"reference_ids,omitzero"`

	// Type == "document_url"
	DocumentURL  string `json:"document_url,omitzero"`
	DocumentName string `json:"document_name,omitzero"`

	// Type == "image_url"
	ImageURL struct {
		URL    string `json:"url,omitzero"`
		Detail string `json:"detail,omitzero"` // undocumented, likely "auto" like OpenAI
	} `json:"image_url,omitzero"`
}

func (c *Content) From(in *genai.Content) error {
	if in.Text != "" {
		c.Type = "text"
		c.Text = in.Text
		return nil
	}
	mimeType, data, err := in.ReadDocument(10 * 1024 * 1024)
	if err != nil {
		return err
	}
	switch {
	case (in.URL != "" && mimeType == "") || strings.HasPrefix(mimeType, "image/"):
		c.Type = "image_url"
		if in.URL == "" {
			c.ImageURL.URL = fmt.Sprintf("data:%s;base64,%s", mimeType, base64.StdEncoding.EncodeToString(data))
		} else {
			c.ImageURL.URL = in.URL
		}
	case mimeType == "application/pdf":
		c.Type = "document_url"
		if in.URL == "" {
			return errors.New("unsupported inline document")
		}
		c.DocumentURL = in.URL
	default:
		return fmt.Errorf("unsupported mime type %s", mimeType)
	}
	return nil
}

type Tool struct {
	Type     string `json:"type,omitzero"` // "function"
	Function struct {
		Name        string             `json:"name,omitzero"`
		Description string             `json:"description,omitzero"`
		Strict      bool               `json:"strict,omitzero"`
		Parameters  *jsonschema.Schema `json:"parameters,omitzero"`
	} `json:"function,omitzero"`
}

type ChatResponse struct {
	ID      string `json:"id"`
	Object  string `json:"object"` // "chat.completion"
	Model   string `json:"model"`
	Created Time   `json:"created"`
	Choices []struct {
		// FinishReason is one of "stop", "length", "content_filter" or "tool_calls".
		FinishReason string          `json:"finish_reason"`
		Index        int64           `json:"index"`
		Message      MessageResponse `json:"message"`
		Logprobs     struct{}        `json:"logprobs"`
	} `json:"choices"`
	Usage struct {
		PromptTokens int64 `json:"prompt_tokens"`
		ChatTokens   int64 `json:"completion_tokens"`
		TotalTokens  int64 `json:"total_tokens"`
	} `json:"usage"`
}

type MessageResponse struct {
	Role      string     `json:"role"`
	Content   string     `json:"content"`
	Prefix    bool       `json:"prefix"`
	ToolCalls []ToolCall `json:"tool_calls"`
}

func (m *MessageResponse) To(out *genai.Message) error {
	switch role := m.Role; role {
	case "system", "assistant", "user":
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
	if m.Content != "" {
		out.Contents = []genai.Content{{Text: m.Content}}
	}
	return nil
}

type ToolCall struct {
	ID       string `json:"id"`
	Type     string `json:"type"` // Omitted
	Function struct {
		Name      string `json:"name"`
		Arguments string `json:"arguments"`
	} `json:"function"`
	Index int64 `json:"index"`
}

func (t *ToolCall) From(in *genai.ToolCall) {
	t.Type = "function"
	t.ID = in.ID
	t.Function.Name = in.Name
	t.Function.Arguments = in.Arguments
	t.Index = 0 // Unsure
}

func (t *ToolCall) To(out *genai.ToolCall) {
	out.ID = t.ID
	out.Name = t.Function.Name
	out.Arguments = t.Function.Arguments
}

func (c *ChatResponse) ToResult() (genai.ChatResult, error) {
	out := genai.ChatResult{
		Usage: genai.Usage{
			InputTokens:  c.Usage.PromptTokens,
			OutputTokens: c.Usage.ChatTokens,
		},
	}
	if len(c.Choices) != 1 {
		return out, fmt.Errorf("server returned an unexpected number of choices, expected 1, got %d", len(c.Choices))
	}
	err := c.Choices[0].Message.To(&out.Message)
	return out, err
}

type ChatStreamChunkResponse struct {
	ID      string `json:"id"`
	Object  string `json:"object"` // chat.completion.chunk
	Created Time   `json:"created"`
	Model   string `json:"model"`
	Choices []struct {
		Index int64 `json:"index"`
		Delta struct {
			Role    genai.Role `json:"role"`
			Content string     `json:"content"`
		} `json:"delta"`
		FinishReason string `json:"finish_reason"`
	} `json:"choices"`
	Usage struct {
		PromptTokens int64 `json:"prompt_tokens"`
		ChatTokens   int64 `json:"completion_tokens"`
		TotalTokens  int64 `json:"total_tokens"`
	} `json:"usage"`
}

//

// errorResponseAuth is used when simple issue like auth failure.
type errorResponseAuth struct {
	Message   string `json:"message"`
	RequestID string `json:"request_id"`
}

type errorResponseAPI1 struct {
	Object  string `json:"object"` // error
	Message string `json:"message"`
	Type    string `json:"type"`
	Param   string `json:"param"`
	Code    int64  `json:"code"`
}

type errorResponseAPI2 struct {
	Detail []struct {
		Type string `json:"type"` // "string_type", "missing"
		Msg  string `json:"msg"`
		Loc  []any  `json:"loc"` // to be joined, a mix of string and nuber
		// Input is either a list or an instance of struct { Type string `json:"type"` }.
		Input any    `json:"input"`
		Ctx   any    `json:"ctx"`
		URL   string `json:"url"`
	} `json:"detail"`
}

type errorResponseAPI3 struct {
	Object  string            `json:"object"` // error
	Message errorResponseAPI2 `json:"message"`
	Type    string            `json:"type"`
	Param   string            `json:"param"`
	Code    int64             `json:"code"`
}

// Client implements the REST JSON based API.
type Client struct {
	// Client is exported for testing replay purposes.
	Client httpjson.Client

	model string
}

// TODO:
// https://codestral.mistral.ai/v1/fim/completions
// https://codestral.mistral.ai/v1/chat/completions

// New creates a new client to talk to the Mistral platform API.
//
// If apiKey is not provided, it tries to load it from the MISTRAL_API_KEY environment variable.
// If none is found, it returns an error.
// Get your API key at https://console.mistral.ai/api-keys or https://console.mistral.ai/codestral
// If no model is provided, only functions that do not require a model, like ListModels, will work.
// To use multiple models, create multiple clients.
// Use one of the model from https://docs.mistral.ai/getting-started/models/models_overview/
func New(apiKey, model string) (*Client, error) {
	if apiKey == "" {
		if apiKey = os.Getenv("MISTRAL_API_KEY"); apiKey == "" {
			return nil, errors.New("mistral API key is required; get one at " + apiKeyURL)
		}
	}
	return &Client{
		model: model,
		Client: httpjson.Client{
			Client: &http.Client{Transport: &roundtrippers.Header{
				Transport: http.DefaultTransport,
				Header:    http.Header{"Authorization": {"Bearer " + apiKey}},
			}},
			Lenient: internal.BeLenient,
		},
	}, nil
}

func (c *Client) Chat(ctx context.Context, msgs genai.Messages, opts genai.Validatable) (genai.ChatResult, error) {
	rpcin := ChatRequest{Model: c.model}
	if err := rpcin.Init(msgs, opts); err != nil {
		return genai.ChatResult{}, err
	}
	rpcout := ChatResponse{}
	if err := c.ChatRaw(ctx, &rpcin, &rpcout); err != nil {
		return genai.ChatResult{}, fmt.Errorf("failed to get chat response: %w", err)
	}
	return rpcout.ToResult()
}

func (c *Client) ChatRaw(ctx context.Context, in *ChatRequest, out *ChatResponse) error {
	// https://docs.mistral.ai/api/#tag/chat/operation/chat_completion_v1_chat_completions_post
	if err := c.validate(); err != nil {
		return err
	}
	in.Stream = false
	return c.post(ctx, "https://api.mistral.ai/v1/chat/completions", in, out)
}

func (c *Client) ChatStream(ctx context.Context, msgs genai.Messages, opts genai.Validatable, chunks chan<- genai.MessageFragment) error {
	in := ChatRequest{Model: c.model}
	if err := in.Init(msgs, opts); err != nil {
		return err
	}
	ch := make(chan ChatStreamChunkResponse)
	eg, ctx := errgroup.WithContext(ctx)
	eg.Go(func() error {
		return processStreamPackets(ch, chunks)
	})
	err := c.ChatStreamRaw(ctx, &in, ch)
	close(ch)
	if err2 := eg.Wait(); err2 != nil {
		err = err2
	}
	return err
}

func processStreamPackets(ch <-chan ChatStreamChunkResponse, chunks chan<- genai.MessageFragment) error {
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
		case "assistant", "":
		default:
			return fmt.Errorf("unexpected role %q", role)
		}
		if word := pkt.Choices[0].Delta.Content; word != "" {
			chunks <- genai.MessageFragment{TextFragment: word}
		}
	}
	return nil
}

func (c *Client) ChatStreamRaw(ctx context.Context, in *ChatRequest, out chan<- ChatStreamChunkResponse) error {
	if err := c.validate(); err != nil {
		return err
	}
	in.Stream = true
	resp, err := c.Client.PostRequest(ctx, "https://api.mistral.ai/v1/chat/completions", nil, in)
	if err != nil {
		return fmt.Errorf("failed to get server response: %w", err)
	}
	defer resp.Body.Close()
	for r := bufio.NewReader(resp.Body); ; {
		line, err := r.ReadBytes('\n')
		if line = bytes.TrimSpace(line); err == io.EOF {
			if len(line) == 0 {
				return nil
			}
		} else if err != nil {
			return fmt.Errorf("failed to get server response: %w", err)
		}
		if len(line) != 0 {
			if err := parseStreamLine(line, out); err != nil {
				return err
			}
		}
	}
}

func parseStreamLine(line []byte, out chan<- ChatStreamChunkResponse) error {
	const prefix = "data: "
	if !bytes.HasPrefix(line, []byte(prefix)) {
		d := json.NewDecoder(bytes.NewReader(line))
		d.DisallowUnknownFields()
		d.UseNumber()
		erAuth := errorResponseAuth{}
		if err := d.Decode(&erAuth); err != nil {
			return fmt.Errorf("unexpected line. expected \"data: \", got %q", line)
		}
		// This is clunky but the best we can do.
		if erAuth.Message == "Requests rate limit exceeded" {
			return &httpjson.Error{
				ResponseBody: []byte(erAuth.Message),
				StatusCode:   http.StatusTooManyRequests,
				Status:       erAuth.Message,
			}
		}
		return errors.New(erAuth.Message)
	}
	suffix := string(line[len(prefix):])
	if suffix == "[DONE]" {
		return nil
	}
	d := json.NewDecoder(strings.NewReader(suffix))
	d.DisallowUnknownFields()
	d.UseNumber()
	msg := ChatStreamChunkResponse{}
	if err := d.Decode(&msg); err != nil {
		return fmt.Errorf("failed to decode server response %q: %w", string(line), err)
	}
	out <- msg
	return nil
}

// Time is a JSON encoded unix timestamp.
type Time int64

func (t *Time) AsTime() time.Time {
	return time.Unix(int64(*t), 0)
}

// https://docs.mistral.ai/api/#tag/models/operation/retrieve_model_v1_models__model_id__get
type Model struct {
	ID           string `json:"id"`
	Object       string `json:"object"`
	Created      Time   `json:"created"`
	OwnedBy      string `json:"owned_by"`
	Capabilities struct {
		ChatChat        bool `json:"completion_chat"`
		ChatFim         bool `json:"completion_fim"`
		FunctionCalling bool `json:"function_calling"`
		FineTuning      bool `json:"fine_tuning"`
		Vision          bool `json:"vision"`
		Classification  bool `json:"classification"`
	} `json:"capabilities"`
	Name                    string   `json:"name"`
	Description             string   `json:"description"`
	MaxContextLength        int64    `json:"max_context_length"`
	Aliases                 []string `json:"aliases"`
	Deprecation             string   `json:"deprecation"`
	DefaultModelTemperature float64  `json:"default_model_temperature"`
	Type                    string   `json:"type"`
}

func (m *Model) GetID() string {
	return m.ID
}

func (m *Model) String() string {
	var caps []string
	if m.Capabilities.ChatChat {
		caps = append(caps, "chat")
	}
	if m.Capabilities.ChatFim {
		caps = append(caps, "fim")
	}
	if m.Capabilities.FunctionCalling {
		caps = append(caps, "function")
	}
	if m.Capabilities.FineTuning {
		caps = append(caps, "fine-tuning")
	}
	if m.Capabilities.Vision {
		caps = append(caps, "vision")
	}
	suffix := ""
	if m.Deprecation != "" {
		suffix += " (deprecated)"
	}
	prefix := m.ID
	if m.ID != m.Name {
		prefix += " (" + m.Name + ")"
	}
	// Not including Created and Description because Created is not set and Description is not useful.
	return fmt.Sprintf("%s: %s Context: %d%s", prefix, strings.Join(caps, "/"), m.MaxContextLength, suffix)
}

func (m *Model) Context() int64 {
	return m.MaxContextLength
}

func (c *Client) ListModels(ctx context.Context) ([]genai.Model, error) {
	// https://docs.mistral.ai/api/#tag/models
	var out struct {
		Object string  `json:"object"` // list
		Data   []Model `json:"data"`
	}
	if err := c.Client.Get(ctx, "https://api.mistral.ai/v1/models", nil, &out); err != nil {
		return nil, err
	}
	models := make([]genai.Model, len(out.Data))
	for i := range out.Data {
		models[i] = &out.Data[i]
	}
	return models, nil
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
	// This is so cute.
	erAuth := errorResponseAuth{}
	erAPI1 := errorResponseAPI1{}
	erAPI2 := errorResponseAPI2{}
	erAPI3 := errorResponseAPI3{}
	switch i, err := httpjson.DecodeResponse(resp, out, &erAuth, &erAPI1, &erAPI2, &erAPI3); i {
	case 0:
		return nil
	case 1:
		var herr *httpjson.Error
		if errors.As(err, &herr) {
			if herr.StatusCode == http.StatusUnauthorized {
				return fmt.Errorf("%w: %s. You can get a new API key at %s", herr, erAuth.Message, apiKeyURL)
			}
			return fmt.Errorf("%w: %s", herr, erAuth.Message)
		}
		return errors.New(erAuth.Message)
	case 2:
		var herr *httpjson.Error
		if errors.As(err, &herr) {
			return fmt.Errorf("%w: error %s: %s", herr, erAPI1.Type, erAPI1.Message)
		}
		return fmt.Errorf("error %s: %s", erAPI1.Type, erAPI1.Message)
	case 3:
		var herr *httpjson.Error
		if errors.As(err, &herr) {
			return fmt.Errorf("%w: error %s: %s at %s", herr, erAPI2.Detail[0].Type, erAPI2.Detail[0].Msg, erAPI2.Detail[0].Loc)
		}
		return fmt.Errorf("error %s: %s at %s", erAPI2.Detail[0].Type, erAPI2.Detail[0].Msg, erAPI2.Detail[0].Loc)
	case 4:
		var herr *httpjson.Error
		if errors.As(err, &herr) {
			return fmt.Errorf("%w: error %s/%s: %s at %s", herr, erAPI3.Type, erAPI3.Message.Detail[0].Type, erAPI3.Message.Detail[0].Msg, erAPI3.Message.Detail[0].Loc)
		}
		return fmt.Errorf("error %s/%s: %s at %s", erAPI3.Type, erAPI3.Message.Detail[0].Type, erAPI3.Message.Detail[0].Msg, erAPI3.Message.Detail[0].Loc)
	default:
		var herr *httpjson.Error
		if errors.As(err, &herr) {
			slog.WarnContext(ctx, "mistral", "url", url, "err", err, "response", string(herr.ResponseBody), "status", herr.StatusCode)
		} else {
			slog.WarnContext(ctx, "mistral", "url", url, "err", err)
		}
		return err
	}
}

const apiKeyURL = "https://console.mistral.ai/api-keys"

var (
	_ genai.ChatProvider  = &Client{}
	_ genai.ModelProvider = &Client{}
)
