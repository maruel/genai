// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Package cohere implements a client for the Cohere API.
//
// It is described at https://docs.cohere.com/reference/
package cohere

// See official client at https://github.com/cohere-ai/cohere-go

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
	"sort"
	"strings"

	"github.com/invopop/jsonschema"
	"github.com/maruel/genai"
	"github.com/maruel/genai/internal"
	"github.com/maruel/httpjson"
	"github.com/maruel/roundtrippers"
	"golang.org/x/sync/errgroup"
)

// https://docs.cohere.com/reference/chat
type ChatRequest struct {
	Stream          bool       `json:"stream"`
	Model           string     `json:"model"`
	Messages        []Message  `json:"messages"`
	Documents       []Document `json:"documents,omitzero"`
	CitationOptions struct {
		Mode string `json:"mode,omitzero"` // "fast", "accurate", "off"; default "fast"
	} `json:"citation_options,omitzero"`
	ResponseFormat struct {
		Type       string             `json:"type,omitzero"` // "text", "json_object"
		JSONSchema *jsonschema.Schema `json:"json_schema,omitzero"`
	} `json:"response_format,omitzero"`
	SafetyMode       string   `json:"safety_mode,omitzero"` // "CONTEXTUAL", "STRICT", "OFF"
	MaxTokens        int64    `json:"max_tokens,omitzero"`
	StopSequences    []string `json:"stop_sequences,omitzero"` // keywords to stop completion
	Temperature      float64  `json:"temperature,omitzero"`
	Seed             int64    `json:"seed,omitzero"`
	FrequencyPenalty float64  `json:"frequency_penalty,omitzero"` // [0, 1.0]
	PresencePenalty  float64  `json:"presence_penalty,omitzero"`  // [0, 1.0]
	K                int64    `json:"k,omitzero"`                 // [0, 500]
	P                float64  `json:"p,omitzero"`                 // [0.01, 0.99]
	Logprobs         bool     `json:"logprobs,omitzero"`
	Tools            []Tool   `json:"tools,omitzero"`
	ToolChoice       string   `json:"tool_choice,omitzero"` // "required", "none"
	StrictTools      bool     `json:"strict_tools,omitzero"`
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
				c.P = v.TopP
				sp = v.SystemPrompt
				c.Seed = v.Seed
				c.K = v.TopK
				c.StopSequences = v.Stop
				if v.ReplyAsJSON {
					c.ResponseFormat.Type = "json_object"
				}
				if v.DecodeAs != nil {
					c.ResponseFormat.Type = "json_schema"
					c.ResponseFormat.JSONSchema = jsonschema.Reflect(v.DecodeAs)
				}
				if len(v.Tools) != 0 {
					// Let's assume if the user provides tools, they want to use them.
					c.ToolChoice = "required"
					c.StrictTools = true
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

// https://docs.cohere.com/reference/chat
type Message struct {
	Role string `json:"role"` // "system", "assistant", "user"
	// Type == "system", "assistant", or "user".
	Content []Content `json:"content"`
	// Type == "assistant"
	Citations any `json:"citations,omitzero"` // TODO
	// Type == "assistant"
	ToolCalls  []ToolCall `json:"tool_calls,omitzero"`
	ToolCallID string     `json:"tool_call_id,omitzero"` // TODO
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
		m.ToolCalls = make([]ToolCall, len(in.ToolCalls))
		for i := range in.ToolCalls {
			m.ToolCalls[i].From(&in.ToolCalls[i])
		}
	}
	return nil
}

type Content struct {
	Type     string `json:"type,omitzero"` // "text", "image_url" or "document"
	Text     string `json:"text,omitzero"`
	ImageURL struct {
		URL string `json:"url,omitzero"`
	} `json:"image_url,omitzero"`
	Document struct {
		Data map[string]any `json:"data,omitzero"` // TODO
		ID   string         `json:"id,omitzero"`   // TODO
	} `json:"document,omitzero"`
}

func (c *Content) From(in *genai.Content) error {
	if in.Text != "" {
		c.Type = "text"
		c.Text = in.Text
		return nil
	}
	// Currently fails with: http 400: error: invalid request: all elements in history must have a message
	// TODO: Investigate one day. Maybe because trial key.
	mimeType, data, err := in.ReadDocument(10 * 1024 * 1024)
	if err != nil {
		return err
	}
	switch {
	case (in.URL != "" && mimeType == "") || strings.HasPrefix(mimeType, "image/"):
		c.Type = "image_url"
		if in.URL != "" {
			c.ImageURL.URL = in.URL
		} else {
			c.ImageURL.URL = fmt.Sprintf("data:%s;base64,%s", mimeType, base64.StdEncoding.EncodeToString(data))
		}
	default:
		return fmt.Errorf("unsupported mime type %s", mimeType)
	}
	return nil
}

type Tool struct {
	Type     string `json:"type,omitzero"` // "function"
	Function struct {
		Name        string             `json:"name,omitzero"`
		Parameters  *jsonschema.Schema `json:"parameters,omitzero"`
		Description string             `json:"description,omitzero"`
	} `json:"function,omitzero"`
}

type Document struct {
	// Or a string.
	Document struct {
		ID   string         `json:"id,omitzero"`
		Data map[string]any `json:"data,omitzero"`
	} `json:"document,omitzero"`
}

type ChatResponse struct {
	ID           string          `json:"id"`
	FinishReason string          `json:"finish_reason"` // COMPLETE, STOP_SEQUENCe, MAX_TOKENS, TOOL_CALL, ERROR
	Message      MessageResponse `json:"message"`
	Usage        struct {
		BilledUnits struct {
			InputTokens     int64 `json:"input_tokens"`
			OutputTokens    int64 `json:"output_tokens"`
			SearchUnits     int64 `json:"search_units"`
			Classifications int64 `json:"classifications"`
		} `json:"billed_units"`
		Tokens struct {
			InputTokens  int64 `json:"input_tokens"`
			OutputTokens int64 `json:"output_tokens"`
		} `json:"tokens"`
	} `json:"usage"`
	Logprobs []struct {
		TokenIDs []int64   `json:"token_ids"`
		Text     string    `json:"text"`
		Logprobs []float64 `json:"logprobs"`
	} `json:"logprobs"`
}

func (c *ChatResponse) ToResult() (genai.ChatResult, error) {
	out := genai.ChatResult{
		Usage: genai.Usage{
			// What about BilledUnits, especially for SearchUnits and Classifications?
			InputTokens:  c.Usage.Tokens.InputTokens,
			OutputTokens: c.Usage.Tokens.OutputTokens,
		},
	}
	err := c.Message.To(&out.Message)
	return out, err
}

type MessageResponse struct {
	Role      genai.Role `json:"role"`
	ToolCalls []ToolCall `json:"tool_calls"`
	ToolPlan  string     `json:"tool_plan"`
	Content   []struct {
		Type string `json:"type"` // text
		Text string `json:"text"`
	} `json:"content"`
	Citations []struct {
		Start   int64  `json:"start"`
		End     int64  `json:"end"`
		Text    string `json:"text"`
		Sources []any  `json:"sources"`
		Type    string `json:"type"` // TEXT_CONTENT, PLAN
	} `json:"citations"`
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
	if len(m.Content) != 0 {
		out.Contents = make([]genai.Content, len(m.Content))
		for i := range m.Content {
			out.Contents[i].Text = m.Content[i].Text
		}
	}
	return nil
}

type ToolCall struct {
	ID       string `json:"id"`
	Type     string `json:"type"` // function
	Function struct {
		Name      string `json:"name"`
		Arguments string `json:"arguments"`
	} `json:"function"`
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

type ChatStreamChunkResponse struct {
	ID    string `json:"id"`
	Type  string `json:"type"` // "message_start", "content-start", "message-end"
	Index int64  `json:"index"`
	Delta struct {
		Message struct {
			Role    genai.Role `json:"role"`
			Content struct {
				Type string `json:"type"` // text
				Text string `json:"text"`
			} `json:"content"`
			ToolPlan  string     `json:"tool_plan"`
			ToolCalls []ToolCall `json:"tool_calls"`
			Citations []struct{} `json:"citations"`
		} `json:"message"`
		FinishReason string `json:"finish_reason"` // COMPLETE
		Usage        struct {
			BilledUnits struct {
				InputTokens  int64 `json:"input_tokens"`
				OutputTokens int64 `json:"output_tokens"`
			} `json:"billed_units"`
			Tokens struct {
				InputTokens  int64 `json:"input_tokens"`
				OutputTokens int64 `json:"output_tokens"`
			} `json:"tokens"`
		} `json:"usage"`
	} `json:"delta"`
}

// content can be a struct or an empty list. Go doesn't like that.
type completionStreamChunkMsgStartResponse struct {
	ID    string `json:"id"`
	Type  string `json:"type"` // "message_start"
	Delta struct {
		Message struct {
			Role genai.Role `json:"role"`
			// WTF are they doing?
			Content   []struct{} `json:"content"`
			ToolPlan  string     `json:"tool_plan"`
			ToolCalls []ToolCall `json:"tool_calls"`
			Citations []struct{} `json:"citations"`
		} `json:"message"`
	} `json:"delta"`
}

func (c *completionStreamChunkMsgStartResponse) translateTo(msg *ChatStreamChunkResponse) {
	msg.ID = c.ID
	msg.Type = c.Type
	msg.Delta.Message.Role = c.Delta.Message.Role
	msg.Delta.Message.ToolPlan = c.Delta.Message.ToolPlan
}

//

type errorResponse struct {
	Message   string `json:"message"`
	RequestID string `json:"request_id"`
}

// Client implements the REST JSON based API.
type Client struct {
	// Client is exported for testing replay purposes.
	Client httpjson.Client

	model string
}

// New creates a new client to talk to the Cohere platform API.
//
// If apiKey is not provided, it tries to load it from the COHERE_API_KEY environment variable.
// If none is found, it returns an error.
// Get your API key at https://dashboard.cohere.com/api-keys
// If no model is provided, only functions that do not require a model, like ListModels, will work.
// To use multiple models, create multiple clients.
// Use one of the model from https://cohere.com/pricing and https://docs.cohere.com/v2/docs/models
func New(apiKey, model string) (*Client, error) {
	if apiKey == "" {
		if apiKey = os.Getenv("COHERE_API_KEY"); apiKey == "" {
			return nil, errors.New("cohere API key is required; get one at " + apiKeyURL)
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
	// https://docs.cohere.com/reference/chat
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
	if err := c.validate(); err != nil {
		return err
	}
	in.Stream = false
	return c.post(ctx, "https://api.cohere.com/v2/chat", in, out)
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
		switch role := pkt.Delta.Message.Role; role {
		case "assistant", "":
		default:
			return fmt.Errorf("unexpected role %q", role)
		}
		if word := pkt.Delta.Message.Content.Text; word != "" {
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
	resp, err := c.Client.PostRequest(ctx, "https://api.cohere.com/v2/chat", nil, in)
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
	const dataPrefix = "data: "
	switch {
	case bytes.HasPrefix(line, []byte(dataPrefix)):
		suffix := string(line[len(dataPrefix):])
		if suffix == "[DONE]" {
			return nil
		}
		d := json.NewDecoder(strings.NewReader(suffix))
		d.DisallowUnknownFields()
		d.UseNumber()
		msg := ChatStreamChunkResponse{}
		if err := d.Decode(&msg); err != nil {
			fallback := completionStreamChunkMsgStartResponse{}
			d := json.NewDecoder(strings.NewReader(suffix))
			d.DisallowUnknownFields()
			d.UseNumber()
			if err := d.Decode(&fallback); err != nil {
				return fmt.Errorf("failed to decode server response %q: %w", string(line), err)
			}
			fallback.translateTo(&msg)
		}
		out <- msg
	case string(line) == ": keep-alive":
	case bytes.HasPrefix(line, []byte("event:")):
		// Ignore for now.
	default:
		return fmt.Errorf("unexpected line. expected \"data: \", got %q", line)
	}
	return nil
}

type Model struct {
	Name             string   `json:"name"`
	Endpoints        []string `json:"endpoints"` // chat, embed, classify, summarize, rerank, rate, generate
	Features         []string `json:"features"`  // json_mode, json_schema, safety_modes, strict_tools, tools
	Finetuned        bool     `json:"finetuned"`
	ContextLength    int64    `json:"context_length"`
	TokenizerURL     string   `json:"tokenizer_url"`
	SupportsVision   bool     `json:"supports_vision"`
	DefaultEndpoints []string `json:"default_endpoints"`
}

func (m *Model) GetID() string {
	return m.Name
}

func (m *Model) String() string {
	suffix := ""
	if m.Finetuned {
		suffix += " (finetuned)"
	}
	if m.SupportsVision {
		suffix += " (vision)"
	}
	endpoints := make([]string, len(m.Endpoints))
	copy(endpoints, m.Endpoints)
	sort.Strings(endpoints)
	f := ""
	if len(m.Features) > 0 {
		features := make([]string, len(m.Features))
		copy(features, m.Features)
		sort.Strings(features)
		f = " with " + strings.Join(features, "/")
	}
	return fmt.Sprintf("%s: %s%s. Context: %d%s", m.Name, strings.Join(endpoints, "/"), f, m.ContextLength, suffix)
}

func (m *Model) Context() int64 {
	return m.ContextLength
}

func (c *Client) ListModels(ctx context.Context) ([]genai.Model, error) {
	// https://docs.cohere.com/reference/list-models
	var out struct {
		Models        []Model `json:"models"`
		NextPageToken string  `json:"next_page_token"`
	}
	if err := c.Client.Get(ctx, "https://api.cohere.com/v1/models?page_size=1000", nil, &out); err != nil {
		return nil, err
	}
	models := make([]genai.Model, len(out.Models))
	for i := range out.Models {
		models[i] = &out.Models[i]
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
	er := errorResponse{}
	switch i, err := httpjson.DecodeResponse(resp, out, &er); i {
	case 0:
		return nil
	case 1:
		var herr *httpjson.Error
		if errors.As(err, &herr) {
			if herr.StatusCode == http.StatusUnauthorized {
				return fmt.Errorf("%w: error: %s. You can get a new API key at %s", herr, er.Message, apiKeyURL)
			}
			return fmt.Errorf("%w: error: %s", herr, er.Message)
		}
		return fmt.Errorf("error: %s", er.Message)
	default:
		var herr *httpjson.Error
		if errors.As(err, &herr) {
			slog.WarnContext(ctx, "cohere", "url", url, "err", err, "response", string(herr.ResponseBody), "status", herr.StatusCode)
		} else {
			slog.WarnContext(ctx, "cohere", "url", url, "err", err)
		}
		return err
	}
}

const apiKeyURL = "https://dashboard.cohere.com/api-keys"

var (
	_ genai.ChatProvider  = &Client{}
	_ genai.ModelProvider = &Client{}
)
