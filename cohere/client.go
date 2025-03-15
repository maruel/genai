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

	"github.com/maruel/genai/genaiapi"
	"github.com/maruel/genai/internal"
	"github.com/maruel/httpjson"
)

// https://docs.cohere.com/reference/chat
type CompletionRequest struct {
	Stream          bool       `json:"stream"`
	Model           string     `json:"model"`
	Messages        []Message  `json:"messages"`
	Documents       []Document `json:"documents,omitzero"`
	CitationOptions struct {
		Mode string `json:"mode,omitzero"` // "fast", "accurate", "off"; default "fast"
	} `json:"citation_options,omitzero"`
	ResponseFormat struct {
		Type       string              `json:"type,omitzero"` // "text", "json_object"
		JSONSchema genaiapi.JSONSchema `json:"json_schema,omitzero"`
	} `json:"response_format,omitzero"`
	SafetyMode       string   `json:"safety_mode,omitzero"` // "CONTEXTUAL", "STRICT", "OFF"
	MaxTokens        int64    `json:"max_tokens,omitzero"`
	StopSequences    []string `json:"stop_sequences,omitzero"` // keywords to stop completion
	Temperature      float64  `json:"temperature,omitzero"`
	Seed             int64    `json:"seed,omitzero"`
	FrequencyPenalty float64  `json:"frequency_penalty,omitzero"` // [0, 1.0]
	PresencePenalty  float64  `json:"presence_penalty,omitzero"`  // [0, 1.0]
	K                float64  `json:"k,omitzero"`                 // [0, 500.0]
	P                float64  `json:"p,omitzero"`                 // [0.01, 0.99]
	Logprobs         bool     `json:"logprobs,omitzero"`
	Tools            []Tool   `json:"tools,omitzero"`
	ToolChoice       string   `json:"tool_choice,omitzero"` // "required", "none"
	StrictTools      bool     `json:"strict_tools,omitzero"`
}

func (c *CompletionRequest) Init(msgs []genaiapi.Message, opts any) error {
	if opts != nil {
		switch v := opts.(type) {
		case *genaiapi.CompletionOptions:
			c.MaxTokens = v.MaxTokens
			c.Seed = v.Seed
			c.Temperature = v.Temperature
			if v.ReplyAsJSON {
				c.ResponseFormat.Type = "json_object"
			}
			if !v.JSONSchema.IsZero() {
				c.ResponseFormat.Type = "json_schema"
				c.ResponseFormat.JSONSchema = v.JSONSchema
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
					c.Tools[i].Function.Parameters = t.Parameters
				}
			}
		default:
			return fmt.Errorf("unsupported options type %T", opts)
		}
	}

	c.Messages = make([]Message, len(msgs))
	for i, m := range msgs {
		if err := m.Validate(); err != nil {
			return fmt.Errorf("message %d: %w", i, err)
		}
		switch m.Role {
		case genaiapi.System:
			if i != 0 {
				return fmt.Errorf("message %d: system message must be first message", i)
			}
		case genaiapi.User, genaiapi.Assistant:
		default:
			return fmt.Errorf("message %d: unsupported role %q", i, m.Role)
		}
		c.Messages[i].Role = string(m.Role)
		c.Messages[i].Content = []Content{{}}
		switch m.Type {
		case genaiapi.Text:
			c.Messages[i].Content[0].Type = "text"
			c.Messages[i].Content[0].Text = m.Text
		case genaiapi.Document:
			// Currently fails with: http 400: error: invalid request: all elements in history must have a message
			// TODO: Investigate one day. Maybe because trial key.
			mimeType, data, err := internal.ParseDocument(&m, 10*1024*1024)
			if err != nil {
				return fmt.Errorf("message %d: %w", i, err)
			}
			switch {
			case (m.URL != "" && mimeType == "") || strings.HasPrefix(mimeType, "image/"):
				c.Messages[i].Content[0].Type = "image_url"
				if m.URL != "" {
					c.Messages[i].Content[0].ImageURL.URL = m.URL
				} else {
					c.Messages[i].Content[0].ImageURL.URL = fmt.Sprintf("data:%s;base64,%s", mimeType, base64.StdEncoding.EncodeToString(data))
				}
			default:
				return fmt.Errorf("message %d: unsupported mime type %s", i, mimeType)
			}
		default:
			return fmt.Errorf("message %d: unsupported content type %s", i, m.Type)
		}
	}
	return nil
}

type Message struct {
	Role string `json:"role"`
	// Assistant, System or User.
	Content []Content `json:"content"`
	// Assistant
	Citations any `json:"citations,omitzero"` // TODO
	// Assistant
	ToolCalls []any `json:"tool_calls,omitzero"` // TODO
	// Tool
	ToolCallID string `json:"tool_call_id,omitzero"`
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

type Tool struct {
	Type     string `json:"type,omitzero"` // "function"
	Function struct {
		Name        string              `json:"name,omitzero"`
		Parameters  genaiapi.JSONSchema `json:"parameters,omitzero"`
		Description string              `json:"description,omitzero"`
	} `json:"function,omitzero"`
}

type Document struct {
	// Or a string.
	Document struct {
		ID   string         `json:"id,omitzero"`
		Data map[string]any `json:"data,omitzero"`
	} `json:"document,omitzero"`
}

type CompletionResponse struct {
	ID           string `json:"id"`
	FinishReason string `json:"finish_reason"` // COMPLETE, STOP_SEQUENCe, MAX_TOKENS, TOOL_CALL, ERROR
	Message      struct {
		Role      genaiapi.Role `json:"role"`
		ToolCalls []struct {
			ID       string `json:"id"`
			Type     string `json:"type"` // function
			Function struct {
				Name      string `json:"name"`
				Arguments string `json:"arguments"`
			} `json:"function"`
		} `json:"tool_calls"`
		ToolPlan string `json:"tool_plan"`
		Content  []struct {
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
	} `json:"message"`
	Usage struct {
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

func (c *CompletionResponse) ToResult() (genaiapi.CompletionResult, error) {
	out := genaiapi.CompletionResult{}
	// What about BilledUnits, especially for SearchUnits and Classifications?
	out.InputTokens = c.Usage.Tokens.InputTokens
	out.OutputTokens = c.Usage.Tokens.OutputTokens
	if len(c.Message.ToolCalls) != 0 {
		out.Role = genaiapi.Assistant
		out.Type = genaiapi.ToolCalls
		out.ToolCalls = make([]genaiapi.ToolCall, len(c.Message.ToolCalls))
		for i, t := range c.Message.ToolCalls {
			out.ToolCalls[i].ID = t.ID
			out.ToolCalls[i].Name = t.Function.Name
			out.ToolCalls[i].Arguments = t.Function.Arguments
		}
	} else {
		if len(c.Message.Content) != 1 {
			return out, fmt.Errorf("unexpected number of messages %d", len(c.Message.Content))
		}
		out.Type = genaiapi.Text
		out.Text = c.Message.Content[0].Text
	}
	switch role := c.Message.Role; role {
	case "system", "assistant", "user":
		out.Role = genaiapi.Role(role)
	default:
		return out, fmt.Errorf("unsupported role %q", role)
	}
	return out, nil
}

type CompletionStreamChunkResponse struct {
	ID    string `json:"id"`
	Type  string `json:"type"` // "message_start", "content-start", "message-end"
	Index int64  `json:"index"`
	Delta struct {
		Message struct {
			Role    genaiapi.Role `json:"role"`
			Content struct {
				Type string `json:"type"` // text
				Text string `json:"text"`
			} `json:"content"`
			ToolPlan  string     `json:"tool_plan"`
			ToolCalls []struct{} `json:"tool_calls"`
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
			Role genaiapi.Role `json:"role"`
			// WTF are they doing?
			Content   []struct{} `json:"content"`
			ToolPlan  string     `json:"tool_plan"`
			ToolCalls []struct{} `json:"tool_calls"`
			Citations []struct{} `json:"citations"`
		} `json:"message"`
	} `json:"delta"`
}

func (c *completionStreamChunkMsgStartResponse) translateTo(msg *CompletionStreamChunkResponse) {
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
	apiKey string
	model  string
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
	return &Client{apiKey: apiKey, model: model}, nil
}

func (c *Client) Completion(ctx context.Context, msgs []genaiapi.Message, opts any) (genaiapi.CompletionResult, error) {
	// https://docs.cohere.com/reference/chat
	rpcin := CompletionRequest{Model: c.model}
	if err := rpcin.Init(msgs, opts); err != nil {
		return genaiapi.CompletionResult{}, err
	}
	rpcout := CompletionResponse{}
	if err := c.CompletionRaw(ctx, &rpcin, &rpcout); err != nil {
		return genaiapi.CompletionResult{}, fmt.Errorf("failed to get chat response: %w", err)
	}
	return rpcout.ToResult()
}

func (c *Client) CompletionRaw(ctx context.Context, in *CompletionRequest, out *CompletionResponse) error {
	if err := c.validate(); err != nil {
		return err
	}
	in.Stream = false
	return c.post(ctx, "https://api.cohere.com/v2/chat", in, out)
}

func (c *Client) CompletionStream(ctx context.Context, msgs []genaiapi.Message, opts any, chunks chan<- genaiapi.MessageChunk) error {
	in := CompletionRequest{Model: c.model}
	if err := in.Init(msgs, opts); err != nil {
		return err
	}
	ch := make(chan CompletionStreamChunkResponse)
	end := make(chan error)
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()
	go func() {
		lastRole := genaiapi.System
		for pkt := range ch {
			switch role := pkt.Delta.Message.Role; role {
			case "system", "assistant", "user":
				lastRole = genaiapi.Role(role)
			case "":
			default:
				cancel()
				// We need to empty the channel to avoid blocking the goroutine.
				for range ch {
				}
				end <- fmt.Errorf("unexpected role %q", role)
				return
			}
			if word := pkt.Delta.Message.Content.Text; word != "" {
				chunks <- genaiapi.MessageChunk{Role: lastRole, Type: genaiapi.Text, Text: word}
			}
		}
		end <- nil
	}()
	err := c.CompletionStreamRaw(ctx, &in, ch)
	close(ch)
	if err2 := <-end; err2 != nil {
		err = err2
	}
	return err
}

func (c *Client) CompletionStreamRaw(ctx context.Context, in *CompletionRequest, out chan<- CompletionStreamChunkResponse) error {
	if err := c.validate(); err != nil {
		return err
	}
	in.Stream = true
	h := make(http.Header)
	h.Add("Authorization", "Bearer "+c.apiKey)
	// Cohere doesn't HTTP POST support compression.
	resp, err := httpjson.DefaultClient.PostRequest(ctx, "https://api.cohere.com/v2/chat", h, in)
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

func parseStreamLine(line []byte, out chan<- CompletionStreamChunkResponse) error {
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
		msg := CompletionStreamChunkResponse{}
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

func (c *Client) ListModels(ctx context.Context) ([]genaiapi.Model, error) {
	// https://docs.cohere.com/reference/list-models
	h := make(http.Header)
	h.Add("Authorization", "Bearer "+c.apiKey)
	var out struct {
		Models        []Model `json:"models"`
		NextPageToken string  `json:"next_page_token"`
	}
	err := httpjson.DefaultClient.Get(ctx, "https://api.cohere.com/v1/models?page_size=1000", h, &out)
	if err != nil {
		return nil, err
	}
	models := make([]genaiapi.Model, len(out.Models))
	for i := range out.Models {
		models[i] = &out.Models[i]
	}
	return models, err
}

func (c *Client) validate() error {
	if c.model == "" {
		return errors.New("a model is required")
	}
	return nil
}

func (c *Client) post(ctx context.Context, url string, in, out any) error {
	h := make(http.Header)
	h.Add("Authorization", "Bearer "+c.apiKey)
	// Cohere doesn't HTTP POST support compression.
	resp, err := httpjson.DefaultClient.PostRequest(ctx, url, h, in)
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
	_ genaiapi.CompletionProvider = &Client{}
	_ genaiapi.ModelProvider      = &Client{}
)
