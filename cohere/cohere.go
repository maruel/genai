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
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"sort"
	"strings"

	"github.com/maruel/genai/genaiapi"
	"github.com/maruel/httpjson"
)

// https://docs.cohere.com/reference/chat
type CompletionRequest struct {
	Stream          bool       `json:"stream"`
	Model           string     `json:"model"`
	Messages        []Message  `json:"messages"`
	Tools           []Tool     `json:"tools,omitzero"`
	Documents       []Document `json:"documents,omitzero"`
	CitationOptions struct {
		Mode string `json:"mode,omitzero"` // "fast", "accurate", "off"; default "fast"
	} `json:"citation_options,omitzero"`
	ResponseFormat struct {
		Type       string     `json:"type,omitzero"` // "text", "json_object"
		JSONSchema JSONSchema `json:"json_schema,omitzero"`
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
	ToolChoice       string   `json:"tool_choice,omitzero"` // "required", "none"
	StrictTools      bool     `json:"strict_tools,omitzero"`
}

func (c *CompletionRequest) fromOpts(opts any) error {
	switch v := opts.(type) {
	case *genaiapi.CompletionOptions:
		c.MaxTokens = v.MaxTokens
		c.Seed = v.Seed
		c.Temperature = v.Temperature
	default:
		return fmt.Errorf("unsupported options type %T", opts)
	}
	return nil
}

func (c *CompletionRequest) fromMsgs(msgs []genaiapi.Message) error {
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
		case genaiapi.User, genaiapi.Assistant, genaiapi.Tool:
		default:
			return fmt.Errorf("message %d: unsupported role %q", i, m.Role)
		}
		switch m.Type {
		case genaiapi.Text:
		default:
			return fmt.Errorf("message %d: unsupported content type %s", i, m.Type)
		}
		c.Messages[i].Role = string(m.Role)
		c.Messages[i].Content.Type = "text"
		c.Messages[i].Content.Text = m.Text
	}
	return nil
}

type Message struct {
	Role string `json:"role"`
	// Assistant, System or User.
	Content struct {
		Type     string `json:"type,omitzero"` // "text", "image_url" or "document"
		Text     string `json:"text,omitzero"`
		ImageURL struct {
			URL string `json:"url,omitzero"`
		} `json:"image_url,omitzero"`
		Document struct {
			Data map[string]any `json:"data,omitzero"` // TODO
			ID   string         `json:"id,omitzero"`   // TODO
		} `json:"document,omitzero"`
	} `json:"content"`
	// Assistant
	Citations any `json:"citations,omitzero"` // TODO
	// Assistant
	ToolCalls []any `json:"tool_calls,omitzero"` // TODO
	// Tool
	ToolCallID string `json:"tool_call_id,omitzero"`
}

type Tool struct {
	Type     string `json:"type,omitzero"` // "function"
	Function struct {
		Name        string         `json:"name,omitzero"`
		Parameters  map[string]any `json:"parameters,omitzero"`
		Description string         `json:"description,omitzero"`
	} `json:"function,omitzero"`
}

type Document struct {
	// Or a string.
	Document struct {
		ID   string         `json:"id,omitzero"`
		Data map[string]any `json:"data,omitzero"`
	} `json:"document,omitzero"`
}

type JSONSchema any

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

type Client struct {
	// ApiKey can be retrieved from https://dashboard.cohere.com/api-keys
	ApiKey string
	// Model to use, see https://cohere.com/pricing and https://docs.cohere.com/v2/docs/models
	Model string
}

func (c *Client) Completion(ctx context.Context, msgs []genaiapi.Message, opts any) (string, error) {
	// https://docs.cohere.com/reference/chat
	in := CompletionRequest{Model: c.Model}
	if err := in.fromOpts(opts); err != nil {
		return "", err
	}
	if err := in.fromMsgs(msgs); err != nil {
		return "", err
	}
	out := CompletionResponse{}
	if err := c.CompletionRaw(ctx, &in, &out); err != nil {
		return "", fmt.Errorf("failed to get chat response: %w", err)
	}
	return out.Message.Content[0].Text, nil
}

func (c *Client) CompletionRaw(ctx context.Context, in *CompletionRequest, out *CompletionResponse) error {
	if err := c.validate(true); err != nil {
		return err
	}
	return c.post(ctx, "https://api.cohere.com/v2/chat", in, out)
}

func (c *Client) CompletionStream(ctx context.Context, msgs []genaiapi.Message, opts any, words chan<- string) error {
	in := CompletionRequest{Model: c.Model, Stream: true}
	if err := in.fromOpts(opts); err != nil {
		return err
	}
	if err := in.fromMsgs(msgs); err != nil {
		return err
	}
	ch := make(chan CompletionStreamChunkResponse)
	end := make(chan struct{})
	go func() {
		for msg := range ch {
			word := msg.Delta.Message.Content.Text
			if word != "" {
				words <- word
			}
		}
		end <- struct{}{}
	}()
	err := c.CompletionStreamRaw(ctx, &in, ch)
	close(ch)
	<-end
	return err
}

func (c *Client) CompletionStreamRaw(ctx context.Context, in *CompletionRequest, out chan<- CompletionStreamChunkResponse) error {
	if err := c.validate(true); err != nil {
		return err
	}
	h := make(http.Header)
	h.Add("Authorization", "Bearer "+c.ApiKey)
	// Cohere doesn't HTTP POST support compression.
	resp, err := httpjson.DefaultClient.PostRequest(ctx, "https://api.cohere.com/v2/chat", h, in)
	if err != nil {
		return fmt.Errorf("failed to get server response: %w", err)
	}
	defer resp.Body.Close()
	r := bufio.NewReader(resp.Body)
	for {
		line, err := r.ReadBytes('\n')
		line = bytes.TrimSpace(line)
		if err == io.EOF {
			err = nil
			if len(line) == 0 {
				return nil
			}
		}
		if err != nil {
			return fmt.Errorf("failed to get server response: %w", err)
		}
		if len(line) == 0 {
			continue
		}
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
			if err = d.Decode(&msg); err != nil {
				fallback := completionStreamChunkMsgStartResponse{}
				d := json.NewDecoder(strings.NewReader(suffix))
				d.DisallowUnknownFields()
				d.UseNumber()
				if err = d.Decode(&fallback); err != nil {
					return fmt.Errorf("failed to decode server response %q: %w", string(line), err)
				}
				fallback.translateTo(&msg)
			}
			out <- msg
		case bytes.Equal(line, []byte(": keep-alive")):
		case bytes.HasPrefix(line, []byte("event:")):
			// Ignore for now.
		default:
			return fmt.Errorf("unexpected line. expected \"data: \", got %q", line)
		}
	}
}

func (c *Client) CompletionContent(ctx context.Context, msgs []genaiapi.Message, opts any, mime string, content []byte) (string, error) {
	return "", errors.New("not implemented")
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
	if err := c.validate(false); err != nil {
		return nil, err
	}
	h := make(http.Header)
	h.Add("Authorization", "Bearer "+c.ApiKey)
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

func (c *Client) validate(needModel bool) error {
	if c.ApiKey == "" {
		return errors.New("cohere ApiKey is required; get one at " + apiKeyURL)
	}
	if needModel && c.Model == "" {
		return errors.New("a Model is required")
	}
	return nil
}

func (c *Client) post(ctx context.Context, url string, in, out any) error {
	h := make(http.Header)
	h.Add("Authorization", "Bearer "+c.ApiKey)
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

var _ genaiapi.CompletionProvider = &Client{}
