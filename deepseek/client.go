// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Package deepseek implements a client for the DeepSeek API.
//
// It is described at https://api-docs.deepseek.com/
package deepseek

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
	"os"
	"strings"

	"github.com/invopop/jsonschema"
	"github.com/maruel/genai/genaiapi"
	"github.com/maruel/httpjson"
)

// https://api-docs.deepseek.com/api/create-chat-completion
type CompletionRequest struct {
	Model            string    `json:"model"`
	Messages         []Message `json:"messages"`
	Stream           bool      `json:"stream"`
	Temperature      float64   `json:"temperature,omitzero"`       // [0, 2]
	FrequencyPenalty float64   `json:"frequency_penalty,omitzero"` // [-2, 2]
	MaxToks          int64     `json:"max_tokens,omitzero"`        // [1, 8192]
	PresencePenalty  float64   `json:"presence_penalty,omitzero"`  // [-2, 2]
	ResponseFormat   struct {
		Type string `json:"type,omitzero"` // "text", "json_object"
	} `json:"response_format,omitzero"`
	Stop          []string `json:"stop,omitzero"`
	StreamOptions struct {
		IncludeUsage bool `json:"include_usage,omitzero"`
	} `json:"stream_options,omitzero"`
	TopP float64 `json:"top_p,omitzero"` // [0, 1]
	// Alternative when forcing a specific function. This can probably be achieved
	// by providing a single tool and ToolChoice == "required".
	// ToolChoice struct {
	// 	Type     string `json:"type,omitzero"` // "function"
	// 	Function struct {
	// 		Name string `json:"name,omitzero"`
	// 	} `json:"function,omitzero"`
	// } `json:"tool_choice,omitzero"`
	ToolChoice string `json:"tool_choice,omitzero"` // "none", "auto", "required"
	Tools      []Tool `json:"tools,omitzero"`
	Logprobs   bool   `json:"logprobs,omitzero"`
	TopLogprob int64  `json:"top_logprobs,omitzero"`
}

func (c *CompletionRequest) Init(msgs []genaiapi.Message, opts genaiapi.Validatable) error {
	var errs []error
	if opts != nil {
		if err := opts.Validate(); err != nil {
			errs = append(errs, err)
		} else {
			switch v := opts.(type) {
			case *genaiapi.CompletionOptions:
				c.MaxToks = v.MaxTokens
				c.Temperature = v.Temperature
				if v.Seed != 0 {
					errs = append(errs, errors.New("deepseek does not support seed"))
				}
				c.TopP = v.TopP
				if v.TopK != 0 {
					errs = append(errs, errors.New("deepseek does not support TopK"))
				}
				c.Stop = v.Stop
				if v.ReplyAsJSON {
					c.ResponseFormat.Type = "json_object"
				}
				if v.JSONSchema != nil {
					errs = append(errs, errors.New("deepseek doesn't support JSON schema"))
				}
				if len(v.Tools) != 0 {
					// Let's assume if the user provides tools, they want to use them.
					c.ToolChoice = "required"
					c.Tools = make([]Tool, len(v.Tools))
					for i, t := range v.Tools {
						c.Tools[i].Type = "function"
						c.Tools[i].Function.Name = t.Name
						c.Tools[i].Function.Description = t.Description
						c.Tools[i].Function.Parameters = t.Parameters
					}
				}
			default:
				errs = append(errs, fmt.Errorf("unsupported options type %T", opts))
			}
		}
	}

	if err := genaiapi.ValidateMessages(msgs); err != nil {
		errs = append(errs, err)
	} else {
		c.Messages = make([]Message, len(msgs))
		for i, m := range msgs {
			if err := c.Messages[i].From(m); err != nil {
				errs = append(errs, fmt.Errorf("message %d: %w", i, err))
			}
		}
	}
	return errors.Join(errs...)
}

type Message struct {
	Role             string `json:"role,omitzero"`
	Content          string `json:"content,omitzero"`
	Name             string `json:"name,omitzero"`
	Prefix           bool   `json:"prefix,omitzero"`
	ReasoningContent string `json:"reasoning_content,omitzero"`
	ToolCall         []struct {
		Index    int64  `json:"index,omitzero"`
		ID       string `json:"id,omitzero"`
		Type     string `json:"type,omitzero"` // "function"
		Function struct {
			Name      string `json:"name,omitzero"`
			Arguments string `json:"arguments,omitzero"`
		} `json:"function,omitzero"`
	} `json:"tool_calls,omitzero"`
}

func (msg *Message) From(m genaiapi.Message) error {
	switch m.Role {
	case genaiapi.System, genaiapi.User, genaiapi.Assistant:
		msg.Role = string(m.Role)
	default:
		return fmt.Errorf("unsupported role %q", m.Role)
	}
	switch m.Type {
	case genaiapi.Text:
		msg.Content = m.Text
	default:
		return fmt.Errorf("unsupported content type %s", m.Type)
	}
	return nil
}

type Tool struct {
	Type     string `json:"type"` // "function"
	Function struct {
		Name        string             `json:"name,omitzero"`
		Description string             `json:"description,omitzero"`
		Parameters  *jsonschema.Schema `json:"parameters,omitzero"`
	} `json:"function"`
}

type CompletionResponse struct {
	ID      string `json:"id"`
	Choices []struct {
		FinishReason string   `json:"finish_reason"` // "tool_calls"
		Index        int64    `json:"index"`
		Message      Message  `json:"message"`
		Logprobs     Logprobs `json:"logprobs"`
	} `json:"choices"`
	Created           int64  `json:"created"` // Unix timestamp
	Model             string `json:"model"`
	SystemFingerPrint string `json:"system_fingerprint"`
	Object            string `json:"object"` // chat.completion
	Usage             struct {
		CompletionTokens      int64 `json:"completion_tokens"`
		PromptTokens          int64 `json:"prompt_tokens"`
		PromptCacheHitTokens  int64 `json:"prompt_cache_hit_tokens"`
		PromptCacheMissTokens int64 `json:"prompt_cache_miss_tokens"`
		TotalTokens           int64 `json:"total_tokens"`
		PromptTokensDetails   struct {
			CachedTokens int64 `json:"cached_tokens"`
		} `json:"prompt_tokens_details"`
		CompletionTokensDetails struct {
			ReasoningTokens int64 `json:"reasoning_tokens"`
		} `json:"completion_tokens_details"`
	} `json:"usage"`
}

func (c *CompletionResponse) ToResult() (genaiapi.CompletionResult, error) {
	out := genaiapi.CompletionResult{}
	out.InputTokens = c.Usage.PromptTokens
	out.OutputTokens = c.Usage.CompletionTokens
	if len(c.Choices) != 1 {
		return out, fmt.Errorf("expected 1 choice, got %#v", c.Choices)
	}
	if len(c.Choices[0].Message.ToolCall) != 0 {
		out.Type = genaiapi.ToolCalls
		out.ToolCalls = make([]genaiapi.ToolCall, len(c.Choices[0].Message.ToolCall))
		for i, t := range c.Choices[0].Message.ToolCall {
			out.ToolCalls[i].ID = t.ID
			out.ToolCalls[i].Name = t.Function.Name
			out.ToolCalls[i].Arguments = t.Function.Arguments
		}
	} else {
		out.Type = genaiapi.Text
		out.Text = c.Choices[0].Message.Content
	}
	switch role := c.Choices[0].Message.Role; role {
	case "system", "user":
		out.Role = genaiapi.Role(role)
	case "assistant", "model":
		out.Role = genaiapi.Assistant
	default:
		return out, fmt.Errorf("unsupported role %q", role)
	}
	return out, nil
}

type Logprobs struct {
	Content []struct {
		Token       string  `json:"token"`
		Logprob     float64 `json:"logprob"`
		Bytes       []int64 `json:"bytes"`
		TopLogprobs []struct {
			Token   string  `json:"token"`
			Logprob float64 `json:"logprob"`
			Bytes   []int64 `json:"bytes"`
		} `json:"top_logprobs"`
	} `json:"content"`
}

type CompletionStreamChunkResponse struct {
	ID                string `json:"id"`
	Object            string `json:"object"`  // chat.completion.chunk
	Created           int64  `json:"created"` // Unix timestamp
	Model             string `json:"model"`
	SystemFingerprint string `json:"system_fingerprint"`
	Choices           []struct {
		Index        int64    `json:"index"`
		Delta        Message  `json:"delta"`
		Logprobs     Logprobs `json:"logprobs"`
		FinishReason string   `json:"finish_reason"`
	} `json:"choices"`
	Usage struct {
		CompletionTokens      int64 `json:"completion_tokens"`
		PromptTokens          int64 `json:"prompt_tokens"`
		PromptCacheHitTokens  int64 `json:"prompt_cache_hit_tokens"`
		PromptCacheMissTokens int64 `json:"prompt_cache_miss_tokens"`
		TotalTokens           int64 `json:"total_tokens"`
		PromptTokensDetails   struct {
			CachedTokens int64 `json:"cached_tokens"`
		} `json:"prompt_tokens_details"`
	} `json:"usage"`
}

//

type errorResponse struct {
	// Type  string `json:"type"`
	Error struct {
		Message string `json:"message"`
		Type    string `json:"type"`
		Param   string `json:"param"`
		Code    string `json:"code"`
	} `json:"error"`
}

// Client implements the REST JSON based API.
type Client struct {
	apiKey string
	model  string
}

// New creates a new client to talk to the DeepSeek platform API in China.
//
// If apiKey is not provided, it tries to load it from the DEEPSEEK_API_KEY environment variable.
// If none is found, it returns an error.
// Get your API key at https://platform.deepseek.com/api_keys
// If no model is provided, only functions that do not require a model, like ListModels, will work.
// To use multiple models, create multiple clients.
// Use one of the model from https://api-docs.deepseek.com/quick_start/pricing
func New(apiKey, model string) (*Client, error) {
	if apiKey == "" {
		if apiKey = os.Getenv("DEEPSEEK_API_KEY"); apiKey == "" {
			return nil, errors.New("deepseek API key is required; get one at " + apiKeyURL)
		}
	}
	return &Client{apiKey: apiKey, model: model}, nil
}

func (c *Client) Completion(ctx context.Context, msgs []genaiapi.Message, opts genaiapi.Validatable) (genaiapi.CompletionResult, error) {
	// https://api-docs.deepseek.com/api/create-chat-completion
	rpcin := CompletionRequest{Model: c.model}
	if err := rpcin.Init(msgs, opts); err != nil {
		return genaiapi.CompletionResult{}, err
	}
	rpcout := CompletionResponse{}
	if err := c.CompletionRaw(ctx, &rpcin, &rpcout); err != nil {
		return genaiapi.CompletionResult{}, err
	}
	return rpcout.ToResult()
}

func (c *Client) CompletionRaw(ctx context.Context, in *CompletionRequest, out *CompletionResponse) error {
	if err := c.validate(); err != nil {
		return err
	}
	in.Stream = false
	return c.post(ctx, "https://api.deepseek.com/chat/completions", in, out)
}

func (c *Client) CompletionStream(ctx context.Context, msgs []genaiapi.Message, opts genaiapi.Validatable, chunks chan<- genaiapi.MessageChunk) error {
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
			if len(pkt.Choices) != 1 {
				continue
			}
			switch role := pkt.Choices[0].Delta.Role; role {
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
			if word := pkt.Choices[0].Delta.Content; word != "" {
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
	// DeepSeek doesn't HTTP POST support compression.
	resp, err := httpjson.DefaultClient.PostRequest(ctx, "https://api.deepseek.com/chat/completions", h, in)
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
	if string(line) == ": keep-alive" {
		return nil
	}
	const prefix = "data: "
	if !bytes.HasPrefix(line, []byte(prefix)) {
		return fmt.Errorf("unexpected line. expected \"data: \", got %q", line)
	}
	suffix := string(line[len(prefix):])
	if suffix == "[DONE]" {
		return nil
	}
	d := json.NewDecoder(strings.NewReader(suffix))
	d.DisallowUnknownFields()
	d.UseNumber()
	msg := CompletionStreamChunkResponse{}
	if err := d.Decode(&msg); err != nil {
		return fmt.Errorf("failed to decode server response %q: %w", string(line), err)
	}
	out <- msg
	return nil
}

type Model struct {
	ID      string `json:"id"`
	Object  string `json:"object"` // model
	OwnedBy string `json:"owned_by"`
}

func (m *Model) GetID() string {
	return m.ID
}

func (m *Model) String() string {
	return m.ID
}

func (m *Model) Context() int64 {
	return 0
}

func (c *Client) ListModels(ctx context.Context) ([]genaiapi.Model, error) {
	// https://api-docs.deepseek.com/api/list-models
	h := make(http.Header)
	h.Add("Authorization", "Bearer "+c.apiKey)
	var out struct {
		Object string  `json:"object"` // list
		Data   []Model `json:"data"`
	}
	err := httpjson.DefaultClient.Get(ctx, "https://api.deepseek.com/models", h, &out)
	if err != nil {
		return nil, err
	}
	models := make([]genaiapi.Model, len(out.Data))
	for i := range out.Data {
		models[i] = &out.Data[i]
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
	h.Set("Authorization", "Bearer "+c.apiKey)
	// DeepSeek doesn't HTTP POST support compression.
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
				return fmt.Errorf("%w: error %s: %s. You can get a new API key at %s", herr, er.Error.Type, er.Error.Message, apiKeyURL)
			}
			return fmt.Errorf("%w: error %s: %s", herr, er.Error.Type, er.Error.Message)
		}
		return fmt.Errorf("error %s: %s", er.Error.Type, er.Error.Message)
	default:
		var herr *httpjson.Error
		if errors.As(err, &herr) {
			slog.WarnContext(ctx, "deepseek", "url", url, "err", err, "response", string(herr.ResponseBody), "status", herr.StatusCode)
		} else {
			slog.WarnContext(ctx, "deepseek", "url", url, "err", err)
		}
		return err
	}
}

const apiKeyURL = "https://platform.deepseek.com/api_keys"

var (
	_ genaiapi.CompletionProvider = &Client{}
	_ genaiapi.ModelProvider      = &Client{}
)
