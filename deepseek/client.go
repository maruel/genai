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

func (c *CompletionRequest) fromOpts(opts any) error {
	if opts != nil {
		switch v := opts.(type) {
		case *genaiapi.CompletionOptions:
			c.MaxToks = v.MaxTokens
			c.Temperature = v.Temperature
			if v.Seed != 0 {
				return errors.New("seed is not supported")
			}
			if v.ReplyAsJSON {
				c.ResponseFormat.Type = "json_object"
			}
			if !v.JSONSchema.IsZero() {
				return errors.New("deepseek doesn't support JSON schema")
			}
			if len(v.Tools) != 0 {
				return errors.New("tools support is not implemented yet")
			}
		default:
			return fmt.Errorf("unsupported options type %T", opts)
		}
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
		case genaiapi.User, genaiapi.Assistant:
		default:
			return fmt.Errorf("message %d: unsupported role %q", i, m.Role)
		}
		switch m.Type {
		case genaiapi.Text:
		default:
			return fmt.Errorf("message %d: unsupported content type %s", i, m.Type)
		}
		c.Messages[i].Role = string(m.Role)
		c.Messages[i].Content = m.Text
	}
	return nil
}

type Message struct {
	Role             string `json:"role"`
	Content          string `json:"content"`
	Name             string `json:"name,omitzero"`
	Prefix           bool   `json:"prefix,omitzero"`
	ReasoningContent string `json:"reasoning_content,omitzero"`
	ToolCallID       string `json:"tool_call_id,omitzero"`
}

type Tool struct {
	Type     string `json:"type"` // "function"
	Function struct {
		Name        string              `json:"name,omitzero"`
		Description string              `json:"description,omitzero"`
		Parameters  genaiapi.JSONSchema `json:"parameters,omitzero"`
	} `json:"function"`
}

type CompletionResponse struct {
	ID      string `json:"id"`
	Choices []struct {
		FinishReason string   `json:"finish_reason"`
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

func (c *Client) Completion(ctx context.Context, msgs []genaiapi.Message, opts any) (genaiapi.CompletionResult, error) {
	// https://api-docs.deepseek.com/api/create-chat-completion
	out := genaiapi.CompletionResult{}
	rpcin := CompletionRequest{Model: c.model, Messages: make([]Message, 0, len(msgs))}
	if err := rpcin.fromOpts(opts); err != nil {
		return out, err
	}
	if err := rpcin.fromMsgs(msgs); err != nil {
		return out, err
	}
	rpcout := CompletionResponse{}
	if err := c.CompletionRaw(ctx, &rpcin, &rpcout); err != nil {
		return out, err
	}
	out.InputTokens = rpcout.Usage.PromptTokens
	out.OutputTokens = rpcout.Usage.CompletionTokens
	if len(rpcout.Choices) != 1 {
		return out, fmt.Errorf("expected 1 choice, got %#v", rpcout.Choices)
	}
	out.Type = genaiapi.Text
	out.Text = rpcout.Choices[0].Message.Content
	switch role := rpcout.Choices[0].Message.Role; role {
	case "system", "user":
		out.Role = genaiapi.Role(role)
	case "assistant", "model":
		out.Role = genaiapi.Assistant
	default:
		return out, fmt.Errorf("unsupported role %q", role)
	}
	return out, nil
}

func (c *Client) CompletionRaw(ctx context.Context, in *CompletionRequest, out *CompletionResponse) error {
	if err := c.validate(); err != nil {
		return err
	}
	return c.post(ctx, "https://api.deepseek.com/chat/completions", in, out)
}

func (c *Client) CompletionStream(ctx context.Context, msgs []genaiapi.Message, opts any, chunks chan<- genaiapi.MessageChunk) error {
	in := CompletionRequest{Model: c.model, Messages: make([]Message, 0, len(msgs)), Stream: true}
	if err := in.fromOpts(opts); err != nil {
		return err
	}
	if err := in.fromMsgs(msgs); err != nil {
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
	h := make(http.Header)
	h.Add("Authorization", "Bearer "+c.apiKey)
	// DeepSeek doesn't HTTP POST support compression.
	resp, err := httpjson.DefaultClient.PostRequest(ctx, "https://api.deepseek.com/chat/completions", h, in)
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
		if len(line) == 0 || string(line) == ": keep-alive" {
			continue
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
		if err = d.Decode(&msg); err != nil {
			return fmt.Errorf("failed to decode server response %q: %w", string(line), err)
		}
		out <- msg
	}
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
