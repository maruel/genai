// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Package togetherai implements a client for the Together.ai API.
//
// It is described at https://docs.together.ai/docs/
package togetherai

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
	"math/big"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/invopop/jsonschema"
	"github.com/maruel/genai/genaiapi"
	"github.com/maruel/genai/internal"
	"github.com/maruel/httpjson"
)

// Oficial python client library at https://github.com/togethercomputer/together-python/tree/main/src/together

// https://docs.together.ai/reference/chat-completions-1
type CompletionRequest struct {
	Model                         string             `json:"model"`
	StreamTokens                  bool               `json:"stream_tokens"`
	Messages                      []Message          `json:"messages"`
	MaxTokens                     int64              `json:"max_tokens,omitzero"`
	Stop                          []string           `json:"stop,omitzero"`
	Temperature                   float64            `json:"temperature,omitzero"` // [0, 1]
	TopP                          float64            `json:"top_p,omitzero"`       // [0, 1]
	TopK                          int64              `json:"top_k,omitzero"`
	ContextLengthExceededBehavior string             `json:"context_length_exceeded_behavior,omitzero"` // "error", "truncate"
	RepetitionPenalty             float64            `json:"repetition_penalty,omitzero"`
	Logprobs                      int32              `json:"logprobs,omitzero"` // bool as 0 or 1
	Echo                          bool               `json:"echo,omitzero"`
	N                             int32              `json:"n,omitzero"`                 // Number of completions to generate
	PresencePenalty               float64            `json:"presence_penalty,omitzero"`  // [-2.0, 2.0]
	FrequencyPenalty              float64            `json:"frequency_penalty,omitzero"` // [-2.0, 2.0]
	LogitBias                     map[string]float64 `json:"logit_bias,omitzero"`
	Seed                          int64              `json:"seed,omitzero"`
	ResponseFormat                struct {
		Type   string             `json:"type,omitzero"` // "json_object", "json_schema" according to python library.
		Schema *jsonschema.Schema `json:"schema,omitzero"`
	} `json:"response_format,omitzero"`
	Tools       []Tool `json:"tools,omitzero"`
	ToolChoice  string `json:"tool_choice,omitzero"`  // "auto" or a []Tool
	SafetyModel string `json:"safety_model,omitzero"` // https://docs.together.ai/docs/inference-models#moderation-models
}

func (c *CompletionRequest) Init(msgs genaiapi.Messages, opts genaiapi.Validatable) error {
	var errs []error
	sp := ""
	if opts != nil {
		if err := opts.Validate(); err != nil {
			errs = append(errs, err)
		} else {
			switch v := opts.(type) {
			case *genaiapi.CompletionOptions:
				c.MaxTokens = v.MaxTokens
				c.Temperature = v.Temperature
				c.TopP = v.TopP
				sp = v.SystemPrompt
				c.Seed = v.Seed
				c.TopK = v.TopK
				if v.ReplyAsJSON {
					c.ResponseFormat.Type = "json_object"
				}
				c.Stop = v.Stop
				if v.DecodeAs != nil {
					// Warning: using a model small may fail.
					c.ResponseFormat.Type = "json_schema"
					c.ResponseFormat.Schema = jsonschema.Reflect(v.DecodeAs)
				}
				if len(v.Tools) != 0 {
					c.ToolChoice = "required"
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

type Message struct {
	Role    string    `json:"role,omitzero"`
	Content []Content `json:"content,omitzero"`
}

func (msg *Message) From(m *genaiapi.Message) error {
	switch m.Role {
	case genaiapi.User, genaiapi.Assistant:
		msg.Role = string(m.Role)
	default:
		return fmt.Errorf("unsupported role %q", m.Role)
	}
	msg.Content = []Content{{}}
	switch m.Type {
	case genaiapi.Text:
		msg.Content[0].Type = "text"
		msg.Content[0].Text = m.Text
	case genaiapi.Document:
		mimeType, data, err := internal.ParseDocument(m, 10*1024*1024)
		if err != nil {
			return err
		}
		switch {
		case strings.HasPrefix(mimeType, "image/"):
			msg.Content[0].Type = "image_url"
			if m.URL == "" {
				msg.Content[0].ImageURL.URL = fmt.Sprintf("data:%s;base64,%s", mimeType, base64.StdEncoding.EncodeToString(data))
			} else {
				msg.Content[0].ImageURL.URL = m.URL
			}
		case strings.HasPrefix(mimeType, "video/"):
			msg.Content[0].Type = "video_url"
			if m.URL == "" {
				msg.Content[0].VideoURL.URL = fmt.Sprintf("data:%s;base64,%s", mimeType, base64.StdEncoding.EncodeToString(data))
			} else {
				msg.Content[0].VideoURL.URL = m.URL
			}
		default:
			return fmt.Errorf("unsupported mime type %s", mimeType)
		}
	default:
		return fmt.Errorf("unsupported content type %s", m.Type)
	}
	return nil
}

type Content struct {
	Type string `json:"type,omitzero"` // "text", "image_url", "video_url"

	// Type == "text"
	Text string `json:"text,omitzero"`

	// Type == "image_url"
	ImageURL struct {
		URL string `json:"url,omitzero"`
	} `json:"image_url,omitzero"`

	// Type == "video_url"
	VideoURL struct {
		URL string `json:"url,omitzero"`
	} `json:"video_url,omitzero"`
}

type Tool struct {
	Type     string `json:"type,omitzero"` // "function"
	Function struct {
		Name        string             `json:"name,omitzero"`
		Description string             `json:"description,omitzero"`
		Parameters  *jsonschema.Schema `json:"parameters,omitzero"`
	} `json:"function,omitzero"`
}

type CompletionResponse struct {
	ID      string   `json:"id"`
	Prompt  []string `json:"prompt"`
	Choices []struct {
		// Text  string `json:"text"`
		Index int64 `json:"index"`
		// The seed is returned as a int128.
		Seed big.Int `json:"seed"`
		// FinishReason is one of "stop", "eos", "length", "function_call" or "tool_calls".
		FinishReason string `json:"finish_reason"`
		Message      struct {
			Role      string     `json:"role"`
			Content   string     `json:"content"`
			ToolCalls []ToolCall `json:"tool_calls"`
		} `json:"message"`
		Logprobs struct {
			TokenIDs      []int64   `json:"token_ids"`
			Tokens        []string  `json:"tokens"`
			TokenLogprobs []float64 `json:"token_logprobs"`
		} `json:"logprobs"`
	} `json:"choices"`
	Usage struct {
		PromptTokens     int64 `json:"prompt_tokens"`
		CompletionTokens int64 `json:"completion_tokens"`
		TotalTokens      int64 `json:"total_tokens"`
	} `json:"usage"`
	Created Time   `json:"created"`
	Model   string `json:"model"`
	Object  string `json:"object"` // "chat.completion"
}

func (c *CompletionResponse) ToResult() (genaiapi.CompletionResult, error) {
	out := genaiapi.CompletionResult{}
	out.InputTokens = c.Usage.PromptTokens
	out.OutputTokens = c.Usage.CompletionTokens
	if len(c.Choices) != 1 {
		return out, fmt.Errorf("server returned an unexpected number of choices, expected 1, got %d", len(c.Choices))
	}
	// Warning: using a model small may fail.
	if len(c.Choices[0].Message.ToolCalls) != 0 {
		out.Type = genaiapi.ToolCalls
		for _, t := range c.Choices[0].Message.ToolCalls {
			out.ToolCalls = append(out.ToolCalls, genaiapi.ToolCall{ID: t.ID, Name: t.Function.Name, Arguments: t.Function.Arguments})
		}
	} else {
		out.Type = genaiapi.Text
		out.Text = c.Choices[0].Message.Content
	}
	switch role := c.Choices[0].Message.Role; role {
	case "system", "assistant", "user":
		out.Role = genaiapi.Role(role)
	default:
		return out, fmt.Errorf("unsupported role %q", role)
	}
	return out, nil
}

type ToolCall struct {
	Index    int64  `json:"index"`
	ID       string `json:"id"`
	Type     string `json:"type"` // function
	Function struct {
		Name      string `json:"name"`
		Arguments string `json:"arguments"`
	} `json:"function"`
}

type CompletionStreamChunkResponse struct {
	ID      string `json:"id"`
	Object  string `json:"object"` // "chat.completion.chunk"
	Created Time   `json:"created"`
	Model   string `json:"model"`
	Choices []struct {
		Index int64  `json:"index"`
		Text  string `json:"text"` // Duplicated to Delta.Text
		Seed  int64  `json:"seed"`
		Delta struct {
			TokenID   int64         `json:"token_id"`
			Role      genaiapi.Role `json:"role"`
			Content   string        `json:"content"`
			ToolCalls []ToolCall    `json:"tool_calls"`
		} `json:"delta"`
		Logprobs     struct{} `json:"logprobs"`
		FinishReason string   `json:"finish_reason"` //
	} `json:"choices"`
	// SystemFingerprint string `json:"system_fingerprint"`
	Usage struct {
		PromptTokens     int64 `json:"prompt_tokens"`
		CompletionTokens int64 `json:"completion_tokens"`
		TotalTokens      int64 `json:"total_tokens"`
	} `json:"usage"`
}

//

type errorResponse struct {
	ID    string `json:"id"`
	Error struct {
		Message string `json:"message"`
		Type    string `json:"type"`
		Code    string `json:"code"`
		Param   string `json:"param"`
	} `json:"error"`
}

// Client implements the REST JSON based API.
type Client struct {
	apiKey string
	model  string
}

// New creates a new client to talk to the Together.AI platform API.
//
// If apiKey is not provided, it tries to load it from the TOGETHER_API_KEY environment variable.
// If none is found, it returns an error.
// Get your API key at https://api.together.ai/settings/api-keys
// If no model is provided, only functions that do not require a model, like ListModels, will work.
// To use multiple models, create multiple clients.
// Use one of the model from https://docs.together.ai/docs/serverless-models
func New(apiKey, model string) (*Client, error) {
	if apiKey == "" {
		if apiKey = os.Getenv("TOGETHER_API_KEY"); apiKey == "" {
			return nil, errors.New("together.ai API key is required; get one at " + apiKeyURL)
		}
	}
	return &Client{apiKey: apiKey, model: model}, nil
}

func (c *Client) Completion(ctx context.Context, msgs genaiapi.Messages, opts genaiapi.Validatable) (genaiapi.CompletionResult, error) {
	// https://docs.together.ai/docs/chat-overview
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
	in.StreamTokens = false
	return c.post(ctx, "https://api.together.xyz/v1/chat/completions", in, out)
}

func (c *Client) CompletionStream(ctx context.Context, msgs genaiapi.Messages, opts genaiapi.Validatable, chunks chan<- genaiapi.MessageFragment) error {
	in := CompletionRequest{Model: c.model}
	if err := in.Init(msgs, opts); err != nil {
		return err
	}
	ch := make(chan CompletionStreamChunkResponse)
	end := make(chan error)
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()
	go func() {
		var lastRole genaiapi.Role
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
				chunks <- genaiapi.MessageFragment{Role: lastRole, Type: genaiapi.Text, TextFragment: word}
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
	in.StreamTokens = true
	h := make(http.Header)
	h.Add("Authorization", "Bearer "+c.apiKey)
	// Together.ai doesn't HTTP POST support compression.
	// TODO Test
	// httpjson.DefaultClient.PostCompress = "gzip"
	resp, err := httpjson.DefaultClient.PostRequest(ctx, "https://api.together.xyz/v1/chat/completions", h, in)
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

type Time int64

func (t *Time) AsTime() time.Time {
	return time.Unix(int64(*t), 0)
}

type Model struct {
	ID            string `json:"id"`
	Object        string `json:"object"`
	Created       Time   `json:"created"`
	Type          string `json:"type"` // chat,moderation,image,
	Running       bool   `json:"running"`
	DisplayName   string `json:"display_name"`
	Organization  string `json:"organization"`
	Link          string `json:"link"`
	License       string `json:"license"`
	ContextLength int64  `json:"context_length"`
	Config        struct {
		ChatTemplate string   `json:"chat_template"`
		Stop         []string `json:"stop"`
		BosToken     string   `json:"bos_token"`
		EosToken     string   `json:"eos_token"`
	} `json:"config"`
	Pricing struct {
		Hourly   float64 `json:"hourly"`
		Input    float64 `json:"input"`
		Output   float64 `json:"output"`
		Base     float64 `json:"base"`
		Finetune float64 `json:"finetune"`
	} `json:"pricing"`
}

func (m *Model) GetID() string {
	return m.ID
}

func (m *Model) String() string {
	return fmt.Sprintf("%s (%s): %s Context: %d; in: %.2f$/Mt out: %.2f$/Mt", m.ID, m.Created.AsTime().Format("2006-01-02"), m.Type, m.ContextLength, m.Pricing.Input, m.Pricing.Output)
}

func (m *Model) Context() int64 {
	return m.ContextLength
}

func (c *Client) ListModels(ctx context.Context) ([]genaiapi.Model, error) {
	// https://docs.together.ai/reference/models-1
	h := make(http.Header)
	h.Add("Authorization", "Bearer "+c.apiKey)
	var out []Model
	err := httpjson.DefaultClient.Get(ctx, "https://api.together.xyz/v1/models", h, &out)
	if err != nil {
		return nil, err
	}
	models := make([]genaiapi.Model, len(out))
	for i := range out {
		models[i] = &out[i]
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
	// Together.AI doesn't HTTP POST support compression.
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
				return fmt.Errorf("%w: error %s (%s): %s. You can get a new API key at %s", herr, er.Error.Code, er.Error.Type, er.Error.Message, apiKeyURL)
			}
			return fmt.Errorf("%w: error %s (%s): %s", herr, er.Error.Code, er.Error.Type, er.Error.Message)
		}
		return fmt.Errorf("error %s (%s): %s", er.Error.Code, er.Error.Type, er.Error.Message)
	default:
		var herr *httpjson.Error
		if errors.As(err, &herr) {
			slog.WarnContext(ctx, "togetherai", "url", url, "err", err, "response", string(herr.ResponseBody), "status", herr.StatusCode)
		} else {
			slog.WarnContext(ctx, "togetherai", "url", url, "err", err)
		}
		return err
	}
}

const apiKeyURL = "https://api.together.ai/settings/api-keys"

var (
	_ genaiapi.CompletionProvider = &Client{}
	_ genaiapi.ModelProvider      = &Client{}
)
