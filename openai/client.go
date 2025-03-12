// Copyright 2024 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Package openai implements a client for the OpenAI API.
//
// It is described at https://platform.openai.com/docs/api-reference/
package openai

// See official client at https://github.com/openai/openai-go

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
	"mime"
	"net/http"
	"strings"
	"time"

	"github.com/maruel/genai/genaiapi"
	"github.com/maruel/genai/internal"
	"github.com/maruel/httpjson"
)

// https://platform.openai.com/docs/api-reference/chat/create
type CompletionRequest struct {
	Model               string             `json:"model"`
	MaxTokens           int64              `json:"max_tokens,omitzero"` // Deprecated
	MaxCompletionTokens int64              `json:"max_completion_tokens,omitzero"`
	Stream              bool               `json:"stream"`
	Messages            []Message          `json:"messages"`
	Seed                int64              `json:"seed,omitzero"`
	Temperature         float64            `json:"temperature,omitzero"` // [0, 2]
	Store               bool               `json:"store,omitzero"`
	ReasoningEffort     string             `json:"reasoning_effort,omitzero"` // "low", "medium", "high"
	Metadata            map[string]string  `json:"metadata,omitzero"`
	FrequencyPenalty    float64            `json:"frequency_penalty,omitzero"` // [-2.0, 2.0]
	LogitBias           map[string]float64 `json:"logit_bias,omitzero"`
	// See https://cookbook.openai.com/examples/using_logprobs
	Logprobs    bool     `json:"logprobs,omitzero"`
	TopLogprobs int64    `json:"top_logprobs,omitzero"` // [0, 20]
	N           int64    `json:"n,omitzero"`            // Number of choices
	Modalities  []string `json:"modalities,omitzero"`   // text, audio
	Prediction  struct {
		Type    string `json:"type,omitzero"` // "content"
		Content []struct {
			Type string `json:"type,omitzero"` // "text"
			Text string `json:"text,omitzero"`
		} `json:"content,omitzero"`
	} `json:"prediction,omitzero"`
	Audio struct {
		Voice  string `json:"voice,omitzero"`  // "ash", "ballad", "coral", "sage", "verse", "alloy", "echo", "shimmer"
		Format string `json:"format,omitzero"` // "mp3", "wav", "flac", "opus", "pcm16"
	} `json:"audio,omitzero"`
	PresencePenalty float64 `json:"presence_penalty,omitzero"` // [-2.0, 2.0]
	ResponseFormat  struct {
		Type       string `json:"type,omitzero"` // "text", "json_object", "json_schema"
		JSONSchema struct {
			Description string     `json:"description,omitzero"`
			Name        string     `json:"name,omitzero"`
			Schema      JSONSchema `json:"schema,omitzero"`
			Strict      bool       `json:"strict,omitzero"`
		} `json:"json_schema,omitzero"`
	} `json:"response_format,omitzero"`
	ServiceTier   string   `json:"service_tier,omitzero"` // "auto", "default"
	Stop          []string `json:"stop,omitzero"`         // keywords to stop completion
	StreamOptions struct {
		IncludeUsage bool `json:"include_usage,omitzero"`
	} `json:"stream_options,omitzero"`
	TopP  float64 `json:"top_p,omitzero"` // [0, 1]
	Tools []Tool  `json:"tools,omitzero"`
	// Alternative when forcing a specific function. This can probably be achieved
	// by providing a single tool and ToolChoice == "required".
	// ToolChoice struct {
	// 	Type     string `json:"type,omitzero"` // "function"
	// 	Function struct {
	// 		Name string `json:"name,omitzero"`
	// 	} `json:"function,omitzero"`
	// } `json:"tool_choice,omitzero"`
	ToolChoice        string `json:"tool_choice,omitzero"` // "none", "auto", "required"
	ParallelToolCalls bool   `json:"parallel_tool_calls,omitzero"`
	User              string `json:"user,omitzero"`
}

func (c *CompletionRequest) fromOpts(opts any) error {
	if opts != nil {
		switch v := opts.(type) {
		case *genaiapi.CompletionOptions:
			c.MaxTokens = v.MaxTokens
			c.Seed = v.Seed
			c.Temperature = v.Temperature
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
			// Starting with 01.
			c.Messages[i].Role = "developer"
		case genaiapi.User, genaiapi.Assistant, genaiapi.Tool:
			c.Messages[i].Role = string(m.Role)
		default:
			return fmt.Errorf("message %d: unsupported role %q", i, m.Role)
		}
		c.Messages[i].Content = []Content{{}}
		switch m.Type {
		case genaiapi.Text:
			c.Messages[i].Content[0].Type = "text"
			c.Messages[i].Content[0].Text = m.Text
		case genaiapi.Document:
			// https://platform.openai.com/docs/guides/images?api-mode=chat&format=base64-encoded#image-input-requirements
			mimeType, data, err := internal.ParseDocument(&m, 10*1024*1024)
			if err != nil {
				return fmt.Errorf("message %d: %w", i, err)
			}
			// OpenAI require a mime-type to determine if image, sound or PDF.
			if mimeType == "" {
				return fmt.Errorf("message %d: unspecified mime type for URL %q", i, m.URL)
			}
			switch {
			case strings.HasPrefix(mimeType, "image/"):
				c.Messages[i].Content[0].Type = "image_url"
				if m.URL == "" {
					c.Messages[i].Content[0].ImageURL.URL = fmt.Sprintf("data:%s;base64,%s", mimeType, base64.StdEncoding.EncodeToString(data))
				} else {
					c.Messages[i].Content[0].ImageURL.URL = m.URL
				}
			case mimeType == "audio/mpeg":
				if m.URL != "" {
					return fmt.Errorf("message %d: URL to audio file not supported", i)
				}
				c.Messages[i].Content[0].Type = "input_audio"
				c.Messages[i].Content[0].InputAudio.Data = data
				c.Messages[i].Content[0].InputAudio.Format = "mp3"
			case mimeType == "audio/wav":
				if m.URL != "" {
					return fmt.Errorf("message %d: URL to audio file not supported", i)
				}
				c.Messages[i].Content[0].Type = "input_audio"
				c.Messages[i].Content[0].InputAudio.Data = data
				c.Messages[i].Content[0].InputAudio.Format = "wav"
			default:
				if m.URL != "" {
					return fmt.Errorf("message %d: URL to %s file not supported", i, mimeType)
				}
				filename := m.Filename
				if filename == "" {
					exts, err := mime.ExtensionsByType(mimeType)
					if err != nil {
						return fmt.Errorf("message %d: %w", i, err)
					}
					if len(exts) == 0 {
						return fmt.Errorf("message %d: unknown extension for mime type %s", i, mimeType)
					}
					filename = "content" + exts[0]
				}
				c.Messages[i].Content[0].Type = "input_file"
				c.Messages[i].Content[0].Filename = filename
				c.Messages[i].Content[0].FileData = data
			}
		default:
			return fmt.Errorf("message %d: unsupported content type %s", i, m.Type)
		}
	}
	return nil
}

type Message struct {
	Role    string   `json:"role,omitzero"`
	Content Contents `json:"content,omitzero"`
	Name    string   `json:"name,omitzero"`
	Refusal string   `json:"refusal,omitzero"`
	Audio   struct {
		ID string `json:"id,omitzero"`
	} `json:"audio,omitzero"`
	ToolCalls   []ToolCall `json:"tool_calls,omitzero"`
	ToolCallID  string     `json:"tool_call_id,omitzero"`
	Annotations []struct{} `json:"annotations,omitzero"`
}

type Contents []Content

// OpenAI replies with content as a string.
func (c *Contents) UnmarshalJSON(data []byte) error {
	var v []Content
	if err := json.Unmarshal(data, &v); err != nil {
		s := ""
		if err = json.Unmarshal(data, &s); err != nil {
			return err
		}
		*c = []Content{{Type: "text", Text: s}}
		return nil
	}
	*c = Contents(v)
	return nil
}

type Content struct {
	Type string `json:"type,omitzero"` // "text", "image_url", "input_file", "input_audio"

	// Type == "text"
	Text string `json:"text,omitzero"`

	// Type == "image_url"
	ImageURL struct {
		URL    string `json:"url,omitzero"`
		Detail string `json:"detail,omitzero"` // "auto", "low", "high"
	} `json:"image_url,omitzero"`

	// Type == "input_audio"
	InputAudio struct {
		Data   []byte `json:"data,omitzero"`
		Format string `json:"format,omitzero"` // "mp3", "wav"
	} `json:"input_audio,omitzero"`

	// Type == "input_file"
	// Either FileID or both Filename and FileData.
	FileID   string `json:"file_id,omitzero"`
	Filename string `json:"filename,omitzero"`
	FileData []byte `json:"file_data,omitzero"`
}

type ToolCall struct {
	ID       string `json:"id"`
	Type     string `json:"type"` // function
	Function struct {
		Name      string `json:"name"`
		Arguments string `json:"arguments"` // Generally JSON
	} `json:"function"`
}

type Tool struct {
	Type     string `json:"type,omitzero"` // "function"
	Function struct {
		Description string         `json:"description,omitzero"`
		Name        string         `json:"name,omitzero"`
		Parameters  map[string]any `json:"parameters,omitzero"`
		Strict      bool           `json:"strict,omitzero"`
	} `json:"function,omitzero"`
}

type JSONSchema any

// CompletionResponse is documented at
// https://platform.openai.com/docs/api-reference/chat/object
type CompletionResponse struct {
	Choices []struct {
		// FinishReason is one of "stop", "length", "content_filter" or "tool_calls".
		FinishReason string   `json:"finish_reason"`
		Index        int64    `json:"index"`
		Message      Message  `json:"message"`
		Logprobs     Logprobs `json:"logprobs"`
	} `json:"choices"`
	Created Time   `json:"created"`
	ID      string `json:"id"`
	Model   string `json:"model"`
	Object  string `json:"object"`
	Usage   struct {
		PromptTokens        int64 `json:"prompt_tokens"`
		CompletionTokens    int64 `json:"completion_tokens"`
		TotalTokens         int64 `json:"total_tokens"`
		PromptTokensDetails struct {
			CachedTokens int64 `json:"cached_tokens"`
			AudioTokens  int64 `json:"audio_tokens"`
		} `json:"prompt_tokens_details"`
		CompletionTokensDetails struct {
			ReasoningTokens          int64 `json:"reasoning_tokens"`
			AudioTokens              int64 `json:"audio_tokens"`
			AcceptedPredictionTokens int64 `json:"accepted_prediction_tokens"`
			RejectedPredictionTokens int64 `json:"rejected_prediction_tokens"`
		} `json:"completion_tokens_details"`
	} `json:"usage"`
	ServiceTier       string `json:"service_tier"`
	SystemFingerprint string `json:"system_fingerprint"`
}

type Logprobs struct {
	Content []struct {
		Token       string  `json:"token"`
		Logprob     float64 `json:"logprob"`
		Bytes       []int   `json:"bytes"`
		TopLogprobs []struct {
			TODO string `json:"todo"`
		} `json:"top_logprobs"`
	} `json:"content"`
	Refusal string `json:"refusal"`
}

// CompletionStreamChunkResponse is not documented?
type CompletionStreamChunkResponse struct {
	Choices []struct {
		Delta struct {
			Content  string `json:"content"`
			Role     string `json:"role"`
			Refulsal string `json:"refusal"`
		} `json:"delta"`
		// FinishReason is one of null, "stop", "length", "content_filter" or "tool_calls".
		FinishReason string   `json:"finish_reason"`
		Index        int64    `json:"index"`
		Logprobs     Logprobs `json:"logprobs"`
	} `json:"choices"`
	Created           Time   `json:"created"`
	ID                string `json:"id"`
	Model             string `json:"model"`
	Object            string `json:"object"`
	ServiceTier       string `json:"service_tier"`
	SystemFingerprint string `json:"system_fingerprint"`
	Usage             struct {
		CompletionTokens int64 `json:"completion_tokens"`
		PromptTokens     int64 `json:"prompt_tokens"`
		TotalTokens      int64 `json:"total_tokens"`
	} `json:"usage"`
}

//

type errorResponse struct {
	Error errorResponseError `json:"error"`
}

type errorResponseError struct {
	Code    string `json:"code"`
	Message string `json:"message"`
	Status  string `json:"status"`
	Type    string `json:"type"`
	Param   string `json:"param"`
}

//

type Client struct {
	// ApiKey can be retrieved from https://platform.openai.com/settings/organization/api-keys
	ApiKey string
	// Model to use, from https://platform.openai.com/docs/models
	Model string
}

// TODO: Upload files
// https://platform.openai.com/docs/api-reference/uploads/create
// TTL 1h

func (c *Client) Completion(ctx context.Context, msgs []genaiapi.Message, opts any) (genaiapi.Message, error) {
	msg := genaiapi.Message{}
	in := CompletionRequest{Model: c.Model}
	if err := in.fromOpts(opts); err != nil {
		return msg, err
	}
	if err := in.fromMsgs(msgs); err != nil {
		return msg, err
	}
	out := CompletionResponse{}
	if err := c.CompletionRaw(ctx, &in, &out); err != nil {
		return msg, fmt.Errorf("failed to get chat response: %w", err)
	}
	if len(out.Choices) != 1 {
		return msg, fmt.Errorf("server returned an unexpected number of choices, expected 1, got %d", len(out.Choices))
	}
	msg.Type = genaiapi.Text
	msg.Text = out.Choices[0].Message.Content[0].Text
	switch role := out.Choices[0].Message.Role; role {
	case "assistant", "model":
		msg.Role = genaiapi.Assistant
	case "developer":
		msg.Role = genaiapi.System
	case "user":
		msg.Role = genaiapi.User
	default:
		return msg, fmt.Errorf("unsupported role %q", role)
	}
	return msg, nil
}

func (c *Client) CompletionRaw(ctx context.Context, in *CompletionRequest, out *CompletionResponse) error {
	// https://platform.openai.com/docs/api-reference/chat/create
	if err := c.validate(true); err != nil {
		return err
	}
	return c.post(ctx, "https://api.openai.com/v1/chat/completions", in, out)
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
	start := time.Now()
	go func() {
		for msg := range ch {
			word := msg.Choices[0].Delta.Content
			slog.DebugContext(ctx, "openai", "word", word, "duration", time.Since(start).Round(time.Millisecond))
			// TODO: Remove.
			switch word {
			// Llama-3, Gemma-2, Phi-3
			case "<|eot_id|>", "<end_of_turn>", "<|end|>", "<|endoftext|>":
				continue
			case "":
			default:
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
	// OpenAI doesn't HTTP POST support compression.
	resp, err := httpjson.DefaultClient.PostRequest(ctx, "https://api.openai.com/v1/chat/completions", h, in)
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
		if len(msg.Choices) != 1 {
			return fmt.Errorf("server returned an unexpected number of choices, expected 1, got %d", len(msg.Choices))
		}
		out <- msg
	}
}

type Time int64

func (t *Time) AsTime() time.Time {
	return time.Unix(int64(*t), 0)
}

// https://platform.openai.com/docs/api-reference/models/object
type Model struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Created Time   `json:"created"`
	OwnedBy string `json:"owned_by"`
}

func (m *Model) GetID() string {
	return m.ID
}

func (m *Model) String() string {
	return fmt.Sprintf("%s (%s)", m.ID, m.Created.AsTime().Format("2006-01-02"))
}

func (m *Model) Context() int64 {
	return 0
}

func (c *Client) ListModels(ctx context.Context) ([]genaiapi.Model, error) {
	// https://platform.openai.com/docs/api-reference/models/list
	if err := c.validate(false); err != nil {
		return nil, err
	}
	h := make(http.Header)
	h.Add("Authorization", "Bearer "+c.ApiKey)
	var out struct {
		Object string  `json:"object"` // list
		Data   []Model `json:"data"`
	}
	err := httpjson.DefaultClient.Get(ctx, "https://api.openai.com/v1/models", h, &out)
	if err != nil {
		return nil, err
	}
	models := make([]genaiapi.Model, len(out.Data))
	for i := range out.Data {
		models[i] = &out.Data[i]
	}
	return models, err
}

func (c *Client) validate(needModel bool) error {
	if c.ApiKey == "" {
		return errors.New("openai ApiKey is required; get one at " + apiKeyURL)
	}
	if needModel && c.Model == "" {
		return errors.New("a Model is required")
	}
	return nil
}

func (c *Client) post(ctx context.Context, url string, in, out any) error {
	h := make(http.Header)
	h.Add("Authorization", "Bearer "+c.ApiKey)
	// OpenAI doesn't HTTP POST support compression.
	resp, err := httpjson.DefaultClient.PostRequest(ctx, url, h, in)
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
			if er.Error.Code == "" {
				return fmt.Errorf("%w: error %s: %s", herr, er.Error.Type, er.Error.Message)
			}
			return fmt.Errorf("%w: error %s (%s): %s", herr, er.Error.Code, er.Error.Status, er.Error.Message)
		}
		if er.Error.Code == "" {
			return fmt.Errorf("error %s: %s", er.Error.Type, er.Error.Message)
		}
		return fmt.Errorf("error %s (%s): %s", er.Error.Code, er.Error.Status, er.Error.Message)
	default:
		var herr *httpjson.Error
		if errors.As(err, &herr) {
			slog.WarnContext(ctx, "openai", "url", url, "err", err, "response", string(herr.ResponseBody), "status", herr.StatusCode)
		} else {
			slog.WarnContext(ctx, "openai", "url", url, "err", err)
		}
		return err
	}
}

const apiKeyURL = "https://platform.openai.com/settings/organization/api-keys"

var _ genaiapi.CompletionProvider = &Client{}
