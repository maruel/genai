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

// https://platform.openai.com/docs/api-reference/chat/create
type ChatRequest struct {
	Model            string             `json:"model"`
	MaxTokens        int64              `json:"max_tokens,omitzero"` // Deprecated
	MaxChatTokens    int64              `json:"max_completion_tokens,omitzero"`
	Stream           bool               `json:"stream"`
	Messages         []Message          `json:"messages"`
	Seed             int64              `json:"seed,omitzero"`
	Temperature      float64            `json:"temperature,omitzero"` // [0, 2]
	Store            bool               `json:"store,omitzero"`
	ReasoningEffort  ReasoningEffort    `json:"reasoning_effort,omitzero"`
	Metadata         map[string]string  `json:"metadata,omitzero"`
	FrequencyPenalty float64            `json:"frequency_penalty,omitzero"` // [-2.0, 2.0]
	LogitBias        map[string]float64 `json:"logit_bias,omitzero"`
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
			Description string             `json:"description,omitzero"`
			Name        string             `json:"name,omitzero"`
			Schema      *jsonschema.Schema `json:"schema,omitzero"`
			Strict      bool               `json:"strict,omitzero"`
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
				c.MaxChatTokens = v.MaxTokens
				if (strings.HasPrefix(model, "gpt-4o-") && strings.Contains(model, "-search")) ||
					model == "o1" ||
					strings.HasPrefix(model, "o1-") ||
					strings.HasPrefix(model, "o3-") ||
					strings.HasPrefix(model, "o4-") {
					if v.Temperature != 0 {
						unsupported = append(unsupported, "Temperature")
					}
				} else {
					c.Temperature = v.Temperature
				}
				c.TopP = v.TopP
				sp = v.SystemPrompt
				if strings.HasPrefix(model, "gpt-4o-") && strings.Contains(model, "-search") {
					if v.Seed != 0 {
						unsupported = append(unsupported, "Seed")
					}
				} else {
					c.Seed = v.Seed
				}
				if v.TopK != 0 {
					// Track this as an unsupported feature that can be ignored
					unsupported = append(unsupported, "TopK")
				}
				c.Stop = v.Stop
				if v.ReplyAsJSON {
					c.ResponseFormat.Type = "json_object"
				}
				if v.DecodeAs != nil {
					c.ResponseFormat.Type = "json_schema"
					// OpenAI requires a name.
					c.ResponseFormat.JSONSchema.Name = "response"
					c.ResponseFormat.JSONSchema.Strict = true
					c.ResponseFormat.JSONSchema.Schema = jsonschema.Reflect(v.DecodeAs)
				}
				if len(v.Tools) != 0 {
					c.ParallelToolCalls = true
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
					}
				}
				if v.ThinkingBudget > 0 {
					if v.ThinkingBudget <= v.MaxTokens/3 {
						c.ReasoningEffort = ReasoningEffortLow
					} else if v.ThinkingBudget <= v.MaxTokens/2 {
						c.ReasoningEffort = ReasoningEffortMedium
					} else {
						c.ReasoningEffort = ReasoningEffortHigh
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
			// Starting with o1.
			c.Messages[0].Role = "developer"
			c.Messages[0].Content = Contents{{Type: "text", Text: sp}}
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

// ReasoningEffort is the effort the model should put into reasoning. Default is Medium.
//
// https://platform.openai.com/docs/api-reference/assistants/createAssistant#assistants-createassistant-reasoning_effort
// https://platform.openai.com/docs/guides/reasoning
type ReasoningEffort string

const (
	ReasoningEffortLow    ReasoningEffort = "low"
	ReasoningEffortMedium ReasoningEffort = "medium"
	ReasoningEffortHigh   ReasoningEffort = "high"
)

// https://platform.openai.com/docs/api-reference/chat/create
type Message struct {
	Role    string   `json:"role,omitzero"` // "developer", "assistant", "user"
	Name    string   `json:"name,omitzero"` // An optional name for the participant. Provides the model information to differentiate between participants of the same role.
	Content Contents `json:"content,omitzero"`
	Refusal string   `json:"refusal,omitzero"` // The refusal message by the assistant.
	Audio   struct {
		ID string `json:"id,omitzero"`
	} `json:"audio,omitzero"`
	ToolCalls   []ToolCall `json:"tool_calls,omitzero"`
	ToolCallID  string     `json:"tool_call_id,omitzero"` // TODO
	Annotations []struct{} `json:"annotations,omitzero"`
}

func (m *Message) From(in *genai.Message) error {
	switch in.Role {
	case genai.User, genai.Assistant:
		m.Role = string(in.Role)
	default:
		return fmt.Errorf("unsupported role %q", in.Role)
	}
	m.Name = in.User
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

func (m *Message) To(out *genai.Message) error {
	switch m.Role {
	case "assistant", "user":
		out.Role = genai.Role(m.Role)
	default:
		// case "developer":
		return fmt.Errorf("unsupported role %q", m.Role)
	}
	if len(m.Content) != 0 {
		out.Contents = make([]genai.Content, len(m.Content))
		for i := range m.Content {
			if err := m.Content[i].To(&out.Contents[i]); err != nil {
				return fmt.Errorf("block %d: %w", i, err)
			}
		}
	}
	if len(m.ToolCalls) != 0 {
		out.ToolCalls = make([]genai.ToolCall, len(m.ToolCalls))
		for i := range m.ToolCalls {
			m.ToolCalls[i].To(&out.ToolCalls[i])
		}
	}
	return nil
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
	Type string `json:"type,omitzero"` // "text", "image_url", "input_audio", "refusal", "audio", "file"

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

	// Type == "file"
	File struct {
		// Either FileID or both Filename and FileData.
		FileID   string `json:"file_id,omitzero"` // Use https://platform.openai.com/docs/api-reference/files
		Filename string `json:"filename,omitzero"`
		FileData string `json:"file_data,omitzero"`
	} `json:"file,omitzero"`
}

func (c *Content) From(in *genai.Content) error {
	if in.Text != "" {
		c.Type = "text"
		c.Text = in.Text
		return nil
	}
	// https://platform.openai.com/docs/guides/images?api-mode=chat&format=base64-encoded#image-input-requirements
	mimeType, data, err := in.ReadDocument(10 * 1024 * 1024)
	if err != nil {
		return err
	}
	// OpenAI require a mime-type to determine if image, sound or PDF.
	if mimeType == "" {
		return fmt.Errorf("unspecified mime type for URL %q", in.URL)
	}
	switch {
	case strings.HasPrefix(mimeType, "image/"):
		c.Type = "image_url"
		if in.URL == "" {
			c.ImageURL.URL = fmt.Sprintf("data:%s;base64,%s", mimeType, base64.StdEncoding.EncodeToString(data))
		} else {
			c.ImageURL.URL = in.URL
		}
	case mimeType == "audio/mpeg":
		if in.URL != "" {
			return errors.New("URL to audio file not supported")
		}
		c.Type = "input_audio"
		c.InputAudio.Data = data
		c.InputAudio.Format = "mp3"
	case mimeType == "audio/wav":
		if in.URL != "" {
			return errors.New("URL to audio file not supported")
		}
		c.Type = "input_audio"
		c.InputAudio.Data = data
		c.InputAudio.Format = "wav"
	default:
		if in.URL != "" {
			return fmt.Errorf("URL to %s file not supported", mimeType)
		}
		filename := in.GetFilename()
		if filename == "" {
			exts, err := mime.ExtensionsByType(mimeType)
			if err != nil {
				return err
			}
			if len(exts) == 0 {
				return fmt.Errorf("unknown extension for mime type %s", mimeType)
			}
			filename = "content" + exts[0]
		}
		c.Type = "file"
		c.File.Filename = filename
		c.File.FileData = fmt.Sprintf("data:%s;base64,%s", mimeType, base64.StdEncoding.EncodeToString(data))
	}
	return nil
}

func (c *Content) To(out *genai.Content) error {
	switch c.Type {
	case "text":
		out.Text = c.Text
	default:
		return fmt.Errorf("unsupported content type %q", c.Type)
	}
	return nil
}

type ToolCall struct {
	ID       string `json:"id"`
	Type     string `json:"type"` // function
	Function struct {
		Name      string `json:"name"`
		Arguments string `json:"arguments"` // Generally JSON
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

type Tool struct {
	Type     string `json:"type,omitzero"` // "function"
	Function struct {
		Description string             `json:"description,omitzero"`
		Name        string             `json:"name,omitzero"`
		Parameters  *jsonschema.Schema `json:"parameters,omitzero"`
		Strict      bool               `json:"strict,omitzero"`
	} `json:"function,omitzero"`
}

// ChatResponse is documented at
// https://platform.openai.com/docs/api-reference/chat/object
type ChatResponse struct {
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
		ChatTokens          int64 `json:"completion_tokens"`
		TotalTokens         int64 `json:"total_tokens"`
		PromptTokensDetails struct {
			CachedTokens int64 `json:"cached_tokens"`
			AudioTokens  int64 `json:"audio_tokens"`
			TextTokens   int64 `json:"text_tokens"`
			ImageTokens  int64 `json:"image_tokens"`
		} `json:"prompt_tokens_details"`
		ChatTokensDetails struct {
			ReasoningTokens          int64 `json:"reasoning_tokens"`
			AudioTokens              int64 `json:"audio_tokens"`
			AcceptedPredictionTokens int64 `json:"accepted_prediction_tokens"`
			RejectedPredictionTokens int64 `json:"rejected_prediction_tokens"`
			TextTokens               int64 `json:"text_tokens"`
		} `json:"completion_tokens_details"`
	} `json:"usage"`
	ServiceTier       string `json:"service_tier"`
	SystemFingerprint string `json:"system_fingerprint"`
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
	out.FinishReason = c.Choices[0].FinishReason
	err := c.Choices[0].Message.To(&out.Message)
	return out, err
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

// ChatStreamChunkResponse is not documented?
type ChatStreamChunkResponse struct {
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
		ChatTokens   int64 `json:"completion_tokens"`
		PromptTokens int64 `json:"prompt_tokens"`
		TotalTokens  int64 `json:"total_tokens"`
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

// Client implements the REST JSON based API.
type Client struct {
	// Client is exported for testing replay purposes.
	Client httpjson.Client

	model string
}

// TODO: Upload files
// https://platform.openai.com/docs/api-reference/uploads/create
// TTL 1h

// New creates a new client to talk to the OpenAI platform API.
//
// If apiKey is not provided, it tries to load it from the OPENAI_API_KEY environment variable.
// If none is found, it returns an error.
// Get your API key at https://platform.openai.com/settings/organization/api-keys
// If no model is provided, only functions that do not require a model, like ListModels, will work.
// To use multiple models, create multiple clients.
// Use one of the model from https://platform.openai.com/docs/models
func New(apiKey, model string) (*Client, error) {
	if apiKey == "" {
		if apiKey = os.Getenv("OPENAI_API_KEY"); apiKey == "" {
			return nil, errors.New("openai API key is required; get one at " + apiKeyURL)
		}
	}
	return &Client{
		model: model,
		Client: httpjson.Client{
			Client: &http.Client{Transport: &roundtrippers.Header{
				Transport: &roundtrippers.Retry{
					Transport: &roundtrippers.RequestID{
						Transport: http.DefaultTransport,
					},
				},
				Header: http.Header{"Authorization": {"Bearer " + apiKey}},
			}},
			Lenient: internal.BeLenient,
		},
	}, nil
}

func (c *Client) Chat(ctx context.Context, msgs genai.Messages, opts genai.Validatable) (genai.ChatResult, error) {
	rpcin := ChatRequest{}
	var continuableErr error
	if err := rpcin.Init(msgs, opts, c.model); err != nil {
		// If it's an UnsupportedContinuableError, we can continue
		if uce, ok := err.(*genai.UnsupportedContinuableError); ok {
			// Store the error to return later if no other error occurs
			continuableErr = uce
			// Otherwise log the error but continue
		} else {
			return genai.ChatResult{}, err
		}
	}
	rpcout := ChatResponse{}
	if err := c.ChatRaw(ctx, &rpcin, &rpcout); err != nil {
		return genai.ChatResult{}, fmt.Errorf("failed to get chat response: %w", err)
	}
	result, err := rpcout.ToResult()
	if err != nil {
		return result, err
	}
	// Return the continuable error if no other error occurred
	if continuableErr != nil {
		return result, continuableErr
	}
	return result, nil
}

func (c *Client) ChatRaw(ctx context.Context, in *ChatRequest, out *ChatResponse) error {
	// https://platform.openai.com/docs/api-reference/chat/create
	if err := c.validate(); err != nil {
		return err
	}
	in.Stream = false
	return c.post(ctx, "https://api.openai.com/v1/chat/completions", in, out)
}

func (c *Client) ChatStream(ctx context.Context, msgs genai.Messages, opts genai.Validatable, chunks chan<- genai.MessageFragment) error {
	in := ChatRequest{}
	var continuableErr error
	if err := in.Init(msgs, opts, c.model); err != nil {
		// If it's an UnsupportedContinuableError, we can continue
		if uce, ok := err.(*genai.UnsupportedContinuableError); ok {
			// Store the error to return later if no other error occurs
			continuableErr = uce
			// Otherwise log the error but continue
		} else {
			return err
		}
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
	// Return the continuable error if no other error occurred
	if err == nil && continuableErr != nil {
		return continuableErr
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
		case "", "assistant":
		default:
			return fmt.Errorf("unexpected role %q", role)
		}
		if word := pkt.Choices[0].Delta.Content; word != "" {
			fragment := genai.MessageFragment{TextFragment: word}
			// Only include FinishReason if it's not empty (only on the final fragment)
			if pkt.Choices[0].FinishReason != "" {
				fragment.FinishReason = pkt.Choices[0].FinishReason
			}
			chunks <- fragment
		}
	}
	return nil
}

func (c *Client) ChatStreamRaw(ctx context.Context, in *ChatRequest, out chan<- ChatStreamChunkResponse) error {
	if err := c.validate(); err != nil {
		return err
	}
	in.Stream = true
	resp, err := c.Client.PostRequest(ctx, "https://api.openai.com/v1/chat/completions", nil, in)
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
		return fmt.Errorf("unexpected line. expected \"data: \", got %q", line)
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
	if len(msg.Choices) != 1 {
		return fmt.Errorf("server returned an unexpected number of choices, expected 1, got %d", len(msg.Choices))
	}
	out <- msg
	return nil
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

func (c *Client) ListModels(ctx context.Context) ([]genai.Model, error) {
	// https://platform.openai.com/docs/api-reference/models/list
	var out struct {
		Object string  `json:"object"` // list
		Data   []Model `json:"data"`
	}
	if err := c.Client.Get(ctx, "https://api.openai.com/v1/models", nil, &out); err != nil {
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

var (
	_ genai.ChatProvider  = &Client{}
	_ genai.ModelProvider = &Client{}
)
