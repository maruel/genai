// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Package huggingface implements a client for the HuggingFace serverless
// inference API.
//
// It is described at https://huggingface.co/docs/api-inference/
package huggingface

import (
	"bufio"
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/invopop/jsonschema"
	"github.com/maruel/genai"
	"github.com/maruel/genai/internal"
	"github.com/maruel/httpjson"
	"github.com/maruel/roundtrippers"
	"golang.org/x/sync/errgroup"
)

// Official python SDK: https://github.com/huggingface/huggingface_hub
//
// But the real spec source of truth is
// https://github.com/huggingface/huggingface.js/tree/main/packages/tasks/src/tasks/chat-completion

// https://huggingface.co/docs/api-inference/tasks/chat-completion#api-specification
type ChatRequest struct {
	Model            string    `json:"model,omitempty"` // It's already in the URL.
	Stream           bool      `json:"stream"`
	Messages         []Message `json:"messages"`
	FrequencyPenalty float64   `json:"frequency_penalty,omitzero"` // [-2.0, 2.0]
	Logprobs         bool      `json:"logprobs,omitzero"`
	MaxTokens        int64     `json:"max_tokens,omitzero"`
	PresencePenalty  float64   `json:"presence_penalty,omitzero"` // [-2.0, 2.0]
	ResponseFormat   struct {
		Type string `json:"type"` // "json", "regex"
		// Type == "regexp": a regex string.
		// Type == "json": a JSONSchema.
		Value *jsonschema.Schema `json:"value"`
	} `json:"response_format,omitzero"`
	Seed          int64    `json:"seed,omitzero"`
	Stop          []string `json:"stop,omitzero"` // Up to 4
	StreamOptions struct {
		IncludeUsage bool `json:"include_usage,omitzero"`
	} `json:"stream_options,omitzero"`
	Temperature float64 `json:"temperature,omitzero"` // [0, 2.0]
	// Alternative when forcing a specific function. This can probably be achieved
	// by providing a single tool and ToolChoice == "required".
	// ToolChoice struct {
	// 	Type     string `json:"type,omitzero"` // "function"
	// 	Function struct {
	// 		Name string `json:"name,omitzero"`
	// 	} `json:"function,omitzero"`
	// } `json:"tool_choice,omitzero"`
	ToolChoice  string  `json:"tool_choice,omitzero"` // "auto", "none", "required"
	ToolPrompt  string  `json:"tool_prompt,omitzero"`
	Tools       []Tool  `json:"tools,omitzero"`
	TopLogprobs int64   `json:"top_logprobs,omitzero"`
	TopP        float64 `json:"top_p,omitzero"` // [0, 1]
	// logit_bias, n are documented as ignored.
}

// Init initializes the provider specific completion request with the generic completion request.
func (c *ChatRequest) Init(msgs genai.Messages, opts genai.Validatable, model string) error {
	var errs []error
	var unsupported []string
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
				c.Seed = v.Seed
				if v.TopK != 0 {
					unsupported = append(unsupported, "TopK")
				}
				c.Stop = v.Stop
				if v.ReplyAsJSON {
					c.ResponseFormat.Type = "json"
				}
				if v.DecodeAs != nil {
					c.ResponseFormat.Type = "json"
					c.ResponseFormat.Value = jsonschema.Reflect(v.DecodeAs)
				}
				if len(v.Tools) != 0 {
					switch v.ToolCallRequest {
					case genai.ToolCallAny:
						c.ToolChoice = "auto"
					case genai.ToolCallRequired:
						c.ToolChoice = "required"
					case genai.ToolCallNone:
						c.ToolChoice = "none"
					}
					c.Tools = make([]Tool, len(v.Tools))
					for i, t := range v.Tools {
						c.Tools[i].Type = "function"
						c.Tools[i].Function.Name = t.Name
						c.Tools[i].Function.Description = t.Description
						c.Tools[i].Function.Arguments = t.InputSchema()
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
			c.Messages[0].Content = []Content{{Type: ContentText, Text: sp}}
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

// Message is incorrectly documented at
// https://huggingface.co/docs/api-inference/tasks/chat-completion#api-specification
type Message struct {
	Role      string     `json:"role"` // "system", "assistant", "user", "tool"
	Content   Contents   `json:"content,omitzero"`
	ToolCalls []ToolCall `json:"tool_calls,omitzero"`
	Name      string     `json:"name,omitzero"` // Not really documented
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
		for j := range in.ToolCalls {
			m.ToolCalls[j].From(&in.ToolCalls[j])
		}
	}
	if len(in.ToolCallResults) != 0 {
		if len(in.Contents) != 0 || len(in.ToolCalls) != 0 {
			// This could be worked around.
			return fmt.Errorf("can't have tool call result along content or tool calls")
		}
		// Huggingface doesn't use tool ID in the result, hence only one tool can safely be called at a time.
		if len(in.ToolCallResults) != 1 {
			return fmt.Errorf("can't have more than one tool call result at a time")
		}
		m.Role = "tool"
		m.Content = Contents{{Type: ContentText, Text: in.ToolCallResults[0].Result}}
		m.Name = in.ToolCallResults[0].Name

	}
	return nil
}

type Content struct {
	Type     ContentType `json:"type"`
	Text     string      `json:"text,omitzero"`
	ImageURL struct {
		URL string `json:"url,omitzero"`
	} `json:"image_url,omitzero"`
}

func (c *Content) From(in *genai.Content) error {
	if in.Text != "" {
		c.Type = ContentText
		c.Text = in.Text
		return nil
	}
	mimeType, data, err := in.ReadDocument(10 * 1024 * 1024)
	if err != nil {
		return err
	}
	switch {
	case (in.URL != "" && mimeType == "") || strings.HasPrefix(mimeType, "image/"):
		c.Type = ContentImageURL
		if in.URL == "" {
			c.ImageURL.URL = fmt.Sprintf("data:%s;base64,%s", mimeType, base64.StdEncoding.EncodeToString(data))
		} else {
			c.ImageURL.URL = in.URL
		}
	default:
		return fmt.Errorf("unsupported mime type %s", mimeType)
	}
	return nil
}

// Contents represents a slice of Content with custom unmarshalling to handle
// both string and Content struct types.
type Contents []Content

func (c *Contents) MarshalJSON() ([]byte, error) {
	if len(*c) == 1 && (*c)[0].Type == ContentText {
		return json.Marshal((*c)[0].Text)
	}
	return json.Marshal(([]Content)(*c))
}

type ContentType string

const (
	ContentText     ContentType = "text"
	ContentImageURL ContentType = "image_url"
)

type ToolCall struct {
	Index    int64  `json:"index,omitzero"`
	ID       string `json:"id,omitzero"`
	Type     string `json:"type,omitzero"` // "function"
	Function struct {
		Name        string   `json:"name,omitzero"`
		Description struct{} `json:"description,omitzero"` // Passed in as null in response
		Arguments   string   `json:"arguments,omitzero"`
	} `json:"function,omitzero"`
}

func (t *ToolCall) From(in *genai.ToolCall) {
	t.ID = in.ID
	t.Type = "function"
	t.Function.Name = in.Name
	// The API seems to flip-flop between JSON and string.
	// return json.Unmarshal([]byte(in.Arguments), &t.Function.Arguments)
	t.Function.Arguments = in.Arguments
}

func (t *ToolCall) To(out *genai.ToolCall) {
	out.ID = t.ID
	out.Name = t.Function.Name
	// b, err := json.Marshal(t.Function.Arguments)
	// if err != nil {
	//	return fmt.Errorf("failed to marshal arguments: %w", err)
	// }
	// out.Arguments = string(b)
	out.Arguments = t.Function.Arguments
}

type Tool struct {
	Type     string `json:"type,omitzero"` // "function"
	Function struct {
		Name        string             `json:"name,omitzero"`
		Description string             `json:"description,omitzero"`
		Arguments   *jsonschema.Schema `json:"arguments,omitzero"`
	} `json:"function,omitzero"`
}

type ChatResponse struct {
	Object            string `json:"object"`
	ID                string `json:"id"`
	Created           Time   `json:"created"`
	Model             string `json:"model"`
	SystemFingerprint string `json:"system_fingerprint"`

	Choices []struct {
		FinishReason FinishReason    `json:"finish_reason"`
		Index        int64           `json:"index"`
		Message      MessageResponse `json:"message"`
		Logprobs     struct {
			Content []struct {
				Logprob     float64 `json:"logprob"`
				Token       string  `json:"token"`
				TopLogprobs []struct {
					Token   string  `json:"token"`
					Logprob float64 `json:"logprob"`
				} `json:"top_logprobs"`
			} `json:"content"`
		} `json:"logprobs"`
	} `json:"choices"`
	Usage Usage `json:"usage"`
}

type FinishReason string

const (
	FinishStop FinishReason = "stop"
)

func (f FinishReason) ToFinishReason() string {
	return string(f)
}

type Usage struct {
	PromptTokens     int64 `json:"prompt_tokens"`
	CompletionTokens int64 `json:"completion_tokens"`
	TotalTokens      int64 `json:"total_tokens"`
}

// The structure is different than the request. :(
type MessageResponse struct {
	Role       string     `json:"role"`
	Content    string     `json:"content"`
	ToolCallID string     `json:"tool_call_id"`
	ToolCalls  []ToolCall `json:"tool_calls"`
}

func (m *MessageResponse) To(out *genai.Message) error {
	switch role := m.Role; role {
	case "assistant", "user":
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

func (c *ChatResponse) ToResult() (genai.ChatResult, error) {
	out := genai.ChatResult{
		// At the moment, Huggingface doesn't support caching.
		Usage: genai.Usage{
			InputTokens:  c.Usage.PromptTokens,
			OutputTokens: c.Usage.CompletionTokens,
			FinishReason: c.Choices[0].FinishReason.ToFinishReason(),
		},
	}
	if len(c.Choices) != 1 {
		return out, fmt.Errorf("server returned an unexpected number of choices, expected 1, got %d", len(c.Choices))
	}
	err := c.Choices[0].Message.To(&out.Message)
	return out, err
}

type ChatStreamChunkResponse struct {
	Object            string `json:"object"`
	Created           Time   `json:"created"`
	ID                string `json:"id"`
	Model             string `json:"model"`
	SystemFingerprint string `json:"system_fingerprint"`
	Choices           []struct {
		Index        int64        `json:"index"`
		FinishReason FinishReason `json:"finish_reason"`
		Delta        struct {
			Role    genai.Role `json:"role"`
			Content string     `json:"content"`
			// ToolCallID string     `json:"tool_call_id"`
			ToolCalls []ToolCall `json:"tool_calls"`
		} `json:"delta"`
		Logprobs struct {
			Content []struct {
				Logprob     float64 `json:"logprob"`
				Token       string  `json:"token"`
				TopLogprobs []struct {
					Token   string  `json:"token"`
					Logprob float64 `json:"logprob"`
				} `json:"top_logprobs"`
			} `json:"content"`
		} `json:"logprobs"`
	} `json:"choices"`
	Usage Usage `json:"usage"`
}

type errorResponse struct {
	Error errorError `json:"error"`
}

func (er *errorResponse) String() string {
	if er.Error.HTTPStatusCode != 0 {
		return fmt.Sprintf("http %d: %s", er.Error.HTTPStatusCode, er.Error.Message)
	}
	return "error " + er.Error.Message
}

type errorError struct {
	Message        string `json:"message"`
	HTTPStatusCode int64  `json:"http_status_code"`
}

func (ee *errorError) UnmarshalJSON(d []byte) error {
	s := ""
	if err := json.Unmarshal(d, &s); err == nil {
		ee.Message = s
		return nil
	}
	var x struct {
		Message        string `json:"message"`
		HTTPStatusCode int64  `json:"http_status_code"`
	}
	if err := json.Unmarshal(d, &x); err != nil {
		return err
	}
	ee.Message = x.Message
	ee.HTTPStatusCode = x.HTTPStatusCode
	return nil
}

// Client implements the REST JSON based API.
type Client struct {
	internal.ClientBase

	model   string
	chatURL string
}

// TODO: Investigate https://huggingface.co/blog/inference-providers and https://huggingface.co/docs/inference-endpoints/

// New creates a new client to talk to the HuggingFace serverless inference API.
//
// If apiKey is not provided, it tries to load it from the HUGGINGFACE_API_KEY environment variable.
// Otherwise, it tries to load it from the huggingface python client's cache.
// If none is found, it returns an error.
// Get your API key at https://huggingface.co/settings/tokens
// If no model is provided, only functions that do not require a model, like ListModels, will work.
// To use multiple models, create multiple clients.
// Use one of the tens of thousands of models to chose from at https://huggingface.co/models?inference=warm&sort=trending
func New(apiKey, model string) (*Client, error) {
	if apiKey == "" {
		if apiKey = os.Getenv("HUGGINGFACE_API_KEY"); apiKey == "" {
			// Fallback to loading from the python client's cache.
			h, err := os.UserHomeDir()
			if err != nil {
				return nil, fmt.Errorf("can't find home directory; failed to load hugginface key; get one at %s: %w", apiKeyURL, err)
			}
			// TODO: Windows.
			b, err := os.ReadFile(filepath.Join(h, ".cache", "huggingface", "token"))
			if err != nil {
				return nil, fmt.Errorf("no cached token file; failed to load hugginface key; get one at %s: %w", apiKeyURL, err)
			}
			if apiKey = strings.TrimSpace(string(b)); apiKey == "" {
				return nil, errors.New("token file exist but is empty; huggingface API key is required; get one at " + apiKeyURL)
			}
		}
	}
	return &Client{
		model:   model,
		chatURL: "https://router.huggingface.co/hf-inference/models/" + model + "/v1/chat/completions",
		ClientBase: internal.ClientBase{
			ClientJSON: httpjson.Client{
				Client: &http.Client{Transport: &roundtrippers.Header{
					Header: http.Header{"Authorization": {"Bearer " + apiKey}},
					Transport: &roundtrippers.Retry{
						Transport: &roundtrippers.RequestID{
							Transport: http.DefaultTransport,
						},
					},
				}},
				Lenient: internal.BeLenient,
			},
		},
	}, nil
}

func (c *Client) Chat(ctx context.Context, msgs genai.Messages, opts genai.Validatable) (genai.ChatResult, error) {
	// https://huggingface.co/docs/api-inference/tasks/chat-completion#api-specification
	for i, msg := range msgs {
		for j, content := range msg.Contents {
			if len(content.Opaque) != 0 {
				return genai.ChatResult{}, fmt.Errorf("message #%d content #%d: field Opaque not supported", i, j)
			}
		}
	}
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
		return genai.ChatResult{}, err
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
	if err := c.validate(); err != nil {
		return err
	}
	in.Stream = false
	return c.doRequest(ctx, "POST", c.chatURL, in, out)
}

func (c *Client) ChatStream(ctx context.Context, msgs genai.Messages, opts genai.Validatable, chunks chan<- genai.MessageFragment) (genai.ChatResult, error) {
	result := genai.ChatResult{}
	// Check for non-empty Opaque field
	for i, msg := range msgs {
		for j, content := range msg.Contents {
			if len(content.Opaque) != 0 {
				return result, fmt.Errorf("message #%d content #%d: Opaque field not supported", i, j)
			}
		}
	}

	in := ChatRequest{}
	var continuableErr error
	if err := in.Init(msgs, opts, c.model); err != nil {
		// If it's an UnsupportedContinuableError, we can continue
		if uce, ok := err.(*genai.UnsupportedContinuableError); ok {
			// Store the error to return later if no other error occurs
			continuableErr = uce
			// Otherwise log the error but continue
		} else {
			return result, err
		}
	}
	ch := make(chan ChatStreamChunkResponse)
	eg, ctx := errgroup.WithContext(ctx)
	eg.Go(func() error {
		return processStreamPackets(ch, chunks, &result)
	})
	err := c.ChatStreamRaw(ctx, &in, ch)
	close(ch)
	if err2 := eg.Wait(); err2 != nil {
		err = err2
	}
	// Return the continuable error if no other error occurred
	if err == nil && continuableErr != nil {
		return result, continuableErr
	}
	return result, err
}

func processStreamPackets(ch <-chan ChatStreamChunkResponse, chunks chan<- genai.MessageFragment, result *genai.ChatResult) error {
	defer func() {
		// We need to empty the channel to avoid blocking the goroutine.
		for range ch {
		}
	}()
	pendingCall := ToolCall{}
	for pkt := range ch {
		if pkt.Usage.PromptTokens != 0 {
			result.InputTokens = pkt.Usage.PromptTokens
			result.OutputTokens = pkt.Usage.CompletionTokens
		}
		if len(pkt.Choices) != 1 {
			continue
		}
		if pkt.Choices[0].FinishReason != "" {
			result.FinishReason = pkt.Choices[0].FinishReason.ToFinishReason()
		}
		switch role := pkt.Choices[0].Delta.Role; role {
		case "assistant", "":
		default:
			return fmt.Errorf("unexpected role %q", role)
		}
		// There's only one at a time ever.
		if len(pkt.Choices[0].Delta.ToolCalls) > 1 {
			return fmt.Errorf("implement multiple tool calls: %#v", pkt.Choices[0].Delta.ToolCalls)
		}
		f := genai.MessageFragment{TextFragment: pkt.Choices[0].Delta.Content}
		// Huggingface streams the arguments. Buffer the arguments to send the fragment as a whole tool call.
		if len(pkt.Choices[0].Delta.ToolCalls) == 1 {
			if t := pkt.Choices[0].Delta.ToolCalls[0]; t.ID == pendingCall.ID {
				// Continuation.
				pendingCall.Function.Arguments += t.Function.Arguments
				if !f.IsZero() {
					return fmt.Errorf("implement tool call with metadata: %#v", pkt)
				}
				continue
			} else {
				// A new call.
				if pendingCall.ID == "" {
					pendingCall = t
					if !f.IsZero() {
						return fmt.Errorf("implement tool call with metadata: %#v", pkt)
					}
					continue
				}
				// Flush.
				pendingCall.To(&f.ToolCall)
				pendingCall = t
			}
		} else if pendingCall.ID != "" {
			// Flush.
			pendingCall.To(&f.ToolCall)
			pendingCall = ToolCall{}
		}
		if !f.IsZero() {
			if err := result.Accumulate(f); err != nil {
				return err
			}
			chunks <- f
		}
	}
	// Hugginface doesn't send an "ending" packet, FinishReason isn't even set on the last packet.
	if pendingCall.ID != "" {
		// Flush.
		f := genai.MessageFragment{}
		pendingCall.To(&f.ToolCall)
		if err := result.Accumulate(f); err != nil {
			return err
		}
		chunks <- f
	}
	return nil
}

func (c *Client) ChatStreamRaw(ctx context.Context, in *ChatRequest, out chan<- ChatStreamChunkResponse) error {
	if err := c.validate(); err != nil {
		return err
	}
	in.Stream = true
	in.StreamOptions.IncludeUsage = true
	resp, err := c.ClientJSON.Request(ctx, "POST", c.chatURL, nil, in)
	if err != nil {
		return fmt.Errorf("failed to get server response: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		return internal.DecodeError(ctx, c.chatURL, resp, &errorResponse{}, apiKeyURL)
	}
	return processSSE(resp.Body, out)
}

func processSSE(body io.Reader, out chan<- ChatStreamChunkResponse) error {
	dataPrefix := []byte("data: ")
	eventPrefix := []byte("event:")
	done := []byte("[DONE]")
	r := bufio.NewReader(body)
	for first := true; ; first = false {
		line, err := r.ReadBytes('\n')
		if line = bytes.TrimSpace(line); err == io.EOF {
			if len(line) == 0 {
				return nil
			}
		} else if err != nil {
			return fmt.Errorf("failed to get server response: %w", err)
		}
		if len(line) == 0 {
			continue
		}
		// HuggingFace has the bad habit of returning errors as HTML pages.
		if first && bytes.HasPrefix(line, []byte("<!DOCTYPE html>")) {
			// Often has a 503 in there as a <div>.
			rest, _ := io.ReadAll(r)
			return fmt.Errorf("unexpected error: %s\n%s", line, rest)
		}
		switch {
		case bytes.HasPrefix(line, dataPrefix):
			suffix := line[len(dataPrefix):]
			if bytes.Equal(suffix, done) {
				return nil
			}
			d := json.NewDecoder(bytes.NewReader(suffix))
			d.DisallowUnknownFields()
			d.UseNumber()
			msg := ChatStreamChunkResponse{}
			if err := d.Decode(&msg); err != nil {
				// Errors are returned as data: JSON line.
				d = json.NewDecoder(bytes.NewReader(suffix))
				d.DisallowUnknownFields()
				d.UseNumber()
				er := errorResponse{}
				if err := d.Decode(&er); err != nil {
					return fmt.Errorf("failed to decode server response %q: %w", string(line), err)
				}
				return errors.New(er.String())
			}
			out <- msg
		case bytes.HasPrefix(line, eventPrefix):
			// Ignore.
		default:
			return fmt.Errorf("unexpected line. expected \"data: \", got %q", line)
		}
	}
}

type Time int64

func (t *Time) AsTime() time.Time {
	return time.Unix(int64(*t), 0)
}

type Model struct {
	ID            string    `json:"id"`
	ID2           string    `json:"_id"`
	Likes         int64     `json:"likes"`
	TrendingScore float64   `json:"trendingScore"`
	Private       bool      `json:"private"`
	Downloads     int64     `json:"downloads"`
	Tags          []string  `json:"tags"` // Tags can be a single word or key:value, like base_model, doi, license, region, arxiv.
	PipelineTag   string    `json:"pipeline_tag"`
	LibraryName   string    `json:"library_name"`
	CreatedAt     time.Time `json:"createdAt"`
	ModelId       string    `json:"modelId"`

	// When full=true is specified:
	Author       string    `json:"author"`
	Gated        bool      `json:"gated"`
	LastModified time.Time `json:"lastModified"`
	SHA          string    `json:"sha"`
	Siblings     []struct {
		RFilename string `json:"r_filename"`
	} `json:"siblings"`
}

func (m *Model) GetID() string {
	return m.ID
}

func (m *Model) String() string {
	return fmt.Sprintf("%s (%s) %s Trending: %.1f", m.ID, m.CreatedAt.Format("2006-01-02"), m.PipelineTag, m.TrendingScore)
}

func (m *Model) Context() int64 {
	return 0
}

func (c *Client) ListModels(ctx context.Context) ([]genai.Model, error) {
	// https://huggingface.co/docs/hub/api

	// return nil, errors.New("not implemented; there's just too many, tens of thousands to chose from at https://huggingface.co/models?inference=warm&sort=trending")
	var out []Model
	// There's 20k models warm as of March 2025. There's no way to sort by
	// trending. Sorting by download is not useful.
	if err := c.doRequest(ctx, "GET", "https://huggingface.co/api/models?inference=warm", nil, &out); err != nil {
		return nil, err
	}
	models := make([]genai.Model, len(out))
	for i := range out {
		models[i] = &out[i]
	}
	return models, nil
}

func (c *Client) validate() error {
	if c.model == "" {
		return errors.New("a model is required")
	}
	return nil
}

func (c *Client) doRequest(ctx context.Context, method, url string, in, out any) error {
	return c.DoRequest(ctx, method, url, in, out, &errorResponse{}, apiKeyURL)
}

const apiKeyURL = "https://huggingface.co/settings/tokens"

var (
	_ genai.ChatProvider  = &Client{}
	_ genai.ModelProvider = &Client{}
)
