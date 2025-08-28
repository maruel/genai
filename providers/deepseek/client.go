// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Package deepseek implements a client for the DeepSeek API.
//
// It is described at https://api-docs.deepseek.com/
package deepseek

import (
	"bytes"
	"context"
	_ "embed"
	"encoding/json"
	"errors"
	"fmt"
	"iter"
	"net/http"
	"os"
	"slices"
	"strings"

	"github.com/invopop/jsonschema"
	"github.com/maruel/genai"
	"github.com/maruel/genai/base"
	"github.com/maruel/genai/internal"
	"github.com/maruel/genai/scoreboard"
	"github.com/maruel/roundtrippers"
)

//go:embed scoreboard.json
var scoreboardJSON []byte

// Scoreboard for DeepSeek.
func Scoreboard() scoreboard.Score {
	var s scoreboard.Score
	d := json.NewDecoder(bytes.NewReader(scoreboardJSON))
	d.DisallowUnknownFields()
	if err := d.Decode(&s); err != nil {
		panic(fmt.Errorf("failed to unmarshal scoreboard.json: %w", err))
	}
	return s
}

// ChatRequest is documented at https://api-docs.deepseek.com/api/create-chat-completion
type ChatRequest struct {
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

// Init initializes the provider specific completion request with the generic completion request.
func (c *ChatRequest) Init(msgs genai.Messages, model string, opts ...genai.Options) error {
	c.Model = model
	if err := msgs.Validate(); err != nil {
		return err
	}
	var errs []error
	var unsupported []string
	sp := ""
	for _, opt := range opts {
		if err := opt.Validate(); err != nil {
			return err
		}
		// https://api-docs.deepseek.com/guides/reasoning_model Soon "reasoning_effort"
		switch v := opt.(type) {
		case *genai.OptionsText:
			c.MaxToks = v.MaxTokens
			c.Temperature = v.Temperature
			c.TopP = v.TopP
			sp = v.SystemPrompt
			if v.TopLogprobs > 0 {
				c.TopLogprob = v.TopLogprobs
				c.Logprobs = true
			}
			if v.Seed != 0 {
				unsupported = append(unsupported, "Seed")
			}
			if v.TopK != 0 {
				unsupported = append(unsupported, "TopK")
			}
			c.Stop = v.Stop
			if v.ReplyAsJSON {
				c.ResponseFormat.Type = "json_object"
			}
			if v.DecodeAs != nil {
				errs = append(errs, errors.New("unsupported option DecodeAs"))
			}
			if len(v.Tools) != 0 {
				switch v.ToolCallRequest {
				case genai.ToolCallAny:
					c.ToolChoice = "auto"
				case genai.ToolCallRequired:
					if strings.Contains(model, "reasoner") {
						// "deepseek-reasoner does not support this tool_choice"
						unsupported = append(unsupported, "ToolCallRequired")
					} else {
						c.ToolChoice = "required"
					}
				case genai.ToolCallNone:
					c.ToolChoice = "none"
				}
				c.Tools = make([]Tool, len(v.Tools))
				for i, t := range v.Tools {
					c.Tools[i].Type = "function"
					c.Tools[i].Function.Name = t.Name
					c.Tools[i].Function.Description = t.Description
					if c.Tools[i].Function.Parameters = t.InputSchemaOverride; c.Tools[i].Function.Parameters == nil {
						c.Tools[i].Function.Parameters = t.GetInputSchema()
					}
				}
			}
		default:
			errs = append(errs, fmt.Errorf("unsupported options type %T", opt))
		}
	}

	if sp != "" {
		c.Messages = append(c.Messages, Message{Role: "system", Content: sp})
	}
	for i := range msgs {
		// Split messages into multiple messages as needed.
		if len(msgs[i].ToolCallResults) > 1 {
			// Handle messages with multiple tool call results by creating multiple messages
			for j := range msgs[i].ToolCallResults {
				// Create a copy of the message with only one tool call result
				msgCopy := msgs[i]
				msgCopy.ToolCallResults = []genai.ToolCallResult{msgs[i].ToolCallResults[j]}
				var newMsg Message
				if err := newMsg.From(&msgCopy); err != nil {
					errs = append(errs, fmt.Errorf("message #%d, tool call results #%d: %w", i, j, err))
				} else {
					c.Messages = append(c.Messages, newMsg)
				}
			}
		} else if len(msgs[i].Requests) > 1 {
			// Handle messages with multiple Request by creating multiple messages
			for j := range msgs[i].Requests {
				// Create a copy of the message with only one request
				msgCopy := msgs[i]
				msgCopy.Requests = []genai.Request{msgs[i].Requests[j]}
				var newMsg Message
				if err := newMsg.From(&msgCopy); err != nil {
					errs = append(errs, fmt.Errorf("message #%d, request #%d: %w", i, j, err))
				} else {
					c.Messages = append(c.Messages, newMsg)
				}
			}
		} else {
			var newMsg Message
			if err := newMsg.From(&msgs[i]); err != nil {
				errs = append(errs, fmt.Errorf("message #%d: %w", i, err))
			} else {
				c.Messages = append(c.Messages, newMsg)
			}
		}
	}
	// If we have unsupported features but no other errors, return a continuable error
	if len(unsupported) > 0 && len(errs) == 0 {
		return &genai.UnsupportedContinuableError{Unsupported: unsupported}
	}
	return errors.Join(errs...)
}

func (c *ChatRequest) SetStream(stream bool) {
	c.Stream = stream
}

// Message is documented at https://api-docs.deepseek.com/api/create-chat-completion
type Message struct {
	Role             string     `json:"role,omitzero"` // "system", "assistant", "user"
	Name             string     `json:"name,omitzero"` // An optional name for the participant. Provides the model information to differentiate between participants of the same role.
	Content          string     `json:"content,omitzero"`
	Prefix           bool       `json:"prefix,omitzero"` // Force the model to start its answer by the content of the supplied prefix in this assistant message.
	ReasoningContent string     `json:"reasoning_content,omitzero"`
	ToolCalls        []ToolCall `json:"tool_calls,omitzero"`
	ToolCallID       string     `json:"tool_call_id,omitzero"` // Tool call that this message is responding to, with response in Content field.
}

// From must be called with at most one Request or one ToolCallResults.
func (m *Message) From(in *genai.Message) error {
	if len(in.Requests) > 1 || len(in.ToolCallResults) > 1 {
		return errors.New("internal error")
	}
	switch r := in.Role(); r {
	case "user", "assistant":
		m.Role = r
	case "computer":
		m.Role = "tool"
	default:
		return fmt.Errorf("unsupported role %q", r)
	}
	m.Name = in.User
	if len(in.Requests) == 1 {
		if in.Requests[0].Text != "" {
			m.Content += in.Requests[0].Text
		} else if !in.Requests[0].Doc.IsZero() {
			if in.Requests[0].Doc.URL != "" {
				return errors.New("deepseek doesn't support document content blocks with URLs")
			}
			mimeType, data, err := in.Requests[0].Doc.Read(10 * 1024 * 1024)
			if err != nil {
				return fmt.Errorf("failed to read document: %w", err)
			}
			if !strings.HasPrefix(mimeType, "text/plain") {
				return fmt.Errorf("deepseek only supports text/plain documents, got %s", mimeType)
			}
			m.Content += string(data)
		} else {
			return errors.New("unknown Request type")
		}
		return nil
	}
	for i := range in.Replies {
		if len(in.Replies[i].Opaque) != 0 {
			return fmt.Errorf("reply #%d: field Reply.Opaque not supported", i)
		}
		if in.Replies[i].Text != "" {
			m.Content += in.Replies[i].Text
		} else if !in.Replies[i].Doc.IsZero() {
			mimeType, data, err := in.Replies[i].Doc.Read(10 * 1024 * 1024)
			if err != nil {
				return fmt.Errorf("reply #%d: failed to read document: %w", i, err)
			}
			if in.Replies[i].Doc.URL != "" {
				return fmt.Errorf("reply #%d: deepseek doesn't support document content blocks with URLs", i)
			}
			if !strings.HasPrefix(mimeType, "text/plain") {
				return fmt.Errorf("reply #%d: deepseek only supports text/plain documents, got %s", 0, mimeType)
			}
			m.Content += string(data)
		} else if in.Replies[i].Thinking != "" {
			// Thinking content should not be returned to the model.
		} else if !in.Replies[i].ToolCall.IsZero() {
			m.ToolCalls = append(m.ToolCalls, ToolCall{})
			if err := m.ToolCalls[len(m.ToolCalls)-1].From(&in.Replies[i].ToolCall); err != nil {
				return fmt.Errorf("reply #%d: %w", i, err)
			}
		} else {
			return errors.New("unknown Reply type")
		}
	}
	if len(in.ToolCallResults) != 0 {
		// Process only the first tool call result in this method.
		// The Init method handles multiple tool call results by creating multiple messages.
		m.Content = in.ToolCallResults[0].Result
		m.ToolCallID = in.ToolCallResults[0].ID
	}
	return nil
}

func (m *Message) To(out *genai.Message) error {
	// Both ReasoningContent and Content can be set on the same reply.
	if m.ReasoningContent != "" {
		out.Replies = append(out.Replies, genai.Reply{Thinking: m.ReasoningContent})
	}
	if m.Content != "" {
		out.Replies = append(out.Replies, genai.Reply{Text: m.Content})
	}
	for i := range m.ToolCalls {
		out.Replies = append(out.Replies, genai.Reply{})
		m.ToolCalls[i].To(&out.Replies[len(out.Replies)-1].ToolCall)
	}
	return nil
}

type ToolCall struct {
	Index    int64  `json:"index,omitzero"`
	ID       string `json:"id,omitzero"`
	Type     string `json:"type,omitzero"` // "function"
	Function struct {
		Name      string `json:"name,omitzero"`
		Arguments string `json:"arguments,omitzero"`
	} `json:"function,omitzero"`
}

func (t *ToolCall) From(in *genai.ToolCall) error {
	if len(in.Opaque) != 0 {
		return errors.New("field ToolCall.Opaque not supported")
	}
	t.Type = "function"
	t.ID = in.ID
	t.Function.Name = in.Name
	t.Function.Arguments = in.Arguments
	return nil
}

func (t *ToolCall) To(out *genai.ToolCall) {
	out.ID = t.ID
	out.Name = t.Function.Name
	out.Arguments = t.Function.Arguments
}

type Tool struct {
	Type     string `json:"type"` // "function"
	Function struct {
		Name        string             `json:"name,omitzero"`
		Description string             `json:"description,omitzero"`
		Parameters  *jsonschema.Schema `json:"parameters,omitzero"`
	} `json:"function"`
}

type ChatResponse struct {
	ID      string `json:"id"`
	Choices []struct {
		FinishReason FinishReason `json:"finish_reason"`
		Index        int64        `json:"index"`
		Message      Message      `json:"message"`
		Logprobs     Logprobs     `json:"logprobs"`
	} `json:"choices"`
	Created           int64  `json:"created"` // Unix timestamp
	Model             string `json:"model"`
	SystemFingerPrint string `json:"system_fingerprint"`
	Object            string `json:"object"` // chat.completion
	Usage             Usage  `json:"usage"`
}

func (c *ChatResponse) ToResult() (genai.Result, error) {
	out := genai.Result{
		Usage: genai.Usage{
			InputTokens:       c.Usage.PromptTokens,
			InputCachedTokens: c.Usage.PromptCacheHitTokens,
			ReasoningTokens:   c.Usage.ChatTokensDetails.ReasoningTokens,
			OutputTokens:      c.Usage.CompletionTokens,
		},
	}
	if len(c.Choices) != 1 {
		return out, fmt.Errorf("expected 1 choice, got %#v", c.Choices)
	}
	out.Usage.FinishReason = c.Choices[0].FinishReason.ToFinishReason()
	err := c.Choices[0].Message.To(&out.Message)
	out.Logprobs = c.Choices[0].Logprobs.To()
	return out, err
}

type FinishReason string

const (
	FinishStop          FinishReason = "stop"
	FinishToolCalls     FinishReason = "tool_calls"
	FinishLength        FinishReason = "length"
	FinishContentFilter FinishReason = "content_filter"
	FinishInsufficient  FinishReason = "insufficient_system_resource"
)

func (f FinishReason) ToFinishReason() genai.FinishReason {
	switch f {
	case FinishStop:
		return genai.FinishedStop
	case FinishToolCalls:
		return genai.FinishedToolCalls
	case FinishLength:
		return genai.FinishedLength
	case FinishContentFilter:
		return genai.FinishedContentFilter
	case FinishInsufficient:
		fallthrough
	default:
		if !internal.BeLenient {
			panic(f)
		}
		return genai.FinishReason(f)
	}
}

type Usage struct {
	CompletionTokens      int64 `json:"completion_tokens"`
	PromptTokens          int64 `json:"prompt_tokens"`
	PromptCacheHitTokens  int64 `json:"prompt_cache_hit_tokens"`
	PromptCacheMissTokens int64 `json:"prompt_cache_miss_tokens"`
	TotalTokens           int64 `json:"total_tokens"`
	PromptTokensDetails   struct {
		CachedTokens int64 `json:"cached_tokens"`
	} `json:"prompt_tokens_details"`
	ChatTokensDetails struct {
		ReasoningTokens int64 `json:"reasoning_tokens"`
	} `json:"completion_tokens_details"`
}

type Logprobs struct {
	Content []struct {
		Token       string  `json:"token"`
		Bytes       []byte  `json:"bytes"`
		Logprob     float64 `json:"logprob"`
		TopLogprobs []struct {
			Token   string  `json:"token"`
			Bytes   []byte  `json:"bytes"`
			Logprob float64 `json:"logprob"`
		} `json:"top_logprobs"`
	} `json:"content"`
}

func (l *Logprobs) To() []genai.Logprobs {
	if len(l.Content) == 0 {
		return nil
	}
	out := make([]genai.Logprobs, 0, len(l.Content))
	for i, c := range l.Content {
		out = append(out, genai.Logprobs{Text: c.Token, Bytes: c.Bytes, Logprob: c.Logprob, TopLogprobs: make([]genai.TopLogprob, 0, len(c.TopLogprobs))})
		for _, tlp := range c.TopLogprobs {
			out[i].TopLogprobs = append(out[i].TopLogprobs, genai.TopLogprob{Text: tlp.Token, Bytes: tlp.Bytes, Logprob: tlp.Logprob})
		}
	}
	return out
}

type ChatStreamChunkResponse struct {
	ID                string `json:"id"`
	Object            string `json:"object"`  // chat.completion.chunk
	Created           int64  `json:"created"` // Unix timestamp
	Model             string `json:"model"`
	SystemFingerprint string `json:"system_fingerprint"`
	Choices           []struct {
		Index        int64        `json:"index"`
		Delta        Message      `json:"delta"`
		Logprobs     Logprobs     `json:"logprobs"`
		FinishReason FinishReason `json:"finish_reason"`
	} `json:"choices"`
	Usage Usage `json:"usage"`
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

// ModelsResponse represents the response structure for DeepSeek models listing
type ModelsResponse struct {
	Object string  `json:"object"` // list
	Data   []Model `json:"data"`
}

// ToModels converts DeepSeek models to genai.Model interfaces
func (r *ModelsResponse) ToModels() []genai.Model {
	models := make([]genai.Model, len(r.Data))
	for i := range r.Data {
		models[i] = &r.Data[i]
	}
	return models
}

//

type ErrorResponse struct {
	// Type  string `json:"type"`
	ErrorVal struct {
		Message string `json:"message"`
		Type    string `json:"type"`
		Param   string `json:"param"`
		Code    string `json:"code"`
	} `json:"error"`
}

func (er *ErrorResponse) Error() string {
	return fmt.Sprintf("%s: %s", er.ErrorVal.Type, er.ErrorVal.Message)
}

func (er *ErrorResponse) IsAPIError() bool {
	return true
}

// Client implements genai.Provider.
type Client struct {
	impl base.Provider[*ErrorResponse, *ChatRequest, *ChatResponse, ChatStreamChunkResponse]
}

// New creates a new client to talk to the DeepSeek platform API in China.
//
// If opts.APIKey is not provided, it tries to load it from the DEEPSEEK_API_KEY environment variable.
// If none is found, it will still return a client coupled with an base.ErrAPIKeyRequired error.
// Get your API key at https://platform.deepseek.com/api_keys
//
// To use multiple models, create multiple clients.
// Use one of the model from https://api-docs.deepseek.com/quick_start/pricing
//
// wrapper optionally wraps the HTTP transport. Useful for HTTP recording and playback, or to tweak HTTP
// retries, or to throttle outgoing requests.
func New(ctx context.Context, opts *genai.ProviderOptions, wrapper func(http.RoundTripper) http.RoundTripper) (*Client, error) {
	if opts.AccountID != "" {
		return nil, errors.New("unexpected option AccountID")
	}
	if opts.Remote != "" {
		return nil, errors.New("unexpected option Remote")
	}
	apiKey := opts.APIKey
	const apiKeyURL = "https://platform.deepseek.com/api_keys"
	var err error
	if apiKey == "" {
		if apiKey = os.Getenv("DEEPSEEK_API_KEY"); apiKey == "" {
			err = &base.ErrAPIKeyRequired{EnvVar: "DEEPSEEK_API_KEY", URL: apiKeyURL}
		}
	}
	mod := genai.Modalities{genai.ModalityText}
	if len(opts.OutputModalities) != 0 && !slices.Equal(opts.OutputModalities, mod) {
		return nil, fmt.Errorf("unexpected option Modalities %s, only text is supported", mod)
	}
	t := base.DefaultTransport
	if wrapper != nil {
		t = wrapper(t)
	}
	c := &Client{
		impl: base.Provider[*ErrorResponse, *ChatRequest, *ChatResponse, ChatStreamChunkResponse]{
			GenSyncURL:           "https://api.deepseek.com/chat/completions",
			ProcessStreamPackets: processStreamPackets,
			PreloadedModels:      opts.PreloadedModels,
			ProviderBase: base.ProviderBase[*ErrorResponse]{
				APIKeyURL: apiKeyURL,
				Lenient:   internal.BeLenient,
				Client: http.Client{
					Transport: &roundtrippers.Header{
						Header:    http.Header{"Authorization": {"Bearer " + apiKey}},
						Transport: &roundtrippers.RequestID{Transport: t},
					},
				},
			},
		},
	}
	if err == nil {
		switch opts.Model {
		case genai.ModelNone:
		case genai.ModelCheap, genai.ModelGood, genai.ModelSOTA, "":
			if c.impl.Model, err = c.selectBestTextModel(ctx, opts.Model); err != nil {
				return nil, err
			}
			c.impl.OutputModalities = mod
		default:
			c.impl.Model = opts.Model
			c.impl.OutputModalities = mod
		}
	}
	return c, err
}

// selectBestTextModel selects the most appropriate model based on the preference (cheap, good, or SOTA).
//
// We may want to make this function overridable in the future by the client since this is going to break one
// day or another.
func (c *Client) selectBestTextModel(ctx context.Context, preference string) (string, error) {
	mdls, err := c.ListModels(ctx)
	if err != nil {
		return "", fmt.Errorf("failed to automatically select the model: %w", err)
	}
	cheap := preference == genai.ModelCheap
	selectedModel := ""
	for _, mdl := range mdls {
		m := mdl.(*Model)
		if cheap {
			if strings.Contains(m.ID, "chat") {
				selectedModel = m.ID
			}
		} else {
			if !strings.Contains(m.ID, "chat") {
				selectedModel = m.ID
			}
		}
	}
	if selectedModel == "" {
		return "", errors.New("failed to find a model automatically")
	}
	return selectedModel, nil
}

// Name implements genai.Provider.
//
// It returns the name of the provider.
func (c *Client) Name() string {
	return "deepseek"
}

// ModelID implements genai.Provider.
//
// It returns the selected model ID.
func (c *Client) ModelID() string {
	return c.impl.Model
}

// OutputModalities implements genai.Provider.
//
// It returns the output modalities, i.e. what kind of output the model will generate (text, audio, image,
// video, etc).
func (c *Client) OutputModalities() genai.Modalities {
	return c.impl.OutputModalities
}

// Scoreboard implements genai.Provider.
func (c *Client) Scoreboard() scoreboard.Score {
	return Scoreboard()
}

// HTTPClient returns the HTTP client to fetch results (e.g. videos) generated by the provider.
func (c *Client) HTTPClient() *http.Client {
	return &c.impl.Client
}

// GenSync implements genai.Provider.
func (c *Client) GenSync(ctx context.Context, msgs genai.Messages, opts ...genai.Options) (genai.Result, error) {
	return c.impl.GenSync(ctx, msgs, opts...)
}

// GenSyncRaw provides access to the raw API.
func (c *Client) GenSyncRaw(ctx context.Context, in *ChatRequest, out *ChatResponse) error {
	return c.impl.GenSyncRaw(ctx, in, out)
}

// GenStream implements genai.Provider.
func (c *Client) GenStream(ctx context.Context, msgs genai.Messages, opts ...genai.Options) (iter.Seq[genai.ReplyFragment], func() (genai.Result, error)) {
	return c.impl.GenStream(ctx, msgs, opts...)
}

// GenStreamRaw provides access to the raw API.
func (c *Client) GenStreamRaw(ctx context.Context, in *ChatRequest, out chan<- ChatStreamChunkResponse) error {
	return c.impl.GenStreamRaw(ctx, in, out)
}

// ListModels implements genai.Provider.
func (c *Client) ListModels(ctx context.Context) ([]genai.Model, error) {
	if c.impl.PreloadedModels != nil {
		return c.impl.PreloadedModels, nil
	}
	// https://api-docs.deepseek.com/api/list-models
	var resp ModelsResponse
	if err := c.impl.DoRequest(ctx, "GET", "https://api.deepseek.com/models", nil, &resp); err != nil {
		return nil, err
	}
	return resp.ToModels(), nil
}

// TODO: Caching: https://api-docs.deepseek.com/guides/kv_cache

func processStreamPackets(ch <-chan ChatStreamChunkResponse, chunks chan<- genai.ReplyFragment, result *genai.Result) error {
	defer func() {
		// We need to empty the channel to avoid blocking the goroutine.
		for range ch {
		}
	}()
	pendingCall := ToolCall{}
	for pkt := range ch {
		if len(pkt.Choices) != 1 {
			continue
		}
		if pkt.Usage.CompletionTokens != 0 {
			result.Usage.InputTokens = pkt.Usage.PromptTokens
			result.Usage.InputCachedTokens = pkt.Usage.PromptCacheHitTokens
			result.Usage.ReasoningTokens = pkt.Usage.ChatTokensDetails.ReasoningTokens
			result.Usage.OutputTokens = pkt.Usage.CompletionTokens
			result.Usage.FinishReason = pkt.Choices[0].FinishReason.ToFinishReason()
		}
		if len(pkt.Choices[0].Delta.ToolCalls) > 1 {
			return fmt.Errorf("implement multiple tool calls: %#v", pkt)
		}
		switch role := pkt.Choices[0].Delta.Role; role {
		case "assistant", "":
		default:
			return fmt.Errorf("unexpected role %q", role)
		}
		f := genai.ReplyFragment{
			TextFragment:     pkt.Choices[0].Delta.Content,
			ThinkingFragment: pkt.Choices[0].Delta.ReasoningContent,
		}
		// DeepSeek streams the arguments. Buffer the arguments to send the fragment as a whole tool call.
		if len(pkt.Choices[0].Delta.ToolCalls) == 1 {
			if t := pkt.Choices[0].Delta.ToolCalls[0]; t.ID != "" {
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
			} else if pendingCall.ID != "" {
				// Continuation.
				pendingCall.Function.Arguments += t.Function.Arguments
				if !f.IsZero() {
					return fmt.Errorf("implement tool call with metadata: %#v", pkt)
				}
				continue
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
		if len(pkt.Choices[0].Logprobs.Content) != 0 {
			result.Logprobs = append(result.Logprobs, pkt.Choices[0].Logprobs.To()...)
		}
	}
	return nil
}

var _ genai.Provider = &Client{}
