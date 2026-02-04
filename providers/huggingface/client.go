// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Package huggingface implements a client for the HuggingFace serverless
// inference API.
//
// It is described at https://huggingface.co/docs/api-inference/
package huggingface

import (
	"bytes"
	"context"
	_ "embed"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"iter"
	"net/http"
	"os"
	"path/filepath"
	"reflect"
	"regexp"
	"slices"
	"strconv"
	"strings"
	"time"

	"github.com/invopop/jsonschema"
	"github.com/maruel/genai"
	"github.com/maruel/genai/base"
	"github.com/maruel/genai/internal"
	"github.com/maruel/genai/scoreboard"
	"github.com/maruel/roundtrippers"
)

// Official python SDK: https://github.com/huggingface/huggingface_hub
//
// But the real spec source of truth is
// https://github.com/huggingface/huggingface.js/tree/main/packages/tasks/src/tasks/chat-completion

//go:embed scoreboard.json
var scoreboardJSON []byte

// Scoreboard for Huggingface.
func Scoreboard() scoreboard.Score {
	var s scoreboard.Score
	d := json.NewDecoder(bytes.NewReader(scoreboardJSON))
	d.DisallowUnknownFields()
	if err := d.Decode(&s); err != nil {
		panic(fmt.Errorf("failed to unmarshal scoreboard.json: %w", err))
	}
	return s
}

// ChatRequest is underspecified at
// https://huggingface.co/docs/api-inference/tasks/chat-completion#api-specification
type ChatRequest struct {
	Model            string    `json:"model"`
	Stream           bool      `json:"stream"`
	Messages         []Message `json:"messages"`
	FrequencyPenalty float64   `json:"frequency_penalty,omitzero"` // [-2.0, 2.0]
	Logprobs         bool      `json:"logprobs,omitzero"`
	MaxTokens        int64     `json:"max_tokens,omitzero"`
	PresencePenalty  float64   `json:"presence_penalty,omitzero"` // [-2.0, 2.0]
	ResponseFormat   struct {
		Type       string             `json:"type,omitzero"` // "text", "json_object" or "json_schema".
		JSONSchema *jsonschema.Schema `json:"json_schema,omitzero"`
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
func (c *ChatRequest) Init(msgs genai.Messages, model string, opts ...genai.GenOptions) error {
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
		switch v := opt.(type) {
		case *genai.GenOptionsText:
			c.MaxTokens = v.MaxTokens
			c.Temperature = v.Temperature
			c.TopP = v.TopP
			sp = v.SystemPrompt
			if v.TopLogprobs > 0 {
				c.TopLogprobs = v.TopLogprobs
				c.Logprobs = true
			}
			if v.TopK != 0 {
				unsupported = append(unsupported, "GenOptionsText.TopK")
			}
			c.Stop = v.Stop
			if v.DecodeAs != nil {
				c.ResponseFormat.Type = "json_schema"
				c.ResponseFormat.JSONSchema = internal.JSONSchemaFor(reflect.TypeOf(v.DecodeAs))
				// Huggingface complains otherwise.
				c.ResponseFormat.JSONSchema.Extras = map[string]any{"name": "response"}
			} else if v.ReplyAsJSON {
				c.ResponseFormat.Type = "json_object"
			}
		case *genai.GenOptionsTools:
			if len(v.Tools) != 0 {
				switch v.Force {
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
					if c.Tools[i].Function.Arguments = t.InputSchemaOverride; c.Tools[i].Function.Arguments == nil {
						c.Tools[i].Function.Arguments = t.GetInputSchema()
					}
				}
			}
			if v.WebSearch {
				errs = append(errs, errors.New("unsupported OptionsTools.WebSearch"))
			}
		case genai.GenOptionsSeed:
			c.Seed = int64(v)
		default:
			unsupported = append(unsupported, internal.TypeName(opt))
		}
	}

	if sp != "" {
		c.Messages = append(c.Messages, Message{Role: "system", Content: []Content{{Type: ContentText, Text: sp}}})
	}
	for i := range msgs {
		if len(msgs[i].ToolCallResults) > 1 {
			// Handle messages with multiple tool call results by creating multiple messages
			for j := range msgs[i].ToolCallResults {
				// Create a copy of the message with only one tool call result
				msgCopy := msgs[i]
				msgCopy.ToolCallResults = []genai.ToolCallResult{msgs[i].ToolCallResults[j]}
				var newMsg Message
				if err := newMsg.From(&msgCopy); err != nil {
					errs = append(errs, fmt.Errorf("message #%d: tool call results #%d: %w", i, j, err))
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
	// If we have unsupported features but no other errors, return a structured error.
	if len(unsupported) > 0 && len(errs) == 0 {
		return &base.ErrNotSupported{Options: unsupported}
	}
	return errors.Join(errs...)
}

func (c *ChatRequest) SetStream(stream bool) {
	c.Stream = stream
	c.StreamOptions.IncludeUsage = stream
}

// Message is incorrectly documented at
// https://huggingface.co/docs/api-inference/tasks/chat-completion#api-specification
type Message struct {
	Role      string     `json:"role"` // "system", "assistant", "user", "tool"
	Content   Contents   `json:"content,omitzero"`
	ToolCalls []ToolCall `json:"tool_calls,omitzero"`
	Name      string     `json:"name,omitzero"` // Not really documented
}

// From must be called with at most one ToolCallResults.
func (m *Message) From(in *genai.Message) error {
	if len(in.ToolCallResults) > 1 {
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
	if len(in.Requests) != 0 {
		m.Content = make([]Content, len(in.Requests))
		for i := range in.Requests {
			if err := m.Content[i].FromRequest(&in.Requests[i]); err != nil {
				return fmt.Errorf("request #%d: %w", i, err)
			}
		}
	}
	if len(in.Replies) != 0 {
		m.Content = make([]Content, len(in.Replies))
		for i := range in.Replies {
			if !in.Replies[i].ToolCall.IsZero() {
				m.ToolCalls = append(m.ToolCalls, ToolCall{})
				if err := m.ToolCalls[len(m.ToolCalls)-1].From(&in.Replies[i].ToolCall); err != nil {
					return fmt.Errorf("reply #%d: %w", i, err)
				}
				continue
			}
			if err := m.Content[i].FromReply(&in.Replies[i]); err != nil {
				return fmt.Errorf("reply #%d: %w", i, err)
			}
		}
	}
	if len(in.ToolCallResults) != 0 {
		// Huggingface doesn't use tool ID in the result, hence only one tool can safely be called at a time.
		// Process only the first tool call result in this method.
		// The Init method handles multiple tool call results by creating multiple messages.
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

func (c *Content) FromRequest(in *genai.Request) error {
	if in.Text != "" {
		c.Type = ContentText
		c.Text = in.Text
		return nil
	}
	if !in.Doc.IsZero() {
		mimeType, data, err := in.Doc.Read(10 * 1024 * 1024)
		if err != nil {
			return err
		}
		switch {
		case (in.Doc.URL != "" && mimeType == "") || strings.HasPrefix(mimeType, "image/"):
			c.Type = ContentImageURL
			if in.Doc.URL == "" {
				c.ImageURL.URL = fmt.Sprintf("data:%s;base64,%s", mimeType, base64.StdEncoding.EncodeToString(data))
			} else {
				c.ImageURL.URL = in.Doc.URL
			}
			// text/plain, text/markdown
		case strings.HasPrefix(mimeType, "text/"):
			c.Type = ContentText
			if in.Doc.URL != "" {
				return fmt.Errorf("%s documents must be provided inline, not as a URL", mimeType)
			}
			c.Text = string(data)
		default:
			return fmt.Errorf("unsupported mime type %s", mimeType)
		}
		return nil
	}
	return errors.New("unknown Request type")
}

func (c *Content) FromReply(in *genai.Reply) error {
	if len(in.Opaque) != 0 {
		return &internal.BadError{Err: errors.New("field Reply.Opaque not supported")}
	}
	if in.Text != "" {
		c.Type = ContentText
		c.Text = in.Text
		return nil
	}
	if !in.Doc.IsZero() {
		mimeType, data, err := in.Doc.Read(10 * 1024 * 1024)
		if err != nil {
			return err
		}
		switch {
		case (in.Doc.URL != "" && mimeType == "") || strings.HasPrefix(mimeType, "image/"):
			c.Type = ContentImageURL
			if in.Doc.URL == "" {
				c.ImageURL.URL = fmt.Sprintf("data:%s;base64,%s", mimeType, base64.StdEncoding.EncodeToString(data))
			} else {
				c.ImageURL.URL = in.Doc.URL
			}
			// text/plain, text/markdown
		case strings.HasPrefix(mimeType, "text/"):
			c.Type = ContentText
			if in.Doc.URL != "" {
				return fmt.Errorf("%s documents must be provided inline, not as a URL", mimeType)
			}
			c.Text = string(data)
		default:
			return &internal.BadError{Err: fmt.Errorf("unsupported mime type %s", mimeType)}
		}
		return nil
	}
	return &internal.BadError{Err: errors.New("unknown Reply type")}
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
	Object            string    `json:"object"` // "chat.completion"
	ID                string    `json:"id"`
	Created           base.Time `json:"created"`
	Model             string    `json:"model"`
	SystemFingerprint string    `json:"system_fingerprint"`
	Prompt            []any     `json:"prompt,omitzero"` // Some models return empty prompt field

	Choices []struct {
		FinishReason         FinishReason         `json:"finish_reason"`
		Index                int64                `json:"index"`
		Message              MessageResponse      `json:"message"`
		ContentFilterResults ContentFilterResults `json:"content_filter_results"`
		StopReason           string               `json:"stop_reason"`
		Logprobs             Logprobs             `json:"logprobs,omitzero"`
		Seed                 int64                `json:"seed,omitzero"`
	} `json:"choices"`
	Usage          Usage    `json:"usage"`
	PromptLogprobs struct{} `json:"prompt_logprobs"`
	ServiceTier    struct{} `json:"service_tier"`
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
	// Alternative format used by some models
	Tokens         []any    `json:"tokens,omitzero"`
	TokenLogprobs  []any    `json:"token_logprobs,omitzero"`
	TopLogprobsAlt []any    `json:"top_logprobs,omitzero"`
	Refusal        struct{} `json:"refusal,omitzero"`
}

func (l *Logprobs) IsZero() bool {
	return len(l.Content) == 0
}

func (l *Logprobs) To() [][]genai.Logprob {
	out := make([][]genai.Logprob, 0, len(l.Content))
	for _, p := range l.Content {
		lp := make([]genai.Logprob, 0, len(p.TopLogprobs))
		// The first token is the same as the content token.
		// Intentionally discard Bytes.
		for _, tlp := range p.TopLogprobs {
			lp = append(lp, genai.Logprob{Text: tlp.Token, Logprob: tlp.Logprob})
		}
		out = append(out, lp)
	}
	return out
}

type FinishReason string

const (
	FinishStop         FinishReason = "stop"
	FinishLength       FinishReason = "length"
	FinishStopSequence FinishReason = "stop_sequence"
	FinishToolCalls    FinishReason = "tool_calls"
)

func (f FinishReason) ToFinishReason() genai.FinishReason {
	switch f {
	case FinishStop:
		return genai.FinishedStop
	case FinishLength:
		return genai.FinishedLength
	case FinishStopSequence:
		return genai.FinishedStopSequence
	case FinishToolCalls:
		return genai.FinishedToolCalls
	default:
		if !internal.BeLenient {
			panic(f)
		}
		return genai.FinishReason(f)
	}
}

type Usage struct {
	PromptTokens        int64 `json:"prompt_tokens"`
	CompletionTokens    int64 `json:"completion_tokens"`
	TotalTokens         int64 `json:"total_tokens"`
	PromptTokensDetails struct {
		AudioTokens              int64 `json:"audio_tokens"`
		CachedTokens             int64 `json:"cached_tokens"`
		CacheCreationInputTokens int64 `json:"cache_creation_input_tokens"`
		CacheReadInputTokens     int64 `json:"cache_read_input_tokens"`
	} `json:"prompt_tokens_details"`
	CompletionTokensDetails struct {
		AudioTokens              int64 `json:"audio_tokens"`
		ReasoningTokens          int64 `json:"reasoning_tokens"`
		AcceptedPredictionTokens int64 `json:"accepted_prediction_tokens"`
		RejectedPredictionTokens int64 `json:"rejected_prediction_tokens"`
	} `json:"completion_tokens_details"`
}

type ContentFilterResults struct {
	Hate struct {
		Filtered bool `json:"filtered"`
	} `json:"hate"`
	SelfHarm struct {
		Filtered bool `json:"filtered"`
	} `json:"self_harm"`
	Sexual struct {
		Filtered bool `json:"filtered"`
	} `json:"sexual"`
	Violence struct {
		Filtered bool `json:"filtered"`
	} `json:"violence"`
	Jailbreak struct {
		Filtered bool `json:"filtered"`
		Detected bool `json:"detected"`
	} `json:"jailbreak"`
	Profanity struct {
		Filtered bool `json:"filtered"`
		Detected bool `json:"detected"`
	} `json:"profanity"`
}

// MessageResponse uses a different structure than the request Message. :(
type MessageResponse struct {
	Role             string     `json:"role"`
	Content          string     `json:"content"`
	ToolCallID       string     `json:"tool_call_id"`
	ToolCalls        []ToolCall `json:"tool_calls"`
	Refusal          struct{}   `json:"refusal"`
	FunctionCall     struct{}   `json:"function_call"`
	ReasoningContent struct{}   `json:"reasoning_content"`
	Annotations      struct{}   `json:"annotations"`
	Audio            struct{}   `json:"audio"`
	Reasoning        any        `json:"reasoning,omitzero"`
}

func (m *MessageResponse) To(out *genai.Message) error {
	if m.Content != "" {
		out.Replies = []genai.Reply{{Text: m.Content}}
	}
	for i := range m.ToolCalls {
		out.Replies = []genai.Reply{{}}
		m.ToolCalls[i].To(&out.Replies[len(out.Replies)-1].ToolCall)
	}
	return nil
}

func (c *ChatResponse) ToResult() (genai.Result, error) {
	out := genai.Result{
		// At the moment, Huggingface doesn't support caching.
		Usage: genai.Usage{
			InputTokens:       c.Usage.PromptTokens,
			InputCachedTokens: c.Usage.PromptTokensDetails.CachedTokens,
			ReasoningTokens:   c.Usage.CompletionTokensDetails.ReasoningTokens,
			OutputTokens:      c.Usage.CompletionTokens,
			TotalTokens:       c.Usage.TotalTokens,
		},
	}
	if len(c.Choices) != 1 {
		return out, fmt.Errorf("server returned an unexpected number of choices, expected 1, got %d", len(c.Choices))
	}
	out.Usage.FinishReason = c.Choices[0].FinishReason.ToFinishReason()
	err := c.Choices[0].Message.To(&out.Message)
	if out.Usage.FinishReason == genai.FinishedStop && slices.ContainsFunc(out.Replies, func(r genai.Reply) bool { return !r.ToolCall.IsZero() }) {
		// Lie for the benefit of everyone.
		out.Usage.FinishReason = genai.FinishedToolCalls
	}
	out.Logprobs = c.Choices[0].Logprobs.To()
	return out, err
}

type ChatStreamChunkResponse struct {
	Object            string    `json:"object"` // "chat.completion.chunk"
	Created           base.Time `json:"created"`
	ID                string    `json:"id"`
	Model             string    `json:"model"`
	SystemFingerprint string    `json:"system_fingerprint"`
	Choices           []struct {
		Index        int64        `json:"index,omitzero"`
		FinishReason FinishReason `json:"finish_reason"`
		Text         string       `json:"text,omitzero"`
		Logprobs     Logprobs     `json:"logprobs,omitzero"`
		Delta        struct {
			Role      string     `json:"role"`
			Content   string     `json:"content"`
			ToolCalls []ToolCall `json:"tool_calls"`
			TokenID   int64      `json:"token_id,omitzero"`
			Reasoning any        `json:"reasoning,omitzero"`
		} `json:"delta"`
		ContentFilterResults ContentFilterResults `json:"content_filter_results,omitzero"`
		StopReason           string               `json:"stop_reason,omitzero"`
	} `json:"choices"`
	Usage      Usage `json:"usage"`
	SLAMetrics struct {
		TSUs   int64 `json:"ts_us"`
		TTFTMs int64 `json:"ttft_ms"`
	} `json:"sla_metrics,omitzero"`
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
	ModelID       string    `json:"modelId"`

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

// ModelsResponse represents the response structure for Huggingface models listing
type ModelsResponse []Model

// ToModels converts Huggingface models to genai.Model interfaces
func (r *ModelsResponse) ToModels() []genai.Model {
	models := make([]genai.Model, len(*r))
	for i := range *r {
		models[i] = &(*r)[i]
	}
	return models
}

//

type ErrorResponse struct {
	ErrorVal  ErrorError `json:"error"`
	ErrorType string     `json:"error_type"`
	Detail    string     `json:"detail"`
	Code      int64      `json:"code"`
	Reason    string     `json:"reason"`
	Message   string     `json:"message"`
	Metadata  struct{}   `json:"metadata,omitzero"`
	Type      string     `json:"type"`        // "server_error"
	ID        string     `json:"id,omitzero"` // Some models include request ID in error response
}

func (er *ErrorResponse) Error() string {
	if er.Detail != "" {
		return er.Detail
	}
	if er.ErrorVal.HTTPStatusCode != 0 {
		// This is a relayed http failure from the router.
		return fmt.Sprintf("http %d: %s", er.ErrorVal.HTTPStatusCode, er.ErrorVal.Message)
	}
	if er.ErrorVal.Message != "" {
		if er.ErrorVal.Type != "" {
			return fmt.Sprintf("%s (%s): %s: %s", er.ErrorVal.Type, er.ErrorVal.Code, er.ErrorVal.Param, er.ErrorVal.Message)
		}
		return er.ErrorVal.Message
	}
	return fmt.Sprintf("http %d (%s): %s", er.Code, er.Reason, er.Message)
}

func (er *ErrorResponse) IsAPIError() bool {
	return true
}

type ErrorError struct {
	Message        string `json:"message"`
	HTTPStatusCode int64  `json:"http_status_code"`
	Type           string `json:"type"`
	Param          string `json:"param"`
	Code           string `json:"code"`
}

func (ee *ErrorError) UnmarshalJSON(b []byte) error {
	s := ""
	if err := json.Unmarshal(b, &s); err == nil {
		ee.Message = s
		return nil
	}
	type Alias ErrorError
	a := struct{ *Alias }{Alias: (*Alias)(ee)}
	d := json.NewDecoder(bytes.NewReader(b))
	if !internal.BeLenient {
		d.DisallowUnknownFields()
	}
	if err := d.Decode(&a); err != nil {
		return err
	}
	return nil
}

// Client implements genai.Provider.
type Client struct {
	base.NotImplemented
	impl base.Provider[*ErrorResponse, *ChatRequest, *ChatResponse, ChatStreamChunkResponse]
}

// TODO: Investigate https://huggingface.co/blog/inference-providers and https://huggingface.co/docs/inference-endpoints/

// New creates a new client to talk to the HuggingFace serverless inference API.
//
// If ProviderOptionAPIKey is not provided, it tries to load it from the HUGGINGFACE_API_KEY environment variable.
// Otherwise, it tries to load it from the huggingface python client's cache.
// If none is found, it will still return a client coupled with an base.ErrAPIKeyRequired error.
// Get your API key at https://huggingface.co/settings/tokens
//
// To use multiple models, create multiple clients.
// Use one of the tens of thousands of models to chose from at https://huggingface.co/models?inference=warm&sort=trending
//
// ProviderOptionTransportWrapper can be used to add the HTTP header "X-HF-Bill-To" via roundtrippers.Header. See
// https://huggingface.co/docs/inference-providers/pricing#organization-billing
func New(ctx context.Context, opts ...genai.ProviderOption) (*Client, error) {
	var apiKey, model string
	var modalities genai.Modalities
	var preloadedModels []genai.Model
	var wrapper func(http.RoundTripper) http.RoundTripper
	for _, opt := range opts {
		if err := opt.Validate(); err != nil {
			return nil, err
		}
		switch v := opt.(type) {
		case genai.ProviderOptionAPIKey:
			apiKey = string(v)
		case genai.ProviderOptionModel:
			model = string(v)
		case genai.ProviderOptionModalities:
			modalities = genai.Modalities(v)
		case genai.ProviderOptionPreloadedModels:
			preloadedModels = []genai.Model(v)
		case genai.ProviderOptionTransportWrapper:
			wrapper = v
		default:
			return nil, fmt.Errorf("unsupported option type %T", opt)
		}
	}
	const apiKeyURL = "https://huggingface.co/settings/tokens"
	var err error
	if apiKey == "" {
		if apiKey = os.Getenv("HUGGINGFACE_API_KEY"); apiKey == "" {
			// Fallback to loading from the python client's cache.
			h, errHome := os.UserHomeDir()
			if errHome != nil {
				err = &base.ErrAPIKeyRequired{EnvVar: "HUGGINGFACE_API_KEY", URL: apiKeyURL}
			} else {
				// TODO: Windows.
				b, errRead := os.ReadFile(filepath.Join(h, ".cache", "huggingface", "token"))
				if errRead != nil {
					err = &base.ErrAPIKeyRequired{EnvVar: "HUGGINGFACE_API_KEY", URL: apiKeyURL}
				} else {
					if apiKey = strings.TrimSpace(string(b)); apiKey == "" {
						err = &base.ErrAPIKeyRequired{EnvVar: "HUGGINGFACE_API_KEY", URL: apiKeyURL}
					}
				}
			}
		}
	}
	mod := genai.Modalities{genai.ModalityText}
	if len(modalities) != 0 && !slices.Equal(modalities, mod) {
		// https://huggingface.co/docs/inference-providers/index
		return nil, fmt.Errorf("unexpected option Modalities %s, only text is implemented (send PR to add support)", mod)
	}
	t := base.DefaultTransport
	if wrapper != nil {
		t = wrapper(t)
	}
	c := &Client{
		impl: base.Provider[*ErrorResponse, *ChatRequest, *ChatResponse, ChatStreamChunkResponse]{
			GenSyncURL:      "https://router.huggingface.co/v1/chat/completions",
			ProcessStream:   ProcessStream,
			PreloadedModels: preloadedModels,
			ProcessHeaders:  processHeaders,
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
		switch model {
		case "":
		case genai.ModelCheap, genai.ModelGood, genai.ModelSOTA:
			if c.impl.Model, err = c.selectBestTextModel(ctx, model); err != nil {
				return nil, err
			}
			c.impl.OutputModalities = mod
		default:
			c.impl.Model = model
			c.impl.OutputModalities = mod
		}
	}
	return c, err
}

// selectBestTextModel selects the most recent model based on the preference (cheap, good, or SOTA).
//
// We may want to make this function overridable in the future by the client since this is going to break one
// day or another.
func (c *Client) selectBestTextModel(ctx context.Context, preference string) (string, error) {
	// Warning: listing models from Huggingface takes a while.
	mdls, err := c.ListModels(ctx)
	if err != nil {
		return "", fmt.Errorf("failed to automatically select the model: %w", err)
	}
	cheap := preference == genai.ModelCheap
	good := preference == genai.ModelGood || preference == ""
	selectedModel := ""
	trending := 0.
	weights := 0
	re := regexp.MustCompile(`(\d+)B`)
	for _, mdl := range mdls {
		m := mdl.(*Model)
		if m.TrendingScore < 2 {
			continue
		}
		// TODO: This algorithm would gain to be improved.
		if cheap {
			// HF doesn't report the number of weights in the model. Try to guess it.
			matches := re.FindAllStringSubmatch(m.ID, 1)
			if len(matches) != 1 {
				continue
			}
			w, err2 := strconv.Atoi(matches[0][1])
			if err2 != nil {
				continue
			}
			if strings.HasPrefix(m.ID, "meta-llama/Llama") && strings.HasSuffix(m.ID, "-Instruct") && (weights == 0 || w < weights) {
				weights = w
				selectedModel = m.ID
			}
		} else if good {
			// HF doesn't report the number of weights in the model. Try to guess it.
			matches := re.FindAllStringSubmatch(m.ID, 1)
			if len(matches) != 1 {
				continue
			}
			w, err2 := strconv.Atoi(matches[0][1])
			if err2 != nil {
				continue
			}
			if strings.HasPrefix(m.ID, "Qwen/Qwen") && (weights == 0 || w > weights) {
				weights = w
				selectedModel = m.ID
			}
		} else {
			if strings.HasPrefix(m.ID, "deepseek-ai/") && !strings.Contains(m.ID, "Qwen") && !strings.Contains(m.ID, "Prover") && !strings.Contains(m.ID, "Distill") && (trending == 0 || trending < m.TrendingScore) {
				// Make it a popularity contest.
				trending = m.TrendingScore
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
	return "huggingface"
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
func (c *Client) GenSync(ctx context.Context, msgs genai.Messages, opts ...genai.GenOptions) (genai.Result, error) {
	return c.impl.GenSync(ctx, msgs, opts...)
}

// GenSyncRaw provides access to the raw API.
func (c *Client) GenSyncRaw(ctx context.Context, in *ChatRequest, out *ChatResponse) error {
	return c.impl.GenSyncRaw(ctx, in, out)
}

// GenStream implements genai.Provider.
func (c *Client) GenStream(ctx context.Context, msgs genai.Messages, opts ...genai.GenOptions) (iter.Seq[genai.Reply], func() (genai.Result, error)) {
	return c.impl.GenStream(ctx, msgs, opts...)
}

// GenStreamRaw provides access to the raw API.
func (c *Client) GenStreamRaw(ctx context.Context, in *ChatRequest) (iter.Seq[ChatStreamChunkResponse], func() error) {
	return c.impl.GenStreamRaw(ctx, in)
}

// ListModels implements genai.Provider.
func (c *Client) ListModels(ctx context.Context) ([]genai.Model, error) {
	if c.impl.PreloadedModels != nil {
		return c.impl.PreloadedModels, nil
	}
	// https://huggingface.co/docs/hub/api
	// There's 20k models warm as of March 2025. There's no way to sort by
	// trending. Sorting by download is not useful. There's no pagination.
	var resp ModelsResponse
	if err := c.impl.DoRequest(ctx, "GET", "https://huggingface.co/api/models?inference=warm", nil, &resp); err != nil {
		return nil, err
	}
	return resp.ToModels(), nil
}

// ProcessStream converts the raw packets from the streaming API into Reply fragments.
func ProcessStream(chunks iter.Seq[ChatStreamChunkResponse]) (iter.Seq[genai.Reply], func() (genai.Usage, [][]genai.Logprob, error)) {
	var finalErr error
	u := genai.Usage{}
	var l [][]genai.Logprob

	return func(yield func(genai.Reply) bool) {
			pendingToolCall := ToolCall{}
			for pkt := range chunks {
				if pkt.Usage.PromptTokens != 0 {
					u.InputTokens = pkt.Usage.PromptTokens
					u.InputCachedTokens = pkt.Usage.PromptTokensDetails.CachedTokens
					u.ReasoningTokens = pkt.Usage.CompletionTokensDetails.ReasoningTokens
					u.OutputTokens = pkt.Usage.CompletionTokens
					u.TotalTokens = pkt.Usage.TotalTokens
				}
				if len(pkt.Choices) != 1 {
					continue
				}
				if pkt.Choices[0].FinishReason != "" {
					u.FinishReason = pkt.Choices[0].FinishReason.ToFinishReason()
				}
				if !pkt.Choices[0].Logprobs.IsZero() {
					l = append(l, pkt.Choices[0].Logprobs.To()...)
				}
				switch role := pkt.Choices[0].Delta.Role; role {
				case "assistant", "":
				default:
					finalErr = &internal.BadError{Err: fmt.Errorf("unexpected role %q", role)}
					return
				}
				// There's only one at a time ever.
				if len(pkt.Choices[0].Delta.ToolCalls) > 1 {
					finalErr = &internal.BadError{Err: fmt.Errorf("implement multiple tool calls: %#v", pkt.Choices[0].Delta.ToolCalls)}
					return
				}
				f := genai.Reply{Text: pkt.Choices[0].Delta.Content}
				// Huggingface streams the arguments. Buffer the arguments to send the fragment as a whole tool call.
				if len(pkt.Choices[0].Delta.ToolCalls) == 1 {
					// ID is not consistently set. Use Name for now but that's risky.
					if t := pkt.Choices[0].Delta.ToolCalls[0]; t.Function.Name == pendingToolCall.Function.Name {
						// Continuation.
						pendingToolCall.Function.Arguments += t.Function.Arguments
						if !f.IsZero() {
							finalErr = &internal.BadError{Err: fmt.Errorf("implement tool call with metadata: %#v", pkt)}
							return
						}
						continue
					} else {
						// A new call.
						if pendingToolCall.Function.Name == "" {
							pendingToolCall = t
							if !f.IsZero() {
								finalErr = &internal.BadError{Err: fmt.Errorf("implement tool call with metadata: %#v", pkt)}
								return
							}
							continue
						}
						// Flush.
						pendingToolCall.To(&f.ToolCall)
						pendingToolCall = ToolCall{}
					}
				} else if pendingToolCall.Function.Name != "" {
					// Flush.
					pendingToolCall.To(&f.ToolCall)
					pendingToolCall = ToolCall{}
				}
				if !yield(f) {
					return
				}
			}
			// Hugginface doesn't send an "ending" packet, FinishReason isn't even set on the last packet.
			if pendingToolCall.Function.Name != "" {
				// Flush.
				f := genai.Reply{}
				pendingToolCall.To(&f.ToolCall)
				if !yield(f) {
					return
				}
			}
		}, func() (genai.Usage, [][]genai.Logprob, error) {
			return u, l, finalErr
		}
}

func processHeaders(h http.Header) []genai.RateLimit {
	var limits []genai.RateLimit
	requestsLimit, _ := strconv.ParseInt(h.Get("X-Ratelimit-Limit-Requests"), 10, 64)
	requestsRemaining, _ := strconv.ParseInt(h.Get("X-Ratelimit-Remaining-Requests"), 10, 64)
	tokensLimit, _ := strconv.ParseInt(h.Get("X-Ratelimit-Limit-Tokens"), 10, 64)
	tokensRemaining, _ := strconv.ParseInt(h.Get("X-Ratelimit-Remaining-Tokens"), 10, 64)
	reset, _ := time.ParseDuration(h.Get("X-Ratelimit-Dynamic-Period-Remaining"))

	if requestsLimit > 0 {
		limits = append(limits, genai.RateLimit{
			Type:      genai.Requests,
			Period:    genai.PerOther,
			Limit:     requestsLimit,
			Remaining: requestsRemaining,
			Reset:     time.Now().Add(reset).Round(10 * time.Millisecond),
		})
	}
	if tokensLimit > 0 {
		limits = append(limits, genai.RateLimit{
			Type:      genai.Tokens,
			Period:    genai.PerOther,
			Limit:     tokensLimit,
			Remaining: tokensRemaining,
			Reset:     time.Now().Add(reset).Round(10 * time.Millisecond),
		})
	}
	return limits
}

var _ genai.Provider = &Client{}
