// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Wire types for the Cloudflare Workers AI REST API.
//
// Documentation: https://developers.cloudflare.com/api/resources/ai/methods/run/

package cloudflare

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"reflect"
	"slices"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/maruel/genai"
	"github.com/maruel/genai/base"
	"github.com/maruel/genai/internal"
)

// ChatRequest structure depends on the model used.
//
// The general description is at https://developers.cloudflare.com/api/resources/ai/methods/run/
//
// A specific one is https://developers.cloudflare.com/workers-ai/models/llama-4-scout-17b-16e-instruct/
type ChatRequest struct {
	Messages          []Message `json:"messages"`
	FrequencyPenalty  float64   `json:"frequency_penalty,omitzero"` // [0, 2.0]
	MaxTokens         int64     `json:"max_tokens,omitzero"`
	PresencePenalty   float64   `json:"presence_penalty,omitzero"`   // [0, 2.0]
	RepetitionPenalty float64   `json:"repetition_penalty,omitzero"` // [0, 2.0]
	ResponseFormat    struct {
		Type       string           `json:"type,omitzero"` // json_object, json_schema
		JSONSchema genai.JSONSchema `json:"json_schema,omitzero"`
	} `json:"response_format,omitzero"`
	GuidedJSON  genai.JSONSchema `json:"guided_json,omitzero"`
	Seed        int64            `json:"seed,omitzero"`
	Stream      bool             `json:"stream,omitzero"`
	Temperature float64          `json:"temperature,omitzero"` // [0, 5]
	Tools       []Tool           `json:"tools,omitzero"`
	TopK        int64            `json:"top_k,omitzero"` // [1, 50]
	TopP        float64          `json:"top_p,omitzero"` // [0, 2.0]
}

// Init initializes the provider specific completion request with the generic completion request.
func (c *ChatRequest) Init(msgs genai.Messages, model string, opts ...genai.GenOption) error {
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
		case *genai.GenOptionText:
			c.MaxTokens = v.MaxTokens
			c.Temperature = v.Temperature
			c.TopP = v.TopP
			sp = v.SystemPrompt
			c.TopK = v.TopK
			if v.TopLogprobs > 0 {
				unsupported = append(unsupported, "GenOptionText.TopLogprobs")
			}
			if len(v.Stop) != 0 {
				errs = append(errs, errors.New("unsupported option Stop"))
			}
			if v.DecodeAs != nil {
				c.ResponseFormat.Type = "json_schema"
				s, err := genai.JSONSchemaFor(reflect.TypeOf(v.DecodeAs))
				if err != nil {
					errs = append(errs, err)
				} else {
					c.ResponseFormat.JSONSchema = s
				}
			} else if v.ReplyAsJSON {
				c.ResponseFormat.Type = "json_object"
			}
		case *genai.GenOptionTools:
			if len(v.Tools) != 0 {
				if v.Force != genai.ToolCallAny {
					// Cloudflare doesn't provide a way to force tool use. Don't fail.
					unsupported = append(unsupported, "GenOptionTools.Force")
				}
				c.Tools = make([]Tool, len(v.Tools))
				for i, t := range v.Tools {
					c.Tools[i].Type = "function"
					c.Tools[i].Function.Name = t.Name
					c.Tools[i].Function.Description = t.Description
					s, err := t.GetInputSchema()
					if err != nil {
						errs = append(errs, err)
					}
					c.Tools[i].Function.Parameters = s
				}
			}
		case genai.GenOptionSeed:
			c.Seed = int64(v)
		default:
			unsupported = append(unsupported, internal.TypeName(opt))
		}
	}

	if sp != "" {
		c.Messages = append(c.Messages, Message{Role: "system", Content: sp})
	}
	for i := range msgs {
		// Split messages into multiple messages as needed.
		switch {
		case len(msgs[i].ToolCallResults) > 1:
			// Handle messages with multiple tool call results by creating multiple messages
			for j := range msgs[i].ToolCallResults {
				// Create a copy of the message with only one tool call result
				msgCopy := msgs[i]
				msgCopy.ToolCallResults = []genai.ToolCallResult{msgs[i].ToolCallResults[j]}
				var newMsg Message
				if err := newMsg.From(&msgCopy); err != nil {
					errs = append(errs, fmt.Errorf("message %d, tool result %d: %w", i, j, err))
				} else {
					c.Messages = append(c.Messages, newMsg)
				}
			}
		case len(msgs[i].Requests) > 1:
			for j := range msgs[i].Requests {
				msgCopy := msgs[i]
				msgCopy.Requests = []genai.Request{msgs[i].Requests[j]}
				var newMsg Message
				if err := newMsg.From(&msgCopy); err != nil {
					errs = append(errs, fmt.Errorf("message %d, request %d: %w", i, j, err))
				} else {
					c.Messages = append(c.Messages, newMsg)
				}
			}
		default:
			var newMsg Message
			if err := newMsg.From(&msgs[i]); err != nil {
				errs = append(errs, fmt.Errorf("message %d: %w", i, err))
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

// SetStream sets the streaming mode.
func (c *ChatRequest) SetStream(stream bool) {
	c.Stream = stream
}

// Message is not well specified in the API documentation.
// https://developers.cloudflare.com/api/resources/ai/methods/run/
type Message struct {
	Role       string `json:"role"` // "system", "assistant", "user", "tool"
	Content    string `json:"content,omitzero"`
	ToolCallID string `json:"tool_call_id,omitzero"`
}

// From must be called with at most one Request, one Reply or one ToolCallResults.
func (m *Message) From(in *genai.Message) error {
	// We do not expect cloudflare to send multiple replies.
	if len(in.Requests) > 1 || len(in.Replies) > 1 || len(in.ToolCallResults) > 1 {
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
	if len(in.Requests) == 1 {
		// Process only the first Request in this method.
		// The Init method handles multiple Request by creating multiple messages.
		switch {
		case in.Requests[0].Text != "":
			m.Content = in.Requests[0].Text
		case !in.Requests[0].Doc.IsZero():
			// Check if this is a text document
			mimeType, data, err := in.Requests[0].Doc.Read(10 * 1024 * 1024)
			if err != nil {
				return fmt.Errorf("failed to read document: %w", err)
			}
			if !strings.HasPrefix(mimeType, "text/") {
				return fmt.Errorf("cloudflare only supports text documents, got %s", mimeType)
			}
			if in.Requests[0].Doc.URL != "" {
				return fmt.Errorf("%s documents must be provided inline, not as a URL", mimeType)
			}
			m.Content = string(data)
		default:
			return fmt.Errorf("unsupported content type %#v", in.Requests[0])
		}
		return nil
	}
	if len(in.Replies) != 0 {
		if len(in.Replies[0].Opaque) != 0 {
			return &internal.BadError{Err: errors.New("field Reply.Opaque not supported")}
		}
		switch {
		case in.Replies[0].Text != "":
			m.Content = in.Replies[0].Text
		case !in.Replies[0].Doc.IsZero():
			// Check if this is a text/plain document
			mimeType, data, err := in.Replies[0].Doc.Read(10 * 1024 * 1024)
			if err != nil {
				return fmt.Errorf("failed to read document: %w", err)
			}
			if !strings.HasPrefix(mimeType, "text/") {
				return fmt.Errorf("cloudflare only supports text documents, got %s", mimeType)
			}
			if in.Replies[0].Doc.URL != "" {
				return fmt.Errorf("%s documents must be provided inline, not as a URL", mimeType)
			}
			m.Content = string(data)
		case !in.Replies[0].ToolCall.IsZero():
			if len(in.Replies[0].ToolCall.Opaque) != 0 {
				return &internal.BadError{Err: errors.New("field ToolCall.Opaque not supported")}
			}
			m.ToolCallID = in.Replies[0].ToolCall.ID
			m.Content = in.Replies[0].ToolCall.Arguments
		default:
			return &internal.BadError{Err: fmt.Errorf("unsupported content type %#v", in.Replies[0])}
		}
		return nil
	}
	if len(in.ToolCallResults) == 1 {
		// Process only the first ToolCallResults in this method.
		// The Init method handles multiple ToolCallResults by creating multiple messages.
		m.ToolCallID = in.ToolCallResults[0].ID
		m.Content = in.ToolCallResults[0].Result
		return nil
	}
	return &internal.BadError{Err: errors.New("internal error")}
}

// Tool is a provider-specific tool definition.
type Tool struct {
	Type     string `json:"type"` // "function"
	Function struct {
		Description string           `json:"description"`
		Name        string           `json:"name"`
		Parameters  genai.JSONSchema `json:"parameters"`
	} `json:"function"`
}

/*
Maybe later

type function struct {
	Code string `json:"code"`
	Name string `json:"name"`
}

type prompt struct {
	Prompt            string  `json:"prompt"`
	FrequencyPenalty  float64 `json:"frequency_penalty,omitzero"` // [0, 2.0]
	Lora              string  `json:"lora,omitzero"`
	MaxTokens         int64   `json:"max_tokens,omitzero"`
	PresencePenalty   float64 `json:"presence_penalty,omitzero"`   // [0, 2.0]
	Raw               bool    `json:"raw,omitzero"`                // Do not apply chat template
	RepetitionPenalty float64 `json:"repetition_penalty,omitzero"` // [0, 2.0]
	ResponseFormat    struct {
		Type       string           `json:"type,omitzero"` // json_object, json_schema
		JSONSchema genai.JSONSchema `json:"json_schema,omitzero"`
	} `json:"response_format,omitzero"`
	Seed        int64   `json:"seed,omitzero"`
	Stream      bool    `json:"stream,omitzero"`
	Temperature float64 `json:"temperature,omitzero"` // [0, 5]
	TopK        int64   `json:"top_k,omitzero"`       // [1, 50]
	TopP        float64 `json:"top_p,omitzero"`       // [0, 2.0]
}

type textClassification struct {
	Text string `json:"text"`
}

type textToImage struct {
	Prompt         string  `json:"prompt"`
	Guidance       float64 `json:"guidance,omitzero"`
	Height         int64   `json:"height,omitzero"` // [256, 2048]
	Image          []uint8 `json:"image,omitzero"`
	ImageB64       []byte  `json:"image_b64,omitzero"`
	Mask           []uint8 `json:"mask,omitzero"`
	NegativePrompt string  `json:"negative_prompt,omitzero"`
	NumSteps       int64   `json:"num_steps,omitzero"` // Max 20
	Seed           int64   `json:"seed,omitzero"`
	Strength       float64 `json:"strength,omitzero"` // [0, 1]
	Width          int64   `json:"width,omitzero"`    // [256, 2048]
}

type textToSpeech struct {
	Prompt string `json:"prompt"`
	Lang   string `json:"lang,omitzero"` // en, fr, etc
}

type textEmbeddings struct {
	Text []string `json:"text"`
}

type automaticSpeechRecognition struct {
	Audio      []uint8 `json:"audio"`
	SourceLang string  `json:"source_lang,omitzero"`
	TargetLang string  `json:"target_lang,omitzero"`
}

type imageClassification struct {
	Image []uint8 `json:"image"`
}

type objectDetection struct {
	Image []uint8 `json:"image,omitzero"`
}

type translation struct {
	TargetLang string  `json:"target_lang"`
	Text       string  `json:"text"`
	SourceLang *string `json:"source_lang,omitzero"`
}

type summarization struct {
	InputText string `json:"input_text"`
	MaxLength *int   `json:"max_length,omitzero"`
}

type imageToText struct {
	Image             []uint8 `json:"image"`
	FrequencyPenalty  float64 `json:"frequency_penalty,omitzero"`
	MaxTokens         int64   `json:"max_tokens,omitzero"`
	PresencePenalty   float64 `json:"presence_penalty,omitzero"`
	Prompt            string  `json:"prompt,omitzero"`
	Raw               bool    `json:"raw,omitzero"`
	RepetitionPenalty float64 `json:"repetition_penalty,omitzero"`
	Seed              int64   `json:"seed,omitzero"`
	Temperature       float64 `json:"temperature,omitzero"`
	TopK              int64   `json:"top_k,omitzero"`
	TopP              float64 `json:"top_p,omitzero"`
}
*/

// ChatResponse is somewhat documented at https://developers.cloudflare.com/api/resources/ai/methods/run/
// See UnionMember7.
type ChatResponse struct {
	Result struct {
		MessageResponse
		Usage Usage `json:"usage"`
	} `json:"result"`
	Success  bool       `json:"success"`
	Errors   []struct{} `json:"errors"`   // Annoyingly, it's included all the time
	Messages []struct{} `json:"messages"` // Annoyingly, it's included all the time
}

// MessageResponse is a message in a provider-specific response.
type MessageResponse struct {
	// Normally a string, or an object if response_format.type == "json_schema".
	Response  json.RawMessage `json:"response"`
	ToolCalls []ToolCall      `json:"tool_calls"`
}

// To converts to the genai equivalent.
func (msg *MessageResponse) To(out *genai.Message) error {
	if len(msg.ToolCalls) != 0 {
		out.Replies = make([]genai.Reply, len(msg.ToolCalls))
		for i, tc := range msg.ToolCalls {
			if err := tc.To(&out.Replies[i].ToolCall); err != nil {
				return err
			}
		}
		return nil
	}
	var s string
	if err := json.Unmarshal(msg.Response, &s); err == nil {
		// This is just sad.
		if strings.HasPrefix(s, "<tool_call>") {
			return errors.New("hacked up XML tool calls are not supported")
		} else {
			out.Replies = []genai.Reply{{Text: s}}
		}
	} else if len(msg.Response) != 0 {
		out.Replies = []genai.Reply{{Text: string(msg.Response)}}
	} else {
		out.Replies = []genai.Reply{{Text: "null"}}
	}
	return nil
}

// ToResult converts the response to a genai.Result.
func (c *ChatResponse) ToResult() (genai.Result, error) {
	out := genai.Result{
		Usage: genai.Usage{
			InputTokens:       c.Result.Usage.PromptTokens,
			InputCachedTokens: c.Result.Usage.PromptTokensDetail.CachedTokens,
			OutputTokens:      c.Result.Usage.CompletionTokens,
			TotalTokens:       c.Result.Usage.TotalTokens,
			// Cloudflare doesn't provide FinishReason (!?)
		},
	}
	err := c.Result.To(&out.Message)
	return out, err
}

// ChatStreamChunkResponse is not documented.
// If you find the documentation for this please tell me!
type ChatStreamChunkResponse struct {
	Response  Response   `json:"response"`
	P         string     `json:"p"`
	ToolCalls []ToolCall `json:"tool_calls"`
	Usage     Usage      `json:"usage"`
}

// Response is normally the response but it can be true (bool) sometimes?
type Response string

// UnmarshalJSON implements json.Unmarshaler.
func (r *Response) UnmarshalJSON(b []byte) error {
	v := false
	if err := json.Unmarshal(b, &v); err == nil {
		*r = Response(strconv.FormatBool(v))
		return nil
	}
	return json.Unmarshal(b, (*string)(r))
}

// Usage is the provider-specific token usage.
type Usage struct {
	CompletionTokens   int64 `json:"completion_tokens"`
	PromptTokens       int64 `json:"prompt_tokens"`
	TotalTokens        int64 `json:"total_tokens"`
	PromptTokensDetail struct {
		CachedTokens int64 `json:"cached_tokens"`
	} `json:"prompt_tokens_details,omitzero"`
}

// ToolCall can be populated differently depending on the model used.
type ToolCall struct {
	Type     string `json:"type,omitzero"` // "function"
	ID       string `json:"id,omitzero"`
	Index    int64  `json:"index,omitzero"`
	Function struct {
		Name      string `json:"name,omitzero"`
		Arguments string `json:"arguments"`
	} `json:"function,omitzero"`

	Arguments json.RawMessage `json:"arguments"`
	Name      string          `json:"name"`
}

// To converts to the genai equivalent.
func (c *ToolCall) To(out *genai.ToolCall) error {
	out.ID = c.ID
	if out.Name = c.Name; c.Name == "" {
		out.Name = c.Function.Name
	}
	if c.Function.Arguments != "" {
		out.Arguments = c.Function.Arguments
	} else {
		raw := c.Arguments
		if len(raw) == 0 {
			raw = json.RawMessage("null")
		}
		out.Arguments = string(raw)
	}
	return nil
}

// Time is a wrapper around time.Time to support unmarshalling for cloudflare non-standard encoding.
type Time time.Time

// UnmarshalJSON implements json.Unmarshaler.
func (t *Time) UnmarshalJSON(b []byte) error {
	s := ""
	if err := json.Unmarshal(b, &s); err != nil {
		return err
	}
	t2, err := time.Parse("2006-01-02 15:04:05.999999999", s)
	if err != nil {
		return err
	}
	*t = Time(t2)
	return nil
}

// Model is the provider-specific model metadata.
type Model struct {
	ID          string `json:"id"`
	Source      int64  `json:"source"`
	Name        string `json:"name"`
	Description string `json:"description"`
	CreatedAt   Time   `json:"created_at"`
	Task        struct {
		ID          string `json:"id"`
		Name        string `json:"name"`
		Description string `json:"description"`
	} `json:"task"`
	Tags       []string `json:"tags"`
	Properties []struct {
		PropertyID string          `json:"property_id"`
		Value      json.RawMessage `json:"value"` // sometimes a string, sometimes an array
	} `json:"properties"`
}

// GetID implements genai.Model.
func (m *Model) GetID() string {
	return m.Name
}

// ModelPricing is the pricing information for a model.
type ModelPricing struct {
	Currency string  `json:"currency"`
	Price    float64 `json:"price"`
	Unit     string  `json:"unit"` // "per M input tokens", "per M output tokens"
}

func (m *Model) String() string {
	var suffixes []string
	pp := slices.Clone(m.Properties)
	sort.Slice(pp, func(i, j int) bool {
		return pp[i].PropertyID < pp[j].PropertyID
	})
	for _, p := range pp {
		if p.PropertyID == "info" || p.PropertyID == "terms" {
			continue
		}
		if v, ok := rawScalarString(p.Value); ok {
			suffixes = append(suffixes, fmt.Sprintf("%s=%s", p.PropertyID, v))
		} else {
			d := json.NewDecoder(bytes.NewReader(p.Value))
			d.DisallowUnknownFields()
			var mp []ModelPricing
			if err := d.Decode(&mp); err == nil {
				s := make([]string, 0, len(mp))
				for _, l := range mp {
					// Try to simplify the unit.
					unit := ""
					switch l.Unit {
					case "per M input tokens":
						unit = "/Mt in"
					case "per M output tokens":
						unit = "/Mt out"
					case "per audio minute":
						unit = "/min"
					case "per 512 by 512 tile":
						unit = "/512x512"
					default:
						unit = " " + l.Unit
					}
					s = append(s, fmt.Sprintf("%g$%s%s", l.Price, l.Currency, unit))
				}
				suffixes = append(suffixes, fmt.Sprintf("%s=[%s]", p.PropertyID, strings.Join(s, ", ")))
			} else {
				suffixes = append(suffixes, fmt.Sprintf("%s=%s", p.PropertyID, p.Value))
			}
		}
	}
	suffix := ""
	if len(suffixes) != 0 {
		suffix = " (" + strings.Join(suffixes, ", ") + ")"
	}
	// Description is good but it's verbose and the models are well known.
	return fmt.Sprintf("%s%s", m.Name, suffix)
}

// Context implements genai.Model.
func (m *Model) Context() int64 {
	for _, p := range m.Properties {
		if p.PropertyID == "context_window" || p.PropertyID == "max_input_tokens" {
			if s, ok := rawString(p.Value); ok {
				if v, err := strconv.ParseInt(s, 10, 64); err == nil {
					return v
				}
			}
		}
	}
	return 0
}

// Price returns the input and output price per token.
func (m *Model) Price() (float64, float64) {
	var mp []ModelPricing
	in := 0.
	out := 0.
	for _, p := range m.Properties {
		if p.PropertyID != "price" {
			continue
		}
		d := json.NewDecoder(bytes.NewReader(p.Value))
		d.DisallowUnknownFields()
		if err := d.Decode(&mp); err != nil {
			return in, out
		}
		for _, l := range mp {
			switch l.Unit {
			case "per M output tokens":
				out = l.Price
			case "per M input tokens":
				in = l.Price
			}
		}
		return in, out
	}
	return in, out
}

func rawScalarString(v json.RawMessage) (string, bool) {
	if s, ok := rawString(v); ok {
		return s, true
	}
	v = bytes.TrimSpace(v)
	if len(v) == 0 || v[0] == '[' || v[0] == '{' {
		return "", false
	}
	return string(v), true
}

func rawString(v json.RawMessage) (string, bool) {
	var s string
	if err := json.Unmarshal(v, &s); err != nil {
		return "", false
	}
	return s, true
}

// ModelsResponse represents the response structure for Cloudflare models listing.
type ModelsResponse struct {
	Result     []Model `json:"result"`
	ResultInfo struct {
		Count      int64 `json:"count"`
		Page       int64 `json:"page"`
		PerPage    int64 `json:"per_page"`
		TotalCount int64 `json:"total_count"`
	} `json:"result_info"`
	Success  bool       `json:"success"`
	Errors   []struct{} `json:"errors"`   // Annoyingly, it's included all the time
	Messages []struct{} `json:"messages"` // Annoyingly, it's included all the time
}

// ErrorResponse is the provider-specific error response.
type ErrorResponse struct {
	Errors []struct {
		Message string `json:"message"`
		Code    int    `json:"code"`
	} `json:"errors"`
	Success  bool       `json:"success"`
	Result   struct{}   `json:"result"`
	Messages []struct{} `json:"messages"` // Annoyingly, it's included all the time
}

func (er *ErrorResponse) Error() string {
	if len(er.Errors) == 0 {
		return fmt.Sprintf("unknown (%#v)", er)
	}
	// Sometimes Code is set too.
	return er.Errors[0].Message
}

// IsAPIError implements base.ErrorResponseI.
func (er *ErrorResponse) IsAPIError() bool {
	return true
}
