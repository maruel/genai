// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Package cloudflare implements a client for the Cloudflare AI API.
//
// It is described at https://developers.cloudflare.com/api/resources/ai/
package cloudflare

// See official client at https://github.com/cloudflare/cloudflare-go

import (
	"bytes"
	"context"
	_ "embed"
	"encoding/json"
	"errors"
	"fmt"
	"iter"
	"net/http"
	"net/url"
	"os"
	"reflect"
	"slices"
	"sort"
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

//go:embed scoreboard.json
var scoreboardJSON []byte

// AccountID provides an account ID for Cloudflare Workers AI.
//
// Get your account ID at https://dash.cloudflare.com/profile/api-tokens
type AccountID string

func (a AccountID) Validate() error {
	if a == "" {
		return errors.New("cloudflare.AccountID cannot be empty")
	}
	return nil
}

// Scoreboard for Cloudflare.
func Scoreboard() scoreboard.Score {
	var s scoreboard.Score
	d := json.NewDecoder(bytes.NewReader(scoreboardJSON))
	d.DisallowUnknownFields()
	if err := d.Decode(&s); err != nil {
		panic(fmt.Errorf("failed to unmarshal scoreboard.json: %w", err))
	}
	return s
}

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
		Type       string             `json:"type,omitzero"` // json_object, json_schema
		JSONSchema *jsonschema.Schema `json:"json_schema,omitzero"`
	} `json:"response_format,omitzero"`
	GuidedJSON  *jsonschema.Schema `json:"guided_json,omitzero"`
	Seed        int64              `json:"seed,omitzero"`
	Stream      bool               `json:"stream,omitzero"`
	Temperature float64            `json:"temperature,omitzero"` // [0, 5]
	Tools       []Tool             `json:"tools,omitzero"`
	TopK        int64              `json:"top_k,omitzero"` // [1, 50]
	TopP        float64            `json:"top_p,omitzero"` // [0, 2.0]
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
				c.ResponseFormat.JSONSchema = internal.JSONSchemaFor(reflect.TypeOf(v.DecodeAs))
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
					if c.Tools[i].Function.Parameters = t.InputSchemaOverride; c.Tools[i].Function.Parameters == nil {
						c.Tools[i].Function.Parameters = t.GetInputSchema()
					}
				}
			}
			if v.WebSearch {
				errs = append(errs, errors.New("unsupported OptionsTools.WebSearch"))
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
		if len(msgs[i].ToolCallResults) > 1 {
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
		} else if len(msgs[i].Requests) > 1 {
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
		} else {
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
		if in.Requests[0].Text != "" {
			m.Content = in.Requests[0].Text
		} else if !in.Requests[0].Doc.IsZero() {
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
		} else {
			return fmt.Errorf("unsupported content type %#v", in.Requests[0])
		}
		return nil
	}
	if len(in.Replies) != 0 {
		if len(in.Replies[0].Opaque) != 0 {
			return &internal.BadError{Err: errors.New("field Reply.Opaque not supported")}
		}
		if in.Replies[0].Text != "" {
			m.Content = in.Replies[0].Text
		} else if !in.Replies[0].Doc.IsZero() {
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
		} else if !in.Replies[0].ToolCall.IsZero() {
			if len(in.Replies[0].ToolCall.Opaque) != 0 {
				return &internal.BadError{Err: errors.New("field ToolCall.Opaque not supported")}
			}
			m.ToolCallID = in.Replies[0].ToolCall.ID
			m.Content = in.Replies[0].ToolCall.Arguments
		} else {
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

type Tool struct {
	Type     string `json:"type"` // "function"
	Function struct {
		Description string             `json:"description"`
		Name        string             `json:"name"`
		Parameters  *jsonschema.Schema `json:"parameters"`
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
		Type       string              `json:"type,omitzero"` // json_object, json_schema
		JSONSchema *jsonschema.Schema `json:"json_schema,omitzero"`
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
// See UnionMember7
type ChatResponse struct {
	Result struct {
		MessageResponse
		Usage Usage `json:"usage"`
	} `json:"result"`
	Success  bool       `json:"success"`
	Errors   []struct{} `json:"errors"`   // Annoyingly, it's included all the time
	Messages []struct{} `json:"messages"` // Annoyingly, it's included all the time
}

type MessageResponse struct {
	// Normally a string, or an object if response_format.type == "json_schema".
	Response  any        `json:"response"`
	ToolCalls []ToolCall `json:"tool_calls"`
}

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
	switch v := msg.Response.(type) {
	case string:
		// This is just sad.
		if strings.HasPrefix(v, "<tool_call>") {
			return fmt.Errorf("hacked up XML tool calls are not supported")
		} else {
			out.Replies = []genai.Reply{{Text: v}}
		}
	default:
		// Marshal back into JSON.
		b, err := json.Marshal(v)
		if err != nil {
			return fmt.Errorf("failed to JSON marshal type %T: %v: %w", v, v, err)
		}
		out.Replies = []genai.Reply{{Text: string(b)}}
	}
	return nil
}

func (c *ChatResponse) ToResult() (genai.Result, error) {
	out := genai.Result{
		// At the moment, Cloudflare doesn't support cached tokens.
		Usage: genai.Usage{
			InputTokens:  c.Result.Usage.PromptTokens,
			OutputTokens: c.Result.Usage.CompletionTokens,
			TotalTokens:  c.Result.Usage.TotalTokens,
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

func (r *Response) UnmarshalJSON(b []byte) error {
	v := false
	if err := json.Unmarshal(b, &v); err == nil {
		*r = Response(strconv.FormatBool(v))
		return nil
	}
	return json.Unmarshal(b, (*string)(r))
}

type Usage struct {
	CompletionTokens int64 `json:"completion_tokens"`
	PromptTokens     int64 `json:"prompt_tokens"`
	TotalTokens      int64 `json:"total_tokens"`
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

	Arguments any    `json:"arguments"`
	Name      string `json:"name"`
}

func (c *ToolCall) To(out *genai.ToolCall) error {
	out.ID = c.ID
	if out.Name = c.Name; c.Name == "" {
		out.Name = c.Function.Name
	}
	if c.Function.Arguments != "" {
		out.Arguments = c.Function.Arguments
	} else {
		raw, err := json.Marshal(c.Arguments)
		if err != nil {
			return fmt.Errorf("failed to marshal tool call arguments: %w", err)
		}
		out.Arguments = string(raw)
	}
	return nil
}

// Time is a wrapper around time.Time to support unmarshalling for cloudflare non-standard encoding.
type Time time.Time

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
		PropertyID string `json:"property_id"`
		Value      any    `json:"value"` // sometimes a string, sometimes an array
	} `json:"properties"`
}

func (m *Model) GetID() string {
	return m.Name
}

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
		switch p.Value.(type) {
		case json.Number, string, int:
			suffixes = append(suffixes, fmt.Sprintf("%s=%v", p.PropertyID, p.Value))
		default:
			b, _ := json.Marshal(p.Value)
			d := json.NewDecoder(bytes.NewReader(b))
			d.DisallowUnknownFields()
			var mp []ModelPricing
			if err := d.Decode(&mp); err == nil {
				var s []string
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
				suffixes = append(suffixes, fmt.Sprintf("%s=%v", p.PropertyID, p.Value))
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

func (m *Model) Context() int64 {
	for _, p := range m.Properties {
		if p.PropertyID == "context_window" || p.PropertyID == "max_input_tokens" {
			if s, ok := p.Value.(string); ok {
				if v, err := strconv.ParseInt(s, 10, 64); err == nil {
					return v
				}
			}
		}
	}
	return 0
}

func (m *Model) Price() (float64, float64) {
	var mp []ModelPricing
	in := 0.
	out := 0.
	for _, p := range m.Properties {
		if p.PropertyID != "price" {
			continue
		}
		b, _ := json.Marshal(p.Value)
		d := json.NewDecoder(bytes.NewReader(b))
		d.DisallowUnknownFields()
		if err := d.Decode(&mp); err != nil {
			return in, out
		}
		for _, l := range mp {
			if l.Unit == "per M output tokens" {
				out = l.Price
			} else if l.Unit == "per M input tokens" {
				in = l.Price
			}
		}
		return in, out
	}
	return in, out
}

// ModelsResponse represents the response structure for Cloudflare models listing
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

//

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

func (er *ErrorResponse) IsAPIError() bool {
	return true
}

// Client implements genai.Provider.
type Client struct {
	base.NotImplemented
	impl      base.Provider[*ErrorResponse, *ChatRequest, *ChatResponse, ChatStreamChunkResponse]
	accountID string
}

// New creates a new client to talk to the Cloudflare Workers AI platform API.
//
// If AccountID is not provided, it tries to load it from the CLOUDFLARE_ACCOUNT_ID environment variable.
// If ProviderOptionAPIKey is not provided, it tries to load it from the CLOUDFLARE_API_KEY environment variable.
// If none is found, it will still return a client coupled with an base.ErrAPIKeyRequired error.
// Get your account ID and API key at https://dash.cloudflare.com/profile/api-tokens
//
// To use multiple models, create multiple clients.
// Use one of the model from https://developers.cloudflare.com/workers-ai/models/
func New(ctx context.Context, opts ...genai.ProviderOption) (*Client, error) {
	var apiKey, accountID, model string
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
		case AccountID:
			accountID = string(v)
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
	const apiKeyURL = "https://dash.cloudflare.com/profile/api-tokens"
	var err error
	if accountID == "" {
		if accountID = os.Getenv("CLOUDFLARE_ACCOUNT_ID"); accountID == "" {
			err = &base.ErrAPIKeyRequired{EnvVar: "CLOUDFLARE_ACCOUNT_ID", URL: apiKeyURL}
		}
	}
	if apiKey == "" {
		if apiKey = os.Getenv("CLOUDFLARE_API_KEY"); apiKey == "" {
			err = &base.ErrAPIKeyRequired{EnvVar: "CLOUDFLARE_API_KEY", URL: apiKeyURL}
		}
	}
	mod := genai.Modalities{genai.ModalityText}
	if len(modalities) != 0 && !slices.Equal(modalities, mod) {
		// TODO: Cloudflare supports non-text modalities but it is not currently implemented.
		// https://developers.cloudflare.com/workers-ai/models/?tasks=Text-to-Image
		return nil, fmt.Errorf("unexpected option Modalities %s, only text is implemented (send PR to add support)", mod)
	}
	t := base.DefaultTransport
	if wrapper != nil {
		t = wrapper(t)
	}
	// Investigate websockets?
	// https://blog.cloudflare.com/workers-ai-streaming/ and
	// https://developers.cloudflare.com/workers/examples/websockets/
	c := &Client{
		impl: base.Provider[*ErrorResponse, *ChatRequest, *ChatResponse, ChatStreamChunkResponse]{
			ProcessStream:   ProcessStream,
			PreloadedModels: preloadedModels,
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
		accountID: accountID,
	}
	if err == nil {
		switch model {
		case "":
		case string(genai.ModelCheap), string(genai.ModelGood), string(genai.ModelSOTA):
			if c.impl.Model, err = c.selectBestTextModel(ctx, model); err != nil {
				return nil, err
			}
			// Important: the model must not be path escaped!
			c.impl.GenSyncURL = "https://api.cloudflare.com/client/v4/accounts/" + url.PathEscape(accountID) + "/ai/run/" + c.impl.Model
			c.impl.OutputModalities = mod
		default:
			c.impl.Model = model
			c.impl.GenSyncURL = "https://api.cloudflare.com/client/v4/accounts/" + url.PathEscape(accountID) + "/ai/run/" + c.impl.Model
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
	cheap := preference == string(genai.ModelCheap)
	good := preference == string(genai.ModelGood) || preference == ""
	selectedModel := ""
	price := 100000.
	if !cheap {
		price = 0.
	}
	for _, mdl := range mdls {
		m := mdl.(*Model)
		if strings.Contains(m.Name, "guard") || strings.HasPrefix(m.Name, "@cf/meta/llama-2") {
			// llama-guard is not a generation model.
			// @cf/meta/llama-2-7b-chat-fp16 is super expensive.
			continue
		}
		_, out := m.Price()
		if out == 0 {
			continue
		}
		if cheap {
			if strings.HasPrefix(m.Name, "@cf/meta/") && out < price {
				price = out
				selectedModel = m.Name
			}
		} else if good {
			if strings.HasPrefix(m.Name, "@cf/meta/") && out > price {
				price = out
				selectedModel = m.Name
			}
		} else {
			if strings.HasPrefix(m.Name, "@cf/deepseek-ai/") && out > price {
				price = out
				selectedModel = m.Name
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
	return "cloudflare"
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
func (c *Client) GenSync(ctx context.Context, msgs genai.Messages, opts ...genai.GenOption) (genai.Result, error) {
	return c.impl.GenSync(ctx, msgs, opts...)
}

// GenSyncRaw provides access to the raw API.
func (c *Client) GenSyncRaw(ctx context.Context, in *ChatRequest, out *ChatResponse) error {
	return c.impl.GenSyncRaw(ctx, in, out)
}

// GenStream implements genai.Provider.
func (c *Client) GenStream(ctx context.Context, msgs genai.Messages, opts ...genai.GenOption) (iter.Seq[genai.Reply], func() (genai.Result, error)) {
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
	// https://developers.cloudflare.com/api/resources/ai/subresources/models/methods/list/
	var models []genai.Model
	for page := 1; ; page++ {
		out := ModelsResponse{}
		// Cloudflare's pagination is surprisingly brittle.
		url := fmt.Sprintf("https://api.cloudflare.com/client/v4/accounts/%s/ai/models/search?page=%d&per_page=100&hide_experimental=false", url.PathEscape(c.accountID), page)
		err := c.impl.DoRequest(ctx, "GET", url, nil, &out)
		if err != nil {
			return nil, err
		}
		for i := range out.Result {
			models = append(models, &out.Result[i])
		}
		if len(models) >= int(out.ResultInfo.TotalCount) || len(out.Result) == 0 {
			break
		}
	}
	return models, nil
}

// ProcessStream converts the raw packets from the streaming API into Reply fragments.
func ProcessStream(chunks iter.Seq[ChatStreamChunkResponse]) (iter.Seq[genai.Reply], func() (genai.Usage, [][]genai.Logprob, error)) {
	var finalErr error
	u := genai.Usage{}

	return func(yield func(genai.Reply) bool) {
			for pkt := range chunks {
				if pkt.Usage.TotalTokens != 0 {
					u.InputTokens = pkt.Usage.PromptTokens
					u.OutputTokens = pkt.Usage.CompletionTokens
					u.TotalTokens = pkt.Usage.TotalTokens
					// Cloudflare doesn't provide FinishReason.
				}
				// TODO: Tools.
				if !yield(genai.Reply{Text: string(pkt.Response)}) {
					return
				}
			}
		}, func() (genai.Usage, [][]genai.Logprob, error) {
			return u, nil, finalErr
		}
}

var _ genai.Provider = &Client{}
