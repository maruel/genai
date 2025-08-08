// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Package cohere implements a client for the Cohere API.
//
// It is described at https://docs.cohere.com/reference/
package cohere

// See official client at https://github.com/cohere-ai/cohere-go

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"os"
	"reflect"
	"slices"
	"sort"
	"strings"

	"github.com/invopop/jsonschema"
	"github.com/maruel/genai"
	"github.com/maruel/genai/base"
	"github.com/maruel/genai/internal"
	"github.com/maruel/httpjson"
	"github.com/maruel/roundtrippers"
)

// Scoreboard for Cohere.
//
// # Warnings
//
//   - Cohere doesn't support multimodal inputs yet.
//   - Tool calling works very well but is biased; the model is lazy and when it's unsure, it will use the
//     tool's first argument.
//   - The API has good citations support but it's not well implemented yet.
//   - Free tier rate limit is lower: https://docs.cohere.com/v2/docs/rate-limits
var Scoreboard = genai.Scoreboard{
	Country:      "CA",
	DashboardURL: "https://dashboard.cohere.com/billing",
	Scenarios: []genai.Scenario{
		{
			Models:   []string{"command-r7b-12-2024"},
			Thinking: true, // Only related to tool planning.
			In:       map[genai.Modality]genai.ModalCapability{genai.ModalityText: {Inline: true}},
			Out:      map[genai.Modality]genai.ModalCapability{genai.ModalityText: {Inline: true}},
			GenSync: &genai.FunctionalityText{
				Tools:          genai.True,
				IndecisiveTool: genai.True,
				JSON:           true,
				JSONSchema:     true,
				Citations:      true,
				Seed:           true,
			},
			GenStream: &genai.FunctionalityText{
				Tools:          genai.True,
				IndecisiveTool: genai.True,
				JSON:           true,
				JSONSchema:     true,
				Citations:      true,
				Seed:           true,
			},
		},
		// https://docs.cohere.com/docs/aya-vision
		{
			Models:   []string{"c4ai-aya-vision-8b"},
			Thinking: true, // Only related to tool planning.
			In: map[genai.Modality]genai.ModalCapability{
				genai.ModalityText: {Inline: true},
				genai.ModalityImage: {
					Inline:           true,
					URL:              true,
					SupportedFormats: []string{"image/gif", "image/jpeg", "image/png", "image/webp"},
				},
			},
			Out: map[genai.Modality]genai.ModalCapability{genai.ModalityText: {Inline: true}},
			GenSync: &genai.FunctionalityText{
				Tools:     genai.True,
				Citations: true,
				Seed:      true,
			},
			GenStream: &genai.FunctionalityText{
				Tools:     genai.True,
				Citations: true,
				Seed:      true,
			},
		},
		// To enable.
		{
			Models: []string{
				"c4ai-aya-expanse-32b",
				"c4ai-aya-expanse-8b",
				"c4ai-aya-vision-32b",
				"command",
				"command-a-03-2025",
				"command-a-vision-07-2025",
				"command-light",
				"command-light-nightly",
				"command-nightly",
				"command-r",
				"command-r-08-2024",
				"command-r-plus",
				"command-r-plus-08-2024",
				"command-r7b-arabic-02-2025",
			},
		},
		// Unsupported.
		{
			Models: []string{
				"embed-english-light-v2.0",
				"embed-english-light-v3.0",
				"embed-english-light-v3.0-image",
				"embed-english-v2.0",
				"embed-english-v3.0",
				"embed-english-v3.0-image",
				"embed-multilingual-light-v3.0",
				"embed-multilingual-light-v3.0-image",
				"embed-multilingual-v2.0",
				"embed-multilingual-v3.0",
				"embed-multilingual-v3.0-image",
				"embed-v4.0",
				"rerank-english-v3.0",
				"rerank-multilingual-v3.0",
				"rerank-v3.5",
			},
		},
	},
}

// ChatRequest is documented at https://docs.cohere.com/reference/chat
type ChatRequest struct {
	Stream          bool       `json:"stream"`
	Model           string     `json:"model"`
	Messages        []Message  `json:"messages"`
	Documents       []Document `json:"documents,omitzero"`
	CitationOptions struct {
		Mode string `json:"mode,omitzero"` // "fast", "accurate", "off"; default "fast" for command-r7b-12-2024 and command-a-03-2025, else "accurate".
	} `json:"citation_options,omitzero"`
	ResponseFormat struct {
		Type       string             `json:"type,omitzero"` // "text", "json_object"
		JSONSchema *jsonschema.Schema `json:"json_schema,omitzero"`
	} `json:"response_format,omitzero"`
	SafetyMode       string   `json:"safety_mode,omitzero"` // "CONTEXTUAL", "STRICT", "OFF"
	MaxTokens        int64    `json:"max_tokens,omitzero"`
	StopSequences    []string `json:"stop_sequences,omitzero"` // Up to 5 words
	Temperature      float64  `json:"temperature,omitzero"`
	Seed             int64    `json:"seed,omitzero"`
	FrequencyPenalty float64  `json:"frequency_penalty,omitzero"` // [0, 1.0]
	PresencePenalty  float64  `json:"presence_penalty,omitzero"`  // [0, 1.0]
	K                int64    `json:"k,omitzero"`                 // [0, 500]
	P                float64  `json:"p,omitzero"`                 // [0.01, 0.99]
	Logprobs         bool     `json:"logprobs,omitzero"`
	Tools            []Tool   `json:"tools,omitzero"`
	ToolChoice       string   `json:"tool_choice,omitzero"` // "required", "none"
	StrictTools      bool     `json:"strict_tools,omitzero"`
}

// Init initializes the provider specific completion request with the generic completion request.
func (c *ChatRequest) Init(msgs genai.Messages, opts genai.Options, model string) error {
	c.Model = model
	var errs []error
	var unsupported []string
	sp := ""
	if opts != nil {
		switch v := opts.(type) {
		case *genai.OptionsText:
			c.MaxTokens = v.MaxTokens
			c.Temperature = v.Temperature
			c.P = v.TopP
			sp = v.SystemPrompt
			c.Seed = v.Seed
			c.K = v.TopK
			c.StopSequences = v.Stop
			if v.DecodeAs != nil {
				c.ResponseFormat.Type = "json_schema"
				c.ResponseFormat.JSONSchema = internal.JSONSchemaFor(reflect.TypeOf(v.DecodeAs))
			} else if v.ReplyAsJSON {
				c.ResponseFormat.Type = "json_object"
			}
			if len(v.Tools) != 0 {
				switch v.ToolCallRequest {
				case genai.ToolCallAny:
				case genai.ToolCallRequired:
					c.ToolChoice = "required"
					c.StrictTools = true
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
			errs = append(errs, fmt.Errorf("unsupported options type %T", opts))
		}
	}

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
		d, err := c.Messages[i+offset].From(&msgs[i])
		if err != nil {
			errs = append(errs, fmt.Errorf("message %d: %w", i, err))
		}
		if len(d) != 0 {
			c.Documents = append(c.Documents, d...)
		}
		if len(c.Messages[i+offset].Content) == 0 && len(c.Messages[i+offset].ToolCalls) == 0 {
			errs = append(errs, fmt.Errorf("message %d: must have at least one content or tool call block", i))
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

func (c *ChatRequest) SetStream(stream bool) {
	c.Stream = stream
}

// Message is documented at https://docs.cohere.com/reference/chat
type Message struct {
	Role string `json:"role"` // "system", "assistant", "user", "tool"
	// Type == "system", "assistant", "user" or "tool".
	Content []Content `json:"content,omitzero"`
	// Type == "assistant"
	Citations []Citation `json:"citations,omitzero"`
	ToolCalls []ToolCall `json:"tool_calls,omitzero"`
	ToolPlan  string     `json:"tool_plan,omitzero"`
	// Type == "tool"
	ToolCallID string `json:"tool_call_id,omitzero"`
}

func (m *Message) From(in *genai.Message) ([]Document, error) {
	switch in.Role {
	case genai.User, genai.Assistant:
		m.Role = string(in.Role)
	case genai.Computer:
		fallthrough
	default:
		return nil, fmt.Errorf("unsupported role %q", in.Role)
	}
	var out []Document
	if len(in.Contents) != 0 {
		for i := range in.Contents {
			if in.Contents[i].Thinking != "" {
				// Silently ignore thinking blocks.
				continue
			}
			c := Content{}
			d, err := c.From(&in.Contents[i])
			if err != nil {
				return nil, fmt.Errorf("block %d: %w", i, err)
			}
			if d != nil {
				out = append(out, *d)
			} else {
				m.Content = append(m.Content, c)
			}
		}
	}
	if len(in.ToolCalls) != 0 {
		m.ToolCalls = make([]ToolCall, len(in.ToolCalls))
		for i := range in.ToolCalls {
			m.ToolCalls[i].From(&in.ToolCalls[i])
		}
	}
	if len(in.ToolCallResults) != 0 {
		if len(in.Contents) != 0 || len(in.ToolCalls) != 0 {
			// This could be worked around.
			return out, fmt.Errorf("can't have tool call result along content or tool calls")
		}
		if len(in.ToolCallResults) != 1 {
			// This could be worked around.
			return out, fmt.Errorf("can't have more than one tool call result at a time")
		}
		// Cohere supports Document, but only when using tools.
		m.Role = "tool"
		m.ToolCallID = in.ToolCallResults[0].ID
		m.Content = []Content{{Type: ContentText, Text: in.ToolCallResults[0].Result}}
	}
	return out, nil
}

type Content struct {
	Type ContentType `json:"type,omitzero"`
	Text string      `json:"text,omitzero"`

	// Only used when Type == ContentImageURL and Role == "user".
	ImageURL struct {
		URL string `json:"url,omitzero"`
	} `json:"image_url,omitzero"`

	// Only used when Type == ContentDocument and Role == "tool" for tool results.
	Document Document `json:"document,omitzero"`
}

func (c *Content) IsZero() bool {
	return c.Type == "" && c.Text == "" && c.ImageURL.URL == "" && len(c.Document.Data) == 0 && c.Document.ID == ""
}

func (c *Content) From(in *genai.Content) (*Document, error) {
	if in.Text != "" {
		c.Type = ContentText
		c.Text = in.Text
		return nil, nil
	}

	mimeType, data, err := in.ReadDocument(10 * 1024 * 1024)
	if err != nil {
		return nil, err
	}
	switch {
	case (in.URL != "" && mimeType == "") || strings.HasPrefix(mimeType, "image/"):
		c.Type = ContentImageURL
		if in.URL != "" {
			c.ImageURL.URL = in.URL
		} else {
			c.ImageURL.URL = fmt.Sprintf("data:%s;base64,%s", mimeType, base64.StdEncoding.EncodeToString(data))
		}
		return nil, nil
	case strings.HasPrefix(mimeType, "text/plain"):
		if in.URL != "" {
			return nil, errors.New("text/plain documents must be provided inline, not as a URL")
		}
		name := in.GetFilename()
		d := &Document{
			ID:   name,
			Data: map[string]any{"title": name, "snippet": string(data)},
		}
		// This is handled as ChatRequest.Documents.
		return d, nil
	default:
		return nil, fmt.Errorf("unsupported mime type %s", mimeType)
	}
}

func (c *Content) To(in *genai.Content) error {
	switch c.Type {
	case ContentText:
		in.Text = c.Text
	case ContentDocument, ContentImageURL:
		fallthrough
	default:
		return fmt.Errorf("implement %s", c.Type)
	}
	return nil
}

// ContentType is documented at https://docs.cohere.com/v2/reference/chat
type ContentType string

const (
	ContentText     ContentType = "text"
	ContentImageURL ContentType = "image_url"
	ContentDocument ContentType = "document"
)

type Citations []Citation

// UnmarshalJSON implements custom unmarshalling for Citations type
// to handle cases where citations could be a list or a single object.
func (c *Citations) UnmarshalJSON(b []byte) error {
	if bytes.Equal(b, []byte("null")) {
		*c = nil
		return nil
	}
	d := json.NewDecoder(bytes.NewReader(b))
	if !internal.BeLenient {
		d.DisallowUnknownFields()
	}
	if err := d.Decode((*[]Citation)(c)); err == nil {
		return nil
	}

	v := Citation{}
	d = json.NewDecoder(bytes.NewReader(b))
	if !internal.BeLenient {
		d.DisallowUnknownFields()
	}
	if err := d.Decode(&v); err != nil {
		return err
	}
	*c = Citations{v}
	return nil
}

func (c Citations) To(dst *genai.Content) error {
	dst.Citations = make([]genai.Citation, len(c))
	for i := range c {
		if err := c[i].To(&dst.Citations[i]); err != nil {
			return fmt.Errorf("citation %d: %w", i, err)
		}
	}
	return nil
}

// Citation is only used with Role == "assistant"
type Citation struct {
	Start   int64            `json:"start,omitzero"`
	End     int64            `json:"end,omitzero"`
	Text    string           `json:"text,omitzero"`
	Sources []CitationSource `json:"sources,omitzero"`
	Type    string           `json:"type,omitzero"` // "TEXT_CONTENT", "PLAN"
}

func (c *Citation) To(dst *genai.Citation) error {
	dst.Text = c.Text
	dst.StartIndex = c.Start
	dst.EndIndex = c.End
	dst.Type = c.Type
	dst.Sources = make([]genai.CitationSource, len(c.Sources))
	for i, source := range c.Sources {
		cs := &dst.Sources[i]
		cs.ID = source.ID
		cs.Type = source.Type
		switch source.Type {
		case "tool":
			cs.Metadata = map[string]any{
				"tool_output": source.ToolOutput,
			}
		case "document":
			cs.Metadata = map[string]any{
				"document": source.Document,
			}
		}
	}
	return nil
}

type CitationSource struct {
	Type string `json:"type,omitzero"` // "tool", "document"

	// Type == "tool", "document"
	ID string `json:"id,omitzero"`

	// Type == "tool"
	ToolOutput map[string]any `json:"tool_output,omitzero"`

	// Type == "document"
	Document map[string]any `json:"document,omitzero"`
}

type Tool struct {
	Type     string `json:"type,omitzero"` // "function"
	Function struct {
		Name        string             `json:"name,omitzero"`
		Parameters  *jsonschema.Schema `json:"parameters,omitzero"`
		Description string             `json:"description,omitzero"`
	} `json:"function,omitzero"`
}

// Document can be used in the ChatRequest.Documents field or as a tool result. It's annoying because genai
// passes documents as Content inside a Message.
type Document struct {
	// Or a string.
	ID   string         `json:"id,omitzero"`
	Data map[string]any `json:"data,omitzero"`
}

type ChatResponse struct {
	ID           string          `json:"id"`
	FinishReason FinishReason    `json:"finish_reason"`
	Message      MessageResponse `json:"message"`
	Usage        Usage           `json:"usage"`
	Logprobs     []struct {
		TokenIDs []int64   `json:"token_ids"`
		Text     string    `json:"text"`
		Logprobs []float64 `json:"logprobs"`
	} `json:"logprobs"`
}

func (c *ChatResponse) ToResult() (genai.Result, error) {
	out := genai.Result{
		// At the moment, Cohere doesn't support cached tokens.
		Usage: genai.Usage{
			// What about BilledUnits, especially for SearchUnits and Classifications?
			InputTokens:  c.Usage.Tokens.InputTokens,
			OutputTokens: c.Usage.Tokens.OutputTokens,
			FinishReason: c.FinishReason.ToFinishReason(),
		},
	}
	// It is very frustrating that Cohere uses different message response types.
	err := c.Message.To(&out.Message)
	return out, err
}

type FinishReason string

const (
	FinishComplete     FinishReason = "COMPLETE"
	FinishStopSequence FinishReason = "STOP_SEQUENCE"
	FinishMaxTokens    FinishReason = "MAX_TOKENS"
	FinishToolCall     FinishReason = "TOOL_CALL"
	FinishError        FinishReason = "ERROR"
)

func (f FinishReason) ToFinishReason() genai.FinishReason {
	switch f {
	case FinishComplete:
		return genai.FinishedStop
	case FinishToolCall:
		return genai.FinishedToolCalls
	case FinishMaxTokens:
		return genai.FinishedLength
	case FinishStopSequence:
		return genai.FinishedStopSequence
	case FinishError:
		fallthrough
	default:
		if !internal.BeLenient {
			panic(f)
		}
		return genai.FinishReason(strings.ToLower(string(f)))
	}
}

type Usage struct {
	BilledUnits struct {
		InputTokens     int64 `json:"input_tokens"`
		OutputTokens    int64 `json:"output_tokens"`
		SearchUnits     int64 `json:"search_units"`
		Classifications int64 `json:"classifications"`
	} `json:"billed_units"`
	Tokens struct {
		InputTokens  int64 `json:"input_tokens"`
		OutputTokens int64 `json:"output_tokens"`
		ImageTokens  int64 `json:"image_tokens"`
	} `json:"tokens"`
}

// MessageResponse handles all the various forms that Cohere can reply.
//
//   - For non-stream text, "content" is []Content.
//   - For streaming text, "content" is initially an empty list, then Content (not a list).
//   - For non-stream tool call, Tool* members are set and Content is never present. ToolCalls is a list.
//   - For streaming tool call, Tool* members are set and Content is never present. ToolCalls is a ToolCall (not
//     a list).
type MessageResponse struct {
	Content Contents `json:"content"` // Generally a []Content but will be a Content when streaming text.
	Role    string   `json:"role"`    // "system", "assistant", "user"
	// Type == "assistant"
	Citations Citations `json:"citations,omitzero"`
	// Type == "assistant"
	ToolCalls  ToolCalls `json:"tool_calls,omitzero"` // Generally []ToolCall but will be a ToolCall when streaming tool call.
	ToolCallID string    `json:"tool_call_id,omitzero"`
	ToolPlan   string    `json:"tool_plan,omitzero"`
}

func (m *MessageResponse) To(out *genai.Message) error {
	switch role := m.Role; role {
	case "system", "assistant", "user":
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
	if m.ToolCallID != "" && !internal.BeLenient {
		return fmt.Errorf("implement tool call id")
	}
	if m.ToolPlan != "" {
		out.Contents = []genai.Content{{Thinking: m.ToolPlan}}
	}
	if len(m.Content) != 0 {
		for i := range m.Content {
			out.Contents = append(out.Contents, genai.Content{})
			if err := m.Content[len(m.Content)-1].To(&out.Contents[i]); err != nil {
				return fmt.Errorf("block %d: %w", i, err)
			}
		}
		if len(m.Citations) != 0 {
			for i := range out.Contents {
				if out.Contents[i].Text != "" {
					if err := m.Citations.To(&out.Contents[i]); err != nil {
						return fmt.Errorf("mapping citations: %w", err)
					}
					// TODO: handle multiple citations.
					break
				}
			}
		}
	}
	return nil
}

type Contents []Content

func (c *Contents) UnmarshalJSON(b []byte) error {
	if bytes.Equal(b, []byte("null")) {
		*c = nil
		return nil
	}
	d := json.NewDecoder(bytes.NewReader(b))
	if !internal.BeLenient {
		d.DisallowUnknownFields()
	}
	if err := d.Decode((*[]Content)(c)); err == nil {
		return nil
	}

	v := Content{}
	d = json.NewDecoder(bytes.NewReader(b))
	if !internal.BeLenient {
		d.DisallowUnknownFields()
	}
	if err := d.Decode(&v); err != nil {
		return err
	}
	if !v.IsZero() {
		*c = (Contents)([]Content{v})
	} else {
		*c = nil
	}
	return nil
}

type ToolCall struct {
	ID       string `json:"id"`
	Type     string `json:"type"` // function
	Function struct {
		Name      string `json:"name"`
		Arguments string `json:"arguments"`
	} `json:"function"`
}

func (t *ToolCall) IsZero() bool {
	return t.ID == "" && t.Type == "" && t.Function.Name == "" && t.Function.Arguments == ""
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

type ToolCalls []ToolCall

func (t *ToolCalls) UnmarshalJSON(b []byte) error {
	if bytes.Equal(b, []byte("null")) {
		*t = nil
		return nil
	}
	d := json.NewDecoder(bytes.NewReader(b))
	if !internal.BeLenient {
		d.DisallowUnknownFields()
	}
	if err := d.Decode((*[]ToolCall)(t)); err == nil {
		return nil
	}

	tc := ToolCall{}
	d = json.NewDecoder(bytes.NewReader(b))
	if !internal.BeLenient {
		d.DisallowUnknownFields()
	}
	if err := d.Decode(&tc); err != nil {
		return err
	}
	if !tc.IsZero() {
		*t = (ToolCalls)([]ToolCall{tc})
	} else {
		*t = nil
	}
	return nil
}

type ChatStreamChunkResponse struct {
	ID    string    `json:"id"`
	Type  ChunkType `json:"type"`
	Index int64     `json:"index"`
	Delta struct {
		Message      MessageResponse `json:"message"`
		FinishReason FinishReason    `json:"finish_reason"`
		Usage        Usage           `json:"usage"`
	} `json:"delta"`
}

type ChunkType string

const (
	ChunkMessageStart  ChunkType = "message-start"
	ChunkMessageEnd    ChunkType = "message-end"
	ChunkContentStart  ChunkType = "content-start"
	ChunkContentDelta  ChunkType = "content-delta"
	ChunkContentEnd    ChunkType = "content-end"
	ChunkToolPlanDelta ChunkType = "tool-plan-delta"
	ChunkToolCallStart ChunkType = "tool-call-start"
	ChunkToolCallDelta ChunkType = "tool-call-delta"
	ChunkToolCallEnd   ChunkType = "tool-call-end"
	ChunkCitationStart ChunkType = "citation-start"
	ChunkCitationEnd   ChunkType = "citation-end"
)

type Model struct {
	Name             string   `json:"name"`
	Endpoints        []string `json:"endpoints"` // chat, embed, classify, summarize, rerank, rate, generate
	Features         []string `json:"features"`  // json_mode, json_schema, safety_modes, strict_tools, tools
	Finetuned        bool     `json:"finetuned"`
	ContextLength    int64    `json:"context_length"`
	TokenizerURL     string   `json:"tokenizer_url"`
	SupportsVision   bool     `json:"supports_vision"`
	DefaultEndpoints []string `json:"default_endpoints"`
	IsDeprecated     bool     `json:"is_deprecated"`
}

func (m *Model) GetID() string {
	return m.Name
}

func (m *Model) String() string {
	suffix := ""
	if m.Finetuned {
		suffix += " (finetuned)"
	}
	if m.SupportsVision {
		suffix += " (vision)"
	}
	endpoints := make([]string, len(m.Endpoints))
	copy(endpoints, m.Endpoints)
	sort.Strings(endpoints)
	f := ""
	if len(m.Features) > 0 {
		features := make([]string, len(m.Features))
		copy(features, m.Features)
		sort.Strings(features)
		f = " with " + strings.Join(features, "/")
	}
	return fmt.Sprintf("%s: %s%s. Context: %d%s", m.Name, strings.Join(endpoints, "/"), f, m.ContextLength, suffix)
}

func (m *Model) Context() int64 {
	return m.ContextLength
}

// ModelsResponse represents the response structure for Cohere models listing
type ModelsResponse struct {
	Models        []Model `json:"models"`
	NextPageToken string  `json:"next_page_token"`
}

// ToModels converts Cohere models to genai.Model interfaces
func (r *ModelsResponse) ToModels() []genai.Model {
	models := make([]genai.Model, len(r.Models))
	for i := range r.Models {
		models[i] = &r.Models[i]
	}
	return models
}

//

type ErrorResponse struct {
	ID        string `json:"id"`
	Message   string `json:"message"`
	RequestID string `json:"request_id"`
}

func (er *ErrorResponse) Error() string {
	return er.Message
}

func (er *ErrorResponse) IsAPIError() bool {
	return true
}

// Client implements genai.ProviderGen and genai.ProviderModel.
type Client struct {
	base.ProviderGen[*ErrorResponse, *ChatRequest, *ChatResponse, ChatStreamChunkResponse]
}

// New creates a new client to talk to the Cohere platform API.
//
// If opts.APIKey is not provided, it tries to load it from the COHERE_API_KEY environment variable.
// If none is found, it will still return a client coupled with an base.ErrAPIKeyRequired error.
// Get your API key at https://dashboard.cohere.com/api-keys
//
// Use one of the model from https://cohere.com/pricing and https://docs.cohere.com/v2/docs/models
// To use multiple models, create multiple clients.
//
// wrapper optionally wraps the HTTP transport. Useful for HTTP recording and playback, or to tweak HTTP
// retries, or to throttle outgoing requests.
//
// # Tool use
//
// Tool use requires the use a model that supports structured output.
// https://docs.cohere.com/v2/docs/structured-outputs
func New(opts *genai.OptionsProvider, wrapper func(http.RoundTripper) http.RoundTripper) (*Client, error) {
	if opts.AccountID != "" {
		return nil, errors.New("unexpected option AccountID")
	}
	if opts.Remote != "" {
		return nil, errors.New("unexpected option Remote")
	}
	apiKey := opts.APIKey
	const apiKeyURL = "https://dashboard.cohere.com/api-keys"
	var err error
	if apiKey == "" {
		if apiKey = os.Getenv("COHERE_API_KEY"); apiKey == "" {
			err = &base.ErrAPIKeyRequired{EnvVar: "COHERE_API_KEY", URL: apiKeyURL}
		}
	}
	model := opts.Model
	if model == "" {
		model = base.PreferredGood
	}
	t := base.DefaultTransport
	if wrapper != nil {
		t = wrapper(t)
	}
	c := &Client{
		ProviderGen: base.ProviderGen[*ErrorResponse, *ChatRequest, *ChatResponse, ChatStreamChunkResponse]{
			Model:                model,
			GenSyncURL:           "https://api.cohere.com/v2/chat",
			ProcessStreamPackets: processStreamPackets,
			Provider: base.Provider[*ErrorResponse]{
				ProviderName: "cohere",
				APIKeyURL:    apiKeyURL,
				ClientJSON: httpjson.Client{
					Lenient: internal.BeLenient,
					Client: &http.Client{
						Transport: &roundtrippers.Header{
							Header:    http.Header{"Authorization": {"Bearer " + apiKey}},
							Transport: &roundtrippers.RequestID{Transport: t},
						},
					},
				},
			},
		},
	}
	if model == base.NoModel {
		c.Model = ""
	} else if err == nil && (model == base.PreferredCheap || model == base.PreferredGood || model == base.PreferredSOTA) {
		mdls, err2 := c.ListModels(context.Background())
		if err2 != nil {
			return nil, err2
		}
		// https://cohere.com/pricing
		// https://docs.cohere.com/v2/docs/models
		cheap := model == base.PreferredCheap
		good := model == base.PreferredGood
		c.Model = ""
		var context int64
		for _, mdl := range mdls {
			m := mdl.(*Model)
			if !slices.Contains(m.Endpoints, "chat") || strings.Contains(m.Name, "nightly") {
				continue
			}
			if cheap {
				if strings.Contains(m.Name, "light") && (context == 0 || context > m.ContextLength) {
					// For the cheapest, we want the smallest context.
					context = m.ContextLength
					c.Model = m.Name
				}
			} else if good {
				if strings.Contains(m.Name, "r7b") && (context == 0 || context < m.ContextLength) {
					// For the greatest, we want the largest context.
					context = m.ContextLength
					c.Model = m.Name
				}
			} else {
				// We want to select Command-A. We go by elimination to increase the probability of being future
				// proof.
				if !strings.HasPrefix(m.Name, "command-r") && (context == 0 || context < m.ContextLength) {
					// For the greatest, we want the largest context.
					context = m.ContextLength
					c.Model = m.Name
				}
			}
		}
		if c.Model == "" {
			return nil, errors.New("failed to find a model automatically")
		}
	}
	return c, err
}

func (c *Client) Scoreboard() genai.Scoreboard {
	return Scoreboard
}

func (c *Client) ListModels(ctx context.Context) ([]genai.Model, error) {
	// https://docs.cohere.com/reference/list-models
	return base.ListModels[*ErrorResponse, *ModelsResponse](ctx, &c.Provider, "https://api.cohere.com/v1/models?page_size=1000")
}

func processStreamPackets(ch <-chan ChatStreamChunkResponse, chunks chan<- genai.ContentFragment, result *genai.Result) error {
	defer func() {
		// We need to empty the channel to avoid blocking the goroutine.
		for range ch {
		}
	}()
	pendingCall := ToolCall{}
	for pkt := range ch {
		// These can't happen.
		if len(pkt.Delta.Message.Content) > 1 {
			return errors.New("implement multiple content")
		}
		if len(pkt.Delta.Message.ToolCalls) > 1 {
			return errors.New("implement multiple tool calls")
		}

		switch role := pkt.Delta.Message.Role; role {
		case "assistant", "":
		default:
			return fmt.Errorf("unexpected role %q", role)
		}
		f := genai.ContentFragment{}
		switch pkt.Type {
		case ChunkMessageStart:
			// Nothing useful.
			continue
		case ChunkMessageEnd:
			// Contain usage and finish reason.
			result.InputTokens = pkt.Delta.Usage.Tokens.InputTokens
			result.OutputTokens = pkt.Delta.Usage.Tokens.OutputTokens
			result.FinishReason = pkt.Delta.FinishReason.ToFinishReason()
		case ChunkContentStart:
			if len(pkt.Delta.Message.Content) != 1 {
				return fmt.Errorf("expected content %#v", pkt)
			}
			if t := pkt.Delta.Message.Content[0].Type; t != ContentText {
				return fmt.Errorf("implement content %q", t)
			}
		case ChunkContentDelta:
		case ChunkContentEnd:
			// Will be useful when there's multiple index.
		case ChunkToolPlanDelta:
			f.ThinkingFragment = pkt.Delta.Message.ToolPlan
		case ChunkToolCallStart:
			if len(pkt.Delta.Message.ToolCalls) != 1 {
				return fmt.Errorf("expected tool call %#v", pkt)
			}
			pendingCall = pkt.Delta.Message.ToolCalls[0]
		case ChunkToolCallDelta:
			pendingCall.Function.Arguments += pkt.Delta.Message.ToolCalls[0].Function.Arguments
		case ChunkToolCallEnd:
			pendingCall.To(&f.ToolCall)
			pendingCall = ToolCall{}
		case ChunkCitationStart:
			if len(pkt.Delta.Message.Citations) == 1 {
				if err := pkt.Delta.Message.Citations[0].To(&f.Citation); err != nil {
					return fmt.Errorf("mapping citations: %w", err)
				}
			} else {
				return fmt.Errorf("expected one citation, got %d", len(pkt.Delta.Message.Citations))
			}
		case ChunkCitationEnd:
		default:
			if !internal.BeLenient {
				return fmt.Errorf("unknown packet %q", pkt.Type)
			}
		}

		if len(pkt.Delta.Message.Content) == 1 {
			f.TextFragment = pkt.Delta.Message.Content[0].Text
		}
		if !f.IsZero() {
			if err := result.Accumulate(f); err != nil {
				return err
			}
			chunks <- f
		}
	}
	return nil
}

var (
	_ genai.Provider           = &Client{}
	_ genai.ProviderGen        = &Client{}
	_ genai.ProviderModel      = &Client{}
	_ genai.ProviderScoreboard = &Client{}
)
