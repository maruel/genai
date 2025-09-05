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
	_ "embed"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"iter"
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
	"github.com/maruel/genai/scoreboard"
	"github.com/maruel/roundtrippers"
)

//go:embed scoreboard.json
var scoreboardJSON []byte

// Scoreboard for Cohere.
func Scoreboard() scoreboard.Score {
	var s scoreboard.Score
	d := json.NewDecoder(bytes.NewReader(scoreboardJSON))
	d.DisallowUnknownFields()
	if err := d.Decode(&s); err != nil {
		panic(fmt.Errorf("failed to unmarshal scoreboard.json: %w", err))
	}
	return s
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
		switch v := opt.(type) {
		case *genai.OptionsText:
			c.MaxTokens = v.MaxTokens
			c.Temperature = v.Temperature
			c.P = v.TopP
			sp = v.SystemPrompt
			c.Seed = v.Seed
			c.K = v.TopK
			if v.TopLogprobs > 0 {
				c.Logprobs = true
			}
			c.StopSequences = v.Stop
			if v.DecodeAs != nil {
				c.ResponseFormat.Type = "json_schema"
				c.ResponseFormat.JSONSchema = internal.JSONSchemaFor(reflect.TypeOf(v.DecodeAs))
			} else if v.ReplyAsJSON {
				c.ResponseFormat.Type = "json_object"
			}
		case *genai.OptionsTools:
			if len(v.Tools) != 0 {
				switch v.Force {
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
			if v.WebSearch {
				errs = append(errs, errors.New("unsupported OptionsTools.WebSearch"))
			}
		default:
			errs = append(errs, fmt.Errorf("unsupported options type %T", opt))
		}
	}

	if sp != "" {
		c.Messages = append(c.Messages, Message{Role: "system", Content: []Content{{Type: "text", Text: sp}}})
	}
	for i := range msgs {
		if len(msgs[i].ToolCallResults) > 1 {
			// Handle messages with multiple tool call results by creating multiple messages
			for j := range msgs[i].ToolCallResults {
				// Create a copy of the message with only one tool call result
				msgCopy := msgs[i]
				msgCopy.ToolCallResults = []genai.ToolCallResult{msgs[i].ToolCallResults[j]}
				var newMsg Message
				if d, err := newMsg.From(&msgCopy); err != nil {
					errs = append(errs, fmt.Errorf("message %d, tool result %d: %w", i, j, err))
				} else if c.Messages = append(c.Messages, newMsg); len(d) != 0 {
					c.Documents = append(c.Documents, d...)
				}
			}
		} else {
			var newMsg Message
			if d, err := newMsg.From(&msgs[i]); err != nil {
				errs = append(errs, fmt.Errorf("message %d: %w", i, err))
			} else {
				if c.Messages = append(c.Messages, newMsg); len(d) != 0 {
					c.Documents = append(c.Documents, d...)
				}
				if len(newMsg.Content) == 0 && len(newMsg.ToolCalls) == 0 && len(msgs[i].ToolCallResults) == 0 {
					errs = append(errs, fmt.Errorf("message %d: must have at least one content or tool call block", i))
				}
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

// From must be called with at most one ToolCallResults.
func (m *Message) From(in *genai.Message) ([]Document, error) {
	if len(in.ToolCallResults) > 1 {
		return nil, errors.New("internal error")
	}
	switch r := in.Role(); r {
	case "user", "assistant":
		m.Role = r
	case "computer":
		m.Role = "tool"
	default:
		return nil, fmt.Errorf("unsupported role %q", r)
	}
	var out []Document
	if len(in.Requests) != 0 {
		for i := range in.Requests {
			c := Content{}
			d, err := c.FromRequest(&in.Requests[i])
			if err != nil {
				return nil, fmt.Errorf("request #%d: %w", i, err)
			}
			if d != nil {
				out = append(out, *d)
			} else {
				m.Content = append(m.Content, c)
			}
		}
	}
	if len(in.Replies) != 0 {
		for i := range in.Replies {
			if len(in.Replies[i].Opaque) != 0 {
				return nil, fmt.Errorf("reply #%d: field Reply.Opaque not supported", i)
			}
			if in.Replies[i].Reasoning != "" {
				// Silently ignore thinking blocks.
				continue
			}
			if !in.Replies[i].ToolCall.IsZero() {
				t := ToolCall{}
				if err := t.From(&in.Replies[i].ToolCall); err != nil {
					return nil, fmt.Errorf("reply #%d: %w", i, err)
				}
				m.ToolCalls = append(m.ToolCalls, t)
				continue
			}
			c := Content{}
			d, err := c.FromReply(&in.Replies[i])
			if err != nil {
				return nil, fmt.Errorf("reply #%d: %w", i, err)
			}
			if d != nil {
				out = append(out, *d)
			} else {
				m.Content = append(m.Content, c)
			}
		}
	}
	if len(in.ToolCallResults) != 0 {
		// Process only the first tool call result in this method.
		// The Init method handles multiple tool call results by creating multiple messages.
		// Cohere supports Document, but only when using tools.
		m.ToolCallID = in.ToolCallResults[0].ID
		m.Content = []Content{{Type: ContentText, Text: in.ToolCallResults[0].Result}}
	}
	return out, nil
}

type Content struct {
	Type     ContentType `json:"type,omitzero"`
	Text     string      `json:"text,omitzero"`
	Thinking string      `json:"thinking,omitzero"`

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

func (c *Content) FromRequest(in *genai.Request) (*Document, error) {
	if in.Text != "" {
		c.Type = ContentText
		c.Text = in.Text
	} else if !in.Doc.IsZero() {
		mimeType, data, err := in.Doc.Read(10 * 1024 * 1024)
		if err != nil {
			return nil, err
		}
		switch {
		case (in.Doc.URL != "" && mimeType == "") || strings.HasPrefix(mimeType, "image/"):
			c.Type = ContentImageURL
			if in.Doc.URL != "" {
				c.ImageURL.URL = in.Doc.URL
			} else {
				c.ImageURL.URL = fmt.Sprintf("data:%s;base64,%s", mimeType, base64.StdEncoding.EncodeToString(data))
			}
			return nil, nil
			// text/plain, text/markdown
		case strings.HasPrefix(mimeType, "text/"):
			if in.Doc.URL != "" {
				return nil, fmt.Errorf("%s documents must be provided inline, not as a URL", mimeType)
			}
			name := in.Doc.GetFilename()
			d := &Document{
				ID:   name,
				Data: map[string]any{"title": name, "snippet": string(data)},
			}
			// This is handled as ChatRequest.Documents.
			return d, nil
		default:
			return nil, fmt.Errorf("unsupported mime type %s", mimeType)
		}
	} else {
		return nil, errors.New("unknown Request type")
	}
	return nil, nil
}

func (c *Content) FromReply(in *genai.Reply) (*Document, error) {
	if in.Text != "" {
		c.Type = ContentText
		c.Text = in.Text
	} else if !in.Doc.IsZero() {
		mimeType, data, err := in.Doc.Read(10 * 1024 * 1024)
		if err != nil {
			return nil, err
		}
		switch {
		case (in.Doc.URL != "" && mimeType == "") || strings.HasPrefix(mimeType, "image/"):
			c.Type = ContentImageURL
			if in.Doc.URL != "" {
				c.ImageURL.URL = in.Doc.URL
			} else {
				c.ImageURL.URL = fmt.Sprintf("data:%s;base64,%s", mimeType, base64.StdEncoding.EncodeToString(data))
			}
			return nil, nil
			// text/plain, text/markdown
		case strings.HasPrefix(mimeType, "text/"):
			if in.Doc.URL != "" {
				return nil, fmt.Errorf("%s documents must be provided inline, not as a URL", mimeType)
			}
			name := in.Doc.GetFilename()
			d := &Document{
				ID:   name,
				Data: map[string]any{"title": name, "snippet": string(data)},
			}
			// This is handled as ChatRequest.Documents.
			return d, nil
		default:
			return nil, &internal.BadError{Err: fmt.Errorf("unsupported mime type %s", mimeType)}
		}
	} else if in.Reasoning != "" {
		// Unclear if we should send it back.
		c.Type = ContentThinking
		c.Thinking = in.Reasoning
	} else {
		return nil, &internal.BadError{Err: errors.New("unknown Reply type")}
	}
	return nil, nil
}

func (c *Content) To(in *genai.Reply) error {
	switch c.Type {
	case ContentText:
		in.Text = c.Text
	case ContentThinking:
		in.Reasoning = c.Thinking
	case ContentDocument, ContentImageURL:
		fallthrough
	default:
		return &internal.BadError{Err: fmt.Errorf("implement %s", c.Type)}
	}
	return nil
}

// ContentType is documented at https://docs.cohere.com/v2/reference/chat
type ContentType string

const (
	ContentText     ContentType = "text"
	ContentThinking ContentType = "thinking"
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

func (c Citations) To(dst *genai.Reply) error {
	dst.Citations = make([]genai.Citation, len(c))
	for i := range c {
		if err := c[i].To(&dst.Citations[i]); err != nil {
			return err
		}
	}
	return nil
}

type CitationType string

const (
	CitationTextContent CitationType = "TEXT_CONTENT"
	CitationPlan        CitationType = "PLAN"
)

// Citation is only used with messages from the LLM.
type Citation struct {
	Type         CitationType     `json:"type,omitzero"`
	Start        int64            `json:"start,omitzero"`
	End          int64            `json:"end,omitzero"`
	Text         string           `json:"text,omitzero"`
	Sources      []CitationSource `json:"sources,omitzero"`
	ContentIndex int64            `json:"content_index,omitzero"`
}

func (c *Citation) To(out *genai.Citation) error {
	out.Sources = make([]genai.CitationSource, len(c.Sources))
	for i, source := range c.Sources {
		cs := &out.Sources[i]
		cs.ID = source.ID
		switch source.Type {
		case SourceTool:
			// Triggered in SquareRoot-2.
			cs.Type = genai.CitationTool
			b, _ := json.Marshal(source.ToolOutput)
			cs.Snippet = string(b)
		case SourceDocument:
			// Triggered in Citations-text-plain.
			cs.Type = genai.CitationDocument
			cs.ID = source.Document.ID
			cs.Title = source.Document.Title
			// The snippet is essentially the whole text. This is not ideal.
			// cs.Snippet = source.Document.Snippet
			cs.Snippet = c.Text
			// This is problematic because the citation is from the document but it doesn't seem to be working. If
			// someone take the time to figure it out, please send a PR.
			cs.StartCharIndex = c.Start
			cs.EndCharIndex = c.End
		default:
			return &internal.BadError{Err: fmt.Errorf("implement citation source type %q", source.Type)}
		}
	}
	return nil
}

type SourceType string

const (
	SourceTool     SourceType = "tool"
	SourceDocument SourceType = "document"
)

type CitationSource struct {
	Type SourceType `json:"type,omitzero"`

	// Type == SourceTool, SourceDocument
	ID string `json:"id,omitzero"`

	// Type == SourceTool
	ToolOutput map[string]any `json:"tool_output,omitzero"`

	// Type == SourceDocument
	Document struct {
		ID      string `json:"id,omitzero"`
		Snippet string `json:"snippet,omitzero"`
		Title   string `json:"title,omitzero"`
	} `json:"document,omitzero"`
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
	Logprobs     []Logprobs      `json:"logprobs"`
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
	if len(c.Logprobs) != 0 {
		out.Logprobs = make([]genai.Logprobs, len(c.Logprobs))
		for i, lp := range c.Logprobs {
			out.Logprobs[i] = lp.To()
		}
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
		return "Error"
	default:
		if !internal.BeLenient {
			panic(f)
		}
		return genai.FinishReason(strings.ToLower(string(f)))
	}
}

type Logprobs struct {
	TokenIDs []int64   `json:"token_ids"`
	Text     string    `json:"text"`
	Logprobs []float64 `json:"logprobs"`
}

func (l *Logprobs) To() genai.Logprobs {
	out := genai.Logprobs{Text: l.Text, Logprob: l.Logprobs[0], TopLogprobs: make([]genai.TopLogprob, len(l.Logprobs))}
	for i := range l.Logprobs {
		out.TopLogprobs[i] = genai.TopLogprob{ID: l.TokenIDs[i], Logprob: l.Logprobs[i]}
	}
	return out
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
	if m.ToolCallID != "" && !internal.BeLenient {
		return fmt.Errorf("implement tool call id")
	}
	if m.ToolPlan != "" {
		out.Replies = []genai.Reply{{Reasoning: m.ToolPlan}}
	}
	if len(m.Content) != 0 {
		for i := range m.Content {
			out.Replies = append(out.Replies, genai.Reply{})
			if err := m.Content[len(m.Content)-1].To(&out.Replies[len(out.Replies)-1]); err != nil {
				return fmt.Errorf("reply %d: %w", i, err)
			}
		}
		if len(m.Citations) != 0 {
			for i := range out.Replies {
				if out.Replies[i].Text != "" {
					if err := m.Citations.To(&out.Replies[i]); err != nil {
						return fmt.Errorf("mapping citations: %w", err)
					}
					// TODO: handle multiple citations.
					break
				}
			}
		}
	}
	for i := range m.ToolCalls {
		out.Replies = append(out.Replies, genai.Reply{})
		m.ToolCalls[i].To(&out.Replies[len(out.Replies)-1].ToolCall)
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

type ChatStreamChunkResponse struct {
	Type  ChunkType `json:"type"`
	ID    string    `json:"id"`
	Index int64     `json:"index"`
	Delta struct {
		Message      MessageResponse `json:"message"`
		FinishReason FinishReason    `json:"finish_reason"`
		Usage        Usage           `json:"usage"`
		Error        string          `json:"error"`
	} `json:"delta"`
	Logprobs Logprobs `json:"logprobs"`
}

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
	if m.ContextLength == 0 {
		return fmt.Sprintf("%s: %s%s%s", m.Name, strings.Join(endpoints, "/"), f, suffix)
	}
	return fmt.Sprintf("%s: %s%s Context: %d%s", m.Name, strings.Join(endpoints, "/"), f, m.ContextLength, suffix)
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

// Client implements genai.Provider.
type Client struct {
	impl base.Provider[*ErrorResponse, *ChatRequest, *ChatResponse, ChatStreamChunkResponse]
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
func New(ctx context.Context, opts *genai.ProviderOptions, wrapper func(http.RoundTripper) http.RoundTripper) (*Client, error) {
	if err := opts.Validate(); err != nil {
		return nil, err
	}
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
			GenSyncURL:           "https://api.cohere.com/v2/chat",
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
	good := preference == genai.ModelGood || preference == ""
	selectedModel := ""
	var context int64
	for _, mdl := range mdls {
		m := mdl.(*Model)
		if !slices.Contains(m.Endpoints, "chat") || strings.Contains(m.Name, "nightly") {
			continue
		}
		if !slices.Contains(m.Features, "strict_tools") {
			continue
		}
		if cheap {
			if strings.Contains(m.Name, "r7b") && !strings.Contains(m.Name, "arabic") && (context == 0 || context < m.ContextLength) {
				context = m.ContextLength
				selectedModel = m.Name
			}
		} else if good {
			if !strings.Contains(m.Name, "r7b") && !strings.Contains(m.Name, "reasoning") && !strings.Contains(m.Name, "plus") && (context == 0 || context < m.ContextLength) {
				// For the greatest, we want the largest context.
				context = m.ContextLength
				selectedModel = m.Name
			}
		} else {
			if strings.Contains(m.Name, "reasoning") && (context == 0 || context < m.ContextLength) {
				// For the greatest, we want the largest context.
				context = m.ContextLength
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
	return "cohere"
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
func (c *Client) GenStreamRaw(ctx context.Context, in *ChatRequest) (iter.Seq[ChatStreamChunkResponse], func() error) {
	return c.impl.GenStreamRaw(ctx, in)
}

// ListModels implements genai.Provider.
func (c *Client) ListModels(ctx context.Context) ([]genai.Model, error) {
	if c.impl.PreloadedModels != nil {
		return c.impl.PreloadedModels, nil
	}
	// https://docs.cohere.com/reference/list-models
	var resp ModelsResponse
	if err := c.impl.DoRequest(ctx, "GET", "https://api.cohere.com/v1/models?page_size=1000", nil, &resp); err != nil {
		return nil, err
	}
	return resp.ToModels(), nil
}

func processStreamPackets(chunks iter.Seq[ChatStreamChunkResponse]) (iter.Seq[genai.ReplyFragment], func() (genai.Usage, []genai.Logprobs, error)) {
	var finalErr error
	u := genai.Usage{}
	var l []genai.Logprobs

	return func(yield func(genai.ReplyFragment) bool) {
			wasThinking := false
			pendingToolCall := ToolCall{}
			for pkt := range chunks {
				// These can't happen.
				if len(pkt.Delta.Message.Content) > 1 {
					finalErr = &internal.BadError{Err: errors.New("implement multiple content")}
					return
				}
				if len(pkt.Delta.Message.ToolCalls) > 1 {
					finalErr = &internal.BadError{Err: errors.New("implement multiple tool calls")}
					return
				}
				switch role := pkt.Delta.Message.Role; role {
				case "assistant", "":
				default:
					finalErr = &internal.BadError{Err: fmt.Errorf("unexpected role %q", role)}
					return
				}
				if pkt.Logprobs.Text != "" {
					l = append(l, pkt.Logprobs.To())
				}
				f := genai.ReplyFragment{}
				switch pkt.Type {
				case ChunkMessageStart:
					// Nothing useful.
					if len(pkt.Delta.Message.Content) != 0 {
						finalErr = &internal.BadError{Err: fmt.Errorf("expected no content %#v", pkt)}
						return
					}
				case ChunkMessageEnd:
					// Contain usage and finish reason.
					if pkt.Delta.FinishReason == FinishError {
						finalErr = errors.New(pkt.Delta.Error)
						return
					}
					if len(pkt.Delta.Message.Content) != 0 {
						finalErr = &internal.BadError{Err: fmt.Errorf("expected no content %#v", pkt)}
						return
					}
					u.InputTokens = pkt.Delta.Usage.Tokens.InputTokens
					u.OutputTokens = pkt.Delta.Usage.Tokens.OutputTokens
					u.FinishReason = pkt.Delta.FinishReason.ToFinishReason()
				case ChunkContentStart:
					if len(pkt.Delta.Message.Content) != 1 {
						finalErr = &internal.BadError{Err: fmt.Errorf("expected content %#v", pkt)}
						return
					}
					switch t := pkt.Delta.Message.Content[0].Type; t {
					case ContentText:
						f.TextFragment = pkt.Delta.Message.Content[0].Text
						wasThinking = false
					case ContentThinking:
						f.ReasoningFragment = pkt.Delta.Message.Content[0].Text
						wasThinking = true
					case ContentDocument, ContentImageURL:
						fallthrough
					default:
						finalErr = &internal.BadError{Err: fmt.Errorf("implement content %q", t)}
						return
					}
				case ChunkContentDelta:
					// Sometimes the delta is empty?
					if len(pkt.Delta.Message.Content) == 1 {
						t := pkt.Delta.Message.Content[0].Text
						if t == "" {
							finalErr = &internal.BadError{Err: fmt.Errorf("empty content %#v", pkt)}
							return
						}
						// Type is not set. We need to remember the value from ChunkContentStart.
						if wasThinking {
							f.ReasoningFragment = t
						} else {
							f.TextFragment = t
						}
					}
				case ChunkContentEnd:
					// Will be useful when there's multiple index.
					// Sometimes the delta is empty?
					if len(pkt.Delta.Message.Content) != 0 {
						finalErr = &internal.BadError{Err: fmt.Errorf("unexpected content %#v", pkt)}
						return
					}
				case ChunkToolPlanDelta:
					f.ReasoningFragment = pkt.Delta.Message.ToolPlan
				case ChunkToolCallStart:
					if len(pkt.Delta.Message.ToolCalls) != 1 {
						finalErr = &internal.BadError{Err: fmt.Errorf("expected tool call %#v", pkt)}
						return
					}
					pendingToolCall = pkt.Delta.Message.ToolCalls[0]
				case ChunkToolCallDelta:
					pendingToolCall.Function.Arguments += pkt.Delta.Message.ToolCalls[0].Function.Arguments
				case ChunkToolCallEnd:
					pendingToolCall.To(&f.ToolCall)
					pendingToolCall = ToolCall{}
				case ChunkCitationStart:
					if len(pkt.Delta.Message.Citations) != 1 {
						finalErr = &internal.BadError{Err: fmt.Errorf("expected one citation, got %v", pkt)}
						return
					}
					if err := pkt.Delta.Message.Citations[0].To(&f.Citation); err != nil {
						finalErr = err
						return
					}
				case ChunkCitationEnd:
					if len(pkt.Delta.Message.Citations) != 0 {
						finalErr = &internal.BadError{Err: fmt.Errorf("expected no citations, got %v", pkt)}
						return
					}
				default:
					if !internal.BeLenient {
						finalErr = &internal.BadError{Err: fmt.Errorf("unknown packet %q", pkt.Type)}
						return
					}
				}
				if !yield(f) {
					return
				}
			}
		}, func() (genai.Usage, []genai.Logprobs, error) {
			return u, l, finalErr
		}
}

var _ genai.Provider = &Client{}
