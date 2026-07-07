// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Wire types for the Cohere chat API.
//
// Reference: https://docs.cohere.com/reference/chat

package cohere

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"sort"
	"strings"

	"github.com/maruel/genai"
	"github.com/maruel/genai/base"
	"github.com/maruel/genai/internal"
)

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
		Type       string           `json:"type,omitzero"` // "text", "json_object"
		JSONSchema genai.JSONSchema `json:"json_schema,omitzero"`
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

// SetStream sets the streaming mode.
func (c *ChatRequest) SetStream(stream bool) {
	c.Stream = stream
}

// Init initializes the provider specific completion request with the generic completion request.
func (c *ChatRequest) Init(msgs genai.Messages, model string, opts ...genai.GenOption) error {
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
		case *genai.GenOptionText:
			c.MaxTokens = v.MaxTokens
			c.Temperature = v.Temperature
			c.P = v.TopP
			sp = v.SystemPrompt
			c.K = v.TopK
			if v.TopLogprobs > 0 {
				c.Logprobs = true
			}
			c.StopSequences = v.Stop
			if v.DecodeAs != nil {
				c.ResponseFormat.Type = "json_schema"
				s, err := v.DecodeSchema()
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
	// If we have unsupported features but no other errors, return a structured error.
	if len(unsupported) > 0 && len(errs) == 0 {
		return &base.ErrNotSupported{Options: unsupported}
	}
	return errors.Join(errs...)
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
			if err := c.FromRequest(&in.Requests[i], &out); err != nil {
				return nil, fmt.Errorf("request #%d: %w", i, err)
			}
			if c.Type != "" {
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
			if err := c.FromReply(&in.Replies[i], &out); err != nil {
				return nil, fmt.Errorf("reply #%d: %w", i, err)
			}
			if c.Type != "" {
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

// Content represents a content block in a message.
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

// IsZero returns true if the content is empty.
func (c *Content) IsZero() bool {
	return c.Type == "" && c.Text == "" && c.ImageURL.URL == "" && len(c.Document.Data) == 0 && c.Document.ID == ""
}

// FromRequest converts a genai.Request to a Content.
//
// When a Document is produced, it is appended to docs.
func (c *Content) FromRequest(in *genai.Request, docs *[]Document) error {
	switch {
	case in.Text != "":
		c.Type = ContentText
		c.Text = in.Text
	case !in.Doc.IsZero():
		mimeType, data, err := in.Doc.Read(10 * 1024 * 1024)
		if err != nil {
			return err
		}
		switch {
		case (in.Doc.URL != "" && mimeType == "") || strings.HasPrefix(mimeType, "image/"):
			c.Type = ContentImageURL
			if in.Doc.URL != "" {
				c.ImageURL.URL = in.Doc.URL
			} else {
				c.ImageURL.URL = fmt.Sprintf("data:%s;base64,%s", mimeType, base64.StdEncoding.EncodeToString(data))
			}
			return nil
			// text/plain, text/markdown
		case strings.HasPrefix(mimeType, "text/"):
			if in.Doc.URL != "" {
				return fmt.Errorf("%s documents must be provided inline, not as a URL", mimeType)
			}
			name := in.Doc.GetFilename()
			docData, err := newDocumentData(name, string(data))
			if err != nil {
				return err
			}
			*docs = append(*docs, Document{
				ID:   name,
				Data: docData,
			})
			// This is handled as ChatRequest.Documents.
			return nil
		default:
			return fmt.Errorf("unsupported mime type %s", mimeType)
		}
	default:
		return errors.New("unknown Request type")
	}
	return nil
}

// FromReply converts a genai.Reply to a Content.
//
// When a Document is produced, it is appended to docs.
func (c *Content) FromReply(in *genai.Reply, docs *[]Document) error {
	switch {
	case in.Text != "":
		c.Type = ContentText
		c.Text = in.Text
	case !in.Doc.IsZero():
		mimeType, data, err := in.Doc.Read(10 * 1024 * 1024)
		if err != nil {
			return err
		}
		switch {
		case (in.Doc.URL != "" && mimeType == "") || strings.HasPrefix(mimeType, "image/"):
			c.Type = ContentImageURL
			if in.Doc.URL != "" {
				c.ImageURL.URL = in.Doc.URL
			} else {
				c.ImageURL.URL = fmt.Sprintf("data:%s;base64,%s", mimeType, base64.StdEncoding.EncodeToString(data))
			}
			return nil
			// text/plain, text/markdown
		case strings.HasPrefix(mimeType, "text/"):
			if in.Doc.URL != "" {
				return fmt.Errorf("%s documents must be provided inline, not as a URL", mimeType)
			}
			name := in.Doc.GetFilename()
			docData, err := newDocumentData(name, string(data))
			if err != nil {
				return err
			}
			*docs = append(*docs, Document{
				ID:   name,
				Data: docData,
			})
			// This is handled as ChatRequest.Documents.
			return nil
		default:
			return &internal.BadError{Err: fmt.Errorf("unsupported mime type %s", mimeType)}
		}
	case in.Reasoning != "":
		// Unclear if we should send it back.
		c.Type = ContentThinking
		c.Thinking = in.Reasoning
	default:
		return &internal.BadError{Err: errors.New("unknown Reply type")}
	}
	return nil
}

// To converts the Content to a genai.Reply.
func (c *Content) To(in *genai.Reply) error {
	switch c.Type {
	case ContentText:
		in.Text = c.Text
	case ContentThinking:
		in.Reasoning = c.Thinking
	case ContentDocument, ContentImageURL:
		return &internal.BadError{Err: fmt.Errorf("implement %s", c.Type)}
	default:
		return &internal.BadError{Err: fmt.Errorf("implement %s", c.Type)}
	}
	return nil
}

// ContentType is documented at https://docs.cohere.com/v2/reference/chat
type ContentType string

// Content type values.
const (
	ContentText     ContentType = "text"
	ContentThinking ContentType = "thinking"
	ContentImageURL ContentType = "image_url"
	ContentDocument ContentType = "document"
)

// Citations is a collection of citations.
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

// To converts Citations to genai.Reply slices.
func (c Citations) To() ([]genai.Reply, error) {
	out := make([]genai.Reply, len(c))
	for i := range c {
		if err := c[i].To(&out[i].Citation); err != nil {
			return out, err
		}
	}
	return out, nil
}

// CitationType represents the type of a citation.
type CitationType string

// Citation type values.
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

// To converts a Citation to a genai.Citation.
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
		case SourceWeb:
			cs.Type = genai.CitationWeb
			cs.URL = source.URL
			cs.Title = source.Title
			cs.Snippet = c.Text
		default:
			return &internal.BadError{Err: fmt.Errorf("implement citation source type %q", source.Type)}
		}
	}
	return nil
}

// SourceType represents the type of a citation source.
type SourceType string

// Source type values.
const (
	SourceTool     SourceType = "tool"
	SourceDocument SourceType = "document"
	SourceWeb      SourceType = "web"
)

// CitationSource represents a source in a citation.
type CitationSource struct {
	Type SourceType `json:"type,omitzero"`

	// Type == SourceTool, SourceDocument, SourceWeb
	ID string `json:"id,omitzero"`

	// Type == SourceTool
	ToolOutput map[string]json.RawMessage `json:"tool_output,omitzero"`

	// Type == SourceDocument
	Document struct {
		ID      string `json:"id,omitzero"`
		Snippet string `json:"snippet,omitzero"`
		Title   string `json:"title,omitzero"`
	} `json:"document,omitzero"`

	// Type == SourceWeb
	URL   string `json:"url,omitzero"`
	Title string `json:"title,omitzero"`
}

// Tool represents a tool definition.
type Tool struct {
	Type     string `json:"type,omitzero"` // "function"
	Function struct {
		Name        string           `json:"name,omitzero"`
		Parameters  genai.JSONSchema `json:"parameters,omitzero"`
		Description string           `json:"description,omitzero"`
	} `json:"function,omitzero"`
}

// Document can be used in the ChatRequest.Documents field or as a tool result. It's annoying because genai
// passes documents as Content inside a Message.
type Document struct {
	// Or a string.
	ID   string                     `json:"id,omitzero"`
	Data map[string]json.RawMessage `json:"data,omitzero"`
}

func newDocumentData(title, snippet string) (map[string]json.RawMessage, error) {
	titleJSON, err := json.Marshal(title)
	if err != nil {
		return nil, fmt.Errorf("marshal document title: %w", err)
	}
	snippetJSON, err := json.Marshal(snippet)
	if err != nil {
		return nil, fmt.Errorf("marshal document snippet: %w", err)
	}
	return map[string]json.RawMessage{
		"title":   titleJSON,
		"snippet": snippetJSON,
	}, nil
}

// ChatResponse is the response from the chat API.
type ChatResponse struct {
	ID           string          `json:"id"`
	FinishReason FinishReason    `json:"finish_reason"`
	Message      MessageResponse `json:"message"`
	Usage        Usage           `json:"usage"`
	Logprobs     []Logprobs      `json:"logprobs"`
}

// ToResult converts the ChatResponse to a genai.Result.
func (c *ChatResponse) ToResult() (genai.Result, error) {
	out := genai.Result{
		Usage: genai.Usage{
			// What about BilledUnits, especially for SearchUnits and Classifications?
			InputTokens:       c.Usage.Tokens.InputTokens,
			InputCachedTokens: c.Usage.CachedTokens,
			OutputTokens:      c.Usage.Tokens.OutputTokens,
			FinishReason:      c.FinishReason.ToFinishReason(),
		},
	}
	if len(c.Logprobs) != 0 {
		out.Logprobs = make([][]genai.Logprob, len(c.Logprobs))
		for i, lp := range c.Logprobs {
			out.Logprobs[i] = lp.To()
		}
	}
	// It is very frustrating that Cohere uses different message response types.
	err := c.Message.To(&out.Message)
	return out, err
}

// FinishReason represents the reason a generation finished.
type FinishReason string

// Finish reason values.
const (
	FinishComplete     FinishReason = "COMPLETE"
	FinishStopSequence FinishReason = "STOP_SEQUENCE"
	FinishMaxTokens    FinishReason = "MAX_TOKENS"
	FinishToolCall     FinishReason = "TOOL_CALL"
	FinishError        FinishReason = "ERROR"
)

// ToFinishReason converts to a genai.FinishReason.
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

// Logprobs represents log probability information.
type Logprobs struct {
	TokenIDs []int64   `json:"token_ids"`
	Text     string    `json:"text"`
	Logprobs []float64 `json:"logprobs"`
}

// To converts Logprobs to genai.Logprob slices.
func (l *Logprobs) To() []genai.Logprob {
	out := make([]genai.Logprob, 1, len(l.Logprobs)+1)
	out[0] = genai.Logprob{Text: l.Text, ID: l.TokenIDs[0], Logprob: l.Logprobs[0]}
	for i := 1; i < len(l.Logprobs); i++ {
		out = append(out, genai.Logprob{ID: l.TokenIDs[i], Logprob: l.Logprobs[i]})
	}
	return out
}

// Usage represents token usage information.
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
	CachedTokens int64 `json:"cached_tokens,omitzero"`
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

// To converts a MessageResponse to a genai.Message.
func (m *MessageResponse) To(out *genai.Message) error {
	if m.ToolCallID != "" && !internal.BeLenient {
		return errors.New("implement tool call id")
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
			replies, err := m.Citations.To()
			if err != nil {
				return fmt.Errorf("mapping citations: %w", err)
			}
			out.Replies = append(out.Replies, replies...)
		}
	}
	for i := range m.ToolCalls {
		out.Replies = append(out.Replies, genai.Reply{})
		m.ToolCalls[i].To(&out.Replies[len(out.Replies)-1].ToolCall)
	}
	return nil
}

// Contents is a slice of Content with custom unmarshalling.
type Contents []Content

// UnmarshalJSON implements json.Unmarshaler.
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
		*c = Contents([]Content{v})
	} else {
		*c = nil
	}
	return nil
}

// ToolCall represents a tool call.
type ToolCall struct {
	ID       string `json:"id"`
	Type     string `json:"type"` // function
	Function struct {
		Name      string `json:"name"`
		Arguments string `json:"arguments"`
	} `json:"function"`
}

// IsZero returns true if the tool call is empty.
func (t *ToolCall) IsZero() bool {
	return t.ID == "" && t.Type == "" && t.Function.Name == "" && t.Function.Arguments == ""
}

// From converts a genai.ToolCall to a ToolCall.
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

// To converts a ToolCall to a genai.ToolCall.
func (t *ToolCall) To(out *genai.ToolCall) {
	out.ID = t.ID
	out.Name = t.Function.Name
	out.Arguments = t.Function.Arguments
}

// ToolCalls is a slice of ToolCall with custom unmarshalling.
type ToolCalls []ToolCall

// UnmarshalJSON implements json.Unmarshaler.
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
		*t = ToolCalls([]ToolCall{tc})
	} else {
		*t = nil
	}
	return nil
}

// ChunkType represents the type of a streaming chunk.
type ChunkType string

// Chunk type values.
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

// ChatStreamChunkResponse represents a streaming chunk from the chat API.
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

// Model represents a Cohere model.
type Model struct {
	Name             string                 `json:"name"`
	Endpoints        []string               `json:"endpoints"` // chat, embed, classify, summarize, rerank, rate, generate
	Features         []string               `json:"features"`  // json_mode, json_schema, safety_modes, strict_tools, tools
	Finetuned        bool                   `json:"finetuned"`
	ContextLength    int64                  `json:"context_length"`
	TokenizerURL     string                 `json:"tokenizer_url"`
	SupportsVision   bool                   `json:"supports_vision"`
	DefaultEndpoints []string               `json:"default_endpoints"`
	IsDeprecated     bool                   `json:"is_deprecated"`
	SamplingDefaults map[string]json.Number `json:"sampling_defaults,omitzero"`
}

// GetID returns the model ID.
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

// Context returns the context window size.
func (m *Model) Context() int64 {
	return m.ContextLength
}

// ModelsResponse represents the response structure for Cohere models listing.
type ModelsResponse struct {
	Models        []Model `json:"models"`
	NextPageToken string  `json:"next_page_token"`
}

// ToModels converts Cohere models to genai.Model interfaces.
func (r *ModelsResponse) ToModels() []genai.Model {
	models := make([]genai.Model, len(r.Models))
	for i := range r.Models {
		models[i] = &r.Models[i]
	}
	return models
}

// ErrorResponse represents an API error.
type ErrorResponse struct {
	ID        string `json:"id"`
	Message   string `json:"message"`
	RequestID string `json:"request_id"`
}

func (er *ErrorResponse) Error() string {
	return er.Message
}

// IsAPIError returns true.
func (er *ErrorResponse) IsAPIError() bool {
	return true
}
