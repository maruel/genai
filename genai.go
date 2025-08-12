// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Package genai is the high performance native Go client for LLMs.
//
// It provides a generic interface to interact with various LLM providers.
//
// Check out the examples for a quick start.
package genai

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"maps"
	"path"
	"path/filepath"
	"reflect"
	"strconv"
	"strings"
	"time"

	"github.com/maruel/genai/internal"
	"github.com/maruel/genai/internal/bb"
)

// Provider

// Provider is the base interface that all provider interfaces embed.
type Provider interface {
	// Name returns the name of the provider.
	Name() string
	// ModelID returns the model currently used by the provider. It can be an empty string.
	ModelID() string
}

// ProviderUnwrap is exposed when the Provider is actually a wrapper around another one, like
// ProviderGenThinking or ProviderGenUsage. This is useful when looking for other interfaces.
type ProviderUnwrap interface {
	Unwrap() Provider
}

// Generation

// ProviderGen is the generic interface to interact with a LLM backend.
type ProviderGen interface {
	Provider
	// GenSync runs generation synchronously.
	//
	// opts can be nil, in this case OptionsText is assumed. It can also be other modalities like *OptionsImage,
	// *OptionsText or a provider-specialized option struct.
	GenSync(ctx context.Context, msgs Messages, opts Options) (Result, error)
	// GenStream runs generation synchronously, streaming the results to channel replies.
	//
	// No need to accumulate the replies into the result, the Result contains the accumulated message.
	GenStream(ctx context.Context, msgs Messages, replies chan<- ContentFragment, opts Options) (Result, error)
}

// Result is the result of a completion.
type Result struct {
	Message
	Usage
	Logprobs []Logprobs
}

// Logprobs represents the log probability information for a single token.
type Logprobs struct {
	ID          int64        `json:"id,omitempty"`           // Input token ID. Not always provided.
	Text        string       `json:"text,omitempty"`         // Text in UTF-8.
	Bytes       []byte       `json:"bytes,omitempty"`        // Bytes representation of the text, in case it's not valid UTF-8. Not always provided.
	Logprob     float64      `json:"logprob"`                // The log probability of the token
	TopLogprobs []TopLogprob `json:"top_logprobs,omitempty"` // Top candidates.
}

// TopLogprob represents the log probability information for a top token.
type TopLogprob struct {
	ID      int64   `json:"id,omitempty"`
	Text    string  `json:"text,omitempty"`
	Bytes   []byte  `json:"bytes,omitempty"`
	Logprob float64 `json:"logprob"`
}

// Usage from the LLM provider.
type Usage struct {
	InputTokens       int64
	InputCachedTokens int64
	ReasoningTokens   int64
	OutputTokens      int64
	TotalTokens       int64

	// FinishReason indicates why the model stopped generating tokens.
	FinishReason FinishReason

	// Limits contains a list of rate limit details from the provider.
	Limits []RateLimit
}

func (u *Usage) String() string {
	var s strings.Builder
	fmt.Fprintf(&s, "in: %d (cached %d), reasoning: %d, out: %d, total: %d",
		u.InputTokens, u.InputCachedTokens, u.ReasoningTokens, u.OutputTokens, u.TotalTokens)
	for _, l := range u.Limits {
		fmt.Fprintf(&s, ", %s", l.String())
	}
	return s.String()
}

// Add adds the usage from another result.
func (u *Usage) Add(r Usage) {
	u.InputTokens += r.InputTokens
	u.InputCachedTokens += r.InputCachedTokens
	u.ReasoningTokens += r.ReasoningTokens
	u.OutputTokens += r.OutputTokens
	u.TotalTokens += r.TotalTokens
}

// RateLimitType defines the type of rate limit.
type RateLimitType string

const (
	Requests RateLimitType = "requests"
	Tokens   RateLimitType = "tokens"
)

// RateLimitPeriod defines the time period for a rate limit.
type RateLimitPeriod string

const (
	PerMinute RateLimitPeriod = "minute"
	PerDay    RateLimitPeriod = "day"
	PerMonth  RateLimitPeriod = "month"
	PerOther  RateLimitPeriod = "other" // For non-standard periods
)

// RateLimit contains the limit, remaining, and reset values for a metric.
type RateLimit struct {
	Type      RateLimitType
	Period    RateLimitPeriod
	Limit     int64
	Remaining int64
	Reset     time.Time
}

func (r *RateLimit) String() string {
	if r.Period == PerOther {
		if r.Reset.IsZero() {
			return fmt.Sprintf("%s: %d/%d", r.Type, r.Remaining, r.Limit)
		}
		return fmt.Sprintf("%s/%s: %d/%d", r.Type, r.Reset, r.Remaining, r.Limit)
	}
	if r.Reset.IsZero() {
		return fmt.Sprintf("%s (%s): %d/%d", r.Type, r.Period, r.Remaining, r.Limit)
	}
	return fmt.Sprintf("%s/%s (%s): %d/%d", r.Type, r.Reset, r.Period, r.Remaining, r.Limit)
}

func (r *RateLimit) Validate() error {
	switch r.Type {
	case Requests, Tokens:
	default:
		return fmt.Errorf("unknown limit type %q", r.Type)
	}
	if r.Limit == 0 {
		return errors.New("limit is 0")
	}
	// It is valid for Remaining to be zero. It's rare but it happens.
	switch r.Period {
	case PerMinute, PerDay, PerMonth, PerOther:
	default:
		return fmt.Errorf("unknown limit period %q", r.Period)
	}
	if r.Reset.IsZero() {
		return errors.New("reset is 0")
	}
	return nil
}

// FinishReason is the reason why the model stopped generating tokens.
//
// It can be one of the well known below or a custom value.
type FinishReason string

const (
	// FinishedStop means the assistant was done for the turn. Some providers confuse it with
	// FinishedStopSequence.
	FinishedStop FinishReason = "stop"
	// FinishedLength means the model reached the maximum number of tokens allowed as set in
	// OptionsText.MaxTokens or as limited by the provider.
	FinishedLength FinishReason = "length"
	// FinishedToolCalls means the model called one or multiple tools and needs the replies to continue the turn.
	FinishedToolCalls FinishReason = "tool_calls"
	// FinishedStopSequence means the model stopped because it saw a stop word as listed in OptionsText.Stop.
	FinishedStopSequence FinishReason = "stop"
	// FinishedContentFilter means the model stopped because the reply got caught by a content filter.
	FinishedContentFilter FinishReason = "content_filter"
	// Pending means that it's not finished yet. For use with ProviderGenAsync.
	Pending FinishReason = "pending"
)

// Messages

// Role is one of the LLM known roles.
type Role string

// LLM known roles. Not all systems support all roles.
const (
	// User is the user's inputs. There can be multiple users in a conversation.
	// They are differentiated by the Message.User field.
	User Role = "user"
	// Assistant is the LLM.
	Assistant Role = "assistant"
	// Computer is the user's computer, it replies to tool calls.
	Computer Role = "computer"
)

// Validate ensures the role is valid.
func (r Role) Validate() error {
	switch r {
	case User, Assistant, Computer:
		return nil
	default:
		return fmt.Errorf("role %q is not supported", r)
	}
}

// Messages is a list of valid messages in an exchange with a LLM.
//
// The messages should be alternating between user input, assistant replies, assistant tool cal requests and
// computer tool call results. The exception in the case of multi-user discussion, with different Users.
type Messages []Message

// Validate ensures the messages are valid.
func (m Messages) Validate() error {
	var errs []error
	for i := range m {
		if err := m[i].Validate(); err != nil {
			errs = append(errs, fmt.Errorf("message %d: %w", i, err))
		} else if m[i].IsZero() {
			errs = append(errs, fmt.Errorf("message %d: is empty", i))
		}
		if i > 0 && m[i-1].Role == m[i].Role && (m[i].Role != User || m[i-1].User != m[i].User) {
			errs = append(errs, fmt.Errorf("message %d: role must alternate", i))
		}
	}
	return errors.Join(errs...)
}

// Message is a message to send to the LLM as part of the exchange.
//
// The message may contain content, information to communicate between the user
// and the LLM. This is the Contents section. The content can be text or a
// document. The document may be audio, video, image, PDF or any other format.
//
// The message may also contain tool calls. The tool call is a request from
// the LLM to answer a specific question, so the LLM can continue its process.
type Message struct {
	Role Role `json:"role,omitzero"`
	// User must only be used when Role == User. Only some provider (e.g. OpenAI, Groq, DeepSeek) support it.
	User string `json:"user,omitzero"`

	// Contents slice is generally one item in text-only mode. It is more frequently multiple items when using
	// multi-modal content or with advanced LLM providers that can emit multiple content blocks like a code
	// block and a different block with an explanantion.
	Contents []Content `json:"contents,omitzero"`

	// ToolCall is a tool call that the LLM requested to make.
	ToolCalls       []ToolCall       `json:"tool_calls,omitzero"`
	ToolCallResults []ToolCallResult `json:"tool_call_results,omitzero"`

	_ struct{}
}

// NewTextMessage is a shorthand function to create a Message with a single
// text block.
func NewTextMessage(text string) Message {
	return Message{Role: User, Contents: []Content{{Text: text}}}
}

func (m *Message) IsZero() bool {
	return m.Role == "" && m.User == "" && len(m.Contents) == 0 && len(m.ToolCalls) == 0 && len(m.ToolCallResults) == 0
}

// Validate ensures the messages are valid.
func (m *Message) Validate() error {
	errs := m.validate()
	if len(m.Contents) == 0 && len(m.ToolCalls) == 0 && len(m.ToolCallResults) == 0 {
		errs = append(errs, errors.New("at least one of fields Contents, ToolCalls or ToolCallsResults is required"))
	}
	for i, b := range m.Contents {
		if err := b.Validate(); err != nil {
			errs = append(errs, fmt.Errorf("content %d: %w", i, err))
		}
	}
	if len(m.ToolCalls) != 0 && m.Role != Assistant {
		errs = append(errs, errors.New("only role assistant can call tools"))
	}
	for i, b := range m.ToolCalls {
		if err := b.Validate(); err != nil {
			errs = append(errs, fmt.Errorf("tool call %d: %w", i, err))
		}
	}
	if len(m.ToolCallResults) != 0 && m.Role != User {
		errs = append(errs, errors.New("only role user can provide tool call results"))
	}
	for i, b := range m.ToolCallResults {
		if err := b.Validate(); err != nil {
			errs = append(errs, fmt.Errorf("tool result %d: %w", i, err))
		}
	}
	return errors.Join(errs...)
}

func (m *Message) validate() []error {
	var errs []error
	if err := m.Role.Validate(); err != nil {
		errs = append(errs, fmt.Errorf("field Role: %w", err))
	}
	if m.User != "" {
		errs = append(errs, errors.New("field User: not supported yet"))
	}
	return errs
}

// AsText is a short hand to get the content as text.
//
// It ignores Thinking or multi-modal content.
func (m *Message) AsText() string {
	var data [16]string
	out := data[:0]
	for i := range m.Contents {
		if s := m.Contents[i].Text; s != "" {
			out = append(out, s)
		}
	}
	return strings.Join(out, "")
}

// Decode decodes the JSON message into the struct.
//
// Requires using either ReplyAsJSON or DecodeAs in the OptionsText.
//
// Note: this doesn't verify the type is the same as specified in
// OptionsText.DecodeAs.
func (m *Message) Decode(x any) error {
	s := m.AsText()
	if s == "" {
		return fmt.Errorf("only text messages can be decoded as JSON, can't decode %#v", m)
	}
	d := json.NewDecoder(strings.NewReader(s))
	d.DisallowUnknownFields()
	d.UseNumber()
	if err := d.Decode(x); err != nil {
		return fmt.Errorf("failed to decode message text as JSON: %w; content: %q", err, s)
	}
	return nil
}

// DoToolCalls processes all the ToolCalls if any.
//
// Returns a Message to be added back to the list of messages, only if msg.IsZero() is true.
func (m *Message) DoToolCalls(ctx context.Context, tools []ToolDef) (Message, error) {
	var out Message
	for i := range m.ToolCalls {
		res, err := m.ToolCalls[i].Call(ctx, tools)
		if err != nil {
			return out, err
		}
		// Set User as the role. Some provider will use "tool".
		out.Role = User
		out.ToolCallResults = append(out.ToolCallResults, ToolCallResult{
			ID:     m.ToolCalls[i].ID,
			Name:   m.ToolCalls[i].Name,
			Result: res,
		})
	}
	return out, nil
}

// UnmarshalJSON adds validation during decoding.
func (m *Message) UnmarshalJSON(b []byte) error {
	type Alias Message
	a := struct{ *Alias }{Alias: (*Alias)(m)}
	d := json.NewDecoder(bytes.NewReader(b))
	d.DisallowUnknownFields()
	if err := d.Decode(&a); err != nil {
		return err
	}
	return errors.Join(m.validate()...)
}

func (m *Message) GoString() string {
	b, _ := json.Marshal(m)
	return string(b)
}

// Content is a block of content in the message meant to be visible in a
// chat setting.
//
// The content can be text or a document. The document may be audio, video,
// image, PDF or any other format.
//
// Only Text, Thinking, Opaque or the rest can be set.
//
// If Text and Thinking are not set, then, one of Document or URL must be set.
type Content struct {
	// Text is the content of the text message.
	Text string `json:"text,omitzero"`

	// Thinking is the reasoning done by the LLM.
	Thinking string `json:"thinking,omitzero"`

	// Citations contains references to source material that support the content.
	// Only valid when Text is set and the provider supports citations.
	Citations []Citation `json:"citations,omitzero"`

	// Opaque is added to keep continuity on the processing. A good example is Anthropic's extended thinking. It
	// must be kept during an exchange.
	//
	// A message with only Opaque set is valid.
	Opaque map[string]any `json:"opaque,omitzero"`

	// Filename is the name of the file. For many providers, only the extension
	// is relevant. They only use mime-type, which is derived from the filename's
	// extension. When an URL is provided or when the object provided to Document
	// implements a method with the signature `Name() string`, like an
	// `*os.File`, Filename is optional.
	Filename string `json:"filename,omitzero"`
	// Document is raw document data. It is perfectly fine to use a bytes.NewReader() or *os.File.
	Document io.ReadSeeker `json:"document,omitzero"`
	// URL is the reference to the raw data. When set, the mime-type is derived from the URL.
	URL string `json:"url,omitzero"`

	_ struct{}
}

// Validate ensures the block is valid.
func (c *Content) Validate() error {
	if c.Text != "" {
		if c.Thinking != "" {
			return errors.New("field Thinking can't be used along Text")
		}
		if len(c.Opaque) != 0 {
			return errors.New("field Opaque can't be used along Text")
		}
		if c.Filename != "" {
			return errors.New("field Filename can't be used along Text")
		}
		if c.Document != nil {
			return errors.New("field Document can't be used along Text")
		}
		if c.URL != "" {
			return errors.New("field URL can't be used along Text")
		}
		// Validate citations when text is present
		for i, citation := range c.Citations {
			if err := citation.Validate(); err != nil {
				return fmt.Errorf("citation %d: %w", i, err)
			}
		}
	} else if c.Thinking != "" || len(c.Opaque) != 0 {
		if len(c.Citations) != 0 {
			return errors.New("field Citations can only be used with Text")
		}
		if c.Filename != "" {
			return errors.New("field Filename can't be used along Text")
		}
		if c.Document != nil {
			return errors.New("field Document can't be used along Text")
		}
		if c.URL != "" {
			return errors.New("field URL can't be used along Text")
		}
	} else {
		if len(c.Citations) != 0 {
			return errors.New("field Citations can only be used with Text")
		}
		if len(c.Opaque) != 0 {
			return errors.New("field Opaque can't be used along a document")
		}
		if c.Document == nil {
			if c.URL == "" {
				if c.Filename == "" {
					return errors.New("no content")
				}
				return errors.New("field Document or URL is required when using Filename")
			}
		} else {
			if c.URL != "" {
				return errors.New("field Document and URL are mutually exclusive")
			}
			if c.GetFilename() == "" {
				return errors.New("field Filename is required with Document when not implementing Name()")
			}
		}
	}
	return nil
}

// GetFilename returns the filename to use for the document, querying the
// Document's name if available.
func (c *Content) GetFilename() string {
	if c.Filename == "" {
		if namer, ok := c.Document.(interface{ Name() string }); ok {
			return filepath.Base(namer.Name())
		}
	}
	return c.Filename
}

// ReadDocument reads the document content into memory.
func (c *Content) ReadDocument(maxSize int64) (string, []byte, error) {
	if c.Text != "" {
		return "", nil, errors.New("only document messages can be read as documents")
	}
	// genai cannot depend on base as it would cause a circular import.
	mimeType := internal.MimeByExt(filepath.Ext(c.GetFilename()))
	if c.URL != "" {
		// Not all provider require a mime-type so do not error out.
		if mimeType == "" {
			mimeType = internal.MimeByExt(filepath.Ext(path.Base(c.URL)))
		}
		return mimeType, nil, nil
	}
	if mimeType == "" {
		return "", nil, errors.New("failed to determine mime-type, pass a filename with an extension")
	}
	size, err := c.Document.Seek(0, io.SeekEnd)
	if err != nil {
		return "", nil, fmt.Errorf("failed to seek data: %w", err)
	}
	if size > maxSize {
		return "", nil, fmt.Errorf("large files are not yet supported, max %dMiB", maxSize/1024/1024)
	}
	if _, err = c.Document.Seek(0, io.SeekStart); err != nil {
		return "", nil, fmt.Errorf("failed to seek data: %w", err)
	}
	var data []byte
	if data, err = io.ReadAll(c.Document); err != nil {
		return "", nil, fmt.Errorf("failed to read data: %w", err)
	}
	if len(data) == 0 {
		return "", nil, errors.New("empty data")
	}
	return mimeType, data, nil
}

type contentSerialized struct {
	Text     string         `json:"text,omitzero"`
	Thinking string         `json:"thinking,omitzero"`
	Opaque   map[string]any `json:"opaque,omitzero"`
	Filename string         `json:"filename,omitzero"`
	Document []byte         `json:"document,omitzero"`
	URL      string         `json:"url,omitzero"`
}

func (c *Content) MarshalJSON() ([]byte, error) {
	cc := contentSerialized{
		Text:     c.Text,
		Thinking: c.Thinking,
		Opaque:   c.Opaque,
		Filename: c.Filename,
		URL:      c.URL,
	}
	if c.Document != nil {
		if _, err := c.Document.Seek(0, io.SeekStart); err != nil {
			return nil, err
		}
		var err error
		if cc.Document, err = io.ReadAll(c.Document); err != nil {
			return nil, err
		}
	}
	return json.Marshal(&cc)
}

func (c *Content) UnmarshalJSON(b []byte) error {
	cc := contentSerialized{}
	d := json.NewDecoder(bytes.NewReader(b))
	d.DisallowUnknownFields()
	if err := d.Decode(&cc); err != nil {
		return err
	}
	c.Text = cc.Text
	c.Thinking = cc.Thinking
	c.Opaque = cc.Opaque
	c.Filename = cc.Filename
	c.URL = cc.URL
	if len(cc.Document) != 0 {
		c.Document = &bb.BytesBuffer{D: cc.Document}
	}
	return c.Validate()
}

// ContentFragment is a fragment of a content the LLM is sending back as part
// of the GenStream().
type ContentFragment struct {
	TextFragment string `json:"text,omitzero"`

	ThinkingFragment string         `json:"thinking,omitzero"`
	Opaque           map[string]any `json:"opaque,omitzero"`

	Filename         string `json:"filename,omitzero"`
	DocumentFragment []byte `json:"document,omitzero"`
	URL              string `json:"url,omitzero"`

	// ToolCall is a tool call that the LLM requested to make.
	ToolCall ToolCall `json:"tool_call,omitzero"`

	Citation Citation `json:"citation,omitzero"`

	_ struct{}
}

func (m *ContentFragment) IsZero() bool {
	return m.TextFragment == "" && m.ThinkingFragment == "" && len(m.Opaque) == 0 && m.Filename == "" && len(m.DocumentFragment) == 0 && m.URL == "" && m.ToolCall.IsZero() && m.Citation.IsZero()
}

func (m *ContentFragment) GoString() string {
	b, _ := json.Marshal(m)
	return string(b)
}

// Accumulate adds a ContentFragment to the message being streamed.
//
// The role is always implicitly the assistant.
func (m *Message) Accumulate(mf ContentFragment) error {
	if m.Role == "" {
		m.Role = Assistant
	}
	// Generally the first message fragment.
	if mf.ThinkingFragment != "" {
		if len(m.Contents) != 0 {
			if lastBlock := &m.Contents[len(m.Contents)-1]; lastBlock.Thinking != "" {
				lastBlock.Thinking += mf.ThinkingFragment
				if len(mf.Opaque) != 0 {
					if lastBlock.Opaque == nil {
						lastBlock.Opaque = map[string]any{}
					}
					maps.Copy(lastBlock.Opaque, mf.Opaque)
				}
				return nil
			}
		}
		m.Contents = append(m.Contents, Content{Thinking: mf.ThinkingFragment, Opaque: mf.Opaque})
		return nil
	}
	if len(mf.Opaque) != 0 {
		if len(m.Contents) != 0 {
			// Only add Opaque to Thinking block.
			if lastBlock := &m.Contents[len(m.Contents)-1]; lastBlock.Thinking != "" {
				if lastBlock.Opaque == nil {
					lastBlock.Opaque = map[string]any{}
				}
				maps.Copy(lastBlock.Opaque, mf.Opaque)
				return nil
			}
		}
		// Unlikely.
		m.Contents = append(m.Contents, Content{Opaque: mf.Opaque})
		return nil
	}

	// Content.
	if mf.TextFragment != "" {
		if len(m.Contents) != 0 {
			if lastBlock := &m.Contents[len(m.Contents)-1]; lastBlock.Text != "" {
				lastBlock.Text += mf.TextFragment
				return nil
			}
		}
		m.Contents = append(m.Contents, Content{Text: mf.TextFragment})
		return nil
	}

	if mf.URL != "" {
		m.Contents = append(m.Contents, Content{Filename: mf.Filename, URL: mf.URL})
		return nil
	}
	if mf.DocumentFragment != nil {
		if len(m.Contents) != 0 {
			if lastBlock := &m.Contents[len(m.Contents)-1]; lastBlock.Filename != "" || lastBlock.Document != nil {
				if lastBlock.Document == nil {
					lastBlock.Document = &bb.BytesBuffer{}
				}
				if lastBlock.Filename == "" {
					// Unlikely.
					lastBlock.Filename = mf.Filename
				}
				_, _ = lastBlock.Document.(*bb.BytesBuffer).Write(mf.DocumentFragment)
				return nil
			}
		}
		m.Contents = append(m.Contents, Content{Filename: mf.Filename, Document: &bb.BytesBuffer{D: mf.DocumentFragment}})
		return nil
	}
	if mf.Filename != "" {
		m.Contents = append(m.Contents, Content{Filename: mf.Filename})
		return nil
	}

	if mf.ToolCall.Name != "" {
		m.ToolCalls = append(m.ToolCalls, mf.ToolCall)
		return nil
	}

	if !mf.Citation.IsZero() {
		// For now always add a new block.
		m.Contents = append(m.Contents, Content{Citations: []Citation{mf.Citation}})
		return nil
	}

	// Nothing to accumulate. It should be an error but there are bugs where the system hangs.
	return nil
}

// ToolCall is a tool call that the LLM requested to make.
type ToolCall struct {
	ID        string `json:"id,omitzero"`        // Unique identifier for the tool call. Necessary for parallel tool calling.
	Name      string `json:"name,omitzero"`      // Tool being called.
	Arguments string `json:"arguments,omitzero"` // encoded as JSON

	// Opaque is added to keep continuity on the processing. A good example is Gemini's extended thinking. It
	// must be kept during an exchange.
	//
	// A message with only Opaque set is valid.
	Opaque map[string]any `json:"opaque,omitzero"`

	_ struct{}
}

func (t *ToolCall) IsZero() bool {
	return t.ID == "" && t.Name == "" && t.Arguments == ""
}

// Validate ensures the tool call request from the LLM is valid.
func (t *ToolCall) Validate() error {
	// Some provider like Gemini doesn't set an ID.
	if t.ID == "" && t.Name == "" {
		return errors.New("at least one of field ID or Name is required")
	}
	// Excessive?
	d := json.NewDecoder(strings.NewReader(t.Arguments))
	d.DisallowUnknownFields()
	d.UseNumber()
	var x any
	if err := d.Decode(&x); err != nil {
		return fmt.Errorf("field Arguments: %w", err)
	}
	return nil
}

// Call invokes the ToolDef.Callback with arguments from the ToolCall, returning the result string.
//
// It decodes the ToolCall.Arguments and passes it to the ToolDef.Callback.
func (t *ToolCall) Call(ctx context.Context, tools []ToolDef) (string, error) {
	i := 0
	for ; i < len(tools); i++ {
		if tools[i].Name == t.Name {
			break
		}
	}
	if i == len(tools) {
		return "", fmt.Errorf("failed to find tool named %q", t.Name)
	}
	// This function assumes Validate() was called on both object and that they match. Otherwise this will
	// panic.
	input := reflect.New(reflect.TypeOf(tools[i].Callback).In(1).Elem())
	d := json.NewDecoder(strings.NewReader(t.Arguments))
	d.DisallowUnknownFields()
	d.UseNumber()
	if err := d.Decode(input.Interface()); err != nil {
		return "", fmt.Errorf("failed to decode tool call arguments: %w; arguments: %q", err, t.Arguments)
	}
	res := reflect.ValueOf(tools[i].Callback).Call([]reflect.Value{reflect.ValueOf(ctx), input})
	s := res[0].String()
	if e := res[1].Interface(); e != nil {
		return s, e.(error)
	}
	return s, nil
}

func (t *ToolCall) UnmarshalJSON(data []byte) error {
	type Alias ToolCall
	a := struct{ *Alias }{Alias: (*Alias)(t)}
	if err := json.Unmarshal(data, &a); err != nil {
		return err
	}
	return t.Validate()
}

// ToolCallResult is the result for a tool call that the LLM requested to make.
type ToolCallResult struct {
	ID     string `json:"id,omitzero"`
	Name   string `json:"name,omitzero"`
	Result string `json:"result,omitzero"`

	_ struct{}
}

// Validate ensures the tool result is valid.
func (t *ToolCallResult) Validate() error {
	// Some provider like Gemini doesn't set an ID.
	if t.ID == "" && t.Name == "" {
		return errors.New("at least one of field ID or Name is required")
	}
	if t.Result == "" {
		return errors.New("field Result: required")
	}
	return nil
}

func (t *ToolCallResult) UnmarshalJSON(b []byte) error {
	type Alias ToolCallResult
	a := struct{ *Alias }{Alias: (*Alias)(t)}
	d := json.NewDecoder(bytes.NewReader(b))
	d.DisallowUnknownFields()
	if err := d.Decode(&a); err != nil {
		return err
	}
	return t.Validate()
}

// Citation represents a reference to source material that supports content.
// It provides a unified interface for different provider citation formats.
type Citation struct {
	// Text is the exact text that is being cited.
	Text string `json:"text,omitzero"`

	// StartIndex is the starting character position of the citation in the content (0-based).
	// For providers that support character-level citations.
	StartIndex int64 `json:"start_index,omitzero"`

	// EndIndex is the ending character position of the citation in the content (0-based, exclusive).
	// For providers that support character-level citations.
	EndIndex int64 `json:"end_index,omitzero"`

	// Sources contains information about the source documents or tools that support this citation.
	Sources []CitationSource `json:"sources,omitzero"`

	// Location provides additional location information (page numbers, block indices, etc.)
	// that varies by provider and source type.
	Location map[string]any `json:"location,omitzero"`

	// Type indicates the citation type (e.g., "text", "document", "tool", "web").
	// This is provider-specific and may be empty for unified citations.
	Type string `json:"type,omitzero"`

	_ struct{}
}

// Validate ensures the citation is valid.
func (c *Citation) Validate() error {
	// Text can be empty, e.g. Perplexity web search results.
	if c.StartIndex < 0 {
		return fmt.Errorf("start index must be non-negative, got %d", c.StartIndex)
	}
	if c.EndIndex < 0 {
		return fmt.Errorf("end index must be non-negative, got %d", c.EndIndex)
	}
	if c.EndIndex > 0 && c.EndIndex <= c.StartIndex {
		return fmt.Errorf("end index (%d) must be greater than start index (%d)", c.EndIndex, c.StartIndex)
	}
	for i, source := range c.Sources {
		if err := source.Validate(); err != nil {
			return fmt.Errorf("source %d: %w", i, err)
		}
	}
	return nil
}

func (c *Citation) IsZero() bool {
	return c.Text == "" && c.StartIndex == 0 && c.EndIndex == 0 && len(c.Sources) == 0 && len(c.Location) == 0 && c.Type == ""
}

// CitationSource represents a source that supports a citation.
type CitationSource struct {
	// ID is a unique identifier for the source (e.g., document ID, tool call ID).
	ID string `json:"id,omitzero"`

	// Type indicates the source type (e.g., "document", "tool", "web").
	Type string `json:"type,omitzero"`

	// Title is the human-readable title of the source.
	Title string `json:"title,omitzero"`

	// URL is the web URL for the source, if applicable.
	URL string `json:"url,omitzero"`

	// Metadata contains additional source-specific information.
	// For document sources: document index, page numbers, etc.
	// For tool sources: tool output, function name, etc.
	// For web sources: encrypted index, search result info, etc.
	Metadata map[string]any `json:"metadata,omitzero"`

	_ struct{}
}

// Validate ensures the citation source is valid.
func (cs *CitationSource) Validate() error {
	if cs.ID == "" && cs.URL == "" {
		return fmt.Errorf("citation source must have either ID or URL")
	}
	return nil
}

func (cs *CitationSource) IsZero() bool {
	return cs.ID == "" && cs.Type == "" && cs.Title == "" && cs.URL == "" && len(cs.Metadata) == 0
}

//

// ProviderGenDoc is the interface to interact with a document (audio, image, video, etc) generator.
type ProviderGenDoc interface {
	Provider
	GenDoc(ctx context.Context, msg Message, opts Options) (Result, error)
}

// ProviderGenAsync is the interface to interact with a batch generator.
type ProviderGenAsync interface {
	// GenAsync requests a generation and returns a pending job that can be polled.
	GenAsync(ctx context.Context, msgs Messages, opts Options) (Job, error)
	// PokeResult requests the state of the job.
	//
	// When the job is still pending, Result.Usage.FinishReason is Pending.
	PokeResult(ctx context.Context, job Job) (Result, error)
}

// Job is a pending job.
type Job string

//

// CacheEntry is one file (or GenSync request) cached on the provider for reuse.
type CacheEntry interface {
	GetID() string
	GetDisplayName() string
	GetExpiry() time.Time
}

// ProviderCache provides a high level way to manage files cached on the provider.
type ProviderCache interface {
	CacheAddRequest(ctx context.Context, msgs Messages, opts Options, name, displayName string, ttl time.Duration) (string, error)
	CacheList(ctx context.Context) ([]CacheEntry, error)
	CacheDelete(ctx context.Context, name string) error
}

// Models

// ProviderModel represents a provider that can list models.
type ProviderModel interface {
	Provider
	ListModels(ctx context.Context) ([]Model, error)
}

// Model represents a served model by the provider.
type Model interface {
	GetID() string
	String() string
	Context() int64
}

// Ping

// ProviderPing represents a provider that you can ping.
type ProviderPing interface {
	Provider
	// Ping enables confirming that the provider is accessible, without incurring cost. This is useful for local
	// providers to detect if they are accessible or not.
	Ping(ctx context.Context) error
}

//

// ProviderScoreboard describes the known state of the provider.
type ProviderScoreboard interface {
	Provider
	// Scoreboard returns what the provider supports.
	//
	// Some models have more features than others, e.g. some models may be text-only while others have vision or
	// audio support.
	//
	// The client code may be the limiting factor for some models, and not the provider itself.
	//
	// The values returned here are gone through a smoke test to make sure they are valid.
	Scoreboard() Scoreboard
}

// FunctionalityText defines which functionalites are supported in a scenario for models that support text
// output modality.
//
// The first group is for multi-modal models, either with non-text inputs (e.g. vision, STT) or outputs
// (combined text and image generation).
//
// The second group are supported functional features for agency.
//
// The third group is to identify bugged providers. A provider is considered to be bugged if any of the field
// is false.
type FunctionalityText struct {
	// Tools means that tool call is supported. This is a requirement for MCP. Some provider support tool
	// calling but the model is very flaky at actually requesting the calls. This is more frequent on highly
	// quantized models, small models or MoE models.
	Tools TriState
	// JSON means that the model supports enforcing that the response is valid JSON but not necessarily with a
	// schema.
	JSON bool
	// JSONSchema means that the model supports enforcing that the response is a specific JSON schema.
	JSONSchema bool
	// Citations is set when the provider and model combination supports citations in the response.
	Citations bool
	// Seed is set when the provider and model combination supports seed for reproducibility.
	Seed bool
	// TopLogprobs is set when the provider and model combination supports top_logprobs.
	TopLogprobs bool

	// ReportRateLimits means that the provider reports rate limits in its Usage.
	ReportRateLimits bool
	// BrokenTokenUsage means that the usage is not correctly reported.
	BrokenTokenUsage TriState
	// BrokenFinishReason means that the finish reason (FinishStop, FinishLength, etc) is not correctly reported.
	BrokenFinishReason bool
	// NoMaxTokens means that the provider doesn't support limiting text output. Only relevant on text output.
	NoMaxTokens bool
	// NoStopSequence means that the provider doesn't support stop words. Only relevant on text output.
	NoStopSequence bool
	// BiasedTool is true when we ask the LLM to use a tool in an ambiguous biased question, it will always
	// reply with the first readily available answer.
	//
	// This means that when using enum, it is important to understand that the LLM will put heavy weight on the
	// first option.
	//
	// This is affected by two factors: model size and quantization. Quantization affects this dramatically.
	BiasedTool TriState
	// IndecisiveTool is True when we ask the LLM to use a tool in an ambiguous biased question, it'll call both
	// options. It is Flaky when both can happen.
	//
	// This is actually fine, it means that the LLM will be less opinionated in some cases. The test into which
	// a LLM is indecisive is likely model-specific too.
	IndecisiveTool TriState

	_ struct{}
}

// TriState helps describing support when a feature "kinda work", which is frequent with LLM's inherent
// non-determinism.
type TriState int8

const (
	False TriState = 0
	True  TriState = 1
	Flaky TriState = -1
)

const triStateName = "flakyfalsetrue"

var triStateIndex = [...]uint8{0, 5, 10, 14}

func (i TriState) String() string {
	i -= -1
	if i < 0 || i >= TriState(len(triStateIndex)-1) {
		return "TriState(" + strconv.FormatInt(int64(i+-1), 10) + ")"
	}
	return triStateName[triStateIndex[i]:triStateIndex[i+1]]
}

func (i TriState) GoString() string {
	return i.String()
}

// FunctionalityDoc defines which functionalites are supported in a scenario for non-text output modality.
type FunctionalityDoc struct {
	// Seed is set when the provider and model combination supports seed for reproducibility.
	Seed bool

	// ReportRateLimits means that the provider reports rate limits in its Usage.
	ReportRateLimits bool
	// BrokenTokenUsage means that the usage is not correctly reported.
	BrokenTokenUsage TriState
	// BrokenFinishReason means that the finish reason (FinishStop, FinishLength, etc) is not correctly reported.
	BrokenFinishReason bool

	_ struct{}
}

// Scenario defines one way to use the provider.
type Scenario struct {
	In  map[Modality]ModalCapability
	Out map[Modality]ModalCapability
	// Models is a *non exhaustive* list of models that support this scenario. It can't be exhaustive since
	// providers continuouly release new models. It is still valuable to use the first value
	Models []string

	// Thinking means that the model does either explicit chain-of-thought or hidden thinking. For some
	// providers, this is controlled via a OptionsText. For some models (like Qwen3), a token "/no_think" or
	// "/think" is used to control. ThinkingTokenStart and ThinkingTokenEnd must only be set on explicit inline
	// thinking models. They often use <think> and </think>.
	Thinking           bool
	ThinkingTokenStart string
	ThinkingTokenEnd   string

	// GenSync declares features supported when using ProviderGen.GenSync
	GenSync *FunctionalityText
	// GenStream declares features supported when using ProviderGen.GenStream
	GenStream *FunctionalityText
	// GenDoc declares features supported when using a ProviderGenDoc
	GenDoc *FunctionalityDoc

	_ struct{}
}

// Thinking specifies if a model Scenario supports thinking.
type Thinking int8

const (
	// NoThinking means that no thinking is supported.
	NoThinking Thinking = 0
	// ThinkingInline means that the thinking tokens are inline and must be explicitly parsed from Content.Text
	// with adapters.ProviderGenThinking.
	ThinkingInline Thinking = 1
	// ThinkingAutomatic means that the thinking tokens are properly generated and handled by the provider and
	// are returned as Content.Thinking.
	ThinkingAutomatic Thinking = -1
)

// Scoreboard is a snapshot of the capabilities of the provider. These are smoke tested to confirm the
// accuracy.
type Scoreboard struct {
	// Scenarios is the list of all known supported and tested scenarios.
	//
	// A single provider can provide various distinct use cases, like text-to-text, multi-modal-to-text,
	// text-to-audio, audio-to-text, etc.
	Scenarios []Scenario

	// Country where the provider is based, e.g. "US", "CN", "EU". Two exceptions: "Local" for local and "N/A"
	// for pure routers.
	Country string
	// DashboardURL is the URL to the provider's dashboard, if available.
	DashboardURL string

	_ struct{}
}

var (
	_ Validatable = (*Citation)(nil)
	_ Validatable = (*CitationSource)(nil)
	_ Validatable = (*Content)(nil)
	_ Validatable = (*Message)(nil)
	_ Validatable = (*Messages)(nil)
	_ Validatable = (*RateLimit)(nil)
	_ Validatable = (*Role)(nil)
	_ Validatable = (*ToolCall)(nil)
	_ Validatable = (*ToolCallResult)(nil)
)
