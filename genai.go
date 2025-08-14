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
	GenStream(ctx context.Context, msgs Messages, replies chan<- ReplyFragment, opts Options) (Result, error)
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
	// FinishedStop means the LLM was done for the turn. Some providers confuse it with
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

// Messages is a list of valid messages in an exchange with a LLM.
//
// The messages should be alternating between user input, assistant replies, tool call requests and
// computer tool call results. The exception in the case of multi-user discussion, with different Users.
type Messages []Message

// Validate ensures the messages are valid.
func (m Messages) Validate() error {
	var errs []error
	for i := range m {
		if err := m[i].Validate(); err != nil {
			errs = append(errs, fmt.Errorf("message %d: %w", i, err))
		}
		if i > 0 {
			r := m[i].Role()
			lastR := m[i-1].Role()
			if r == lastR {
				errs = append(errs, fmt.Errorf("message %d: role must alternate; got twice %q", i, r))
			}
		}
	}
	return errors.Join(errs...)
}

// Message is a part of an exchange with a LLM.
//
// It is effectively a union, with the exception of the User field that can be set with In.
type Message struct {
	// Request is the message from the user or the computer.
	//
	// It is more frequently multiple items when using multi-modal content.
	Request []Request `json:"request,omitzero"`
	// User must only be used when sent by the user. Only some provider (e.g. OpenAI, Groq, DeepSeek) support it.
	User string `json:"user,omitzero"`

	// Reply is the message from the LLM.
	//
	// Some models can emit multiple content blocks, either multi modal or multiple text blocks: a code block
	// and a different block with an explanantion.
	Reply []Reply `json:"reply,omitzero"`

	// ToolCall is a tool call that the LLM requested to make.
	ToolCalls []ToolCall `json:"tool_calls,omitzero"`

	// ToolCallResult is the result for a tool call that the LLM requested to make.
	ToolCallResults []ToolCallResult `json:"tool_call_results,omitzero"`

	_ struct{}
}

// NewTextMessage is a shorthand function to create a Message with a single
// text block.
func NewTextMessage(text string) Message {
	return Message{Request: []Request{{Text: text}}}
}

func (m *Message) IsZero() bool {
	return m.User == "" && len(m.Request) == 0 && len(m.Reply) == 0 && len(m.ToolCalls) == 0 && len(m.ToolCallResults) == 0
}

// Validate ensures the message is valid.
func (m *Message) Validate() error {
	errs := m.validateShallow()
	for i := range m.Request {
		if err := m.Request[i].Validate(); err != nil {
			errs = append(errs, fmt.Errorf("request %d: %w", i, err))
		}
	}
	for i := range m.Reply {
		if err := m.Reply[i].Validate(); err != nil {
			errs = append(errs, fmt.Errorf("reply %d: %w", i, err))
		}
	}
	for i := range m.ToolCalls {
		if err := m.ToolCalls[i].Validate(); err != nil {
			errs = append(errs, fmt.Errorf("tool call %d: %w", i, err))
		}
	}
	for i := range m.ToolCallResults {
		if err := m.ToolCallResults[i].Validate(); err != nil {
			errs = append(errs, fmt.Errorf("tool result %d: %w", i, err))
		}
	}
	return errors.Join(errs...)
}

// validateShallow ensures the message is valid but not the inner fields.
func (m *Message) validateShallow() []error {
	var errs []error
	if m.IsZero() {
		errs = append(errs, errors.New("at least one of fields Request, Reply, ToolCalls or ToolCallsResults is required"))
	} else if m.Role() == "invalid" {
		errs = append(errs, errors.New("exactly one of Request, Reply/ToolCalls or ToolCallResults must be set"))
	}
	// We should not accept content along with tool calls except for thinking. It is tricky to evaluate since
	// explicit Chain-of-Thought models like Qwen 3 Thinking or Deepseek R1 return their thinking as text
	// until it is parsed by adapters.ProviderGenThinking.
	//
	// It is possible to use a hack to allow it by assuming all explicit CoT models return thinking as text
	// starting with "<".
	//
	// The problem is with deepseek-reasoner. It returns both thinking, text and tool call as a single reply!
	// The text can be discarded in GenSync which would make this check pass, but the text cannot be discarded
	// in GenStream because the ordering of the generated content is thinking, then text, then tool call.
	//
	// See providers/deepseek/testdata/TestClient_Scoreboard/deepseek-reasoner_thinking/GenStream-Tools-SquareRoot-1-any.yaml
	// for an example.
	//
	// At the very least, assert no document is returned along with tool calls.
	if len(m.Reply) != 0 && len(m.ToolCalls) != 0 {
		for i := range m.Reply {
			if !m.Reply[i].Doc.IsZero() {
				return append(errs, errors.New("field Reply can't contain a Doc along with ToolCalls"))
			}
		}
	}
	if m.User != "" {
		errs = append(errs, errors.New("field User: not supported yet"))
	}
	return errs
}

// Role returns one of "user", "assistant" or "computer".
func (m *Message) Role() string {
	hasRequest := len(m.Request) != 0
	hasReplyOrTool := len(m.Reply) != 0 || len(m.ToolCalls) != 0
	hasToolResult := len(m.ToolCallResults) != 0
	if hasRequest && !hasReplyOrTool && !hasToolResult {
		return "user"
	}
	if !hasRequest && hasReplyOrTool && !hasToolResult {
		return "assistant"
	}
	if !hasRequest && !hasReplyOrTool && hasToolResult {
		return "computer"
	}
	return "invalid"
}

// AsText is a short hand to get the request or reply content as text.
//
// It ignores Thinking or multi-modal content.
func (m *Message) AsText() string {
	var data [32]string
	out := data[:0]
	// Only one of the two slices will be non-empty.
	for i := range m.Request {
		if s := m.Request[i].Text; s != "" {
			out = append(out, s)
		}
	}
	for i := range m.Reply {
		if s := m.Reply[i].Text; s != "" {
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
		return fmt.Errorf("failed to decode message text as JSON: %w; reply: %q", err, s)
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
	// We do not need to check each individual fields since they each implement json.Unmarshaler that explicitly
	// call their own Validate() method.
	return errors.Join(m.validateShallow()...)
}

func (m *Message) GoString() string {
	b, _ := json.Marshal(m)
	return string(b)
}

// Request is a block of content in the message meant to be visible in a
// chat setting.
//
// It is effectively a union, only one of the 2 related field groups can be set.
type Request struct {
	// Text is the content of the text message.
	Text string `json:"text,omitzero"`

	// Doc can be audio, video, image, PDF or any other format, including reference text.
	Doc Doc `json:"doc,omitzero"`

	_ struct{}
}

// Validate ensures the block is valid.
func (r *Request) Validate() error {
	if r.Text != "" {
		if !r.Doc.IsZero() {
			return errors.New("field Doc can't be used along Text")
		}
	} else if !r.Doc.IsZero() {
		if err := r.Doc.Validate(); err != nil {
			return err
		}
	} else {
		return errors.New("an empty Request is invalid")
	}
	return nil
}

func (r *Request) UnmarshalJSON(b []byte) error {
	type Alias Request
	a := struct{ *Alias }{Alias: (*Alias)(r)}
	d := json.NewDecoder(bytes.NewReader(b))
	d.DisallowUnknownFields()
	if err := d.Decode(&a); err != nil {
		return err
	}
	return r.Validate()
}

// Reply is a block of content in the message meant to be visible in a
// chat setting.
type Reply struct {
	// Text is the content of the text message.
	Text string `json:"text,omitzero"`

	// Doc can be audio, video, image, PDF or any other format, including reference text.
	Doc Doc `json:"doc,omitzero"`

	// Citations contains references to source material that support the content.
	// Only valid when Text is set and the provider supports citations.
	Citations []Citation `json:"citations,omitzero"`

	// Thinking is the reasoning done by the LLM.
	Thinking string `json:"thinking,omitzero"`
	// Opaque is added to keep continuity on the processing. A good example is Anthropic's extended thinking. It
	// must be kept during an exchange.
	//
	// A message with only Opaque set is valid.
	Opaque map[string]any `json:"opaque,omitzero"`

	_ struct{}
}

// Validate ensures the block is valid.
func (r *Reply) Validate() error {
	// Validate citations when text is present
	for i, citation := range r.Citations {
		if err := citation.Validate(); err != nil {
			return fmt.Errorf("citation %d: %w", i, err)
		}
	}
	if r.Text != "" {
		if !r.Doc.IsZero() {
			return errors.New("field Doc can't be used along Text")
		}
		if r.Thinking != "" {
			return errors.New("field Thinking can't be used along Text")
		}
		if len(r.Opaque) != 0 {
			return errors.New("field Opaque can't be used along Text")
		}
	} else if !r.Doc.IsZero() {
		if err := r.Doc.Validate(); err != nil {
			return err
		}
	} else if r.Thinking != "" || len(r.Opaque) != 0 {
		if len(r.Citations) != 0 {
			return errors.New("field Citations can only be used with Text")
		}
		if !r.Doc.IsZero() {
			return errors.New("field Doc can't be used along Thinking")
		}
	} else if len(r.Citations) != 0 {
		if len(r.Opaque) != 0 {
			return errors.New("field Opaque can't be used along a Citations")
		}
		if !r.Doc.IsZero() {
			return errors.New("field Doc can't be used along Citations")
		}
	} else {
		return errors.New("an empty Reply is invalid")
	}
	return nil
}

func (r *Reply) UnmarshalJSON(b []byte) error {
	type Alias Reply
	a := struct{ *Alias }{Alias: (*Alias)(r)}
	d := json.NewDecoder(bytes.NewReader(b))
	d.DisallowUnknownFields()
	if err := d.Decode(&a); err != nil {
		return err
	}
	return r.Validate()
}

// Doc is a document.
type Doc struct {
	// Filename is the name of the file. For many providers, only the extension
	// is relevant. They only use mime-type, which is derived from the filename's
	// extension. When an URL is provided or when the object provided to Document
	// implements a method with the signature `Name() string`, like an
	// `*os.File`, Filename is optional.
	Filename string `json:"filename,omitzero"`
	// Src is raw document data. It is perfectly fine to use a bytes.NewReader() or *os.File.
	Src io.ReadSeeker `json:"bytes,omitzero"`
	// URL is the reference to the raw data. When set, the mime-type is derived from the URL.
	URL string `json:"url,omitzero"`

	_ struct{}
}

func (d *Doc) IsZero() bool {
	return d.Filename == "" && d.Src == nil && d.URL == ""
}

// Validate ensures the block is valid.
func (d *Doc) Validate() error {
	if d.Src != nil && d.URL != "" {
		return errors.New("field Document and URL are mutually exclusive")
	}
	if d.Filename != "" && d.Src == nil && d.URL == "" {
		return errors.New("field Document or URL is required when using Filename")
	}
	if d.Filename == "" && d.Src != nil {
		if _, ok := d.Src.(interface{ Name() string }); !ok {
			return errors.New("field Filename is required with Document when not implementing Name()")
		}
	}
	return nil
}

// GetFilename returns the filename to use for the document, querying the
// Document's name if available.
func (d *Doc) GetFilename() string {
	if d.Filename == "" {
		if namer, ok := d.Src.(interface{ Name() string }); ok {
			return filepath.Base(namer.Name())
		}
	}
	return d.Filename
}

type serializedDoc struct {
	Filename string `json:"filename,omitzero"`
	Bytes    []byte `json:"bytes,omitzero"`
	URL      string `json:"url,omitzero"`
}

func (d *Doc) MarshalJSON() ([]byte, error) {
	dd := serializedDoc{Filename: d.Filename, URL: d.URL}
	if d.Src != nil {
		if _, err := d.Src.Seek(0, io.SeekStart); err != nil {
			return nil, err
		}
		var err error
		if dd.Bytes, err = io.ReadAll(d.Src); err != nil {
			return nil, err
		}
		if d.Filename == "" {
			if namer, ok := d.Src.(interface{ Name() string }); ok {
				dd.Filename = filepath.Base(namer.Name())
			}
		}
	}
	return json.Marshal(&dd)
}

func (d *Doc) UnmarshalJSON(b []byte) error {
	dd := serializedDoc{}
	de := json.NewDecoder(bytes.NewReader(b))
	de.DisallowUnknownFields()
	if err := de.Decode(&dd); err != nil {
		return err
	}
	d.Filename = dd.Filename
	d.URL = dd.URL
	if len(dd.Bytes) != 0 {
		d.Src = &bb.BytesBuffer{D: dd.Bytes}
	}
	return d.Validate()
}

// Read reads the document content into memory.
func (d *Doc) Read(maxSize int64) (string, []byte, error) {
	// genai cannot depend on base as it would cause a circular import.
	mimeType := internal.MimeByExt(filepath.Ext(d.GetFilename()))
	if d.URL != "" {
		// Not all provider require a mime-type so do not error out.
		if mimeType == "" {
			mimeType = internal.MimeByExt(filepath.Ext(path.Base(d.URL)))
		}
		return mimeType, nil, nil
	}
	if mimeType == "" {
		return "", nil, errors.New("failed to determine mime-type, pass a filename with an extension")
	}
	size, err := d.Src.Seek(0, io.SeekEnd)
	if err != nil {
		return "", nil, fmt.Errorf("failed to seek data: %w", err)
	}
	if size > maxSize {
		return "", nil, fmt.Errorf("large files are not yet supported, max %dMiB", maxSize/1024/1024)
	}
	if _, err = d.Src.Seek(0, io.SeekStart); err != nil {
		return "", nil, fmt.Errorf("failed to seek data: %w", err)
	}
	var data []byte
	if data, err = io.ReadAll(d.Src); err != nil {
		return "", nil, fmt.Errorf("failed to read data: %w", err)
	}
	if len(data) == 0 {
		return "", nil, errors.New("empty data")
	}
	return mimeType, data, nil
}

// ReplyFragment is a fragment of a content the LLM is sending back as part
// of the GenStream().
type ReplyFragment struct {
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

func (m *ReplyFragment) IsZero() bool {
	return m.TextFragment == "" && m.ThinkingFragment == "" && len(m.Opaque) == 0 && m.Filename == "" && len(m.DocumentFragment) == 0 && m.URL == "" && m.ToolCall.IsZero() && m.Citation.IsZero()
}

func (m *ReplyFragment) GoString() string {
	b, _ := json.Marshal(m)
	return string(b)
}

// Accumulate adds a ReplyFragment to the message being streamed.
func (m *Message) Accumulate(mf ReplyFragment) error {
	// Generally the first message fragment.
	if mf.ThinkingFragment != "" {
		if len(m.Reply) != 0 {
			if lastBlock := &m.Reply[len(m.Reply)-1]; lastBlock.Thinking != "" {
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
		m.Reply = append(m.Reply, Reply{Thinking: mf.ThinkingFragment, Opaque: mf.Opaque})
		return nil
	}
	if len(mf.Opaque) != 0 {
		if len(m.Reply) != 0 {
			// Only add Opaque to Thinking block.
			if lastBlock := &m.Reply[len(m.Reply)-1]; lastBlock.Thinking != "" {
				if lastBlock.Opaque == nil {
					lastBlock.Opaque = map[string]any{}
				}
				maps.Copy(lastBlock.Opaque, mf.Opaque)
				return nil
			}
		}
		// Unlikely.
		m.Reply = append(m.Reply, Reply{Opaque: mf.Opaque})
		return nil
	}

	// Content.
	if mf.TextFragment != "" {
		if len(m.Reply) != 0 {
			if lastBlock := &m.Reply[len(m.Reply)-1]; lastBlock.Text != "" {
				lastBlock.Text += mf.TextFragment
				return nil
			}
		}
		m.Reply = append(m.Reply, Reply{Text: mf.TextFragment})
		return nil
	}

	if mf.URL != "" {
		m.Reply = append(m.Reply, Reply{Doc: Doc{Filename: mf.Filename, URL: mf.URL}})
		return nil
	}
	if mf.DocumentFragment != nil {
		if len(m.Reply) != 0 {
			if lastBlock := &m.Reply[len(m.Reply)-1]; lastBlock.Doc.Filename != "" || lastBlock.Doc.Src != nil {
				if lastBlock.Doc.Src == nil {
					lastBlock.Doc.Src = &bb.BytesBuffer{}
				}
				if lastBlock.Doc.Filename == "" {
					// Unlikely.
					lastBlock.Doc.Filename = mf.Filename
				}
				_, _ = lastBlock.Doc.Src.(*bb.BytesBuffer).Write(mf.DocumentFragment)
				return nil
			}
		}
		m.Reply = append(m.Reply, Reply{Doc: Doc{Filename: mf.Filename, Src: &bb.BytesBuffer{D: mf.DocumentFragment}}})
		return nil
	}
	if mf.Filename != "" {
		m.Reply = append(m.Reply, Reply{Doc: Doc{Filename: mf.Filename}})
		return nil
	}

	if mf.ToolCall.Name != "" {
		m.ToolCalls = append(m.ToolCalls, mf.ToolCall)
		return nil
	}

	if !mf.Citation.IsZero() {
		// For now always add a new block.
		m.Reply = append(m.Reply, Reply{Citations: []Citation{mf.Citation}})
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

func (t *ToolCall) UnmarshalJSON(b []byte) error {
	type Alias ToolCall
	a := struct{ *Alias }{Alias: (*Alias)(t)}
	d := json.NewDecoder(bytes.NewReader(b))
	d.DisallowUnknownFields()
	if err := d.Decode(&a); err != nil {
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
	_ Validatable = (*Message)(nil)
	_ Validatable = (*Messages)(nil)
	_ Validatable = (*RateLimit)(nil)
	_ Validatable = (*Reply)(nil)
	_ Validatable = (*Request)(nil)
	_ Validatable = (*ToolCall)(nil)
	_ Validatable = (*ToolCallResult)(nil)
)
