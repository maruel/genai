// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Package genai is the opiniated high performance professional-grade AI package for Go.
//
// It provides a generic interface to interact with various LLM providers, while allowing full access to each
// provider's full capabilities.
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
	"iter"
	"maps"
	"net/http"
	"path"
	"path/filepath"
	"reflect"
	"strings"
	"time"

	"github.com/maruel/genai/internal"
	"github.com/maruel/genai/internal/bb"
	"github.com/maruel/genai/scoreboard"
)

// Provider is the base interface that all provider interfaces embed.
//
// The first group contains local methods. Calling these methods will not make an HTTP request.
//
// The second group is supported by the majority of providers.
//
// The rest is supported by a limited number of providers.
type Provider interface {
	// Name returns the name of the provider.
	Name() string
	// ModelID returns the model currently used by the provider. It can be an empty string.
	ModelID() string
	// OutputModalities returns the output modalities supported by this specific client configuration.
	//
	// This states what kind of output the model will generate (text, audio, image, video). It varies per
	// provider and models. The vast majority of providers and models support only output modality like
	// text-only, image-only, etc.
	OutputModalities() Modalities
	// Capabilities returns the optional capabilities this provider supports.
	Capabilities() ProviderCapabilities
	// Scoreboard returns what the provider supports.
	//
	// Some models have more features than others, e.g. some models may be text-only while others have vision or
	// audio support.
	//
	// The client code may be the limiting factor for some models, and not the provider itself.
	//
	// The values returned here should have gone through a smoke test to make sure they are valid.
	Scoreboard() scoreboard.Score
	// HTTPClient returns the underlying http client. It may be necessary to use it to fetch the results from
	// the provider. An example is retrieving Veo 3 generated videos from Gemini requires the authentication
	// headers to be set.
	HTTPClient() *http.Client

	// GenSync runs generation synchronously.
	//
	// Multiple options can be mixed together, both standard ones like *GenOptionImage, *GenOptionText,
	// *GenOptionTools and provider-specialized options struct, e.g. *anthropic.GenOptionText, *gemini.GenOption.
	GenSync(ctx context.Context, msgs Messages, opts ...GenOption) (Result, error)
	// GenStream runs generation synchronously, yielding the fragments of replies as the server sends them.
	//
	// No need to accumulate the fragments into a Message since the Result contains the accumulated message.
	GenStream(ctx context.Context, msgs Messages, opts ...GenOption) (iter.Seq[Reply], func() (Result, error))
	// ListModels returns the list of models the provider supports. Not all providers support it, some will
	// return an ErrorNotSupported. For local providers like llamacpp and ollama, they may return only the
	// model currently loaded.
	ListModels(ctx context.Context) ([]Model, error)

	// GenAsync requests a generation and returns a pending job that can be polled.
	//
	// Requires ProviderCapabilities.GenAsync to be set. Returns base.ErrNotSupported otherwise.
	GenAsync(ctx context.Context, msgs Messages, opts ...GenOption) (Job, error)
	// PokeResult requests the state of the job.
	//
	// When the job is still pending, Result.Usage.FinishReason is Pending.
	//
	// Requires ProviderCapabilities.GenAsync to be set. Returns base.ErrNotSupported otherwise.
	PokeResult(ctx context.Context, job Job) (Result, error)
	// CacheAddRequest caches a request.
	//
	// Requires ProviderCapabilities.Caching to be set. Returns base.ErrNotSupported otherwise.
	//
	// # Warning
	//
	// May be changed in the future.
	CacheAddRequest(ctx context.Context, msgs Messages, name, displayName string, ttl time.Duration, opts ...GenOption) (string, error)
	// CacheList lists the caches entries.
	//
	// Requires ProviderCapabilities.Caching to be set. Returns base.ErrNotSupported otherwise.
	//
	// # Warning
	//
	// May be changed in the future.
	CacheList(ctx context.Context) ([]CacheEntry, error)
	// CacheDelete deletes a cache entry.
	//
	// Requires ProviderCapabilities.Caching to be set. Returns base.ErrNotSupported otherwise.
	//
	// # Warning
	//
	// May be changed in the future.
	CacheDelete(ctx context.Context, name string) error
}

// ProviderCapabilities describes optional capabilities a provider supports.
type ProviderCapabilities struct {
	// GenAsync indicates the provider supports GenAsync and PokeResult for batch operations.
	GenAsync bool
	// Caching indicates the provider supports CacheAddRequest, CacheList, and CacheDelete.
	Caching bool

	_ struct{}
}

// ProviderUnwrap is exposed when the Provider is actually a wrapper around another one, like
// adapters.ProviderReasoning or ProviderUsage. This is useful when looking for other interfaces.
type ProviderUnwrap interface {
	Unwrap() Provider
}

// Generation

// Result is the result of a completion.
//
// It is a Message along with Usage metadata about the operation. It optionally include Logprobs if requested
// and the provider support it.
type Result struct {
	Message
	Usage Usage
	// Logprobs is a list of multiple log probabilities, each for a token.
	//
	// The first item of each subslice is the chosen token. The next items are the candidates not chosen.
	//
	// Some providers only return the probability for the chosen tokens and not for the candidates.
	Logprobs [][]Logprob
}

// Logprob represents a single log probability information for a token.
//
// One of ID or Text must be set.
type Logprob struct {
	ID      int64   `json:"id,omitempty"`   // Input token ID.
	Text    string  `json:"text,omitempty"` // Text in UTF-8.
	Logprob float64 `json:"logprob"`        // Log probability of the token. It should normally be non-zero but sometimes it is.
}

// GoString returns a JSON representation of the reply for debugging purposes.
func (l *Logprob) GoString() string {
	b, _ := json.Marshal(l)
	return string(b)
}

// Usage from the LLM provider.
type Usage struct {
	// Token usage for the current request.
	InputTokens       int64
	InputCachedTokens int64
	ReasoningTokens   int64
	OutputTokens      int64
	TotalTokens       int64
	// FinishReason indicates why the model stopped generating tokens.
	FinishReason FinishReason
	// ServiceTier is the quality of service tier used to process the request,
	// as reported by the provider. Common values include "standard", "batch",
	// "flex", "default", "auto", etc. Empty when not reported.
	ServiceTier string
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

// Add accumulates the usage from another result.
func (u *Usage) Add(r *Usage) {
	u.InputTokens += r.InputTokens
	u.InputCachedTokens += r.InputCachedTokens
	u.ReasoningTokens += r.ReasoningTokens
	u.OutputTokens += r.OutputTokens
	u.TotalTokens += r.TotalTokens
}

// RateLimitType defines the type of rate limit.
type RateLimitType int32

// RateLimitType values.
const (
	Requests RateLimitType = iota + 1
	Tokens
)

// RateLimitPeriod defines the time period for a rate limit.
type RateLimitPeriod int32

// RateLimitPeriod values.
const (
	PerOther RateLimitPeriod = iota // For non-standard periods
	PerMinute
	PerDay
	PerMonth
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
	out := ""
	switch r.Type {
	case Requests:
		out = "requests"
	case Tokens:
		out = "tokens"
	default:
		return "invalid RateLimit"
	}
	if !r.Reset.IsZero() {
		out += "/" + r.Reset.Format(time.DateTime)
	}
	switch r.Period {
	case PerMinute:
		out += " (minute)"
	case PerDay:
		out += " (day)"
	case PerMonth:
		out += " (month)"
	case PerOther:
	default:
		return "invalid RateLimit"
	}
	return out + fmt.Sprintf(": %d/%d", r.Remaining, r.Limit)
}

// Validate ensures the rate limit is valid.
func (r *RateLimit) Validate() error {
	switch r.Type {
	case Requests, Tokens:
	default:
		return fmt.Errorf("invalid limit type %v", r.Type)
	}
	if r.Limit == 0 {
		return errors.New("limit is 0")
	}
	// It is valid for Remaining to be zero. It's rare but it happens.
	switch r.Period {
	case PerMinute, PerDay, PerMonth, PerOther:
	default:
		return fmt.Errorf("invalid limit period %v", r.Period)
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
	// GenOptionText.MaxTokens or as limited by the provider.
	FinishedLength FinishReason = "length"
	// FinishedToolCalls means the model called one or multiple tools and needs the replies to continue the turn.
	FinishedToolCalls FinishReason = "tool_calls"
	// FinishedStopSequence means the model stopped because it saw a stop word as listed in GenOptionText.Stop.
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
			errs = append(errs, fmt.Errorf("message #%d: %w", i, err))
		}
		if i > 0 {
			r := m[i].Role()
			lastR := m[i-1].Role()
			if r == lastR {
				errs = append(errs, fmt.Errorf("message #%d: role must alternate; got twice %q", i, r))
			}
		}
	}
	return errors.Join(errs...)
}

// Message is a part of an exchange with a LLM.
//
// It is effectively a union, with the exception of the User field that can be set with In.
type Message struct {
	// Requests is the content from the user.
	//
	// It is normally a single message. It is more frequently multiple items when using multi-modal content.
	Requests []Request `json:"request,omitzero"`
	// User must only be used when sent by the user. Only some provider (e.g. OpenAI, Groq, DeepSeek) support it.
	User string `json:"user,omitzero"`

	// Replies is the message from the LLM.
	//
	// It is generally a single reply with text or a tool call. Some models can emit multiple content blocks,
	// either multi modal or multiple text blocks: a code block and a different block with an explanantion. Some
	// models can emit multiple tool calls at once.
	Replies []Reply `json:"reply,omitzero"`

	// ToolCallResult is the result for a tool call that the LLM requested to make.
	//
	// These messages are generated by the "computer".
	ToolCallResults []ToolCallResult `json:"tool_call_results,omitzero"`

	_ struct{}
}

// NewTextMessage is a shorthand function to create a Message with a single
// text block.
func NewTextMessage(text string) Message {
	return Message{Requests: []Request{{Text: text}}}
}

// IsZero returns true if the message is empty.
func (m *Message) IsZero() bool {
	return m.User == "" && len(m.Requests) == 0 && len(m.Replies) == 0 && len(m.ToolCallResults) == 0
}

// Validate ensures the message is valid.
func (m *Message) Validate() error {
	errs := m.validateShallow()
	for i := range m.Requests {
		if err := m.Requests[i].Validate(); err != nil {
			errs = append(errs, fmt.Errorf("request #%d: %w", i, err))
		}
	}
	for i := range m.Replies {
		if err := m.Replies[i].Validate(); err != nil {
			errs = append(errs, fmt.Errorf("reply #%d: %w", i, err))
		}
	}
	for i := range m.ToolCallResults {
		if err := m.ToolCallResults[i].Validate(); err != nil {
			errs = append(errs, fmt.Errorf("tool result #%d: %w", i, err))
		}
	}
	return errors.Join(errs...)
}

// validateShallow ensures the message is valid but not the inner fields.
func (m *Message) validateShallow() []error {
	var errs []error
	if m.IsZero() {
		errs = append(errs, errors.New("at least one of fields Request, Reply or ToolCallsResults is required"))
	} else if m.Role() == "invalid" {
		errs = append(errs, errors.New("exactly one of Request, Reply or ToolCallResults must be set"))
	}
	if m.User != "" {
		errs = append(errs, errors.New("field User: not supported yet"))
	}
	return errs
}

// Role returns one of "user", "assistant" or "computer".
func (m *Message) Role() string {
	hasRequest := len(m.Requests) != 0
	hasReply := len(m.Replies) != 0
	hasToolResult := len(m.ToolCallResults) != 0
	if hasRequest && !hasReply && !hasToolResult {
		return "user"
	}
	if !hasRequest && hasReply && !hasToolResult {
		return "assistant"
	}
	if !hasRequest && !hasReply && hasToolResult {
		return "computer"
	}
	return "invalid"
}

// String is a short hand to get the request or reply content as text.
//
// It ignores reasoning or multi-modal content.
func (m *Message) String() string {
	var data [32]string
	out := data[:0]
	// Only one of the two slices will be non-empty.
	for i := range m.Requests {
		if s := m.Requests[i].Text; s != "" {
			out = append(out, s)
		}
	}
	for i := range m.Replies {
		if s := m.Replies[i].Text; s != "" {
			out = append(out, s)
		}
	}
	return strings.Join(out, "")
}

// Reasoning returns all the reasoning concatenated, if any.
func (m *Message) Reasoning() string {
	var data [32]string
	out := data[:0]
	for i := range m.Replies {
		if s := m.Replies[i].Reasoning; s != "" {
			out = append(out, s)
		}
	}
	return strings.Join(out, "")
}

// Decode decodes the JSON message into the struct.
//
// Requires using either ReplyAsJSON or DecodeAs in the GenOptionText.
//
// Note: this doesn't verify the type is the same as specified in
// GenOptionText.DecodeAs.
func (m *Message) Decode(x any) error {
	s := m.String()
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

// DoToolCalls processes all the ToolCall in the Reply if any.
//
// Returns a Message to be added back to the list of messages, only if msg.IsZero() is true.
func (m *Message) DoToolCalls(ctx context.Context, tools []ToolDef) (Message, error) {
	var out Message
	for i := range m.Replies {
		if m.Replies[i].ToolCall.IsZero() {
			continue
		}
		res, err := m.Replies[i].ToolCall.Call(ctx, tools)
		if err != nil {
			return out, err
		}
		out.ToolCallResults = append(out.ToolCallResults, ToolCallResult{
			ID:     m.Replies[i].ToolCall.ID,
			Name:   m.Replies[i].ToolCall.Name,
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
		return fmt.Errorf("failed to decode message: %w", err)
	}
	// We do not need to check each individual fields since they each implement json.Unmarshaler that explicitly
	// call their own Validate() method.
	return errors.Join(m.validateShallow()...)
}

// GoString returns a JSON representation of the reply for debugging purposes.
func (m *Message) GoString() string {
	b, _ := json.Marshal(m)
	return string(b)
}

// Accumulate adds a Reply to the message being streamed.
//
// It is used by GenStream. There's generally no need to call it by end users.
func (m *Message) Accumulate(mf *Reply) error {
	// Generally the first message fragment.
	if mf.Reasoning != "" {
		if len(m.Replies) != 0 {
			if lastBlock := &m.Replies[len(m.Replies)-1]; lastBlock.Reasoning != "" {
				lastBlock.Reasoning += mf.Reasoning
				if len(mf.Opaque) != 0 {
					return &internal.BadError{Err: errors.New("cannot add Opaque to a Reasoning block")}
				}
				return nil
			}
		}
		m.Replies = append(m.Replies, Reply{Reasoning: mf.Reasoning, Opaque: mf.Opaque})
		return nil
	}

	// Content.
	if mf.Text != "" {
		if len(m.Replies) != 0 {
			if lastBlock := &m.Replies[len(m.Replies)-1]; lastBlock.Text != "" {
				lastBlock.Text += mf.Text
				return nil
			}
		}
		m.Replies = append(m.Replies, Reply{Text: mf.Text})
		return nil
	}

	if mf.Doc.URL != "" {
		m.Replies = append(m.Replies, Reply{Doc: Doc{Filename: mf.Doc.Filename, URL: mf.Doc.URL}})
		return nil
	}
	if mf.Doc.Src != nil {
		src, ok := mf.Doc.Src.(*bb.BytesBuffer)
		if !ok {
			raw, err := io.ReadAll(mf.Doc.Src)
			if err != nil {
				return &internal.BadError{Err: fmt.Errorf("failed to read document source: %w", err)}
			}
			src = &bb.BytesBuffer{D: raw}
		}
		if len(m.Replies) != 0 {
			if lastBlock := &m.Replies[len(m.Replies)-1]; lastBlock.Doc.Filename != "" || lastBlock.Doc.Src != nil {
				if lastBlock.Doc.Src == nil {
					lastBlock.Doc.Src = &bb.BytesBuffer{}
				}
				if lastBlock.Doc.Filename == "" {
					// Unlikely.
					lastBlock.Doc.Filename = mf.Doc.Filename
				}
				// We are almost guaranteed that the last block is a bb.BytesBuffer but we go safe and just look for
				// io.Writer. This permits users to use their own types.
				lastSrc, ok := lastBlock.Doc.Src.(io.Writer)
				if !ok {
					return &internal.BadError{Err: fmt.Errorf("invalid document source type: %T", lastBlock.Doc.Src)}
				}
				_, _ = lastSrc.Write(src.D)
				return nil
			}
		}
		m.Replies = append(m.Replies, Reply{Doc: Doc{Filename: mf.Doc.Filename, Src: src}})
		return nil
	}
	if mf.Doc.Filename != "" {
		m.Replies = append(m.Replies, Reply{Doc: Doc{Filename: mf.Doc.Filename}})
		return nil
	}

	if mf.ToolCall.Name != "" {
		m.Replies = append(m.Replies, Reply{ToolCall: mf.ToolCall})
		return nil
	}

	if !mf.Citation.IsZero() {
		// For now always add a new block.
		m.Replies = append(m.Replies, Reply{Citation: mf.Citation})
		return nil
	}
	if len(mf.Opaque) != 0 {
		if len(m.Replies) != 0 {
			// Only add Opaque to Reasoning block.
			if lastBlock := &m.Replies[len(m.Replies)-1]; lastBlock.Reasoning != "" {
				if lastBlock.Opaque == nil {
					lastBlock.Opaque = map[string]any{}
				}
				maps.Copy(lastBlock.Opaque, mf.Opaque)
				return nil
			}
		}
		// Unlikely.
		m.Replies = append(m.Replies, Reply{Opaque: mf.Opaque})
		return nil
	}

	// Nothing to accumulate. It should be an error but there are bugs where the system hangs.
	return nil
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
	switch {
	case r.Text != "":
		if !r.Doc.IsZero() {
			return errors.New("field Doc can't be used along Text")
		}
	case !r.Doc.IsZero():
		if err := r.Doc.Validate(); err != nil {
			return err
		}
	default:
		return errors.New("an empty Request is invalid")
	}
	return nil
}

// UnmarshalJSON implements the json.Unmarshaler interface.
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

// Reply is a block of information returned by the provider.
//
// Normally only one of the field must be set. The exception is the Opaque field.
//
// Reply generally represents content returned by the provider, like a block of text or a document returned by
// the model. It can be a silent tool call request. It can also be an opaque block. A good example is traces
// of server side tool calling like WebSearch or MCP tool calling.
type Reply struct {
	// Text is the content of the text message.
	Text string `json:"text,omitzero"`

	// Doc can be audio, video, image, PDF or any other format, including reference text.
	Doc Doc `json:"doc,omitzero"`

	// Citation contains references to source material that support the content.
	Citation Citation `json:"citation,omitzero"`

	// Reasoning is the reasoning done by the LLM.
	Reasoning string `json:"reasoning,omitzero"`

	// ToolCall is a tool call that the LLM requested to make.
	ToolCall ToolCall `json:"tool_call,omitzero"`

	// Opaque is added to keep continuity on the processing. A good example is Anthropic's extended thinking, or
	// server-side tool calling. It must be kept during an exchange.
	//
	// A message with only Opaque set is valid. It can be used in combination with other fields. This field is
	// specific to both the provider and the model.
	//
	// The data must be JSON-serializable.
	Opaque map[string]any `json:"opaque,omitzero"`

	_ struct{}
}

// IsZero returns true if the Reply is empty.
//
// An empty reply is not valid.
func (r *Reply) IsZero() bool {
	return r.Text == "" && r.Doc.IsZero() && r.Citation.IsZero() && r.Reasoning == "" && len(r.Opaque) == 0 && r.ToolCall.IsZero()
}

// GoString returns a JSON representation of the reply for debugging purposes.
func (r *Reply) GoString() string {
	b, _ := json.Marshal(r)
	return string(b)
}

// Validate ensures the block is valid.
func (r *Reply) Validate() error {
	switch {
	case r.Text != "":
		if !r.Doc.IsZero() {
			return errors.New("field Doc can't be used along Text")
		}
		if !r.Citation.IsZero() {
			return errors.New("field Citation can't be used along Text")
		}
		if r.Reasoning != "" {
			return errors.New("field Reasoning can't be used along Text")
		}
		if !r.ToolCall.IsZero() {
			return errors.New("field ToolCall can't be used along Text")
		}
		// Reasoning is allowed.
		//
		// We should not accept Text along with ToolCall. It is tricky to evaluate since explicit Chain-of-Thought
		// models like Qwen 3 Thinking or Deepseek R1 return their reasoning as text until it is parsed by
		// adapters.ProviderReasoning.
		//
		// It is possible to use a hack to allow it by assuming all explicit CoT models return reasoning as text
		// starting with "<".
		//
		// The problem is with deepseek-reasoner. It returns both Text, Reasoning, and ToolCall as a single reply!
		// The text can be discarded in GenSync which would make this check pass, but the Text cannot be discarded
		// in GenStream because the ordering of the generated content is Reasoning, then Text, then ToolCall.
		//
		// See
		// providers/deepseek/testdata/TestClient/Scoreboard/deepseek-reasoner_thinking/GenStream-Tools-SquareRoot-1.yaml
		// for an example.
	case !r.Doc.IsZero():
		if err := r.Doc.Validate(); err != nil {
			return err
		}
		if !r.Citation.IsZero() {
			return errors.New("field Citation can't be used along Doc")
		}
		if r.Reasoning != "" {
			return errors.New("field Reasoning can't be used along Doc")
		}
		if !r.ToolCall.IsZero() {
			return errors.New("field ToolCall can't be used along Doc")
		}
	case !r.Citation.IsZero():
		if err := r.Citation.Validate(); err != nil {
			return err
		}
		if r.Reasoning != "" {
			return errors.New("field Reasoning can't be used along Citation")
		}
		if !r.ToolCall.IsZero() {
			return errors.New("field ToolCall can't be used along Citation")
		}
	case r.Reasoning != "":
		if !r.ToolCall.IsZero() {
			return errors.New("field ToolCall can't be used along Reasoning")
		}
	case !r.ToolCall.IsZero():
		if err := r.ToolCall.Validate(); err != nil {
			return err
		}
	case len(r.Opaque) == 0:
		return errors.New("an empty Reply is invalid")
	}
	return nil
}

// UnmarshalJSON implements the json.Unmarshaler interface.
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

// IsZero returns true if the document is empty.
func (d *Doc) IsZero() bool {
	return d.Filename == "" && d.Src == nil && d.URL == ""
}

// Validate ensures the block is valid.
func (d *Doc) Validate() error {
	if d.Src != nil && d.URL != "" {
		return errors.New("field Document and URL are mutually exclusive")
	}
	if d.Filename != "" {
		if filepath.Base(d.Filename) != d.Filename {
			return errors.New("field Filename must be a valid filename with no path")
		}
		if d.Src == nil && d.URL == "" {
			return errors.New("field Document or URL is required when using Filename")
		}
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
	base := d.Filename
	if base == "" {
		if namer, ok := d.Src.(interface{ Name() string }); ok {
			base = filepath.Base(namer.Name())
		}
	}
	if base == "stdin" {
		base = "stdin.txt"
	}
	return base
}

type serializedDoc struct {
	Filename string `json:"filename,omitzero"`
	Bytes    []byte `json:"bytes,omitzero"`
	URL      string `json:"url,omitzero"`
}

// MarshalJSON implements the json.Marshaler interface.
func (d *Doc) MarshalJSON() ([]byte, error) {
	dd := serializedDoc{Filename: d.GetFilename(), URL: d.URL}
	if d.Src != nil {
		// Try to seek to the beginning; if that fails (e.g., os.Stdin), buffer the whole input.
		if _, err := d.Src.Seek(0, io.SeekStart); err != nil {
			// Unseekable input: buffer it all into a BytesBuffer.
			buf := &bytes.Buffer{}
			if _, err := io.Copy(buf, d.Src); err != nil {
				return nil, err
			}
			dd.Bytes = buf.Bytes()
			// Update d.Src to the buffered version for potential future reads.
			d.Src = &bb.BytesBuffer{D: dd.Bytes}
		} else {
			// Seekable: read from the beginning.
			var err error
			if dd.Bytes, err = io.ReadAll(d.Src); err != nil {
				return nil, err
			}
		}
	}
	return json.Marshal(&dd)
}

// UnmarshalJSON implements the json.Unmarshaler interface.
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
//
// It returns the mime type, the raw bytes and an error if any.
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
	var data []byte
	// Try to seek to end to check size; if that fails (e.g., os.Stdin), buffer the whole input.
	if size, err := d.Src.Seek(0, io.SeekEnd); err != nil {
		// Unseekable input: buffer it all into a BytesBuffer.
		buf := &bytes.Buffer{}
		if _, err = io.Copy(buf, io.LimitReader(d.Src, maxSize)); err != nil {
			return "", nil, fmt.Errorf("failed to copy data into temporary buffer: %w", err)
		}
		data = buf.Bytes()
		// Update d.Src to the buffered version for potential future reads.
		d.Src = &bb.BytesBuffer{D: data}
	} else {
		// Seekable: check size and read.
		if size > maxSize {
			return "", nil, fmt.Errorf("large files are not yet supported, max %dMiB", maxSize/1024/1024)
		}
		if _, err = d.Src.Seek(0, io.SeekStart); err != nil {
			return "", nil, fmt.Errorf("failed to seek data at beginning: %w", err)
		}
		if data, err = io.ReadAll(d.Src); err != nil {
			return "", nil, fmt.Errorf("failed to read data: %w", err)
		}
	}
	if len(data) == 0 {
		return "", nil, errors.New("empty data")
	}
	return mimeType, data, nil
}

// ToolCall is a tool call that the LLM requested to make.
type ToolCall struct {
	ID        string `json:"id,omitzero"`        // Unique identifier for the tool call. Necessary for parallel tool calling.
	Name      string `json:"name,omitzero"`      // Tool being called.
	Arguments string `json:"arguments,omitzero"` // encoded as JSON

	// Opaque is added to keep continuity on the processing. A good example is Anthropic's extended thinking. It
	// must be kept during an exchange.
	//
	// A message with only Opaque set is valid.
	Opaque map[string]any `json:"opaque,omitzero"`

	_ struct{}
}

// IsZero returns true if the tool call is empty.
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
		return fmt.Errorf("failed to decode tool call arguments %q: %w", t.Arguments, err)
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

// UnmarshalJSON implements the json.Unmarshaler interface.
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

// UnmarshalJSON implements the json.Unmarshaler interface.
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
//
// Normally one of CitedText or StartIndex/EndIndex is set.
type Citation struct {
	// CitedText is the text that was cited.
	CitedText string `json:"cited_text,omitzero"`
	// StartIndex is the starting character position of the citation in the answer (0-based).
	StartIndex int64 `json:"start_index,omitzero"`
	// EndIndex is the ending character position of the citation in the answer (0-based, exclusive).
	EndIndex int64 `json:"end_index,omitzero"`

	// Sources contains information about the source documents or tools that support this citation.
	Sources []CitationSource `json:"sources,omitzero"`

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
	for i := range c.Sources {
		if err := c.Sources[i].Validate(); err != nil {
			return fmt.Errorf("source %d: %w", i, err)
		}
	}
	return nil
}

// IsZero returns true if the citation is empty.
func (c *Citation) IsZero() bool {
	return c.CitedText == "" && c.StartIndex == 0 && c.EndIndex == 0 && len(c.Sources) == 0
}

// CitationType is a citation that a model returned as part of its reply.
type CitationType int32

const (
	// CitationWebQuery is an query used as part of a web search.
	CitationWebQuery CitationType = iota + 1
	// CitationWeb is an URL from a web search.
	CitationWeb
	// CitationWebImage is an URL to an image from a web search.
	CitationWebImage
	// CitationDocument is from a document provided as input or explicitly referenced.
	CitationDocument
	// CitationTool is when the provider refers to the result of a tool call in its answer.
	CitationTool
)

// CitationSource represents a source that supports a citation.
type CitationSource struct {
	// Type indicates the source type.
	Type CitationType `json:"type,omitzero"`
	// ID is a unique identifier for the source (e.g., document ID, tool call ID).
	ID string `json:"id,omitzero"`
	// Title is the human-readable title of the source.
	Title string `json:"title,omitzero"`
	// URL is the web URL for the source, if applicable.
	URL string `json:"url,omitzero"`
	// Snippet is a snippet from the source, if applicable. It is the web search query for CitationWebQuery.
	Snippet string `json:"snippet,omitzero"`
	// StartCharIndex is the starting character position of the citation in the sourced document (0-based).
	StartCharIndex int64 `json:"start_index,omitzero"`
	// EndCharIndex is the ending character position of the citation in the sourced document (0-based, exclusive).
	EndCharIndex int64 `json:"end_index,omitzero"`
	// StartPageNumber is the starting page number of the citation in the sourced document (1-based).
	StartPageNumber int64 `json:"start_page_number,omitzero"`
	EndPageNumber   int64 `json:"end_page_number,omitzero"`
	// StartBlockIndex is the starting block index of the citation in the sourced document (0-based).
	StartBlockIndex int64 `json:"start_block_index,omitzero"`
	EndBlockIndex   int64 `json:"end_block_index,omitzero"`

	// Date is the date of the source, if applicable.
	Date string `json:"date,omitzero"`
	// Metadata contains additional source-specific information.
	// For document sources: document index, page numbers, etc.
	// For tool sources: tool output, function name, etc.
	// For web sources: encrypted index, search result info, etc.
	Metadata map[string]any `json:"metadata,omitzero"`

	_ struct{}
}

// Validate ensures the citation source is valid.
func (cs *CitationSource) Validate() error {
	if cs.ID == "" && cs.URL == "" && cs.Type != CitationWebQuery {
		return errors.New("citation source must have either ID or URL")
	}
	return nil
}

// IsZero returns true if the citation source is empty.
func (cs *CitationSource) IsZero() bool {
	return cs.Type == 0 && cs.ID == "" && cs.Title == "" && cs.URL == "" &&
		cs.Snippet == "" && cs.Date == "" && len(cs.Metadata) == 0
}

// Job is a pending job.
type Job string

// CacheEntry is one file (or GenSync request) cached on the provider for reuse.
type CacheEntry interface {
	GetID() string
	GetDisplayName() string
	GetExpiry() time.Time
}

// Models

// Model represents a served model by the provider.
//
// Use Provider.ListModels() to get a list of models.
type Model interface {
	GetID() string
	String() string
	// Context returns the number of tokens the model can process as input.
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

var (
	_ internal.Validatable = (*Citation)(nil)
	_ internal.Validatable = (*CitationSource)(nil)
	_ internal.Validatable = (*Message)(nil)
	_ internal.Validatable = (*Messages)(nil)
	_ internal.Validatable = (*RateLimit)(nil)
	_ internal.Validatable = (*Reply)(nil)
	_ internal.Validatable = (*Request)(nil)
	_ internal.Validatable = (*ToolCall)(nil)
	_ internal.Validatable = (*ToolCallResult)(nil)
)
