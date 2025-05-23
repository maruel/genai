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
	"mime"
	"path"
	"path/filepath"
	"reflect"
	"strings"
	"unicode"

	"github.com/invopop/jsonschema"
)

// UnsupportedContinuableError is an error when an unsupported option is used but the operation still
// succeeded.
type UnsupportedContinuableError struct {
	// Unsupported is the list of arguments that were not supported and were silently ignored.
	Unsupported []string
}

func (u *UnsupportedContinuableError) Error() string {
	if len(u.Unsupported) == 0 {
		return "no unsupported options"
	}
	return fmt.Sprintf("unsupported options: %s", strings.Join(u.Unsupported, ", "))
}

// Generic

// ReflectedToJSON must be a pointer to a struct that can be decoded by
// encoding/json and can have jsonschema tags.
//
// It is recommended to use jsonscheme_description tags to describe each
// field or argument.
//
// Use jsonschema:"enum=..." to enforce a specific value within a set.
type ReflectedToJSON any

// Validatable is an interface to an object that can be validated.
type Validatable interface {
	Validate() error
}

// Chat

// ChatProvider is the generic interface to interact with a LLM backend.
type ChatProvider interface {
	// Chat runs completion synchronously.
	//
	// opts must be either nil, *ChatOptions or a provider-specialized
	// option struct.
	Chat(ctx context.Context, msgs Messages, opts Validatable) (ChatResult, error)
	// ChatStream runs completion synchronously, streaming the results to channel replies.
	//
	// opts must be either nil, *ChatOptions or a provider-specialized
	// option struct.
	//
	// No need to accumulate the replies into the result, the ChatResult contains the accumulated message.
	ChatStream(ctx context.Context, msgs Messages, opts Validatable, replies chan<- MessageFragment) (ChatResult, error)
	// ModelID returns the model currently used by the provider. It can be an empty string.
	ModelID() string
}

// ChatOptions is a list of frequent options supported by most ChatProvider.
// Each provider is free to support more options through a specialized struct.
type ChatOptions struct {
	// Options supported by all providers.

	// Temperature adjust the creativity of the sampling. Generally between 0 and 2.
	Temperature float64
	// TopP adjusts correctness sampling between 0 and 1. The higher the more diverse the output.
	TopP float64
	// MaxTokens is the maximum number of tokens to generate. Used to limit it
	// lower than the default maximum, for budget reasons.
	MaxTokens int64
	// SystemPrompt is the prompt to use for the system role.
	SystemPrompt string

	// Options supported only by some providers. Using them may cause the
	// chat operation to succeed while returning a UnsupportedContinuableError.

	// Seed for the random number generator. Default is 0 which means
	// non-deterministic.
	Seed int64
	// TopK adjusts sampling where only the N first candidates are considered.
	TopK int64
	// Stop is the list of tokens to stop generation.
	Stop []string

	// Options supported by a few providers and a few models on each, that will
	// slow down generation (increase latency) and will increase token use
	// (cost).

	// ReplyAsJSON enforces the output to be valid JSON, any JSON. It is
	// important to tell the model to reply in JSON in the prompt itself.
	ReplyAsJSON bool
	// DecodeAs enforces a reply with a specific JSON structure. It is important
	// to tell the model to reply in JSON in the prompt itself.
	DecodeAs ReflectedToJSON
	// Tools is the list of tools that the LLM can request to call.
	Tools []ToolDef
	// ToolCallRequest tells the LLM a tool call must be done.
	ToolCallRequest ToolCallRequest

	_ struct{}
}

// Validate ensures the completion options are valid.
func (c *ChatOptions) Validate() error {
	if c.Seed < 0 {
		return errors.New("field Seed: must be non-negative")
	}
	if c.Temperature < 0 || c.Temperature > 100 {
		return errors.New("field Temperature: must be [0, 100]")
	}
	if c.MaxTokens < 0 || c.MaxTokens > 1024*1024*1024 {
		return errors.New("field MaxTokens: must be [0, 1 GiB]")
	}
	if c.TopP < 0 || c.TopP > 1 {
		return errors.New("field TopP: must be [0, 1]")
	}
	if c.TopK < 0 || c.TopK > 1024 {
		return errors.New("field TopK: must be [0, 1024]")
	}
	if c.DecodeAs != nil {
		if err := validateReflectedToJSON(c.DecodeAs); err != nil {
			return fmt.Errorf("field DecodeAs: %w", err)
		}
	}
	names := map[string]int{}
	for i, t := range c.Tools {
		if err := t.Validate(); err != nil {
			return fmt.Errorf("tool %d: %w", i, err)
		}
		if j, ok := names[t.Name]; ok {
			return fmt.Errorf("tool %d: has name %q which is the same as tool %d", i, t.Name, j)
		}
		names[t.Name] = i
	}
	if len(c.Tools) == 0 && c.ToolCallRequest == ToolCallRequired {
		return fmt.Errorf("field ToolCallRequest is ToolCallRequired: Tools are required")
	}
	return nil
}

// ToolCallRequest determines if we want the LLM to request a tool call.
type ToolCallRequest int

const (
	// ToolCallAny is the default, the model is free to choose if a tool is called or not. For some models (like
	// llama family), it may be a bit too "tool call happy".
	ToolCallAny ToolCallRequest = iota
	// ToolCallRequired means a tool call is required. Don't forget to change the value after sending the
	// response!
	ToolCallRequired
	// ToolCallNone means that while tools are described, they should not be called. It is useful when a LLM did
	// tool calls, got the response and now it's time to generate some text to present to the end user.
	ToolCallNone
)

// ChatResult is the result of a completion.
type ChatResult struct {
	Message
	Usage
}

// Usage from the LLM provider.
type Usage struct {
	InputTokens       int64
	InputCachedTokens int64
	OutputTokens      int64

	// FinishReason indicates why the model stopped generating tokens.
	// Common values include "stop", "length", "content_filter", "tool_calls", etc.
	// The exact values depend on the specific provider.
	FinishReason string
}

func (u *Usage) String() string {
	return fmt.Sprintf("in: %d (cached %d), out: %d", u.InputTokens, u.InputCachedTokens, u.OutputTokens)
}

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
// The messages should be alternating between User and Assistant roles, or in the
// case of multi-user discussion, with different Users.
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
		// if i > 0 && msgs[i-1].Role == m.Role {
		// 	errs = append(errs, fmt.Errorf("message %d: role must alternate", i))
		// }
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
	Role Role   `json:"role,omitzero"`
	User string `json:"user,omitzero"` // Only used when Role == User. Only some provider (e.g. OpenAI, Groq, DeepSeek) support it.

	Contents []Content `json:"contents,omitzero"` // For example when the LLM replies with multiple content blocks, an explanation and a code block.

	// ToolCall is a tool call that the LLM requested to make.
	ToolCalls       []ToolCall       `json:"tool_calls,omitzero"`
	ToolCallResults []ToolCallResult `json:"tool_call_results,omitzero"`

	_ struct{}
}

// NewTextMessage is a shorthand function to create a Message with a single
// text block.
func NewTextMessage(role Role, text string) Message {
	return Message{Role: role, Contents: []Content{{Text: text}}}
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
			out = append(out, strings.TrimRightFunc(s, unicode.IsSpace))
		}
	}
	return strings.Join(out, "\n")
}

// Decode decodes the JSON message into the struct.
//
// Requires using either ReplyAsJSON or DecodeAs in the ChatOptions.
//
// Note: this doesn't verify the type is the same as specified in
// ChatOptions.DecodeAs.
func (m *Message) Decode(x any) error {
	s := m.AsText()
	if s == "" {
		return errors.New("only text messages can be decoded as JSON")
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

// Content is a block of content in the message meant to be visible in a
// chat setting.
//
// The content can be text or a document. The document may be audio, video,
// image, PDF or any other format.
type Content struct {
	// Only Text, Thinking, Opaque or the rest can be set.

	// Text is the content of the text message.
	Text string `json:"text,omitzero"`

	// Thinking is the reasoning done by the LLM.
	Thinking string `json:"thinking,omitzero"`

	// Opaque is added to keep continuity on the processing. A good example is Anthropic's extended thinking. It
	// must be kept during an exchange.
	//
	// A message with only Opaque set is valid.
	Opaque map[string]any `json:"opaque,omitzero"`

	// If Text and Thinking are not set, then, one of Document or URL must be set.

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
	} else if c.Thinking != "" || len(c.Opaque) != 0 {
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
			return namer.Name()
		}
	}
	return c.Filename
}

// ReadDocument reads the document content into memory.
func (c *Content) ReadDocument(maxSize int64) (string, []byte, error) {
	if c.Text != "" {
		return "", nil, errors.New("only document messages can be read as documents")
	}
	mimeType := mime.TypeByExtension(filepath.Ext(c.GetFilename()))
	if c.URL != "" {
		// Not all provider require a mime-type so do not error out.
		if mimeType == "" {
			mimeType = mime.TypeByExtension(filepath.Ext(path.Base(c.URL)))
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
		c.Document = bytes.NewReader(cc.Document)
	}
	return c.Validate()
}

// MessageFragment is a fragment of a message the LLM is sending back as part
// of the ChatStream().
//
// The role is always implicitly the assistant.
type MessageFragment struct {
	TextFragment string

	ThinkingFragment string
	Opaque           map[string]any

	Filename         string
	DocumentFragment []byte

	// ToolCall is a tool call that the LLM requested to make.
	ToolCall ToolCall

	_ struct{}
}

func (m *MessageFragment) IsZero() bool {
	return m.TextFragment == "" && m.ThinkingFragment == "" && len(m.Opaque) == 0 && m.Filename == "" && len(m.DocumentFragment) == 0 && m.ToolCall.IsZero()
}

// Accumulate adds a MessageFragment to the message being streamed.
func (m *Message) Accumulate(mf MessageFragment) error {
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

	if mf.Filename != "" || mf.DocumentFragment != nil {
		if len(m.Contents) != 0 {
			if lastBlock := &m.Contents[len(m.Contents)-1]; lastBlock.Filename != "" {
				_, _ = lastBlock.Document.(*bytesBuffer).Write(mf.DocumentFragment)
				return nil
			}
		}
		m.Contents = append(m.Contents, Content{Filename: mf.Filename, Document: &bytesBuffer{d: mf.DocumentFragment}})
		return nil
	}

	if mf.ToolCall.Name != "" {
		m.ToolCalls = append(m.ToolCalls, mf.ToolCall)
		return nil
	}

	// Nothing to accumulate. It should be an error but there are bugs where the system hangs.
	return nil
}

type bytesBuffer struct {
	d   []byte
	pos int
}

func (b *bytesBuffer) Read(p []byte) (int, error) {
	n := copy(p, b.d[b.pos:])
	if n == 0 {
		return 0, io.EOF
	}
	b.pos += n
	return n, nil
}

func (b *bytesBuffer) Seek(offset int64, whence int) (int64, error) {
	var p int64
	if whence == io.SeekCurrent {
		offset += int64(b.pos)
		whence = io.SeekStart
	}
	switch whence {
	case io.SeekEnd:
		offset = int64(len(b.d)) - offset
		fallthrough
	case io.SeekStart:
		if offset < 0 || offset > int64(len(b.d)) {
			return p, errors.New("out of bound")
		}
		p = offset
		b.pos = int(p)
	default:
		return p, fmt.Errorf("unknown whence %d", whence)
	}
	return p, nil
}

func (b *bytesBuffer) Write(p []byte) (int, error) {
	b.d = append(b.d, p...)
	return len(p), nil
}

// Tools

// ToolDef describes a tool that the LLM can request to use.
type ToolDef struct {
	// Name must be unique among all tools.
	Name string
	// Description must be a LLM-friendly short description of the tool.
	Description string
	// Callback is the function to call with the inputs.
	// It must accept a context.Context one struct pointer as input: (ctx context.Context, input *struct{}). The
	// struct must use json_schema to be serializable as JSON.
	// It must return the result and an error: (string, error).
	Callback any

	_ struct{}
}

// Validate ensures the tool definition is valid.
func (t *ToolDef) Validate() error {
	if t.Name == "" {
		return errors.New("field Name: required")
	}
	if t.Description == "" {
		return errors.New("field Description: required")
	}
	if t.Callback != nil {
		cbType := reflect.TypeOf(t.Callback)
		if cbType.Kind() != reflect.Func {
			return errors.New("field Callback: must be a function")
		}
		if cbType.NumIn() != 2 {
			return errors.New("field Callback: must accept exactly two parameters: (context.Context, input *struct{})")
		}
		paramType := cbType.In(0)
		if paramType != reflect.TypeFor[context.Context]() {
			return fmt.Errorf("field Callback: must accept exactly two parameters, first that is a context.Context, not a %q", paramType.Name())
		}
		paramType = cbType.In(1)
		if paramType.Kind() != reflect.Ptr {
			return fmt.Errorf("field Callback: must accept exactly two parameters, second that is a pointer to a struct, not a %q", paramType.Name())
		}
		paramType = paramType.Elem()
		if paramType.Kind() != reflect.Struct {
			return fmt.Errorf("field Callback: must accept exactly two parameters, second that is a pointer to a struct, not a %q", paramType.Name())
		}
		if err := validateReflectedToJSON(paramType); err != nil {
			return fmt.Errorf("field Callback: must accept exactly two parameters, second that is a pointer to a struct that has valid json schema: %w", err)
		}
		if cbType.NumOut() != 2 {
			return errors.New("field Callback: must return exactly two values: (string, error)")
		}
		if cbType.Out(0).Kind() != reflect.String {
			return fmt.Errorf("field Callback: must return a string first, not %q", cbType.Out(0).Name())
		}
		if !isErrorType(cbType.Out(1)) {
			return fmt.Errorf("field Callback: must return an error second, not %q", cbType.Out(1).Name())
		}
	}
	return nil
}

// InputSchema returns the json schema for the input argument of the callback.
func (t *ToolDef) InputSchema() *jsonschema.Schema {
	// This function assumes Validate() was called.
	// No need to set an ID on the struct, it's unnecessary data that may confuse the tool.
	r := jsonschema.Reflector{Anonymous: true, DoNotReference: true}
	return r.ReflectFromType(reflect.TypeOf(t.Callback).In(1))
}

// ToolCall is a tool call that the LLM requested to make.
type ToolCall struct {
	ID        string `json:"id,omitzero"`        // Unique identifier for the tool call. Necessary for parallel tool calling.
	Name      string `json:"name,omitzero"`      // Tool being called.
	Arguments string `json:"arguments,omitzero"` // encoded as JSON

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

// Models

// ModelProvider represents a provider that can list models.
type ModelProvider interface {
	ListModels(ctx context.Context) ([]Model, error)
}

// Model represents a served model by the provider.
type Model interface {
	GetID() string
	String() string
	Context() int64
}

// Private

func validateReflectedToJSON(r ReflectedToJSON) error {
	tp := reflect.TypeOf(r)
	if tp.Kind() == reflect.Ptr {
		tp = tp.Elem()
		if _, ok := r.(*jsonschema.Schema); ok {
			return errors.New("must be an actual struct serializable as JSON, not a *jsonschema.Schema")
		}
	}
	if tp.Kind() != reflect.Struct {
		return fmt.Errorf("must be a struct, not %T", r)
	}
	return nil
}

// isErrorType returns true if the type is of error type.
func isErrorType(t reflect.Type) bool {
	return t == reflect.TypeOf((*error)(nil)).Elem()
}

var (
	_ Validatable = (*ChatOptions)(nil)
	_ Validatable = (*Role)(nil)
	_ Validatable = (*Messages)(nil)
	_ Validatable = (*Message)(nil)
	_ Validatable = (*Content)(nil)
	_ Validatable = (*ToolDef)(nil)
	_ Validatable = (*ToolCall)(nil)
	_ Validatable = (*ToolCallResult)(nil)
)
