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
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
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
	ChatStream(ctx context.Context, msgs Messages, opts Validatable, replies chan<- MessageFragment) (Usage, error)
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
	// ToolCallRequired tells the LLM a tool call must be done.
	ToolCallRequired bool

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
	for i, t := range c.Tools {
		if err := t.Validate(); err != nil {
			return fmt.Errorf("tool %d: %w", i, err)
		}
	}
	if len(c.Tools) == 0 && c.ToolCallRequired {
		return fmt.Errorf("field ToolCallRequired: Tools are required")
	}
	return nil
}

// ChatResult is the result of a completion.
type ChatResult struct {
	Message
	Usage

	// FinishReason indicates why the model stopped generating tokens.
	// Common values include "stop", "length", "content_filter", "tool_calls", etc.
	// The exact values depend on the specific provider.
	FinishReason string

	_ struct{}
}

// Usage from the LLM provider.
type Usage struct {
	InputTokens       int64
	InputCachedTokens int64
	OutputTokens      int64
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
func (msgs Messages) Validate() error {
	var errs []error
	for i, m := range msgs {
		if err := m.Validate(); err != nil {
			errs = append(errs, fmt.Errorf("message %d: %w", i, err))
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
	Role Role
	User string // Only used when Role == User. Only some provider (e.g. OpenAI, Groq, DeepSeek) support it.

	Contents []Content // For example when the LLM replies with multiple content blocks, an explanation and a code block.

	// ToolCall is a tool call that the LLM requested to make.
	ToolCalls []ToolCall

	// TODO: Tool replies

	_ struct{}
}

// NewTextMessage is a shorthand function to create a Message with a single
// text block.
func NewTextMessage(role Role, text string) Message {
	return Message{Role: role, Contents: []Content{{Text: text}}}
}

// Validate ensures the messages are valid.
func (m *Message) Validate() error {
	var errs []error
	if err := m.Role.Validate(); err != nil {
		errs = append(errs, fmt.Errorf("field Role: %w", err))
	}
	if m.User != "" {
		errs = append(errs, errors.New("field User: not supported yet"))
	}
	if len(m.Contents) == 0 {
		errs = append(errs, errors.New("field Contents: required"))
	}
	for i, b := range m.Contents {
		if err := b.Validate(); err != nil {
			errs = append(errs, fmt.Errorf("block %d: %w", i, err))
		}
	}
	return errors.Join(errs...)
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

// Content is a block of content in the message meant to be visible in a
// chat setting.
//
// The content can be text or a document. The document may be audio, video,
// image, PDF or any other format.
type Content struct {
	// Only Text, Thinking, Opaque or the rest can be set.

	// Text is the content of the text message.
	Text string

	// Thinking is the reasoning done by the LLM.
	Thinking string

	// Opaque is added to keep continuity on the processing. A good example is Anthropic's extended thinking. It
	// must be kept during an exchange.
	//
	// A message with only Opaque set is valid.
	Opaque map[string]any

	// If Text and Thinking are not set, then, one of Document or URL must be set.

	// Filename is the name of the file. For many providers, only the extension
	// is relevant. They only use mime-type, which is derived from the filename's
	// extension. When an URL is provided or when the object provided to Document
	// implements a method with the signature `Name() string`, like an
	// `*os.File`, Filename is optional.
	Filename string
	// Document is raw document data. It is perfectly fine to use a
	// bytes.Buffer{}, bytes.NewReader() or *os.File.
	Document io.ReadSeeker
	// URL is the reference to the raw data. When set, the mime-type is derived from the URL.
	URL string

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

// MessageFragment is a fragment of a message the LLM is sending back as part
// of the ChatStream().
type MessageFragment struct {
	TextFragment     string
	ThinkingFragment string

	Filename         string
	DocumentFragment []byte

	// ToolCall is a tool call that the LLM requested to make.
	ToolCall ToolCall

	// FinishReason indicates why the model stopped generating tokens.
	// Common values include "stop", "length", "content_filter", "tool_calls", etc.
	// The exact values depend on the specific provider.
	// This is only populated in the final MessageFragment of a stream.
	FinishReason string

	_ struct{}
}

// Accumulate accumulates the message fragment into the list of messages.
//
// The assumption is that the fragment is always a message from the Assistant.
func (m *MessageFragment) Accumulate(msgs Messages) (Messages, error) {
	if len(msgs) == 0 {
		// First message.
		return append(msgs, m.toMessage()), nil
	}
	lastMsg := &msgs[len(msgs)-1]
	if lastMsg.Role != Assistant {
		// First reply.
		return append(msgs, m.toMessage()), nil
	}

	// Generally the first message fragment.
	if m.ThinkingFragment != "" {
		if len(lastMsg.Contents) == 0 {
			lastMsg.Contents = append(lastMsg.Contents, Content{Thinking: m.ThinkingFragment})
			return msgs, nil
		}
		if lastBlock := &lastMsg.Contents[len(lastMsg.Contents)-1]; lastBlock.Thinking != "" {
			lastBlock.Thinking += m.ThinkingFragment
			return msgs, nil
		}
		return append(msgs, m.toMessage()), nil
	}

	// Content.
	if m.TextFragment != "" {
		if len(lastMsg.Contents) == 0 {
			lastMsg.Contents = append(lastMsg.Contents, Content{Text: m.TextFragment})
			return msgs, nil
		}
		if lastBlock := &lastMsg.Contents[len(lastMsg.Contents)-1]; lastBlock.Text != "" {
			lastBlock.Text += m.TextFragment
			return msgs, nil
		}
		return append(msgs, m.toMessage()), nil
	}
	if m.DocumentFragment != nil {
		return nil, fmt.Errorf("cannot accumulate documents yet")
	}
	if m.ToolCall.Name != "" {
		lastMsg.ToolCalls = append(lastMsg.ToolCalls, m.ToolCall)
		return msgs, nil
	}

	// Generally the last message fragment.
	if m.FinishReason != "" {
		// TODO: lastMsg.FinishReason = m.FinishReason
		return msgs, nil
	}
	// Nothing to accumulate. It should be an error but there are bugs where the system hangs.
	return msgs, nil
}

// toMessage converts the fragment to a standalone message.
func (m *MessageFragment) toMessage() Message {
	if m.ThinkingFragment != "" {
		return Message{Role: Assistant, Contents: []Content{{Thinking: m.ThinkingFragment}}}
	}
	if m.TextFragment != "" {
		return NewTextMessage(Assistant, m.TextFragment)
	}
	if m.DocumentFragment != nil {
		// TODO: We need a seekable memory buffer, bytes.Buffer{} is not seekable.
		return Message{
			Role:     Assistant,
			Contents: []Content{{Document: strings.NewReader(m.TextFragment)}},
		}
	}
	if m.ToolCall.Name != "" {
		return Message{
			Role:      Assistant,
			ToolCalls: []ToolCall{m.ToolCall},
		}
	}
	return Message{}
}

// Tools

// ToolDef describes a tool that the LLM can request to use.
type ToolDef struct {
	// Name must be unique among all tools.
	Name string
	// Description must be a LLM-friendly short description of the tool.
	Description string
	// InputsAs enforces a tool call with a specific JSON structure for
	// arguments.
	InputsAs ReflectedToJSON
	// Callback is the function to call with the inputs in InputAs. It must return a string.
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
	if t.InputsAs != nil {
		if err := validateReflectedToJSON(t.InputsAs); err != nil {
			return fmt.Errorf("field InputsAs: %w", err)
		}
	}
	if t.Callback != nil {
		cbType := reflect.TypeOf(t.Callback)
		if cbType.Kind() != reflect.Func {
			return errors.New("field Callback: must be a function")
		}
		if cbType.NumOut() != 1 {
			return errors.New("field Callback: must return exactly one value")
		}
		if cbType.Out(0).Kind() != reflect.String {
			return errors.New("field Callback: must return a string")
		}
		if t.InputsAs != nil {
			if cbType.NumIn() != 1 {
				return errors.New("field Callback: must accept exactly one parameter")
			}
			inputType := reflect.TypeOf(t.InputsAs)
			if inputType.Kind() == reflect.Ptr {
				inputType = inputType.Elem()
			}
			paramType := cbType.In(0)
			isParamPtr := false
			if paramType.Kind() == reflect.Ptr {
				isParamPtr = true
				paramType = paramType.Elem()
			}
			if paramType != inputType {
				return fmt.Errorf("field Callback: parameter type %v does not match InputsAs type %v", cbType.In(0), reflect.TypeOf(t.InputsAs))
			}
			if reflect.TypeOf(t.InputsAs).Kind() == reflect.Ptr && !isParamPtr {
				return errors.New("field Callback: InputsAs is a pointer but parameter is not")
			}
		}
	}
	return nil
}

// ToolCall is a tool call that the LLM requested to make.
type ToolCall struct {
	ID        string // Unique identifier for the tool call. Necessary for parallel tool calling.
	Name      string // Tool being called.
	Arguments string // encoded as JSON

	_ struct{}
}

// Decode decodes the JSON tool call.
//
// This function doesn't validate x is the same as InputsAs in the ToolDef.
func (t *ToolCall) Decode(x any) error {
	d := json.NewDecoder(strings.NewReader(t.Arguments))
	d.DisallowUnknownFields()
	d.UseNumber()
	if err := d.Decode(x); err != nil {
		return fmt.Errorf("failed to decode tool call arguments: %w; arguments: %q", err, t.Arguments)
	}
	return nil
}

// Call invokes the ToolDef.Callback with arguments from the ToolCall, returning the result string.
//
// It decodes the ToolCall.Arguments into a new instance of ToolDef.InputsAs and passes it to the Callback.
func (t *ToolCall) Call(toolDef *ToolDef) (string, error) {
	if toolDef == nil {
		return "", errors.New("toolDef is nil")
	}
	if toolDef.Callback == nil {
		return "", errors.New("toolDef.Callback is nil")
	}
	if toolDef.InputsAs == nil {
		return reflect.ValueOf(toolDef.Callback).Call(nil)[0].String(), nil
	}

	inputType := reflect.TypeOf(toolDef.InputsAs)
	var input reflect.Value
	if inputType.Kind() == reflect.Ptr {
		input = reflect.New(inputType.Elem())
	} else {
		input = reflect.New(inputType)
	}
	if err := t.Decode(input.Interface()); err != nil {
		return "", fmt.Errorf("failed to decode arguments: %w", err)
	}
	if inputType.Kind() != reflect.Ptr {
		input = input.Elem()
	}
	return reflect.ValueOf(toolDef.Callback).Call([]reflect.Value{input})[0].String(), nil
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

var (
	_ Validatable = (*ChatOptions)(nil)
	_ Validatable = (*Role)(nil)
	_ Validatable = (*Messages)(nil)
	_ Validatable = (*Message)(nil)
	_ Validatable = (*Content)(nil)
	_ Validatable = (*ToolDef)(nil)
)
