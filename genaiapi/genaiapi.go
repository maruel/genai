// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Package genaiapi provides a generic interface to interact with a LLM backend.
package genaiapi

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"reflect"
	"strings"

	"github.com/invopop/jsonschema"
)

type Validatable interface {
	Validate() error
}

// CompletionOptions is a list of frequent options supported by most
// CompletionProvider. Each provider is free to support more options through a
// specialized struct.
type CompletionOptions struct {
	// Seed for the random number generator. Default is 0 which means
	// non-deterministic. Some providers do not support this.
	Seed int64
	// Temperature adjust the creativity of the sampling. Generally between 0 and 2.
	Temperature float64
	// TopP adjusts correctness sampling between 0 and 1. The higher the more diverse the output.
	TopP float64
	// TopK adjusts sampling where only the N first candidates are considered. Some providers do not support this.
	TopK int64
	// MaxTokens is the maximum number of tokens to generate. Used to limit it
	// lower than the default maximum, for budget reasons.
	MaxTokens int64
	// Stop is the list of tokens to stop generation. Some providers do not support this.
	Stop []string
	// ReplyAsJSON enforces the output to be valid JSON, any JSON. Not all
	// providers support this, and even, only a limited number of models.
	// Increases latency and token use (cost). It is important to tell the model
	// to reply in JSON in the prompt itself.
	ReplyAsJSON bool
	// DecodeAs enforces a reply with a specific JSON structure. Not all
	// providers support this, and even, only a limited number of models.
	// Increases latency and token use (cost). It is important to tell the model
	// to reply in JSON in the prompt itself. The struct must be a pointer to a
	// struct that can be decoded by encoding/json and can have jsonschema tags.
	DecodeAs any
	// Tools is the list of tools that the LLM can request to call. Not all
	// providers support this, and even, only a limited number of models.
	// Increases latency and token use (cost).
	Tools []ToolDef
}

func (c *CompletionOptions) Validate() error {
	if c.Seed < 0 {
		return errors.New("invalid Seed: must be non-negative")
	}
	if c.Temperature < 0 || c.Temperature > 100 {
		return errors.New("invalid Temperature: must be [0, 100]")
	}
	if c.MaxTokens < 0 || c.MaxTokens > 1024*1024*1024 {
		return errors.New("invalid MaxTokens: must be [0, 1 GiB]")
	}
	if c.TopP < 0 || c.TopP > 1 {
		return errors.New("invalid TopP: must be [0, 1]")
	}
	if c.TopK < 0 || c.TopK > 1024 {
		return errors.New("invalid TopK: must be [0, 1024]")
	}
	if c.DecodeAs != nil {
		t := reflect.TypeOf(c.DecodeAs)
		if t.Kind() == reflect.Ptr {
			t = t.Elem()
			if _, ok := c.DecodeAs.(*jsonschema.Schema); ok {
				return errors.New("invalid DecodeAs: must be an actual struct serializable as JSON, not a *jsonschema.Schema")
			}
		}
		if t.Kind() != reflect.Struct {
			return errors.New("invalid DecodeAs: must be a struct")
		}
	}
	return nil
}

// CompletionProvider is the generic interface to interact with a LLM backend.
type CompletionProvider interface {
	// Completion runs completion synchronously.
	//
	// opts must be either nil, *CompletionOptions or a provider-specialized
	// option struct.
	Completion(ctx context.Context, msgs []Message, opts Validatable) (CompletionResult, error)
	// CompletionStream runs completion synchronously, streaming the results to channel replies.
	//
	// opts must be either nil, *CompletionOptions or a provider-specialized
	// option struct.
	CompletionStream(ctx context.Context, msgs []Message, opts Validatable, replies chan<- MessageChunk) error
}

// Model represents a served model by the provider.
type Model interface {
	GetID() string
	String() string
	Context() int64
}

// ModelProvider represents a provider that can list models.
type ModelProvider interface {
	ListModels(ctx context.Context) ([]Model, error)
}

// Role is one of the LLM known roles.
type Role string

// LLM known roles. Not all systems support all roles.
const (
	System    Role = "system"
	User      Role = "user"
	Assistant Role = "assistant"
)

// Validate ensures the role is valid.
func (r Role) Validate() error {
	switch r {
	case System, User, Assistant:
		return nil
	case "":
		return errors.New("a valid role is required")
	default:
		return fmt.Errorf("role %q is not supported", r)
	}
}

type ContentType string

const (
	Text      ContentType = "text"
	Document  ContentType = "document"
	ToolCalls ContentType = "tool_calls"
)

// Validate ensures the content type is valid.
func (c ContentType) Validate() error {
	switch c {
	case Text, Document, ToolCalls:
		return nil
	case "":
		return errors.New("a valid content type is required")
	default:
		return fmt.Errorf("content type %q is not supported", c)
	}
}

// Message is a message to send to the LLM as part of the exchange.
type Message struct {
	Role Role
	Type ContentType

	// Type == "text"
	// Text is the content of the text message.
	Text string

	// Type == "document"
	// In this case, one of Document or URL must be set.
	//
	// Filename is the name of the file. For many providers, only the extension
	// is relevant. They only use mime-type, which is derived from the filename's
	// extension. When an URL is provided, Filename is optional.
	Filename string
	// Document is raw document data. It is perfectly fine to use a
	// bytes.Buffer{}, bytes.NewReader() or *os.File.
	Document io.ReadSeeker
	// URL is the reference to the raw data. When set, the mime-type is derived from the URL.
	URL string

	// Type == "tool_calls"
	// ToolCalls is a list of tool calls that the LLM requested to make.
	ToolCalls []ToolCall

	_ struct{}
}

// Decode decodes the JSON message into the struct.
//
// Requires using either ReplyAsJSON or JSONSchema in the CompletionOptions.
func (m *Message) Decode(x any) error {
	if m.Type != Text {
		return errors.New("only text messages can be decoded as JSON")
	}
	d := json.NewDecoder(strings.NewReader(m.Text))
	d.DisallowUnknownFields()
	d.UseNumber()
	if err := d.Decode(x); err != nil {
		return fmt.Errorf("failed to decode message text as JSON: %w; content: %q", err, m.Text)
	}
	return nil
}

// CompletionResult is the result of a completion.
type CompletionResult struct {
	Message
	Usage
}

// Usage from the LLM provider.
type Usage struct {
	InputTokens  int64
	OutputTokens int64
}

// MessageChunk is a fragment of a message the LLM is sending back as part of the CompletionStream().
type MessageChunk struct {
	Role Role // Almost (?) always (?) Assistant.
	Type ContentType

	// Type == "text"
	Text string

	// Type == "tool_calls"
	// ToolCalls is a list of tool calls that the LLM requested to make.
	ToolCalls []ToolCall
}

// Validate ensures the message is valid.
func (m Message) Validate() error {
	if err := m.Role.Validate(); err != nil {
		return fmt.Errorf("field Role: %w", err)
	}
	if err := m.Type.Validate(); err != nil {
		return fmt.Errorf("field ContentType: %w", err)
	}
	switch m.Role {
	case System:
		if m.Type != Text {
			return errors.New("field Role is system but Type is not text")
		}
	}
	switch m.Type {
	case Text:
		if m.Text == "" {
			return errors.New("field Type is text but no text is provided")
		}
		if m.Filename != "" {
			return errors.New("field Filename is not supported for text")
		}
		if m.Document != nil {
			return errors.New("field Document is not supported for text")
		}
		if m.URL != "" {
			return errors.New("field URL is not supported for text")
		}
	case Document:
		if m.Text != "" {
			return errors.New("field Type is document but text is provided")
		}
		if m.Document == nil {
			if m.URL == "" {
				return errors.New("field Document or URL is required")
			}
		} else {
			if m.URL != "" {
				return errors.New("field Document and URL are mutually exclusive")
			}
			if m.Filename == "" {
				return errors.New("field Filename is required with Document")
			}
		}
	case ToolCalls:
		return errors.New("todo")
	}
	return nil
}

// Validate ensures the message chunk is valid.
func ValidateMessages(msgs []Message) error {
	var errs []error
	for i, m := range msgs {
		if err := m.Validate(); err != nil {
			errs = append(errs, fmt.Errorf("message %d: %w", i, err))
		}
		if m.Role == System && i != 0 {
			errs = append(errs, fmt.Errorf("message %d: system role is only allowed for the first message", i))
		}
	}
	return errors.Join(errs...)
}

// ToolDef describes a tool that the LLM can request to use.
type ToolDef struct {
	Name        string
	Description string
	Parameters  *jsonschema.Schema
}

// ToolCall is a tool call that the LLM requested to make.
type ToolCall struct {
	ID        string // Unique identifier for the tool call. Necessary for parallel tool calling.
	Name      string // Tool being called.
	Arguments string // encoded as JSON
}

// Decode decodes the JSON message into the struct.
//
// Requires using either ReplyAsJSON or JSONSchema in the CompletionOptions.
func (t *ToolCall) Decode(x any) error {
	d := json.NewDecoder(strings.NewReader(t.Arguments))
	d.DisallowUnknownFields()
	d.UseNumber()
	if err := d.Decode(x); err != nil {
		return fmt.Errorf("failed to decode tool call arguments: %w; arguments: %q", err, t.Arguments)
	}
	return nil
}
