// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Package genaiapi provides a generic interface to interact with a LLM backend.
package genaiapi

import (
	"context"
	"errors"
	"fmt"
	"io"
)

// CompletionOptions is a list of frequent options supported by most
// CompletionProvider. Each provider is free to support more options through a
// specialized struct.
type CompletionOptions struct {
	Seed        int64      // Seed for the random number generator. Default is 0 which means non-deterministic.
	Temperature float64    // Temperature of the sampling.
	MaxTokens   int64      // Maximum number of tokens to generate.
	ReplyAsJSON bool       // If true, the output is JSON. If false, the output is text. It is important to tell the model to reply in JSON.
	JSONSchema  JSONSchema // Enforces a reply JSON format. Not all providers support this.
	Tools       []ToolDef  // List of tools that the LLM can request to call. Not all providers support this.

	_ struct{}
}

// CompletionProvider is the generic interface to interact with a LLM backend.
type CompletionProvider interface {
	// Completion runs completion synchronously.
	//
	// opts must be either nil, *CompletionOptions or a provider-specialized
	// option struct.
	Completion(ctx context.Context, msgs []Message, opts any) (Message, error)
	// CompletionStream runs completion synchronously, streaming the results to channel replies.
	//
	// opts must be either nil, *CompletionOptions or a provider-specialized
	// option struct.
	CompletionStream(ctx context.Context, msgs []Message, opts any, replies chan<- MessageChunk) error
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
	// The following has to be revised.
	Tool           Role = "tool"
	AvailableTools Role = "available_tools"
	ToolCall       Role = "tool_call"
	ToolCallResult Role = "tool_call_result"
)

type ContentType string

const (
	Text     ContentType = "text"
	Document ContentType = "document"
)

// Message is a message to send to the LLM as part of the exchange.
type Message struct {
	Role Role
	Type ContentType

	// Type == "text"
	// Text is the content of the text message.
	Text string

	// Type == "document". In this case, one of Document or URL must be set.
	// Filename is the name of the file. For many providers, only the extension
	// is relevant. They only use mime-type, which is derived from the filename's
	// extension. When an URL is provided, Filename is optional.
	Filename string
	// Document is raw document data. It is perfectly fine to use a
	// bytes.Buffer{}, bytes.NewReader() or *os.File.
	Document io.ReadSeeker
	// URL is the reference to the raw data. When set, the mime-type is derived from the URL.
	URL string

	_ struct{}
}

type MessageChunk struct {
	Role Role
	Type ContentType
	Text string
}

// Validate ensures the message is valid.
func (m Message) Validate() error {
	switch m.Role {
	case System:
		if m.Type != Text {
			return errors.New("field Role is system but Type is not text")
		}
	case User, Assistant, Tool, AvailableTools, ToolCall, ToolCallResult:
	case "":
		return errors.New("field Role is required")
	default:
		return fmt.Errorf("field Role %q is not supported", m.Role)
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
	case "":
		return errors.New("field Type is required")
	default:
		return fmt.Errorf("field Type %q is not supported", m.Type)
	}
	return nil
}

// JSONSchema is a minimalist representation of a normalized JSON schema to be
// used to force the LLM to return a specific JSON schema.
//
// It doesn't implement dependentSchemas, patternProperties,
// additionalProperties, unevaluatedProperties, allOf, nor if/then/else.
type JSONSchema struct {
	Type string `json:"type,omitzero"` // "object", "array", "string", "integer", "number", "boolean", "null" or empty for enum.

	// Type == "object"
	Properties    map[string]JSONSchema `json:"properties,omitzero"`
	Required      []string              `json:"required,omitzero"`
	MinProperties int64                 `json:"minProperties,omitzero"`
	MaxProperties int64                 `json:"maxProperties,omitzero"`

	// Type == "array"
	Items *JSONSchema `json:"items,omitzero"`

	// Type == "string", "integer", "boolean"
	Description string `json:"description,omitzero"`

	// Type == "string"
	Pattern   string `json:"pattern,omitzero"` // regexp
	MinLength int64  `json:"minLength,omitzero"`
	MaxLength int64  `json:"maxLength,omitzero"`

	// Type == "integer", "number"
	// TODO: This is strictly incorrect. It should be a union of int64 and
	// float64 and they should be pointers.
	Minimum          int64 `json:"minimum,omitzero"`
	ExclusiveMinimum int64 `json:"exclusiveMinimum,omitzero"`
	Maximum          int64 `json:"maximum,omitzero"`
	ExclusiveMaximum int64 `json:"exclusiveMaximum,omitzero"`
	MultipleOf       int64 `json:"multipleOf,omitzero"`

	Enum []any `json:"enum,omitzero"`

	_ struct{}
}

func (j *JSONSchema) IsZero() bool {
	return j.Type == "" && len(j.Enum) == 0
}

// Tool describes a tool that the LLM can request to use.
type ToolDef struct {
	Name        string
	Description string
	Parameters  JSONSchema
}
