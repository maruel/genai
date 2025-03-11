// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Package genaiapi provides a generic interface to interact with a LLM backend.
package genaiapi

import (
	"context"
	"errors"
	"fmt"
)

// CompletionOptions is a list of frequent options supported by most CompletionProvider.
type CompletionOptions struct {
	Seed        int64   // Seed for the random number generator. Default is 0 which means non-deterministic.
	Temperature float64 // Temperature of the sampling.
	MaxTokens   int64   // Maximum number of tokens to generate.
}

// CompletionProvider is the generic interface to interact with a LLM backend.
type CompletionProvider interface {
	Completion(ctx context.Context, msgs []Message, opts any) (string, error)
	CompletionStream(ctx context.Context, msgs []Message, opts any, words chan<- string) error
}

type Model interface {
	GetID() string
	String() string
	Context() int64
}

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

	// Type == "document"
	// Inline determines if the data is embedded in the message or externally referenced.
	Inline bool
	// Data is raw document data.
	Data []byte
	// MimeType is the MIME type of the data if relevant.
	MimeType string
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
		if m.Inline {
			return errors.New("field Inline is not supported for text")
		}
		if len(m.Data) != 0 {
			return errors.New("field Data is not supported for text")
		}
		if m.MimeType != "" {
			return errors.New("field MimeType is not supported for text")
		}
	case Document:
		if m.Text != "" {
			return errors.New("field Type is document but text is provided")
		}
		if len(m.Data) == 0 {
			return errors.New("field Data is required")
		}
		if m.MimeType == "" {
			return errors.New("field MimeType is required")
		}
	case "":
		return errors.New("field Type is required")
	default:
		return fmt.Errorf("field Type %q is not supported", m.Type)
	}
	return nil
}
