// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package genaiapi

import (
	"strings"
	"testing"
)

func TestCompletionOptions_Validate(t *testing.T) {
	o := CompletionOptions{
		Seed:        0,
		Temperature: 0.5,
		MaxTokens:   100,
		TopP:        0.5,
		TopK:        10,
	}
	if err := o.Validate(); err != nil {
		t.Fatal(err)
	}
}

func TestCompletionOptions_Validate_error(t *testing.T) {
	tests := []struct {
		name    string
		options CompletionOptions
		errMsg  string
	}{
		{
			name: "Invalid Seed",
			options: CompletionOptions{
				Seed: -1,
			},
			errMsg: "invalid Seed: must be non-negative",
		},
		{
			name: "Invalid Temperature",
			options: CompletionOptions{
				Temperature: -1,
			},
			errMsg: "invalid Temperature: must be [0, 100]",
		},
		{
			name: "Invalid MaxTokens",
			options: CompletionOptions{
				MaxTokens: 1024*1024*1024 + 1,
			},
			errMsg: "invalid MaxTokens: must be [0, 1 GiB]",
		},
		{
			name: "Invalid TopP",
			options: CompletionOptions{
				TopP: -1,
			},
			errMsg: "invalid TopP: must be [0, 1]",
		},
		{
			name: "Invalid TopK",
			options: CompletionOptions{
				TopK: 1025,
			},
			errMsg: "invalid TopK: must be [0, 1024]",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if err := tt.options.Validate(); err == nil || err.Error() != tt.errMsg {
				t.Fatalf("expected error %q, got %v", tt.errMsg, err)
			}
		})
	}
}

func TestRole_Validate(t *testing.T) {
	for _, role := range []Role{System, User, Assistant} {
		if err := role.Validate(); err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
	}
	for _, role := range []Role{"invalid", ""} {
		if err := role.Validate(); err == nil {
			t.Fatalf("expected error, got nil")
		}
	}
}

func TestContentType_Validate(t *testing.T) {
	for _, contentType := range []ContentType{Text, Document, ToolCalls} {
		if err := contentType.Validate(); err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
	}
	for _, contentType := range []ContentType{"invalid", ""} {
		if err := contentType.Validate(); err == nil {
			t.Fatalf("expected error, got nil")
		}
	}
}

func TestMessage_Validate(t *testing.T) {
	tests := []struct {
		name    string
		message Message
	}{
		{
			name: "Valid system message",
			message: Message{
				Role: System,
				Type: Text,
				Text: "System instruction",
			},
		},
		{
			name: "Valid user text message",
			message: Message{
				Role: User,
				Type: Text,
				Text: "Hello",
			},
		},
		{
			name: "Valid user document message",
			message: Message{
				Role:     User,
				Type:     Document,
				Filename: "document.txt",
				Document: strings.NewReader("document content"),
			},
		},
		{
			name: "Valid assistant message",
			message: Message{
				Role: Assistant,
				Type: Text,
				Text: "I can help with that",
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if err := tt.message.Validate(); err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
		})
	}
}

func TestMessage_Validate_error(t *testing.T) {
	tests := []struct {
		name    string
		message Message
		errMsg  string
	}{
		{
			name: "Invalid system message with wrong type",
			message: Message{
				Role:     System,
				Type:     Document,
				Filename: "document.txt",
				Document: strings.NewReader("document content"),
			},
			errMsg: "field Role is system but Type is not text",
		},
		{
			name: "Missing role",
			message: Message{
				Type: Text,
				Text: "Content",
			},
			errMsg: "field Role: a valid role is required",
		},
		{
			name: "Unsupported role",
			message: Message{
				Role: "UnsupportedRole",
				Type: Text,
				Text: "Content",
			},
			errMsg: "field Role: role \"UnsupportedRole\" is not supported",
		},
		{
			name: "Missing type",
			message: Message{
				Role: User,
				Text: "Content",
			},
			errMsg: "field ContentType: a valid content type is required",
		},
		{
			name: "Unsupported type",
			message: Message{
				Role: User,
				Type: "UnsupportedType",
				Text: "Content",
			},
			errMsg: "field ContentType: content type \"UnsupportedType\" is not supported",
		},
		{
			name: "Text type without text",
			message: Message{
				Role: User,
				Type: Text,
			},
			errMsg: "field Type is text but no text is provided",
		},
		{
			name: "Text type with invalid Filename flag",
			message: Message{
				Role:     User,
				Type:     Text,
				Text:     "Content",
				Filename: "document.txt",
			},
			errMsg: "field Filename is not supported for text",
		},
		{
			name: "Text type with invalid Document",
			message: Message{
				Role:     User,
				Type:     Text,
				Text:     "Content",
				Document: strings.NewReader("document content"),
			},
			errMsg: "field Document is not supported for text",
		},
		{
			name: "Text type with invalid MimeType",
			message: Message{
				Role: User,
				Type: Text,
				Text: "Content",
				URL:  "http://localhost",
			},
			errMsg: "field URL is not supported for text",
		},
		{
			name: "Document type with invalid Text",
			message: Message{
				Role:     User,
				Type:     Document,
				Text:     "Content",
				Filename: "document.txt",
				Document: strings.NewReader("document content"),
			},
			errMsg: "field Type is document but text is provided",
		},
		{
			name: "Document type without Document",
			message: Message{
				Role:     User,
				Type:     Document,
				Filename: "document.txt",
			},
			errMsg: "field Document or URL is required",
		},
		{
			name: "Document type without Filename",
			message: Message{
				Role:     User,
				Type:     Document,
				Document: strings.NewReader("document content"),
			},
			errMsg: "field Filename is required with Document",
		},
		{
			name: "Document type with URL",
			message: Message{
				Role:     User,
				Type:     Document,
				Filename: "document.txt",
				Document: strings.NewReader("document content"),
				URL:      "http://localhost",
			},
			errMsg: "field Document and URL are mutually exclusive",
		},
		{
			name: "ToolCalls not implemented",
			message: Message{
				Role: Assistant,
				Type: ToolCalls,
			},
			errMsg: "todo",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.message.Validate()
			if err == nil || err.Error() != tt.errMsg {
				t.Fatalf("expected error %q, got %q", tt.errMsg, err.Error())
			}
		})
	}
}

func TestValidateMessages(t *testing.T) {
	messages := []Message{
		{
			Role: System,
			Type: Text,
			Text: "System instruction",
		},
		{
			Role: User,
			Type: Text,
			Text: "Hello",
		},
		{
			Role: Assistant,
			Type: Text,
			Text: "I can help with that",
		},
	}
	if err := ValidateMessages(messages); err != nil {
		t.Fatal(err)
	}
}

func TestValidateMessages_error(t *testing.T) {
	tests := []struct {
		name     string
		messages []Message
		errMsg   string
	}{
		{
			name: "Invalid system message with wrong type",
			messages: []Message{
				{
					Role:     System,
					Type:     Document,
					Filename: "document.txt",
					Document: strings.NewReader("document content"),
				},
			},
			errMsg: "message 0: field Role is system but Type is not text",
		},
		{
			name: "Invalid system message with wrong type",
			messages: []Message{
				{
					Role: System,
					Type: Text,
					Text: "System instruction",
				},
				{
					Role:     System,
					Type:     Document,
					Filename: "document.txt",
					Document: strings.NewReader("document content"),
				},
			},
			errMsg: "message 1: field Role is system but Type is not text\nmessage 1: system role is only allowed for the first message",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if err := ValidateMessages(tt.messages); err == nil || err.Error() != tt.errMsg {
				t.Fatalf("expected error %q, got %v", tt.errMsg, err)
			}
		})
	}
}
