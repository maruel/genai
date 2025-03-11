// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package genaiapi

import (
	"testing"
)

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
				Data:     []byte("document content"),
				MimeType: "text/plain",
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
		{
			name: "Valid tool message",
			message: Message{
				Role: Tool,
				Type: Text,
				Text: "Tool response",
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

func TestMessage_Validate_Error(t *testing.T) {
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
				Data:     []byte("data"),
				MimeType: "application/pdf",
			},
			errMsg: "field Role is system but Type is not text",
		},
		{
			name: "Missing role",
			message: Message{
				Type: Text,
				Text: "Content",
			},
			errMsg: "field Role is required",
		},
		{
			name: "Unsupported role",
			message: Message{
				Role: "UnsupportedRole",
				Type: Text,
				Text: "Content",
			},
			errMsg: "field Role \"UnsupportedRole\" is not supported",
		},
		{
			name: "Missing type",
			message: Message{
				Role: User,
				Text: "Content",
			},
			errMsg: "field Type is required",
		},
		{
			name: "Unsupported type",
			message: Message{
				Role: User,
				Type: "UnsupportedType",
				Text: "Content",
			},
			errMsg: "field Type \"UnsupportedType\" is not supported",
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
			name: "Text type with invalid Inline flag",
			message: Message{
				Role:   User,
				Type:   Text,
				Text:   "Content",
				Inline: true,
			},
			errMsg: "field Inline is not supported for text",
		},
		{
			name: "Text type with invalid Data",
			message: Message{
				Role: User,
				Type: Text,
				Text: "Content",
				Data: []byte("data"),
			},
			errMsg: "field Data is not supported for text",
		},
		{
			name: "Text type with invalid MimeType",
			message: Message{
				Role:     User,
				Type:     Text,
				Text:     "Content",
				MimeType: "text/plain",
			},
			errMsg: "field MimeType is not supported for text",
		},
		{
			name: "Document type with invalid Text",
			message: Message{
				Role:     User,
				Type:     Document,
				Text:     "Content",
				Data:     []byte("data"),
				MimeType: "text/plain",
			},
			errMsg: "field Type is document but text is provided",
		},
		{
			name: "Document type without Data",
			message: Message{
				Role:     User,
				Type:     Document,
				MimeType: "text/plain",
			},
			errMsg: "field Data is required",
		},
		{
			name: "Document type without MimeType",
			message: Message{
				Role: User,
				Type: Document,
				Data: []byte("data"),
			},
			errMsg: "field MimeType is required",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.message.Validate()
			if err == nil || err.Error() != tt.errMsg {
				t.Fatalf("expected error %q, got %v", tt.errMsg, err)
			}
		})
	}
}
