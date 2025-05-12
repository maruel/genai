// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package cerebras

import (
	"encoding/json"
	"testing"

	"github.com/google/go-cmp/cmp"
)

func TestContentsUnmarshalJSON(t *testing.T) {
	tests := []struct {
		name      string
		json      string
		want      Contents
		wantError bool
	}{
		{
			name: "string_content",
			json: `"hello world"`,
			want: Contents{{
				Type: "text",
				Text: "hello world",
			}},
			wantError: false,
		},
		{
			name: "content_array",
			json: `[{"type":"text","text":"hello world"}]`,
			want: Contents{{
				Type: "text",
				Text: "hello world",
			}},
			wantError: false,
		},
		{
			name:      "empty_array",
			json:      `[]`,
			want:      Contents{},
			wantError: false,
		},
		{
			name:      "invalid_json",
			json:      `{"invalid": true}`,
			want:      nil,
			wantError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var got Contents
			err := json.Unmarshal([]byte(tt.json), &got)

			if (err != nil) != tt.wantError {
				t.Errorf("UnmarshalJSON() error = %v, wantError %v", err, tt.wantError)
				return
			}

			if !tt.wantError {
				if diff := cmp.Diff(tt.want, got); diff != "" {
					t.Errorf("UnmarshalJSON() mismatch (-want +got):\n%s", diff)
				}
			}
		})
	}
}

func TestMessageWithContent(t *testing.T) {
	tests := []struct {
		name      string
		json      string
		wantRole  string
		wantType  string
		wantText  string
		wantError bool
	}{
		{
			name:      "string_content",
			json:      `{"role":"assistant","content":"hello world"}`,
			wantRole:  "assistant",
			wantType:  "text",
			wantText:  "hello world",
			wantError: false,
		},
		{
			name:      "array_content",
			json:      `{"role":"user","content":[{"type":"text","text":"hello world"}]}`,
			wantRole:  "user",
			wantType:  "text",
			wantText:  "hello world",
			wantError: false,
		},
		{
			name:      "empty_content",
			json:      `{"role":"user","content":[]}`,
			wantRole:  "user",
			wantText:  "",
			wantError: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var msg Message
			err := json.Unmarshal([]byte(tt.json), &msg)

			if (err != nil) != tt.wantError {
				t.Errorf("UnmarshalJSON() error = %v, wantError %v", err, tt.wantError)
				return
			}

			if !tt.wantError {
				if msg.Role != tt.wantRole {
					t.Errorf("Role = %v, want %v", msg.Role, tt.wantRole)
				}

				if len(msg.Content) > 0 {
					if msg.Content[0].Type != tt.wantType && tt.wantType != "" {
						t.Errorf("Content[0].Type = %v, want %v", msg.Content[0].Type, tt.wantType)
					}
					if msg.Content[0].Text != tt.wantText {
						t.Errorf("Content[0].Text = %v, want %v", msg.Content[0].Text, tt.wantText)
					}
				} else if tt.wantText != "" {
					t.Errorf("Content is empty, want text %v", tt.wantText)
				}
			}
		})
	}
}
