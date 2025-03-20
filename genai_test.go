// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package genai

import (
	"errors"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/invopop/jsonschema"
)

func TestCompletionOptions_Validate(t *testing.T) {
	t.Run("valid", func(t *testing.T) {
		o := ChatOptions{
			Seed:        1,
			Temperature: 0.5,
			TopP:        0.5,
			TopK:        10,
			MaxTokens:   100,
			Stop:        []string{"stop"},
			ReplyAsJSON: true,
			DecodeAs:    struct{}{},
			Tools: []ToolDef{
				{
					Name:        "tool",
					Description: "do stuff",
				},
			},
		}
		if err := o.Validate(); err != nil {
			t.Fatalf("unexpected error: %q", err)
		}
		o = ChatOptions{
			DecodeAs: &struct{}{},
		}
		if err := o.Validate(); err != nil {
			t.Fatalf("unexpected error: %q", err)
		}
	})
	t.Run("error", func(t *testing.T) {
		tests := []struct {
			name    string
			options ChatOptions
			errMsg  string
		}{
			{
				name: "Invalid Seed",
				options: ChatOptions{
					Seed: -1,
				},
				errMsg: "field Seed: must be non-negative",
			},
			{
				name: "Invalid Temperature",
				options: ChatOptions{
					Temperature: -1,
				},
				errMsg: "field Temperature: must be [0, 100]",
			},
			{
				name: "Invalid MaxTokens",
				options: ChatOptions{
					MaxTokens: 1024*1024*1024 + 1,
				},
				errMsg: "field MaxTokens: must be [0, 1 GiB]",
			},
			{
				name: "Invalid TopP",
				options: ChatOptions{
					TopP: -1,
				},
				errMsg: "field TopP: must be [0, 1]",
			},
			{
				name: "Invalid TopK",
				options: ChatOptions{
					TopK: 1025,
				},
				errMsg: "field TopK: must be [0, 1024]",
			},
			{
				name: "Invalid DecodeAs jsonschema.Schema",
				options: ChatOptions{
					DecodeAs: &jsonschema.Schema{},
				},
				errMsg: "field DecodeAs: must be an actual struct serializable as JSON, not a *jsonschema.Schema",
			},
			{
				name: "Invalid DecodeAs string",
				options: ChatOptions{
					DecodeAs: "string",
				},
				errMsg: "field DecodeAs: must be a struct, not string",
			},
		}
		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				if err := tt.options.Validate(); err == nil || err.Error() != tt.errMsg {
					t.Fatalf("\nwant %q\ngot  %q", tt.errMsg, err)
				}
			})
		}
	})
}

func TestRole_Validate(t *testing.T) {
	for _, role := range []Role{User, Assistant, Computer} {
		if err := role.Validate(); err != nil {
			t.Fatalf("unexpected error: %q", err)
		}
	}
	for _, role := range []Role{"invalid", ""} {
		if err := role.Validate(); err == nil {
			t.Fatalf("expected error, got nil")
		}
	}
}

func TestMessages_Validate(t *testing.T) {
	t.Run("valid", func(t *testing.T) {
		m := Messages{
			NewTextMessage(User, "Hello"),
			NewTextMessage(Assistant, "I can help with that"),
		}
		if err := m.Validate(); err != nil {
			t.Fatalf("unexpected error: %q", err)
		}
	})
	t.Run("error", func(t *testing.T) {
		tests := []struct {
			name     string
			messages Messages
			errMsg   string
		}{
			{
				name: "Invalid messages",
				messages: Messages{
					{
						Role:     User,
						Contents: []Content{{Text: "Hi", Filename: "hi.txt"}},
					},
					{
						Role:     User,
						Contents: []Content{{}},
					},
				},
				errMsg: "message 0: block 0: field Filename can't be used along Text\nmessage 1: block 0: no content",
			},
		}
		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				if err := tt.messages.Validate(); err == nil || err.Error() != tt.errMsg {
					t.Fatalf("\nwant %q\ngot  %q", tt.errMsg, err)
				}
			})
		}
	})
}

func TestMessage_Validate(t *testing.T) {
	t.Run("valid", func(t *testing.T) {
		tests := []struct {
			name    string
			message Message
		}{
			{
				name:    "Valid user text message",
				message: NewTextMessage(User, "Hello"),
			},
			{
				name: "Valid user document message",
				message: Message{
					Role: User,
					Contents: []Content{
						{
							Filename: "document.txt",
							Document: strings.NewReader("document content"),
						},
					},
				},
			},
			{
				name:    "Valid assistant message",
				message: NewTextMessage(Assistant, "I can help with that"),
			},
		}
		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				if err := tt.message.Validate(); err != nil {
					t.Fatalf("unexpected error: %q", err)
				}
			})
		}
	})
	t.Run("error", func(t *testing.T) {
		tests := []struct {
			name    string
			message Message
			errMsg  string
		}{
			{
				name:    "empty",
				message: Message{},
				errMsg:  "field Role: role \"\" is not supported\nfield Contents: required",
			},
			{
				name:    "user",
				message: Message{Role: User, User: "Joe", Contents: []Content{{Text: "Hi"}}},
				errMsg:  "field User: not supported yet",
			},
		}
		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				err := tt.message.Validate()
				if err == nil || err.Error() != tt.errMsg {
					t.Fatalf("\nwant %q\ngot  %q", tt.errMsg, err)
				}
			})
		}
	})
}

func TestContent_Validate(t *testing.T) {
	t.Run("valid", func(t *testing.T) {
		tests := []struct {
			name string
			in   Content
		}{
			{
				name: "Valid text block",
				in:   Content{Text: "Hello"},
			},
			{
				name: "Valid document block",
				in: Content{
					Filename: "document.txt",
					Document: strings.NewReader("document content"),
				},
			},
		}
		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				if err := tt.in.Validate(); err != nil {
					t.Fatalf("unexpected error: %q", err)
				}
			})
		}
	})
	// TODO: error
}

func TestContent_ReadDocument(t *testing.T) {
	t.Run("valid", func(t *testing.T) {
		c := Content{
			Filename: "document.txt",
			Document: strings.NewReader("document content"),
		}
		mime, got, err := c.ReadDocument(1000)
		if err != nil {
			t.Fatalf("unexpected error: %q", err)
		}
		if mime != "text/plain; charset=utf-8" {
			t.Fatalf("unexpected mime type: %q", mime)
		}
		if string(got) != "document content" {
			t.Fatalf("unexpected content: %q", got)
		}
	})
	// TODO: error
}

func TestContent_Decode(t *testing.T) {
	t.Run("valid", func(t *testing.T) {
		m := NewTextMessage(Assistant, "{\"key\": \"value\"}")
		if err := m.Contents[0].Decode(&struct{ Key string }{}); err != nil {
			t.Fatalf("unexpected error: %q", err)
		}
	})
	t.Run("error", func(t *testing.T) {
		tests := []struct {
			name    string
			message Message
			errMsg  string
		}{
			{
				name:    "Invalid JSON message",
				message: NewTextMessage(Assistant, "invalid"),
				errMsg:  "failed to decode message text as JSON: invalid character 'i' looking for beginning of value; content: \"invalid\"",
			},
			{
				name: "Invalid DecodeAs",
				message: Message{
					Role: Assistant,
					Contents: []Content{
						{
							Document: strings.NewReader("document content"),
						},
					},
				},
				errMsg: "only text messages can be decoded as JSON",
			},
		}
		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				if err := tt.message.Contents[0].Decode("invalid"); err == nil || err.Error() != tt.errMsg {
					t.Fatalf("\nwant %q\ngot  %q", tt.errMsg, err)
				}
			})
		}
	})
}

func TestMessageFragment_Accumulate(t *testing.T) {
	tests := []struct {
		name string
		msgs Messages
		f    MessageFragment
		want Messages
	}{
		{
			name: "Join assistant text",
			msgs: Messages{NewTextMessage(Assistant, "Hello")},
			f:    MessageFragment{TextFragment: " world"},
			want: Messages{NewTextMessage(Assistant, "Hello world")},
		},
		{
			name: "User then assistant",
			msgs: Messages{NewTextMessage(User, "Make me a sandwich")},
			f:    MessageFragment{TextFragment: "No"},
			want: Messages{
				NewTextMessage(User, "Make me a sandwich"),
				NewTextMessage(Assistant, "No"),
			},
		},
		{
			name: "Document then text",
			msgs: Messages{
				{
					Role:     Assistant,
					Contents: []Content{{Filename: "document.txt", Document: &buffer{"document content"}}},
				},
			},
			f: MessageFragment{TextFragment: "No"},
			want: Messages{
				{
					Role: Assistant,
					Contents: []Content{
						{
							Filename: "document.txt",
							Document: &buffer{"document content"},
						},
					},
				},
				NewTextMessage(Assistant, "No"),
			},
		},
		{
			name: "Tool then text",
			msgs: Messages{{Role: Assistant, ToolCalls: []ToolCall{{Name: "tool"}}}},
			f:    MessageFragment{TextFragment: "No"},
			want: Messages{
				{
					Role: Assistant,
					// Merge together.
					Contents:  []Content{{Text: "No"}},
					ToolCalls: []ToolCall{{Name: "tool"}},
				},
			},
		},
		{
			name: "Tool then tool",
			msgs: Messages{{Role: Assistant, ToolCalls: []ToolCall{{Name: "tool"}}}},
			f:    MessageFragment{ToolCall: ToolCall{Name: "tool2"}},
			want: Messages{
				{
					Role: Assistant,
					// Merge together.
					ToolCalls: []ToolCall{{Name: "tool"}, {Name: "tool2"}},
				},
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := tt.f.Accumulate(tt.msgs)
			if err != nil {
				t.Fatalf("unexpected error: %q", err)
			}
			if diff := cmp.Diff(tt.want, got); diff != "" {
				t.Fatalf("unexpected result: %s", diff)
			}
		})
	}
}

func TestMessageFragment_toMessage(t *testing.T) {
	tests := []struct {
		name string
		f    MessageFragment
		want Message
	}{
		{
			name: "Text",
			f:    MessageFragment{TextFragment: "Hello"},
			want: NewTextMessage(Assistant, "Hello"),
		},
		/* TODO: Implement document while streaming.
		{
			name: "Document",
			f: MessageFragment{
				Filename:         "document.txt",
				DocumentFragment: []byte("document content"),
			},
			want: Message{
				Role:   Assistant,
				Contents: []Content{{Document: &buffer{"document content"}}},
			},
		},
		*/
		{
			name: "Tool",
			f: MessageFragment{
				ToolCall: ToolCall{Name: "tool"},
			},
			want: Message{
				Role:      Assistant,
				ToolCalls: []ToolCall{{Name: "tool"}},
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := tt.f.toMessage()
			if diff := cmp.Diff(tt.want, got); diff != "" {
				t.Fatalf("unexpected result: %s", diff)
			}
		})
	}
}

func TestToolDef_Validate_error(t *testing.T) {
	tests := []struct {
		name    string
		toolDef ToolDef
		errMsg  string
	}{
		{
			name: "Missing Name",
			toolDef: ToolDef{
				Description: "do stuff",
			},
			errMsg: "field Name: required",
		},
		{
			name: "Missing Description",
			toolDef: ToolDef{
				Name: "tool",
			},
			errMsg: "field Description: required",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if err := tt.toolDef.Validate(); err == nil || err.Error() != tt.errMsg {
				t.Fatalf("\nwant %q\ngot  %q", tt.errMsg, err)
			}
		})
	}
}

func TestToolCall_Validate(t *testing.T) {
	// TODO.
}

func TestToolCall_Decode(t *testing.T) {
	tc := ToolCall{
		ID:        "call1",
		Name:      "tool",
		Arguments: "{\"round\":true}",
	}
	var expected struct {
		Round bool `json:"round"`
	}
	if err := tc.Decode(&expected); err != nil {
		t.Fatalf("unexpected error: %q", err)
	}
}

type buffer struct {
	Data string
}

func (b *buffer) Read(p []byte) (int, error) {
	return copy(p, b.Data), nil
}

func (b *buffer) Close() error {
	return nil
}

func (b *buffer) Seek(offset int64, whence int) (int64, error) {
	if whence == 0 {
		return 0, nil
	}
	if whence == 2 {
		return int64(len(b.Data)), nil
	}
	return 0, errors.New("unsupported whence")
}
