// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package genai

import (
	"context"
	"errors"
	"fmt"
	"io"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/invopop/jsonschema"
	"github.com/maruel/genai/internal/bb"
)

func TestOptionsText_Validate(t *testing.T) {
	t.Run("valid", func(t *testing.T) {
		o := OptionsText{
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
		o = OptionsText{
			DecodeAs: &struct{}{},
		}
		if err := o.Validate(); err != nil {
			t.Fatalf("unexpected error: %q", err)
		}
	})
	t.Run("error", func(t *testing.T) {
		tests := []struct {
			name    string
			options OptionsText
			errMsg  string
		}{
			{
				name: "Invalid Seed",
				options: OptionsText{
					Seed: -1,
				},
				errMsg: "field Seed: must be non-negative",
			},
			{
				name: "Invalid Temperature",
				options: OptionsText{
					Temperature: -1,
				},
				errMsg: "field Temperature: must be [0, 100]",
			},
			{
				name: "Invalid MaxTokens",
				options: OptionsText{
					MaxTokens: 1024*1024*1024 + 1,
				},
				errMsg: "field MaxTokens: must be [0, 1 GiB]",
			},
			{
				name: "Invalid TopP",
				options: OptionsText{
					TopP: -1,
				},
				errMsg: "field TopP: must be [0, 1]",
			},
			{
				name: "Invalid TopK",
				options: OptionsText{
					TopK: 1025,
				},
				errMsg: "field TopK: must be [0, 1024]",
			},
			{
				name: "Invalid DecodeAs jsonschema.Schema",
				options: OptionsText{
					DecodeAs: &jsonschema.Schema{},
				},
				errMsg: "field DecodeAs: must be an actual struct serializable as JSON, not a *jsonschema.Schema",
			},
			{
				name: "Invalid DecodeAs string",
				options: OptionsText{
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
			NewTextMessage("Hello"),
			Message{Role: Assistant, Contents: []Content{{Text: "I can help with that"}}},
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
				errMsg: "message 0: content 0: field Filename can't be used along Text\nmessage 1: content 0: no content",
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
				message: NewTextMessage("Hello"),
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
				message: Message{Role: Assistant, Contents: []Content{{Text: "I can help with that"}}},
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
				errMsg:  "field Role: role \"\" is not supported\nat least one of fields Contents, ToolCalls or ToolCallsResults is required",
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

func TestMessage_Decode(t *testing.T) {
	t.Run("valid", func(t *testing.T) {
		m := Message{Role: Assistant, Contents: []Content{{Text: "{\"key\": \"value\"}"}}}
		if err := m.Decode(&struct{ Key string }{}); err != nil {
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
				message: Message{Role: Assistant, Contents: []Content{{Text: "invalid"}}},
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
				errMsg: "only text messages can be decoded as JSON, can't decode {\"role\":\"assistant\",\"contents\":[{\"document\":\"ZG9jdW1lbnQgY29udGVudA==\"}]}",
			},
		}
		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				if err := tt.message.Decode("invalid"); err == nil || err.Error() != tt.errMsg {
					t.Fatalf("\nwant %q\ngot  %q", tt.errMsg, err)
				}
			})
		}
	})
}

func TestAccumulateContentFragment(t *testing.T) {
	tests := []struct {
		name string
		msgs Messages
		f    ContentFragment
		want Messages
	}{
		{
			name: "Join assistant text",
			msgs: Messages{Message{Role: Assistant, Contents: []Content{{Text: "Hello"}}}},
			f:    ContentFragment{TextFragment: " world"},
			want: Messages{Message{Role: Assistant, Contents: []Content{{Text: "Hello world"}}}},
		},
		{
			name: "User then assistant",
			msgs: Messages{NewTextMessage("Make me a sandwich")},
			f:    ContentFragment{TextFragment: "No"},
			want: Messages{
				NewTextMessage("Make me a sandwich"),
				Message{Role: Assistant, Contents: []Content{{Text: "No"}}},
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
			f: ContentFragment{TextFragment: "No"},
			want: Messages{
				{
					Role: Assistant,
					Contents: []Content{
						{
							Filename: "document.txt",
							Document: &buffer{"document content"},
						},
						{
							Text: "No",
						},
					},
				},
			},
		},
		{
			name: "Tool then text",
			msgs: Messages{{Role: Assistant, ToolCalls: []ToolCall{{Name: "tool"}}}},
			f:    ContentFragment{TextFragment: "No"},
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
			f:    ContentFragment{ToolCall: ToolCall{Name: "tool2"}},
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
			got := make(Messages, len(tt.msgs))
			copy(got, tt.msgs)

			// Similar logic to GenStreamWithToolCallLoop
			var assistantMsg *Message
			if len(got) == 0 || got[len(got)-1].Role != Assistant {
				got = append(got, Message{Role: Assistant})
				assistantMsg = &got[len(got)-1]
			} else {
				assistantMsg = &got[len(got)-1]
			}

			err := assistantMsg.Accumulate(tt.f)
			if err != nil {
				t.Fatalf("unexpected error: %q", err)
			}

			if diff := cmp.Diff(tt.want, got); diff != "" {
				t.Fatalf("unexpected result: %s", diff)
			}
		})
	}
}

func TestMessage_Accumulate(t *testing.T) {
	tests := []struct {
		name     string
		message  Message
		fragment ContentFragment
		want     Message
	}{
		{
			name:     "Text",
			message:  Message{Role: Assistant},
			fragment: ContentFragment{TextFragment: "Hello"},
			want:     Message{Role: Assistant, Contents: []Content{{Text: "Hello"}}},
		},
		{
			name:    "Document",
			message: Message{Role: Assistant},
			fragment: ContentFragment{
				Filename:         "document.txt",
				DocumentFragment: []byte("document content"),
			},
			want: Message{
				Role:     Assistant,
				Contents: []Content{{Filename: "document.txt", Document: &bb.BytesBuffer{D: []byte("document content")}}},
			},
		},
		{
			name:     "Tool",
			message:  Message{Role: Assistant},
			fragment: ContentFragment{ToolCall: ToolCall{Name: "tool"}},
			want: Message{
				Role:      Assistant,
				ToolCalls: []ToolCall{{Name: "tool"}},
			},
		},
		{
			name:     "Add text to existing text",
			message:  Message{Role: Assistant, Contents: []Content{{Text: "Hello"}}},
			fragment: ContentFragment{TextFragment: " world"},
			want:     Message{Role: Assistant, Contents: []Content{{Text: "Hello world"}}},
		},
		{
			name:     "Add thinking to existing thinking",
			message:  Message{Role: Assistant, Contents: []Content{{Thinking: "I think "}}},
			fragment: ContentFragment{ThinkingFragment: "therefore I am"},
			want:     Message{Role: Assistant, Contents: []Content{{Thinking: "I think therefore I am"}}},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			message := tt.message
			err := message.Accumulate(tt.fragment)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if diff := cmp.Diff(tt.want, message); diff != "" {
				t.Fatalf("unexpected result: %s", diff)
			}
		})
	}
}

// Define test structs for validation
type TestInputStruct struct {
	Name string
}

type TestDifferentStruct struct {
	Age int
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
		{
			name: "Callback not a function",
			toolDef: ToolDef{
				Name:        "tool",
				Description: "do stuff",
				Callback:    "not a function",
			},
			errMsg: "field Callback: must be a function",
		},
		{
			name: "Callback returns wrong type first",
			toolDef: ToolDef{
				Name:        "tool",
				Description: "do stuff",
				Callback:    func(ctx context.Context, b *TestInputStruct) (int, error) { return 1, nil },
			},
			errMsg: "field Callback: must return a string first, not \"int\"",
		},
		{
			name: "Callback returns wrong type second",
			toolDef: ToolDef{
				Name:        "tool",
				Description: "do stuff",
				Callback:    func(ctx context.Context, b *TestInputStruct) (string, string) { return "", "" },
			},
			errMsg: "field Callback: must return an error second, not \"string\"",
		},
		{
			name: "Callback with wrong parameter count",
			toolDef: ToolDef{
				Name:        "tool",
				Description: "do stuff",
				Callback:    func(a, b, c *TestInputStruct) string { return "" },
			},
			errMsg: "field Callback: must accept exactly two parameters: (context.Context, input *struct{})",
		},
		{
			name: "parameter not pointer",
			toolDef: ToolDef{
				Name:        "tool",
				Description: "do stuff",
				Callback:    func(ctx context.Context, input TestInputStruct) string { return "" },
			},
			errMsg: "field Callback: must accept exactly two parameters, second that is a pointer to a struct, not a \"TestInputStruct\"",
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

func TestToolDef_Validate_success(t *testing.T) {
	tests := []struct {
		name    string
		toolDef ToolDef
	}{
		{
			name: "Valid ToolDef with function and pointer InputsAs",
			toolDef: ToolDef{
				Name:        "tool",
				Description: "do stuff",
				Callback:    func(ctx context.Context, input *TestInputStruct) (string, error) { return "", nil },
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if err := tt.toolDef.Validate(); err != nil {
				t.Fatalf("Expected no error, got: %v", err)
			}
		})
	}
}

func TestToolCall_Validate(t *testing.T) {
	// TODO.
}

func TestToolCall_Call(t *testing.T) {
	type CalculateInput struct {
		A int `json:"a"`
		B int `json:"b"`
	}

	t.Run("with struct arguments", func(t *testing.T) {
		ctx := t.Context()
		structTool := ToolDef{
			Name:        "calculateTool",
			Description: "A tool that performs a calculation",
			Callback: func(ctx context.Context, input *CalculateInput) (string, error) {
				return fmt.Sprintf("%d + %d = %d", input.A, input.B, input.A+input.B), nil
			},
		}
		if err := structTool.Validate(); err != nil {
			t.Fatal(err)
		}

		tc := ToolCall{
			ID:        "call2",
			Name:      "calculateTool",
			Arguments: `{"a": 5, "b": 3}`,
		}
		if err := tc.Validate(); err != nil {
			t.Fatal(err)
		}

		result, err := tc.Call(ctx, []ToolDef{structTool})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if result != "5 + 3 = 8" {
			t.Fatalf("unexpected result: got %q, want %q", result, "5 + 3 = 8")
		}
	})

	t.Run("with pointer arguments", func(t *testing.T) {
		ctx := t.Context()
		pointerTool := ToolDef{
			Name:        "pointerTool",
			Description: "A tool that takes a pointer argument",
			Callback: func(ctx context.Context, input *CalculateInput) (string, error) {
				return fmt.Sprintf("%d * %d = %d", input.A, input.B, input.A*input.B), nil
			},
		}
		if err := pointerTool.Validate(); err != nil {
			t.Fatal(err)
		}

		tc := ToolCall{
			ID:        "call3",
			Name:      "pointerTool",
			Arguments: `{"a": 5, "b": 3}`,
		}
		if err := tc.Validate(); err != nil {
			t.Fatal(err)
		}

		result, err := tc.Call(ctx, []ToolDef{pointerTool})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if result != "5 * 3 = 15" {
			t.Fatalf("unexpected result: got %q, want %q", result, "5 * 3 = 15")
		}
	})

	t.Run("with callback returning error", func(t *testing.T) {
		ctx := t.Context()
		errorTool := ToolDef{
			Name:        "errorTool",
			Description: "A tool that returns an error",
			Callback: func(ctx context.Context, input *CalculateInput) (string, error) {
				return "operation failed", errors.New("intentional error from callback")
			},
		}
		if err := errorTool.Validate(); err != nil {
			t.Fatal(err)
		}

		tc := ToolCall{
			ID:        "call5",
			Name:      "errorTool",
			Arguments: `{"a": 5, "b": 3}`,
		}
		if err := tc.Validate(); err != nil {
			t.Fatal(err)
		}

		result, err := tc.Call(ctx, []ToolDef{errorTool})
		if err == nil {
			t.Fatal("expected error, got nil")
		}
		if err.Error() != "intentional error from callback" {
			t.Fatalf("unexpected error message: got %q, want %q", err.Error(), "intentional error from callback")
		}
		if result != "operation failed" {
			t.Fatalf("unexpected result: got %q, want %q", result, "operation failed")
		}
	})

	t.Run("with invalid arguments", func(t *testing.T) {
		ctx := t.Context()
		structTool := ToolDef{
			Name:        "calculateTool",
			Description: "A tool that performs a calculation",
			Callback: func(ctx context.Context, input *CalculateInput) (string, error) {
				t.Error("unexpected call")
				return fmt.Sprintf("%d + %d = %d", input.A, input.B, input.A+input.B), nil
			},
		}
		if err := structTool.Validate(); err != nil {
			t.Fatal(err)
		}

		tc := ToolCall{
			ID:        "call4",
			Name:      "calculateTool",
			Arguments: `{"a": "not an integer", "b": 3}`,
		}
		if err := tc.Validate(); err != nil {
			t.Fatal(err)
		}

		result, err := tc.Call(ctx, []ToolDef{structTool})
		if err == nil {
			t.Fatalf("expected error, got nil")
		}
		if want := ""; result != want {
			t.Fatal(result)
		}
	})
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

func TestUnsupportedContinuableError(t *testing.T) {
	// Create an UnsupportedContinuableError
	unsupported := []string{"TopK", "ThinkingBudget"}
	uce := &UnsupportedContinuableError{Unsupported: unsupported}

	// Test the Error method
	expectedMsg := "unsupported options: TopK, ThinkingBudget"
	if uce.Error() != expectedMsg {
		t.Errorf("Expected error message %q, got %q", expectedMsg, uce.Error())
	}

	// Test empty unsupported list
	uce = &UnsupportedContinuableError{Unsupported: nil}
	expectedMsg = "no unsupported options"
	if uce.Error() != expectedMsg {
		t.Errorf("Expected error message %q, got %q", expectedMsg, uce.Error())
	}
}

// UnmarshalJSON Tests

func TestMessage_UnmarshalJSON(t *testing.T) {
	t.Run("valid", func(t *testing.T) {
		tests := []struct {
			name string
			json string
			want Message
		}{
			{
				name: "User text message",
				json: `{"role": "user", "contents": [{"text": "Hello"}]}`,
				want: Message{
					Role:     User,
					Contents: []Content{{Text: "Hello"}},
				},
			},
			{
				name: "Assistant message with tool call",
				json: `{"role": "assistant", "tool_calls": [{"id": "1", "name": "tool", "arguments": "{}"}]}`,
				want: Message{
					Role:      Assistant,
					ToolCalls: []ToolCall{{ID: "1", Name: "tool", Arguments: "{}"}},
				},
			},
			{
				name: "Computer message with tool result",
				json: `{"role": "computer", "tool_call_results": [{"id": "1", "name": "tool", "result": "success"}]}`,
				want: Message{
					Role:            Computer,
					ToolCallResults: []ToolCallResult{{ID: "1", Name: "tool", Result: "success"}},
				},
			},
		}
		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				var got Message
				if err := got.UnmarshalJSON([]byte(tt.json)); err != nil {
					t.Fatalf("unexpected error: %v", err)
				}
				if diff := cmp.Diff(tt.want, got); diff != "" {
					t.Errorf("Message mismatch (-want +got):\n%s", diff)
				}
			})
		}
	})

	t.Run("error", func(t *testing.T) {
		tests := []struct {
			name   string
			json   string
			errMsg string
		}{
			{
				name:   "Invalid JSON",
				json:   `{"role": "user", "contents": invalid}`,
				errMsg: "invalid character 'i' looking for beginning of value",
			},
			{
				name:   "Unknown field",
				json:   `{"role": "user", "contents": [{"text": "Hi"}], "unknown_field": "value"}`,
				errMsg: "json: unknown field \"unknown_field\"",
			},
			{
				name:   "Invalid role",
				json:   `{"role": "invalid", "contents": [{"text": "Hi"}]}`,
				errMsg: "field Role: role \"invalid\" is not supported",
			},
			// Note: Empty message (just role) passes UnmarshalJSON validation,
			// but would fail full Validate() method. UnmarshalJSON only does partial validation.
			{
				name:   "User with User field",
				json:   `{"role": "user", "user": "joe", "contents": [{"text": "Hi"}]}`,
				errMsg: "field User: not supported yet",
			},
		}
		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				var got Message
				err := got.UnmarshalJSON([]byte(tt.json))
				if err == nil {
					t.Fatal("expected error, got nil")
				}
				if !strings.Contains(err.Error(), tt.errMsg) {
					t.Errorf("expected error to contain %q, got %q", tt.errMsg, err.Error())
				}
			})
		}
	})
}

func TestContent_UnmarshalJSON(t *testing.T) {
	t.Run("valid", func(t *testing.T) {
		tests := []struct {
			name string
			json string
			want Content
		}{
			{
				name: "Text content",
				json: `{"text": "Hello world"}`,
				want: Content{Text: "Hello world"},
			},
			{
				name: "Thinking content",
				json: `{"thinking": "Let me think about this"}`,
				want: Content{Thinking: "Let me think about this"},
			},
			{
				name: "Opaque content",
				json: `{"opaque": {"key": "value", "num": 42}}`,
				want: Content{Opaque: map[string]any{"key": "value", "num": float64(42)}},
			},
			{
				name: "URL content",
				json: `{"filename": "image.jpg", "url": "https://example.com/image.jpg"}`,
				want: Content{Filename: "image.jpg", URL: "https://example.com/image.jpg"},
			},
			{
				name: "Document content",
				json: `{"filename": "doc.txt", "document": "SGVsbG8gV29ybGQ="}`,
				want: Content{
					Filename: "doc.txt",
					Document: strings.NewReader("Hello World"),
				},
			},
		}
		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				var got Content
				if err := got.UnmarshalJSON([]byte(tt.json)); err != nil {
					t.Fatalf("unexpected error: %v", err)
				}
				// For Document comparison, read the content since we can't directly compare io.ReadSeeker
				if tt.want.Document != nil {
					wantData, _ := io.ReadAll(tt.want.Document)
					gotData, _ := io.ReadAll(got.Document)
					if string(wantData) != string(gotData) {
						t.Errorf("Document content mismatch: want %q, got %q", string(wantData), string(gotData))
					}
					// Reset Document field for comparison
					tt.want.Document = nil
					got.Document = nil
				}
				if diff := cmp.Diff(tt.want, got); diff != "" {
					t.Errorf("Content mismatch (-want +got):\n%s", diff)
				}
			})
		}
	})

	t.Run("error", func(t *testing.T) {
		tests := []struct {
			name   string
			json   string
			errMsg string
		}{
			{
				name:   "Invalid JSON",
				json:   `{"text": invalid}`,
				errMsg: "invalid character 'i' looking for beginning of value",
			},
			{
				name:   "Unknown field",
				json:   `{"text": "Hi", "unknown_field": "value"}`,
				errMsg: "json: unknown field \"unknown_field\"",
			},
			{
				name:   "Text and Thinking together",
				json:   `{"text": "Hello", "thinking": "Let me think"}`,
				errMsg: "field Thinking can't be used along Text",
			},
			{
				name:   "Document without filename",
				json:   `{"document": "SGVsbG8="}`,
				errMsg: "field Filename is required with Document when not implementing Name()",
			},
			{
				name:   "Empty content",
				json:   `{}`,
				errMsg: "no content",
			},
		}
		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				var got Content
				err := got.UnmarshalJSON([]byte(tt.json))
				if err == nil {
					t.Fatal("expected error, got nil")
				}
				if !strings.Contains(err.Error(), tt.errMsg) {
					t.Errorf("expected error to contain %q, got %q", tt.errMsg, err.Error())
				}
			})
		}
	})
}

func TestToolCall_UnmarshalJSON(t *testing.T) {
	t.Run("valid", func(t *testing.T) {
		tests := []struct {
			name string
			json string
			want ToolCall
		}{
			{
				name: "Complete tool call",
				json: `{"id": "call_123", "name": "calculator", "arguments": "{\"a\": 5, \"b\": 3}"}`,
				want: ToolCall{ID: "call_123", Name: "calculator", Arguments: "{\"a\": 5, \"b\": 3}"},
			},
			{
				name: "Tool call with only name",
				json: `{"name": "weather", "arguments": "{}"}`,
				want: ToolCall{Name: "weather", Arguments: "{}"},
			},
			{
				name: "Tool call with only ID",
				json: `{"id": "call_456", "arguments": "{}"}`,
				want: ToolCall{ID: "call_456", Arguments: "{}"},
			},
		}
		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				var got ToolCall
				if err := got.UnmarshalJSON([]byte(tt.json)); err != nil {
					t.Fatalf("unexpected error: %v", err)
				}
				if diff := cmp.Diff(tt.want, got); diff != "" {
					t.Errorf("ToolCall mismatch (-want +got):\n%s", diff)
				}
			})
		}
	})

	t.Run("error", func(t *testing.T) {
		tests := []struct {
			name   string
			json   string
			errMsg string
		}{
			{
				name:   "Invalid JSON",
				json:   `{"name": "tool", "arguments": invalid}`,
				errMsg: "invalid character 'i' looking for beginning of value",
			},
			{
				name:   "Invalid arguments JSON",
				json:   `{"name": "tool", "arguments": "invalid json"}`,
				errMsg: "field Arguments: invalid character 'i' looking for beginning of value",
			},
			{
				name:   "Missing both ID and name",
				json:   `{"arguments": "{}"}`,
				errMsg: "at least one of field ID or Name is required",
			},
		}
		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				var got ToolCall
				err := got.UnmarshalJSON([]byte(tt.json))
				if err == nil {
					t.Fatal("expected error, got nil")
				}
				if !strings.Contains(err.Error(), tt.errMsg) {
					t.Errorf("expected error to contain %q, got %q", tt.errMsg, err.Error())
				}
			})
		}
	})
}

func TestToolCallResult_UnmarshalJSON(t *testing.T) {
	t.Run("valid", func(t *testing.T) {
		tests := []struct {
			name string
			json string
			want ToolCallResult
		}{
			{
				name: "Complete tool call result",
				json: `{"id": "call_123", "name": "calculator", "result": "8"}`,
				want: ToolCallResult{ID: "call_123", Name: "calculator", Result: "8"},
			},
			{
				name: "Tool call result with only name",
				json: `{"name": "weather", "result": "sunny"}`,
				want: ToolCallResult{Name: "weather", Result: "sunny"},
			},
			{
				name: "Tool call result with only ID",
				json: `{"id": "call_456", "result": "success"}`,
				want: ToolCallResult{ID: "call_456", Result: "success"},
			},
		}
		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				var got ToolCallResult
				if err := got.UnmarshalJSON([]byte(tt.json)); err != nil {
					t.Fatalf("unexpected error: %v", err)
				}
				if diff := cmp.Diff(tt.want, got); diff != "" {
					t.Errorf("ToolCallResult mismatch (-want +got):\n%s", diff)
				}
			})
		}
	})

	t.Run("error", func(t *testing.T) {
		tests := []struct {
			name   string
			json   string
			errMsg string
		}{
			{
				name:   "Invalid JSON",
				json:   `{"name": "tool", "result": invalid}`,
				errMsg: "invalid character 'i' looking for beginning of value",
			},
			{
				name:   "Unknown field",
				json:   `{"name": "tool", "result": "success", "unknown_field": "value"}`,
				errMsg: "json: unknown field \"unknown_field\"",
			},
			{
				name:   "Missing both ID and name",
				json:   `{"result": "success"}`,
				errMsg: "at least one of field ID or Name is required",
			},
			{
				name:   "Missing result",
				json:   `{"name": "tool"}`,
				errMsg: "field Result: required",
			},
		}
		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				var got ToolCallResult
				err := got.UnmarshalJSON([]byte(tt.json))
				if err == nil {
					t.Fatal("expected error, got nil")
				}
				if !strings.Contains(err.Error(), tt.errMsg) {
					t.Errorf("expected error to contain %q, got %q", tt.errMsg, err.Error())
				}
			})
		}
	})
}

func TestCitation_Validate(t *testing.T) {
	tests := []struct {
		name     string
		citation Citation
		wantErr  bool
	}{
		{
			name: "valid citation",
			citation: Citation{
				Text:       "example text",
				StartIndex: 0,
				EndIndex:   12,
				Sources: []CitationSource{{
					ID:   "doc1",
					Type: "document",
				}},
			},
		},
		{
			name: "empty text",
			citation: Citation{
				Text:       "",
				StartIndex: 0,
				EndIndex:   10,
			},
		},
		{
			name: "negative start index",
			citation: Citation{
				Text:       "example",
				StartIndex: -1,
				EndIndex:   10,
			},
			wantErr: true,
		},
		{
			name: "end index before start",
			citation: Citation{
				Text:       "example",
				StartIndex: 10,
				EndIndex:   5,
			},
			wantErr: true,
		},
		{
			name: "end index equal to start",
			citation: Citation{
				Text:       "example",
				StartIndex: 10,
				EndIndex:   10,
			},
			wantErr: true,
		},
		{
			name: "zero end index is valid",
			citation: Citation{
				Text:       "example",
				StartIndex: 0,
				EndIndex:   0, // Zero is allowed as it may indicate position-only citation
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.citation.Validate()
			if (err != nil) != tt.wantErr {
				t.Errorf("Citation.Validate() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestCitationSource_Validate(t *testing.T) {
	tests := []struct {
		name    string
		source  CitationSource
		wantErr bool
	}{
		{
			name: "valid with ID",
			source: CitationSource{
				ID:   "doc1",
				Type: "document",
			},
			wantErr: false,
		},
		{
			name: "valid with URL",
			source: CitationSource{
				URL:  "https://example.com",
				Type: "web",
			},
			wantErr: false,
		},
		{
			name: "valid with both ID and URL",
			source: CitationSource{
				ID:   "doc1",
				URL:  "https://example.com",
				Type: "document",
			},
			wantErr: false,
		},
		{
			name: "invalid without ID or URL",
			source: CitationSource{
				Type: "document",
			},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.source.Validate()
			if (err != nil) != tt.wantErr {
				t.Errorf("CitationSource.Validate() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestTriState_String(t *testing.T) {
	tests := []struct {
		name string
		ts   TriState
		want string
	}{
		{
			name: "False",
			ts:   False,
			want: "false",
		},
		{
			name: "True",
			ts:   True,
			want: "true",
		},
		{
			name: "Flaky",
			ts:   Flaky,
			want: "flaky",
		},
		{
			name: "Unknown value",
			ts:   TriState(99),
			want: "TriState(99)",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.ts.String(); got != tt.want {
				t.Errorf("TriState.String() got = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestContent_ValidateWithCitations(t *testing.T) {
	tests := []struct {
		name    string
		content Content
		wantErr bool
	}{
		{
			name: "valid text with citations",
			content: Content{
				Text: "The capital is Paris.",
				Citations: []Citation{{
					Text:       "Paris",
					StartIndex: 15,
					EndIndex:   20,
					Sources: []CitationSource{{
						ID:   "doc1",
						Type: "document",
					}},
				}},
			},
		},
		{
			name: "citations without text",
			content: Content{
				Citations: []Citation{{
					Text: "example",
				}},
			},
			wantErr: true,
		},
		{
			name: "citations with thinking",
			content: Content{
				Thinking: "reasoning",
				Citations: []Citation{{
					Text: "example",
				}},
			},
			wantErr: true,
		},
		{
			name: "invalid citation",
			content: Content{
				Text: "The capital is Paris.",
				Citations: []Citation{{
					Text: "", // Empty text
				}},
			},
			// Technically it need a source. wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.content.Validate()
			if (err != nil) != tt.wantErr {
				t.Errorf("Content.Validate() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}
