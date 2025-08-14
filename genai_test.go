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
	"github.com/maruel/genai/internal/bb"
)

func TestUsage(t *testing.T) {
	t.Run("String", func(t *testing.T) {
		u := Usage{
			InputTokens:       10,
			InputCachedTokens: 5,
			ReasoningTokens:   15,
			OutputTokens:      20,
			TotalTokens:       10 + 5 + 15 + 20,
			Limits: []RateLimit{
				{
					Type:      Requests,
					Period:    PerMinute,
					Limit:     100,
					Remaining: 99,
				},
				{
					Type:      Tokens,
					Period:    PerDay,
					Limit:     10000,
					Remaining: 9980,
				},
			},
		}
		want := "in: 10 (cached 5), reasoning: 15, out: 20, total: 50, requests (minute): 99/100, tokens (day): 9980/10000"
		if got := u.String(); got != want {
			t.Errorf("Usage.String()\nwant %q\ngot  %q", want, got)
		}
	})
}

func TestRole(t *testing.T) {
	t.Run("Validate", func(t *testing.T) {
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
	})
}

func TestMessages(t *testing.T) {
	t.Run("Validate", func(t *testing.T) {
		t.Run("valid", func(t *testing.T) {
			m := Messages{
				NewTextMessage("Hello"),
				Message{Role: Assistant, Reply: []Content{{Text: "I can help with that"}}},
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
						{Role: User, Request: []Content{{Text: "Hi", Doc: Doc{Filename: "hi.txt"}}}},
						{Role: User, Request: []Content{{}}},
					},
					errMsg: "message 0: request 0: field Doc can't be used along Text\nmessage 1: request 0: an empty Content is invalid",
				},
			}
			for _, tt := range tests {
				t.Run(tt.name, func(t *testing.T) {
					if err := tt.messages.Validate(); err == nil || err.Error() != tt.errMsg {
						t.Fatalf("error mismatch\nwant %q\ngot  %q", tt.errMsg, err)
					}
				})
			}
		})
	})
}

func TestMessage(t *testing.T) {
	t.Run("Validate", func(t *testing.T) {
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
						Request: []Content{
							{Doc: Doc{Filename: "document.txt", Src: strings.NewReader("document content")}},
						},
					},
				},
				{
					name:    "Valid assistant message",
					message: Message{Role: Assistant, Reply: []Content{{Text: "I can help with that"}}},
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
					errMsg:  "field Role: role \"\" is not supported\nat least one of fields Request, Reply, ToolCalls or ToolCallsResults is required",
				},
				{
					name:    "user",
					message: Message{Role: User, User: "Joe", Request: []Content{{Text: "Hi"}}},
					errMsg:  "field User: not supported yet",
				},
			}
			for _, tt := range tests {
				t.Run(tt.name, func(t *testing.T) {
					if err := tt.message.Validate(); err == nil || err.Error() != tt.errMsg {
						t.Fatalf("error mismatch\nwant %q\ngot  %q", tt.errMsg, err)
					}
				})
			}
		})
	})
	t.Run("Decode", func(t *testing.T) {
		t.Run("valid", func(t *testing.T) {
			m := Message{Role: Assistant, Reply: []Content{{Text: "{\"key\": \"value\"}"}}}
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
					message: Message{Role: Assistant, Reply: []Content{{Text: "invalid"}}},
					errMsg:  "failed to decode message text as JSON: invalid character 'i' looking for beginning of value; reply: \"invalid\"",
				},
				{
					name: "Invalid DecodeAs",
					message: Message{
						Role:  Assistant,
						Reply: []Content{{Doc: Doc{Src: strings.NewReader("document content")}}},
					},
					errMsg: "only text messages can be decoded as JSON, can't decode {\"role\":\"assistant\",\"reply\":[{\"doc\":{\"bytes\":\"ZG9jdW1lbnQgY29udGVudA==\"}}]}",
				},
			}
			for _, tt := range tests {
				t.Run(tt.name, func(t *testing.T) {
					if err := tt.message.Decode("invalid"); err == nil || err.Error() != tt.errMsg {
						t.Fatalf("error mismatch\nwant %q\ngot  %q", tt.errMsg, err)
					}
				})
			}
		})
	})
	t.Run("UnmarshalJSON", func(t *testing.T) {
		t.Run("valid", func(t *testing.T) {
			tests := []struct {
				name string
				json string
				want Message
			}{
				{
					name: "User text message",
					json: `{"role": "user", "request": [{"text": "Hello"}]}`,
					want: Message{
						Role:    User,
						Request: []Content{{Text: "Hello"}},
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
					json:   `{"role": "user", "request": invalid}`,
					errMsg: "invalid character 'i' looking for beginning of value",
				},
				{
					name:   "Unknown field",
					json:   `{"role": "user", "request": [{"text": "Hi"}], "unknown_field": "value"}`,
					errMsg: "json: unknown field \"unknown_field\"",
				},
				{
					name:   "Invalid role",
					json:   `{"role": "invalid", "request": [{"text": "Hi"}]}`,
					errMsg: "field Role: role \"invalid\" is not supported",
				},
				// Note: Empty message (just role) passes UnmarshalJSON validation,
				// but would fail full Validate() method. UnmarshalJSON only does partial validation.
				{
					name:   "User with User field",
					json:   `{"role": "user", "user": "joe", "request": [{"text": "Hi"}]}`,
					errMsg: "field User: not supported yet",
				},
			}
			for _, tt := range tests {
				t.Run(tt.name, func(t *testing.T) {
					var got Message
					if err := got.UnmarshalJSON([]byte(tt.json)); err == nil || err.Error() != tt.errMsg {
						t.Fatalf("error mismatch\nwant %q\ngot  %q", tt.errMsg, err)
					}
				})
			}
		})
	})
	t.Run("Accumulate", func(t *testing.T) {
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
				want:     Message{Role: Assistant, Reply: []Content{{Text: "Hello"}}},
			},
			{
				name:    "Document",
				message: Message{Role: Assistant},
				fragment: ContentFragment{
					Filename:         "document.txt",
					DocumentFragment: []byte("document content"),
				},
				want: Message{
					Role:  Assistant,
					Reply: []Content{{Doc: Doc{Filename: "document.txt", Src: &bb.BytesBuffer{D: []byte("document content")}}}},
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
				message:  Message{Role: Assistant, Reply: []Content{{Text: "Hello"}}},
				fragment: ContentFragment{TextFragment: " world"},
				want:     Message{Role: Assistant, Reply: []Content{{Text: "Hello world"}}},
			},
			{
				name:     "Add thinking to existing thinking",
				message:  Message{Role: Assistant, Reply: []Content{{Thinking: "I think "}}},
				fragment: ContentFragment{ThinkingFragment: "therefore I am"},
				want:     Message{Role: Assistant, Reply: []Content{{Thinking: "I think therefore I am"}}},
			},
			{
				name:     "Join assistant text",
				message:  Message{Role: Assistant, Reply: []Content{{Text: "Hello"}}},
				fragment: ContentFragment{TextFragment: " world"},
				want:     Message{Role: Assistant, Reply: []Content{{Text: "Hello world"}}},
			},
			{
				name: "Document then text",
				message: Message{
					Role:  Assistant,
					Reply: []Content{{Doc: Doc{Filename: "document.txt", Src: &bb.BytesBuffer{D: []byte("document content")}}}},
				},
				fragment: ContentFragment{TextFragment: "No"},
				want: Message{
					Role: Assistant,
					Reply: []Content{
						{Doc: Doc{Filename: "document.txt", Src: &bb.BytesBuffer{D: []byte("document content")}}},
						{Text: "No"},
					},
				},
			},
			{
				name:     "Tool then text",
				message:  Message{Role: Assistant, ToolCalls: []ToolCall{{Name: "tool"}}},
				fragment: ContentFragment{TextFragment: "No"},
				want: Message{
					Role: Assistant,
					// Merge together.
					Reply:     []Content{{Text: "No"}},
					ToolCalls: []ToolCall{{Name: "tool"}},
				},
			},
			{
				name:     "Tool then tool",
				message:  Message{Role: Assistant, ToolCalls: []ToolCall{{Name: "tool"}}},
				fragment: ContentFragment{ToolCall: ToolCall{Name: "tool2"}},
				want: Message{
					Role: Assistant,
					// Merge together.
					ToolCalls: []ToolCall{{Name: "tool"}, {Name: "tool2"}},
				},
			},
		}
		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				message := tt.message
				if err := message.Accumulate(tt.fragment); err != nil {
					t.Fatalf("unexpected error: %v", err)
				}
				if diff := cmp.Diff(tt.want, message); diff != "" {
					t.Fatalf("unexpected result: %s", diff)
				}
			})
		}
	})
}

func TestContent(t *testing.T) {
	t.Run("Validate", func(t *testing.T) {
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
					in:   Content{Doc: Doc{Filename: "document.txt", Src: strings.NewReader("document content")}},
				},
				{
					name: "valid text with citations",
					in: Content{
						Text: "The capital is Paris.",
						Citations: []Citation{{
							Text:       "Paris",
							StartIndex: 15,
							EndIndex:   20,
							Sources:    []CitationSource{{ID: "doc1", Type: "document"}},
						}},
					},
				},
				// Can happen with tool calling, e.g. cohere
				{
					name: "citations without text",
					in:   Content{Citations: []Citation{{Text: "example"}}},
				},
				{
					// Technically it need a source. wantErr: true,
					name: "invalid citation",
					in: Content{
						Text:      "The capital is Paris.",
						Citations: []Citation{{Text: ""}}, // Empty text
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
		t.Run("error", func(t *testing.T) {
			tests := []struct {
				name    string
				content Content
				errMsg  string
			}{
				{
					name: "citations with thinking",
					content: Content{
						Thinking:  "reasoning",
						Citations: []Citation{{Text: "example"}},
					},
					errMsg: "field Citations can only be used with Text",
				},
			}
			for _, tt := range tests {
				t.Run(tt.name, func(t *testing.T) {
					if err := tt.content.Validate(); err == nil || err.Error() != tt.errMsg {
						t.Fatalf("error mismatch\nwant %q\ngot  %q", tt.errMsg, err)
					}
				})
			}
		})
	})
	t.Run("Read", func(t *testing.T) {
		t.Run("valid", func(t *testing.T) {
			c := Content{
				Doc: Doc{Filename: "document.txt", Src: strings.NewReader("document content")},
			}
			mime, got, err := c.Doc.Read(1000)
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
	})
	t.Run("UnmarshalJSON", func(t *testing.T) {
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
					json: `{"doc":{"filename": "image.jpg", "url": "https://example.com/image.jpg"}}`,
					want: Content{Doc: Doc{Filename: "image.jpg", URL: "https://example.com/image.jpg"}},
				},
				{
					name: "Document content",
					json: `{"doc":{"filename": "doc.txt", "bytes": "SGVsbG8gV29ybGQ="}}`,
					want: Content{Doc: Doc{Filename: "doc.txt", Src: strings.NewReader("Hello World")}},
				},
			}
			for _, tt := range tests {
				t.Run(tt.name, func(t *testing.T) {
					var got Content
					if err := got.UnmarshalJSON([]byte(tt.json)); err != nil {
						t.Fatalf("unexpected error: %v", err)
					}
					// For Document comparison, read the content since we can't directly compare io.ReadSeeker
					if tt.want.Doc.Src != nil {
						wantData, _ := io.ReadAll(tt.want.Doc.Src)
						gotData, _ := io.ReadAll(got.Doc.Src)
						if string(wantData) != string(gotData) {
							t.Errorf("Document content mismatch: want %q, got %q", string(wantData), string(gotData))
						}
						// Reset Document field for comparison
						tt.want.Doc.Src = nil
						got.Doc.Src = nil
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
					json:   `{"doc":{"bytes": "SGVsbG8="}}`,
					errMsg: "field Filename is required with Document when not implementing Name()",
				},
				{
					name:   "Empty content",
					json:   `{}`,
					errMsg: "an empty Content is invalid",
				},
			}
			for _, tt := range tests {
				t.Run(tt.name, func(t *testing.T) {
					var got Content
					if err := got.UnmarshalJSON([]byte(tt.json)); err == nil || err.Error() != tt.errMsg {
						t.Fatalf("error mismatch\nwant %q\ngot  %q", tt.errMsg, err)
					}
				})
			}
		})
	})
}

func TestToolCall(t *testing.T) {
	t.Run("Validate", func(t *testing.T) {
		// TODO.
	})
	t.Run("Call", func(t *testing.T) {
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
	})
	t.Run("UnmarshalJSON", func(t *testing.T) {
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
					if err := got.UnmarshalJSON([]byte(tt.json)); err == nil || err.Error() != tt.errMsg {
						t.Fatalf("error mismatch\nwant %q\ngot  %q", tt.errMsg, err)
					}
				})
			}
		})
	})
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

func TestToolCallResult(t *testing.T) {
	t.Run("UnmarshalJSON", func(t *testing.T) {
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
					if err := got.UnmarshalJSON([]byte(tt.json)); err == nil || err.Error() != tt.errMsg {
						t.Fatalf("error mismatch\nwant %q\ngot  %q", tt.errMsg, err)
					}
				})
			}
		})
	})
}

func TestCitation(t *testing.T) {
	t.Run("Validate", func(t *testing.T) {
		t.Run("valid", func(t *testing.T) {
			tests := []struct {
				name     string
				citation Citation
				errMsg   string
			}{
				{
					name: "valid citation",
					citation: Citation{
						Text:       "example text",
						StartIndex: 0,
						EndIndex:   12,
						Sources:    []CitationSource{{ID: "doc1", Type: "document"}},
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
					if err := tt.citation.Validate(); err != nil {
						t.Fatalf("unexpected error: %q", err)
					}
				})
			}
		})
		t.Run("error", func(t *testing.T) {
			tests := []struct {
				name     string
				citation Citation
				errMsg   string
			}{
				{
					name: "negative start index",
					citation: Citation{
						Text:       "example",
						StartIndex: -1,
						EndIndex:   10,
					},
					errMsg: "start index must be non-negative, got -1",
				},
				{
					name: "end index before start",
					citation: Citation{
						Text:       "example",
						StartIndex: 10,
						EndIndex:   5,
					},
					errMsg: "end index (5) must be greater than start index (10)",
				},
				{
					name: "end index equal to start",
					citation: Citation{
						Text:       "example",
						StartIndex: 10,
						EndIndex:   10,
					},
					errMsg: "end index (10) must be greater than start index (10)",
				},
			}
			for _, tt := range tests {
				t.Run(tt.name, func(t *testing.T) {
					if err := tt.citation.Validate(); err == nil || err.Error() != tt.errMsg {
						t.Fatalf("error mismatch\nwant %q\ngot  %q", tt.errMsg, err)
					}
				})
			}
		})
	})
}

func TestCitationSource(t *testing.T) {
	t.Run("Validate", func(t *testing.T) {
		t.Run("valid", func(t *testing.T) {
			tests := []struct {
				name   string
				source CitationSource
			}{
				{
					name:   "valid with ID",
					source: CitationSource{ID: "doc1", Type: "document"},
				},
				{
					name:   "valid with URL",
					source: CitationSource{URL: "https://example.com", Type: "web"},
				},
				{
					name:   "valid with both ID and URL",
					source: CitationSource{ID: "doc1", URL: "https://example.com", Type: "document"},
				},
			}
			for _, tt := range tests {
				t.Run(tt.name, func(t *testing.T) {
					if err := tt.source.Validate(); err != nil {
						t.Fatalf("unexpected error: %q", err)
					}
				})
			}
		})
		t.Run("error", func(t *testing.T) {
			tests := []struct {
				name   string
				source CitationSource
				errMsg string
			}{
				{
					name:   "invalid without ID or URL",
					source: CitationSource{Type: "document"},
					errMsg: "citation source must have either ID or URL",
				},
			}
			for _, tt := range tests {
				t.Run(tt.name, func(t *testing.T) {
					if err := tt.source.Validate(); err == nil || err.Error() != tt.errMsg {
						t.Fatalf("error mismatch\nwant %q\ngot  %q", tt.errMsg, err)
					}
				})
			}
		})
	})
}

func TestTriState(t *testing.T) {
	t.Run("String", func(t *testing.T) {
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
	})
}
