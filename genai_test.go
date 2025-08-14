// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package genai

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"strings"
	"testing"
	"time"

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
			t.Fatalf("Usage.String()\nwant %q\ngot  %q", want, got)
		}
	})
	t.Run("Add", func(t *testing.T) {
		u1 := Usage{
			InputTokens:       10,
			InputCachedTokens: 5,
			ReasoningTokens:   15,
			OutputTokens:      20,
			TotalTokens:       50,
		}
		u2 := Usage{
			InputTokens:       20,
			InputCachedTokens: 10,
			ReasoningTokens:   30,
			OutputTokens:      40,
			TotalTokens:       100,
		}
		expected := Usage{
			InputTokens:       30,
			InputCachedTokens: 15,
			ReasoningTokens:   45,
			OutputTokens:      60,
			TotalTokens:       150,
		}
		u1.Add(u2)
		if diff := cmp.Diff(expected, u1); diff != "" {
			t.Fatalf("Usage.Add() mismatch (-want +got):\n%s", diff)
		}
	})
}

func TestMessages(t *testing.T) {
	t.Run("Validate", func(t *testing.T) {
		t.Run("valid", func(t *testing.T) {
			m := Messages{
				NewTextMessage("Hello"),
				Message{Reply: []Reply{{Text: "I can help with that"}}},
			}
			if err := m.Validate(); err != nil {
				t.Fatalf("unexpected error: %q", err)
			}
		})
		t.Run("error", func(t *testing.T) {
			tests := []struct {
				name   string
				in     Messages
				errMsg string
			}{
				{
					name: "Invalid messages",
					in: Messages{
						{Request: []Request{{Text: "Hi", Doc: Doc{Filename: "hi.txt"}}}},
						{Request: []Request{{}}},
					},
					errMsg: "message 0: request 0: field Doc can't be used along Text\nmessage 1: request 0: an empty Request is invalid\nmessage 1: role must alternate; got twice \"user\"",
				},
			}
			for _, tt := range tests {
				t.Run(tt.name, func(t *testing.T) {
					if err := tt.in.Validate(); err == nil || err.Error() != tt.errMsg {
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
				name string
				in   Message
			}{
				{
					name: "Valid user text message",
					in:   NewTextMessage("Hello"),
				},
				{
					name: "Valid user document message",
					in: Message{
						Request: []Request{
							{Doc: Doc{Filename: "document.txt", Src: strings.NewReader("document content")}},
						},
					},
				},
				{
					name: "Valid assistant message",
					in:   Message{Reply: []Reply{{Text: "I can help with that"}}},
				},
				{
					name: "Valid assistant with tool calls",
					in: Message{
						ToolCalls: []ToolCall{{Name: "tool", Arguments: "{}"}},
					},
				},
				{
					name: "Valid user with tool call results",
					in: Message{
						ToolCallResults: []ToolCallResult{{Name: "tool", Result: "result"}},
					},
				},
				{
					name: "Valid assistant with reply and tool calls",
					in: Message{
						Reply:     []Reply{{Text: "I'll call a tool"}},
						ToolCalls: []ToolCall{{Name: "tool", Arguments: "{}"}},
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
				name   string
				in     Message
				errMsg string
			}{
				{
					name:   "empty",
					in:     Message{},
					errMsg: "at least one of fields Request, Reply, ToolCalls or ToolCallsResults is required",
				},
				{
					name:   "User field",
					in:     Message{User: "Joe", Request: []Request{{Text: "Hi"}}},
					errMsg: "field User: not supported yet",
				},
				{
					name: "both request and tool call results",
					in: Message{
						Request:         []Request{{Text: "request"}},
						ToolCallResults: []ToolCallResult{{Name: "tool", Result: "result"}},
					},
					errMsg: "exactly one of Request, Reply/ToolCalls or ToolCallResults must be set",
				},
				{
					name: "reply containing doc and tool calls",
					in: Message{
						Reply:     []Reply{{Doc: Doc{Filename: "file.txt", Src: strings.NewReader("content")}}},
						ToolCalls: []ToolCall{{Name: "tool", Arguments: "{}"}},
					},
					errMsg: "field Reply can't contain a Doc along with ToolCalls",
				},
			}
			for _, tt := range tests {
				t.Run(tt.name, func(t *testing.T) {
					if err := tt.in.Validate(); err == nil || err.Error() != tt.errMsg {
						t.Fatalf("error mismatch\nwant %q\ngot  %q", tt.errMsg, err)
					}
				})
			}
		})
	})
	t.Run("Decode", func(t *testing.T) {
		t.Run("valid", func(t *testing.T) {
			m := Message{Reply: []Reply{{Text: "{\"key\": \"value\"}"}}}
			if err := m.Decode(&struct{ Key string }{}); err != nil {
				t.Fatalf("unexpected error: %q", err)
			}
		})
		t.Run("error", func(t *testing.T) {
			tests := []struct {
				name   string
				in     Message
				errMsg string
			}{
				{
					name:   "Invalid JSON message",
					in:     Message{Reply: []Reply{{Text: "invalid"}}},
					errMsg: "failed to decode message text as JSON: invalid character 'i' looking for beginning of value; reply: \"invalid\"",
				},
				{
					name: "Invalid DecodeAs",
					in: Message{
						Reply: []Reply{{Doc: Doc{Src: strings.NewReader("document content")}}},
					},
					errMsg: "only text messages can be decoded as JSON, can't decode {\"reply\":[{\"doc\":{\"bytes\":\"ZG9jdW1lbnQgY29udGVudA==\"}}]}",
				},
			}
			for _, tt := range tests {
				t.Run(tt.name, func(t *testing.T) {
					if err := tt.in.Decode("invalid"); err == nil || err.Error() != tt.errMsg {
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
				in   string
				want Message
			}{
				{
					name: "User text message",
					in:   `{"request": [{"text": "Hello"}]}`,
					want: Message{
						Request: []Request{{Text: "Hello"}},
					},
				},
				{
					name: "Assistant message with tool call",
					in:   `{"tool_calls": [{"id": "1", "name": "tool", "arguments": "{}"}]}`,
					want: Message{
						ToolCalls: []ToolCall{{ID: "1", Name: "tool", Arguments: "{}"}},
					},
				},
				{
					name: "Computer message with tool result",
					in:   `{"tool_call_results": [{"id": "1", "name": "tool", "result": "success"}]}`,
					want: Message{
						ToolCallResults: []ToolCallResult{{ID: "1", Name: "tool", Result: "success"}},
					},
				},
			}
			for _, tt := range tests {
				t.Run(tt.name, func(t *testing.T) {
					var got Message
					d := json.NewDecoder(strings.NewReader(tt.in))
					d.DisallowUnknownFields()
					if err := d.Decode(&got); err != nil {
						t.Fatalf("unexpected error: %v", err)
					}
					if diff := cmp.Diff(tt.want, got); diff != "" {
						t.Fatalf("Message mismatch (-want +got):\n%s", diff)
					}
				})
			}
		})

		t.Run("error", func(t *testing.T) {
			tests := []struct {
				name   string
				in     string
				errMsg string
			}{
				{
					name:   "Invalid JSON",
					in:     `{"request": invalid}`,
					errMsg: "invalid character 'i' looking for beginning of value",
				},
				{
					name:   "Unknown field",
					in:     `{"request": [{"text": "Hi"}], "unknown_field": "value"}`,
					errMsg: "json: unknown field \"unknown_field\"",
				},
				{
					name:   "User with User field",
					in:     `{"user": "joe", "request": [{"text": "Hi"}]}`,
					errMsg: "field User: not supported yet",
				},
			}
			for _, tt := range tests {
				t.Run(tt.name, func(t *testing.T) {
					var got Message
					d := json.NewDecoder(strings.NewReader(tt.in))
					d.DisallowUnknownFields()
					if err := d.Decode(&got); err == nil || err.Error() != tt.errMsg {
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
				fragment: ContentFragment{TextFragment: "Hello"},
				want:     Message{Reply: []Reply{{Text: "Hello"}}},
			},
			{
				name: "Document",
				fragment: ContentFragment{
					Filename:         "document.txt",
					DocumentFragment: []byte("document content"),
				},
				want: Message{
					Reply: []Reply{{Doc: Doc{Filename: "document.txt", Src: &bb.BytesBuffer{D: []byte("document content")}}}},
				},
			},
			{
				name:     "Tool",
				fragment: ContentFragment{ToolCall: ToolCall{Name: "tool"}},
				want: Message{
					ToolCalls: []ToolCall{{Name: "tool"}},
				},
			},
			{
				name:     "Add text to existing text",
				message:  Message{Reply: []Reply{{Text: "Hello"}}},
				fragment: ContentFragment{TextFragment: " world"},
				want:     Message{Reply: []Reply{{Text: "Hello world"}}},
			},
			{
				name:     "Add thinking to existing thinking",
				message:  Message{Reply: []Reply{{Thinking: "I think "}}},
				fragment: ContentFragment{ThinkingFragment: "therefore I am"},
				want:     Message{Reply: []Reply{{Thinking: "I think therefore I am"}}},
			},
			{
				name:     "Join assistant text",
				message:  Message{Reply: []Reply{{Text: "Hello"}}},
				fragment: ContentFragment{TextFragment: " world"},
				want:     Message{Reply: []Reply{{Text: "Hello world"}}},
			},
			{
				name: "Document then text",
				message: Message{
					Reply: []Reply{{Doc: Doc{Filename: "document.txt", Src: &bb.BytesBuffer{D: []byte("document content")}}}},
				},
				fragment: ContentFragment{TextFragment: "No"},
				want: Message{
					Reply: []Reply{
						{Doc: Doc{Filename: "document.txt", Src: &bb.BytesBuffer{D: []byte("document content")}}},
						{Text: "No"},
					},
				},
			},
			{
				name:     "Tool then text",
				message:  Message{ToolCalls: []ToolCall{{Name: "tool"}}},
				fragment: ContentFragment{TextFragment: "No"},
				want: Message{
					// Merge together.
					Reply:     []Reply{{Text: "No"}},
					ToolCalls: []ToolCall{{Name: "tool"}},
				},
			},
			{
				name:     "Tool then tool",
				message:  Message{ToolCalls: []ToolCall{{Name: "tool"}}},
				fragment: ContentFragment{ToolCall: ToolCall{Name: "tool2"}},
				want: Message{
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
	t.Run("DoToolCalls", func(t *testing.T) {
		type calculateInput struct {
			A int `json:"a"`
			B int `json:"b"`
		}

		t.Run("valid", func(t *testing.T) {
			ctx := t.Context()
			tool := ToolDef{
				Name:        "calculator",
				Description: "A calculator tool",
				Callback: func(ctx context.Context, input *calculateInput) (string, error) {
					return fmt.Sprintf("%d", input.A+input.B), nil
				},
			}

			msg := Message{
				ToolCalls: []ToolCall{
					{
						ID:        "call1",
						Name:      "calculator",
						Arguments: `{"a": 5, "b": 3}`,
					},
				},
			}

			result, err := msg.DoToolCalls(ctx, []ToolDef{tool})
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}

			expected := Message{
				ToolCallResults: []ToolCallResult{
					{
						ID:     "call1",
						Name:   "calculator",
						Result: "8",
					},
				},
			}

			if diff := cmp.Diff(expected, result); diff != "" {
				t.Fatalf("DoToolCalls() mismatch (-want +got):\n%s", diff)
			}
		})

		t.Run("tool not found", func(t *testing.T) {
			ctx := t.Context()
			tool := ToolDef{
				Name:        "calculator",
				Description: "A calculator tool",
				Callback: func(ctx context.Context, input *calculateInput) (string, error) {
					return fmt.Sprintf("%d", input.A+input.B), nil
				},
			}

			msg := Message{
				ToolCalls: []ToolCall{
					{
						ID:        "call1",
						Name:      "nonexistent",
						Arguments: `{"a": 5, "b": 3}`,
					},
				},
			}

			_, err := msg.DoToolCalls(ctx, []ToolDef{tool})
			if err == nil {
				t.Fatal("expected error, got nil")
			}

			if !strings.Contains(err.Error(), "failed to find tool named \"nonexistent\"") {
				t.Fatalf("unexpected error message: %v", err)
			}
		})

		t.Run("no tool calls", func(t *testing.T) {
			ctx := t.Context()
			msg := Message{
				Reply: []Reply{{Text: "Hello"}},
			}

			result, err := msg.DoToolCalls(ctx, []ToolDef{})
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}

			// Should return a zero message
			if !result.IsZero() {
				t.Fatalf("expected zero message, got: %+v", result)
			}
		})
	})
}

func TestRequest(t *testing.T) {
	t.Run("Validate", func(t *testing.T) {
		t.Run("valid", func(t *testing.T) {
			tests := []struct {
				name string
				in   Request
			}{
				{
					name: "Valid text block",
					in:   Request{Text: "Hello"},
				},
				{
					name: "Valid document block",
					in:   Request{Doc: Doc{Filename: "document.txt", Src: strings.NewReader("document content")}},
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
				name   string
				in     Request
				errMsg string
			}{
				{
					name:   "empty",
					in:     Request{},
					errMsg: "an empty Request is invalid",
				},
			}
			for _, tt := range tests {
				t.Run(tt.name, func(t *testing.T) {
					if err := tt.in.Validate(); err == nil || err.Error() != tt.errMsg {
						t.Fatalf("error mismatch\nwant %q\ngot  %q", tt.errMsg, err)
					}
				})
			}
		})
	})
	t.Run("Read", func(t *testing.T) {
		t.Run("valid", func(t *testing.T) {
			c := Request{
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
				in   string
				want Request
			}{
				{
					name: "Text content",
					in:   `{"text": "Hello world"}`,
					want: Request{Text: "Hello world"},
				},
				{
					name: "URL content",
					in:   `{"doc":{"filename": "image.jpg", "url": "https://example.com/image.jpg"}}`,
					want: Request{Doc: Doc{Filename: "image.jpg", URL: "https://example.com/image.jpg"}},
				},
				{
					name: "Document content",
					in:   `{"doc":{"filename": "doc.txt", "bytes": "SGVsbG8gV29ybGQ="}}`,
					want: Request{Doc: Doc{Filename: "doc.txt", Src: strings.NewReader("Hello World")}},
				},
			}
			for _, tt := range tests {
				t.Run(tt.name, func(t *testing.T) {
					var got Request
					d := json.NewDecoder(strings.NewReader(tt.in))
					d.DisallowUnknownFields()
					if err := d.Decode(&got); err != nil {
						t.Fatalf("unexpected error: %v", err)
					}
					// For Document comparison, read the content since we can't directly compare io.ReadSeeker
					if tt.want.Doc.Src != nil {
						wantData, _ := io.ReadAll(tt.want.Doc.Src)
						gotData, _ := io.ReadAll(got.Doc.Src)
						if string(wantData) != string(gotData) {
							t.Fatalf("Document content mismatch: want %q, got %q", string(wantData), string(gotData))
						}
						// Reset Document field for comparison
						tt.want.Doc.Src = nil
						got.Doc.Src = nil
					}
					if diff := cmp.Diff(tt.want, got); diff != "" {
						t.Fatalf("Request mismatch (-want +got):\n%s", diff)
					}
				})
			}
		})

		t.Run("error", func(t *testing.T) {
			tests := []struct {
				name   string
				in     string
				errMsg string
			}{
				{
					name:   "Invalid JSON",
					in:     `{"text": invalid}`,
					errMsg: "invalid character 'i' looking for beginning of value",
				},
				{
					name:   "Unknown field",
					in:     `{"text": "Hi", "unknown_field": "value"}`,
					errMsg: "json: unknown field \"unknown_field\"",
				},
				{
					name:   "Document without filename",
					in:     `{"doc":{"bytes": "SGVsbG8="}}`,
					errMsg: "field Filename is required with Document when not implementing Name()",
				},
				{
					name:   "Empty content",
					in:     `{}`,
					errMsg: "an empty Request is invalid",
				},
			}
			for _, tt := range tests {
				t.Run(tt.name, func(t *testing.T) {
					var got Request
					d := json.NewDecoder(strings.NewReader(tt.in))
					d.DisallowUnknownFields()
					if err := d.Decode(&got); err == nil || err.Error() != tt.errMsg {
						t.Fatalf("error mismatch\nwant %q\ngot  %q", tt.errMsg, err)
					}
				})
			}
		})
	})
}

func TestReply(t *testing.T) {
	t.Run("Validate", func(t *testing.T) {
		t.Run("valid", func(t *testing.T) {
			tests := []struct {
				name string
				in   Reply
			}{
				{
					name: "Valid text block",
					in:   Reply{Text: "Hello"},
				},
				{
					name: "Valid document block",
					in:   Reply{Doc: Doc{Filename: "document.txt", Src: strings.NewReader("document content")}},
				},
				{
					name: "valid text with citations",
					in: Reply{
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
					in:   Reply{Citations: []Citation{{Text: "example"}}},
				},
				{
					// Technically it need a source. wantErr: true,
					name: "invalid citation",
					in: Reply{
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
				name   string
				in     Reply
				errMsg string
			}{
				{
					name: "citations with thinking",
					in: Reply{
						Thinking:  "reasoning",
						Citations: []Citation{{Text: "example"}},
					},
					errMsg: "field Citations can only be used with Text",
				},
			}
			for _, tt := range tests {
				t.Run(tt.name, func(t *testing.T) {
					if err := tt.in.Validate(); err == nil || err.Error() != tt.errMsg {
						t.Fatalf("error mismatch\nwant %q\ngot  %q", tt.errMsg, err)
					}
				})
			}
		})
	})
	t.Run("Read", func(t *testing.T) {
		t.Run("valid", func(t *testing.T) {
			c := Reply{
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
				in   string
				want Reply
			}{
				{
					name: "Text content",
					in:   `{"text": "Hello world"}`,
					want: Reply{Text: "Hello world"},
				},
				{
					name: "Thinking content",
					in:   `{"thinking": "Let me think about this"}`,
					want: Reply{Thinking: "Let me think about this"},
				},
				{
					name: "Opaque content",
					in:   `{"opaque": {"key": "value", "num": 42}}`,
					want: Reply{Opaque: map[string]any{"key": "value", "num": float64(42)}},
				},
				{
					name: "URL content",
					in:   `{"doc":{"filename": "image.jpg", "url": "https://example.com/image.jpg"}}`,
					want: Reply{Doc: Doc{Filename: "image.jpg", URL: "https://example.com/image.jpg"}},
				},
				{
					name: "Document content",
					in:   `{"doc":{"filename": "doc.txt", "bytes": "SGVsbG8gV29ybGQ="}}`,
					want: Reply{Doc: Doc{Filename: "doc.txt", Src: strings.NewReader("Hello World")}},
				},
			}
			for _, tt := range tests {
				t.Run(tt.name, func(t *testing.T) {
					var got Reply
					d := json.NewDecoder(strings.NewReader(tt.in))
					d.DisallowUnknownFields()
					if err := d.Decode(&got); err != nil {
						t.Fatalf("unexpected error: %v", err)
					}
					// For Document comparison, read the content since we can't directly compare io.ReadSeeker
					if tt.want.Doc.Src != nil {
						wantData, _ := io.ReadAll(tt.want.Doc.Src)
						gotData, _ := io.ReadAll(got.Doc.Src)
						if string(wantData) != string(gotData) {
							t.Fatalf("Document content mismatch: want %q, got %q", string(wantData), string(gotData))
						}
						// Reset Document field for comparison
						tt.want.Doc.Src = nil
						got.Doc.Src = nil
					}
					if diff := cmp.Diff(tt.want, got); diff != "" {
						t.Fatalf("Reply mismatch (-want +got):\n%s", diff)
					}
				})
			}
		})

		t.Run("error", func(t *testing.T) {
			tests := []struct {
				name   string
				in     string
				errMsg string
			}{
				{
					name:   "Invalid JSON",
					in:     `{"text": invalid}`,
					errMsg: "invalid character 'i' looking for beginning of value",
				},
				{
					name:   "Unknown field",
					in:     `{"text": "Hi", "unknown_field": "value"}`,
					errMsg: "json: unknown field \"unknown_field\"",
				},
				{
					name:   "Text and Thinking together",
					in:     `{"text": "Hello", "thinking": "Let me think"}`,
					errMsg: "field Thinking can't be used along Text",
				},
				{
					name:   "Document without filename",
					in:     `{"doc":{"bytes": "SGVsbG8="}}`,
					errMsg: "field Filename is required with Document when not implementing Name()",
				},
				{
					name:   "Empty content",
					in:     `{}`,
					errMsg: "an empty Reply is invalid",
				},
			}
			for _, tt := range tests {
				t.Run(tt.name, func(t *testing.T) {
					var got Reply
					d := json.NewDecoder(strings.NewReader(tt.in))
					d.DisallowUnknownFields()
					if err := d.Decode(&got); err == nil || err.Error() != tt.errMsg {
						t.Fatalf("error mismatch\nwant %q\ngot  %q", tt.errMsg, err)
					}
				})
			}
		})
	})
}

func TestContentFragment(t *testing.T) {
	t.Run("IsZero", func(t *testing.T) {
		tests := []struct {
			name string
			in   ContentFragment
			want bool
		}{
			{
				name: "zero fragment",
				in:   ContentFragment{},
				want: true,
			},
			{
				name: "text fragment",
				in:   ContentFragment{TextFragment: "Hello"},
				want: false,
			},
			{
				name: "thinking fragment",
				in:   ContentFragment{ThinkingFragment: "thinking"},
				want: false,
			},
			{
				name: "opaque fragment",
				in:   ContentFragment{Opaque: map[string]any{"key": "value"}},
				want: false,
			},
			{
				name: "document fragment",
				in:   ContentFragment{DocumentFragment: []byte("data")},
				want: false,
			},
			{
				name: "tool call fragment",
				in:   ContentFragment{ToolCall: ToolCall{Name: "tool"}},
				want: false,
			},
			{
				name: "citation fragment",
				in:   ContentFragment{Citation: Citation{Text: "citation"}},
				want: false,
			},
		}
		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				if got := tt.in.IsZero(); got != tt.want {
					t.Fatalf("ContentFragment.IsZero() = %v, want %v", got, tt.want)
				}
			})
		}
	})

	t.Run("GoString", func(t *testing.T) {
		fragment := ContentFragment{
			TextFragment: "Hello",
		}
		got := fragment.GoString()
		// Just check that it returns a valid JSON string
		if !strings.HasPrefix(got, "{") || !strings.HasSuffix(got, "}") {
			t.Fatalf("ContentFragment.GoString() = %q, want JSON object", got)
		}
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
				in   string
				want ToolCall
			}{
				{
					name: "Complete tool call",
					in:   `{"id": "call_123", "name": "calculator", "arguments": "{\"a\": 5, \"b\": 3}"}`,
					want: ToolCall{ID: "call_123", Name: "calculator", Arguments: "{\"a\": 5, \"b\": 3}"},
				},
				{
					name: "Tool call with only name",
					in:   `{"name": "weather", "arguments": "{}"}`,
					want: ToolCall{Name: "weather", Arguments: "{}"},
				},
				{
					name: "Tool call with only ID",
					in:   `{"id": "call_456", "arguments": "{}"}`,
					want: ToolCall{ID: "call_456", Arguments: "{}"},
				},
			}
			for _, tt := range tests {
				t.Run(tt.name, func(t *testing.T) {
					var got ToolCall
					d := json.NewDecoder(strings.NewReader(tt.in))
					d.DisallowUnknownFields()
					if err := d.Decode(&got); err != nil {
						t.Fatalf("unexpected error: %v", err)
					}
					if diff := cmp.Diff(tt.want, got); diff != "" {
						t.Fatalf("ToolCall mismatch (-want +got):\n%s", diff)
					}
				})
			}
		})

		t.Run("error", func(t *testing.T) {
			tests := []struct {
				name   string
				in     string
				errMsg string
			}{
				{
					name:   "Invalid JSON",
					in:     `{"name": "tool", "arguments": invalid}`,
					errMsg: "invalid character 'i' looking for beginning of value",
				},
				{
					name:   "Invalid arguments JSON",
					in:     `{"name": "tool", "arguments": "invalid json"}`,
					errMsg: "field Arguments: invalid character 'i' looking for beginning of value",
				},
				{
					name:   "Missing both ID and name",
					in:     `{"arguments": "{}"}`,
					errMsg: "at least one of field ID or Name is required",
				},
			}
			for _, tt := range tests {
				t.Run(tt.name, func(t *testing.T) {
					var got ToolCall
					d := json.NewDecoder(strings.NewReader(tt.in))
					d.DisallowUnknownFields()
					if err := d.Decode(&got); err == nil || err.Error() != tt.errMsg {
						t.Fatalf("error mismatch\nwant %q\ngot  %q", tt.errMsg, err)
					}
				})
			}
		})
	})
}

func TestToolCallResult(t *testing.T) {
	t.Run("UnmarshalJSON", func(t *testing.T) {
		t.Run("valid", func(t *testing.T) {
			tests := []struct {
				name string
				in   string
				want ToolCallResult
			}{
				{
					name: "Complete tool call result",
					in:   `{"id": "call_123", "name": "calculator", "result": "8"}`,
					want: ToolCallResult{ID: "call_123", Name: "calculator", Result: "8"},
				},
				{
					name: "Tool call result with only name",
					in:   `{"name": "weather", "result": "sunny"}`,
					want: ToolCallResult{Name: "weather", Result: "sunny"},
				},
				{
					name: "Tool call result with only ID",
					in:   `{"id": "call_456", "result": "success"}`,
					want: ToolCallResult{ID: "call_456", Result: "success"},
				},
			}
			for _, tt := range tests {
				t.Run(tt.name, func(t *testing.T) {
					var got ToolCallResult
					d := json.NewDecoder(strings.NewReader(tt.in))
					d.DisallowUnknownFields()
					if err := d.Decode(&got); err != nil {
						t.Fatalf("unexpected error: %v", err)
					}
					if diff := cmp.Diff(tt.want, got); diff != "" {
						t.Fatalf("ToolCallResult mismatch (-want +got):\n%s", diff)
					}
				})
			}
		})

		t.Run("error", func(t *testing.T) {
			tests := []struct {
				name   string
				in     string
				errMsg string
			}{
				{
					name:   "Invalid JSON",
					in:     `{"name": "tool", "result": invalid}`,
					errMsg: "invalid character 'i' looking for beginning of value",
				},
				{
					name:   "Unknown field",
					in:     `{"name": "tool", "result": "success", "unknown_field": "value"}`,
					errMsg: "json: unknown field \"unknown_field\"",
				},
				{
					name:   "Missing both ID and name",
					in:     `{"result": "success"}`,
					errMsg: "at least one of field ID or Name is required",
				},
				{
					name:   "Missing result",
					in:     `{"name": "tool"}`,
					errMsg: "field Result: required",
				},
			}
			for _, tt := range tests {
				t.Run(tt.name, func(t *testing.T) {
					var got ToolCallResult
					d := json.NewDecoder(strings.NewReader(tt.in))
					d.DisallowUnknownFields()
					if err := d.Decode(&got); err == nil || err.Error() != tt.errMsg {
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
				name   string
				in     Citation
				errMsg string
			}{
				{
					name: "valid citation",
					in: Citation{
						Text:       "example text",
						StartIndex: 0,
						EndIndex:   12,
						Sources:    []CitationSource{{ID: "doc1", Type: "document"}},
					},
				},
				{
					name: "empty text",
					in: Citation{
						Text:       "",
						StartIndex: 0,
						EndIndex:   10,
					},
				},
				{
					name: "zero end index is valid",
					in: Citation{
						Text:       "example",
						StartIndex: 0,
						EndIndex:   0, // Zero is allowed as it may indicate position-only citation
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
				name   string
				in     Citation
				errMsg string
			}{
				{
					name: "negative start index",
					in: Citation{
						Text:       "example",
						StartIndex: -1,
						EndIndex:   10,
					},
					errMsg: "start index must be non-negative, got -1",
				},
				{
					name: "end index before start",
					in: Citation{
						Text:       "example",
						StartIndex: 10,
						EndIndex:   5,
					},
					errMsg: "end index (5) must be greater than start index (10)",
				},
				{
					name: "end index equal to start",
					in: Citation{
						Text:       "example",
						StartIndex: 10,
						EndIndex:   10,
					},
					errMsg: "end index (10) must be greater than start index (10)",
				},
				{
					name: "invalid citation source",
					in: Citation{
						Text:    "example",
						Sources: []CitationSource{{}},
					},
					errMsg: "source 0: citation source must have either ID or URL",
				},
				{
					name: "end index equal to start",
					in: Citation{
						Text:       "example",
						StartIndex: 10,
						EndIndex:   10,
					},
					errMsg: "end index (10) must be greater than start index (10)",
				},
			}
			for _, tt := range tests {
				t.Run(tt.name, func(t *testing.T) {
					if err := tt.in.Validate(); err == nil || err.Error() != tt.errMsg {
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
				name string
				in   CitationSource
			}{
				{
					name: "valid with ID",
					in:   CitationSource{ID: "doc1", Type: "document"},
				},
				{
					name: "valid with URL",
					in:   CitationSource{URL: "https://example.com", Type: "web"},
				},
				{
					name: "valid with both ID and URL",
					in:   CitationSource{ID: "doc1", URL: "https://example.com", Type: "document"},
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
				name   string
				in     CitationSource
				errMsg string
			}{
				{
					name:   "invalid without ID or URL",
					in:     CitationSource{Type: "document"},
					errMsg: "citation source must have either ID or URL",
				},
			}
			for _, tt := range tests {
				t.Run(tt.name, func(t *testing.T) {
					if err := tt.in.Validate(); err == nil || err.Error() != tt.errMsg {
						t.Fatalf("error mismatch\nwant %q\ngot  %q", tt.errMsg, err)
					}
				})
			}
		})
	})

	t.Run("IsZero", func(t *testing.T) {
		tests := []struct {
			name string
			in   CitationSource
			want bool
		}{
			{
				name: "zero value",
				in:   CitationSource{},
				want: true,
			},
			{
				name: "with ID",
				in:   CitationSource{ID: "doc1"},
				want: false,
			},
			{
				name: "with Type",
				in:   CitationSource{Type: "document"},
				want: false,
			},
			{
				name: "with Title",
				in:   CitationSource{Title: "title"},
				want: false,
			},
			{
				name: "with URL",
				in:   CitationSource{URL: "https://example.com"},
				want: false,
			},
			{
				name: "with Metadata",
				in:   CitationSource{Metadata: map[string]any{"key": "value"}},
				want: false,
			},
		}

		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				if got := tt.in.IsZero(); got != tt.want {
					t.Fatalf("CitationSource.IsZero() got = %v, want %v", got, tt.want)
				}
			})
		}
	})
}

func TestRateLimit(t *testing.T) {
	t.Run("Validate", func(t *testing.T) {
		t.Run("valid", func(t *testing.T) {
			tests := []struct {
				name string
				in   RateLimit
			}{
				{
					name: "valid requests per minute",
					in: RateLimit{
						Type:      Requests,
						Period:    PerMinute,
						Limit:     100,
						Remaining: 50,
						Reset:     time.Now(),
					},
				},
				{
					name: "valid tokens per day",
					in: RateLimit{
						Type:      Tokens,
						Period:    PerDay,
						Limit:     10000,
						Remaining: 5000,
						Reset:     time.Now(),
					},
				},
				{
					name: "valid with other period",
					in: RateLimit{
						Type:      Tokens,
						Period:    PerOther,
						Limit:     1000,
						Remaining: 500,
						Reset:     time.Now(),
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
			now := time.Now()
			tests := []struct {
				name   string
				in     RateLimit
				errMsg string
			}{
				{
					name: "invalid type",
					in: RateLimit{
						Type:      "invalid",
						Period:    PerMinute,
						Limit:     100,
						Remaining: 50,
						Reset:     now,
					},
					errMsg: "unknown limit type \"invalid\"",
				},
				{
					name: "zero limit",
					in: RateLimit{
						Type:      Requests,
						Period:    PerMinute,
						Limit:     0,
						Remaining: 50,
						Reset:     now,
					},
					errMsg: "limit is 0",
				},
				{
					name: "invalid period",
					in: RateLimit{
						Type:      Requests,
						Period:    "invalid",
						Limit:     100,
						Remaining: 50,
						Reset:     now,
					},
					errMsg: "unknown limit period \"invalid\"",
				},
				{
					name: "zero reset",
					in: RateLimit{
						Type:      Requests,
						Period:    PerMinute,
						Limit:     100,
						Remaining: 50,
						Reset:     time.Time{},
					},
					errMsg: "reset is 0",
				},
			}
			for _, tt := range tests {
				t.Run(tt.name, func(t *testing.T) {
					if err := tt.in.Validate(); err == nil || err.Error() != tt.errMsg {
						t.Fatalf("error mismatch\nwant %q\ngot  %q", tt.errMsg, err)
					}
				})
			}
		})
	})

	t.Run("String", func(t *testing.T) {
		now := time.Now()
		tests := []struct {
			name string
			in   RateLimit
			want string
		}{
			{
				name: "requests per minute",
				in: RateLimit{
					Type:      Requests,
					Period:    PerMinute,
					Limit:     100,
					Remaining: 50,
					Reset:     now,
				},
				want: fmt.Sprintf("requests/%s (minute): 50/100", now),
			},
			{
				name: "tokens per day",
				in: RateLimit{
					Type:      Tokens,
					Period:    PerDay,
					Limit:     10000,
					Remaining: 5000,
					Reset:     now,
				},
				want: fmt.Sprintf("tokens/%s (day): 5000/10000", now),
			},
			{
				name: "other period",
				in: RateLimit{
					Type:      Tokens,
					Period:    PerOther,
					Limit:     1000,
					Remaining: 500,
					Reset:     now,
				},
				want: fmt.Sprintf("tokens/%s: 500/1000", now),
			},
			{
				name: "zero reset",
				in: RateLimit{
					Type:      Requests,
					Period:    PerMinute,
					Limit:     100,
					Remaining: 50,
					Reset:     time.Time{},
				},
				want: "requests (minute): 50/100",
			},
		}
		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				if got := tt.in.String(); got != tt.want {
					t.Fatalf("RateLimit.String()\nwant %q\ngot  %q", tt.want, got)
				}
			})
		}
	})
}

func TestTriState(t *testing.T) {
	t.Run("String", func(t *testing.T) {
		tests := []struct {
			name string
			in   TriState
			want string
		}{
			{
				name: "False",
				in:   False,
				want: "false",
			},
			{
				name: "True",
				in:   True,
				want: "true",
			},
			{
				name: "Flaky",
				in:   Flaky,
				want: "flaky",
			},
			{
				name: "Unknown value",
				in:   TriState(99),
				want: "TriState(99)",
			},
		}

		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				if got := tt.in.String(); got != tt.want {
					t.Fatalf("TriState.String() got = %q, want %q", got, tt.want)
				}
			})
		}
	})

	t.Run("GoString", func(t *testing.T) {
		tests := []struct {
			name string
			in   TriState
			want string
		}{
			{
				name: "False",
				in:   False,
				want: "false",
			},
			{
				name: "True",
				in:   True,
				want: "true",
			},
			{
				name: "Flaky",
				in:   Flaky,
				want: "flaky",
			},
		}

		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				if got := tt.in.GoString(); got != tt.want {
					t.Fatalf("TriState.GoString() got = %q, want %q", got, tt.want)
				}
			})
		}
	})
}
