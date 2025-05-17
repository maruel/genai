// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package genai_test

import (
	"context"
	"errors"
	"strings"
	"sync"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/maruel/genai"
)

func TestThinkingChat(t *testing.T) {
	tests := []struct {
		name        string
		tagName     string
		in          string
		want        []genai.Content
		expectError bool
	}{
		{
			name: "No thinking tags",
			in:   "Just regular text without thinking tags",
			want: []genai.Content{
				{Text: "Just regular text without thinking tags"},
			},
		},
		{
			name: "With thinking tags at the beginning",
			in:   "<thinking>\nThis is my thinking process</thinking>\nThis is the response",
			want: []genai.Content{
				{Thinking: "This is my thinking process"},
				{Text: "This is the response"},
			},
		},
		{
			name:        "With non-empty content before tag",
			in:          "Text before <thinking>\nThis is thinking</thinking>\nThis is response",
			expectError: true,
		},
		{
			name: "With only whitespace before tag",
			in:   "  \n\t<thinking>\nThinking with whitespace before</thinking>\nResponse",
			want: []genai.Content{
				{Thinking: "Thinking with whitespace before"},
				{Text: "Response"},
			},
		},
		{
			name: "With empty text content",
			in:   "",
			want: []genai.Content{{}},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			mp := &mockChatProvider{response: genai.ChatResult{
				Message: genai.Message{
					Role:     genai.Assistant,
					Contents: []genai.Content{{Text: tc.in}},
				},
			}}

			tp := &genai.ThinkingChatProvider{Provider: mp, TagName: "thinking"}
			got, err := tp.Chat(t.Context(), genai.Messages{}, nil)
			if tc.expectError {
				if err == nil {
					t.Fatal("expected error but got none")
				}
				return
			} else if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if diff := cmp.Diff(tc.want, got.Contents); diff != "" {
				t.Fatalf("diff:\n%s", diff)
			}
		})
	}
}

func TestThinkingChatStream(t *testing.T) {
	tests := []struct {
		name        string
		tagName     string
		in          string
		want        []genai.Content
		expectError bool
	}{
		{
			name: "No thinking tags",
			in:   "Just regular text without thinking tags",
			want: []genai.Content{
				{Thinking: "Just regular text without thinking tags"},
			},
		},
		{
			name: "short",
			in:   "<thinking>\nfoo</thinking>\nbar",
			want: []genai.Content{
				{Thinking: "foo"},
				{Text: "bar"},
			},
		},
		{
			name: "With thinking tags at the beginning",
			in:   "<thinking>\nThis is my thinking process</thinking>\nThis is the response",
			want: []genai.Content{
				{Thinking: "This is my thinking process"},
				{Text: "This is the response"},
			},
		},
		{
			name:        "With non-empty content before tag",
			in:          "Text before <thinking>\nThis is thinking</thinking>\nThis is response",
			expectError: true,
		},
		{
			name: "With only whitespace before tag",
			in:   "  \n\t<thinking>\nThinking with whitespace before</thinking>\nResponse",
			want: []genai.Content{
				{Thinking: "Thinking with whitespace before"},
				{Text: "Response"},
			},
		},
		{
			name: "With empty text content",
			in:   "",
			want: []genai.Content{},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			mp := &mockChatStreamProvider{in: tc.in}
			tp := &genai.ThinkingChatProvider{Provider: mp, TagName: "thinking"}
			ch := make(chan genai.MessageFragment)
			var msgs genai.Messages
			wg := sync.WaitGroup{}
			wg.Add(1)
			go func() {
				for pkt := range ch {
					var err2 error
					if msgs, err2 = pkt.Accumulate(msgs); err2 != nil {
						t.Error(err2)
					}
				}
				wg.Done()
			}()
			_, err := tp.ChatStream(t.Context(), genai.Messages{}, nil, ch)
			close(ch)
			wg.Wait()
			if tc.expectError {
				if err == nil {
					t.Fatal("expected error but got none")
				}
				return
			} else if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if len(msgs) != 1 {
				if len(tc.want) == 0 {
					return
				}
				t.Fatalf("expected one message, got %#v", msgs)
			}
			if diff := cmp.Diff(tc.want, msgs[0].Contents); diff != "" {
				t.Fatalf("diff:\n%s", diff)
			}
		})
	}
}

//

// mockChatProvider is a mock implementation of genai.ChatProvider for testing.
// It returns a predefined ChatResult for the Chat method and returns an error
// for the ChatStream method.
type mockChatProvider struct {
	response genai.ChatResult
}

func (m *mockChatProvider) Chat(ctx context.Context, msgs genai.Messages, opts genai.Validatable) (genai.ChatResult, error) {
	return m.response, nil
}

func (m *mockChatProvider) ChatStream(ctx context.Context, msgs genai.Messages, opts genai.Validatable, replies chan<- genai.MessageFragment) (genai.Usage, error) {
	return genai.Usage{}, errors.New("unexpected")
}

// mockChatStreamProvider is a mock implementation of genai.ChatProvider for testing.
// It sends the predefined fragments to the replies channel for the ChatStream method
// and returns an error for the Chat method.
type mockChatStreamProvider struct {
	in string
}

func (m *mockChatStreamProvider) Chat(ctx context.Context, msgs genai.Messages, opts genai.Validatable) (genai.ChatResult, error) {
	return genai.ChatResult{}, errors.New("unexpected")
}

func (m *mockChatStreamProvider) ChatStream(ctx context.Context, msgs genai.Messages, opts genai.Validatable, replies chan<- genai.MessageFragment) (genai.Usage, error) {
	for in := m.in; in != ""; {
		i := strings.IndexAny(in, "\n ")
		if i == -1 {
			replies <- genai.MessageFragment{TextFragment: in}
			break
		}
		replies <- genai.MessageFragment{TextFragment: in[:i+1]}
		in = in[i+1:]
	}
	return genai.Usage{}, nil
}
