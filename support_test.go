// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package genai_test

import (
	"context"
	"errors"
	"sync"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"github.com/maruel/genai"
)

func TestProviderGenThinking_GenSync(t *testing.T) {
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
			name: "With only whitespace before tag and cut off",
			in:   "  \n\t<thinking>\nThinking with whitespace",
			want: []genai.Content{
				{Thinking: "Thinking with whitespace"},
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
			mp := &mockProviderGen{response: genai.Result{
				Message: genai.Message{
					Role:     genai.Assistant,
					Contents: []genai.Content{{Text: tc.in}},
				},
			}}

			tp := &genai.ProviderGenThinking{ProviderGen: mp, TagName: "thinking"}
			got, err := tp.GenSync(t.Context(), genai.Messages{}, nil)
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

func TestProviderGenThinking_GenStream(t *testing.T) {
	tests := []struct {
		name        string
		in          []string
		want        []genai.Content
		expectError bool
	}{
		{
			name: "No thinking tags - assumed to be thinking content",
			in:   []string{"Just ", "regular", " text", " without ", "thinking ", "tags"},
			want: []genai.Content{
				{Thinking: "Just regular text without thinking tags"},
			},
		},
		{
			name: "With thinking tags in separate fragments",
			in:   []string{"<thinking>", "This is my ", "thinking process", "</thinking>", "This is the response"},
			want: []genai.Content{
				{Thinking: "This is my thinking process"},
				{Text: "This is the response"},
			},
		},
		{
			name:        "With text before tag in same fragment - error case",
			in:          []string{"Text before <thinking>", "This is thinking", "</thinking>", "This is response"},
			expectError: true,
		},
		{
			name: "With whitespace before tag",
			in:   []string{"  \n\t<thinking>", "Thinking content", "</thinking>", "Response"},
			want: []genai.Content{
				{Thinking: "Thinking content"},
				{Text: "Response"},
			},
		},
		{
			name: "With whitespace before tag as a separate packet",
			in:   []string{"  \n\t", "<thinking>", "Thinking content", "</thinking>", "Response"},
			want: []genai.Content{
				{Thinking: "Thinking content"},
				{Text: "Response"},
			},
		},
		{
			name: "Handling fragmented content with regular text only",
			in:   []string{"Fragment1", "Fragment2", "Fragment3"},
			want: []genai.Content{
				{Thinking: "Fragment1Fragment2Fragment3"},
			},
		},
		{
			name: "With thinking tag at the end",
			in:   []string{"<thinking>", "This is thinking only"},
			want: []genai.Content{
				{Thinking: "This is thinking only"},
			},
		},
		{
			name: "With start tag and text in same fragment",
			in:   []string{"<thinking>Some text", " after tag", "</thinking>", "Response"},
			want: []genai.Content{
				{Thinking: "Some text after tag"},
				{Text: "Response"},
			},
		},

		{
			name: "With end tag and response in same fragment",
			in:   []string{"<thinking>", "Thinking", "</thinking>Response"},
			want: []genai.Content{
				{Thinking: "Thinking"},
				{Text: "Response"},
			},
		},
		{
			name: "End tag not at start of fragment",
			in:   []string{"<thinking>", "Thinking1", "Thinking2</thinking>", "Response"},
			want: []genai.Content{
				{Thinking: "Thinking1Thinking2"},
				{Text: "Response"},
			},
		},
		{
			name: "Text state fragments",
			in:   []string{"<thinking>", "Thinking", "</thinking>", "Response1", "Response2"},
			want: []genai.Content{
				{Thinking: "Thinking"},
				{Text: "Response1Response2"},
			},
		},
		{
			name: "End tag at the start of a fragment",
			in:   []string{"<thinking>", "Thinking content", "</thinking>", "Response"},
			want: []genai.Content{
				{Thinking: "Thinking content"},
				{Text: "Response"},
			},
		},
		{
			name: "Text after start tag",
			in:   []string{"<thinking>\nOkay", " content", "</thinking>", "Response"},
			want: []genai.Content{
				{Thinking: "Okay content"},
				{Text: "Response"},
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			mp := &mockGenStreamProvider{in: tc.in}
			tp := &genai.ProviderGenThinking{ProviderGen: mp, TagName: "thinking"}
			ch := make(chan genai.ContentFragment)
			ctx, cancel := context.WithTimeout(t.Context(), 1*time.Second)
			defer cancel()
			wg := sync.WaitGroup{}
			wg.Add(1)
			accumErrCh := make(chan error, 1)
			accumulated := genai.Message{}
			go func() {
				defer wg.Done()
				for pkt := range ch {
					var err2 error
					if err2 = accumulated.Accumulate(pkt); err2 != nil {
						accumErrCh <- err2
						return
					}
				}
			}()

			result, err := tp.GenStream(ctx, genai.Messages{}, nil, ch)
			close(ch)
			wg.Wait()
			select {
			case accErr := <-accumErrCh:
				t.Fatalf("error accumulating messages: %v", accErr)
			default:
			}
			if tc.expectError {
				if err == nil {
					t.Fatal("expected error but got none")
				}
				return
			} else if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if diff := cmp.Diff(tc.want, accumulated.Contents); diff != "" {
				t.Fatalf("diff:\n%s", diff)
			}
			if diff := cmp.Diff(result.Message, accumulated); diff != "" {
				t.Fatalf("diff:\n%s", diff)
			}
		})
	}
}

//

// mockProviderGen is a mock implementation of genai.ProviderGen for testing.
// It returns a predefined Result for the GenSync method and returns an error
// for the GenStream method.
type mockProviderGen struct {
	response genai.Result
}

func (m *mockProviderGen) Name() string {
	return "mock"
}

func (m *mockProviderGen) GenSync(ctx context.Context, msgs genai.Messages, opts genai.Options) (genai.Result, error) {
	return m.response, nil
}

func (m *mockProviderGen) GenStream(ctx context.Context, msgs genai.Messages, opts genai.Options, replies chan<- genai.ContentFragment) (genai.Result, error) {
	return genai.Result{}, errors.New("unexpected")
}

func (m *mockProviderGen) ModelID() string {
	return "llm-sota"
}

// mockGenStreamProvider is a mock implementation of genai.ProviderGen for testing.
// It sends the predefined fragments to the replies channel for the GenStream method
// and returns an error for the GenSync method.
type mockGenStreamProvider struct {
	in []string
}

func (m *mockGenStreamProvider) Name() string {
	return "mockstream"
}

func (m *mockGenStreamProvider) GenSync(ctx context.Context, msgs genai.Messages, opts genai.Options) (genai.Result, error) {
	return genai.Result{}, errors.New("unexpected")
}

func (m *mockGenStreamProvider) GenStream(ctx context.Context, msgs genai.Messages, opts genai.Options, replies chan<- genai.ContentFragment) (genai.Result, error) {
	result := genai.Result{
		Usage:   genai.Usage{},
		Message: genai.Message{Role: genai.Assistant},
	}

	for _, f := range m.in {
		fragment := genai.ContentFragment{TextFragment: f}
		select {
		case <-ctx.Done():
			return result, ctx.Err()
		case replies <- fragment:
			// We don't accumulate in the result.Message since the wrapper will transform TextFragment to ThinkingFragment
			// and the actual accumulation happens in the test code. The final message will be constructed properly
			// by the ProviderGenThinking wrapper.
		}
	}
	return result, nil
}

func (m *mockGenStreamProvider) ModelID() string {
	return "llm-sota"
}
