// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package adapters_test

import (
	"context"
	"errors"
	"fmt"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/maruel/genai"
	"github.com/maruel/genai/adapters"
	"golang.org/x/sync/errgroup"
)

func TestProviderGenThinking_GenSync(t *testing.T) {
	tests := []struct {
		name    string
		tagName string
		in      string
		opts    genai.Options
		want    []genai.Content
	}{
		{
			name: "No thinking tags",
			in:   "Just regular text without thinking tags",
			want: []genai.Content{{Text: "Just regular text without thinking tags"}},
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
			want: []genai.Content{{Thinking: "Thinking with whitespace"}},
		},
		{
			name: "With empty text content",
			want: []genai.Content{{}},
		},
		{
			name: "SkipJSON with ReplyAsJSON",
			in:   "<thinking>This is thinking</thinking>Response",
			opts: &genai.OptionsText{ReplyAsJSON: true},
			want: []genai.Content{{Text: "<thinking>This is thinking</thinking>Response"}},
		},
		{
			name: "SkipJSON with DecodeAs",
			in:   "<thinking>This is thinking</thinking>Response",
			opts: &genai.OptionsText{DecodeAs: &struct{ A string }{}},
			want: []genai.Content{{Text: "<thinking>This is thinking</thinking>Response"}},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			mp := &mockProviderGenSync{response: genai.Result{Message: genai.NewTextMessage(genai.Assistant, tc.in)}}
			tp := &adapters.ProviderGenThinking{ProviderGen: mp, TagName: "thinking", SkipJSON: true}
			got, err := tp.GenSync(t.Context(), genai.Messages{}, tc.opts)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if diff := cmp.Diff(tc.want, got.Contents); diff != "" {
				t.Fatalf("diff:\n%s", diff)
			}
		})
	}
}

func TestProviderGenThinking_GenSync_errors(t *testing.T) {
	tests := []struct {
		name    string
		tagName string
		in      []genai.Content
		err     error
		want    string
	}{
		{
			name: "With non-empty content before tag",
			in:   []genai.Content{{Text: "Text before <thinking>\nThis is thinking</thinking>\nThis is response"}},
			want: "failed to parse thinking tokens",
		},
		{
			name: "Error from underlying GenSync",
			err:  errors.New("mock error"),
			want: "mock error",
		},
		{
			name: "Multiple content blocks",
			in:   []genai.Content{{Text: "First part. "}, {Text: "<thinking>Thinking part</thinking>"}, {Text: " Second part."}},
			want: "failed to parse thinking tokens",
		},
		{
			name: "Message with existing thinking content",
			in:   []genai.Content{{Thinking: "Existing thinking"}, {Text: "Some text"}},
			want: `got unexpected thinking content: "Existing thinking"; do not use ProviderGenThinking with an explicit thinking CoT model`,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			mp := &mockProviderGenSync{response: genai.Result{Message: genai.Message{Role: genai.Assistant, Contents: tc.in}}, err: tc.err}
			tp := &adapters.ProviderGenThinking{ProviderGen: mp, TagName: "thinking"}
			_, err := tp.GenSync(t.Context(), genai.Messages{}, nil)
			if err == nil {
				t.Fatal("expected error but got none")
			}
			if got := err.Error(); got != tc.want {
				t.Fatalf("expected %q but got %q", tc.want, got)
			}
		})
	}
}

func TestProviderGenThinking_GenStream(t *testing.T) {
	tests := []struct {
		name string
		in   []string
		opts genai.Options
		want []genai.Content
	}{
		{
			name: "No thinking tags - assumed to be thinking content",
			in:   []string{"Just ", "regular", " text", " without ", "thinking ", "tags"},
			want: []genai.Content{{Thinking: "Just regular text without thinking tags"}},
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
			want: []genai.Content{{Thinking: "Fragment1Fragment2Fragment3"}},
		},
		{
			name: "With thinking tag at the end",
			in:   []string{"<thinking>", "This is thinking only"},
			want: []genai.Content{{Thinking: "This is thinking only"}},
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
		{
			name: "SkipJSON with ReplyAsJSON",
			in:   []string{"<thinking>This is thinking</thinking>Response"},
			opts: &genai.OptionsText{ReplyAsJSON: true},
			want: []genai.Content{{Text: "<thinking>This is thinking</thinking>Response"}},
		},
		{
			name: "SkipJSON with DecodeAs",
			in:   []string{"<thinking>This is thinking</thinking>Response"},
			opts: &genai.OptionsText{DecodeAs: &struct{ A string }{}},
			want: []genai.Content{{Text: "<thinking>This is thinking</thinking>Response"}},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			mp := &mockProviderGenStream{streamResponses: []streamResponse{{}}}
			for _, i := range tc.in {
				mp.streamResponses[0].fragments = append(mp.streamResponses[0].fragments, genai.ContentFragment{TextFragment: i})
			}
			tp := &adapters.ProviderGenThinking{ProviderGen: mp, TagName: "thinking", SkipJSON: true}
			ch := make(chan genai.ContentFragment, 100)
			eg, ctx := errgroup.WithContext(t.Context())
			accumulated := genai.Message{}
			eg.Go(func() error {
				for pkt := range ch {
					if err2 := accumulated.Accumulate(pkt); err2 != nil {
						t.Error(err2)
						return err2
					}
				}
				return nil
			})
			result, err := tp.GenStream(ctx, genai.Messages{}, ch, tc.opts)
			close(ch)
			_ = eg.Wait()
			if err != nil {
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

func TestProviderGenThinking_GenStream_errors(t *testing.T) {
	tests := []struct {
		name      string
		in        []string
		fragments []genai.ContentFragment
		err       error
		want      string
	}{
		{
			name: "With text before tag in same fragment - error case",
			in:   []string{"Text before <thinking>", "This is thinking", "</thinking>", "This is response"},
			want: "unexpected prefix before thinking tag: \"\"",
		},
		{
			name: "Error from underlying GenStream",
			in:   []string{},
			err:  errors.New("mock stream error"),
			want: "mock stream error",
		},
		{
			name: "Unexpected thinking fragment in stream",
			in:   []string{},
			fragments: []genai.ContentFragment{
				{ThinkingFragment: "This is an unexpected thinking fragment"},
			},
			want: `got unexpected thinking fragment: "This is an unexpected thinking fragment"; do not use ProviderGenThinking with an explicit thinking CoT model`,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			mp := &mockProviderGenStream{streamResponses: []streamResponse{{}}, err: tc.err}
			if len(tc.fragments) > 0 {
				mp.streamResponses[0].fragments = tc.fragments
			} else {
				for _, i := range tc.in {
					mp.streamResponses[0].fragments = append(mp.streamResponses[0].fragments, genai.ContentFragment{TextFragment: i})
				}
			}
			tp := &adapters.ProviderGenThinking{ProviderGen: mp, TagName: "thinking"}
			ch := make(chan genai.ContentFragment, 100)
			eg := errgroup.Group{}
			eg.Go(func() error {
				for range ch {
				}
				return nil
			})
			_, err := tp.GenStream(t.Context(), genai.Messages{}, ch, nil)
			close(ch)
			_ = eg.Wait()
			if err == nil {
				t.Fatal("expected error but got none")
			}
			if got := err.Error(); got != tc.want {
				t.Fatalf("expected %q but got %q", tc.want, got)
			}
		})
	}
}

//

// TestGenStreamWithToolCallLoop tests the GenStreamWithToolCallLoop function.
func TestGenStreamWithToolCallLoop(t *testing.T) {
	provider := &mockProviderGenStream{
		streamResponses: []streamResponse{
			{
				fragments: []genai.ContentFragment{
					{TextFragment: "I'll help you calculate that. "},
					{TextFragment: "Let me use the calculator tool."},
					{ToolCall: genai.ToolCall{ID: "1", Name: "calculator", Arguments: `{"a": 5, "b": 3, "operation": "add"}`}},
				},
				usage: genai.Usage{InputTokens: 10, OutputTokens: 20},
			},
			{
				fragments: []genai.ContentFragment{
					{TextFragment: "The result of 5 + 3 is 8."},
				},
				usage: genai.Usage{InputTokens: 15, OutputTokens: 10},
			},
		},
	}
	msgs := genai.Messages{genai.NewTextMessage(genai.User, "Calculate 5 + 3")}
	type CalculatorArgs struct {
		A         int    `json:"a"`
		B         int    `json:"b"`
		Operation string `json:"operation"`
	}
	opts := &genai.OptionsText{
		Tools: []genai.ToolDef{
			{
				Name:        "calculator",
				Description: "A simple calculator",
				Callback: func(ctx context.Context, args *CalculatorArgs) (string, error) {
					switch args.Operation {
					case "add":
						return fmt.Sprintf("%d", args.A+args.B), nil
					default:
						return "", fmt.Errorf("unsupported operation: %s", args.Operation)
					}
				},
			},
		},
	}
	chunks := make(chan genai.ContentFragment)
	var frags []genai.ContentFragment
	ctx := t.Context()
	go func() error {
		for {
			select {
			case <-ctx.Done():
				return ctx.Err()
			case fragment, ok := <-chunks:
				if !ok {
					return nil
				}
				frags = append(frags, fragment)
			}
		}
	}()

	respMsgs, usage, err := adapters.GenStreamWithToolCallLoop(ctx, provider, msgs, chunks, opts)
	close(chunks)
	if err != nil {
		t.Fatalf("GenStreamWithToolCallLoop returned an error: %v", err)
	}
	// Verify we got the expected number of messages
	if len(respMsgs) != 3 { // original + 1 LLM response + 1 tool result
		t.Fatalf("Expected 3 messages, got %d", len(respMsgs))
	}
	t.Logf("Messages: %+v", respMsgs)
	expectedUsage := genai.Usage{InputTokens: 25, OutputTokens: 30}
	if usage.InputTokens != expectedUsage.InputTokens || usage.OutputTokens != expectedUsage.OutputTokens {
		t.Fatalf("Expected usage %+v, got %+v", expectedUsage, usage)
	}
	// Verify we received all fragments
	if len(frags) != 4 { // 2 text fragments + 1 tool call + 1 text fragment
		t.Fatalf("Expected 4 fragments, got %d", len(frags))
	}
}

// Mock types for testing

type mockProviderGenSync struct {
	response genai.Result
	err      error
}

func (m *mockProviderGenSync) Name() string {
	return "mock"
}

func (m *mockProviderGenSync) GenSync(ctx context.Context, msgs genai.Messages, opts genai.Options) (genai.Result, error) {
	return m.response, m.err
}

func (m *mockProviderGenSync) GenStream(ctx context.Context, msgs genai.Messages, replies chan<- genai.ContentFragment, opts genai.Options) (genai.Result, error) {
	return genai.Result{}, errors.New("unexpected")
}

func (m *mockProviderGenSync) ModelID() string {
	return "llm-sota"
}

type streamResponse struct {
	fragments []genai.ContentFragment
	usage     genai.Usage
}

type mockProviderGenStream struct {
	streamResponses []streamResponse
	callIndex       int
	err             error
}

func (m *mockProviderGenStream) Name() string {
	return "mock"
}

func (m *mockProviderGenStream) GenSync(ctx context.Context, msgs genai.Messages, opts genai.Options) (genai.Result, error) {
	return genai.Result{}, errors.New("unexpected")
}

func (m *mockProviderGenStream) GenStream(ctx context.Context, msgs genai.Messages, replies chan<- genai.ContentFragment, opts genai.Options) (genai.Result, error) {
	if m.err != nil {
		return genai.Result{}, m.err
	}
	if m.callIndex >= len(m.streamResponses) {
		return genai.Result{}, fmt.Errorf("no more mock responses")
	}
	resp := m.streamResponses[m.callIndex]
	m.callIndex++
	result := genai.Result{
		Usage:   resp.usage,
		Message: genai.Message{Role: genai.Assistant},
	}
	for _, fragment := range resp.fragments {
		select {
		case <-ctx.Done():
			return result, ctx.Err()
		case replies <- fragment:
			if err := result.Accumulate(fragment); err != nil {
				return result, err
			}
		}
	}
	return result, nil
}

func (m *mockProviderGenStream) ModelID() string {
	return "llm-sota"
}
