// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package adapters_test

import (
	"context"
	"errors"
	"fmt"
	"iter"
	"net/http"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/maruel/genai"
	"github.com/maruel/genai/adapters"
	"github.com/maruel/genai/base"
	"github.com/maruel/genai/scoreboard"
)

func TestProviderReasoning(t *testing.T) {
	t.Run("GenSync", func(t *testing.T) {
		t.Run("valid", func(t *testing.T) {
			tests := []struct {
				name string
				in   string
				opts genai.Options
				want []genai.Reply
			}{
				{
					name: "No thinking tags",
					in:   "Just regular text without thinking tags",
					want: []genai.Reply{{Text: "Just regular text without thinking tags"}},
				},
				{
					name: "With thinking tags at the beginning",
					in:   "<thinking>\nThis is my thinking process</thinking>\nThis is the response",
					want: []genai.Reply{
						{Reasoning: "This is my thinking process"},
						{Text: "This is the response"},
					},
				},
				{
					name: "With only whitespace before tag",
					in:   "  \n\t<thinking>\nThinking with whitespace before</thinking>\nResponse",
					want: []genai.Reply{
						{Reasoning: "Thinking with whitespace before"},
						{Text: "Response"},
					},
				},
				{
					name: "With only whitespace before tag and cut off",
					in:   "  \n\t<thinking>\nThinking with whitespace",
					want: []genai.Reply{{Reasoning: "Thinking with whitespace"}},
				},
				{
					name: "JSON",
					in:   "{\"is_fruit\": true}",
					want: []genai.Reply{{Text: "{\"is_fruit\": true}"}},
				},
				{
					name: "With empty text content",
					want: []genai.Reply{{}},
				},
			}

			for _, tc := range tests {
				t.Run(tc.name, func(t *testing.T) {
					mp := &mockProviderGenSync{
						responses: []genai.Result{{Message: genai.Message{Replies: []genai.Reply{{Text: tc.in}}}}},
					}
					tp := &adapters.ProviderReasoning{
						Provider:            mp,
						ReasoningTokenStart: "<thinking>",
						ReasoningTokenEnd:   "</thinking>",
					}
					got, err := tp.GenSync(t.Context(), genai.Messages{}, tc.opts)
					if err != nil {
						t.Fatalf("unexpected error: %v", err)
					}
					if diff := cmp.Diff(tc.want, got.Replies); diff != "" {
						t.Fatalf("diff:\n%s", diff)
					}
				})
			}
			t.Run("errors", func(t *testing.T) {
				tests := []struct {
					name string
					in   []genai.Reply
					err  error
					want string
				}{
					{
						name: "With non-empty content before tag",
						in:   []genai.Reply{{Text: "Text before <thinking>\nThis is thinking</thinking>\nThis is response"}},
						want: "unexpected prefix before reasoning tag: \"Text before \"",
					},
					{
						name: "Error from underlying GenSync",
						err:  errors.New("mock error"),
						want: "mock error",
					},
					{
						name: "Multiple content blocks",
						in:   []genai.Reply{{Text: "First part. "}, {Text: "<thinking>Thinking part</thinking>"}, {Text: " Second part."}},
						want: "unexpected prefix before reasoning tag: \"First part. \"",
					},
					{
						name: "Message with existing thinking content",
						in:   []genai.Reply{{Reasoning: "Existing thinking"}, {Text: "Some text"}},
						want: `got unexpected reasoning content: "Existing thinking"; do not use ProviderReasoning with an explicit reasoning CoT model`,
					},
				}

				for _, tc := range tests {
					t.Run(tc.name, func(t *testing.T) {
						mp := &mockProviderGenSync{
							responses: []genai.Result{{Message: genai.Message{Replies: tc.in}}},
							err:       tc.err,
						}
						tp := &adapters.ProviderReasoning{
							Provider:            mp,
							ReasoningTokenStart: "<thinking>",
							ReasoningTokenEnd:   "</thinking>",
						}
						_, err := tp.GenSync(t.Context(), genai.Messages{})
						if err == nil {
							t.Fatal("expected error but got none")
						}
						if got := err.Error(); got != tc.want {
							t.Fatalf("invalid error\nwant %q\ngot  %q", tc.want, got)
						}
					})
				}
			})
		})
	})

	t.Run("GenSync", func(t *testing.T) {
		t.Run("valid", func(t *testing.T) {
			tests := []struct {
				name string
				in   []string
				opts genai.Options
				want []genai.Reply
			}{
				{
					name: "No thinking tag",
					in:   []string{"Just ", "regular", " text", " without ", "thinking ", "tags"},
					want: []genai.Reply{{Text: "Just regular text without thinking tags"}},
				},
				{
					name: "With thinking tags in separate fragments",
					in:   []string{"<thinking>", "This is my ", "thinking process", "</thinking>", "This is the response"},
					want: []genai.Reply{
						{Reasoning: "This is my thinking process"},
						{Text: "This is the response"},
					},
				},
				{
					name: "With whitespace before tag",
					in:   []string{"  \n\t<thinking>", "Thinking content", "</thinking>", "Response"},
					want: []genai.Reply{
						{Reasoning: "Thinking content"},
						{Text: "Response"},
					},
				},
				{
					name: "With whitespace before tag as a separate packet",
					in:   []string{"  \n\t", "<thinking>", "Thinking content", "</thinking>", "Response"},
					want: []genai.Reply{
						{Reasoning: "Thinking content"},
						{Text: "Response"},
					},
				},
				{
					name: "With thinking tag at the end",
					in:   []string{"<thinking>", "This is thinking only"},
					want: []genai.Reply{{Reasoning: "This is thinking only"}},
				},
				{
					name: "With start tag and text in same fragment",
					in:   []string{"<thinking>Some text", " after tag", "</thinking>", "Response"},
					want: []genai.Reply{
						{Reasoning: "Some text after tag"},
						{Text: "Response"},
					},
				},
				{
					name: "With end tag and response in same fragment",
					in:   []string{"<thinking>", "Thinking", "</thinking>Response"},
					want: []genai.Reply{
						{Reasoning: "Thinking"},
						{Text: "Response"},
					},
				},
				{
					name: "End tag not at start of fragment",
					in:   []string{"<thinking>", "Thinking1", "Thinking2</thinking>", "Response"},
					want: []genai.Reply{
						{Reasoning: "Thinking1Thinking2"},
						{Text: "Response"},
					},
				},
				{
					name: "Text state fragments",
					in:   []string{"<thinking>", "Thinking", "</thinking>", "Response1", "Response2"},
					want: []genai.Reply{
						{Reasoning: "Thinking"},
						{Text: "Response1Response2"},
					},
				},
				{
					name: "End tag at the start of a fragment",
					in:   []string{"<thinking>", "Thinking content", "</thinking>", "Response"},
					want: []genai.Reply{
						{Reasoning: "Thinking content"},
						{Text: "Response"},
					},
				},
				{
					name: "Text after start tag",
					in:   []string{"<thinking>\nOkay", " content", "</thinking>", "Response"},
					want: []genai.Reply{
						{Reasoning: "Okay content"},
						{Text: "Response"},
					},
				},
				{
					name: "JSON",
					in:   []string{"{\"is_fruit\": ", "true}"},
					want: []genai.Reply{{Text: "{\"is_fruit\": true}"}},
				},
			}

			for _, tc := range tests {
				t.Run(tc.name, func(t *testing.T) {
					mp := &mockProviderGenStream{streamResponses: []streamResponse{{}}}
					for _, i := range tc.in {
						mp.streamResponses[0].fragments = append(mp.streamResponses[0].fragments, genai.ReplyFragment{TextFragment: i})
					}
					tp := &adapters.ProviderReasoning{
						Provider:            mp,
						ReasoningTokenStart: "<thinking>",
						ReasoningTokenEnd:   "</thinking>",
					}
					accumulated := genai.Message{}
					fragments, finish := tp.GenStream(t.Context(), genai.Messages{}, tc.opts)
					for f := range fragments {
						if err2 := accumulated.Accumulate(f); err2 != nil {
							t.Fatal(err2)
						}
					}
					result, err := finish()
					if err != nil {
						t.Fatalf("unexpected error: %v", err)
					}
					if diff := cmp.Diff(tc.want, accumulated.Replies); diff != "" {
						t.Fatalf("diff:\n%s", diff)
					}
					if diff := cmp.Diff(result.Message, accumulated); diff != "" {
						t.Fatalf("diff:\n%s", diff)
					}
				})
			}
		})
		t.Run("errors", func(t *testing.T) {
			tests := []struct {
				name      string
				in        []string
				fragments []genai.ReplyFragment
				err       error
				want      string
			}{
				{
					name: "With text before tag in same fragment - error case",
					in:   []string{"Text before <thinking>", "This is thinking", "</thinking>", "This is response"},
					want: "unexpected prefix before reasoning tag: \"Text before\"",
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
					fragments: []genai.ReplyFragment{
						{ReasoningFragment: "This is an unexpected thinking fragment"},
					},
					want: `got unexpected reasoning fragment: "This is an unexpected thinking fragment"; do not use ProviderReasoning with an explicit reasoning CoT model`,
				},
			}

			for _, tc := range tests {
				t.Run(tc.name, func(t *testing.T) {
					mp := &mockProviderGenStream{streamResponses: []streamResponse{{}}, err: tc.err}
					if len(tc.fragments) > 0 {
						mp.streamResponses[0].fragments = tc.fragments
					} else {
						for _, i := range tc.in {
							mp.streamResponses[0].fragments = append(mp.streamResponses[0].fragments, genai.ReplyFragment{TextFragment: i})
						}
					}
					tp := &adapters.ProviderReasoning{
						Provider:            mp,
						ReasoningTokenStart: "<thinking>",
						ReasoningTokenEnd:   "</thinking>",
					}
					fragments, finish := tp.GenStream(t.Context(), genai.Messages{})
					for range fragments {
					}
					_, err := finish()
					if err == nil {
						t.Fatal("expected error but got none")
					}
					if got := err.Error(); got != tc.want {
						t.Fatalf("expected %q but got %q", tc.want, got)
					}
				})
			}
		})
	})
}

type mockProviderGenSync struct {
	base.NotImplemented
	responses []genai.Result
	msgs      genai.Messages // Messages from the client
	err       error
}

func (m *mockProviderGenSync) Name() string {
	return "mock"
}

func (m *mockProviderGenSync) ModelID() string {
	return "llm-sota"
}

func (m *mockProviderGenSync) OutputModalities() genai.Modalities {
	return nil
}

func (m *mockProviderGenSync) HTTPClient() *http.Client {
	return nil
}

func (m *mockProviderGenSync) Scoreboard() scoreboard.Score {
	return scoreboard.Score{}
}

func (m *mockProviderGenSync) GenSync(ctx context.Context, msgs genai.Messages, opts ...genai.Options) (genai.Result, error) {
	// Store messages
	m.msgs = msgs
	r := m.responses[0]
	m.responses = m.responses[1:]
	return r, m.err
}

type streamResponse struct {
	fragments []genai.ReplyFragment
	usage     genai.Usage
}

type mockProviderGenStream struct {
	base.NotImplemented
	streamResponses []streamResponse
	msgs            genai.Messages // Messages from the client
	callIndex       int
	err             error
}

func (m *mockProviderGenStream) Name() string {
	return "mock"
}

func (m *mockProviderGenStream) ModelID() string {
	return "llm-sota"
}

func (m *mockProviderGenStream) OutputModalities() genai.Modalities {
	return nil
}

func (m *mockProviderGenStream) HTTPClient() *http.Client {
	return nil
}

func (m *mockProviderGenStream) Scoreboard() scoreboard.Score {
	return scoreboard.Score{}
}

func (m *mockProviderGenStream) GenStream(ctx context.Context, msgs genai.Messages, opts ...genai.Options) (iter.Seq[genai.ReplyFragment], func() (genai.Result, error)) {
	// Store messages
	m.msgs = msgs
	res := genai.Result{}
	var finalErr error
	fnFragments := func(yield func(genai.ReplyFragment) bool) {
		if m.err != nil {
			finalErr = m.err
			return
		}
		if m.callIndex >= len(m.streamResponses) {
			finalErr = fmt.Errorf("no more mock responses")
			return
		}
		resp := m.streamResponses[m.callIndex]
		m.callIndex++
		res.Usage = resp.usage
		for _, fragment := range resp.fragments {
			if err := res.Accumulate(fragment); err != nil {
				finalErr = err
				return
			}
			if !yield(fragment) {
				return
			}
		}
	}
	fnFinish := func() (genai.Result, error) {
		return res, finalErr
	}
	return fnFragments, fnFinish
}
