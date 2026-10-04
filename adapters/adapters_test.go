// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Tests for the adapters package.

package adapters_test

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"slices"
	"strconv"
	"testing"

	"github.com/google/go-cmp/cmp"

	"github.com/maruel/genai"
	"github.com/maruel/genai/adapters"
	"github.com/maruel/genai/providers/llamacpp"
)

func TestGenSyncWithToolCallLoop(t *testing.T) {
	provider := &mockProviderGenSync{
		responses: []genai.Result{
			{
				Message: genai.Message{
					Replies: []genai.Reply{
						{Text: "I'll help you calculate that. "},
						{Text: "Let me use the calculator tool."},
						{ToolCall: genai.ToolCall{ID: "1", Name: "calculator", Arguments: `{"a": 5, "b": 3, "operation": "add"}`}},
					},
				},
				Usage: genai.Usage{InputTokens: 10, OutputTokens: 20},
			},
			{
				Message: genai.Message{
					Replies: []genai.Reply{
						{Text: "The result of 5 + 3 is 8."},
					},
				},
				Usage: genai.Usage{InputTokens: 15, OutputTokens: 10},
			},
		},
	}
	msgs := genai.Messages{genai.NewTextMessage("Calculate 5 + 3")}
	type CalculatorArgs struct {
		A         int    `json:"a"`
		B         int    `json:"b"`
		Operation string `json:"operation"`
	}
	opts := &genai.GenOptionTools{
		Tools: []genai.ToolDef{
			{
				Name:        "calculator",
				Description: "A simple calculator",
				Callback: func(ctx context.Context, args *CalculatorArgs) (string, error) {
					switch args.Operation {
					case "add":
						return strconv.Itoa(args.A + args.B), nil
					default:
						return "", fmt.Errorf("unsupported operation: %s", args.Operation)
					}
				},
			},
		},
	}
	respMsgs, usage, err := adapters.GenSyncWithToolCallLoop(t.Context(), provider, msgs, opts)
	if err != nil {
		t.Fatalf("GenSyncWithToolCallLoop returned an error: %v", err)
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
}

func TestGenStreamWithToolCallLoop(t *testing.T) {
	provider := &mockProviderGenStream{
		streamResponses: []streamResponse{
			{
				fragments: []genai.Reply{
					{Text: "I'll help you calculate that. "},
					{Text: "Let me use the calculator tool."},
					{ToolCall: genai.ToolCall{ID: "1", Name: "calculator", Arguments: `{"a": 5, "b": 3, "operation": "add"}`}},
				},
				usage: genai.Usage{InputTokens: 10, OutputTokens: 20},
			},
			{
				fragments: []genai.Reply{
					{Text: "The result of 5 + 3 is 8."},
				},
				usage: genai.Usage{InputTokens: 15, OutputTokens: 10},
			},
		},
	}
	msgs := genai.Messages{genai.NewTextMessage("Calculate 5 + 3")}
	type CalculatorArgs struct {
		A         int    `json:"a"`
		B         int    `json:"b"`
		Operation string `json:"operation"`
	}
	opts := &genai.GenOptionTools{
		Tools: []genai.ToolDef{
			{
				Name:        "calculator",
				Description: "A simple calculator",
				Callback: func(ctx context.Context, args *CalculatorArgs) (string, error) {
					switch args.Operation {
					case "add":
						return strconv.Itoa(args.A + args.B), nil
					default:
						return "", fmt.Errorf("unsupported operation: %s", args.Operation)
					}
				},
			},
		},
	}
	fragments, finish := adapters.GenStreamWithToolCallLoop(t.Context(), provider, msgs, opts)
	got := slices.Collect(fragments)
	respMsgs, usage, err := finish()
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
	if len(got) != 4 { // 2 text fragments + 1 tool call + 1 text fragment
		t.Fatalf("Expected 4 fragments, got %d", len(got))
	}
}

func TestProviderUsage(t *testing.T) {
	t.Run("SystemOne", func(t *testing.T) {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			if _, err := w.Write([]byte(`{"model":"kev","answers":{"yes":{"type":"noul","noul":0.8}},"usage":{"input_tokens":42,"output_tokens":3,"reasoning_tokens":2}}`)); err != nil {
				t.Error(err)
			}
		}))
		t.Cleanup(srv.Close)
		c, err := llamacpp.New(t.Context(), genai.ProviderOptionRemote(srv.URL))
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() {
			if err := c.Close(); err != nil {
				t.Error(err)
			}
		})
		p := adapters.ProviderUsage{Provider: c}
		for range 2 {
			if _, err := p.SystemOne(t.Context(), &genai.SystemOneRequest{State: genai.Text("state"), Questions: genai.Questions{"yes": {Type: genai.QuestionNoul, Instructions: genai.Text("yes?")}}}); err != nil {
				t.Fatal(err)
			}
		}
		if got := p.GetAccumulatedUsage(); got.InputTokens != 84 || got.OutputTokens != 6 || got.ReasoningTokens != 4 {
			t.Fatalf("unexpected usage: %+v", got)
		}
	})
	t.Run("GenSync", func(t *testing.T) {
		provider := &mockProviderGenSync{
			responses: []genai.Result{
				{Usage: genai.Usage{InputTokens: 10, OutputTokens: 20}},
				{Usage: genai.Usage{InputTokens: 15, OutputTokens: 25}},
			},
		}
		wrapped := &adapters.ProviderUsage{Provider: provider}
		_, _ = wrapped.GenSync(t.Context(), nil)
		_, _ = wrapped.GenSync(t.Context(), nil)
		expected := genai.Usage{InputTokens: 25, OutputTokens: 45}
		if diff := cmp.Diff(expected, wrapped.GetAccumulatedUsage()); diff != "" {
			t.Fatalf("unexpected usage: %s", diff)
		}
	})
	t.Run("GenStream", func(t *testing.T) {
		provider := &mockProviderGenStream{
			streamResponses: []streamResponse{
				{usage: genai.Usage{InputTokens: 10, OutputTokens: 20}},
				{usage: genai.Usage{InputTokens: 15, OutputTokens: 25}},
			},
		}
		wrapped := &adapters.ProviderUsage{Provider: provider}
		fragments, finish := wrapped.GenStream(t.Context(), nil)
		for range fragments {
		}
		_, _ = finish()
		fragments, finish = wrapped.GenStream(t.Context(), nil)
		for range fragments {
		}
		_, _ = finish()
		expected := genai.Usage{InputTokens: 25, OutputTokens: 45}
		if diff := cmp.Diff(expected, wrapped.GetAccumulatedUsage()); diff != "" {
			t.Fatalf("unexpected usage: %s", diff)
		}
	})
	t.Run("Unwrap", func(t *testing.T) {
		provider := &mockProviderGenSync{}
		wrapped := &adapters.ProviderUsage{Provider: provider}
		if wrapped.Unwrap() != provider {
			t.Fatal("expected unwrapped provider to be the original provider")
		}
	})
}

func TestProviderAppend(t *testing.T) {
	t.Run("GenSync", func(t *testing.T) {
		provider := &mockProviderGenSync{
			responses: []genai.Result{{}},
		}
		wrapped := &adapters.ProviderAppend{
			Provider: provider,
			Append:   genai.Request{Text: "appended"},
		}
		msgs := genai.Messages{{Requests: []genai.Request{{Text: "original"}}}}
		_, _ = wrapped.GenSync(t.Context(), msgs)
		expected := genai.Messages{{Requests: []genai.Request{{Text: "original"}, {Text: "appended"}}}}
		if diff := cmp.Diff(expected, provider.msgs); diff != "" {
			t.Fatalf("unexpected messages: %s", diff)
		}
	})
	t.Run("GenStream", func(t *testing.T) {
		provider := &mockProviderGenStream{
			streamResponses: []streamResponse{{}},
		}
		wrapped := &adapters.ProviderAppend{
			Provider: provider,
			Append:   genai.Request{Text: "appended"},
		}
		msgs := genai.Messages{{Requests: []genai.Request{{Text: "original"}}}}
		_, _ = wrapped.GenStream(t.Context(), msgs)
		expected := genai.Messages{{Requests: []genai.Request{{Text: "original"}, {Text: "appended"}}}}
		if diff := cmp.Diff(expected, provider.msgs); diff != "" {
			t.Fatalf("unexpected messages: %s", diff)
		}
	})
	t.Run("Unwrap", func(t *testing.T) {
		provider := &mockProviderGenSync{}
		wrapped := &adapters.ProviderAppend{Provider: provider}
		if wrapped.Unwrap() != provider {
			t.Fatal("expected unwrapped provider to be the original provider")
		}
	})
}
