// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Tests for Cohere thinking request conversion.

package cohere_test

import (
	"encoding/json"
	"errors"
	"slices"
	"testing"

	"github.com/maruel/genai"
	"github.com/maruel/genai/base"
	"github.com/maruel/genai/providers/cohere"
)

func TestChatRequest(t *testing.T) {
	t.Run("thinking", func(t *testing.T) {
		for _, tc := range []struct {
			name      string
			model     string
			level     genai.EffortLevel
			maxTokens int64
			want      cohere.Thinking
		}{
			{"off", "command-a-reasoning-08-2025", genai.EffortOff, 0, cohere.Thinking{Type: cohere.ThinkingDisabled}},
			{"medium", "command-a-reasoning-08-2025", genai.EffortMedium, 0, cohere.Thinking{Type: cohere.ThinkingEnabled, TokenBudget: 8192}},
			{"high", "command-a-reasoning-08-2025", genai.EffortHigh, 0, cohere.Thinking{Type: cohere.ThinkingEnabled, TokenBudget: 16384}},
			{"max", "command-a-reasoning-08-2025", genai.EffortMax, 0, cohere.Thinking{Type: cohere.ThinkingEnabled}},
			{"nonreasoning", "command-a-03-2025", genai.EffortOff, 0, cohere.Thinking{}},
			{"plus", "command-a-plus-05-2026", genai.EffortXHigh, 0, cohere.Thinking{Type: cohere.ThinkingEnabled, TokenBudget: 32768}},
		} {
			t.Run(tc.name, func(t *testing.T) {
				var r cohere.ChatRequest
				if err := r.Init(genai.Messages{genai.NewTextMessage("hello")}, tc.model, tc.level, &genai.GenOptionText{MaxTokens: tc.maxTokens}); err != nil {
					t.Fatal(err)
				}
				if r.Thinking != tc.want {
					t.Errorf("thinking = %#v, want %#v", r.Thinking, tc.want)
				}
			})
		}
	})
	t.Run("error", func(t *testing.T) {
		t.Run("unsupported_model", func(t *testing.T) {
			var r cohere.ChatRequest
			err := r.Init(genai.Messages{genai.NewTextMessage("hello")}, "command-a-03-2025", genai.EffortHigh)
			if _, ok := errors.AsType[*base.ErrNotSupported](err); !ok {
				t.Fatalf("got %v, want unsupported thinking", err)
			}
		})
		t.Run("no_response_room", func(t *testing.T) {
			var r cohere.ChatRequest
			if err := r.Init(genai.Messages{genai.NewTextMessage("hello")}, "command-a-reasoning-08-2025", &genai.GenOptionText{MaxTokens: 1024}, genai.EffortLow); err == nil {
				t.Fatal("expected budget error")
			}
		})
	})
}

func TestMessage(t *testing.T) {
	t.Run("native_reasoning", func(t *testing.T) {
		src := cohere.MessageResponse{Role: "assistant", ToolPlan: "plan", Content: []cohere.Content{
			{Type: cohere.ContentThinking, Thinking: "check"},
			{Type: cohere.ContentText, Text: "answer"},
		}}
		var in genai.Message
		if err := src.To(&in); err != nil {
			t.Fatal(err)
		}
		var got cohere.Message
		if _, err := got.From(&in); err != nil {
			t.Fatal(err)
		}
		if got.ToolPlan != "plan" || len(got.Content) != 2 || got.Content[0].Type != cohere.ContentThinking || got.Content[0].Thinking != "check" || got.Content[1].Text != "answer" {
			t.Fatalf("lost native reasoning: %+v", got)
		}
	})
}

func TestProcessStream(t *testing.T) {
	chunks := make([]cohere.ChatStreamChunkResponse, 0, 6)
	for _, body := range []string{
		`{"type":"content-start","delta":{"message":{"content":{"type":"thinking","thinking":"first"}}}}`,
		`{"type":"content-delta","delta":{"message":{"content":{"thinking":" second"}}}}`,
		`{"type":"content-end"}`,
		`{"type":"tool-plan-delta","delta":{"message":{"tool_plan":"plan"}}}`,
		`{"type":"content-start","delta":{"message":{"content":{"type":"text","text":"answer"}}}}`,
		`{"type":"content-end"}`,
	} {
		var c cohere.ChatStreamChunkResponse
		if err := json.Unmarshal([]byte(body), &c); err != nil {
			t.Fatal(err)
		}
		chunks = append(chunks, c)
	}
	seq, finish := cohere.ProcessStream(slices.Values(chunks))
	var msg genai.Message
	for f := range seq {
		if err := msg.Accumulate(&f); err != nil {
			t.Fatal(err)
		}
	}
	if _, _, err := finish(); err != nil {
		t.Fatal(err)
	}
	var got cohere.Message
	if _, err := got.From(&msg); err != nil {
		t.Fatal(err)
	}
	if got.ToolPlan != "plan" || len(got.Content) != 2 || got.Content[0].Thinking != "first second" || got.Content[1].Text != "answer" {
		t.Fatalf("lost native reasoning: %+v", got)
	}
}
