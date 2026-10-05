// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Tests for Mistral provider DTOs.

package mistral_test

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"github.com/maruel/genai"
	"github.com/maruel/genai/base"
	"github.com/maruel/genai/providers/mistral"
)

func TestChatRequest(t *testing.T) {
	t.Run("Init", func(t *testing.T) {
		a, b := mistral.ToolCall{ID: "A", Type: "function"}, mistral.ToolCall{ID: "B", Type: "function"}
		a.Function.Name, a.Function.Arguments = "status", `{"task":3}`
		b.Function.Name, b.Function.Arguments = "status", `{"task":7}`
		ca := genai.Reply{ToolCall: genai.ToolCall{ID: "A", Name: "status", Arguments: `{"task":3}`}}
		cb := genai.Reply{ToolCall: genai.ToolCall{ID: "B", Name: "status", Arguments: `{"task":7}`}}
		t.Run("valid/content_before_calls", func(t *testing.T) {
			msgs := genai.Messages{
				genai.NewTextMessage("user"),
				{Replies: []genai.Reply{{Text: "before"}, ca, cb}},
				{ToolCallResults: []genai.ToolCallResult{{ID: "B", Name: "status", Result: "waiting"}, {ID: "A", Name: "status", Result: "running"}}},
			}
			var req mistral.ChatRequest
			if err := req.Init(msgs, "test"); err != nil {
				t.Fatal(err)
			}
			want := []mistral.Message{
				{Role: "user", Content: []mistral.Content{{Type: mistral.ContentText, Text: "user"}}},
				{Role: "assistant", Content: []mistral.Content{{Type: mistral.ContentText, Text: "before"}}, ToolCalls: []mistral.ToolCall{a, b}},
				{Role: "tool", ToolCallID: "B", Name: "status", Content: []mistral.Content{{Type: mistral.ContentText, Text: "waiting"}}},
				{Role: "tool", ToolCallID: "A", Name: "status", Content: []mistral.Content{{Type: mistral.ContentText, Text: "running"}}},
			}
			if diff := cmp.Diff(want, req.Messages); diff != "" {
				t.Fatal(diff)
			}
		})
		t.Run("error/content_after_calls", func(t *testing.T) {
			for _, tc := range []struct {
				name    string
				replies []genai.Reply
			}{
				{"interleaved", []genai.Reply{{Text: "before"}, ca, {Text: "after"}, cb}},
				{"call_then_content", []genai.Reply{ca, {Text: "after"}}},
				{"content_after_calls", []genai.Reply{{Text: "before"}, ca, cb, {Text: "after"}}},
			} {
				t.Run(tc.name, func(t *testing.T) {
					msgs := genai.Messages{genai.NewTextMessage("user"), {Replies: tc.replies}}
					before, err := json.Marshal(msgs)
					if err != nil {
						t.Fatal(err)
					}
					var req mistral.ChatRequest
					if err := req.Init(msgs, "test"); err == nil || !strings.Contains(err.Error(), "mistral cannot represent content after a tool call") {
						t.Fatalf("error=%v", err)
					}
					after, err := json.Marshal(msgs)
					if err != nil {
						t.Fatal(err)
					}
					if !bytes.Equal(before, after) {
						t.Fatal("mutated input on rejection")
					}
				})
			}
		})
	})
}

func TestUsageDurationS(t *testing.T) {
	const input = `{"prompt_audio_seconds":2.5}`
	var got mistral.Usage
	if err := json.Unmarshal([]byte(input), &got); err != nil {
		t.Fatal(err)
	}
	if got.PromptAudio != base.DurationS(2.5) {
		t.Errorf("PromptAudio = %v, want 2.5", got.PromptAudio)
	}
	if got.PromptAudio.AsDuration() != 2*time.Second+500*time.Millisecond {
		t.Errorf("PromptAudio.AsDuration() = %v", got.PromptAudio.AsDuration())
	}
}
