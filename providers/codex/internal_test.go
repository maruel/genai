// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Internal tests for the Codex provider.

package codex

import (
	"encoding/json"
	"errors"
	"slices"
	"testing"
	"time"

	"github.com/maruel/genai"
	"github.com/maruel/genai/base"
	"github.com/maruel/genai/internal"
)

func TestReasoningEffort(t *testing.T) {
	t.Run("valid", func(t *testing.T) {
		for _, v := range []ReasoningEffort{
			ReasoningEffortNone, ReasoningEffortMinimal, ReasoningEffortLow,
			ReasoningEffortMedium, ReasoningEffortHigh, ReasoningEffortXHigh,
		} {
			c, err := New(v)
			if err != nil {
				t.Fatalf("New(%q): %v", v, err)
			}
			if c.effort != v {
				t.Errorf("effort: got %q, want %q", c.effort, v)
			}
		}
	})
	t.Run("invalid", func(t *testing.T) {
		if _, err := New(ReasoningEffort("turbo")); err == nil {
			t.Fatal("expected error for invalid effort")
		}
	})
	t.Run("default", func(t *testing.T) {
		c, err := New()
		if err != nil {
			t.Fatal(err)
		}
		if c.effort != ReasoningEffortMedium {
			t.Errorf("default effort: got %q, want %q", c.effort, ReasoningEffortMedium)
		}
	})
}

func TestParseOpts(t *testing.T) {
	t.Run("system_prompt", func(t *testing.T) {
		co, err := parseOpts([]genai.GenOption{&genai.GenOptionText{SystemPrompt: "Be helpful"}})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if co.systemPrompt != "Be helpful" {
			t.Errorf("systemPrompt: got %q, want %q", co.systemPrompt, "Be helpful")
		}
	})
	t.Run("unsupported", func(t *testing.T) {
		for _, tc := range []struct {
			name string
			opts []genai.GenOption
			want string
		}{
			{"Temperature", []genai.GenOption{&genai.GenOptionText{Temperature: 0.5}}, "GenOptionText.Temperature"},
			{"Seed", []genai.GenOption{genai.GenOptionSeed(42)}, "GenOptionSeed"},
		} {
			t.Run(tc.name, func(t *testing.T) {
				_, err := parseOpts(tc.opts)
				uerr, ok := errors.AsType[*base.ErrNotSupported](err)
				if !ok {
					t.Fatalf("expected ErrNotSupported, got %v", err)
				}
				if !slices.Contains(uerr.Options, tc.want) {
					t.Errorf("expected %q in unsupported, got %v", tc.want, uerr.Options)
				}
			})
		}
	})
}

func TestNotificationTimeMS(t *testing.T) {
	t.Run("item_started", func(t *testing.T) {
		const input = `{"item":{"id":"u1","type":"userMessage"},"threadId":"t1","turnId":"turn_1","startedAtMs":1780832660165}`
		var got ItemStartedNotification
		if err := json.Unmarshal([]byte(input), &got); err != nil {
			t.Fatal(err)
		}
		if got.StartedAt != base.TimeMS(1780832660165) {
			t.Errorf("StartedAt = %v, want 1780832660165", got.StartedAt)
		}
		if got.StartedAt.AsTime() != time.Date(2026, 6, 7, 11, 44, 20, 165000000, time.UTC) {
			t.Errorf("StartedAt.AsTime() = %v", got.StartedAt.AsTime())
		}
	})
	t.Run("guardian_review_completed", func(t *testing.T) {
		const input = `{"threadId":"t1","turnId":"turn_1","startedAtMs":1780832660165,"completedAtMs":1780832661123,"reviewId":"r1","targetItemId":null,"decisionSource":"agent_decision","review":{"status":"approved"},"action":{"type":"run_command"}}`
		var got ItemGuardianApprovalReviewCompletedNotification
		if err := json.Unmarshal([]byte(input), &got); err != nil {
			t.Fatal(err)
		}
		if got.StartedAt != base.TimeMS(1780832660165) {
			t.Errorf("StartedAt = %v, want 1780832660165", got.StartedAt)
		}
		if got.CompletedAt != base.TimeMS(1780832661123) {
			t.Errorf("CompletedAt = %v, want 1780832661123", got.CompletedAt)
		}
	})
}

func TestDurationMS(t *testing.T) {
	t.Run("turn", func(t *testing.T) {
		const input = `{"id":"turn_1","status":"completed","startedAt":1780832660.165,"completedAt":1780832661.25,"durationMs":123.5}`
		var got Turn
		if err := json.Unmarshal([]byte(input), &got); err != nil {
			t.Fatal(err)
		}
		if got.StartedAt != base.TimeS(1780832660.165) {
			t.Errorf("StartedAt = %v, want 1780832660.165", got.StartedAt)
		}
		if got.CompletedAt != base.TimeS(1780832661.25) {
			t.Errorf("CompletedAt = %v, want 1780832661.25", got.CompletedAt)
		}
		if got.Duration == nil {
			t.Fatal("Duration = nil")
		}
		if *got.Duration != base.DurationMS(123.5) {
			t.Errorf("Duration = %v, want 123.5", *got.Duration)
		}
		if got.Duration.AsDuration() != 123*time.Millisecond+500*time.Microsecond {
			t.Errorf("Duration.AsDuration() = %v", got.Duration.AsDuration())
		}
	})
	t.Run("command_execution", func(t *testing.T) {
		const input = `{"id":"cmd_1","type":"commandExecution","durationMs":12.25}`
		var got CommandExecutionItem
		if err := json.Unmarshal([]byte(input), &got); err != nil {
			t.Fatal(err)
		}
		if got.Duration == nil {
			t.Fatal("Duration = nil")
		}
		if *got.Duration != base.DurationMS(12.25) {
			t.Errorf("Duration = %v, want 12.25", *got.Duration)
		}
		if got.Duration.AsDuration() != 12*time.Millisecond+250*time.Microsecond {
			t.Errorf("Duration.AsDuration() = %v", got.Duration.AsDuration())
		}
	})
	t.Run("dynamic_tool_call", func(t *testing.T) {
		const input = `{"id":"dyn_1","type":"dynamicToolCall","durationMs":7.75}`
		var got DynamicToolCallItem
		if err := json.Unmarshal([]byte(input), &got); err != nil {
			t.Fatal(err)
		}
		if got.Duration != base.DurationMS(7.75) {
			t.Errorf("Duration = %v, want 7.75", got.Duration)
		}
		if got.Duration.AsDuration() != 7*time.Millisecond+750*time.Microsecond {
			t.Errorf("Duration.AsDuration() = %v", got.Duration.AsDuration())
		}
	})
}

func TestDurationS(t *testing.T) {
	const input = `{"threadId":"t1","objective":"ship","status":"active","tokensUsed":12,"timeUsedSeconds":3.25,"createdAt":1,"updatedAt":2}`
	var got ThreadGoal
	if err := json.Unmarshal([]byte(input), &got); err != nil {
		t.Fatal(err)
	}
	if got.TimeUsed != base.DurationS(3.25) {
		t.Errorf("TimeUsed = %v, want 3.25", got.TimeUsed)
	}
	if got.TimeUsed.AsDuration() != 3*time.Second+250*time.Millisecond {
		t.Errorf("TimeUsed.AsDuration() = %v", got.TimeUsed.AsDuration())
	}
	if got.CreatedAt != base.TimeS(1) {
		t.Errorf("CreatedAt = %v, want 1", got.CreatedAt)
	}
	if got.UpdatedAt != base.TimeS(2) {
		t.Errorf("UpdatedAt = %v, want 2", got.UpdatedAt)
	}
}

func TestTimeS(t *testing.T) {
	t.Run("thread", func(t *testing.T) {
		const input = `{"id":"t1","createdAt":1780832660.165,"updatedAt":1780832661.25}`
		var got Thread
		if err := json.Unmarshal([]byte(input), &got); err != nil {
			t.Fatal(err)
		}
		if got.CreatedAt != base.TimeS(1780832660.165) {
			t.Errorf("CreatedAt = %v, want 1780832660.165", got.CreatedAt)
		}
		if got.UpdatedAt != base.TimeS(1780832661.25) {
			t.Errorf("UpdatedAt = %v, want 1780832661.25", got.UpdatedAt)
		}
		if got.CreatedAt.AsTime() != time.Date(2026, 6, 7, 11, 44, 20, 165000000, time.UTC) {
			t.Errorf("CreatedAt.AsTime() = %v", got.CreatedAt.AsTime())
		}
	})
	t.Run("rate_limit_window", func(t *testing.T) {
		const input = `{"usedPercent":50,"resetsAt":1780832660.165}`
		var got RateLimitWindow
		if err := json.Unmarshal([]byte(input), &got); err != nil {
			t.Fatal(err)
		}
		if got.ResetsAt != base.TimeS(1780832660.165) {
			t.Errorf("ResetsAt = %v, want 1780832660.165", got.ResetsAt)
		}
	})
	t.Run("spend_control_limit_snapshot", func(t *testing.T) {
		got := SpendControlLimitSnapshot{
			Limit:            "100",
			Used:             "50",
			RemainingPercent: 50,
			ResetsAt:         base.TimeS(1780832660.165),
		}
		b, err := json.Marshal(got)
		if err != nil {
			t.Fatal(err)
		}
		var fields map[string]json.RawMessage
		if err := json.Unmarshal(b, &fields); err != nil {
			t.Fatal(err)
		}
		if _, ok := fields["resetsAt"]; !ok {
			t.Errorf("marshaled fields = %s, want resetsAt", b)
		}
		if _, ok := fields["ResetsAt"]; ok {
			t.Errorf("marshaled fields = %s, did not want ResetsAt", b)
		}
	})
}

func TestContextCompactionThreadItem(t *testing.T) {
	t.Run("valid", func(t *testing.T) {
		const input = `{"id":"cc1","type":"contextCompaction"}`
		var item ContextCompactionThreadItem
		if err := json.Unmarshal([]byte(input), &item); err != nil {
			t.Fatal(err)
		}
		if item.ID != "cc1" {
			t.Errorf("ID = %q, want cc1", item.ID)
		}
		if item.Type != ItemTypeContextCompaction {
			t.Errorf("Type = %q, want %q", item.Type, ItemTypeContextCompaction)
		}
	})
}

func TestUserMessageItem(t *testing.T) {
	t.Run("valid", func(t *testing.T) {
		const input = `{"id":"u1","type":"userMessage","clientId":null,"content":[{"type":"text","text":"hello","text_elements":[]}]}`
		var item UserMessageItem
		if err := json.Unmarshal([]byte(input), &item); err != nil {
			t.Fatal(err)
		}
		if item.ID != "u1" {
			t.Errorf("ID = %q, want u1", item.ID)
		}
		if item.Type != ItemTypeUserMessage {
			t.Errorf("Type = %q, want %q", item.Type, ItemTypeUserMessage)
		}
		if len(item.Content) != 1 {
			t.Fatalf("len(Content) = %d, want 1", len(item.Content))
		}
		if item.Content[0].Type != TurnInputTypeText {
			t.Errorf("Content[0].Type = %q, want %q", item.Content[0].Type, TurnInputTypeText)
		}
		if item.Content[0].Text != "hello" {
			t.Errorf("Content[0].Text = %q, want hello", item.Content[0].Text)
		}
		if len(item.Content[0].TextElements) != 0 {
			t.Errorf("len(Content[0].TextElements) = %d, want 0", len(item.Content[0].TextElements))
		}
	})
}

func init() {
	internal.BeLenient = false
}
