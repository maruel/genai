// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Tests for HuggingFace provider DTOs.

package huggingface_test

import (
	"encoding/json"
	"testing"
	"time"

	"github.com/maruel/genai/base"
	"github.com/maruel/genai/providers/huggingface"
)

func TestChatResponseToResult(t *testing.T) {
	const input = `{
  "choices": [{
    "finish_reason": "length",
    "message": {
      "content": null,
      "name": null,
      "role": "assistant",
      "reasoning_content": "Need answer."
    }
  }],
  "usage": {
    "prompt_tokens": 40,
    "completion_tokens": 16,
    "total_tokens": 56,
    "completion_tokens_details": {"reasoning_tokens": 16},
    "estimated_cost": 4.4e-7
  },
  "service_tier": "default"
}`
	var c huggingface.ChatResponse
	if err := json.Unmarshal([]byte(input), &c); err != nil {
		t.Fatal(err)
	}
	got, err := c.ToResult()
	if err != nil {
		t.Fatal(err)
	}
	if len(got.Replies) != 1 || got.Replies[0].Reasoning != "Need answer." || got.Replies[0].Text != "" {
		t.Errorf("Replies = %#v, want one reasoning reply", got.Replies)
	}
	if got.Usage.ReasoningTokens != 16 {
		t.Errorf("ReasoningTokens = %d, want 16", got.Usage.ReasoningTokens)
	}
}

func TestChatStreamChunkResponseDurationMS(t *testing.T) {
	const input = `{"service_tier":null,"choices":[{"token_ids":null}],"sla_metrics":{"ts_us":1,"ttft_ms":12.5}}`
	var got huggingface.ChatStreamChunkResponse
	if err := json.Unmarshal([]byte(input), &got); err != nil {
		t.Fatal(err)
	}
	if got.SLAMetrics.TTFT != base.DurationMS(12.5) {
		t.Errorf("TTFT = %v, want 12.5", got.SLAMetrics.TTFT)
	}
	if got.SLAMetrics.TTFT.AsDuration() != 12*time.Millisecond+500*time.Microsecond {
		t.Errorf("TTFT.AsDuration() = %v", got.SLAMetrics.TTFT.AsDuration())
	}
}
