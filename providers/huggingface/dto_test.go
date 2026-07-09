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

func TestChatStreamChunkResponseDurationMS(t *testing.T) {
	const input = `{"choices":[],"sla_metrics":{"ts_us":1,"ttft_ms":12.5}}`
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
