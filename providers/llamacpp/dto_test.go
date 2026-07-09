// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Tests for llama.cpp provider DTOs.

package llamacpp_test

import (
	"encoding/json"
	"testing"
	"time"

	"github.com/maruel/genai/base"
	"github.com/maruel/genai/providers/llamacpp"
)

func TestTimingsDurationMS(t *testing.T) {
	const input = `{"prompt_ms":12.5,"prompt_per_token_ms":1.25,"predicted_ms":34.75,"predicted_per_token_ms":2.5}`
	var got llamacpp.Timings
	if err := json.Unmarshal([]byte(input), &got); err != nil {
		t.Fatal(err)
	}
	if got.Prompt != base.DurationMS(12.5) {
		t.Errorf("Prompt = %v, want 12.5", got.Prompt)
	}
	if got.Prompt.AsDuration() != 12*time.Millisecond+500*time.Microsecond {
		t.Errorf("Prompt.AsDuration() = %v", got.Prompt.AsDuration())
	}
	if got.PredictedPerToken != base.DurationMS(2.5) {
		t.Errorf("PredictedPerTokenMS = %v, want 2.5", got.PredictedPerToken)
	}
}
