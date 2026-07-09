// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Tests for Mistral provider DTOs.

package mistral_test

import (
	"encoding/json"
	"testing"
	"time"

	"github.com/maruel/genai/base"
	"github.com/maruel/genai/providers/mistral"
)

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
