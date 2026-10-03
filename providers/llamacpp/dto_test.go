// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Tests for llama.cpp provider DTOs.

package llamacpp_test

import (
	"encoding/json"
	"strings"
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

func TestGenerationSettings(t *testing.T) {
	// Fields emitted by task_params::to_json in llama.cpp v0.5.0.
	input := `{"adaptive_target":0.8,"adaptive_decay":0.9,"generation_prompt":"<assistant>","backend_sampling":true,"speculative.types":"draft-simple","grammar_triggers":[]}`
	var got llamacpp.GenerationSettings
	d := json.NewDecoder(strings.NewReader(input))
	d.DisallowUnknownFields()
	if err := d.Decode(&got); err != nil {
		t.Fatal(err)
	}
	if got.AdaptiveTarget != 0.8 || got.AdaptiveDecay != 0.9 || got.GenerationPrompt != "<assistant>" || !got.BackendSampling || got.SpeculativeTypes != "draft-simple" {
		t.Fatalf("unexpected generation settings: %+v", got)
	}
}

func TestChatStreamChunkResponse(t *testing.T) {
	const input = `{"prompt_progress":{"total":100,"cache":20,"processed":30,"time_ms":12.5}}`
	var got llamacpp.ChatStreamChunkResponse
	d := json.NewDecoder(strings.NewReader(input))
	d.DisallowUnknownFields()
	if err := d.Decode(&got); err != nil {
		t.Fatal(err)
	}
	if got.PromptProgress.Total != 100 || got.PromptProgress.Cache != 20 || got.PromptProgress.Processed != 30 || got.PromptProgress.Time != base.DurationMS(12.5) {
		t.Fatalf("unexpected prompt progress: %+v", got.PromptProgress)
	}
}
