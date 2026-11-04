// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Tests for the OpenRouter wire types.

package openrouter_test

import (
	"encoding/json"
	"slices"
	"testing"

	"github.com/maruel/genai"
	"github.com/maruel/genai/providers/openrouter"
)

func TestModel(t *testing.T) {
	t.Run("String", func(t *testing.T) {
		m := openrouter.Model{
			ID:            "openai/gpt-test",
			Name:          "GPT Test",
			Created:       1735689600,
			ContextLength: 1048576,
			Pricing: openrouter.ModelPricing{
				Prompt:     "0.0000005",
				Completion: "0.000002",
			},
		}
		m.Architecture.Modality = "text+image->text"
		m.TopProvider.MaxCompletionTokens = 32768
		want := "openai/gpt-test (2025-01-01): GPT Test (text+image->text) Context: 1048576/32768; in: 0.50$/Mt out: 2.00$/Mt"
		if got := m.String(); got != want {
			t.Fatalf("String() = %q, want %q", got, want)
		}
	})
	t.Run("String sparse", func(t *testing.T) {
		m := openrouter.Model{ID: "test/model", Name: "Test Model", ContextLength: 4096}
		want := "test/model: Test Model Context: 4096"
		if got := m.String(); got != want {
			t.Fatalf("String() = %q, want %q", got, want)
		}
	})
	t.Run("String malformed pricing", func(t *testing.T) {
		m := openrouter.Model{
			ID:            "test/model",
			Name:          "Test Model",
			ContextLength: 4096,
			Pricing: openrouter.ModelPricing{
				Prompt:     "malformed",
				Completion: "0.000001",
			},
		}
		want := "test/model: Test Model Context: 4096; in: malformed$/t out: 1.00$/Mt"
		if got := m.String(); got != want {
			t.Fatalf("String() = %q, want %q", got, want)
		}
	})
}

func TestProcessStream(t *testing.T) {
	chunks := make([]openrouter.ChatStreamChunkResponse, 0, 3)
	for _, body := range []string{
		`{"choices":[{"delta":{"reasoning":"first ","reasoning_details":[{"type":"reasoning.text","index":0,"text":"first "}]}}]}`,
		`{"choices":[{"delta":{"reasoning":"second","reasoning_details":[{"type":"reasoning.text","index":0,"text":"second","signature":"signed"}]}}]}`,
		`{"choices":[{"delta":{"content":"answer"}}]}`,
	} {
		var c openrouter.ChatStreamChunkResponse
		if err := json.Unmarshal([]byte(body), &c); err != nil {
			t.Fatal(err)
		}
		chunks = append(chunks, c)
	}
	seq, finish := openrouter.ProcessStream(slices.Values(chunks))
	var msg genai.Message
	for f := range seq {
		if err := msg.Accumulate(&f); err != nil {
			t.Fatal(err)
		}
	}
	if _, _, err := finish(); err != nil {
		t.Fatal(err)
	}
	b, err := json.Marshal(msg)
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(b, &msg); err != nil {
		t.Fatal(err)
	}
	var got openrouter.Message
	if err := got.From(&msg); err != nil {
		t.Fatal(err)
	}
	if got.Reasoning == nil || *got.Reasoning != "first second" || len(got.ReasoningDetails) != 1 || got.ReasoningDetails[0].Text != "first second" || got.ReasoningDetails[0].Signature == nil || *got.ReasoningDetails[0].Signature != "signed" {
		t.Fatalf("lost streamed reasoning: %+v", got)
	}
}
