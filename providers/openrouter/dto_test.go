// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Tests for the OpenRouter wire types.

package openrouter_test

import (
	"testing"

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
