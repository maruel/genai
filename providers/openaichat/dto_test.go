// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Tests for OpenAI Chat provider DTOs.

package openaichat

import (
	"testing"

	"github.com/maruel/genai"
)

func TestChatRequest(t *testing.T) {
	t.Run("Init", func(t *testing.T) {
		t.Run("web search", func(t *testing.T) {
			msgs := genai.Messages{genai.NewTextMessage("search")}
			t.Run("legacy model", func(t *testing.T) {
				var r ChatRequest
				if err := r.Init(msgs, "gpt-4o-search-preview", &genai.GenOptionWeb{Search: true}); err != nil {
					t.Fatal(err)
				}
				if r.WebSearchOptions == nil || r.WebSearchOptions.SearchContextSize != "high" {
					t.Fatalf("WebSearchOptions = %#v, want high context", r.WebSearchOptions)
				}
			})
		})
		t.Run("tools", func(t *testing.T) {
			t.Run("valid", func(t *testing.T) {
				t.Run("snapshot disables reasoning by default", func(t *testing.T) {
					opts := &genai.GenOptionTools{
						Tools: []genai.ToolDef{
							{
								Name:                "square_root",
								Description:         "Calculates and returns the square root of a number",
								InputSchemaOverride: genai.JSONSchema(`{"type":"object","properties":{"number":{"type":"number"}},"required":["number"]}`),
							},
						},
						Force: genai.ToolCallRequired,
					}
					var r ChatRequest
					if err := r.Init(genai.Messages{genai.NewTextMessage("calculate")}, "gpt-6-luna-2026-10-01", opts); err != nil {
						t.Fatal(err)
					}
					if r.ReasoningEffort != ReasoningEffortNone {
						t.Fatalf("ReasoningEffort = %q, want none", r.ReasoningEffort)
					}
				})
			})
		})
	})
}
