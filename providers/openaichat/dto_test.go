// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package openaichat

import (
	"errors"
	"testing"

	"github.com/maruel/genai"
	"github.com/maruel/genai/base"
)

func testToolOption() *genai.GenOptionTools {
	return &genai.GenOptionTools{
		Tools: []genai.ToolDef{
			{
				Name:                "square_root",
				Description:         "Calculates and returns the square root of a number",
				InputSchemaOverride: genai.JSONSchema(`{"type":"object","properties":{"number":{"type":"number"}},"required":["number"]}`),
			},
		},
		Force: genai.ToolCallRequired,
	}
}

func TestChatRequest(t *testing.T) {
	t.Run("Init/tools/gpt-5.6 defaults to no reasoning", func(t *testing.T) {
		var r ChatRequest
		err := r.Init(genai.Messages{genai.NewTextMessage("calculate")}, "gpt-5.6-luna", testToolOption(), &GenOptionText{ServiceTier: ServiceTierFlex})
		if err != nil {
			t.Fatal(err)
		}
		if r.ReasoningEffort != ReasoningEffortNone {
			t.Fatalf("got %q, want %q", r.ReasoningEffort, ReasoningEffortNone)
		}
	})

	t.Run("Init/tools/gpt-5.6 rejects explicit reasoning", func(t *testing.T) {
		var r ChatRequest
		err := r.Init(genai.Messages{genai.NewTextMessage("calculate")}, "gpt-5.6-luna", testToolOption(), &GenOptionText{ReasoningEffort: ReasoningEffortLow})
		uerr, ok := errors.AsType[*base.ErrNotSupported](err)
		if !ok {
			t.Fatalf("got %v, want ErrNotSupported", err)
		}
		if len(uerr.Options) != 1 || uerr.Options[0] != "GenOptionText.ReasoningEffort" {
			t.Fatalf("got unsupported options %#v, want GenOptionText.ReasoningEffort", uerr.Options)
		}
	})
}
