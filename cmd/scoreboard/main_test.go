// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Tests for the scoreboard command.

package main

import (
	"strings"
	"testing"

	"github.com/maruel/genai/internal/internaltest"
	"github.com/maruel/genai/scoreboard"
)

func TestPrintList(t *testing.T) {
	t.Parallel()
	_ = printList(t.Context(), &internaltest.WriterToLog{T: t})
}

func TestPrintTable(t *testing.T) {
	t.Parallel()
	ctx := t.Context()
	_ = printTable(ctx, &internaltest.WriterToLog{T: t}, "")
	_ = printTable(ctx, &internaltest.WriterToLog{T: t}, "openaicompatible")
	// Test a provider with scoreboard variants.
	_ = printTable(ctx, &internaltest.WriterToLog{T: t}, "alibaba")
}

func TestTableDataRow(t *testing.T) {
	t.Run("initFromScenario", func(t *testing.T) {
		for _, tc := range []struct {
			name   string
			f      scoreboard.Functionality
			symbol string
		}{
			{"unknown", scoreboard.Functionality{Tools: scoreboard.True}, "✅"},
			{"true", scoreboard.Functionality{Tools: scoreboard.True, OutOfOrder: new(scoreboard.True)}, "✅🔀"},
			{"false", scoreboard.Functionality{Tools: scoreboard.True, OutOfOrder: new(scoreboard.False)}, "✅"},
			{"flaky", scoreboard.Functionality{Tools: scoreboard.True, OutOfOrder: new(scoreboard.Flaky)}, "✅"},
		} {
			t.Run(tc.name, func(t *testing.T) {
				var row tableDataRow
				row.initFromScenario(&scoreboard.Scenario{GenSync: &tc.f}, &tc.f)
				if row.Tools != tc.symbol {
					t.Fatalf("row=%+v", row)
				}
				want := "✅tools"
				if tc.symbol == "✅🔀" {
					want += "🔀"
				}
				if got, _, _ := strings.Cut(functionality(&tc.f), " "); got != want {
					t.Fatalf("list=%s", got)
				}
			})
		}
	})
	t.Run("summary", func(t *testing.T) {
		var row tableDataRow
		for _, value := range []*scoreboard.TriState{nil, new(scoreboard.False), new(scoreboard.True), nil, new(scoreboard.True)} {
			f := scoreboard.Functionality{Tools: scoreboard.True, ToolCallRequired: true, WebSearch: true, OutOfOrder: value}
			row.initFromScenario(&scoreboard.Scenario{GenSync: &f}, &f)
		}
		if row.Tools != "✅🪨🕸️🔀" {
			t.Fatalf("tools=%s", row.Tools)
		}
		if w := visibleWidth(row.Tools); w != 8 {
			t.Fatalf("tools width=%d want=8", w)
		}
	})
	t.Run("Embed", func(t *testing.T) {
		var row tableDataRow
		row.initFromEmbed(&scoreboard.Scenario{Models: []string{"embed"}, In: map[scoreboard.Modality]scoreboard.ModalCapability{scoreboard.ModalityText: {Inline: true}}, Out: map[scoreboard.Modality]scoreboard.ModalCapability{scoreboard.ModalityEmbedding: {Inline: true}}, Embed: &scoreboard.EmbeddingFunctionality{Dimensions: 768, RequestedDimensions: new(true), ReportTokenUsage: scoreboard.True}})
		if row.Mode != "Embed" || row.Inputs != "💬" || row.Outputs != "🧬" || row.Usage != "✅" {
			t.Fatalf("row %+v", row)
		}
	})
	t.Run("SystemOne", func(t *testing.T) {
		var row tableDataRow
		row.initFromSystemOne(&scoreboard.Scenario{Models: []string{"decision"}, In: map[scoreboard.Modality]scoreboard.ModalCapability{scoreboard.ModalityText: {Inline: true}}, Out: map[scoreboard.Modality]scoreboard.ModalCapability{scoreboard.ModalityDecision: {Inline: true}}, SystemOne: &scoreboard.DecisionFunctionality{Noul: true}})
		if row.Mode != "SystemOne" || row.Inputs != "💬" || row.Outputs != "🎯" {
			t.Fatalf("unexpected row: %+v", row)
		}
	})
}
