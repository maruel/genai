// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Tests for the scoreboard command.

package main

import (
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
	t.Run("SystemOne", func(t *testing.T) {
		var row tableDataRow
		row.initFromSystemOne(&scoreboard.Scenario{Models: []string{"decision"}, In: map[scoreboard.Modality]scoreboard.ModalCapability{scoreboard.ModalityText: {Inline: true}}, Out: map[scoreboard.Modality]scoreboard.ModalCapability{scoreboard.ModalityDecision: {Inline: true}}, SystemOne: &scoreboard.DecisionFunctionality{Noul: true}})
		if row.Mode != "SystemOne" || row.Inputs != "💬" || row.Outputs != "🎯" {
			t.Fatalf("unexpected row: %+v", row)
		}
	})
}
