// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Tests for the scoreboard table.

package main

import (
	"testing"

	"github.com/maruel/genai/scoreboard"
)

func TestTableDataRow(t *testing.T) {
	t.Run("initFromRealtime", func(t *testing.T) {
		s := scoreboard.Scenario{
			In:       map[scoreboard.Modality]scoreboard.ModalCapability{scoreboard.ModalityAudio: {Inline: true}, scoreboard.ModalityText: {Inline: true}},
			Out:      map[scoreboard.Modality]scoreboard.ModalCapability{scoreboard.ModalityAudio: {Inline: true}, scoreboard.ModalityText: {Inline: true}},
			Realtime: &scoreboard.RealtimeFunctionality{Tools: scoreboard.True},
		}
		row := tableDataRow{}
		row.initFromRealtime(&s)
		if row.Mode != "Realtime" || row.Tools != "✅" {
			t.Fatalf("row=%+v", row)
		}
		if row.Inputs != "🎤💬" || row.Outputs != "🎤💬" {
			t.Fatalf("row=%+v", row)
		}
	})

	t.Run("audio tool results", func(t *testing.T) {
		s := scoreboard.Scenario{
			In:       map[scoreboard.Modality]scoreboard.ModalCapability{scoreboard.ModalityAudio: {Inline: true}},
			Out:      map[scoreboard.Modality]scoreboard.ModalCapability{scoreboard.ModalityAudio: {Inline: true}},
			Realtime: &scoreboard.RealtimeFunctionality{Tools: scoreboard.True},
		}
		row := tableDataRow{}
		row.initFromRealtime(&s)
		if row.Tools != "✅" {
			t.Fatalf("row=%+v", row)
		}
	})

	t.Run("usage", func(t *testing.T) {
		s := scoreboard.Scenario{
			In:  map[scoreboard.Modality]scoreboard.ModalCapability{scoreboard.ModalityAudio: {Inline: true}},
			Out: map[scoreboard.Modality]scoreboard.ModalCapability{scoreboard.ModalityAudio: {Inline: true}},
			Realtime: &scoreboard.RealtimeFunctionality{
				ReportRateLimits: true,
				ReportTokenUsage: scoreboard.True,
			},
		}
		row := tableDataRow{}
		row.initFromRealtime(&s)
		if row.RateLimits != "✅" || row.Usage != "✅" {
			t.Fatalf("row=%+v", row)
		}
	})

	t.Run("flaky tools", func(t *testing.T) {
		s := scoreboard.Scenario{
			In:       map[scoreboard.Modality]scoreboard.ModalCapability{scoreboard.ModalityAudio: {Inline: true}},
			Out:      map[scoreboard.Modality]scoreboard.ModalCapability{scoreboard.ModalityAudio: {Inline: true}},
			Realtime: &scoreboard.RealtimeFunctionality{Tools: scoreboard.Flaky},
		}
		row := tableDataRow{}
		row.initFromRealtime(&s)
		if row.Tools != "💨" {
			t.Fatalf("row=%+v", row)
		}
	})
}
