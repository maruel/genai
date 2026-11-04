// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Tests for the scoreboard table rows.

package main

import (
	"slices"
	"testing"

	"github.com/maruel/genai"
	"github.com/maruel/genai/scoreboard"
)

func realtimeRow(in, out map[scoreboard.Modality]scoreboard.ModalCapability, f *scoreboard.RealtimeFunctionality) scoreboard.Realtime {
	return scoreboard.Realtime{
		Reasoned: scoreboard.Reasoned{Row: scoreboard.Row{In: in}},
		Out:      out,
		Realtime: f,
	}
}

func TestTableRows(t *testing.T) {
	// A model no section measured is one queue entry, and it comes after every
	// measured row.
	sb := scoreboard.Score{
		Chat: []scoreboard.Chat{
			{Reasoned: scoreboard.Reasoned{Row: scoreboard.Row{Models: []string{"measured"}}}, GenSync: &scoreboard.Functionality{}},
			{Reasoned: scoreboard.Reasoned{Row: scoreboard.Row{Models: []string{"sota"}, SOTA: true}}, GenSync: &scoreboard.Functionality{}},
			{Reasoned: scoreboard.Reasoned{Row: scoreboard.Row{Models: []string{"queued"}}}},
		},
		Embed:    []scoreboard.Embed{{Row: scoreboard.Row{Models: []string{"queued"}}}},
		Realtime: []scoreboard.Realtime{{Reasoned: scoreboard.Reasoned{Row: scoreboard.Row{Models: []string{"audio"}}}}},
	}
	rows := tableRows(&sb, genai.ProviderCapabilities{})
	got := make([]string, 0, len(rows))
	for _, r := range rows {
		got = append(got, r.Model+" "+r.Mode)
	}
	want := []string{"sota🥇 Sync", "measured Sync", "audio ?", "queued ?"}
	if !slices.Equal(got, want) {
		t.Fatalf("rows:\n%q\nwant:\n%q", got, want)
	}
}

func TestTableDataRow(t *testing.T) {
	audioText := map[scoreboard.Modality]scoreboard.ModalCapability{
		scoreboard.ModalityAudio: {Inline: true},
		scoreboard.ModalityText:  {Inline: true},
	}
	audio := map[scoreboard.Modality]scoreboard.ModalCapability{scoreboard.ModalityAudio: {Inline: true}}
	t.Run("wire", func(t *testing.T) {
		// A row measured over the WebSocket wire says so; an HTTP row says nothing.
		rows := []tableModelRow{{Mode: "Sync"}}
		markWire(rows, &scoreboard.Row{Wire: scoreboard.WireWebSocket})
		if rows[0].Mode != "SyncWS" {
			t.Fatalf("rows=%+v", rows)
		}
		rows = []tableModelRow{{Mode: "Sync"}}
		markWire(rows, &scoreboard.Row{Wire: scoreboard.WireHTTP})
		if rows[0].Mode != "Sync" {
			t.Fatalf("rows=%+v", rows)
		}
	})
	t.Run("initFromRealtime", func(t *testing.T) {
		s := realtimeRow(audioText, audioText, &scoreboard.RealtimeFunctionality{Tools: scoreboard.True})
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
		s := realtimeRow(audio, audio, &scoreboard.RealtimeFunctionality{Tools: scoreboard.True})
		row := tableDataRow{}
		row.initFromRealtime(&s)
		if row.Tools != "✅" {
			t.Fatalf("row=%+v", row)
		}
	})
	t.Run("usage", func(t *testing.T) {
		s := realtimeRow(audio, audio, &scoreboard.RealtimeFunctionality{ReportRateLimits: true, ReportTokenUsage: scoreboard.True})
		row := tableDataRow{}
		row.initFromRealtime(&s)
		if row.RateLimits != "✅" || row.Usage != "✅" {
			t.Fatalf("row=%+v", row)
		}
	})
	t.Run("flaky tools", func(t *testing.T) {
		s := realtimeRow(audio, audio, &scoreboard.RealtimeFunctionality{Tools: scoreboard.Flaky})
		row := tableDataRow{}
		row.initFromRealtime(&s)
		if row.Tools != "💨" {
			t.Fatalf("row=%+v", row)
		}
	})
	t.Run("initFromChat", func(t *testing.T) {
		s := scoreboard.Chat{
			Reasoned: scoreboard.Reasoned{Row: scoreboard.Row{In: audioText}, Reason: true},
			Out:      audioText,
			GenSync:  &scoreboard.Functionality{Seed: true},
		}
		row := tableDataRow{}
		row.initFromChat(&s, s.GenSync)
		if row.Mode != "Sync🧠" || row.Inputs != "🎤💬" || row.Outputs != "🎤💬" {
			t.Fatalf("row=%+v", row)
		}
	})
	t.Run("initFromEmbed", func(t *testing.T) {
		s := scoreboard.Embed{
			Row:   scoreboard.Row{In: map[scoreboard.Modality]scoreboard.ModalCapability{scoreboard.ModalityText: {Inline: true}}},
			Embed: &scoreboard.EmbeddingFunctionality{Dimensions: 2},
		}
		row := tableDataRow{}
		row.initFromEmbed(&s)
		if row.Mode != "Embed" || row.Inputs != "💬" || row.Outputs != "🧬" {
			t.Fatalf("row=%+v", row)
		}
	})
}
