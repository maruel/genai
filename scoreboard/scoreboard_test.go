// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Tests for the sectioned scoreboard.

package scoreboard

import (
	"bytes"
	"encoding/json"
	"testing"
)

func TestKeyString(t *testing.T) {
	cases := []struct {
		key  Key
		want string
	}{
		{Key{Model: "gpt-4"}, "gpt-4"},
		{Key{Model: "gpt-4", Reason: true}, "gpt-4_thinking"},
		{Key{Model: "gpt-4", Wire: WireHTTP}, "gpt-4"},
		{Key{Model: "gpt-4", Wire: WireSession}, "gpt-4"},
		{Key{Model: "gpt-4", Wire: WireWebSocket}, "gpt-4_ws"},
		{Key{Model: "gpt-4", Wire: WireSubprocess}, "gpt-4_cli"},
		{Key{Model: "model:tag", Reason: true}, "model-tag_thinking"},
	}
	for _, c := range cases {
		if got := c.key.String(); got != c.want {
			t.Errorf("%#v: got %q, want %q", c.key, got, c.want)
		}
	}
}

func TestRealtimeFunctionality(t *testing.T) {
	t.Run("Validate", func(t *testing.T) {
		t.Run("valid", func(t *testing.T) {
			f := RealtimeFunctionality{Voices: []string{"alpha", "zulu"}}
			if err := f.Validate(); err != nil {
				t.Fatal(err)
			}
		})
		t.Run("error", func(t *testing.T) {
			for _, voices := range [][]string{{" "}, {"alpha", "alpha"}, {"zulu", "alpha"}} {
				f := RealtimeFunctionality{Voices: voices}
				if err := f.Validate(); err == nil {
					t.Errorf("invalid voices accepted: %v", voices)
				}
			}
		})
	})
}

func TestScoreValidate(t *testing.T) {
	text := map[Modality]ModalCapability{ModalityText: {Inline: true}}
	cases := []struct {
		name  string
		score Score
		want  string
	}{
		{"empty", Score{}, ""},
		{"chat", Score{Chat: []Chat{{Reasoned: Reasoned{Row: Row{Models: []string{"m"}, In: text}}, Out: text, GenSync: &Functionality{}}}}, ""},
		{"chat without models", Score{Chat: []Chat{{Reasoned: Reasoned{Row: Row{In: text}}, Out: text}}}, "chat: scenario must have at least one model"},
		{"chat with in only", Score{Chat: []Chat{{Reasoned: Reasoned{Row: Row{Models: []string{"m"}, In: text}}}}}, "chat: scenario must have either both or none of In or Out"},
		{"duplicate", Score{Chat: []Chat{
			{Reasoned: Reasoned{Row: Row{Models: []string{"m"}, In: text}}, Out: text},
			{Reasoned: Reasoned{Row: Row{Models: []string{"m"}, In: text}}, Out: text},
		}}, `chat: duplicate model in scoreboard: "m" (reason=false, wire=http)`},
		{"the same model in two sections", Score{
			Chat:  []Chat{{Reasoned: Reasoned{Row: Row{Models: []string{"m"}, In: text}}, Out: text}},
			Embed: []Embed{{Row: Row{Models: []string{"m"}, In: text}, Embed: &EmbeddingFunctionality{Dimensions: 2}}},
		}, ""},
		{"embed without in", Score{Embed: []Embed{{Row: Row{Models: []string{"m"}}, Embed: &EmbeddingFunctionality{Dimensions: 2}}}}, "embed: embedding functionality requires inline text input"},
		{"embed with an output", Score{Embed: []Embed{{Row: Row{Models: []string{"m"}, In: text}, Embed: &EmbeddingFunctionality{Dimensions: 2}}}}, ""},
		{"decision without modalities", Score{SystemOne: []SystemOne{{Row: Row{Models: []string{"m"}}, SystemOne: &DecisionFunctionality{Noul: true}}}}, "systemone: scenario with decision functionality must define input and output modalities"},
		{"realtime", Score{Realtime: []Realtime{{
			Reasoned: Reasoned{Row: Row{Models: []string{"m"}, In: text}},
			Out:      text,
			Realtime: &RealtimeFunctionality{InputSampleRate: 16000, OutputSampleRate: 24000},
		}}}, ""},
		{"two sota in one modality", Score{Chat: []Chat{
			{Reasoned: Reasoned{Row: Row{Models: []string{"a"}, SOTA: true, In: text}}, Out: text},
			{Reasoned: Reasoned{Row: Row{Models: []string{"b"}, SOTA: true, In: text}}, Out: text},
		}}, `chat: multiple sota models marked for modality "text" (count: 2)`},
		{"good before sota", Score{Chat: []Chat{
			{Reasoned: Reasoned{Row: Row{Models: []string{"a"}, Good: true, In: text}}, Out: text},
			{Reasoned: Reasoned{Row: Row{Models: []string{"b"}, SOTA: true, In: text}}, Out: text},
		}}, `chat: sota model "b" for modality "text" should be first in that modality's scenarios (at position 1)`},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			err := c.score.Validate()
			if c.want == "" {
				if err != nil {
					t.Fatalf("got %v, want no error", err)
				}
				return
			}
			if err == nil || err.Error() != c.want {
				t.Fatalf("got %v, want %q", err, c.want)
			}
		})
	}
}

func TestScoreRows(t *testing.T) {
	text := map[Modality]ModalCapability{ModalityText: {Inline: true}}
	s := Score{
		Chat:      []Chat{{Reasoned: Reasoned{Row: Row{Models: []string{"chat"}, In: text}, Reason: true}, Out: text}},
		Embed:     []Embed{{Row: Row{Models: []string{"embed"}, In: text}}},
		SystemOne: []SystemOne{{Row: Row{Models: []string{"decision"}, In: text}}},
		Realtime:  []Realtime{{Reasoned: Reasoned{Row: Row{Models: []string{"live"}, In: text}}, Out: text}},
	}
	rows := s.Rows()
	if len(rows) != 4 {
		t.Fatalf("got %d rows, want 4", len(rows))
	}
	if got := rows[0].Models[0]; got != "chat" {
		t.Errorf("first row is %q, want chat", got)
	}
	if got := rows[3].Models[0]; got != "live" {
		t.Errorf("last row is %q, want live", got)
	}
	models := s.Models()
	if len(models) != 4 {
		t.Fatalf("got %d models, want 4", len(models))
	}
	if !models[0].Reason || models[1].Reason || models[2].Reason || models[3].Reason {
		t.Errorf("unexpected reasoning modes: %v", models)
	}
}

func TestScoreEmbedding(t *testing.T) {
	s := Score{Embed: []Embed{
		{Row: Row{Models: []string{"pending"}}, Embed: &EmbeddingFunctionality{}},
		{Row: Row{Models: []string{"measured"}}, Embed: &EmbeddingFunctionality{Dimensions: 768}},
	}}
	if got := s.Embedding("measured"); got == nil || got.Dimensions != 768 {
		t.Fatalf("got %#v", got)
	}
	if got := s.Embedding("pending"); got != nil {
		t.Fatalf("an empty marker is not a measurement: %#v", got)
	}
	if got := s.Embedding("absent"); got != nil {
		t.Fatalf("got %#v", got)
	}
}

func TestSortRows(t *testing.T) {
	text := map[Modality]ModalCapability{ModalityText: {Inline: true}}
	rows := []Chat{
		{Reasoned: Reasoned{Row: Row{Models: []string{"zzz"}, In: text}}, Out: text},
		{Reasoned: Reasoned{Row: Row{Models: []string{"cheap"}, Cheap: true, In: text}}, Out: text},
		{Reasoned: Reasoned{Row: Row{Models: []string{"sota"}, SOTA: true, In: text}}, Out: text},
		{Reasoned: Reasoned{Row: Row{Models: []string{"queue"}}}},
		{Reasoned: Reasoned{Row: Row{Models: []string{"good"}, Good: true, In: text}}, Out: text},
	}
	SortRows[Chat](rows)
	want := []string{"sota", "good", "cheap", "zzz", "queue"}
	for i, w := range want {
		if rows[i].Models[0] != w {
			t.Fatalf("position %d is %q, want %q (%v)", i, rows[i].Models[0], w, rows)
		}
	}
}

func TestConsolidateQueue(t *testing.T) {
	rows := []Embed{
		{Row: Row{Models: []string{"b"}, Comments: "pending"}},
		{Row: Row{Models: []string{"a"}, Comments: "pending"}},
		{Row: Row{Models: []string{"c"}, Comments: "other"}},
		{Row: Row{Models: []string{"measured"}, In: map[Modality]ModalCapability{ModalityText: {Inline: true}}}, Embed: &EmbeddingFunctionality{Dimensions: 2}},
	}
	got := ConsolidateQueue[Embed](rows)
	if len(got) != 2 {
		t.Fatalf("got %d queue entries, want 2: %#v", len(got), got)
	}
	if want := []string{"a", "b"}; !slicesEqual(got[0].Models, want) {
		t.Errorf("got %v, want %v", got[0].Models, want)
	}
	if got[1].Comments != "other" {
		t.Errorf("got %q", got[1].Comments)
	}
}

func TestScoreJSON(t *testing.T) {
	text := map[Modality]ModalCapability{ModalityText: {Inline: true}}
	in := Score{
		Country:      "US",
		DashboardURL: "https://example.com",
		Chat: []Chat{{
			Reasoned:            Reasoned{Row: Row{Models: []string{"chat"}, In: text}, Reason: true},
			Out:                 text,
			ReasoningTokenStart: "<think>",
			ReasoningTokenEnd:   "</think>",
			GenSync:             &Functionality{Seed: true},
		}},
		Embed: []Embed{{Row: Row{Models: []string{"embed"}, In: text}, Embed: &EmbeddingFunctionality{Dimensions: 2}}},
	}
	raw, err := json.Marshal(in)
	if err != nil {
		t.Fatal(err)
	}
	var out Score
	d := json.NewDecoder(bytes.NewReader(raw))
	d.DisallowUnknownFields()
	if err = d.Decode(&out); err != nil {
		t.Fatal(err)
	}
	if len(out.Chat) != 1 || out.Chat[0].ReasoningTokenStart != "<think>" || !out.Chat[0].Reason {
		t.Fatalf("chat lost: %#v", out.Chat)
	}
	if len(out.Embed) != 1 || out.Embed[0].Embed == nil || out.Embed[0].Embed.Dimensions != 2 {
		t.Fatalf("embed lost: %#v", out.Embed)
	}
	if err = out.Validate(); err != nil {
		t.Fatal(err)
	}
}

func TestModalityValidate(t *testing.T) {
	if err := ModalityText.Validate(); err != nil {
		t.Fatal(err)
	}
	if err := Modality("bogus").Validate(); err == nil {
		t.Fatal("expected an error")
	}
}

func TestTriState(t *testing.T) {
	for _, c := range []struct {
		in   TriState
		want string
	}{{False, "false"}, {True, "true"}, {Flaky, "flaky"}} {
		if got := c.in.String(); got != c.want {
			t.Errorf("got %q, want %q", got, c.want)
		}
	}
	if err := True.Validate(); err != nil {
		t.Fatal(err)
	}
	if err := TriState(42).Validate(); err == nil {
		t.Fatal("expected an error")
	}
}

func TestFunctionalityValidate(t *testing.T) {
	f := Functionality{Seed: true, ReportTokenUsage: True}
	if err := f.Validate(); err != nil {
		t.Fatal(err)
	}
	bad := Functionality{ReportTokenUsage: TriState(42)}
	if err := bad.Validate(); err == nil {
		t.Fatal("expected an error")
	}
}

func TestReason(t *testing.T) {
	for _, c := range []struct {
		in   Reason
		want string
	}{{ReasonNone, `"none"`}, {ReasonInline, `"inline"`}, {ReasonAuto, `"auto"`}} {
		got, err := c.in.MarshalJSON()
		if err != nil {
			t.Fatal(err)
		}
		if string(got) != c.want {
			t.Errorf("got %s, want %s", got, c.want)
		}
		var back Reason
		if err = back.UnmarshalJSON(got); err != nil {
			t.Fatal(err)
		}
		if back != c.in {
			t.Errorf("round trip got %v, want %v", back, c.in)
		}
	}
	var r Reason
	if err := r.UnmarshalJSON([]byte(`"bogus"`)); err == nil {
		t.Fatal("expected an error")
	}
}

func slicesEqual(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
