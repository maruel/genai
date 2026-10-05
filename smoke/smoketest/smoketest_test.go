// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Tests for the smoke test scoreboard updater.

package smoketest

import (
	"context"
	"encoding/json"
	"iter"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/maruel/genai"
	"github.com/maruel/genai/base"
	"github.com/maruel/genai/scoreboard"
)

func TestRunOneModel(t *testing.T) {
	old := *updateScoreboard
	t.Cleanup(func() { *updateScoreboard = old })
	*updateScoreboard = true
	var existingProbes, outOfOrder atomic.Int64
	gc := func(_ testing.TB, name string) genai.Provider {
		if strings.Contains(name, "/GenSync-OutOfOrder-") || strings.Contains(name, "/GenStream-OutOfOrder-") {
			outOfOrder.Add(1)
		} else if name != "" {
			existingProbes.Add(1)
		}
		return &scoreboardProvider{}
	}
	_, measured := runOneModel(t, gc, &scoreboard.Scenario{Models: []string{"model"}}, false)
	if measured == nil || measured.GenSync == nil || !measured.GenSync.Seed {
		t.Fatalf("generated scenario = %#v, want seeded GenSync scenario", measured)
	}
	if outOfOrder.Load() != 0 || measured.GenSync.OutOfOrder != nil || measured.GenStream.OutOfOrder != nil {
		t.Fatal("update measured OutOfOrder without successful basic tool calls")
	}
	baseline := existingProbes.Load()
	*updateScoreboard = false
	t.Run("replay", func(t *testing.T) {
		existingProbes.Store(0)
		outOfOrder.Store(0)
		_, got := runOneModel(t, gc, measured, false)
		if existingProbes.Load() != baseline || outOfOrder.Load() != 0 {
			t.Fatalf("existingProbes=%d want=%d OutOfOrder=%d want=0", existingProbes.Load(), baseline, outOfOrder.Load())
		}
		if got.GenSync.OutOfOrder != nil || got.GenStream.OutOfOrder != nil {
			t.Fatal("replay measured OutOfOrder without successful basic tool calls")
		}
	})
}

func TestRunOptions(t *testing.T) {
	t.Run("qualifies", func(t *testing.T) {
		m := scoreboard.Model{Model: "model", Reason: true}
		opts := &RunOptions{Qualify: []scoreboard.Model{m}}
		old := *updateScoreboard
		t.Cleanup(func() { *updateScoreboard = old })
		for _, tc := range []struct {
			name   string
			update bool
			model  scoreboard.Model
			want   bool
		}{
			{name: "update", update: true, model: m, want: true},
			{name: "normal", model: m},
			{name: "wrong reasoning", update: true, model: scoreboard.Model{Model: m.Model}},
		} {
			t.Run(tc.name, func(t *testing.T) {
				*updateScoreboard = tc.update
				if got := opts.qualifies(tc.model); got != tc.want {
					t.Errorf("qualifies(%v) = %t, want %t", tc.model, got, tc.want)
				}
			})
		}
	})
}

func TestGenerateUpdatedScoreboard(t *testing.T) {
	t.Run("OutOfOrder", func(t *testing.T) {
		path := filepath.Join(t.TempDir(), "scoreboard.json")
		old := []byte(`{"country":"Local","dashboardURL":"","scenarios":[{"models":["first","sibling"],"good":true,"in":{"text":{"inline":true}},"out":{"text":{"inline":true}},"GenSync":{"tools":"true"}}]}`)
		if err := os.WriteFile(path, old, 0o600); err != nil {
			t.Fatal(err)
		}
		sc := scoreboard.Scenario{
			Models:  []string{"first"},
			In:      map[scoreboard.Modality]scoreboard.ModalCapability{scoreboard.ModalityText: {Inline: true}},
			Out:     map[scoreboard.Modality]scoreboard.ModalCapability{scoreboard.ModalityText: {Inline: true}},
			GenSync: &scoreboard.Functionality{Tools: scoreboard.True, OutOfOrder: new(scoreboard.True)},
		}
		_, raw := generateUpdatedScoreboard(t, path, []scoreboard.Scenario{sc}, nil, false)
		var sb scoreboard.Score
		if err := json.Unmarshal(raw, &sb); err != nil {
			t.Fatal(err)
		}
		if len(sb.Scenarios) != 2 {
			t.Fatalf("measured model consolidated with unknown: %s", raw)
		}
		if !sb.Scenarios[0].Good || sb.Scenarios[0].Models[0] != "first" || sb.Scenarios[0].GenSync.OutOfOrder == nil {
			t.Fatal("lost measured model metadata")
		}
		if sb.Scenarios[1].Models[0] != "sibling" || sb.Scenarios[1].GenSync.OutOfOrder != nil || sb.Scenarios[1].GenSync.Tools != scoreboard.True {
			t.Fatal("manufactured measurement for sibling or lost existing Tools score")
		}
	})
	t.Run("filtered preserves tiers", func(t *testing.T) {
		path := filepath.Join(t.TempDir(), "scoreboard.json")
		old := []byte(`{"country":"US","dashboardURL":"","scenarios":[{"models":["chat"],"sota":true,"GenSync":{}},{"models":["decision"],"SystemOne":{}}]}`)
		if err := os.WriteFile(path, old, 0o600); err != nil {
			t.Fatal(err)
		}
		_, raw := generateUpdatedScoreboard(t, path, []scoreboard.Scenario{{Models: []string{"decision"}, In: map[scoreboard.Modality]scoreboard.ModalCapability{scoreboard.ModalityText: {Inline: true}}, Out: map[scoreboard.Modality]scoreboard.ModalCapability{scoreboard.ModalityText: {Inline: true}}, SystemOne: &scoreboard.DecisionFunctionality{Noul: true}}}, nil, true)
		var sb scoreboard.Score
		if err := json.Unmarshal(raw, &sb); err != nil {
			t.Fatal(err)
		}
		if len(sb.Scenarios) != 2 || sb.Scenarios[0].Models[0] != "chat" || !sb.Scenarios[0].SOTA || sb.Scenarios[0].GenSync == nil {
			t.Fatalf("unfiltered model changed: %+v", sb.Scenarios)
		}
		if sb.Scenarios[1].SystemOne == nil || !sb.Scenarios[1].SystemOne.Noul {
			t.Fatalf("decision result lost: %+v", sb.Scenarios[1])
		}
	})
}

type scoreboardProvider struct{ base.NotImplemented }

func (*scoreboardProvider) Close() error    { return nil }
func (*scoreboardProvider) Name() string    { return "scoreboard" }
func (*scoreboardProvider) ModelID() string { return "model" }
func (*scoreboardProvider) OutputModalities() genai.Modalities {
	return genai.Modalities{genai.ModalityText}
}
func (*scoreboardProvider) HTTPClient() *http.Client     { return nil }
func (*scoreboardProvider) Scoreboard() scoreboard.Score { return scoreboard.Score{} }
func (*scoreboardProvider) GenSync(context.Context, genai.Messages, ...genai.GenOption) (genai.Result, error) {
	return scoreboardResult(), nil
}
func (*scoreboardProvider) GenStream(context.Context, genai.Messages, ...genai.GenOption) (iter.Seq[genai.Reply], func() (genai.Result, error)) {
	r := scoreboardResult()
	return func(yield func(genai.Reply) bool) { yield(r.Replies[0]) }, func() (genai.Result, error) { return r, nil }
}
func scoreboardResult() genai.Result {
	return genai.Result{Message: genai.Message{Replies: []genai.Reply{{Text: "hello"}}}, Usage: genai.Usage{InputTokens: 1, OutputTokens: 1, TotalTokens: 2, FinishReason: genai.FinishedStop}}
}
