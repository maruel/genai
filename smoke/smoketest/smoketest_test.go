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
	"testing"

	"github.com/maruel/genai"
	"github.com/maruel/genai/base"
	"github.com/maruel/genai/scoreboard"
)

func TestRunOneModelUpdateScoreboard(t *testing.T) {
	old := *updateScoreboard
	*updateScoreboard = true
	t.Cleanup(func() {
		*updateScoreboard = old
	})
	want := &scoreboard.Scenario{Models: []string{"model"}}
	_, got := runOneModel(t, func(testing.TB, string) genai.Provider {
		return &scoreboardProvider{}
	}, want, false)
	if got == nil || got.GenSync == nil || got.GenSync.Seed != true {
		t.Fatalf("generated scenario = %#v, want seeded GenSync scenario", got)
	}
}

func TestRunOptionsQualifies(t *testing.T) {
	m := scoreboard.Model{Model: "model", Reason: true}
	opts := &RunOptions{Qualify: []scoreboard.Model{m}}
	old := *updateScoreboard
	t.Cleanup(func() {
		*updateScoreboard = old
	})
	for _, test := range []struct {
		name   string
		update bool
		model  scoreboard.Model
		want   bool
	}{
		{name: "update", update: true, model: m, want: true},
		{name: "normal", model: m},
		{name: "wrong reasoning", update: true, model: scoreboard.Model{Model: m.Model}},
	} {
		t.Run(test.name, func(t *testing.T) {
			*updateScoreboard = test.update
			if got := opts.qualifies(test.model); got != test.want {
				t.Errorf("qualifies(%v) = %t, want %t", test.model, got, test.want)
			}
		})
	}
}

type scoreboardProvider struct {
	base.NotImplemented
}

func (*scoreboardProvider) Close() error {
	return nil
}

func (s *scoreboardProvider) Name() string {
	return "scoreboard"
}

func (s *scoreboardProvider) ModelID() string {
	return "model"
}

func (s *scoreboardProvider) OutputModalities() genai.Modalities {
	return genai.Modalities{genai.ModalityText}
}

func (s *scoreboardProvider) HTTPClient() *http.Client {
	return nil
}

func (s *scoreboardProvider) Scoreboard() scoreboard.Score {
	return scoreboard.Score{}
}

func (s *scoreboardProvider) GenSync(context.Context, genai.Messages, ...genai.GenOption) (genai.Result, error) {
	return scoreboardResult(), nil
}

func (s *scoreboardProvider) GenStream(context.Context, genai.Messages, ...genai.GenOption) (iter.Seq[genai.Reply], func() (genai.Result, error)) {
	r := scoreboardResult()
	return func(yield func(genai.Reply) bool) {
			yield(r.Replies[0])
		}, func() (genai.Result, error) {
			return r, nil
		}
}

func scoreboardResult() genai.Result {
	return genai.Result{
		Message: genai.Message{Replies: []genai.Reply{{Text: "hello"}}},
		Usage: genai.Usage{
			InputTokens:  1,
			OutputTokens: 1,
			TotalTokens:  2,
			FinishReason: genai.FinishedStop,
		},
	}
}

func TestGenerateUpdatedScoreboard(t *testing.T) {
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
