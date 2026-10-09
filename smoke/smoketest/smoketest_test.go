// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Tests for the smoke test scoreboard updater.

package smoketest

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"iter"
	"net/http"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/google/go-cmp/cmp"

	"github.com/maruel/genai"
	"github.com/maruel/genai/base"
	"github.com/maruel/genai/internal/myrecorder"
	"github.com/maruel/genai/scoreboard"
)

func TestGenerateUpdatedScoreboard(t *testing.T) {
	t.Run("embedding media splits tested siblings", func(t *testing.T) {
		for _, added := range []bool{true, false} {
			name := "removed"
			if added {
				name = "added"
			}
			t.Run(name, func(t *testing.T) {
				oldIn := map[genai.Modality]scoreboard.ModalCapability{genai.ModalityText: {Inline: true}}
				newIn := map[genai.Modality]scoreboard.ModalCapability{genai.ModalityText: {Inline: true}}
				media := scoreboard.ModalCapability{Inline: true, SupportedFormats: []string{"image/jpeg"}}
				if added {
					newIn[genai.ModalityImage] = media
				} else {
					oldIn[genai.ModalityImage] = media
				}
				oldSc := scoreboard.Scenario{Models: []string{"measured", "sibling"}, In: oldIn, Out: map[genai.Modality]scoreboard.ModalCapability{genai.ModalityEmbedding: {Inline: true}}, Embed: &scoreboard.EmbeddingFunctionality{Dimensions: 768}}
				path := filepath.Join(t.TempDir(), "scoreboard.json")
				raw, err := json.Marshal(scoreboard.Score{Scenarios: []scoreboard.Scenario{oldSc}})
				if err != nil {
					t.Fatal(err)
				}
				if err := os.WriteFile(path, raw, 0o600); err != nil {
					t.Fatal(err)
				}
				measured := oldSc
				measured.Models = []string{"measured"}
				measured.In = newIn
				_, raw = generateUpdatedScoreboard(t, path, []scoreboard.Scenario{measured}, nil, true)
				var got scoreboard.Score
				if err := json.Unmarshal(raw, &got); err != nil {
					t.Fatal(err)
				}
				if len(got.Scenarios) != 2 {
					t.Fatalf("media change spread to sibling: %+v", got.Scenarios)
				}
				for _, sc := range got.Scenarios {
					if len(sc.Models) != 1 {
						t.Fatalf("grouped models %+v", sc)
					}
					want := oldIn
					if sc.Models[0] == "measured" {
						want = newIn
					}
					if !cmp.Equal(sc.In, want) || !cmp.Equal(sc.Embed, oldSc.Embed) {
						t.Fatalf("model %s changed unrelated qualification: %+v", sc.Models[0], sc)
					}
				}
			})
		}
	})
	t.Run("embedding preserves tested model notes", func(t *testing.T) {
		path := filepath.Join(t.TempDir(), "scoreboard.json")
		if err := os.WriteFile(path, []byte(`{"scenarios":[{"comments":"model-specific notes","models":["embedding"],"in":{"text":{"inline":true}},"out":{"embedding":{"inline":true}},"Embed":{"dimensions":768}}]}`), 0o600); err != nil {
			t.Fatal(err)
		}
		sc := scoreboard.Scenario{Models: []string{"embedding"}, In: map[scoreboard.Modality]scoreboard.ModalCapability{scoreboard.ModalityText: {Inline: true}}, Out: map[scoreboard.Modality]scoreboard.ModalCapability{scoreboard.ModalityEmbedding: {Inline: true}}, Embed: &scoreboard.EmbeddingFunctionality{Dimensions: 768, RequestedDimensions: new(true)}}
		_, raw := generateUpdatedScoreboard(t, path, []scoreboard.Scenario{sc}, nil, true)
		var sb scoreboard.Score
		if err := json.Unmarshal(raw, &sb); err != nil {
			t.Fatal(err)
		}
		if len(sb.Scenarios) != 1 || sb.Scenarios[0].Comments != "model-specific notes" {
			t.Fatalf("tested model metadata lost: %+v", sb)
		}
	})
	t.Run("embedding splits untested siblings", func(t *testing.T) {
		path := filepath.Join(t.TempDir(), "scoreboard.json")
		if err := os.WriteFile(path, []byte(`{"country":"US","dashboardURL":"dashboard","scenarios":[{"models":["chat"],"sota":true,"GenSync":{}},{"comments":"pending","models":["first","embedding","last"]}]}`), 0o600); err != nil {
			t.Fatal(err)
		}
		sc := scoreboard.Scenario{Models: []string{"embedding"}, In: map[scoreboard.Modality]scoreboard.ModalCapability{scoreboard.ModalityText: {Inline: true}}, Out: map[scoreboard.Modality]scoreboard.ModalCapability{scoreboard.ModalityEmbedding: {Inline: true}}, Embed: &scoreboard.EmbeddingFunctionality{Dimensions: 768, RequestedDimensions: new(false)}}
		_, raw := generateUpdatedScoreboard(t, path, []scoreboard.Scenario{sc}, nil, true)
		var sb scoreboard.Score
		if err := json.Unmarshal(raw, &sb); err != nil {
			t.Fatal(err)
		}
		if len(sb.Scenarios) != 3 || sb.Country != "US" || sb.DashboardURL != "dashboard" {
			t.Fatalf("metadata %+v", sb)
		}
		for i := range sb.Scenarios {
			s := &sb.Scenarios[i]
			switch s.Models[0] {
			case "chat":
				if !s.SOTA || s.GenSync == nil {
					t.Fatal("generation metadata lost")
				}
			case "embedding":
				if s.Embed == nil || s.Embed.Dimensions != 768 || s.Comments != "" {
					t.Fatalf("measurement lost: %+v", s)
				}
			case "first":
				if !s.Untested() || len(s.Models) != 2 || s.Models[1] != "last" || s.Comments != "pending" {
					t.Fatalf("manufactured sibling support: %+v", s)
				}
			default:
				t.Fatalf("unexpected scenario %+v", s)
			}
		}
	})
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

func TestRun(t *testing.T) {
	t.Run("Qualify", func(t *testing.T) {
		old := *updateScoreboard
		t.Cleanup(func() { *updateScoreboard = old })
		for _, tc := range []qualificationRunCase{{name: "update", update: true}, {name: "normal"}} {
			t.Run(tc.name, func(t *testing.T) {
				*updateScoreboard = tc.update
				sb := scoreboard.Score{Scenarios: []scoreboard.Scenario{
					{Models: []string{"aaa-first", "embedding", "embedding2", "zzz-last"}, Comments: "pending embedding models"},
					{Models: []string{"generation-first", "generation-qualified", "generation-last"}, Comments: "pending generation models"},
				}}
				for i := range sb.Scenarios {
					slices.Sort(sb.Scenarios[i].Models)
				}
				t.Chdir(t.TempDir())
				path := "scoreboard.json"
				b, err := json.Marshal(sb)
				if err != nil {
					t.Fatal(err)
				}
				if err := os.WriteFile(path, b, 0o600); err != nil {
					t.Fatal(err)
				}
				_, original := generateUpdatedScoreboard(t, path, nil, nil, true)
				if err := os.WriteFile(path, original, 0o600); err != nil {
					t.Fatal(err)
				}
				rec, err := myrecorder.NewRecords(t.TempDir())
				if err != nil {
					t.Fatal(err)
				}
				t.Cleanup(func() {
					if err := rec.Close(); err != nil {
						t.Error(err)
					}
				})
				clients := map[string]*qualificationProvider{}
				var models []scoreboard.Model
				for i := range sb.Scenarios {
					for _, id := range sb.Scenarios[i].Models {
						models = append(models, scoreboard.Model{Model: id})
						p := &qualificationProvider{model: id, score: sb, output: genai.Modalities{genai.ModalityText}}
						if id == "embedding" || id == "embedding2" {
							p.output = genai.Modalities{genai.ModalityEmbedding}
						}
						clients[id] = p
					}
				}
				if tc.update {
					models = append(models, scoreboard.Model{Model: "new-embedding"})
					clients["new-embedding"] = &qualificationProvider{model: "new-embedding", score: sb, output: genai.Modalities{genai.ModalityEmbedding}}
				}
				Run(t, func(_ testing.TB, m scoreboard.Model, _ func(http.RoundTripper) http.RoundTripper) genai.Provider {
					if m.Model == "" {
						return &qualificationProvider{score: sb}
					}
					return clients[m.Model]
				}, models, rec, &RunOptions{ScoreboardFile: path, Qualify: []scoreboard.Model{{Model: "embedding"}, {Model: "embedding2"}, {Model: "new-embedding"}, {Model: "generation-qualified"}}})
				raw, err := os.ReadFile(path)
				if err != nil {
					t.Fatal(err)
				}
				if !tc.update {
					if !bytes.Equal(raw, original) {
						t.Fatal("qualification modified scoreboard outside update mode")
					}
					for id, p := range clients {
						if len(p.dimensions) != 0 || p.syncCalls.Load() != 0 || p.streamCalls.Load() != 0 {
							t.Fatalf("qualification called untested model %q outside update mode", id)
						}
					}
					return
				}
				for id, p := range clients {
					switch id {
					case "embedding", "embedding2", "new-embedding":
						if !slices.Equal(p.dimensions, []int{0, 0, 0, 32, 0, 0}) || p.syncCalls.Load() != 0 || p.streamCalls.Load() != 0 {
							t.Fatalf("embedding %q probes: dimensions=%v sync=%d stream=%d", id, p.dimensions, p.syncCalls.Load(), p.streamCalls.Load())
						}
					case "generation-qualified":
						if len(p.dimensions) != 0 || p.syncCalls.Load() == 0 || p.streamCalls.Load() == 0 {
							t.Fatal("generation qualification did not retain generation probes")
						}
					default:
						if len(p.dimensions) != 0 || p.syncCalls.Load() != 0 || p.streamCalls.Load() != 0 {
							t.Fatalf("unqualified sibling %q was probed", id)
						}
					}
				}
				var got scoreboard.Score
				if err := json.Unmarshal(raw, &got); err != nil {
					t.Fatal(err)
				}
				measured := map[string]scoreboard.Scenario{}
				var untested []string
				for i := range got.Scenarios {
					sc := &got.Scenarios[i]
					if sc.Untested() {
						untested = append(untested, sc.Models...)
						if sc.Comments == "" {
							t.Fatal("untouched sibling notes were lost")
						}
						continue
					}
					if len(sc.Models) != 1 {
						t.Fatalf("measured scenario claims unqualified siblings: %+v", sc)
					}
					measured[sc.Models[0]] = *sc
				}
				for _, id := range []string{"embedding", "embedding2", "new-embedding"} {
					sc := measured[id]
					if !sc.In[genai.ModalityImage].Inline || !sc.In[genai.ModalityAudio].Inline {
						t.Fatalf("media measurements lost in scoreboard: %+v", sc)
					}
					if sc.Embed == nil || sc.Embed.Dimensions != 2 || sc.Embed.RequestedDimensions == nil || !*sc.Embed.RequestedDimensions || sc.Comments != "" || sc.GenSync != nil || sc.GenStream != nil {
						t.Fatalf("embedding measurement %q: %+v", id, sc)
					}
				}
				if sc := measured["generation-qualified"]; sc.GenSync == nil || sc.GenStream == nil || !sc.GenSync.Seed || sc.Comments != "pending generation models" {
					t.Fatalf("generation measurement: %+v", sc)
				}
				slices.Sort(untested)
				if !slices.Equal(untested, []string{"aaa-first", "generation-first", "generation-last", "zzz-last"}) {
					t.Fatalf("untouched siblings = %v", untested)
				}
			})
		}
	})
	t.Run("QualificationNotes", func(t *testing.T) {
		old := *updateScoreboard
		t.Cleanup(func() { *updateScoreboard = old })
		*updateScoreboard = true
		for _, tc := range []qualificationNotesCase{
			{name: "new unqualified model"},
			{name: "failed qualified sibling", qualify: true},
			{name: "tested becomes untested", tested: true},
		} {
			t.Run(tc.name, func(t *testing.T) {
				t.Chdir(t.TempDir())
				sc := scoreboard.Scenario{Models: []string{"a"}, Comments: "pending"}
				if tc.qualify {
					sc.Models = append(sc.Models, "new")
				}
				if tc.tested {
					sc.GenSync = &scoreboard.Functionality{}
				}
				sb := scoreboard.Score{Scenarios: []scoreboard.Scenario{sc}}
				raw, err := json.Marshal(sb)
				if err != nil {
					t.Fatal(err)
				}
				if err := os.WriteFile("scoreboard.json", raw, 0o600); err != nil {
					t.Fatal(err)
				}
				rec, err := myrecorder.NewRecords(t.TempDir())
				if err != nil {
					t.Fatal(err)
				}
				t.Cleanup(func() {
					if err := rec.Close(); err != nil {
						t.Error(err)
					}
				})
				opts := &RunOptions{}
				if tc.qualify {
					opts.Qualify = []scoreboard.Model{{Model: "new"}}
				}
				models := []scoreboard.Model{{Model: "a"}}
				if !tc.tested {
					models = append(models, scoreboard.Model{Model: "new"})
				}
				clients := map[string]*qualificationProvider{}
				for _, m := range append(slices.Clone(models), scoreboard.Model{}) {
					clients[m.Model] = &qualificationProvider{model: m.Model, score: sb, output: genai.Modalities{genai.ModalityText}, generationError: errors.New("generation unavailable")}
				}
				Run(t, func(_ testing.TB, m scoreboard.Model, _ func(http.RoundTripper) http.RoundTripper) genai.Provider {
					return clients[m.Model]
				}, models, rec, opts)
				measured := "new"
				if tc.tested {
					measured = "a"
				}
				if tc.qualify || tc.tested {
					p := clients[measured]
					if p == nil || p.syncCalls.Load() == 0 || p.streamCalls.Load() == 0 {
						t.Fatal("generation failure was not measured")
					}
				} else {
					for _, p := range clients {
						if p.syncCalls.Load() != 0 || p.streamCalls.Load() != 0 {
							t.Fatal("unqualified models were probed")
						}
					}
				}
				raw, err = os.ReadFile("scoreboard.json")
				if err != nil {
					t.Fatal(err)
				}
				var got scoreboard.Score
				if err := json.Unmarshal(raw, &got); err != nil {
					t.Fatal(err)
				}
				wantModels := []string{"a", "new"}
				if tc.tested {
					wantModels = []string{"a"}
				}
				if len(got.Scenarios) != 1 || !got.Scenarios[0].Untested() || got.Scenarios[0].Comments != "pending" || !slices.Equal(got.Scenarios[0].Models, wantModels) {
					t.Fatalf("group and notes lost: %s", raw)
				}
			})
		}
	})
}

type qualificationRunCase struct {
	name   string
	update bool
}

type qualificationProvider struct {
	scoreboardProvider
	model           string
	score           scoreboard.Score
	output          genai.Modalities
	dimensions      []int
	syncCalls       atomic.Int64
	streamCalls     atomic.Int64
	generationError error
}

func (p *qualificationProvider) ModelID() string                    { return p.model }
func (p *qualificationProvider) Scoreboard() scoreboard.Score       { return p.score }
func (p *qualificationProvider) OutputModalities() genai.Modalities { return p.output }
func (p *qualificationProvider) GenSync(ctx context.Context, msgs genai.Messages, opts ...genai.GenOption) (genai.Result, error) {
	p.syncCalls.Add(1)
	if p.generationError != nil {
		return genai.Result{}, p.generationError
	}
	return p.scoreboardProvider.GenSync(ctx, msgs, opts...)
}
func (p *qualificationProvider) GenStream(ctx context.Context, msgs genai.Messages, opts ...genai.GenOption) (iter.Seq[genai.Reply], func() (genai.Result, error)) {
	p.streamCalls.Add(1)
	if p.generationError != nil {
		return func(func(genai.Reply) bool) {}, func() (genai.Result, error) { return genai.Result{}, p.generationError }
	}
	return p.scoreboardProvider.GenStream(ctx, msgs, opts...)
}
func (p *qualificationProvider) Embed(_ context.Context, in *genai.EmbeddingRequest) (*genai.EmbeddingResponse, error) {
	p.dimensions = append(p.dimensions, in.Dimensions)
	n := in.Dimensions
	if n == 0 {
		n = 2
	}
	out := &genai.EmbeddingResponse{Usage: genai.Usage{InputTokens: 7, TotalTokens: 7}}
	for _, text := range in.Inputs {
		v := make([]float32, n)
		switch {
		case strings.Contains(text.Text, "query:"):
			v[0] = 2
		case strings.Contains(text.Text, "charged particles"):
			v[0] = 3
			v[1] = 1
		default:
			v[0] = -1
			v[1] = 2
		}
		out.Embeddings = append(out.Embeddings, v)
	}
	return out, nil
}

// qualificationNotesCase exercises comment ownership across skipped and failed measurements.
type qualificationNotesCase struct {
	name    string
	qualify bool
	tested  bool
}
