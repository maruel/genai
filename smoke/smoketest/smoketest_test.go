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
	"github.com/maruel/genai/smoke"
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
				oldSc := scoreboard.Embed{Row: scoreboard.Row{Models: []string{"measured", "sibling"}, In: oldIn}, Embed: &scoreboard.EmbeddingFunctionality{Dimensions: 768}}
				path := filepath.Join(t.TempDir(), "scoreboard.json")
				raw, err := json.Marshal(scoreboard.Score{Embed: []scoreboard.Embed{oldSc}})
				if err != nil {
					t.Fatal(err)
				}
				if err := os.WriteFile(path, raw, 0o600); err != nil {
					t.Fatal(err)
				}
				measured := oldSc
				measured.Models = []string{"measured"}
				measured.In = newIn
				_, raw = generateUpdatedScoreboard(t, path, &collected{Embed: []scoreboard.Embed{measured}}, nil, true, []scoreboard.Wire{scoreboard.WireHTTP})
				var got scoreboard.Score
				if err := json.Unmarshal(raw, &got); err != nil {
					t.Fatal(err)
				}
				if len(got.Embed) != 2 {
					t.Fatalf("media change spread to sibling: %+v", got.Embed)
				}
				for _, sc := range got.Embed {
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
		if err := os.WriteFile(path, []byte(`{"embed":[{"comments":"model-specific notes","models":["embedding"],"in":{"text":{"inline":true}},"Embed":{"dimensions":768}}]}`), 0o600); err != nil {
			t.Fatal(err)
		}
		sc := scoreboard.Embed{Row: scoreboard.Row{Models: []string{"embedding"}, In: map[scoreboard.Modality]scoreboard.ModalCapability{scoreboard.ModalityText: {Inline: true}}}, Embed: &scoreboard.EmbeddingFunctionality{Dimensions: 768, RequestedDimensions: new(true)}}
		_, raw := generateUpdatedScoreboard(t, path, &collected{Embed: []scoreboard.Embed{sc}}, nil, true, []scoreboard.Wire{scoreboard.WireHTTP})
		var sb scoreboard.Score
		if err := json.Unmarshal(raw, &sb); err != nil {
			t.Fatal(err)
		}
		if len(sb.Embed) != 1 || sb.Embed[0].Comments != "model-specific notes" {
			t.Fatalf("tested model metadata lost: %+v", sb)
		}
	})
	t.Run("embedding splits untested siblings", func(t *testing.T) {
		path := filepath.Join(t.TempDir(), "scoreboard.json")
		if err := os.WriteFile(path, []byte(`{"country":"US","dashboardURL":"dashboard","chat":[{"models":["chat"],"sota":true,"GenSync":{}}],"embed":[{"comments":"pending","models":["first","embedding","last"]}]}`), 0o600); err != nil {
			t.Fatal(err)
		}
		sc := scoreboard.Embed{Row: scoreboard.Row{Models: []string{"embedding"}, In: map[scoreboard.Modality]scoreboard.ModalCapability{scoreboard.ModalityText: {Inline: true}}}, Embed: &scoreboard.EmbeddingFunctionality{Dimensions: 768, RequestedDimensions: new(false)}}
		_, raw := generateUpdatedScoreboard(t, path, &collected{Embed: []scoreboard.Embed{sc}}, nil, true, []scoreboard.Wire{scoreboard.WireHTTP})
		var sb scoreboard.Score
		if err := json.Unmarshal(raw, &sb); err != nil {
			t.Fatal(err)
		}
		if len(sb.Embed)+len(sb.Chat) != 3 || sb.Country != "US" || sb.DashboardURL != "dashboard" {
			t.Fatalf("metadata %+v", sb)
		}
		for i := range sb.Chat {
			r := &sb.Chat[i]
			if !r.SOTA || r.GenSync == nil {
				t.Fatalf("generation metadata lost: %+v", r)
			}
		}
		for i := range sb.Embed {
			r := &sb.Embed[i]
			switch r.Models[0] {
			case "embedding":
				if r.Embed == nil || r.Embed.Dimensions != 768 || r.Comments != "" {
					t.Fatalf("measurement lost: %+v", r)
				}
			case "first":
				if !r.Untested() || len(r.Models) != 2 || r.Models[1] != "last" || r.Comments != "pending" {
					t.Fatalf("manufactured sibling support: %+v", r)
				}
			default:
				t.Fatalf("unexpected row %+v", r)
			}
		}
	})
	t.Run("OutOfOrder", func(t *testing.T) {
		path := filepath.Join(t.TempDir(), "scoreboard.json")
		old := []byte(`{"country":"Local","dashboardURL":"","chat":[{"models":["first","sibling"],"good":true,"in":{"text":{"inline":true}},"out":{"text":{"inline":true}},"GenSync":{"tools":"true"}}]}`)
		if err := os.WriteFile(path, old, 0o600); err != nil {
			t.Fatal(err)
		}
		sc := scoreboard.Chat{
			Reasoned: scoreboard.Reasoned{Row: scoreboard.Row{
				Models: []string{"first"},
				In:     map[scoreboard.Modality]scoreboard.ModalCapability{scoreboard.ModalityText: {Inline: true}},
			}},
			Out:     map[scoreboard.Modality]scoreboard.ModalCapability{scoreboard.ModalityText: {Inline: true}},
			GenSync: &scoreboard.Functionality{Tools: scoreboard.True, OutOfOrder: new(scoreboard.True)},
		}
		_, raw := generateUpdatedScoreboard(t, path, &collected{Chat: []scoreboard.Chat{sc}}, nil, false, []scoreboard.Wire{scoreboard.WireHTTP})
		var sb scoreboard.Score
		if err := json.Unmarshal(raw, &sb); err != nil {
			t.Fatal(err)
		}
		if len(sb.Chat) != 2 {
			t.Fatalf("measured model consolidated with unknown: %s", raw)
		}
		if !sb.Chat[0].Good || sb.Chat[0].Models[0] != "first" || sb.Chat[0].GenSync.OutOfOrder == nil {
			t.Fatal("lost measured model metadata")
		}
		if sb.Chat[1].Models[0] != "sibling" || sb.Chat[1].GenSync.OutOfOrder != nil || sb.Chat[1].GenSync.Tools != scoreboard.True {
			t.Fatal("manufactured measurement for sibling or lost existing Tools score")
		}
	})
	t.Run("filtered preserves tiers", func(t *testing.T) {
		path := filepath.Join(t.TempDir(), "scoreboard.json")
		old := []byte(`{"country":"US","dashboardURL":"","chat":[{"models":["chat"],"sota":true,"GenSync":{}}],"systemone":[{"models":["decision"],"SystemOne":{}}]}`)
		if err := os.WriteFile(path, old, 0o600); err != nil {
			t.Fatal(err)
		}
		_, raw := generateUpdatedScoreboard(t, path, &collected{SystemOne: []scoreboard.SystemOne{
			{Row: scoreboard.Row{Models: []string{"decision"}, In: map[scoreboard.Modality]scoreboard.ModalCapability{scoreboard.ModalityText: {Inline: true}}}, Out: map[scoreboard.Modality]scoreboard.ModalCapability{scoreboard.ModalityText: {Inline: true}}, SystemOne: &scoreboard.DecisionFunctionality{Noul: true}},
		}}, nil, true, []scoreboard.Wire{scoreboard.WireHTTP})
		var sb scoreboard.Score
		if err := json.Unmarshal(raw, &sb); err != nil {
			t.Fatal(err)
		}
		if len(sb.Chat) != 1 || sb.Chat[0].Models[0] != "chat" || !sb.Chat[0].SOTA || sb.Chat[0].GenSync == nil {
			t.Fatalf("unfiltered model changed: %+v", sb.Chat)
		}
		if len(sb.SystemOne) != 1 || sb.SystemOne[0].SystemOne == nil || !sb.SystemOne[0].SystemOne.Noul {
			t.Fatalf("decision result lost: %+v", sb.SystemOne[0])
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
				sb := scoreboard.Score{
					Chat: []scoreboard.Chat{
						{Reasoned: scoreboard.Reasoned{Row: scoreboard.Row{Models: []string{"generation-first", "generation-qualified", "generation-last"}, Comments: "pending generation models"}}},
					},
					Embed: []scoreboard.Embed{
						{Row: scoreboard.Row{Models: []string{"aaa-first", "embedding", "embedding2", "zzz-last"}, Comments: "pending embedding models"}},
					},
				}
				for i := range sb.Chat {
					slices.Sort(sb.Chat[i].Models)
				}
				for i := range sb.Embed {
					slices.Sort(sb.Embed[i].Models)
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
				_, original := generateUpdatedScoreboard(t, path, &collected{}, nil, true, []scoreboard.Wire{scoreboard.WireHTTP})
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
				for _, id := range append(slices.Clone(sb.Embed[0].Models), sb.Chat[0].Models...) {
					{
						models = append(models, scoreboard.Model{Model: id})
						p := &qualificationProvider{model: id, score: sb, output: genai.Modalities{genai.ModalityText}, realtime: true}
						if id == "embedding" || id == "embedding2" {
							p.output = genai.Modalities{genai.ModalityEmbedding}
						}
						clients[id] = p
					}
				}
				if tc.update {
					models = append(models, scoreboard.Model{Model: "new-embedding"})
					clients["new-embedding"] = &qualificationProvider{model: "new-embedding", score: sb, output: genai.Modalities{genai.ModalityEmbedding}, realtime: true}
				}
				Run(t, smoke.HTTPSession(rec, func(_ testing.TB, m scoreboard.Model, _ func(http.RoundTripper) http.RoundTripper) genai.Provider {
					if m.Model == "" {
						return &qualificationProvider{score: sb}
					}
					return clients[m.Model]
				}), models, &RunOptions{ScoreboardFile: path, Qualify: []scoreboard.Model{{Model: "embedding"}, {Model: "embedding2"}, {Model: "new-embedding"}, {Model: "generation-qualified"}}})
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
				measured := map[string]scoreboard.ScoreRow{}
				var untested []string
				note := func(r scoreboard.ScoreRow) {
					if r.Untested() {
						untested = append(untested, r.Base().Models...)
						if r.Base().Comments == "" {
							t.Fatal("untouched sibling notes were lost")
						}
						return
					}
					if len(r.Base().Models) != 1 {
						t.Fatalf("measured row claims unqualified siblings: %+v", r)
					}
					measured[r.Base().Models[0]] = r
				}
				for i := range got.Embed {
					note(&got.Embed[i])
				}
				for i := range got.Chat {
					note(&got.Chat[i])
				}
				for _, tc := range []struct{ id, comments string }{
					{"embedding", "pending embedding models"},
					{"embedding2", "pending embedding models"},
					// A model with no placeholder inherits no notes.
					{"new-embedding", ""},
				} {
					row := measured[tc.id]
					e, ok := row.(*scoreboard.Embed)
					if !ok {
						t.Fatalf("%q is not an embedding row: %#v", tc.id, row)
					}
					if !e.In[genai.ModalityImage].Inline || !e.In[genai.ModalityAudio].Inline {
						t.Fatalf("media measurements lost in scoreboard: %+v", e)
					}
					if e.Embed == nil || e.Embed.Dimensions != 2 || e.Embed.RequestedDimensions == nil || !*e.Embed.RequestedDimensions || e.Comments != tc.comments {
						t.Fatalf("embedding measurement %q: %+v", tc.id, e)
					}
				}
				if c, ok := measured["generation-qualified"].(*scoreboard.Chat); !ok || c.GenSync == nil || c.GenStream == nil || !c.GenSync.Seed || c.Comments != "pending generation models" {
					t.Fatalf("generation measurement: %#v", measured["generation-qualified"])
				}
				slices.Sort(untested)
				if !slices.Equal(untested, []string{"aaa-first", "generation-first", "generation-last", "zzz-last"}) {
					t.Fatalf("untouched siblings = %v", untested)
				}
				// A measured model that returns speech or text is probed for a session
				// once; a skipped model, and one that returns images or embeddings, stays
				// untouched.
				for id, p := range clients {
					if id == "" {
						continue
					}
					measured := p.syncCalls.Load() != 0 || p.streamCalls.Load() != 0 || len(p.dimensions) != 0
					serves := slices.Contains(p.output, genai.ModalityText) || slices.Contains(p.output, genai.ModalityAudio)
					if probed := p.realtimeCalls.Load(); (probed == 1) != (measured && serves) {
						t.Fatalf("%s: measured=%v serves=%v realtime probes=%d", id, measured, serves, probed)
					}
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
				sc := scoreboard.Chat{Reasoned: scoreboard.Reasoned{Row: scoreboard.Row{Models: []string{"a"}, Comments: "pending", Wire: scoreboard.WireHTTP}}}
				if tc.qualify {
					sc.Models = append(sc.Models, "new")
				}
				if tc.tested {
					sc.GenSync = &scoreboard.Functionality{}
				}
				sb := scoreboard.Score{Chat: []scoreboard.Chat{sc}}
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
				Run(t, smoke.HTTP(rec, func(_ testing.TB, m scoreboard.Model, _ func(http.RoundTripper) http.RoundTripper) genai.Provider {
					return clients[m.Model]
				}), models, opts)
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
				if len(got.Chat) != 1 || !got.Chat[0].Untested() || got.Chat[0].Comments != "pending" || !slices.Equal(got.Chat[0].Models, wantModels) {
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
	realtimeCalls   atomic.Int64
	realtime        bool
	generationError error
}

func (p *qualificationProvider) ModelID() string                    { return p.model }
func (p *qualificationProvider) Scoreboard() scoreboard.Score       { return p.score }
func (p *qualificationProvider) OutputModalities() genai.Modalities { return p.output }
func (p *qualificationProvider) Capabilities() genai.ProviderCapabilities {
	return genai.ProviderCapabilities{Realtime: p.realtime}
}
func (p *qualificationProvider) ConnectRealtime(context.Context, ...genai.RealtimeOption) (genai.RealtimeSession, error) {
	p.realtimeCalls.Add(1)
	return nil, errors.New("model does not support realtime")
}
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

func TestUnexpectedRecordings(t *testing.T) {
	dir := t.TempDir()
	for _, name := range []string{"used.ndjson", "extra.ndjson", "Realtime-1.ndjson", "ignored.yaml", "sub/extra.ndjson"} {
		path := filepath.Join(dir, filepath.FromSlash(name))
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, []byte("{}\n"), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	usedPath := filepath.Join(dir, "used.ndjson")
	used, err := filepath.Abs(usedPath)
	if err != nil {
		t.Fatal(err)
	}
	got, err := unexpectedRecordings(dir, map[string]struct{}{used: {}})
	if err != nil {
		t.Fatal(err)
	}
	want := []string{"Realtime-1.ndjson", "extra.ndjson", "sub/extra.ndjson"}
	if !slices.Equal(got, want) {
		t.Fatalf("unexpected recordings: got %v, want %v", got, want)
	}
	if got, err := unexpectedRecordings(filepath.Join(dir, "missing"), nil); err != nil || got != nil {
		t.Fatalf("missing directory: got %v, %v", got, err)
	}
}
