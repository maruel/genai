// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Package smoketest runs a scoreboard in test mode.
package smoketest

import (
	"flag"
	"net/http"
	"slices"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	"github.com/maruel/genai"
	"github.com/maruel/genai/internal/internaltest"
	"github.com/maruel/genai/internal/myrecorder"
	"github.com/maruel/genai/scoreboard"
	"github.com/maruel/genai/smoke"
)

// Model is a model to test. It specifies if the model should run in "thinking  mode" or not. Most models only
// support one or the other but a few support both. Often their functionality is different depending if
// thinking is enabled or not.
type Model struct {
	Model    string
	Thinking bool
}

func (m *Model) String() string {
	if m.Thinking {
		return m.Model + "_thinking"
	}
	return m.Model
}

// ProviderFactory is the client to assert the scoreboard. It will have the HTTP requests recorded.
type ProviderFactory func(t testing.TB, m Model, fn func(http.RoundTripper) http.RoundTripper) genai.Provider

// Run regenerates the scoreboard and asserts it is up to date.
func Run(t *testing.T, pf ProviderFactory, models []Model, rec *myrecorder.Records) {
	if len(models) == 0 {
		t.Fatal("no models")
	}
	seen := map[Model]struct{}{}
	for _, m := range models {
		if _, ok := seen[m]; ok {
			t.Fatalf("duplicate model in ListModel: %v", m)
		}
		seen[m] = struct{}{}
	}

	usage := genai.Usage{}

	// Find the reference.
	cc := pf(t, Model{Model: genai.ModelNone}, nil)
	sb := cc.Scoreboard()
	if err := sb.Validate(); err != nil {
		t.Fatal(err)
	}

	modelsToTest := map[Model]struct{}{}
	for _, sc := range sb.Scenarios {
		for _, model := range sc.Models {
			modelsToTest[Model{Model: model, Thinking: sc.Thinking}] = struct{}{}
		}
	}

	for _, m := range models {
		t.Run(m.String(), func(t *testing.T) {
			// Find the reference.
			var want scoreboard.Scenario
			for _, sc := range sb.Scenarios {
				if m.Thinking != sc.Thinking {
					continue
				}
				if slices.Contains(sc.Models, m.Model) {
					if sc.Models[0] != m.Model {
						// We only run the first model in the scenario for cost savings purposes. Create one scenario per
						// model to smoke test.
						t.Skip("Only run first model in scenario for cost savings")
					}
					want = sc
					want.Models = []string{m.Model}
					break
				}
			}
			if len(want.Models) == 0 {
				t.Fatalf("no scenario for model %v", m)
			}
			if want.In == nil && want.Out == nil {
				t.Skip("Explicitly unsupported model")
			}

			// Run one model at a time otherwise we can't collect the total usage.
			u := runOneModel(t, func(t testing.TB, sn string) genai.Provider {
				fn := func(h http.RoundTripper) http.RoundTripper {
					if sn == "" {
						return h
					}
					r, err2 := rec.Record(sn, h)
					if err2 != nil {
						t.Fatal(err2)
					}
					t.Cleanup(func() {
						if err3 := r.Stop(); err3 != nil {
							t.Error(err3)
						}
					})
					return r
				}
				return pf(t, m, fn)
			}, want)
			usage.Add(u)
		})
	}
	t.Logf("Usage: %#v", usage)

	// Do this at the end.
	filtered := false
	flag.Visit(func(f *flag.Flag) {
		if f.Name == "test.run" {
			filtered = true
		}
	})
	if !filtered {
		for model := range modelsToTest {
			if _, ok := seen[model]; !ok {
				t.Errorf("stale model in scoreboard: %v", model)
			}
		}
	}
}

// getClientOneModel returns a provider client for a specific model.
type getClientOneModel func(t testing.TB, scenarioName string) genai.Provider

// runOneModel runs the scoreboard on one model.
func runOneModel(t testing.TB, gc getClientOneModel, want scoreboard.Scenario) genai.Usage {
	// Calculate the scenario.
	providerFactory := func(name string) genai.Provider {
		if name == "" {
			return gc(t, name)
		}
		return gc(t, t.Name()+"/"+name)
	}
	ctx, _ := internaltest.Log(t)
	got, usage, err := smoke.Run(ctx, providerFactory)
	t.Logf("Usage: %#v", usage)
	if err != nil {
		t.Fatalf("CreateScenario failed: %v", err)
	}
	if !want.Thinking {
		if want.ThinkingTokenStart != "" {
			t.Fatal("unexpected ThinkingTokenStart")
		}
		if want.ThinkingTokenEnd != "" {
			t.Fatal("unexpected ThinkingTokenEnd")
		}
	} else {
		want.ThinkingTokenStart = ""
		want.ThinkingTokenEnd = ""
	}
	// Check if valid.
	// optTriState,
	if diff := cmp.Diff(want, got, optScenario); diff != "" {
		t.Errorf("mismatch (-want +got):\n%s", diff)
	}
	return usage
}

//

var optScenario = cmpopts.IgnoreFields(scoreboard.Scenario{}, "Comments")

/*
var optTriState = cmp.Comparer(func(x, y scoreboard.TriState) bool {
	// TODO: Make this more solid. This requires a better assessment of what "Flaky" is.
	if x == scoreboard.Flaky || y == scoreboard.Flaky {
		return true
	}
	return x == y
})
*/

// var optFunctionalityText = cmpopts.IgnoreFields(scoreboard.FunctionalityText{}, "Tools", "IndecisiveTool", "BiasedTool")
