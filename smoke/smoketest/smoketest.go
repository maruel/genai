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

// Model is a model to test. It specifies if the model should run in "reasoning mode" or not. Most models only
// support one or the other but a few support both. Often their functionality is different depending if
// reasoning is enabled or not.
type Model struct {
	Model  string
	Reason bool
}

func (m *Model) String() string {
	if m.Reason {
		// I may change this later.
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
			modelsToTest[Model{Model: model, Reason: sc.Reason}] = struct{}{}
		}
	}

	for _, m := range models {
		t.Run(m.String(), func(t *testing.T) {
			// Find the reference.
			var want scoreboard.Scenario
			for _, sc := range sb.Scenarios {
				if m.Reason != sc.Reason {
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

	// Check if the order is SOTA, Good, Cheap then the rest.
	// HACK FOR llamacpp; there's only one model and the reported ID and the scoreboard name do not match.
	if len(sb.Scenarios) > 1 {
		// It'd be nicer to use PreloadedModels. In practice most do already.
		name := "sota"
		fn := func(h http.RoundTripper) http.RoundTripper {
			r, err2 := rec.Record(name, h)
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
		sota := pf(t, Model{Model: genai.ModelSOTA}, fn).ModelID()
		name = "good"
		good := pf(t, Model{Model: genai.ModelGood}, fn).ModelID()
		name = "cheap"
		cheap := pf(t, Model{Model: genai.ModelCheap}, fn).ModelID()
		// Some models support both reasoning and non-reasoning. We want to keep them aside, otherwise the table
		// looks weird. But we skip the duplicate row.
		// TODO: Have a way to query from the Provider if the currently selected model supports reasoning.

		i := 0
		if sb.Scenarios[i].Models[0] != sota {
			t.Errorf("SOTA should be first: %q", sota)
		} else if !sb.Scenarios[i].SOTA {
			t.Errorf("SOTA should be true: %q", sota)
		}
		// There's two options, the SOTA model is also Good, or there's a duplicate of the first row for
		// reasoning/non-reasoning.
		if sota == good {
			if !sb.Scenarios[i].Good {
				t.Errorf("Good should be true: %q", good)
			}
			if good == cheap {
				if !sb.Scenarios[i].Cheap {
					t.Errorf("Cheap should be true: %q", cheap)
				}
			} else {
				i++
				if sb.Scenarios[i].Models[0] == good {
					i++
				}
				if !sb.Scenarios[i].Cheap {
					t.Errorf("Cheap should be true: %q", cheap)
				}
			}
		} else {
			i++
			if sb.Scenarios[i].Models[0] == sota {
				i++
			}
			if sb.Scenarios[i].Models[0] != good {
				t.Errorf("Good should be true: %q", good)
			}
			if good == cheap {
				if !sb.Scenarios[i].Cheap {
					t.Errorf("Cheap should be true: %q", cheap)
				}
			} else {
				i++
				if sb.Scenarios[i].Models[0] == good {
					i++
				}
				if !sb.Scenarios[i].Cheap {
					t.Errorf("Cheap should be true: %q", cheap)
				}
			}
		}

		// TODO: Make sure other models are not marked as SOTA, Good or Cheap!
	}

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
	if !want.Reason {
		if want.ReasoningTokenStart != "" {
			t.Fatal("unexpected ReasoningTokenStart")
		}
		if want.ReasoningTokenEnd != "" {
			t.Fatal("unexpected ReasoningTokenEnd")
		}
	} else {
		want.ReasoningTokenStart = ""
		want.ReasoningTokenEnd = ""
	}
	// Check if valid.
	// optTriState,
	if diff := cmp.Diff(want, got, optScenario); diff != "" {
		t.Errorf("mismatch (-want +got):\n%s", diff)
	}
	return usage
}

//

var optScenario = cmpopts.IgnoreFields(scoreboard.Scenario{}, "Comments", "SOTA", "Good", "Cheap")

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
