// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Package smoketest runs a scoreboard in test mode.
//
// # Automatic Scoreboard Updates
//
// The package supports automatic updates to scoreboard.json files when tests are run with the
// -update-scoreboard flag. This is useful when you want to update the provider's capabilities
// based on the latest test results.
//
// Run your tests with:
//
//	go test ./providers/yourprovider -update-scoreboard
//
// The scoreboard.json file will be automatically updated with the new scenario results while
// preserving existing metadata like Comments, SOTA, Good, and Cheap flags.
//
// When a model is no longer being tested (e.g., the provider stopped supporting it), the
// corresponding HTTP recordings in testdata/TestClient/Scoreboard/<model>/ are automatically
// deleted to keep the repository clean and remove stale test data.
package smoketest

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
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

var updateScoreboard = flag.Bool("update-scoreboard", false, "Update scoreboard.json for each provider with current test results")

// Run regenerates the scoreboard and asserts it is up to date.
// If the -update-scoreboard flag is set, it will update the scoreboard files
// and automatically delete stale recordings for models that are no longer available.
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
	updatedScenarios := []scoreboard.Scenario{}
	var staleScenarios []scoreboard.Scenario

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
			u, got := runOneModel(t, func(t testing.TB, sn string) genai.Provider {
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
			if *updateScoreboard && got != nil {
				updatedScenarios = append(updatedScenarios, *got)
			}
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
				// Track stale models when updating scoreboard
				for _, sc := range sb.Scenarios {
					if len(sc.Models) > 0 && sc.Models[0] == model.Model && sc.Reason == model.Reason {
						staleScenarios = append(staleScenarios, sc)
					}
				}
				if !*updateScoreboard {
					t.Errorf("stale model in scoreboard: %v", model)
				}
			}
		}
	}

	// Update scoreboards if requested
	if *updateScoreboard {
		if len(updatedScenarios) > 0 {
			if err := defaultUpdateScoreboard(t, ".", updatedScenarios); err != nil {
				t.Errorf("failed to update scoreboard: %v", err)
			}
		}
		if len(staleScenarios) > 0 {
			if err := deleteStaleRecordings(t, staleScenarios); err != nil {
				t.Errorf("failed to delete stale recordings: %v", err)
			}
		}
	}
}

// getClientOneModel returns a provider client for a specific model.
type getClientOneModel func(t testing.TB, scenarioName string) genai.Provider

// runOneModel runs the scoreboard on one model.
// Returns the usage and the generated scenario (or nil if not updated).
func runOneModel(t testing.TB, gc getClientOneModel, want scoreboard.Scenario) (genai.Usage, *scoreboard.Scenario) {
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
	return usage, &got
}

//

var optScenario = cmpopts.IgnoreFields(scoreboard.Scenario{}, "Comments", "SOTA", "Good", "Cheap", "ReasoningTokenStart", "ReasoningTokenEnd")

// deleteStaleRecordings removes HTTP recordings for models that are no longer being tested.
func deleteStaleRecordings(t testing.TB, staleScenarios []scoreboard.Scenario) error {
	scoreBoardDir := filepath.Join("testdata", "TestClient", "Scoreboard")
	for _, sc := range staleScenarios {
		if len(sc.Models) == 0 {
			continue
		}
		model := sc.Models[0]
		modelDir := model
		if sc.Reason {
			modelDir += "_thinking"
		}
		path := filepath.Join(scoreBoardDir, modelDir)
		if _, err := os.Stat(path); err == nil {
			t.Logf("Deleting stale model recordings for %s", modelDir)
			if err := os.RemoveAll(path); err != nil {
				return fmt.Errorf("failed to delete stale recordings for %s: %w", modelDir, err)
			}
		}
	}
	return nil
}

// defaultUpdateScoreboard is the default implementation for updating scoreboards.
// It merges the updated scenarios with the existing scoreboard while preserving metadata like comments, SOTA, Good, Cheap flags, and reasoning tokens.
func defaultUpdateScoreboard(t testing.TB, providerDir string, scenarios []scoreboard.Scenario) error {
	scoreboardPath := filepath.Join(providerDir, "scoreboard.json")

	// Read the existing scoreboard
	rawOld, err := os.ReadFile(scoreboardPath)
	if err != nil {
		return fmt.Errorf("failed to read scoreboard.json: %w", err)
	}

	d := json.NewDecoder(bytes.NewReader(rawOld))
	d.DisallowUnknownFields()
	existingScore := scoreboard.Score{}
	if err := d.Decode(&existingScore); err != nil {
		return fmt.Errorf("failed to decode scoreboard.json: %w", err)
	}

	// Create a map of new scenarios by model and reason for quick lookup
	newScenarioMap := make(map[string]map[bool]*scoreboard.Scenario)
	for i := range scenarios {
		s := scenarios[i]
		if len(s.Models) == 0 {
			continue
		}
		model := s.Models[0]
		if newScenarioMap[model] == nil {
			newScenarioMap[model] = make(map[bool]*scoreboard.Scenario)
		}
		newScenarioMap[model][s.Reason] = &scenarios[i]
	}

	// Update existing scenarios with new results, preserving metadata
	for i := range existingScore.Scenarios {
		sc := &existingScore.Scenarios[i]
		if len(sc.Models) == 0 {
			continue
		}
		model := sc.Models[0]
		if updated, ok := newScenarioMap[model][sc.Reason]; ok {
			// Preserve metadata from existing scenario
			comments := sc.Comments
			sota := sc.SOTA
			good := sc.Good
			cheap := sc.Cheap
			reasoningTokenStart := sc.ReasoningTokenStart
			reasoningTokenEnd := sc.ReasoningTokenEnd

			// Replace with updated scenario
			*sc = *updated

			// Restore metadata
			sc.Comments = comments
			sc.SOTA = sota
			sc.Good = good
			sc.Cheap = cheap
			sc.ReasoningTokenStart = reasoningTokenStart
			sc.ReasoningTokenEnd = reasoningTokenEnd
		}
	}

	// Encode the updated scoreboard
	rawNew, err := json.MarshalIndent(existingScore, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to encode scoreboard: %w", err)
	}
	rawNew = append(rawNew, '\n')

	// Write back if changed
	if !bytes.Equal(rawNew, rawOld) {
		t.Logf("Updating %s", scoreboardPath)
		if err = os.WriteFile(scoreboardPath, rawNew, 0o644); err != nil {
			return fmt.Errorf("failed to write scoreboard.json: %w", err)
		}
	} else {
		t.Logf("No changes to %s", scoreboardPath)
	}

	return nil
}

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
