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
	"maps"
	"net/http"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	"github.com/maruel/genai"
	"github.com/maruel/genai/internal/internaltest"
	"github.com/maruel/genai/internal/myrecorder"
	"github.com/maruel/genai/scoreboard"
	"github.com/maruel/genai/smoke"
)

// Model is a model to test.
//
// It specifies if the model should run in "reasoning mode" or not. Most models only support one or the other
// but a few support both. Often their functionality is different depending if reasoning is enabled or not.
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
//
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

	// Find the reference.
	cc := pf(t, Model{Model: genai.ModelNone}, nil)
	sb := cc.Scoreboard()
	if err := sb.Validate(); err != nil {
		t.Fatal(err)
	}

	modelsToTest := map[Model]struct{}{}
	allScoreboardModels := map[Model]struct{}{}
	for i, sc := range sb.Scenarios {
		if len(sc.Models) == 0 {
			t.Fatalf("scenario #%d has no models", i)
		}
		// Track all models from scenarios for stale detection
		for _, m := range sc.Models {
			allScoreboardModels[Model{Model: m, Reason: sc.Reason}] = struct{}{}
		}
		// Only mark first model as expected to be tested (others skipped for cost savings)
		modelsToTest[Model{Model: sc.Models[0], Reason: sc.Reason}] = struct{}{}
	}

	usage := genai.Usage{}
	updatedScenarios := []scoreboard.Scenario{}
	for _, m := range models {
		t.Run(m.String(), func(t *testing.T) {
			// First try to find exact match with the requested Reason value
			var want scoreboard.Scenario
			for _, sc := range sb.Scenarios {
				if m.Reason == sc.Reason && slices.Contains(sc.Models, m.Model) {
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
				// Model not in scoreboard yet
				if !*updateScoreboard {
					t.Fatalf("no scenario for model %v", m)
				}
				// When updating scoreboard, create a new scenario for this model
				// TODO(maruel): Maybe not test new models by default?
				want = scoreboard.Scenario{
					Models: []string{m.Model},
					Reason: m.Reason,
					In: map[scoreboard.Modality]scoreboard.ModalCapability{
						scoreboard.ModalityText: {Inline: true},
					},
					Out: map[scoreboard.Modality]scoreboard.ModalCapability{
						scoreboard.ModalityText: {Inline: true},
					},
					GenSync:   &scoreboard.Functionality{},
					GenStream: &scoreboard.Functionality{},
				}
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
	preferredModels := map[string]string{} // maps genai.ModelXXX to actual model ID
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
		preferredModels["sota"] = sota
		name = "good"
		good := pf(t, Model{Model: genai.ModelGood}, fn).ModelID()
		preferredModels["good"] = good
		name = "cheap"
		cheap := pf(t, Model{Model: genai.ModelCheap}, fn).ModelID()
		preferredModels["cheap"] = cheap
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
	staleModels := map[Model]struct{}{}
	if !filtered {
		// Only mark models as truly stale on complete test runs (no filtering)
		// Check for models in scoreboard that are not in discovered models
		// These are truly stale (removed from the API).
		for sbModel := range allScoreboardModels {
			if _, ok := seen[sbModel]; !ok {
				for _, sc := range sb.Scenarios {
					for _, m := range sc.Models {
						if m == sbModel.Model && sc.Reason == sbModel.Reason {
							// This model is in scoreboard but not in API
							staleModels[sbModel] = struct{}{}
							break
						}
					}
				}
				if !*updateScoreboard {
					t.Errorf("stale model in scoreboard: %v", sbModel)
				}
			}
		}
	}

	// Update scoreboards if requested
	if *updateScoreboard {
		stale := slices.Collect(maps.Keys(staleModels))
		if len(updatedScenarios) > 0 || len(staleModels) > 0 {
			doUpdateScoreboard(t, ".", updatedScenarios, preferredModels, stale)
		}
		if len(staleModels) > 0 {
			if err := deleteStaleRecordings(t, stale); err != nil {
				t.Errorf("failed to delete stale recordings: %v", err)
			}
		}
		if false {
			// Delete orphaned recordings that don't correspond to any model in the scoreboard.
			// Use allScoreboardModels instead of seen to avoid deleting secondary models
			// that are in the scoreboard but weren't tested in this run (skipped for cost savings).
			deleteOrphanedRecordingsRecForModels(t, filepath.Join("testdata", "TestClient", "Scoreboard"), "", allScoreboardModels)
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
func deleteStaleRecordings(t testing.TB, staleModels []Model) error {
	scoreBoardDir := filepath.Join("testdata", "TestClient", "Scoreboard")
	for _, sc := range staleModels {
		modelDir := strings.ReplaceAll(sc.Model, ":", "-")
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

// deleteOrphanedRecordingsRecForModels recursively checks and deletes recordings that don't correspond to any
// scoreboard model.
func deleteOrphanedRecordingsRecForModels(t testing.TB, dir, prefix string, scoreboardModels map[Model]struct{}) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		if os.IsNotExist(err) {
			return // No recordings directory yet, nothing to clean
		}
		t.Fatalf("failed to read directory %s: %v", dir, err)
	}

	// TODO(maruel): Rewrite to list all the directories, and use that as the list of models.
	for _, entry := range entries {
		if !entry.IsDir() {
			continue
		}
		dirName := entry.Name()
		fullPath := dirName
		if prefix != "" {
			fullPath = prefix + "/" + dirName
		}

		// Check if this directory corresponds to any model in the scoreboard
		// Normalize model names by converting colons to dashes (filesystem convention)
		found := false
		for m := range scoreboardModels {
			normalizedModel := strings.ReplaceAll(m.Model, ":", "-")
			if m.Reason {
				normalizedModel += "_thinking"
			}
			// Exact match
			if fullPath == normalizedModel {
				found = true
				break
			}
			// Check if this directory is a parent of a scoreboard model
			if strings.HasPrefix(normalizedModel, fullPath+"/") {
				found = true
				break
			}
		}
		path := filepath.Join("testdata", "TestClient", "Scoreboard", fullPath)
		if !found {
			t.Logf("Deleting orphaned model recordings for %s", fullPath)
			if err := os.RemoveAll(path); err != nil {
				t.Fatalf("failed to delete orphaned recordings for %s: %v", fullPath, err)
			}
		} else {
			// Recursively check subdirectories for partially matched paths
			deleteOrphanedRecordingsRecForModels(t, path, fullPath, scoreboardModels)
		}
	}
}

// doUpdateScoreboard update a scoreboard.
//
// It merges the updated scenarios with the existing scoreboard while preserving metadata like comments and
// reasoning tokens.
//
// It also updates the SOTA, Good, Cheap flags based on the preferred models.
// Stale scenarios (models no longer being tested) are removed from the scoreboard.
func doUpdateScoreboard(t testing.TB, providerDir string, scenarios []scoreboard.Scenario, preferredModels map[string]string, staleModels []Model) {
	newScenarioMap := map[Model]*scoreboard.Scenario{}
	for i := range scenarios {
		s := scenarios[i]
		for _, m := range s.Models {
			newScenarioMap[Model{m, s.Reason}] = &scenarios[i]
		}
	}

	scoreboardPath := filepath.Join(providerDir, "scoreboard.json")
	rawOld, err := os.ReadFile(scoreboardPath)
	if err != nil {
		t.Fatalf("failed to read scoreboard.json: %v", err)
	}
	d := json.NewDecoder(bytes.NewReader(rawOld))
	d.DisallowUnknownFields()
	existingScore := scoreboard.Score{}
	if err = d.Decode(&existingScore); err != nil {
		t.Fatalf("failed to decode scoreboard.json: %v", err)
	}
	if err = existingScore.Validate(); err != nil {
		t.Fatalf("failed to validate scoreboard.json: %v", err)
	}

	// Update existing scenarios with new results, preserving metadata and model lists
	seenModels := make(map[Model]struct{})
	for i := range existingScore.Scenarios {
		sc := &existingScore.Scenarios[i]
		seenModels[Model{sc.Models[0], sc.Reason}] = struct{}{}
		model := sc.Models[0]
		if updated, ok := newScenarioMap[Model{model, sc.Reason}]; ok {
			// Preserve metadata and original model list from existing scenario
			comments := sc.Comments
			reasoningTokenStart := sc.ReasoningTokenStart
			reasoningTokenEnd := sc.ReasoningTokenEnd
			originalModels := sc.Models
			// Replace with updated scenario (which only has the tested model)
			*sc = *updated
			// Restore metadata and original model list
			sc.Comments = comments
			sc.ReasoningTokenStart = reasoningTokenStart
			sc.ReasoningTokenEnd = reasoningTokenEnd
			sc.Models = originalModels
		}
		// Always update SOTA, Good, Cheap flags based on preferred models, even for untested scenarios.
		// This ensures old flags are cleared when preferred models change.
		sc.SOTA = preferredModels["sota"] == model
		sc.Good = preferredModels["good"] == model
		sc.Cheap = preferredModels["cheap"] == model
	}

	for _, newScenario := range scenarios {
		model := newScenario.Models[0]
		if _, exists := seenModels[Model{model, newScenario.Reason}]; !exists {
			// This is a new model/reason combination, add it to the scoreboard
			existingScore.Scenarios = append(existingScore.Scenarios, newScenario)
		}
	}

	// Remove stale models from scenarios
	if len(staleModels) > 0 {
		staleSet := map[Model]struct{}{}
		for _, sc := range staleModels {
			staleSet[sc] = struct{}{}
		}
		// Filter out stale models from scenarios
		filtered := make([]scoreboard.Scenario, 0, len(existingScore.Scenarios))
		for _, sc := range existingScore.Scenarios {
			if len(sc.Models) == 0 {
				filtered = append(filtered, sc)
				continue
			}
			// Remove stale models from this scenario
			remainingModels := []string{}
			for _, model := range sc.Models {
				if _, isStale := staleSet[Model{model, sc.Reason}]; !isStale {
					remainingModels = append(remainingModels, model)
				} else {
					t.Logf("Removing stale model %q with reason=%v from scenario", model, sc.Reason)
				}
			}
			// Only add scenario back if it still has models
			if len(remainingModels) > 0 {
				sc.Models = remainingModels
				filtered = append(filtered, sc)
			}
		}
		existingScore.Scenarios = filtered
	}

	// Consolidate models with the same comments and reason into a single scenario,
	// but only if they're not currently tested.
	consolidatedScenarios := make([]scoreboard.Scenario, 0, len(existingScore.Scenarios))
	scenariosByKey := make(map[string]*scoreboard.Scenario) // key is "comments|reason"
	for _, sc := range existingScore.Scenarios {
		// Only merge scenarios that are not yet tested
		isUntested := sc.GenSync == nil && sc.GenStream == nil && len(sc.In) == 0 && len(sc.Out) == 0
		if !isUntested {
			// Already tested - don't merge, add as-is
			consolidatedScenarios = append(consolidatedScenarios, sc)
			continue
		}

		// Create a key based on comments and reason for untested scenarios
		key := fmt.Sprintf("%s|%v", sc.Comments, sc.Reason)
		if existing, found := scenariosByKey[key]; found {
			// Merge models, avoiding duplicates
			modelSet := make(map[string]struct{})
			for _, m := range existing.Models {
				modelSet[m] = struct{}{}
			}
			for _, m := range sc.Models {
				modelSet[m] = struct{}{}
			}
			existing.Models = slices.Sorted(maps.Keys(modelSet))
		} else {
			// New untested scenario - add to consolidatedScenarios and track it
			consolidatedScenarios = append(consolidatedScenarios, sc)
			scenariosByKey[key] = &consolidatedScenarios[len(consolidatedScenarios)-1]
		}
	}

	existingScore.Scenarios = consolidatedScenarios

	// Sort scenarios so SOTA, Good, Cheap appear first in the expected order
	// Build a priority map for sorting
	priority := make(map[string]map[bool]int)
	if sota := preferredModels["sota"]; sota != "" {
		if priority[sota] == nil {
			priority[sota] = make(map[bool]int)
		}
		priority[sota][false] = 0
		priority[sota][true] = 0 // SOTA with reasoning also gets priority 0
	}
	if good := preferredModels["good"]; good != "" && good != preferredModels["sota"] {
		if priority[good] == nil {
			priority[good] = make(map[bool]int)
		}
		priority[good][false] = 1
		priority[good][true] = 1
	}
	if cheap := preferredModels["cheap"]; cheap != "" && cheap != preferredModels["sota"] && cheap != preferredModels["good"] {
		if priority[cheap] == nil {
			priority[cheap] = make(map[bool]int)
		}
		priority[cheap][false] = 2
		priority[cheap][true] = 2
	}

	slices.SortFunc(existingScore.Scenarios, func(a, b scoreboard.Scenario) int {
		// Get priority for each scenario (lower = comes first)
		getPriority := func(sc scoreboard.Scenario) int {
			if len(sc.Models) == 0 {
				return 999 // Scenarios with no models go to the end
			}
			model := sc.Models[0]
			if p, ok := priority[model]; ok {
				if pval, ok := p[sc.Reason]; ok {
					return pval
				}
				// If this model has priority but not for this reason value, use a higher priority
				return 100
			}
			return 999 // No priority means it goes to the end
		}
		aPrio := getPriority(a)
		bPrio := getPriority(b)
		if aPrio != bPrio {
			return aPrio - bPrio
		}
		// If same priority, maintain original order
		return 0
	})

	if err := existingScore.Validate(); err != nil {
		t.Fatalf("failed to validate scoreboard: %v", err)
	}

	// Encode the updated scoreboard
	rawNew, err := json.MarshalIndent(existingScore, "", "  ")
	if err != nil {
		t.Fatalf("failed to encode scoreboard: %v", err)
	}
	rawNew = append(rawNew, '\n')

	// Write back if changed
	if !bytes.Equal(rawNew, rawOld) {
		t.Logf("Updating %s", scoreboardPath)
		if err = os.WriteFile(scoreboardPath, rawNew, 0o644); err != nil {
			t.Fatalf("failed to write scoreboard.json: %v", err)
		}
	} else {
		t.Logf("No changes to %s", scoreboardPath)
	}
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
