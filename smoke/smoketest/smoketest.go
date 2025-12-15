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

// ProviderFactory is the client to assert the scoreboard. It will have the HTTP requests recorded.
type ProviderFactory func(t testing.TB, m scoreboard.Model, fn func(http.RoundTripper) http.RoundTripper) genai.Provider

var updateScoreboard = flag.Bool("update-scoreboard", false, "Update scoreboard.json for each provider with current test results")

// Run regenerates the scoreboard and asserts it is up to date.
//
// If the -update-scoreboard flag is set, it will update the scoreboard files
// and automatically delete stale recordings for models that are no longer available.
func Run(t *testing.T, pf ProviderFactory, models []scoreboard.Model, rec *myrecorder.Records) {
	if len(models) == 0 {
		t.Fatal("no models")
	}

	filtered := false
	flag.Visit(func(f *flag.Flag) {
		if f.Name == "test.run" {
			filtered = true
		}
	})
	if filtered && *updateScoreboard {
		t.Fatal("cannot use -update-scoreboard with -test.run")
	}

	seen := map[scoreboard.Model]struct{}{}
	for _, m := range models {
		if _, ok := seen[m]; ok {
			t.Fatalf("duplicate model in ListModel: %v", m)
		}
		seen[m] = struct{}{}
	}
	t.Run("Stale Recordings", func(t *testing.T) {
		// Delete orphaned recordings that don't correspond to any model in the list of models to test.
		deleteOrphanedRecordings(t, filepath.Join("testdata", "TestClient", "Scoreboard"), seen)
	})

	cc := pf(t, scoreboard.Model{Model: genai.ModelNone}, nil)
	sb := cc.Scoreboard()
	if err := sb.Validate(); err != nil {
		t.Fatal(err)
	}
	modelsToTest := map[scoreboard.Model]struct{}{}
	allScoreboardModels := map[scoreboard.Model]struct{}{}
	for i, sc := range sb.Scenarios {
		if len(sc.Models) == 0 {
			t.Fatalf("scenario #%d has no models", i)
		}
		// Track all models from scenarios for stale detection
		for _, m := range sc.Models {
			allScoreboardModels[scoreboard.Model{Model: m, Reason: sc.Reason}] = struct{}{}
		}
		// Only mark first model as expected to be tested (others skipped for cost savings)
		modelsToTest[scoreboard.Model{Model: sc.Models[0], Reason: sc.Reason}] = struct{}{}
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
				// When updating scoreboard, create a new untested scenario for this model
				want = scoreboard.Scenario{
					Models: []string{m.Model},
					Reason: m.Reason,
				}
			}
			if want.Untested() {
				// If updating scoreboard, collect the untested scenario to be added
				if *updateScoreboard {
					updatedScenarios = append(updatedScenarios, want)
				}
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
			r, err3 := rec.Record(name, h)
			if err3 != nil {
				t.Fatal(err3)
			}
			t.Cleanup(func() {
				if err3 := r.Stop(); err3 != nil {
					t.Error(err3)
				}
			})
			return r
		}
		sota := pf(t, scoreboard.Model{Model: genai.ModelSOTA}, fn).ModelID()
		preferredModels["sota"] = sota
		name = "good"
		good := pf(t, scoreboard.Model{Model: genai.ModelGood}, fn).ModelID()
		preferredModels["good"] = good
		name = "cheap"
		cheap := pf(t, scoreboard.Model{Model: genai.ModelCheap}, fn).ModelID()
		preferredModels["cheap"] = cheap
		// Some models support both reasoning and non-reasoning. We want to keep them aside, otherwise the table
		// looks weird. But we skip the duplicate row.
		// TODO: Have a way to query from the Provider if the currently selected model supports reasoning.

		// Check if the preferred models exist in the scenarios
		// If not, skip validation (they may not have been tested yet)
		sotaInScenarios := false
		for _, sc := range sb.Scenarios {
			if sotaInScenarios = slices.Contains(sc.Models, sota); sotaInScenarios {
				break
			}
		}
		if sotaInScenarios {
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
		} else {
			t.Logf("SOTA model %q not in scenarios, skipping preferred model validation", sota)
		}
	}

	staleModels := map[scoreboard.Model]struct{}{}
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
	if *updateScoreboard && (len(updatedScenarios) > 0 || len(staleModels) > 0) {
		doUpdateScoreboard(t, ".", updatedScenarios, preferredModels, slices.Collect(maps.Keys(staleModels)))
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

// deleteOrphanedRecordings recursively checks and deletes recordings that don't correspond to any
// scoreboard model.
func deleteOrphanedRecordings(t testing.TB, dir string, scoreboardModels map[scoreboard.Model]struct{}) {
	// 1. Gather all the directories in the recordings directory.
	var allDirs []string
	if err := filepath.Walk(dir, func(path string, info os.FileInfo, err error) error {
		if err == nil && info.IsDir() && path != dir {
			relPath, _ := filepath.Rel(dir, path)
			allDirs = append(allDirs, relPath)
		}
		return err
	}); err != nil {
		t.Fatalf("failed to walk directory %s: %v", dir, err)
	}

	// Only keep leaf directories (directories with no subdirectories).
	modelDirs := make([]string, 0, len(allDirs))
	for _, d := range allDirs {
		isLeaf := true
		for _, other := range allDirs {
			if other != d && strings.HasPrefix(other, d+string(filepath.Separator)) {
				isLeaf = false
				break
			}
		}
		if isLeaf {
			modelDirs = append(modelDirs, d)
		}
	}

	// 2. Convert all the models into directory names, with the _thinking suffix.
	expectedDirs := make(map[string]struct{})
	for m := range scoreboardModels {
		expectedDirs[m.String()] = struct{}{}
	}

	// 3. Find the ones that are not expected and delete them.
	for _, dirName := range modelDirs {
		if _, found := expectedDirs[dirName]; !found {
			t.Logf("Deleting orphaned model recordings for %s", dirName)
			if err := os.RemoveAll(filepath.Join(dir, dirName)); err != nil {
				t.Fatalf("failed to delete orphaned recordings for %s: %v", dirName, err)
			}
		}
	}
}

// doUpdateScoreboard regenerates the scoreboard from scratch in one pass.
//
// It merges tested scenarios with the existing scoreboard to preserve metadata,
// removes stale models, and sorts scenarios so SOTA/Good/Cheap appear first.
func doUpdateScoreboard(t testing.TB, providerDir string, scenarios []scoreboard.Scenario, preferredModels map[string]string, staleModels []scoreboard.Model) {
	scoreboardPath := filepath.Join(providerDir, "scoreboard.json")
	rawOld, err := os.ReadFile(scoreboardPath)
	if err != nil {
		t.Fatalf("failed to read scoreboard.json: %v", err)
	}
	d := json.NewDecoder(bytes.NewReader(rawOld))
	d.DisallowUnknownFields()
	oldScore := scoreboard.Score{}
	if err = d.Decode(&oldScore); err != nil {
		t.Fatalf("failed to decode scoreboard.json: %v", err)
	}
	if err = oldScore.Validate(); err != nil {
		t.Fatalf("failed to validate scoreboard.json: %v", err)
	}

	// Build a map of old scenarios for quick lookup and metadata preservation
	oldScenarios := make(map[scoreboard.Model]*scoreboard.Scenario)
	for i := range oldScore.Scenarios {
		sc := &oldScore.Scenarios[i]
		for _, m := range sc.Models {
			oldScenarios[scoreboard.Model{Model: m, Reason: sc.Reason}] = sc
		}
	}

	// Mark stale models for removal
	staleSet := make(map[scoreboard.Model]struct{})
	for _, m := range staleModels {
		staleSet[m] = struct{}{}
	}

	// Regenerate scenarios in a single pass
	result := make([]scoreboard.Scenario, 0, len(scenarios)+len(oldScore.Scenarios))
	seenPairs := make(map[scoreboard.Model]struct{})
	usedOldScenarios := make(map[*scoreboard.Scenario]struct{})

	// First pass: add tested scenarios, preserving metadata from old scenarios
	for _, newSc := range scenarios {
		if len(newSc.Models) == 0 {
			continue
		}
		// Skip untested scenarios - they'll be handled in the third pass
		if newSc.Untested() {
			continue
		}
		model := newSc.Models[0]
		key := scoreboard.Model{Model: model, Reason: newSc.Reason}

		// Skip if already seen
		if _, seen := seenPairs[key]; seen {
			continue
		}
		seenPairs[key] = struct{}{}

		// Use old metadata if available, otherwise use new scenario
		sc := newSc
		if oldSc, found := oldScenarios[key]; found {
			// Preserve metadata from old scenario
			sc.Comments = oldSc.Comments
			sc.ReasoningTokenStart = oldSc.ReasoningTokenStart
			sc.ReasoningTokenEnd = oldSc.ReasoningTokenEnd
			// Only reuse the Models list if we haven't already processed this old scenario
			if _, used := usedOldScenarios[oldSc]; !used {
				sc.Models = oldSc.Models
				usedOldScenarios[oldSc] = struct{}{}
				// Mark all models in this old scenario as seen to avoid processing them again
				for _, m := range oldSc.Models {
					seenPairs[scoreboard.Model{Model: m, Reason: oldSc.Reason}] = struct{}{}
				}
			} else {
				sc.Models = []string{model}
			}
		} else {
			sc.Models = []string{model}
		}

		// Remove stale models from the model list
		remainingModels := make([]string, 0, len(sc.Models))
		for _, m := range sc.Models {
			if _, isStale := staleSet[scoreboard.Model{Model: m, Reason: sc.Reason}]; !isStale {
				remainingModels = append(remainingModels, m)
			} else {
				t.Logf("Removing stale model %q with reason=%v", m, sc.Reason)
			}
		}

		if len(remainingModels) > 0 {
			sc.Models = remainingModels
			// Preserve SOTA/Good/Cheap flags from old scenario only
			var wasSOTA, wasGood, wasCheap bool
			if oldSc, found := oldScenarios[key]; found {
				wasSOTA = oldSc.SOTA
				wasGood = oldSc.Good
				wasCheap = oldSc.Cheap
			}
			// Only preserve flags from old scenario, don't set new ones
			sc.SOTA = wasSOTA
			sc.Good = wasGood
			sc.Cheap = wasCheap
			result = append(result, sc)
		}
	}

	// Second pass: add tested but non-preferred scenarios from old scoreboard
	for _, oldSc := range oldScore.Scenarios {
		// Skip if this scenario was already processed as tested
		if _, seen := seenPairs[scoreboard.Model{Model: oldSc.Models[0], Reason: oldSc.Reason}]; seen {
			continue
		}

		// Skip untested scenarios for now
		if oldSc.Untested() {
			continue
		}

		// Include tested scenarios but clear preference flags since they weren't re-tested
		oldSc.SOTA = false
		oldSc.Good = false
		oldSc.Cheap = false
		result = append(result, oldSc)
	}

	// Third pass: consolidate untested scenarios by comments/reason
	// Include both newly added untested scenarios and old ones
	untestedByKey := make(map[string]*scoreboard.Scenario)
	allUntested := make([]scoreboard.Scenario, 0)

	// First, collect all untested scenarios from both new and old
	for _, sc := range scenarios {
		if len(sc.Models) > 0 && sc.Untested() {
			allUntested = append(allUntested, sc)
		}
	}
	for _, oldSc := range oldScore.Scenarios {
		// Skip if this scenario was already processed
		if _, seen := seenPairs[scoreboard.Model{Model: oldSc.Models[0], Reason: oldSc.Reason}]; seen {
			continue
		}
		// Only process untested scenarios
		if !oldSc.Untested() {
			continue
		}
		allUntested = append(allUntested, oldSc)
	}

	// Now consolidate by comments/reason
	for _, sc := range allUntested {
		// Consolidate with others of same comments/reason
		key := fmt.Sprintf("%s|%v", sc.Comments, sc.Reason)
		if existing, found := untestedByKey[key]; found {
			// Merge models, avoiding duplicates and stale models
			modelSet := make(map[string]struct{})
			for _, m := range existing.Models {
				if _, isStale := staleSet[scoreboard.Model{Model: m, Reason: sc.Reason}]; !isStale {
					modelSet[m] = struct{}{}
				}
			}
			for _, m := range sc.Models {
				if _, isStale := staleSet[scoreboard.Model{Model: m, Reason: sc.Reason}]; !isStale {
					modelSet[m] = struct{}{}
				}
			}
			existing.Models = slices.Sorted(maps.Keys(modelSet))
		} else {
			// New untested scenario: remove stale models
			remainingModels := make([]string, 0, len(sc.Models))
			for _, m := range sc.Models {
				if _, isStale := staleSet[scoreboard.Model{Model: m, Reason: sc.Reason}]; !isStale {
					remainingModels = append(remainingModels, m)
				}
			}
			if len(remainingModels) > 0 {
				sc.Models = remainingModels
				result = append(result, sc)
				untestedByKey[key] = &result[len(result)-1]
			}
		}
	}

	// Sort scenarios by preference flags
	slices.SortFunc(result, scoreboard.CompareScenarios)

	// Validate and write
	newScore := oldScore
	newScore.Scenarios = result
	if err = newScore.Validate(); err != nil {
		t.Fatalf("failed to validate scoreboard: %v", err)
	}

	rawNew, err := json.MarshalIndent(newScore, "", "  ")
	if err != nil {
		t.Fatalf("failed to encode scoreboard: %v", err)
	}
	rawNew = append(rawNew, '\n')

	if !bytes.Equal(rawNew, rawOld) {
		t.Logf("Updating %s", scoreboardPath)
		if err = os.WriteFile(scoreboardPath, rawNew, 0o644); err != nil {
			t.Fatalf("failed to write scoreboard.json: %v", err)
		}
	} else {
		t.Logf("No changes to %s", scoreboardPath)
	}
}
