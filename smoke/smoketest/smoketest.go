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
// Realtime WebSocket sessions record one fixture per session under
// testdata/TestClient/Scoreboard/<model>/Realtime-N.ndjson. Normal tests replay
// them; -update-scoreboard replaces them.
//
// When a model is no longer being tested (e.g., the provider stopped supporting it), the
// corresponding HTTP recordings in testdata/TestClient/Scoreboard/<model>/ are automatically
// deleted to keep the repository clean and remove stale test data. Recordings (.ndjson) that no
// probe consumed are reported at the end of a full run; remove them or restore the probe.
// deleted to keep the repository clean and remove stale test data.
package smoketest

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"flag"
	"io/fs"
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

// ProviderFactory creates a client to assert the scoreboard with recorded HTTP requests.
//
// The factory must register client cleanup with t.
type ProviderFactory func(t testing.TB, m scoreboard.Model, fn func(http.RoundTripper) http.RoundTripper) genai.Provider

var updateScoreboard = flag.Bool("update-scoreboard", false, "Update scoreboard.json for each provider with current test results")

// RunOptions configures optional behavior of Run.
type RunOptions struct {
	// ScoreboardFile overrides the default scoreboard filename ("scoreboard.json").
	ScoreboardFile string
	// Qualify lists untested models that update mode should smoke test. Model output
	// selects the probes. It has no effect unless -update-scoreboard is set.
	Qualify []scoreboard.Model
	// TolerateReasoning lists model name substrings for which reasoning
	// content in non-thinking scenarios is tolerated instead of causing a test
	// failure. Some models (e.g. Gemma 4) emit reasoning even with
	// enable_thinking=false.
	TolerateReasoning []string
}

func (r *RunOptions) qualifies(m scoreboard.Model) bool {
	return *updateScoreboard && slices.Contains(r.Qualify, m)
}

// UpdatingScoreboard reports whether -update-scoreboard is set.
func UpdatingScoreboard() bool {
	return *updateScoreboard
}

// modelSource identifies a model and whether it comes from a realtime session.
// The reasoning mode stays out of it: untested models consolidate across
// reasoning modes.
type modelSource struct {
	model    string
	realtime bool
}

func sourceOf(m string, sc *scoreboard.Scenario) modelSource {
	return modelSource{model: m, realtime: sc.Realtime != nil}
}

// Run regenerates the scoreboard and asserts it is up to date.
//
// If the -update-scoreboard flag is set, it will update the scoreboard files
// and automatically delete stale recordings for models that are no longer available.
// After a full run, it reports recordings that no probe consumed.
func Run(t *testing.T, pf ProviderFactory, models []scoreboard.Model, rec *myrecorder.Records, opts *RunOptions) {
	if opts == nil {
		opts = &RunOptions{}
	}
	scoreboardFile := opts.ScoreboardFile
	if scoreboardFile == "" {
		scoreboardFile = "scoreboard.json"
	}

	if len(models) == 0 {
		t.Fatal("no models")
	}

	filtered := false
	flag.Visit(func(f *flag.Flag) {
		if f.Name == "test.run" {
			filtered = true
		}
	})
	// A model that holds several scenarios is listed once per scenario.
	seen := map[scoreboard.Model]struct{}{}
	unique := make([]scoreboard.Model, 0, len(models))
	for _, m := range models {
		if _, ok := seen[m]; ok {
			continue
		}
		seen[m] = struct{}{}
		unique = append(unique, m)
	}
	models = unique
	runName := t.Name()
	recordingsDir := filepath.Join("testdata", filepath.FromSlash(t.Name()))
	t.Run("Stale Recordings", func(t *testing.T) {
		// Delete orphaned recordings that don't correspond to any model in the list of models to test.
		deleteOrphanedRecordings(t, recordingsDir, seen)
	})

	cc := pf(t, scoreboard.Model{}, nil)
	sb := cc.Scoreboard()
	if err := sb.Validate(); err != nil {
		t.Fatal(err)
	}
	allScoreboardModels := map[scoreboard.Key]struct{}{}
	for i := range sb.Scenarios {
		sc := &sb.Scenarios[i]
		if len(sc.Models) == 0 {
			t.Fatalf("scenario #%d has no models", i)
		}
		for _, m := range sc.Models {
			allScoreboardModels[sc.Key(m)] = struct{}{}
		}
	}
	usage := genai.Usage{}
	updatedScenarios := []scoreboard.Scenario{}
	failed := false
	for _, m := range models {
		if !t.Run(m.String(), func(t *testing.T) {
			// A model holds one scenario per probe family, so collect every scenario
			// that names it: a realtime session and its GenSync/GenStream counterpart
			// coexist.
			runs := make([]scoreboard.Scenario, 0, 2)
			kept := make([]scoreboard.Scenario, 0, 1)
			for i := range sb.Scenarios {
				sc := &sb.Scenarios[i]
				if sc.Reason != m.Reason || !slices.Contains(sc.Models, m.Model) {
					continue
				}
				if sc.Models[0] != m.Model && (!sc.Untested() || !opts.qualifies(m)) {
					// Only the first model of a scenario runs, for cost savings. Create
					// one scenario per model to smoke test the others.
					t.Logf("Skipping scenario: only %s runs", sc.Models[0])
					continue
				}
				want := *sc
				want.Models = []string{m.Model}
				if want.Untested() && !opts.qualifies(m) {
					// Keep the placeholder so the scoreboard still lists the model.
					kept = append(kept, want)
					continue
				}
				runs = append(runs, want)
			}
			if len(runs) == 0 && len(kept) == 0 {
				// The model has no scenario yet.
				want := scoreboard.Scenario{Models: []string{m.Model}, Reason: m.Reason}
				if !opts.qualifies(m) {
					for i := range sb.Scenarios {
						sc := &sb.Scenarios[i]
						if sc.Reason == m.Reason && sc.Untested() {
							want.Comments = sc.Comments
							break
						}
					}
					kept = append(kept, want)
				} else {
					runs = append(runs, want)
				}
			}
			updatedScenarios = append(updatedScenarios, kept...)
			newProvider := func(t testing.TB, sn string) genai.Provider {
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
			}
			tolerate := !m.Reason && slices.ContainsFunc(opts.TolerateReasoning, func(s string) bool {
				return strings.Contains(m.Model, s)
			})
			if len(runs) == 0 {
				if len(kept) != 0 {
					t.Skip("Explicitly unsupported model")
				}
				t.Skip("Only run first model in scenario for cost savings")
			}
			u, got := runOneModel(t, newProvider, m, runs, tolerate)
			usage.Add(&u)
			updatedScenarios = append(updatedScenarios, got...)
		}) {
			failed = true
		}
	}
	t.Logf("Usage: %#v", usage)
	// A failed model has no generated scenario. Do not replace the complete
	// scoreboard with the partial successes from a live refresh.
	if failed {
		return
	}

	staleModels := map[scoreboard.Key]struct{}{}
	if !filtered {
		// Only mark models as truly stale on complete test runs (no filtering)
		// Check for models in scoreboard that are not in discovered models
		// These are truly stale (removed from the API).
		// Untested scenarios are skipped since they intentionally have no test results.
		for key := range allScoreboardModels {
			if _, ok := seen[scoreboard.Model{Model: key.Model, Reason: key.Reason}]; ok {
				continue
			}
			// Untested scenarios intentionally have no test results.
			isUntested := false
			for i := range sb.Scenarios {
				sc := &sb.Scenarios[i]
				if sc.Key(key.Model) == key && sc.Untested() {
					isUntested = true
					break
				}
			}
			if !isUntested {
				staleModels[key] = struct{}{}
			}
			if !*updateScoreboard && !isUntested {
				t.Errorf("stale model in scoreboard: %v", key)
			}
		}
	}
	// Check scoreboard and update if requested
	if len(updatedScenarios) > 0 || len(staleModels) > 0 {
		scoreboardPath := filepath.Join(".", scoreboardFile)
		rawOld, rawNew := generateUpdatedScoreboard(t, scoreboardPath, updatedScenarios, slices.Collect(maps.Keys(staleModels)), filtered)
		if !bytes.Equal(rawNew, rawOld) {
			if !*updateScoreboard {
				t.Fatalf("%s is out of date, run with -update-scoreboard to update it", scoreboardFile)
			}
			t.Logf("Updating %s", scoreboardPath)
			if err := os.WriteFile(scoreboardPath, rawNew, 0o644); err != nil {
				t.Fatalf("failed to write scoreboard.json: %v", err)
			}
		} else {
			t.Logf("No changes to %s", scoreboardPath)
		}
	}
	if !filtered {
		t.Run("Unexpected Recordings", func(t *testing.T) {
			extra, err := unexpectedRecordings(recordingsDir, internaltest.TakeRecordings(runName))
			if err != nil {
				t.Fatalf("failed to find unexpected recordings: %v", err)
			}
			if len(extra) != 0 {
				t.Errorf("unexpected recordings:\n  %s", strings.Join(extra, "\n  "))
			}
		})
	}
}

// getClientOneModel returns a provider client for a specific model.
type getClientOneModel func(t testing.TB, scenarioName string) genai.Provider

// runOneModel runs the scoreboard on one model.
//
// When tolerateReasoning is true, reasoning content discovered in
// non-thinking scenarios is silently ignored instead of causing a mismatch.
// This is useful for models like Gemma 4 that emit reasoning even with
// enable_thinking=false.
//
// Returns the usage and the generated scenario (or nil if not updated).
func runOneModel(t testing.TB, gc getClientOneModel, m scoreboard.Model, wants []scoreboard.Scenario, tolerateReasoning bool) (genai.Usage, []scoreboard.Scenario) {
	providerFactory := func(name string) genai.Provider {
		if name == "" {
			return gc(t, name)
		}
		return gc(t, t.Name()+"/"+name)
	}
	ctx, _ := internaltest.Log(t)
	usage := genai.Usage{}
	got := make([]scoreboard.Scenario, 0, len(wants)+1)
	hasRealtime := false
	for i := range wants {
		// Run one scenario at a time otherwise we can't collect the total usage.
		u, sc := measureScenario(t, ctx, providerFactory, &wants[i], tolerateReasoning)
		usage.Add(&u)
		hasRealtime = hasRealtime || wants[i].Realtime != nil
		if sc != nil {
			got = append(got, *sc)
		}
	}
	// A model may serve both families, and only a session tells whether it accepts
	// one. The session is recorded either way, so replay confirms the answer.
	// Realtime returns speech or text, so an image or embedding model is out of
	// scope even though the provider serves sessions.
	if !hasRealtime && servesSpeechOrText(got) {
		c := providerFactory("")
		if c.Capabilities().Realtime {
			ok, reason, err := smoke.RealtimeSupported(ctx, c)
			if err != nil {
				t.Fatal(err)
			}
			if !ok {
				t.Logf("realtime unsupported: %s", reason)
			} else if sc, u, err2 := smoke.RunRealtimeScenario(ctx, c, m.Reason); err2 != nil {
				// A model that accepts a session and then fails the probes stays untested,
				// and it does not abort the run for the models after it.
				t.Logf("realtime probes failed: %v", err2)
			} else {
				if tolerateReasoning {
					sc.Reason = false
				}
				usage.Add(&u)
				got = append(got, sc)
			}
		}
	}
	return usage, got
}

// measureScenario runs the probes of one scenario and compares the measurement.
func measureScenario(t testing.TB, ctx context.Context, providerFactory func(string) genai.Provider, want *scoreboard.Scenario, tolerateReasoning bool) (genai.Usage, *scoreboard.Scenario) {
	var got scoreboard.Scenario
	var usage genai.Usage
	var err error
	// A model may be scored for both families, so pick the one this scenario names.
	if want.Realtime != nil {
		got, usage, err = smoke.RunRealtimeScenario(ctx, providerFactory(""), want.Reason)
	} else {
		got, usage, err = smoke.RunGenai(ctx, providerFactory)
	}
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
		// Some models leak reasoning content even when thinking is disabled
		// (e.g. Gemma 4). When opted in, tolerate this.
		if tolerateReasoning {
			got.Reason = false
		}
	} else {
		want.ReasoningTokenStart = ""
		want.ReasoningTokenEnd = ""
	}
	if diff := cmp.Diff(*want, got, optScenario); diff != "" && !*updateScoreboard {
		t.Errorf("mismatch (-want +got):\n%s", diff)
	}
	// Preserve Reason from want when got is untested, so the scoreboard
	// update can match the old scenario key correctly.
	if got.Untested() && want.Reason {
		got.Reason = true
	}
	return usage, &got
}

//

var optScenario = cmpopts.IgnoreFields(scoreboard.Scenario{}, "Comments", "SOTA", "Good", "Cheap", "ReasoningTokenStart", "ReasoningTokenEnd")

// deleteOrphanedRecordings recursively checks and deletes recordings that don't correspond to any
// scoreboard model.
func deleteOrphanedRecordings(t testing.TB, dir string, scoreboardModels map[scoreboard.Model]struct{}) {
	if _, err := os.Stat(dir); errors.Is(err, fs.ErrNotExist) {
		return
	}
	// 1. Gather all the directories in the recordings directory.
	var allDirs []string
	if err := filepath.Walk(dir, func(path string, info os.FileInfo, err error) error {
		if err == nil && info.IsDir() && path != dir {
			relPath, _ := filepath.Rel(dir, path)
			allDirs = append(allDirs, strings.ReplaceAll(relPath, "\\", "/"))
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
			// allDirs entries were normalized to use '/' separators above, so
			// compare with '/' to be platform independent.
			if other != d && strings.HasPrefix(other, d+"/") {
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
			// dirName uses '/' separators; convert to OS-specific before joining.
			target := filepath.Join(dir, filepath.FromSlash(dirName))
			if err := os.RemoveAll(target); err != nil {
				t.Fatalf("failed to delete orphaned recordings for %s: %v", dirName, err)
			}
		}
	}
}

// generateUpdatedScoreboard regenerates the scoreboard from scratch in one pass.
//
// It merges tested scenarios with the existing scoreboard to preserve metadata,
// removes stale models, and sorts scenarios so SOTA/Good/Cheap appear first.
// Returns the old and new scoreboard JSON bytes.
func generateUpdatedScoreboard(t testing.TB, scoreboardPath string, scenarios []scoreboard.Scenario, staleModels []scoreboard.Key, filtered bool) (rawOld, rawNew []byte) {
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
	oldScenarios := make(map[scoreboard.Key]*scoreboard.Scenario)
	for i := range oldScore.Scenarios {
		sc := &oldScore.Scenarios[i]
		for _, m := range sc.Models {
			oldScenarios[sc.Key(m)] = sc
		}
	}

	// Mark stale models for removal
	staleSet := make(map[scoreboard.Key]struct{})
	for _, m := range staleModels {
		staleSet[m] = struct{}{}
	}

	// Regenerate scenarios in a single pass
	result := make([]scoreboard.Scenario, 0, len(scenarios)+len(oldScore.Scenarios))
	seenPairs := make(map[scoreboard.Key]struct{})
	usedOldScenarios := make(map[*scoreboard.Scenario]struct{})

	// First pass: add tested scenarios, preserving metadata from old scenarios
	for i := range scenarios {
		newSc := &scenarios[i]
		if len(newSc.Models) == 0 {
			continue
		}
		// Skip untested scenarios - they'll be handled in the third pass.
		// But mark them as seen so the second pass doesn't re-add the old
		// tested scenario for the same model.
		if newSc.Untested() {
			if len(newSc.Models) > 0 {
				seenPairs[newSc.Key(newSc.Models[0])] = struct{}{}
			}
			continue
		}
		model := newSc.Models[0]
		key := newSc.Key(model)

		// Skip if already seen
		if _, seen := seenPairs[key]; seen {
			continue
		}
		seenPairs[key] = struct{}{}

		// Use old metadata if available, otherwise use new scenario
		sc := *newSc
		if oldSc, found := oldScenarios[key]; found {
			// Preserve metadata from old scenario
			if oldSc.Untested() && sc.Embed != nil {
				sc.Comments = ""
			} else {
				sc.Comments = oldSc.Comments
			}
			sc.ReasoningTokenStart = oldSc.ReasoningTokenStart
			sc.ReasoningTokenEnd = oldSc.ReasoningTokenEnd
			// Qualification measures one model; the third pass retains untested siblings.
			// Existing tested groups can still share their previous capability scores.
			if oldSc.Untested() {
				sc.Models = []string{model}
			} else if _, used := usedOldScenarios[oldSc]; !used {
				sc.Models = oldSc.Models
				// New measurement is evidence only for this model, not grouped
				// siblings that shared existing capability scores.
				outOfOrderChanged := func(a, b *scoreboard.Functionality) bool {
					if a == nil || b == nil {
						return false
					}
					return !cmp.Equal(a.OutOfOrder, b.OutOfOrder)
				}
				if len(oldSc.Models) > 1 && (sc.Embed != nil && (!cmp.Equal(oldSc.Embed, sc.Embed) || !cmp.Equal(oldSc.In, sc.In)) || outOfOrderChanged(oldSc.GenSync, sc.GenSync) || outOfOrderChanged(oldSc.GenStream, sc.GenStream)) {
					rest := *oldSc
					rest.Models = slices.DeleteFunc(slices.Clone(oldSc.Models), func(id string) bool { return id == model })
					rest.SOTA, rest.Good, rest.Cheap = false, false, false
					result = append(result, rest)
					sc.Models = []string{model}
				}
				usedOldScenarios[oldSc] = struct{}{}
				// Mark all models in this old scenario as seen to avoid processing them again
				for _, m := range oldSc.Models {
					seenPairs[oldSc.Key(m)] = struct{}{}
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
			if _, isStale := staleSet[sc.Key(m)]; !isStale {
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
	for i := range oldScore.Scenarios {
		oldSc := oldScore.Scenarios[i]
		// Skip if this scenario was already processed as tested
		if _, seen := seenPairs[oldSc.Key(oldSc.Models[0])]; seen {
			continue
		}

		// Skip untested scenarios for now
		if oldSc.Untested() {
			continue
		}

		// Remove stale models from the old scenario
		remainingModels := make([]string, 0, len(oldSc.Models))
		for _, m := range oldSc.Models {
			if _, isStale := staleSet[oldSc.Key(m)]; !isStale {
				remainingModels = append(remainingModels, m)
			} else {
				t.Logf("Removing stale model %q with reason=%v", m, oldSc.Reason)
			}
		}
		if len(remainingModels) == 0 {
			continue
		}
		oldSc.Models = remainingModels

		// Include tested scenarios but clear preference flags since they weren't re-tested
		if !filtered {
			oldSc.SOTA = false
			oldSc.Good = false
			oldSc.Cheap = false
		}
		result = append(result, oldSc)
	}

	// Collect all model names from tested scenarios to avoid including them
	// in untested scenarios (a model should only appear once in the doc).
	testedModels := map[modelSource]struct{}{}
	for i := range result {
		for _, m := range result[i].Models {
			testedModels[sourceOf(m, &result[i])] = struct{}{}
		}
	}

	// Third pass: consolidate untested scenarios by comments/reason
	// Include both newly added untested scenarios and old ones
	allUntested := make([]scoreboard.Scenario, 0)

	// Track which old scenarios have models represented in the new scenarios
	// to avoid adding the same scenario twice
	newScenarioModels := make(map[scoreboard.Key]struct{})

	// First, collect all untested scenarios from new results, removing stale
	// models and models already present in tested scenarios.
	for i := range scenarios {
		sc := scenarios[i]
		if len(sc.Models) > 0 && sc.Untested() {
			if oldSc := oldScenarios[sc.Key(sc.Models[0])]; oldSc != nil {
				sc.Comments = oldSc.Comments
			}
			// Remove stale models
			remainingModels := make([]string, 0, len(sc.Models))
			for _, m := range sc.Models {
				modelKey := sc.Key(m)
				if _, isStale := staleSet[modelKey]; !isStale {
					if _, isTested := testedModels[sourceOf(m, &sc)]; !isTested {
						remainingModels = append(remainingModels, m)
						newScenarioModels[modelKey] = struct{}{}
					}
				}
			}
			if len(remainingModels) > 0 {
				sc.Models = remainingModels
				allUntested = append(allUntested, sc)
			}
		}
	}

	// Then, collect untested scenarios from old scoreboard, but skip any models
	// that were already added from new scenarios to avoid duplicates
	for i := range oldScore.Scenarios {
		oldSc := &oldScore.Scenarios[i]
		// Only process untested scenarios
		if !oldSc.Untested() {
			continue
		}

		// Remove stale models, models already in new scenarios, and
		// models already present in tested scenarios.
		remainingModels := make([]string, 0, len(oldSc.Models))
		for _, m := range oldSc.Models {
			modelKey := oldSc.Key(m)
			if _, isStale := staleSet[modelKey]; !isStale {
				if _, inNew := newScenarioModels[modelKey]; !inNew {
					if _, isTested := testedModels[sourceOf(m, oldSc)]; !isTested {
						remainingModels = append(remainingModels, m)
					}
				}
			}
		}

		if len(remainingModels) > 0 {
			// Create a copy with non-stale models for consolidation
			scCopy := *oldSc
			scCopy.Models = remainingModels
			allUntested = append(allUntested, scCopy)
		}
	}

	// Deduplicate models across untested scenarios: a model should only
	// appear in one untested scenario. Prefer the non-reasoning scenario
	// first to avoid speculative 🧠 markers.
	seenUntestedModels := map[modelSource]struct{}{}
	for _, reason := range []bool{false, true} {
		for i := range allUntested {
			if allUntested[i].Reason != reason {
				continue
			}
			deduped := make([]string, 0, len(allUntested[i].Models))
			for _, m := range allUntested[i].Models {
				key := sourceOf(m, &allUntested[i])
				if _, seen := seenUntestedModels[key]; !seen {
					seenUntestedModels[key] = struct{}{}
					deduped = append(deduped, m)
				}
			}
			allUntested[i].Models = deduped
		}
	}

	// Consolidate untested by comments/reason. Filter out empty scenarios
	// (all models were deduplicated into another scenario).
	nonEmpty := make([]scoreboard.Scenario, 0, len(allUntested))
	for i := range allUntested {
		sc := &allUntested[i]
		if len(sc.Models) > 0 {
			nonEmpty = append(nonEmpty, *sc)
		}
	}
	allUntested = nonEmpty
	untestedConsolidated := scoreboard.ConsolidateUntestedScenarios(allUntested)
	result = append(result, untestedConsolidated...)

	// Validate and write
	newScore := oldScore
	newScore.Scenarios = result
	newScore.SortScenarios()
	if err = newScore.Validate(); err != nil {
		t.Fatalf("failed to validate scoreboard: %v", err)
	}

	var err2 error
	rawNew, err2 = json.MarshalIndent(newScore, "", "  ")
	if err2 != nil {
		t.Fatalf("failed to encode scoreboard: %v", err2)
	}
	rawNew = append(rawNew, '\n')
	return rawOld, rawNew
}

// unexpectedRecordings returns recordings that the run did not consume.
func unexpectedRecordings(dir string, used map[string]struct{}) ([]string, error) {
	if _, err := os.Stat(dir); errors.Is(err, fs.ErrNotExist) {
		return nil, nil
	}
	var extra []string
	err := filepath.WalkDir(dir, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() || !strings.HasSuffix(path, ".ndjson") {
			return nil
		}
		abs, err := filepath.Abs(path)
		if err != nil {
			return err
		}
		if _, ok := used[abs]; !ok {
			rel, err := filepath.Rel(dir, path)
			if err != nil {
				return err
			}
			extra = append(extra, filepath.ToSlash(rel))
		}
		return nil
	})
	slices.Sort(extra)
	return extra, err
}

// servesSpeechOrText reports whether a measured scenario of the model returns
// speech or text, the two outputs a realtime session produces.
func servesSpeechOrText(scenarios []scoreboard.Scenario) bool {
	for i := range scenarios {
		for mod := range scenarios[i].Out {
			if mod == genai.ModalityText || mod == genai.ModalityAudio {
				return true
			}
		}
	}
	return false
}
