// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Package smoketest runs a scoreboard in test mode.
//
// It measures every model of a provider through one [smoke.Recorder], fills the
// scoreboard sections, and writes the file back when -update-scoreboard is set.
//
// # Recording
//
// -update-scoreboard replays the cassettes and merges them. Recording needs
// RECORD=failure_only, scoped with -run: it fills the missing fixtures and replays the
// rest. RECORD=all replaces the existing ones and stays off-limits unscoped.
//
//	go test ./providers/yourprovider -update-scoreboard
//
// A -run pattern matches from the first subtest, so -run 'Scoreboard' selects nothing.
// Rewriting a file takes a full package run with -update-scoreboard.
//
// A model that is no longer tested is dropped from the scoreboard, and its HTTP
// recordings under testdata/<test>/Scoreboard/<model>/ are deleted with it. A run
// reports the recordings no probe consumed; remove them or restore the probe.
//
// A nested run owns the recordings below it: the sweep and the unexpected-recording
// check both stop at their own depth, so a second wire can record under the first
// one's directory.
package smoketest

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"flag"
	"io/fs"
	"maps"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"

	"github.com/maruel/genai"
	"github.com/maruel/genai/internal/internaltest"
	"github.com/maruel/genai/scoreboard"
	"github.com/maruel/genai/smoke"
)

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
	// SkipURLMedia skips image, audio, and video URL probes for this provider.
	// Document URL probes still run.
	SkipURLMedia bool
}

func (r *RunOptions) qualifies(m scoreboard.Model) bool {
	return *updateScoreboard && slices.Contains(r.Qualify, m)
}

// UpdatingScoreboard reports whether -update-scoreboard is set.
func UpdatingScoreboard() bool {
	return *updateScoreboard
}

// Run regenerates the scoreboard and asserts it is up to date.
//
// If the -update-scoreboard flag is set, it will update the scoreboard files
// and automatically delete stale recordings for models that are no longer available.
// After a full run, it reports recordings that no probe consumed.
func Run(t *testing.T, rf smoke.RecorderFactory, models []scoreboard.Model, opts *RunOptions) {
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
		if f.Name == "test.run" || f.Name == "test.skip" {
			filtered = true
		}
	})
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
	recordingsDir := filepath.Join("testdata", filepath.FromSlash(runName))
	t.Run("Stale Recordings", func(t *testing.T) {
		deleteOrphanedRecordings(t, recordingsDir, seen)
	})

	// Read the scoreboard through the first wire the recorder serves: a command line
	// provider has no HTTP surface to ask.
	first := rf(t, scoreboard.Model{})
	sb := first.Score()
	if err := sb.Validate(); err != nil {
		t.Fatal(err)
	}
	allKeys := map[scoreboard.Key]struct{}{}
	addKeys(allKeys, sb.Chat)
	addKeys(allKeys, sb.Embed)
	addKeys(allKeys, sb.SystemOne)
	addKeys(allKeys, sb.Realtime)

	usage := genai.Usage{}
	measured := collected{}
	failed := false
	for _, m := range models {
		if !t.Run(m.String(), func(t *testing.T) {
			r := rf(t, m)
			ctx, _ := internaltest.Log(t)
			tolerate := !m.Reason && slices.ContainsFunc(opts.TolerateReasoning, func(s string) bool {
				return strings.Contains(m.Model, s)
			})
			qualify := opts.qualifies(m)
			// A model a section lists without a measurement is queued, not discovered:
			// only an explicit qualification asks the families about it.
			queued := !qualify && isQueuedFor(&sb, m)
			// Each section measures its own rows: a model may hold a chat row and an
			// embedding row at the same time.
			measured.Chat = append(measured.Chat, runSection(ctx, t, r, m, smoke.Chat{SkipURLMedia: opts.SkipURLMedia}, sb.Chat, &sb, qualify, queued, true, tolerate, &usage, nil)...)
			// A family sees what this run already measured for the model.
			view := sb
			view.Chat = append(slices.Clone(sb.Chat), measured.Chat...)
			measured.Embed = append(measured.Embed, runSection(ctx, t, r, m, smoke.Embed{}, sb.Embed, &view, qualify, queued, false, tolerate, &usage, nil)...)
			measured.SystemOne = append(measured.SystemOne, runSection(ctx, t, r, m, smoke.SystemOne{}, sb.SystemOne, &view, qualify, queued, false, tolerate, &usage, nil)...)
			measured.Realtime = append(measured.Realtime, runSection(ctx, t, r, m, smoke.Realtime{}, sb.Realtime, &view, qualify, queued, false, tolerate, &usage, nil)...)
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

	stale := map[scoreboard.Key]struct{}{}
	if !filtered {
		// Only mark models as truly stale on complete test runs (no filtering).
		for key := range allKeys {
			if _, ok := seen[scoreboard.Model{Model: key.Model, Reason: key.Reason}]; ok {
				continue
			}
			if !slices.Contains(first.Wires(), key.Wire) {
				// Another wire owns this row; this run says nothing about it.
				continue
			}
			if isQueued(&sb, key) {
				continue
			}
			stale[key] = struct{}{}
			if !*updateScoreboard {
				t.Errorf("stale model in scoreboard: %v", key)
			}
		}
	}
	if len(measured.total()) > 0 || len(stale) > 0 {
		scoreboardPath := filepath.Join(".", scoreboardFile)
		rawOld, rawNew := generateUpdatedScoreboard(t, scoreboardPath, &measured, slices.Collect(maps.Keys(stale)), filtered, first.Wires())
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
			usage := internaltest.TakeRecordings(runName)
			extra, err := unexpectedRecordings(recordingsDir, usage.Used)
			if err != nil {
				t.Fatalf("failed to find unexpected recordings: %v", err)
			}
			if len(extra) != 0 {
				t.Errorf("unexpected recordings:\n  %s", strings.Join(extra, "\n  "))
			}
			for path := range usage.Missing {
				t.Logf("missing recording: %s", path)
			}
		})
	}
}

// collected holds the rows a run measured, one list per section.
type collected struct {
	Chat      []scoreboard.Chat
	Embed     []scoreboard.Embed
	SystemOne []scoreboard.SystemOne
	Realtime  []scoreboard.Realtime
}

func (c *collected) total() []int {
	out := make([]int, 0, 4)
	for _, n := range []int{len(c.Chat), len(c.Embed), len(c.SystemOne), len(c.Realtime)} {
		if n > 0 {
			out = append(out, n)
		}
	}
	return out
}

// addKeys adds the identity of every row of a section to keys.
func addKeys[T any, PT interface {
	*T
	scoreboard.ScoreRow
}](keys map[scoreboard.Key]struct{}, rows []T) {
	for i := range rows {
		r := PT(&rows[i])
		for _, m := range r.Base().Models {
			keys[r.Key(m)] = struct{}{}
		}
	}
}

// sharedWire returns the first wire both lists carry.
func sharedWire(a, b []scoreboard.Wire) (scoreboard.Wire, bool) {
	for _, w := range a {
		if slices.Contains(b, w) {
			return w, true
		}
	}
	return "", false
}

// queueNotes returns the notes the queue of a section carries.
func queueNotes[T any, PT interface {
	*T
	scoreboard.ScoreRow
}](section []T) string {
	for i := range section {
		row := PT(&section[i])
		if row.Untested() && row.Base().Comments != "" {
			return row.Base().Comments
		}
	}
	return ""
}

// knownElsewhere reports whether a section other than chat names the model.
func knownElsewhere(sb *scoreboard.Score, model string) bool {
	for i := range sb.Embed {
		if slices.Contains(sb.Embed[i].Models, model) {
			return true
		}
	}
	for i := range sb.SystemOne {
		if slices.Contains(sb.SystemOne[i].Models, model) {
			return true
		}
	}
	for i := range sb.Realtime {
		if slices.Contains(sb.Realtime[i].Models, model) {
			return true
		}
	}
	return false
}

// isQueuedFor reports whether any section lists the model without measuring it.
func isQueuedFor(sb *scoreboard.Score, m scoreboard.Model) bool {
	for i := range sb.Chat {
		if sb.Chat[i].Reason == m.Reason && slices.Contains(sb.Chat[i].Models, m.Model) && sb.Chat[i].Untested() {
			return true
		}
	}
	for i := range sb.Embed {
		if slices.Contains(sb.Embed[i].Models, m.Model) && sb.Embed[i].Untested() {
			return true
		}
	}
	for i := range sb.SystemOne {
		if slices.Contains(sb.SystemOne[i].Models, m.Model) && sb.SystemOne[i].Untested() {
			return true
		}
	}
	for i := range sb.Realtime {
		if sb.Realtime[i].Reason == m.Reason && slices.Contains(sb.Realtime[i].Models, m.Model) && sb.Realtime[i].Untested() {
			return true
		}
	}
	return false
}

// isQueued reports whether the row that holds key holds no measurement.
func isQueued(sb *scoreboard.Score, key scoreboard.Key) bool {
	for _, rows := range []func() []bool{
		func() []bool {
			out := make([]bool, len(sb.Chat))
			for i := range sb.Chat {
				out[i] = sb.Chat[i].Key(key.Model) == key && sb.Chat[i].Untested()
			}
			return out
		},
		func() []bool {
			out := make([]bool, len(sb.Embed))
			for i := range sb.Embed {
				out[i] = sb.Embed[i].Key(key.Model) == key && sb.Embed[i].Untested()
			}
			return out
		},
		func() []bool {
			out := make([]bool, len(sb.SystemOne))
			for i := range sb.SystemOne {
				out[i] = sb.SystemOne[i].Key(key.Model) == key && sb.SystemOne[i].Untested()
			}
			return out
		},
		func() []bool {
			out := make([]bool, len(sb.Realtime))
			for i := range sb.Realtime {
				out[i] = sb.Realtime[i].Key(key.Model) == key && sb.Realtime[i].Untested()
			}
			return out
		},
	} {
		if slices.Contains(rows(), true) {
			return true
		}
	}
	return false
}

// runSection measures what one section owes for one model.
//
// A row that names the model is measured. A row that measures nothing is a queue
// entry: it is kept, and handed to the family only when the run qualifies the
// model. A section without a row asks the family whether the model serves it.
func runSection[T any, PT interface {
	*T
	scoreboard.ScoreRow
}](ctx context.Context, t *testing.T, r smoke.Recorder, m scoreboard.Model, fam smoke.Family[T], section []T, sb *scoreboard.Score, qualify, queued, defaultQueue, tolerate bool, usage *genai.Usage, out []T) []T {
	wire, ok := sharedWire(fam.Wires(), r.Wires())
	if !ok {
		// The recorder does not serve this wire, so the family cannot run here.
		return out
	}
	var runs []T
	for i := range section {
		row := PT(&section[i])
		if row.Reasoning() != m.Reason || !slices.Contains(row.Base().Models, m.Model) {
			continue
		}
		if row.Key(m.Model).Wire != wire {
			// Another wire measured this row; this run would overwrite it.
			continue
		}
		if row.Base().Models[0] != m.Model && (!row.Untested() || !qualify) {
			// Only the first model of a row runs, for cost savings.
			t.Logf("Skipping scenario: only %s runs", row.Base().Models[0])
			continue
		}
		if row.Untested() && !qualify {
			// The scoreboard already holds this queue entry; the merge keeps it.
			continue
		}
		runs = append(runs, section[i])
	}
	measure := func(want *T) error {
		// Run one row at a time otherwise we can't collect the total usage.
		sc, u, err := fam.Run(ctx, r, m, wire, want)
		if err != nil {
			return err
		}
		usage.Add(&u)
		// The row records the wire it ran over: the same model can answer the same
		// suite over another one.
		PT(&sc).Base().Wire = wire
		// A queue entry hands its notes to the row that replaces it.
		if PT(&sc).Base().Comments == "" {
			PT(&sc).Base().Comments = PT(want).Base().Comments
		}
		sc = compare[T, PT](t, want, &sc, tolerate)
		out = append(out, sc)
		return nil
	}
	if len(runs) > 0 {
		for i := range runs {
			err := measure(&runs[i])
			if err == nil {
				continue
			}
			// The row keeps its place whatever happened: a refusal refreshes its
			// reason, and a failure stays queued so it never disappears.
			kept := runs[i]
			PT(&kept).Base().Wire = wire
			PT(&kept).SetTokens("", "")
			// The outcome is about the model under test, not the siblings the row
			// groups.
			PT(&kept).Base().Models = []string{m.Model}
			if errors.Is(err, smoke.ErrUnsupported) {
				PT(&kept).Base().Refused = unsupportedReason(err)
				t.Logf("%T: refuses: %s", fam, PT(&kept).Base().Refused)
			} else {
				// The model is unknown again: a failed probe is not a refusal.
				PT(&kept).Base().Refused = ""
				t.Logf("%T: %v", fam, err)
			}
			out = append(out, kept)
		}
		return out
	}
	if len(out) > 0 || queued {
		if queued {
			t.Log("Explicitly unsupported model")
		}
		return out
	}
	if !qualify {
		// An unqualified model joins the queue of the suite that claims unknown
		// models; a model another section already knows is not unknown.
		if defaultQueue && !knownElsewhere(sb, m.Model) {
			out = append(out, queueRow[T, PT](m, wire, queueNotes[T, PT](section)))
		}
		return out
	}
	// The family decides on the model as the provider describes it, not on a probe:
	// asking here keeps a placeholder row from constructing a client it cannot build.
	if !fam.Fits(r.Spec(m), sb, m.Model) {
		return out
	}
	var empty T
	err := measure(&empty)
	if err == nil {
		return out
	}
	if errors.Is(err, smoke.ErrUnsupported) {
		// A refusal is a result: the row records why and keeps the model out of the
		// queue. The fixture it read says what the model answered.
		refused := queueRow[T, PT](m, wire, "")
		PT(&refused).Base().Refused = unsupportedReason(err)
		t.Logf("%T: refuses: %s", fam, PT(&refused).Base().Refused)
		return append(out, refused)
	}
	// A probe that failed says nothing about the model, so the model stays queued.
	t.Logf("%T: %v", fam, err)
	return append(out, queueRow[T, PT](m, wire, queueNotes[T, PT](section)))
}

// queueRow returns a row that lists the model over one wire and holds no measurement.
func queueRow[T any, PT interface {
	*T
	scoreboard.ScoreRow
}](m scoreboard.Model, w scoreboard.Wire, notes string) T {
	var row T
	PT(&row).Base().Models = []string{m.Model}
	PT(&row).Base().Wire = w
	PT(&row).Base().Comments = notes
	PT(&row).SetReasoning(m.Reason)
	return row
}

// unsupportedReason returns the message a refusal carries.
func unsupportedReason(err error) string {
	msg := strings.TrimPrefix(err.Error(), smoke.ErrUnsupported.Error())
	return strings.TrimSpace(strings.TrimPrefix(msg, ":"))
}

// compare checks a measured row against the row it was expected to match.
//
// When tolerate is true, reasoning content discovered in a non-thinking row is
// ignored instead of causing a mismatch. This suits models like Gemma 4 that emit
// reasoning even with enable_thinking=false.
func compare[T any, PT interface {
	*T
	scoreboard.ScoreRow
}](t testing.TB, want, got *T, tolerate bool) T {
	if !PT(want).Reasoning() {
		if tolerate {
			PT(got).SetReasoning(false)
		}
	} else {
		PT(got).SetTokens("", "")
	}
	w, g := *want, *got
	clearNotes(PT(&w))
	clearNotes(PT(&g))
	if diff := cmp.Diff(w, g); diff != "" && !*updateScoreboard {
		t.Errorf("mismatch (-want +got):\n%s", diff)
	}
	return *got
}

// clearNotes drops the fields a measurement does not re-derive.
func clearNotes[T scoreboard.ScoreRow](r T) {
	r.Base().Comments = ""
	r.Base().SOTA = false
	r.Base().Good = false
	r.Base().Cheap = false
	r.SetTokens("", "")
}

// generateUpdatedScoreboard regenerates the scoreboard from scratch in one pass.
//
// It merges each section with the rows this run measured, removes stale models,
// and sorts every section so SOTA/Good/Cheap appear first.
// Returns the old and new scoreboard JSON bytes.
func generateUpdatedScoreboard(t testing.TB, scoreboardPath string, measured *collected, staleModels []scoreboard.Key, filtered bool, served []scoreboard.Wire) (rawOld, rawNew []byte) {
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
	stale := make(map[scoreboard.Key]struct{}, len(staleModels))
	for _, m := range staleModels {
		stale[m] = struct{}{}
	}
	newScore := scoreboard.Score{Warnings: oldScore.Warnings, Country: oldScore.Country, DashboardURL: oldScore.DashboardURL}
	newScore.Chat = mergeSection(t, oldScore.Chat, measured.Chat, stale, filtered, served)
	newScore.Embed = mergeSection(t, oldScore.Embed, measured.Embed, stale, filtered, served)
	newScore.SystemOne = mergeSection(t, oldScore.SystemOne, measured.SystemOne, stale, filtered, served)
	newScore.Realtime = mergeSection(t, oldScore.Realtime, measured.Realtime, stale, filtered, served)
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

// mergeSection merges one section with the rows a run measured.
//
// A measured row takes the notes and the tier flags of the row it replaces. A row
// of the old section that this run did not measure is kept, without its stale
// models. A queue entry keeps the models that no row measures, then consolidates
// with the other queue entries that share its notes.
func mergeSection[T any, PT interface {
	*T
	scoreboard.ScoreRow
}](t testing.TB, old, measured []T, stale map[scoreboard.Key]struct{}, filtered bool, served []scoreboard.Wire) []T {
	byKey := make(map[scoreboard.Key]*T, len(old))
	for i := range old {
		row := PT(&old[i])
		for _, m := range row.Base().Models {
			byKey[row.Key(m)] = &old[i]
		}
	}
	var out []T
	var queue []T
	// The keys a measured row now covers, so no old row repeats them.
	measuredKeys := map[scoreboard.Key]struct{}{}
	// The keys any row of this run covers, including a queue entry that replaces a
	// failed measurement.
	coveredKeys := map[scoreboard.Key]struct{}{}
	for i := range measured {
		row := &measured[i]
		rowPtr := PT(row)
		key := rowPtr.Key(rowPtr.Base().Models[0])
		sc := *row
		if prev, ok := byKey[key]; ok {
			prevRow := PT(prev)
			// A queue entry's notes describe the queue, so they stay with it. Notes on
			// a measurement describe the model, so they follow it.
			if !prevRow.Untested() {
				PT(&sc).Base().Comments = prevRow.Base().Comments
				// The delimiters of an inline reasoning reply are a property of the
				// model, not something a probe measures.
				start, end := prevRow.Tokens()
				PT(&sc).SetTokens(start, end)
			}
			// The tier flags describe the model, so they follow the measurement.
			PT(&sc).Base().SOTA = prevRow.Base().SOTA
			PT(&sc).Base().Good = prevRow.Base().Good
			PT(&sc).Base().Cheap = prevRow.Base().Cheap
			// A measurement is evidence for one model: a sibling keeps its own row.
		}
		coveredKeys[key] = struct{}{}
		if PT(&sc).Untested() {
			queue = append(queue, sc)
			continue
		}
		// A measurement covers the model it measured, nothing else.
		measuredKeys[PT(&sc).Key(PT(&sc).Base().Models[0])] = struct{}{}
		out = append(out, sc)
	}
	for i := range old {
		row := &old[i]
		rowPtr := PT(row)
		kept := *row
		models := make([]string, 0, len(rowPtr.Base().Models))
		for _, m := range rowPtr.Base().Models {
			key := rowPtr.Key(m)
			if _, isStale := stale[key]; isStale {
				t.Logf("Removing stale model %q with reason=%v", m, rowPtr.Reasoning())
				continue
			}
			if _, isCovered := coveredKeys[key]; isCovered {
				continue
			}
			models = append(models, m)
		}
		if rowPtr.Untested() {
			// A queue entry that lost every model to a measurement is gone.
			if len(models) == 0 {
				continue
			}
			PT(&kept).Base().Models = models
			queue = append(queue, kept)
			continue
		}
		if len(models) == 0 {
			continue
		}
		PT(&kept).Base().Models = models
		if !filtered && slices.Contains(served, rowPtr.Key(models[0]).Wire) {
			// This run did not re-test the row, so it keeps its notes but loses the
			// tier flags that a complete run re-derives. A row of another wire is not
			// this run's to re-derive.
			PT(&kept).Base().SOTA = false
			PT(&kept).Base().Good = false
			PT(&kept).Base().Cheap = false
		}
		out = append(out, kept)
	}
	// Fold the queue entries together, then drop the models a measured or kept row
	// already covers.
	for _, r := range scoreboard.ConsolidateQueue[T, PT](queue) {
		rowPtr := PT(&r)
		models := make([]string, 0, len(rowPtr.Base().Models))
		for _, m := range rowPtr.Base().Models {
			if _, isMeasured := measuredKeys[rowPtr.Key(m)]; isMeasured {
				continue
			}
			models = append(models, m)
		}
		if len(models) == 0 {
			continue
		}
		rowPtr.Base().Models = models
		out = append(out, r)
	}
	// One row per model and key: an earlier row claims the model, a later one loses
	// it. Sections never carry the same model twice.
	claimed := map[scoreboard.Key]struct{}{}
	survivors := make([]T, 0, len(out))
	for i := range out {
		row := PT(&out[i])
		models := make([]string, 0, len(row.Base().Models))
		for _, m := range row.Base().Models {
			key := row.Key(m)
			if _, ok := claimed[key]; ok {
				continue
			}
			claimed[key] = struct{}{}
			models = append(models, m)
		}
		if len(models) == 0 {
			continue
		}
		row.Base().Models = models
		survivors = append(survivors, out[i])
	}
	scoreboard.SortRows[T, PT](survivors)
	return survivors
}

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

	// A nested run sweeps its own recordings, so this one leaves it alone.
	nested := map[string]struct{}{}
	for _, d := range allDirs {
		if i := strings.IndexByte(d, '/'); i != -1 {
			nested[d[:i]] = struct{}{}
		}
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
		if strings.ContainsRune(d, '/') {
			// A nested run owns everything below it.
			continue
		}
		if _, ok := nested[d]; ok {
			continue
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
		if d.IsDir() || !strings.HasSuffix(path, ".ndjson") || strings.HasSuffix(path, ".failed.ndjson") {
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
			relSlash := filepath.ToSlash(rel)
			if strings.Count(relSlash, "/") > 1 {
				// A nested run owns the recordings below it, and reports its own.
				return nil
			}
			extra = append(extra, relSlash)
		}
		return nil
	})
	slices.Sort(extra)
	return extra, err
}
