// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package internaltest

import (
	"os"
	"os/exec"
	"path/filepath"
	"testing"

	"github.com/maruel/genai"
	"github.com/maruel/genai/subprocessrecord"
)

// SubprocessRecorder provides recording and replay of subprocess I/O for
// testing CLI-based providers (claudecode, codex, opencode).
//
// It wraps subprocessrecord.Recorder with test-specific logic: it checks the
// RECORD environment variable and ensures the binary is available.
//
// Use it as a ProviderOptionStarterWrapper:
//
//	rec := internaltest.NewSubprocessRecorder(t, "scenario", "claude")
//	c, err := claudecode.New(genai.ProviderOptionStarterWrapper(rec.Wrap))
type SubprocessRecorder struct {
	rec     *subprocessrecord.Recorder
	forceRR bool // true when a fresh trace should be recorded
}

// NewSubprocessRecorder returns a recorder whose fixture file lives at
// testdata/<name>.ndjson.
//
// When RECORD is "all", a fresh trace is always recorded. When RECORD is
// "failure_only", recording happens only when the fixture is missing or empty.
// Otherwise the existing fixture is replayed.
func NewSubprocessRecorder(t testing.TB, name, binaryName string) *SubprocessRecorder {
	fixture := filepath.Join("testdata", name)
	rec := os.Getenv("RECORD")
	forceRR := false
	if rec == "all" || rec == "failure_only" {
		if _, err := exec.LookPath(binaryName); err != nil {
			t.Fatalf("RECORD=%s but %s not found: %v", rec, binaryName, err)
		}
		if rec == "all" {
			forceRR = true
		} else {
			info, err := os.Stat(fixture + ".ndjson")
			if err != nil || info.Size() == 0 {
				forceRR = true
			}
		}
	}
	if forceRR {
		// Remove the fixture so subprocessrecord.New records fresh.
		_ = os.Remove(fixture + ".ndjson")
	}
	r, err := subprocessrecord.New(fixture)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := r.Stop(); err != nil {
			t.Error(err)
		}
	})
	return &SubprocessRecorder{rec: r, forceRR: forceRR}
}

// Wrap returns a starter wrapper that either records or replays subprocess I/O.
//
// It implements the genai.ProviderOptionStarterWrapper signature.
func (s *SubprocessRecorder) Wrap(inner genai.Starter) genai.Starter {
	return s.rec.Wrap(inner)
}
