// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Subprocess test utilities.

package internaltest

import (
	"context"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	"github.com/maruel/genai"
	"github.com/maruel/genai/subprocessrecord"
)

// SubprocessRecorder provides recording and replay of subprocess I/O for
// testing CLI-based providers.
//
// It wraps subprocessrecord.Recorder with test-specific logic: it checks the
// RECORD environment variable and ensures the binary is available.
//
// Use it as a ProviderOptionStarterWrapper:
//
//	rec := internaltest.NewSubprocessRecorder(t, "scenario", "claude", nil)
//	c, err := claudecode.New(t.Context(), genai.ProviderOptionStarterWrapper(rec.Wrap))
//	if err != nil {
//		t.Fatal(err)
//	}
//	internaltest.CleanupCloser(t, c)
type SubprocessRecorder struct {
	rec     *subprocessrecord.Recorder
	missing string // fixture path when it is absent outside recording mode
}

// NewSubprocessRecorder returns a recorder whose fixture file lives at
// testdata/<name>.ndjson.
//
// When RECORD is "all", a fresh trace is always recorded. When RECORD is
// "failure_only", recording happens only when the fixture is missing or empty.
// Otherwise the existing fixture is replayed, and a missing or empty fixture
// makes the starter fail without launching the binary. If sanitize is non-nil,
// it transforms each stdout line before storage while preserving the original
// stream for the client.
func NewSubprocessRecorder(t testing.TB, name, binaryName string, sanitize subprocessrecord.LineSanitizer) *SubprocessRecorder {
	fixture := filepath.Join("testdata", name)
	markSubprocessRecording(t, fixture+".ndjson")
	st, err := os.Stat(fixture + ".ndjson")
	exists := err == nil && st.Size() != 0
	switch rec := os.Getenv("RECORD"); rec {
	case "all", "failure_only":
		if _, err := exec.LookPath(binaryName); err != nil {
			t.Fatalf("RECORD=%s but %s not found: %v", rec, binaryName, err)
		}
		if rec == "all" || !exists {
			// Remove the fixture so subprocessrecord.New records fresh.
			_ = os.Remove(fixture + ".ndjson")
		}
	default:
		if !exists {
			return &SubprocessRecorder{missing: fixture + ".ndjson"}
		}
	}
	r, err := subprocessrecord.New(fixture, sanitize)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := r.Stop(); err != nil {
			t.Error(err)
		}
	})
	return &SubprocessRecorder{rec: r}
}

// Wrap returns a starter wrapper that either records or replays subprocess I/O.
//
// It implements the genai.ProviderOptionStarterWrapper signature.
func (s *SubprocessRecorder) Wrap(inner genai.Starter) genai.Starter {
	if s.missing != "" {
		return func(context.Context, []string) (io.WriteCloser, io.ReadCloser, func() error, error) {
			return nil, nil, nil, fmt.Errorf("no recording at %s; record it with RECORD=failure_only", s.missing)
		}
	}
	return s.rec.Wrap(inner)
}

var subprocessRecordings = struct {
	sync.Mutex
	used map[string]map[string]struct{}
}{used: map[string]map[string]struct{}{}}

// TakeSubprocessRecordings returns the fixture paths used by tests named prefix
// or one of its subtests, then clears those entries.
func TakeSubprocessRecordings(prefix string) map[string]struct{} {
	subprocessRecordings.Lock()
	defer subprocessRecordings.Unlock()
	out := map[string]struct{}{}
	for name, paths := range subprocessRecordings.used {
		if name == prefix || strings.HasPrefix(name, prefix+"/") {
			for path := range paths {
				out[path] = struct{}{}
			}
			delete(subprocessRecordings.used, name)
		}
	}
	return out
}

func markSubprocessRecording(t testing.TB, fixture string) {
	abs, err := filepath.Abs(fixture)
	if err != nil {
		t.Fatalf("resolve fixture path: %v", err)
	}
	name := t.Name()
	subprocessRecordings.Lock()
	defer subprocessRecordings.Unlock()
	if subprocessRecordings.used[name] == nil {
		subprocessRecordings.used[name] = map[string]struct{}{}
	}
	subprocessRecordings.used[name][abs] = struct{}{}
}
