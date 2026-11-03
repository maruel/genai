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
	"regexp"
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
// makes the starter fail without launching the binary. Recorded lines are
// sanitized to redact IPv4 addresses and emails. A non-nil sanitize also
// transforms each stdout line before storage; the client always receives the
// original stream.
func NewSubprocessRecorder(t testing.TB, name, binaryName string, sanitize subprocessrecord.LineSanitizer) *SubprocessRecorder {
	fixture := filepath.Join("testdata", name)
	sanitize = composeSanitizers(sanitizeSubprocessLine, sanitize)
	markRecording(t, fixture+".ndjson")
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

var (
	ipv4Pattern  = regexp.MustCompile(`\b(?:\d{1,3}\.){3}\d{1,3}\b`)
	emailPattern = regexp.MustCompile(`\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b`)
)

// sanitizeSubprocessLine removes identifying network data from a recorded line.
func sanitizeSubprocessLine(line []byte) ([]byte, error) {
	line = ipv4Pattern.ReplaceAll(line, []byte("<ip>"))
	line = emailPattern.ReplaceAll(line, []byte("<email>"))
	return line, nil
}

func composeSanitizers(first, second subprocessrecord.LineSanitizer) subprocessrecord.LineSanitizer {
	if second == nil {
		return first
	}
	return func(line []byte) ([]byte, error) {
		var err error
		if line, err = first(line); err != nil {
			return nil, err
		}
		return second(line)
	}
}
