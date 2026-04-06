// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package internaltest

import (
	"context"
	"errors"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/maruel/genai"
)

// SubprocessRecorder provides recording and replay of subprocess I/O for
// testing CLI-based providers (claudecode, codex, opencode).
//
// In record mode it delegates to the real subprocess starter (received via Wrap)
// and tees stdout into an NDJSON fixture file. In replay mode it reads the
// fixture directly without launching any subprocess.
//
// Use it as a ProviderOptionStarterWrapper:
//
//	rec := internaltest.NewSubprocessRecorder(t, "scenario", "claude")
//	c, err := claudecode.New(genai.ProviderOptionStarterWrapper(rec.Wrap))
type SubprocessRecorder struct {
	fixture string
	record  bool // true when a fresh trace should be recorded
}

// NewSubprocessRecorder returns a recorder whose fixture file lives at
// testdata/<name>.ndjson.
//
// When RECORD is "all", a fresh trace is always recorded. When RECORD is
// "failure_only", recording happens only when the fixture is missing or empty.
// Otherwise the existing fixture is replayed.
func NewSubprocessRecorder(t testing.TB, name, binaryName string) *SubprocessRecorder {
	fixture := filepath.Join("testdata", name+".ndjson")
	r := &SubprocessRecorder{fixture: fixture}
	rec := os.Getenv("RECORD")
	if rec == "all" || rec == "failure_only" {
		if _, err := exec.LookPath(binaryName); err != nil {
			t.Fatalf("RECORD=%s but %s not found: %v", rec, binaryName, err)
		}
		if rec == "all" {
			r.record = true
		} else {
			info, err := os.Stat(fixture)
			if err != nil || info.Size() == 0 {
				r.record = true
			}
		}
	}
	return r
}

// Wrap returns a starter that either records a fresh subprocess interaction or
// replays an existing fixture.
//
// It implements the genai.ProviderOptionStarterWrapper signature.
func (r *SubprocessRecorder) Wrap(inner genai.Starter) genai.Starter {
	if r.record {
		return func(ctx context.Context, args []string) (io.WriteCloser, io.ReadCloser, func() error, error) {
			stdin, stdout, wait, err := inner(ctx, args)
			if err != nil {
				return nil, nil, nil, err
			}
			if err := os.MkdirAll(filepath.Dir(r.fixture), 0o755); err != nil {
				return nil, nil, nil, err
			}
			f, err := os.Create(r.fixture)
			if err != nil {
				return nil, nil, nil, err
			}
			tee := io.TeeReader(stdout, f)
			rc := &teeReadCloser{Reader: tee, file: f, orig: stdout}
			return stdin, rc, wait, nil
		}
	}
	return func(_ context.Context, _ []string) (io.WriteCloser, io.ReadCloser, func() error, error) {
		return r.replay()
	}
}

func (r *SubprocessRecorder) replay() (io.WriteCloser, io.ReadCloser, func() error, error) {
	data, err := os.ReadFile(r.fixture)
	if err != nil {
		return nil, nil, nil, err
	}
	pr, pw := io.Pipe()
	go func() { _, _ = io.Copy(io.Discard, pr) }()
	return pw, io.NopCloser(strings.NewReader(string(data))), func() error { return nil }, nil
}

// teeReadCloser closes both the fixture file and the original ReadCloser.
type teeReadCloser struct {
	io.Reader
	file *os.File
	orig io.ReadCloser
}

func (t *teeReadCloser) Close() error {
	return errors.Join(t.orig.Close(), t.file.Close())
}
