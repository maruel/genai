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
)

// Starter launches a subprocess and returns its stdin, stdout, a wait function,
// and any error.
//
// It mirrors the unexported executor interface used by CLI-based providers.
type Starter func(ctx context.Context, args []string) (io.WriteCloser, io.ReadCloser, func() error, error)

// SubprocessRecorder provides recording and replay of subprocess I/O for
// testing CLI-based providers (claudecode, codex, opencode).
//
// In record mode it delegates to a real subprocess via a Starter and tees
// stdout into an NDJSON fixture file. In replay mode it reads the fixture
// directly without launching any subprocess.
type SubprocessRecorder struct {
	fixture string
	starter Starter // non-nil only in record mode
}

// NewSubprocessRecorder returns a recorder whose fixture file lives at
// testdata/<name>.ndjson.
//
// makeStarter is called with the resolved binary path when recording is
// needed. It should return a Starter that launches the real subprocess
// (typically wrapping the provider's cmdExecutor).
//
// When RECORD is "all", a fresh trace is always recorded. When RECORD is
// "failure_only", recording happens only when the fixture is missing or empty.
// Otherwise the existing fixture is replayed.
func NewSubprocessRecorder(t testing.TB, name, binaryName string, makeStarter func(bin string) Starter) *SubprocessRecorder {
	fixture := filepath.Join("testdata", name+".ndjson")
	r := &SubprocessRecorder{fixture: fixture}
	rec := os.Getenv("RECORD")
	if rec == "all" || rec == "failure_only" {
		bin, err := exec.LookPath(binaryName)
		if err != nil {
			t.Skipf("RECORD=%s but %s not found: %v", rec, binaryName, err)
		}
		if rec == "all" {
			r.starter = makeStarter(bin)
		} else {
			info, err := os.Stat(fixture)
			if err != nil || info.Size() == 0 {
				r.starter = makeStarter(bin)
			}
		}
	}
	return r
}

// Start either records a fresh subprocess interaction or replays an existing
// fixture, returning stdin, stdout, a wait function, and any error.
func (r *SubprocessRecorder) Start(ctx context.Context, args []string) (io.WriteCloser, io.ReadCloser, func() error, error) {
	if r.starter != nil {
		return r.record(ctx, args)
	}
	return r.replay()
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

func (r *SubprocessRecorder) record(ctx context.Context, args []string) (io.WriteCloser, io.ReadCloser, func() error, error) {
	stdin, stdout, wait, err := r.starter(ctx, args)
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

// teeReadCloser closes both the fixture file and the original ReadCloser.
type teeReadCloser struct {
	io.Reader
	file *os.File
	orig io.ReadCloser
}

func (t *teeReadCloser) Close() error {
	return errors.Join(t.orig.Close(), t.file.Close())
}
