// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Package subprocessrecord provides recording and replay of subprocess I/O for
// CLI-based providers (claudecode, codex, opencode).
//
// It is the subprocess analog of httprecord: httprecord wraps
// http.RoundTripper, subprocessrecord wraps genai.Starter.
//
// In record mode the real subprocess is launched and its stdout is teed into an
// NDJSON file. In replay mode the fixture is read back without launching any
// subprocess.
package subprocessrecord

import (
	"context"
	"errors"
	"io"
	"os"
	"path/filepath"
	"strings"

	"github.com/maruel/genai"
)

// Recorder records or replays subprocess I/O.
//
// Use Wrap as a genai.ProviderOptionStarterWrapper:
//
//	rec, err := subprocessrecord.New("testdata/scenario")
//	opts := []genai.ProviderOption{genai.ProviderOptionStarterWrapper(rec.Wrap)}
//	c, err := codex.New(opts...)
//	defer rec.Stop()
type Recorder struct {
	fixture string
	replay  bool
}

// New creates a Recorder for the given path.
//
// The path should not include the ".ndjson" extension; it is appended
// automatically. If the fixture file already exists, the recorder replays it.
// Otherwise it records.
func New(path string) (*Recorder, error) {
	fixture := path + ".ndjson"
	r := &Recorder{fixture: fixture}
	if _, err := os.Stat(fixture); err == nil {
		r.replay = true
	}
	return r, nil
}

// Stop is called when the recording session is done.
//
// It is a no-op for now but is provided for symmetry with httprecord and
// future use (e.g. flushing, validation).
func (r *Recorder) Stop() error {
	return nil
}

// Wrap returns a starter that either records subprocess I/O or replays the
// fixture.
//
// It implements the genai.ProviderOptionStarterWrapper signature.
func (r *Recorder) Wrap(inner genai.Starter) genai.Starter {
	if r.replay {
		return func(_ context.Context, _ []string) (io.WriteCloser, io.ReadCloser, func() error, error) {
			return replayFixture(r.fixture)
		}
	}
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

func replayFixture(fixture string) (io.WriteCloser, io.ReadCloser, func() error, error) {
	data, err := os.ReadFile(fixture)
	if err != nil {
		return nil, nil, nil, err
	}
	// Provide a stdin that discards writes, matching the subprocess interface.
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
