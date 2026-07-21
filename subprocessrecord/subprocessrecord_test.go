// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Tests for the subprocessrecord package.

package subprocessrecord

import (
	"context"
	"errors"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/maruel/genai"
)

// fakeStarter returns a Starter that produces output without launching a real
// subprocess.
func fakeStarter(output string) genai.Starter {
	return func(_ context.Context, _ []string) (io.WriteCloser, io.ReadCloser, func() error, error) {
		pr, pw := io.Pipe()
		go func() { _, _ = io.Copy(io.Discard, pr) }()
		return pw, io.NopCloser(strings.NewReader(output)), func() error { return nil }, nil
	}
}

func TestNew(t *testing.T) {
	t.Run("valid", func(t *testing.T) {
		t.Run("record_when_no_fixture", func(t *testing.T) {
			rec, err := New(filepath.Join(t.TempDir(), "test"))
			if err != nil {
				t.Fatal(err)
			}
			if rec.replay {
				t.Fatal("expected record mode")
			}
		})
		t.Run("record_when_empty_fixture", func(t *testing.T) {
			dir := t.TempDir()
			path := filepath.Join(dir, "test")
			if err := os.WriteFile(path+".ndjson", nil, 0o644); err != nil {
				t.Fatal(err)
			}
			rec, err := New(path)
			if err != nil {
				t.Fatal(err)
			}
			if rec.replay {
				t.Fatal("empty fixture should trigger record mode")
			}
		})
		t.Run("replay_when_fixture_exists", func(t *testing.T) {
			dir := t.TempDir()
			path := filepath.Join(dir, "test")
			if err := os.WriteFile(path+".ndjson", []byte("data\n"), 0o644); err != nil {
				t.Fatal(err)
			}
			rec, err := New(path)
			if err != nil {
				t.Fatal(err)
			}
			if !rec.replay {
				t.Fatal("expected replay mode")
			}
		})
	})
}

func TestRecorder(t *testing.T) {
	t.Run("valid", func(t *testing.T) {
		t.Run("record", func(t *testing.T) {
			dir := t.TempDir()
			path := filepath.Join(dir, "test")
			fixture := path + ".ndjson"
			rec, err := New(path)
			if err != nil {
				t.Fatal(err)
			}
			want := `{"msg":"hello"}` + "\n"
			starter := rec.Wrap(fakeStarter(want))
			stdin, stdout, wait, err := starter(t.Context(), []string{"fake"})
			if err != nil {
				t.Fatal(err)
			}
			got, err := io.ReadAll(stdout)
			if err != nil {
				t.Fatal(err)
			}
			if string(got) != want {
				t.Fatalf("stdout: got %q, want %q", got, want)
			}
			if err := stdin.Close(); err != nil {
				t.Fatal(err)
			}
			if err := stdout.Close(); err != nil {
				t.Fatal(err)
			}
			if err := wait(); err != nil {
				t.Fatal(err)
			}
			if err := rec.Stop(); err != nil {
				t.Fatal(err)
			}
			data, err := os.ReadFile(fixture)
			if err != nil {
				t.Fatal(err)
			}
			if string(data) != want {
				t.Fatalf("fixture: got %q, want %q", data, want)
			}
		})
		t.Run("delete_empty_recording", func(t *testing.T) {
			dir := t.TempDir()
			path := filepath.Join(dir, "test")
			fixture := path + ".ndjson"
			rec, err := New(path)
			if err != nil {
				t.Fatal(err)
			}
			starter := rec.Wrap(fakeStarter(""))
			stdin, stdout, wait, err := starter(t.Context(), []string{"fake"})
			if err != nil {
				t.Fatal(err)
			}
			if _, err := io.ReadAll(stdout); err != nil {
				t.Fatal(err)
			}
			if err := stdin.Close(); err != nil {
				t.Fatal(err)
			}
			if err := stdout.Close(); err != nil {
				t.Fatal(err)
			}
			if err := wait(); err != nil {
				t.Fatal(err)
			}
			if err := rec.Stop(); err != nil {
				t.Fatal(err)
			}
			if _, err := os.Stat(fixture); !errors.Is(err, os.ErrNotExist) {
				t.Fatalf("fixture: got %v, want not exist", err)
			}
		})
		t.Run("replay", func(t *testing.T) {
			dir := t.TempDir()
			path := filepath.Join(dir, "test")
			want := `{"msg":"replayed"}` + "\n"
			if err := os.WriteFile(path+".ndjson", []byte(want), 0o644); err != nil {
				t.Fatal(err)
			}
			rec, err := New(path)
			if err != nil {
				t.Fatal(err)
			}
			// Inner starter must not be called during replay.
			boom := func(_ context.Context, _ []string) (io.WriteCloser, io.ReadCloser, func() error, error) {
				t.Fatal("inner starter called during replay")
				return nil, nil, nil, nil
			}
			starter := rec.Wrap(boom)
			stdin, stdout, wait, err := starter(t.Context(), []string{"ignored"})
			if err != nil {
				t.Fatal(err)
			}
			got, err := io.ReadAll(stdout)
			if err != nil {
				t.Fatal(err)
			}
			if string(got) != want {
				t.Fatalf("stdout: got %q, want %q", got, want)
			}
			if err := stdin.Close(); err != nil {
				t.Fatal(err)
			}
			if err := wait(); err != nil {
				t.Fatal(err)
			}
			if err := rec.Stop(); err != nil {
				t.Fatal(err)
			}
		})
		t.Run("record_then_replay", func(t *testing.T) {
			dir := t.TempDir()
			path := filepath.Join(dir, "test")
			want := `{"line":1}` + "\n" + `{"line":2}` + "\n"

			// Record.
			rec, err := New(path)
			if err != nil {
				t.Fatal(err)
			}
			starter := rec.Wrap(fakeStarter(want))
			stdin, stdout, _, err := starter(t.Context(), nil)
			if err != nil {
				t.Fatal(err)
			}
			if _, err := io.ReadAll(stdout); err != nil {
				t.Fatal(err)
			}
			if err := stdin.Close(); err != nil {
				t.Fatal(err)
			}
			if err := stdout.Close(); err != nil {
				t.Fatal(err)
			}
			if err := rec.Stop(); err != nil {
				t.Fatal(err)
			}

			// Replay.
			rec2, err := New(path)
			if err != nil {
				t.Fatal(err)
			}
			if !rec2.replay {
				t.Fatal("expected replay after recording")
			}
			starter2 := rec2.Wrap(nil)
			_, stdout2, _, err := starter2(t.Context(), nil)
			if err != nil {
				t.Fatal(err)
			}
			got, err := io.ReadAll(stdout2)
			if err != nil {
				t.Fatal(err)
			}
			if string(got) != want {
				t.Fatalf("replay: got %q, want %q", got, want)
			}
		})
	})
	t.Run("error", func(t *testing.T) {
		t.Run("record_inner_error", func(t *testing.T) {
			dir := t.TempDir()
			path := filepath.Join(dir, "test")
			rec, err := New(path)
			if err != nil {
				t.Fatal(err)
			}
			wantErr := errors.New("spawn failed")
			failing := func(_ context.Context, _ []string) (io.WriteCloser, io.ReadCloser, func() error, error) {
				return nil, nil, nil, wantErr
			}
			starter := rec.Wrap(failing)
			_, _, _, err = starter(t.Context(), nil)
			if !errors.Is(err, wantErr) {
				t.Fatalf("got %v, want %v", err, wantErr)
			}
		})
		t.Run("replay_missing_fixture", func(t *testing.T) {
			dir := t.TempDir()
			path := filepath.Join(dir, "test")
			// Manually force replay mode with a missing fixture.
			rec := &Recorder{fixture: path + ".ndjson", replay: true}
			starter := rec.Wrap(nil)
			_, _, _, err := starter(t.Context(), nil)
			if err == nil {
				t.Fatal("expected error for missing fixture")
			}
		})
	})
}
