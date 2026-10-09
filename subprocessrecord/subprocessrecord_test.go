// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Tests for the subprocessrecord package.

package subprocessrecord

import (
	"bytes"
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
		return &discardWriteCloser{}, io.NopCloser(strings.NewReader(output)), func() error { return nil }, nil
	}
}

func TestNew(t *testing.T) {
	t.Run("valid", func(t *testing.T) {
		t.Run("record_when_empty_fixture", func(t *testing.T) {
			dir := t.TempDir()
			path := filepath.Join(dir, "test")
			if err := os.WriteFile(path+".ndjson", nil, 0o644); err != nil {
				t.Fatal(err)
			}
			rec, err := New(path, nil)
			if err != nil {
				t.Fatal(err)
			}
			if rec.replay {
				t.Fatal("empty fixture should trigger record mode")
			}
		})
		t.Run("record_sanitized", func(t *testing.T) {
			dir := t.TempDir()
			path := filepath.Join(dir, "test")
			rec, err := New(path, func(line []byte) ([]byte, error) {
				return bytes.ReplaceAll(line, []byte("host-secret"), []byte("<redacted>")), nil
			})
			if err != nil {
				t.Fatal(err)
			}
			want := "first host-secret\nsecond host-secret\n"
			stdin, stdout, wait, err := rec.Wrap(fakeStarter(want))(t.Context(), nil)
			if err != nil {
				t.Fatal(err)
			}
			got, err := io.ReadAll(stdout)
			if err != nil {
				t.Fatal(err)
			}
			if string(got) != want {
				t.Fatalf("stdout = %q, want raw %q", got, want)
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
			fixture, err := os.ReadFile(path + ".ndjson")
			if err != nil {
				t.Fatal(err)
			}
			if string(fixture) != "first <redacted>\nsecond <redacted>\n" {
				t.Fatalf("fixture = %q", fixture)
			}
		})
	})
}

func TestRecorder(t *testing.T) {
	t.Run("valid", func(t *testing.T) {
		t.Run("delete_empty_recording", func(t *testing.T) {
			dir := t.TempDir()
			path := filepath.Join(dir, "test")
			fixture := path + ".ndjson"
			rec, err := New(path, nil)
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
			rec, err := New(path, nil)
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
			if n, err := stdin.Write([]byte("ignored input")); n != 13 || err != nil {
				t.Fatalf("stdin.Write = %d, %v", n, err)
			}
			if err := stdin.Close(); err != nil {
				t.Fatal(err)
			}
			if n, err := stdin.Write(nil); n != 0 || !errors.Is(err, io.ErrClosedPipe) {
				t.Fatalf("closed stdin.Write = %d, %v", n, err)
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
		})
	})
	t.Run("error", func(t *testing.T) {
		t.Run("record_inner_error", func(t *testing.T) {
			dir := t.TempDir()
			path := filepath.Join(dir, "test")
			rec, err := New(path, nil)
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
