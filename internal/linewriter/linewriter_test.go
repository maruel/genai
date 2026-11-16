// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Tests for line-atomic forwarding of child process output.

package linewriter

import (
	"errors"
	"io"
	"os/exec"
	"slices"
	"strings"
	"testing"
)

func TestAttach(t *testing.T) {
	t.Run("valid", func(t *testing.T) {
		w := &writeRecorder{}
		cmd := &exec.Cmd{}
		flush := Attach(cmd, w)
		long := strings.Repeat("x", MaxLine)
		for _, c := range []struct {
			w io.Writer
			s string
		}{
			{cmd.Stdout, "first "},
			{cmd.Stderr, "err\n"},
			{cmd.Stdout, "line\nsecond\nthi"},
			{cmd.Stdout, "rd\n"},
			{cmd.Stdout, long},
			{cmd.Stderr, "err tail"},
			{cmd.Stdout, "tail"},
		} {
			if n, err := c.w.Write([]byte(c.s)); n != len(c.s) || err != nil {
				t.Fatalf("Write(%q) = %d, %v", c.s, n, err)
			}
		}
		if err := flush(); err != nil {
			t.Fatal(err)
		}
		want := []string{"err\n", "first line\n", "second\n", "third\n", long, "tail", "err tail"}
		if !slices.Equal(w.writes, want) {
			t.Errorf("writes = %q\nwant %q", w.writes, want)
		}
	})
	t.Run("error", func(t *testing.T) {
		want := errors.New("closed")
		cmd := &exec.Cmd{}
		flush := Attach(cmd, &writeRecorder{err: want})
		if n, err := cmd.Stdout.Write([]byte("a\nb")); n != 0 || !errors.Is(err, want) {
			t.Errorf("Write = %d, %v", n, err)
		}
		if _, err := cmd.Stderr.Write([]byte("c")); err != nil {
			t.Fatal(err)
		}
		if err := flush(); !errors.Is(err, want) {
			t.Errorf("flush = %v", err)
		}
	})
}

// writeRecorder records each Write call.
type writeRecorder struct {
	writes []string
	err    error
}

func (w *writeRecorder) Write(p []byte) (int, error) {
	if w.err != nil {
		return 0, w.err
	}
	w.writes = append(w.writes, string(p))
	return len(p), nil
}
