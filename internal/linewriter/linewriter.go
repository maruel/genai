// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Package linewriter forwards a child process's output one complete line per
// Write. A pipe read can split or merge lines, so forwarding reads as is
// would make a per-Write logger, such as slog.NewLogLogger's writer, emit
// fragments or several lines as one record.
package linewriter

import (
	"bytes"
	"errors"
	"io"
	"os/exec"
	"sync"
)

// MaxLine bounds a buffered line, so a stream without newlines cannot grow
// memory without limit. A longer line is forwarded in pieces.
const MaxLine = 64 << 10

// Attach sets cmd's stdout and stderr to forward each complete line to w in
// one Write, one Write at a time. Call the returned flush after cmd.Wait
// returns, which happens after both streams were copied, to forward an
// unterminated last line.
func Attach(cmd *exec.Cmd, w io.Writer) (flush func() error) {
	mu := &sync.Mutex{}
	stdout := &writer{mu: mu, w: w}
	stderr := &writer{mu: mu, w: w}
	cmd.Stdout, cmd.Stderr = stdout, stderr
	return func() error { return errors.Join(stdout.flush(), stderr.flush()) }
}

// writer buffers one stream up to its newline.
type writer struct {
	mu  *sync.Mutex // Shared by both streams; serializes writes to w.
	w   io.Writer
	buf []byte
}

func (l *writer) Write(p []byte) (int, error) {
	n := len(p)
	for {
		i := bytes.IndexByte(p, '\n')
		if i < 0 {
			break
		}
		line := p[:i+1]
		if len(l.buf) != 0 {
			l.buf = append(l.buf, line...)
			line = l.buf
		}
		if err := l.emit(line); err != nil {
			return n - len(p), err
		}
		l.buf = l.buf[:0]
		p = p[i+1:]
	}
	l.buf = append(l.buf, p...)
	if len(l.buf) >= MaxLine {
		if err := l.emit(l.buf); err != nil {
			return n - len(p), err
		}
		l.buf = l.buf[:0]
	}
	return n, nil
}

func (l *writer) flush() error {
	if len(l.buf) == 0 {
		return nil
	}
	err := l.emit(l.buf)
	l.buf = nil
	return err
}

func (l *writer) emit(line []byte) error {
	l.mu.Lock()
	defer l.mu.Unlock()
	_, err := l.w.Write(line)
	return err
}
