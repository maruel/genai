// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Package bb is a separate package so it can be imported by genai while being internal and exported so
// cmp.Diff() isn't unhappy.
package bb

import (
	"errors"
	"fmt"
	"io"
)

// BytesBuffer wraps bytes.Buffer with io.Seeker support.
type BytesBuffer struct {
	D   []byte
	Pos int
}

func (b *BytesBuffer) Read(p []byte) (int, error) {
	n := copy(p, b.D[b.Pos:])
	if n == 0 {
		return 0, io.EOF
	}
	b.Pos += n
	return n, nil
}

// Seek implements io.Seeker.
func (b *BytesBuffer) Seek(offset int64, whence int) (int64, error) {
	var p int64
	if whence == io.SeekCurrent {
		offset += int64(b.Pos)
		whence = io.SeekStart
	}
	switch whence {
	case io.SeekEnd:
		offset = int64(len(b.D)) - offset
		fallthrough
	case io.SeekStart:
		if offset < 0 || offset > int64(len(b.D)) {
			return p, errors.New("out of bound")
		}
		p = offset
		b.Pos = int(p)
	default:
		return p, fmt.Errorf("unknown whence %d", whence)
	}
	return p, nil
}

func (b *BytesBuffer) Write(p []byte) (int, error) {
	b.D = append(b.D, p...)
	return len(p), nil
}
