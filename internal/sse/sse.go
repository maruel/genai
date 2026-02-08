// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Package sse provides Server-Sent Events (SSE) processing utilities.
package sse

import (
	"bufio"
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"iter"
	"reflect"

	"github.com/maruel/genai/internal"
)

// Process reads and processes Server-Sent Events (SSE) from the provided reader.
//
// It parses the SSE format and decodes JSON messages into values of type T.
// The decoded values are sent to the iterator.
//
// If decoding into T fails, it tries to decode into er, which the error code path. If this succeeds, the
// error is returned and the iterator is stopped.
//
// https://developer.mozilla.org/en-US/docs/Web/API/Server-sent%5Fevents/Using%5Fserver-sent%5Fevents
func Process[T any](body io.Reader, er error, lenient bool) (iter.Seq[T], func() error) {
	var finalErr error
	it := func(yield func(T) bool) {
		for r := bufio.NewReader(body); ; {
			line, err := r.ReadBytes('\n')
			if line = bytes.TrimSpace(line); errors.Is(err, io.EOF) {
				if len(line) == 0 {
					return
				}
			} else if err != nil {
				finalErr = &internal.BadError{Err: fmt.Errorf("sse: failed to get server response: %w", err)}
				return
			}
			if len(line) == 0 {
				continue
			}

			switch {
			case bytes.HasPrefix(line, dataPrefix):
				suffix := line[len(dataPrefix):]
				if bytes.Equal(suffix, done) {
					return
				}
				r := bytes.NewReader(suffix)
				d := json.NewDecoder(r)
				var r2 io.ReadSeeker
				if !lenient {
					d.DisallowUnknownFields()
					r2 = r
				}
				var msg T
				if _, err = internal.DecodeJSON(d, &msg, r2); err == nil {
					// It may have succeeded but not decoded anything.
					if v := reflect.ValueOf(&msg); !reflect.DeepEqual(&msg, reflect.Zero(v.Type()).Interface()) {
						if !yield(msg) {
							return
						}
						continue
					}
				}
				if er == nil {
					if err == nil {
						finalErr = &internal.BadError{Err: fmt.Errorf("sse: failed to decode server response %q", string(line))}
						return
					}
					finalErr = &internal.BadError{Err: fmt.Errorf("sse: failed to decode server response %q: %w", string(line), err)}
					return
				}
				if _, err2 := r.Seek(0, 0); err2 != nil {
					finalErr = &internal.BadError{Err: fmt.Errorf("sse: failed to seek: %w", err2)}
					return
				}
				d = json.NewDecoder(r)
				if !lenient {
					d.DisallowUnknownFields()
				}
				if _, err2 := internal.DecodeJSON(d, er, r2); err2 == nil {
					// Do not wrap the error here since we decoded the error from the server.
					finalErr = er
					return
				}
				// Falling back or when in strict mode, return the decoding error instead.
				finalErr = &internal.BadError{Err: fmt.Errorf("sse: failed to decode server response %q: %w", string(line), err)}
				return
			case bytes.Equal(line, keepAlive):
				// Ignore keep-alive messages. Very few send this.
			case bytes.Equal(line, keepAliveHuggingface):
				// Huggingface...
			case bytes.HasPrefix(line, eventPrefix):
				// Ignore event headers. Very few send this.
			default:
				finalErr = &internal.BadError{Err: fmt.Errorf("sse: unexpected line. expected \"data: \", got %q", line)}
				return
			}
		}
	}
	return it, func() error {
		return finalErr
	}
}

var (
	dataPrefix           = []byte("data: ")
	eventPrefix          = []byte("event:")
	done                 = []byte("[DONE]")
	keepAlive            = []byte(": keep-alive")
	keepAliveHuggingface = []byte(":")
)
