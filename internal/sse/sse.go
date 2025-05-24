// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Package sse provides Server-Sent Events (SSE) processing utilities.
package sse

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
)

// Process reads and processes Server-Sent Events (SSE) from the provided reader.
// It parses the SSE format and decodes JSON messages into values of type T.
// The decoded values are sent to the provided channel.
//
// https://developer.mozilla.org/en-US/docs/Web/API/Server-sent%5Fevents/Using%5Fserver-sent%5Fevents
func Process[T any](body io.Reader, out chan<- T, er any, lenient bool) error {
	for r := bufio.NewReader(body); ; {
		line, err := r.ReadBytes('\n')
		if line = bytes.TrimSpace(line); err == io.EOF {
			if len(line) == 0 {
				return nil
			}
		} else if err != nil {
			return fmt.Errorf("failed to get server response: %w", err)
		}
		if len(line) == 0 {
			continue
		}

		switch {
		case bytes.HasPrefix(line, dataPrefix):
			suffix := line[len(dataPrefix):]
			if bytes.Equal(suffix, done) {
				return nil
			}
			d := json.NewDecoder(bytes.NewReader(suffix))
			if !lenient {
				d.DisallowUnknownFields()
			}
			d.UseNumber()
			var msg T
			if err = d.Decode(&msg); err != nil {
				if er != nil {
					d = json.NewDecoder(bytes.NewReader(suffix))
					if !lenient {
						d.DisallowUnknownFields()
					}
					d.UseNumber()
					if err = d.Decode(er); err == nil {
						return fmt.Errorf("%s", er)
					}
				}
				return fmt.Errorf("failed to decode server response %q: %w", string(line), err)
			}
			out <- msg
		case bytes.Equal(line, keepAlive):
			// Ignore keep-alive messages. Very few send this.
		case bytes.HasPrefix(line, eventPrefix):
			// Ignore event headers. Very few send this.
		default:
			return fmt.Errorf("unexpected line. expected \"data: \", got %q", line)
		}
	}
}

var (
	dataPrefix  = []byte("data: ")
	eventPrefix = []byte("event:")
	done        = []byte("[DONE]")
	keepAlive   = []byte(": keep-alive")
)
