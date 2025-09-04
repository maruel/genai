// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package sse

import (
	"errors"
	"strings"
	"testing"
)

type testResponse struct {
	Text string `json:"text"`
}

func TestProcess(t *testing.T) {
	t.Run("valid", func(t *testing.T) {
		tests := []struct {
			name  string
			input string
			want  []testResponse
		}{
			{
				name:  "basic processing",
				input: "data: {\"text\":\"message 1\"}\n\ndata: {\"text\":\"message 2\"}\n\ndata: [DONE]\n\n",
				want: []testResponse{
					{Text: "message 1"},
					{Text: "message 2"},
				},
			},
			{
				name:  "with keep-alive",
				input: "data: {\"text\":\"message 1\"}\n\n: keep-alive\n\ndata: {\"text\":\"message 2\"}\n\n",
				want: []testResponse{
					{Text: "message 1"},
					{Text: "message 2"},
				},
			},
			{
				name:  "event prefix is ignored",
				input: "event: message\n\ndata: {\"text\":\"message\"}\n\n",
				want:  []testResponse{{Text: "message"}},
			},
		}

		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				outChan := make(chan testResponse, len(tt.want)+1)
				err := Process(strings.NewReader(tt.input), outChan, nil, false)
				close(outChan)
				if err != nil {
					t.Fatal(err)
				}
				var got []testResponse
				for msg := range outChan {
					got = append(got, msg)
				}
				if len(got) != len(tt.want) {
					t.Fatalf("got %d messages, want %d", len(got), len(tt.want))
				}
				for i, expected := range tt.want {
					if got[i].Text != expected.Text {
						t.Errorf("unexpected message\ngot:  [%d] %v\nwant: %v", i, got[i], expected)
					}
				}
			})
		}
	})

	t.Run("errors", func(t *testing.T) {
		tests := []struct {
			name  string
			input string
			want  string
		}{
			{
				name:  "invalid json",
				input: "data: {invalid json}\n\n",
				want:  "failed to decode server response \"data: {invalid json}\": invalid character 'i' looking for beginning of object key string",
			},
			{
				name:  "unexpected format",
				input: "unexpected: {\"text\":\"message\"}\n\n",
				want:  "unexpected line. expected \"data: \", got \"unexpected: {\\\"text\\\":\\\"message\\\"}\"",
			},
		}
		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				outChan := make(chan testResponse, 100)
				err := Process(strings.NewReader(tt.input), outChan, nil, false)
				close(outChan)
				if err == nil {
					t.Fatal("expected error")
				}
				if s := err.Error(); s != tt.want {
					t.Fatalf("unexpected error\ngot:  %q\nwant: %q", err, tt.want)
				}
			})
		}
	})

	t.Run("ReaderError", func(t *testing.T) {
		// Test with a reader that returns an error
		errorReader := &errorReaderMock{err: errors.New("read error")}
		outChan := make(chan testResponse)
		err := Process(errorReader, outChan, nil, false)
		if err == nil {
			t.Fatal("expected error")
		}
		if !errors.Is(err, errorReader.err) {
			t.Fatal("incorrect error")
		}
	})
}

// Mock implementation of io.Reader that returns an error
type errorReaderMock struct {
	err error
}

func (e *errorReaderMock) Read(p []byte) (n int, err error) {
	return 0, e.err
}
