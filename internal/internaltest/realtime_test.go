// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Tests for realtime recording helpers.

package internaltest

import (
	"errors"
	"io/fs"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"github.com/coder/websocket"
)

func TestMaskRealtimeAudio(t *testing.T) {
	tests := []struct {
		name string
		typ  websocket.MessageType
		in   string
		want string
	}{
		{
			name: "client audio append",
			typ:  websocket.MessageText,
			in:   `{"type":"input_audio_buffer.append","audio":"AQIDBA=="}`,
			want: `{"audio":"AAA=","type":"input_audio_buffer.append"}`,
		},
		{
			name: "server audio delta",
			typ:  websocket.MessageText,
			in:   `{"type":"response.output_audio.delta","delta":"AQIDBA==","response_id":"r"}`,
			want: `{"delta":"AAA=","response_id":"r","type":"response.output_audio.delta"}`,
		},
		{
			name: "server audio transcript",
			typ:  websocket.MessageText,
			in:   `{"type":"response.output_audio_transcript.delta","delta":"hello"}`,
			want: `{"type":"response.output_audio_transcript.delta","delta":"hello"}`,
		},
		{
			name: "gemini audio",
			typ:  websocket.MessageBinary,
			in:   `{"serverContent":{"modelTurn":{"parts":[{"inlineData":{"mimeType":"audio/pcm;rate=24000","data":"AQIDBA=="}}]}}}`,
			want: `{"serverContent":{"modelTurn":{"parts":[{"inlineData":{"data":"AAA=","mimeType":"audio/pcm;rate=24000"}}]}}}`,
		},
		{
			name: "not json",
			typ:  websocket.MessageBinary,
			in:   "binary",
			want: "binary",
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := string(MaskRealtimeAudio(tc.typ, []byte(tc.in)))
			if !slices.Equal([]byte(got), []byte(tc.want)) {
				t.Fatalf("got %s, want %s", got, tc.want)
			}
		})
	}
}

func TestRetainFailedRecording(t *testing.T) {
	dir := t.TempDir()
	src := filepath.Join(dir, "Realtime-1.ndjson")
	if err := os.WriteFile(src, []byte("{}\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := retainFailedRecording(src); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(src); !errors.Is(err, fs.ErrNotExist) {
		t.Fatalf("source still exists: %v", err)
	}
	if got, err := os.ReadFile(filepath.Join(dir, "Realtime-1.failed.ndjson")); err != nil || string(got) != "{}\n" {
		t.Fatalf("failed recording: %q, %v", got, err)
	}
}
