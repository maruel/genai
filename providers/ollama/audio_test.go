// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Tests for audio endpoint parameter validation and transport failures.

package ollama_test

import (
	"errors"
	"io"
	"math"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/maruel/genai"
	"github.com/maruel/genai/providers/ollama"
)

func TestSpeechRequest(t *testing.T) {
	t.Run("Validate", func(t *testing.T) {
		t.Run("error", func(t *testing.T) {
			for _, r := range []ollama.SpeechRequest{{}, {Model: "m", Input: strings.Repeat("x", 4097)}, {Model: "m", Input: "hello", ResponseFormat: "mp3"}, {Model: "m", Input: "hello", Speed: new(2.0)}, {Model: "m", Input: "hello", Instructions: "whisper"}, {Model: "m", Input: "hello", StreamFormat: "sse"}} {
				if err := r.Validate(); err == nil {
					t.Fatalf("accepted %+v", r)
				}
			}
		})
	})
}

func TestAudioTextRequest(t *testing.T) {
	t.Run("Validate", func(t *testing.T) {
		t.Run("error", func(t *testing.T) {
			for _, r := range []ollama.AudioTextRequest{{}, {Model: "m", File: strings.NewReader("wav"), ResponseFormat: "srt"}, {Model: "m", File: strings.NewReader("wav"), Temperature: math.NaN()}} {
				if err := r.Validate(); err == nil {
					t.Fatalf("accepted %+v", r)
				}
			}
		})
	})
}

func TestClientAudioErrors(t *testing.T) {
	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/v1/audio/transcriptions" {
			w.WriteHeader(http.StatusBadRequest)
			_, _ = io.WriteString(w, `{"error":{"message":"invalid WAV","type":"invalid_request_error","param":null,"code":null}}`)
			return
		}
		_, _ = io.WriteString(w, "audio")
	}))
	t.Cleanup(s.Close)
	c, err := ollama.New(t.Context(), genai.ProviderOptionRemote(s.URL))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := c.Close(); err != nil {
			t.Error(err)
		}
	})
	t.Run("HTTP error", func(t *testing.T) {
		err := c.TranscriptionRaw(t.Context(), &ollama.AudioTextRequest{Model: "m", File: strings.NewReader("invalid")}, io.Discard)
		if err == nil || !strings.Contains(err.Error(), "invalid WAV") {
			t.Fatalf("unexpected error: %v", err)
		}
	})
	t.Run("writer error", func(t *testing.T) {
		err := c.SpeechRaw(t.Context(), &ollama.SpeechRequest{Model: "m", Input: "hello"}, audioErrorWriter{})
		if !errors.Is(err, io.ErrClosedPipe) {
			t.Fatalf("unexpected error: %v", err)
		}
	})
}

type audioErrorWriter struct{}

func (audioErrorWriter) Write([]byte) (int, error) { return 0, io.ErrClosedPipe }
