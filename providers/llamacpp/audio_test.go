// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Tests for speech request validation.

package llamacpp_test

import (
	"math"
	"testing"

	"github.com/maruel/genai/providers/llamacpp"
)

func TestSpeechRequest(t *testing.T) {
	t.Run("Validate", func(t *testing.T) {
		t.Run("error", func(t *testing.T) {
			for _, r := range []llamacpp.SpeechRequest{
				{},
				{Input: "hello", ResponseFormat: "mp3"},
				{Input: "hello", Voice: "female", SpeakerRef: []byte{1}},
				{Input: "hello", TopP: 2},
				{Input: "hello", Temperature: new(-1.0)},
				{Input: "hello", NPredict: -2},
				{Input: "hello", RepeatPenalty: new(math.Inf(1))},
			} {
				if err := r.Validate(); err == nil {
					t.Fatalf("accepted %+v", r)
				}
			}
		})
	})
}
