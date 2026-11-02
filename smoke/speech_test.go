// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Tests for realtime speech audio helpers.

package smoke

import (
	"context"
	"encoding/binary"
	"slices"
	"testing"
)

func TestResamplePCM16(t *testing.T) {
	t.Run("valid", func(t *testing.T) {
		in := pcm16(0, 1000, 2000, 3000)
		got, err := ResamplePCM16(in, 2, 1)
		if err != nil {
			t.Fatal(err)
		}
		if want := pcm16(0, 2000); !slices.Equal(got, want) {
			t.Fatalf("got %v, want %v", pcmSamples(got), pcmSamples(want))
		}

		got, err = ResamplePCM16(in, 1, 2)
		if err != nil {
			t.Fatal(err)
		}
		if want := pcm16(0, 500, 1000, 1500, 2000, 2500, 3000, 3000); !slices.Equal(got, want) {
			t.Fatalf("got %v, want %v", pcmSamples(got), pcmSamples(want))
		}

		if got, err = ResamplePCM16(nil, 32000, 16000); err != nil || len(got) != 0 {
			t.Fatalf("empty: got %v, err=%v", got, err)
		}
	})

	t.Run("error", func(t *testing.T) {
		for _, tc := range []struct {
			name     string
			in       []byte
			from, to int
		}{
			{"rates", nil, 0, 16000},
			{"negative rate", nil, 16000, -1},
			{"partial sample", []byte{0}, 16000, 24000},
		} {
			t.Run(tc.name, func(t *testing.T) {
				if _, err := ResamplePCM16(tc.in, tc.from, tc.to); err == nil {
					t.Fatal("accepted invalid input")
				}
			})
		}
	})
}

func TestSpeech(t *testing.T) {
	t.Run("valid", func(t *testing.T) {
		for _, fixture := range []speechFixture{speechFixtureCode, speechFixtureOrange} {
			for _, rate := range []int{8000, 16000, 24000} {
				got, err := speech(t.Context(), fixture, rate)
				if err != nil {
					t.Fatalf("fixture=%q rate=%d: %v", fixture, rate, err)
				}
				if len(got) == 0 || len(got)%2 != 0 {
					t.Fatalf("fixture=%q rate=%d: %d bytes", fixture, rate, len(got))
				}
			}
		}
	})

	t.Run("error", func(t *testing.T) {
		if _, err := speech(t.Context(), speechFixture("unknown"), 16000); err == nil {
			t.Fatal("accepted unknown fixture")
		}
		if _, err := speech(t.Context(), speechFixtureOrange, 0); err == nil {
			t.Fatal("accepted zero rate")
		}
		ctx, cancel := context.WithCancel(t.Context())
		cancel()
		if _, err := speech(ctx, speechFixtureOrange, 16000); err == nil {
			t.Fatal("accepted canceled context")
		}
	})
}

func pcm16(samples ...int16) []byte {
	out := make([]byte, len(samples)*2)
	for i, s := range samples {
		binary.LittleEndian.PutUint16(out[i*2:], uint16(s))
	}
	return out
}

func pcmSamples(in []byte) []int16 {
	out := make([]int16, len(in)/2)
	for i := range out {
		out[i] = int16(binary.LittleEndian.Uint16(in[i*2:]))
	}
	return out
}
