// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Internal tests for realtime fixture parsing.

package smoke

import (
	"bytes"
	"encoding/binary"
	"slices"
	"testing"

	"github.com/maruel/genai/scoreboard"
)

func TestDecodePCM16WAV(t *testing.T) {
	raw, err := EncodePCM16WAV([]byte{0, 0, 1, 0, 255, 255}, 16000)
	if err != nil {
		t.Fatal(err)
	}

	t.Run("valid", func(t *testing.T) {
		got, rate, err := DecodePCM16WAV(raw)
		if err != nil {
			t.Fatal(err)
		}
		if rate != 16000 || len(got) == 0 || len(got)%2 != 0 {
			t.Fatalf("rate=%d bytes=%d", rate, len(got))
		}
	})

	t.Run("error", func(t *testing.T) {
		chunkOffset := func(id string) int {
			i := bytes.Index(raw[12:], []byte(id))
			if i < 0 {
				t.Fatalf("missing %q chunk", id)
			}
			return 12 + i
		}
		tests := map[string][]byte{
			"not RIFF": func() []byte {
				b := slices.Clone(raw)
				copy(b, "NOPE")
				return b
			}(),
			"wrong RIFF size": func() []byte {
				b := slices.Clone(raw)
				binary.LittleEndian.PutUint32(b[4:8], 4)
				return b
			}(),
			"trailing data": append(slices.Clone(raw), 0),
			"truncated chunk header": func() []byte {
				b := slices.Clone(raw[:len(raw)-2])
				binary.LittleEndian.PutUint32(b[4:8], uint32(len(b)-8))
				return b
			}(),
			"truncated chunk": func() []byte {
				b := slices.Clone(raw)
				binary.LittleEndian.PutUint32(b[chunkOffset("data")+4:], uint32(len(b)))
				return b
			}(),
		}
		for name, b := range tests {
			t.Run(name, func(t *testing.T) {
				if _, _, err := DecodePCM16WAV(b); err == nil {
					t.Fatal("accepted invalid WAV")
				}
			})
		}
	})
}

func TestEncodePCM16WAV(t *testing.T) {
	pcm := []byte{0, 0, 1, 0, 255, 255}
	raw, err := EncodePCM16WAV(pcm, 16000)
	if err != nil {
		t.Fatal(err)
	}
	got, rate, err := DecodePCM16WAV(raw)
	if err != nil {
		t.Fatal(err)
	}
	if rate != 16000 || !slices.Equal(got, pcm) {
		t.Fatalf("rate=%d data=%v", rate, got)
	}

	for _, tc := range []struct {
		name string
		pcm  []byte
		rate int
	}{
		{name: "empty", rate: 16000},
		{name: "partial sample", pcm: []byte{0}, rate: 16000},
		{name: "zero rate", pcm: pcm},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := EncodePCM16WAV(tc.pcm, tc.rate); err == nil {
				t.Fatal("accepted invalid input")
			}
		})
	}
}

func TestDecodePCM16FLAC(t *testing.T) {
	raw, err := scoreboard.TestdataFiles.ReadFile("testdata/speech/orange-24000.flac")
	if err != nil {
		t.Fatal(err)
	}

	t.Run("valid", func(t *testing.T) {
		got, rate, err := DecodePCM16FLAC(raw)
		if err != nil {
			t.Fatal(err)
		}
		if rate != 24000 || len(got) == 0 || len(got)%2 != 0 {
			t.Fatalf("rate=%d bytes=%d", rate, len(got))
		}
	})

	t.Run("round trip", func(t *testing.T) {
		want, rate, err := DecodePCM16FLAC(raw)
		if err != nil {
			t.Fatal(err)
		}
		encoded, err := EncodePCM16FLAC(want, rate)
		if err != nil {
			t.Fatal(err)
		}
		got, gotRate, err := DecodePCM16FLAC(encoded)
		if err != nil {
			t.Fatal(err)
		}
		if gotRate != rate || !slices.Equal(got, want) {
			t.Fatalf("rate=%d bytes=%d, want rate=%d bytes=%d", gotRate, len(got), rate, len(want))
		}
	})

	t.Run("error", func(t *testing.T) {
		for _, b := range [][]byte{
			nil,
			[]byte("not FLAC"),
			raw[:len(raw)/2],
		} {
			if _, _, err := DecodePCM16FLAC(b); err == nil {
				t.Fatal("accepted invalid FLAC")
			}
		}
	})
}

func TestEncodePCM16FLAC(t *testing.T) {
	pcm := []byte{0, 0, 1, 0, 255, 255}
	raw, err := EncodePCM16FLAC(pcm, 16000)
	if err != nil {
		t.Fatal(err)
	}
	got, rate, err := DecodePCM16FLAC(raw)
	if err != nil {
		t.Fatal(err)
	}
	if rate != 16000 || !slices.Equal(got, pcm) {
		t.Fatalf("rate=%d data=%v", rate, got)
	}

	for _, tc := range []struct {
		name string
		pcm  []byte
		rate int
	}{
		{name: "empty", rate: 16000},
		{name: "partial sample", pcm: []byte{0}, rate: 16000},
		{name: "zero rate", pcm: pcm},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := EncodePCM16FLAC(tc.pcm, tc.rate); err == nil {
				t.Fatal("accepted invalid input")
			}
		})
	}
}
