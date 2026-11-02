// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Realtime speech fixtures and PCM helpers.

package smoke

import (
	"bytes"
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"strings"

	flacpcm "github.com/tphakala/go-flac/pcm"

	"github.com/maruel/genai/scoreboard"
)

// ResamplePCM16 resamples mono PCM16LE audio using linear interpolation.
//
// It is intentionally test-only. Production callers must resample at their
// capture or playback boundary.
func ResamplePCM16(in []byte, fromRate, toRate int) ([]byte, error) {
	if fromRate <= 0 || toRate <= 0 {
		return nil, errors.New("resample rates must be positive")
	}
	if len(in)%2 != 0 {
		return nil, errors.New("PCM16 input must contain complete samples")
	}
	n := len(in) / 2
	if n == 0 {
		return []byte{}, nil
	}
	outN := int((int64(n)*int64(toRate) + int64(fromRate)/2) / int64(fromRate))
	if outN == 0 {
		outN = 1
	}
	out := make([]byte, outN*2)
	from := int64(fromRate)
	to := int64(toRate)
	sample := func(i int) int64 {
		return int64(int16(binary.LittleEndian.Uint16(in[i*2:])))
	}
	for i := range outN {
		pos := int64(i) * from
		j := pos / to
		frac := pos % to
		if j >= int64(n) {
			j = int64(n - 1)
			frac = 0
		}
		s0 := sample(int(j))
		s1 := s0
		if j+1 < int64(n) {
			s1 = sample(int(j) + 1)
		}
		v := (s0*(to-frac) + s1*frac) / to
		binary.LittleEndian.PutUint16(out[i*2:], uint16(int16(v)))
	}
	return out, nil
}

// speech returns synthesized speech for a realtime audio probe.
//
// The recorded phrases match the smoke runner's audio prompts. It returns mono
// PCM16LE at rate. It uses the 16 kHz or 24 kHz FLAC fixture and resamples only
// when the session requests another rate.
func speech(ctx context.Context, text string, rate int) ([]byte, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if rate <= 0 {
		return nil, errors.New("speech rate must be positive")
	}
	base := ""
	switch {
	case strings.Contains(text, "orange"):
		base = "orange"
	case strings.Contains(text, "access code"):
		base = "code"
	default:
		return nil, fmt.Errorf("no speech fixture for %q", text)
	}
	fixtureRate := 16000
	if rate >= 24000 {
		fixtureRate = 24000
	}
	raw, err := scoreboard.TestdataFiles.ReadFile(fmt.Sprintf("testdata/speech/%s-%d.flac", base, fixtureRate))
	if err != nil {
		return nil, err
	}
	pcm, gotRate, err := DecodePCM16FLAC(raw)
	if err != nil {
		return nil, fmt.Errorf("decode %s speech: %w", base, err)
	}
	if gotRate == rate {
		return pcm, nil
	}
	return ResamplePCM16(pcm, gotRate, rate)
}

// DecodePCM16WAV returns the data chunk and sample rate of a mono PCM16 WAV.
func DecodePCM16WAV(raw []byte) ([]byte, int, error) {
	if len(raw) < 12 || string(raw[:4]) != "RIFF" || string(raw[8:12]) != "WAVE" {
		return nil, 0, errors.New("not a RIFF/WAVE file")
	}
	riffSize := int(binary.LittleEndian.Uint32(raw[4:8]))
	if riffSize < 4 || riffSize != len(raw)-8 {
		return nil, 0, errors.New("WAV RIFF size does not match file size")
	}
	const (
		formatPCM = 1
	)
	var data []byte
	var sampleRate, channels, bits int
	format := 0
	end := len(raw)
	for off := 12; off < end; {
		if off+8 > end {
			return nil, 0, errors.New("truncated WAV chunk header")
		}
		id := string(raw[off : off+4])
		size := int(binary.LittleEndian.Uint32(raw[off+4 : off+8]))
		if size > end-(off+8) {
			return nil, 0, errors.New("truncated WAV chunk")
		}
		body := raw[off+8 : off+8+size]
		switch id {
		case "fmt ":
			if len(body) < 16 {
				return nil, 0, errors.New("short WAV format chunk")
			}
			format = int(binary.LittleEndian.Uint16(body[0:2]))
			channels = int(binary.LittleEndian.Uint16(body[2:4]))
			sampleRate = int(binary.LittleEndian.Uint32(body[4:8]))
			bits = int(binary.LittleEndian.Uint16(body[14:16]))
		case "data":
			data = body
		}
		off += 8 + size
		if size%2 != 0 {
			if off >= end {
				return nil, 0, errors.New("missing WAV chunk padding")
			}
			off++
		}
	}
	if format != formatPCM || channels != 1 || bits != 16 {
		return nil, 0, fmt.Errorf("unsupported WAV format: format=%d channels=%d bits=%d", format, channels, bits)
	}
	if sampleRate <= 0 {
		return nil, 0, errors.New("WAV sample rate must be positive")
	}
	if len(data) == 0 || len(data)%2 != 0 {
		return nil, 0, errors.New("WAV data must contain complete PCM16 samples")
	}
	return data, sampleRate, nil
}

// DecodePCM16FLAC returns the data chunk and sample rate of a mono PCM16 FLAC.
func DecodePCM16FLAC(raw []byte) ([]byte, int, error) {
	pcm, info, err := flacpcm.DecodeInterleaved(bytes.NewReader(raw))
	if err != nil {
		return nil, 0, err
	}
	if info.Channels != 1 || info.BitDepth != 16 {
		return nil, 0, fmt.Errorf("unsupported FLAC format: channels=%d bits=%d", info.Channels, info.BitDepth)
	}
	if info.SampleRate <= 0 {
		return nil, 0, errors.New("FLAC sample rate must be positive")
	}
	if len(pcm) == 0 || len(pcm)%2 != 0 {
		return nil, 0, errors.New("FLAC data must contain complete PCM16 samples")
	}
	return pcm, info.SampleRate, nil
}

// EncodePCM16FLAC returns a mono PCM16 FLAC containing pcm at rate.
func EncodePCM16FLAC(pcm []byte, rate int) ([]byte, error) {
	if rate <= 0 {
		return nil, errors.New("FLAC sample rate must be positive")
	}
	if len(pcm) == 0 {
		return nil, errors.New("FLAC data must not be empty")
	}
	if len(pcm)%2 != 0 {
		return nil, errors.New("FLAC data must contain complete PCM16 samples")
	}
	var buf bytes.Buffer
	if err := flacpcm.EncodeInterleaved(&buf, flacpcm.Config{SampleRate: rate, BitDepth: 16, Channels: 1, CompressionLevel: 8}, pcm); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

// EncodePCM16WAV returns a mono PCM16 WAV containing pcm at rate.
func EncodePCM16WAV(pcm []byte, rate int) ([]byte, error) {
	if rate <= 0 {
		return nil, errors.New("WAV sample rate must be positive")
	}
	if len(pcm) == 0 {
		return nil, errors.New("WAV data must not be empty")
	}
	if len(pcm)%2 != 0 {
		return nil, errors.New("WAV data must contain complete PCM16 samples")
	}
	const (
		headerSize = 44
		formatPCM  = 1
	)
	out := make([]byte, headerSize+len(pcm))
	copy(out[0:4], "RIFF")
	binary.LittleEndian.PutUint32(out[4:8], uint32(len(out)-8))
	copy(out[8:12], "WAVE")
	copy(out[12:16], "fmt ")
	binary.LittleEndian.PutUint32(out[16:20], 16)
	binary.LittleEndian.PutUint16(out[20:22], formatPCM)
	binary.LittleEndian.PutUint16(out[22:24], 1)
	binary.LittleEndian.PutUint32(out[24:28], uint32(rate))
	binary.LittleEndian.PutUint32(out[28:32], uint32(rate*2))
	binary.LittleEndian.PutUint16(out[32:34], 2)
	binary.LittleEndian.PutUint16(out[34:36], 16)
	copy(out[36:40], "data")
	binary.LittleEndian.PutUint32(out[40:44], uint32(len(pcm)))
	copy(out[headerSize:], pcm)
	return out, nil
}
