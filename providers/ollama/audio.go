// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Audio endpoint requests and streamed response handling.
// Wire definitions follow github.com/ollama/ollama at
// 2e7b870fbcec32da4b8864a1f715d7809c801490, server/speech.go,
// server/transcription.go, and openai/openai.go.

package ollama

import (
	"errors"
	"io"
	"math"
	"strings"
	"unicode/utf8"
)

// SpeechRequest generates speech through /v1/audio/speech.
// Output is mono 24 kHz signed 16-bit audio. Format defaults to wav.
type SpeechRequest struct {
	Model          string   `json:"model"`
	Input          string   `json:"input"`
	Voice          string   `json:"voice,omitzero"`
	ResponseFormat string   `json:"response_format,omitzero"`
	Speed          *float64 `json:"speed,omitzero"`
	Instructions   string   `json:"instructions,omitzero"`
	StreamFormat   string   `json:"stream_format,omitzero"`
}

// Validate checks the speech endpoint's supported parameters.
func (r *SpeechRequest) Validate() error {
	if strings.TrimSpace(r.Model) == "" || strings.TrimSpace(r.Input) == "" {
		return errors.New("model and input are required")
	}
	if utf8.RuneCountInString(r.Input) > 4096 {
		return errors.New("input must be at most 4096 characters")
	}
	if r.ResponseFormat != "" && r.ResponseFormat != "wav" && r.ResponseFormat != "pcm" {
		return errors.New("response_format must be wav or pcm")
	}
	if r.Speed != nil && *r.Speed != 1 {
		return errors.New("only speed 1 is supported")
	}
	if r.Instructions != "" {
		return errors.New("instructions are not supported")
	}
	if r.StreamFormat != "" && r.StreamFormat != "audio" {
		return errors.New("only stream_format audio is supported")
	}
	return nil
}

// AudioTextRequest uploads audio for transcription or translation.
// File is read incrementally and remains owned by the caller. Ollama accepts WAV.
// ResponseFormat defaults to json; text and verbose_json are also supported.
// Stream requests SSE transcript.text.delta and transcript.text.done events.
type AudioTextRequest struct {
	Model          string
	File           io.Reader
	Filename       string
	Language       string
	ToLanguage     string
	Prompt         string
	ResponseFormat string
	Temperature    float64
	Stream         bool
}

// Validate checks parameters common to transcription and translation.
func (r *AudioTextRequest) Validate() error {
	if strings.TrimSpace(r.Model) == "" || r.File == nil {
		return errors.New("model and file are required")
	}
	switch r.ResponseFormat {
	case "", "json", "text", "verbose_json":
	default:
		return errors.New("response_format must be json, text or verbose_json")
	}
	if math.IsNaN(r.Temperature) || math.IsInf(r.Temperature, 0) {
		return errors.New("temperature must be finite")
	}
	return nil
}

// AudioTextResponse describes json and verbose_json transcription responses.
type AudioTextResponse struct {
	Text     string             `json:"text"`
	Task     string             `json:"task,omitzero"`
	Language string             `json:"language,omitzero"`
	Duration float64            `json:"duration,omitzero"`
	Segments []AudioTextSegment `json:"segments,omitzero"`
}

// AudioTextSegment identifies the text and time range of one audio clip.
type AudioTextSegment struct {
	ID    int     `json:"id"`
	Start float64 `json:"start"`
	End   float64 `json:"end"`
	Text  string  `json:"text"`
}

// AudioTextEvent is an SSE transcription delta or completed transcript.
type AudioTextEvent struct {
	Type  string        `json:"type"`
	Delta string        `json:"delta,omitzero"`
	Text  string        `json:"text,omitzero"`
	Error *ErrorDetails `json:"error,omitzero"`
}
