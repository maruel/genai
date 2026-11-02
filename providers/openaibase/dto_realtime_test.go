// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Tests for OpenAI realtime session update wire types.

package openaibase_test

import (
	"encoding/json"
	"testing"

	"github.com/maruel/genai/providers/openaibase"
)

func TestSessionUpdateMessage(t *testing.T) {
	t.Parallel()
	instructions := ""
	tools := openaibase.ToolList{}
	choice := openaibase.ToolChoice{Mode: openaibase.ToolChoiceNone}
	tokens := openaibase.MaxOutputTokens{Value: 12}
	mods := openaibase.Modalities{"text"}
	voice := openaibase.Voice{Name: "marin"}
	audio := openaibase.SessionUpdateAudio{Input: openaibase.SessionUpdateAudioInput{TurnDetection: openaibase.Nullable[openaibase.TurnDetection]{Set: true}}, Output: openaibase.SessionUpdateAudioOutput{Voice: &voice}}
	for _, tc := range []struct {
		name string
		msg  openaibase.SessionUpdateMessage
		want string
	}{
		{
			name: "one field",
			msg:  openaibase.SessionUpdateMessage{Type: openaibase.EventSessionUpdate, Session: &openaibase.SessionUpdate{Instructions: strPtr("hello")}},
			want: `{"type":"session.update","session":{"instructions":"hello"}}`,
		},
		{
			name: "clear and replace",
			msg: openaibase.SessionUpdateMessage{Type: openaibase.EventSessionUpdate, Session: &openaibase.SessionUpdate{
				Instructions:     &instructions,
				Tools:            tools,
				ToolChoice:       choice,
				MaxOutputTokens:  tokens,
				OutputModalities: mods,
				Audio:            audio,
			}},
			want: `{"type":"session.update","session":{"instructions":"","audio":{"input":{"turn_detection":null},"output":{"voice":"marin"}},"max_output_tokens":12,"output_modalities":["text"],"tool_choice":"none","tools":[]}}`,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			b, err := json.Marshal(&tc.msg)
			if err != nil {
				t.Fatal(err)
			}
			if string(b) != tc.want {
				t.Fatalf("%s != %s", b, tc.want)
			}
		})
	}
}

func strPtr(v string) *string { return &v }

func TestSessionUpdateFull(t *testing.T) {
	t.Parallel()
	instructions := "be terse"
	parallel := false
	prompt := openaibase.Prompt{ID: "pmt_1", Version: "1"}
	speed := 1.5
	voice := openaibase.Voice{Name: "marin"}
	truncation := openaibase.Truncation{Strategy: openaibase.TruncationAuto}
	now := openaibase.Nullable[openaibase.Tracing]{Set: true}
	noise := openaibase.Nullable[openaibase.NoiseReduction]{Set: true, Value: &openaibase.NoiseReduction{Type: openaibase.NoiseNearField}}
	transcription := openaibase.Nullable[openaibase.Transcription]{Set: true}
	turn := openaibase.Nullable[openaibase.TurnDetection]{Set: true}
	msg := openaibase.SessionUpdateMessage{EventID: "evt_1", Type: openaibase.EventSessionUpdate, Session: &openaibase.SessionUpdate{
		Type:              openaibase.SessionRealtime,
		Instructions:      &instructions,
		ParallelToolCalls: &parallel,
		Prompt:            &prompt,
		Tracing:           now,
		Audio: openaibase.SessionUpdateAudio{
			Input: openaibase.SessionUpdateAudioInput{
				Format:         openaibase.AudioFormat{Type: openaibase.AudioPCM, Rate: 24000},
				NoiseReduction: noise,
				Transcription:  transcription,
				TurnDetection:  turn,
			},
			Output: openaibase.SessionUpdateAudioOutput{
				Format: openaibase.AudioFormat{Type: openaibase.AudioPCM, Rate: 24000},
				Speed:  &speed,
				Voice:  &voice,
			},
		},
		Include:          []openaibase.IncludeKind{openaibase.IncludeLogprobs},
		MaxOutputTokens:  openaibase.MaxOutputTokens{Value: 128},
		OutputModalities: openaibase.Modalities{"audio"},
		Reasoning:        &openaibase.ReasoningConfig{Effort: openaibase.ReasoningEffortLow},
		ToolChoice:       openaibase.ToolChoice{Mode: openaibase.ToolChoiceAuto},
		Tools:            openaibase.ToolList{{Type: openaibase.ToolFunction, Name: "lookup", Parameters: json.RawMessage(`{"type":"object"}`)}},
		Truncation:       &truncation,
	}}
	b, err := json.Marshal(&msg)
	if err != nil {
		t.Fatal(err)
	}
	want := `{"event_id":"evt_1","type":"session.update","session":{"type":"realtime","instructions":"be terse","parallel_tool_calls":false,"prompt":{"id":"pmt_1","version":"1"},"tracing":null,"audio":{"input":{"format":{"type":"audio/pcm","rate":24000},"noise_reduction":{"type":"near_field"},"transcription":null,"turn_detection":null},"output":{"format":{"type":"audio/pcm","rate":24000},"speed":1.5,"voice":"marin"}},"include":["item.input_audio_transcription.logprobs"],"max_output_tokens":128,"output_modalities":["audio"],"reasoning":{"effort":"low"},"tool_choice":"auto","tools":[{"type":"function","name":"lookup","parameters":{"type":"object"}}],"truncation":"auto"}}`
	if string(b) != want {
		t.Fatalf("%s != %s", b, want)
	}
}

func TestNullable(t *testing.T) {
	t.Parallel()
	type wrapper struct {
		V openaibase.Nullable[openaibase.Tracing] `json:"v,omitzero"`
	}
	for _, tc := range []struct {
		name string
		in   wrapper
		want string
	}{
		{"absent", wrapper{}, `{}`},
		{"null", wrapper{V: openaibase.Nullable[openaibase.Tracing]{Set: true}}, `{"v":null}`},
		{"value", wrapper{V: openaibase.Nullable[openaibase.Tracing]{Set: true, Value: &openaibase.Tracing{Auto: true}}}, `{"v":"auto"}`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			b, err := json.Marshal(&tc.in)
			if err != nil {
				t.Fatal(err)
			}
			if string(b) != tc.want {
				t.Fatalf("%s != %s", b, tc.want)
			}
		})
	}
}
