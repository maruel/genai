// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Tests for llama.cpp realtime wire types.

package llamacpp

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/maruel/genai"
)

func decodeRealtimeStrict(data string, out any) error {
	d := json.NewDecoder(strings.NewReader(data))
	d.DisallowUnknownFields()
	return d.Decode(out)
}

func TestServerMessage(t *testing.T) {
	t.Parallel()
	// Shapes emitted by server-realtime.cpp in llama.cpp v0.9.0.
	for _, data := range []string{
		`{"type":"input_audio_buffer.committed","previous_item_id":null,"item_id":"item_1","event_id":"event_1"}`,
		`{"type":"conversation.item.added","previous_item_id":null,"item":{"id":"item_1","object":"realtime.item","type":"message","role":"user","status":"completed","content":[{"type":"input_audio"}]},"event_id":"event_1"}`,
		`{"type":"response.content_part.added","response_id":"resp_1","item_id":"item_1","output_index":0,"content_index":0,"part":{"type":"output_audio","transcript":""},"event_id":"event_1"}`,
		`{"type":"response.output_audio.done","response_id":"resp_1","item_id":"item_1","output_index":0,"content_index":0,"event_id":"event_1"}`,
		`{"type":"response.output_item.done","response_id":"resp_1","output_index":0,"item":{"id":"item_1","object":"realtime.item","type":"function_call","status":"cancelled","name":"lookup","call_id":"call_1","arguments":"{}"},"event_id":"event_1"}`,
		`{"type":"response.done","response":{"id":"resp_1","object":"realtime.response","status":"completed","output":[{"id":"item_1","object":"realtime.item","type":"message","role":"assistant","status":"completed","content":[{"type":"output_text","text":"answer"}]}],"usage":{"output_tokens":3}},"event_id":"event_1"}`,
		`{"type":"session.created","session":{"id":"sess_1","object":"realtime.transcription_session","type":"transcription","audio":{"input":{"format":{"type":"audio/pcm","rate":16000},"turn_detection":null,"transcription":{"model":"asr","language":null}}}},"event_id":"event_1"}`,
	} {
		var msg ServerMessage
		if err := decodeRealtimeStrict(data, &msg); err != nil {
			t.Fatalf("%s: %v", data, err)
		}
	}
}

func TestResponseConfig(t *testing.T) {
	t.Parallel()
	msg := ClientMessage{Type: EventCreate, Response: &ResponseConfig{
		OutputModalities: Modalities{"text"},
		MaxOutputTokens:  &MaxOutputTokens{Value: 128},
		ToolChoice:       ToolChoiceRequired,
	}}
	b, err := json.Marshal(msg)
	if err != nil {
		t.Fatal(err)
	}
	if want := `{"type":"response.create","response":{"output_modalities":["text"],"max_output_tokens":128,"tool_choice":"required"}}`; string(b) != want {
		t.Fatalf("got %s, want %s", b, want)
	}
}

func TestMaxOutputTokens(t *testing.T) {
	t.Parallel()
	for _, c := range []struct {
		in   string
		want MaxOutputTokens
	}{
		{`128`, MaxOutputTokens{Value: 128}},
		{`"inf"`, MaxOutputTokens{Infinite: true}},
	} {
		var m MaxOutputTokens
		if err := json.Unmarshal([]byte(c.in), &m); err != nil {
			t.Fatal(err)
		}
		if m != c.want {
			t.Errorf("%s: got %+v, want %+v", c.in, m, c.want)
		}
		b, err := json.Marshal(m)
		if err != nil {
			t.Fatal(err)
		}
		if string(b) != c.in {
			t.Errorf("%+v: got %s, want %s", m, b, c.in)
		}
	}
}

func TestSessionUpdateMessage(t *testing.T) {
	t.Parallel()
	instructions := ""
	tools := ToolList{}
	choice := ToolChoiceNone
	tokens := MaxOutputTokens{Value: 12}
	mods := Modalities{genai.ModalityText}
	for _, tc := range []struct {
		name string
		msg  SessionUpdateMessage
		want string
	}{
		{
			name: "one field",
			msg:  SessionUpdateMessage{Type: EventSessionUpdate, Session: &SessionUpdate{Instructions: &instructions}},
			want: `{"type":"session.update","session":{"instructions":""}}`,
		},
		{
			name: "replace",
			msg: SessionUpdateMessage{Type: EventSessionUpdate, Session: &SessionUpdate{
				Tools:            tools,
				ToolChoice:       choice,
				MaxOutputTokens:  tokens,
				OutputModalities: mods,
			}},
			want: `{"type":"session.update","session":{"tools":[],"tool_choice":"none","max_output_tokens":12,"output_modalities":["text"]}}`,
		},
		{
			name: "session and audio",
			msg: SessionUpdateMessage{Type: EventSessionUpdate, Session: &SessionUpdate{
				Type: SessionRealtime,
				Audio: SessionUpdateAudio{Input: SessionUpdateAudioInput{
					Format:        SessionUpdateAudioFormat{Type: AudioPCM, Rate: 16000},
					Transcription: SessionUpdateTranscription{Language: "en"},
				}},
			}},
			want: `{"type":"session.update","session":{"type":"realtime","audio":{"input":{"format":{"type":"audio/pcm","rate":16000},"transcription":{"language":"en"}}}}}`,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
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
