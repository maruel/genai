// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Tests dedicated live-ASR clients against native WebSocket peers.

package llamacpp_test

import (
	"context"
	"errors"
	"net/http"
	"testing"

	"github.com/coder/websocket"
	"github.com/maruel/genai/internal"
	"github.com/maruel/genai/internal/realtimetest"
	"github.com/maruel/genai/providers/llamacpp"
)

// The transcription acknowledgement is distinct from a conversational session.
// Source: ggml-org/llama.cpp cd959bae99ff4bed22bef2604f45241c09bbb8c8,
// tools/server/server-context.cpp session_config in handle_realtime.
// ID/model are representative dynamic values; all native keys are retained.
const nativeTranscriptionSession = `{"type":"session.created","session":{"id":"sess_native","object":"realtime.transcription_session","type":"transcription","audio":{"input":{"format":{"type":"audio/pcm","rate":16000},"transcription":{"model":"native-asr"},"turn_detection":null,"noise_reduction":null}}}}`

func testClientConnectTranscriptionRaw(t *testing.T) {
	t.Parallel()
	t.Run("native deltas and explicit drain", func(t *testing.T) {
		t.Parallel()
		if internal.BeLenient {
			t.Fatal("native handshake requires strict decoding")
		}
		endpoint := realtimetest.Peer(t, func(r *http.Request) {
			if r.URL.RawQuery != "intent=transcription" {
				t.Error(r.URL)
			}
		}, func(ctx context.Context, ws *websocket.Conn) {
			realtimetest.Write(t, ctx, ws, nativeTranscriptionSession)
			m := realtimetest.Read(t, ctx, ws)
			if m["type"] != "input_audio_buffer.append" {
				t.Error(m)
			}
			realtimetest.Write(t, ctx, ws, `{"type":"conversation.item.input_audio_transcription.delta","item_id":"u","delta":"hello"}`)
			m = realtimetest.Read(t, ctx, ws)
			if m["final"] != true {
				t.Error(m)
			}
			realtimetest.Write(t, ctx, ws, `{"type":"conversation.item.input_audio_transcription.completed","item_id":"u","transcript":"Hello."}`)
		})
		s, err := realtimeClient(t, endpoint).ConnectTranscriptionRaw(t.Context())
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() {
			if err := s.Close(); err != nil {
				t.Error(err)
			}
		})
		if s.Initial.Session == nil || s.Initial.Session.ID != "sess_native" || s.Initial.Session.Object != llamacpp.ObjectTranscriptionSession || s.Initial.Session.Audio.Input.Format.Rate != 16000 || s.Initial.Session.Audio.Input.Transcription == nil || s.Initial.Session.Audio.Input.Transcription.Model != "native-asr" || string(s.Initial.Session.Audio.Input.NoiseReduction) != "null" {
			t.Fatalf("native session %+v", s.Initial)
		}
		if err := s.Send(t.Context(), nil); err == nil {
			t.Fatal("nil native message accepted")
		}
		if err := s.Send(t.Context(), &llamacpp.ClientMessage{Type: llamacpp.EventAudioAppend, Audio: []byte{0, 0}}); err != nil {
			t.Fatal(err)
		}
		event, err := s.Receive(t.Context())
		if err != nil || event.Type != llamacpp.EventInputTranscriptDelta || event.ItemID != "u" || event.Delta != "hello" {
			t.Fatalf("native delta %+v: %v", event, err)
		}
		if err := s.Send(t.Context(), &llamacpp.ClientMessage{Type: llamacpp.EventAudioCommit, Final: true}); err != nil {
			t.Fatal(err)
		}
		event, err = s.Receive(t.Context())
		if err != nil || event.Type != llamacpp.EventInputTranscriptDone || event.ItemID != "u" || event.Transcript != "Hello." {
			t.Fatalf("native completed event %+v: %v", event, err)
		}
		cause := errors.New("caller ended native session")
		s.Abort(cause)
		if _, err := s.Receive(t.Context()); !errors.Is(err, cause) {
			t.Fatalf("abort cause %v", err)
		}
	})
}
