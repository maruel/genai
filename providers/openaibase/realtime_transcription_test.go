// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Tests for OpenAI transcription session paths the recorded smoke probe never takes.

package openaibase

import (
	"context"
	"errors"
	"net/http"
	"strings"
	"testing"

	"github.com/coder/websocket"

	"github.com/maruel/genai"
	"github.com/maruel/genai/base"
	"github.com/maruel/genai/internal/realtimetest"
)

const transcriptionSessionJSON = `{"id":"sess","object":"realtime.transcription_session","type":"transcription","audio":{"input":{"format":{"type":"audio/pcm","rate":24000},"turn_detection":null}}}`

func TestConnectTranscriptionSession(t *testing.T) {
	connect := func(t *testing.T, endpoint string, opts ...genai.RealtimeOption) (genai.RealtimeSession, error) {
		opts = append([]genai.RealtimeOption{genai.RealtimeOptionTranscription{}}, opts...)
		if endpoint != "" {
			opts = append(opts, &RealtimeOption{Endpoint: "ws" + strings.TrimPrefix(endpoint, "http")})
		}
		return ConnectRealtimeSession(t.Context(), http.DefaultClient, "gpt-live-transcribe", opts...)
	}
	t.Run("options", func(t *testing.T) {
		for _, opts := range [][]genai.RealtimeOption{
			{&genai.RealtimeOptionAudio{InputFormat: genai.AudioFormat{SampleRate: 16000}, Voice: "v", OutputTranscription: true}},
			{genai.RealtimeTurnServer},
			{&RealtimeOption{InputTranscriptionModel: "m", MaxOutputTokens: 1, Truncation: &Truncation{Strategy: TruncationAuto}}},
			{genai.RealtimeOptionInstructions("x")},
			{&genai.RealtimeOptionAudio{InputLanguage: "fil"}},
		} {
			if _, err := connect(t, "", opts...); !errors.As(err, new(*base.ErrNotSupported)) {
				t.Errorf("%+v: %v", opts, err)
			}
		}
		for _, opts := range [][]genai.RealtimeOption{{nil}, {genai.RealtimeTurnManual, genai.RealtimeTurnManual}} {
			if _, err := connect(t, "", opts...); err == nil {
				t.Errorf("%+v accepted", opts)
			}
		}
		if _, err := ConnectRealtimeSession(t.Context(), http.DefaultClient, "", genai.RealtimeOptionTranscription{}); err == nil {
			t.Error("empty model accepted")
		}
	})
	t.Run("rejected setup", func(t *testing.T) {
		endpoint := realtimetest.Peer(t, func(r *http.Request) {
			if r.URL.RawQuery != "intent=transcription" {
				t.Error(r.URL)
			}
		}, func(ctx context.Context, ws *websocket.Conn) {
			realtimetest.Write(t, ctx, ws, `{"type":"session.created","event_id":"e","session":`+transcriptionSessionJSON+`}`)
			realtimetest.Read(t, ctx, ws)
			realtimetest.Write(t, ctx, ws, `{"type":"error","event_id":"e","error":{"code":"invalid_value","message":"secret"}}`)
		})
		_, err := connect(t, endpoint)
		if e, ok := errors.AsType[*genai.RealtimeError](err); !ok || e.Code != "invalid_value" || strings.Contains(e.Message, "secret") {
			t.Fatal(err)
		}
	})
	t.Run("session", func(t *testing.T) {
		endpoint := realtimetest.Peer(t, nil, func(ctx context.Context, ws *websocket.Conn) {
			realtimetest.Write(t, ctx, ws, `{"type":"session.created","event_id":"e","session":`+transcriptionSessionJSON+`}`)
			realtimetest.Read(t, ctx, ws)
			realtimetest.Write(t, ctx, ws, `{"type":"session.updated","event_id":"e","session":`+transcriptionSessionJSON+`}`)
			realtimetest.Write(t, ctx, ws, `{"type":"conversation.item.input_audio_transcription.failed","event_id":"e","item_id":"i","error":{"code":"c"}}`)
			_, _, _ = ws.Read(ctx)
		})
		s, err := connect(t, endpoint)
		if err != nil {
			t.Fatal(err)
		}
		realtimetest.Close(t, s)
		for _, in := range []genai.RealtimeInput{genai.RealtimeInputContent{Audio: []byte{0, 0}, Generate: true}, genai.RealtimeCancel{}} {
			if err := s.Send(t.Context(), in); !errors.As(err, new(*base.ErrNotSupported)) {
				t.Errorf("%+v: %v", in, err)
			}
		}
		if err := s.Send(t.Context(), genai.RealtimeInputContent{Commit: true}); err != nil {
			t.Fatal(err)
		}
		if err := s.Send(t.Context(), genai.RealtimeInputContent{Audio: []byte{0, 0}}); err == nil {
			t.Error("audio after commit accepted")
		}
		if e := realtimetest.Receive(t, s, genai.RealtimeEventError); e.Error.Code != "c" || e.ItemID != "i" {
			t.Errorf("%+v", e)
		}
		realtimetest.Receive(t, s, genai.RealtimeEventTurnDone)
	})
}
