// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Tests for Mistral transcription session paths the recorded smoke probe never takes.

package mistral_test

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/coder/websocket"

	"github.com/maruel/genai"
	"github.com/maruel/genai/base"
	"github.com/maruel/genai/internal/realtimetest"
	"github.com/maruel/genai/providers/mistral"
)

const sessionJSON = `{"request_id":"r","model":"m","audio_format":{"encoding":"pcm_s16le","sample_rate":16000}}`

func TestClientConnectRealtime(t *testing.T) {
	c, err := mistral.New(t.Context(), genai.ProviderOptionAPIKey("k"), genai.ProviderOptionModel("voxtral-mini-transcribe-realtime-3"))
	if err != nil {
		t.Fatal(err)
	}
	t.Run("options", func(t *testing.T) {
		for _, opts := range [][]genai.RealtimeOption{
			{},
			{genai.RealtimeOptionTranscription{}, &genai.RealtimeOptionAudio{Voice: "v", OutputTranscription: true}},
			{genai.RealtimeOptionTranscription{}, genai.RealtimeTurnServer},
			{genai.RealtimeOptionTranscription{}, genai.RealtimeOptionInstructions("x")},
			{genai.RealtimeOptionTranscription{}, &genai.RealtimeOptionAudio{InputLanguage: "fil"}},
		} {
			if _, err := c.ConnectRealtime(t.Context(), opts...); !errors.As(err, new(*base.ErrNotSupported)) {
				t.Errorf("%+v: %v", opts, err)
			}
		}
		for _, opts := range [][]genai.RealtimeOption{
			{nil},
			{genai.RealtimeOptionTranscription{}, genai.RealtimeOptionTranscription{}},
			{genai.RealtimeOptionTranscription{}, &mistral.RealtimeOption{Endpoint: "http://x"}},
		} {
			if _, err := c.ConnectRealtime(t.Context(), opts...); err == nil {
				t.Errorf("%+v accepted", opts)
			}
		}
	})
	t.Run("rejected update", func(t *testing.T) {
		endpoint := peer(t, func(ctx context.Context, ws *websocket.Conn) {
			realtimetest.Write(t, ctx, ws, `{"type":"session.created","session":`+sessionJSON+`}`)
			realtimetest.Read(t, ctx, ws)
			realtimetest.Write(t, ctx, ws, `{"type":"error","error":{"message":"bad language","code":4000}}`)
		})
		_, err := c.ConnectRealtime(t.Context(), genai.RealtimeOptionTranscription{}, &genai.RealtimeOptionAudio{InputLanguage: "fr"}, &mistral.RealtimeOption{Endpoint: endpoint})
		if e, ok := errors.AsType[*genai.RealtimeError](err); !ok || e.Code != "4000" || e.Message != "bad language" {
			t.Fatal(err)
		}
	})
	t.Run("session", func(t *testing.T) {
		endpoint := peer(t, func(ctx context.Context, ws *websocket.Conn) {
			realtimetest.Write(t, ctx, ws, `{"type":"session.created","session":`+sessionJSON+`}`)
			realtimetest.Read(t, ctx, ws)
			realtimetest.Write(t, ctx, ws, `{"type":"session.updated","session":`+sessionJSON+`}`)
			realtimetest.Write(t, ctx, ws, `{"type":"error","error":{"message":{"detail":"x"},"code":5000}}`)
			_, _, _ = ws.Read(ctx)
		})
		s, err := c.ConnectRealtime(t.Context(), genai.RealtimeOptionTranscription{}, &mistral.RealtimeOption{Endpoint: endpoint})
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
		e := realtimetest.Receive(t, s, genai.RealtimeEventError)
		if e.Error.Code != "5000" || e.Error.Message != `{"detail":"x"}` {
			t.Errorf("%+v", e.Error)
		}
	})
}

func peer(t *testing.T, script func(context.Context, *websocket.Conn)) string {
	return "ws" + strings.TrimPrefix(realtimetest.Peer(t, nil, script), "http")
}
