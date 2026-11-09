// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Tests dedicated live-ASR clients against native WebSocket peers.

package gemini_test

import (
	"context"
	"errors"
	"net/http"
	"net/url"
	"testing"

	"github.com/coder/websocket"
	"github.com/maruel/genai"
	"github.com/maruel/genai/internal/internaltest"
	"github.com/maruel/genai/internal/realtimetest"
	"github.com/maruel/genai/providers/gemini"
)

func testClientConnectLiveRaw(t *testing.T) {
	t.Parallel()
	t.Run("native segments and generation completion", func(t *testing.T) {
		t.Parallel()
		endpoint := livePeer(t, func(ctx context.Context, ws *websocket.Conn, setup map[string]any) {
			if internaltest.AssertType[map[string]any](t, setup["setup"])["model"] != "models/gemini-3.5-transcribe-live" {
				t.Error(setup)
			}
			realtimetest.Read(t, ctx, ws)
			realtimetest.Read(t, ctx, ws)
			realtimetest.Write(t, ctx, ws, `{"serverContent":{"interimInputTranscription":{"text":"pear"}}}`)
			realtimetest.Write(t, ctx, ws, `{"serverContent":{"interimInputTranscription":{"text":"banana"}}}`)
			realtimetest.Write(t, ctx, ws, `{"serverContent":{"inputTranscription":{"text":"Banana."}}}`)
			realtimetest.Read(t, ctx, ws)
			realtimetest.Write(t, ctx, ws, `{"serverContent":{"inputTranscription":{"text":" Hello."}}}`)
			realtimetest.Write(t, ctx, ws, `{"serverContent":{"generationComplete":true}}`)
		})
		target, err := url.Parse(endpoint)
		if err != nil {
			t.Fatal(err)
		}
		c, err := gemini.New(t.Context(), genai.ProviderOptionAPIKey("test-live-key"), genai.ProviderOptionModel("gemini-3.5-transcribe-live"), genai.ProviderOptionTransportWrapper(func(next http.RoundTripper) http.RoundTripper {
			return speechTransport(func(r *http.Request) (*http.Response, error) {
				r = r.Clone(r.Context())
				r.URL.Scheme = "http"
				r.URL.Host = target.Host
				return next.RoundTrip(r)
			})
		}))
		if err != nil {
			t.Fatal(err)
		}
		s, err := c.ConnectLiveRaw(t.Context(), &gemini.LiveSetup{Model: "models/gemini-3.5-transcribe-live", InputAudioTranscription: &gemini.LiveTranscriptionConfig{}, GenerationConfig: gemini.LiveGenerationConfig{ResponseModalities: []gemini.Modality{gemini.ModalityText}}})
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() {
			if err := s.Close(); err != nil {
				t.Error(err)
			}
		})
		if s.Initial.SetupComplete == nil || s.Initial.SetupComplete.SessionID != "live-test" {
			t.Fatalf("native setup %+v", s.Initial)
		}
		if err := s.Send(t.Context(), nil); err == nil {
			t.Fatal("nil native message accepted")
		}
		if err := s.Send(t.Context(), &gemini.LiveClientMessage{RealtimeInput: &gemini.LiveRealtimeInput{ActivityStart: &gemini.LiveActivity{}}}); err != nil {
			t.Fatal(err)
		}
		if err := s.Send(t.Context(), &gemini.LiveClientMessage{RealtimeInput: &gemini.LiveRealtimeInput{Audio: &gemini.LiveBlob{MIMEType: "audio/pcm;rate=16000", Data: []byte{0, 0}}}}); err != nil {
			t.Fatal(err)
		}
		for _, text := range []string{"pear", "banana", "Banana."} {
			event, err := s.Receive(t.Context())
			if err != nil {
				t.Fatal(err)
			}
			content := event.ServerContent
			if content == nil {
				t.Fatalf("native content missing: %+v", event)
			}
			got := ""
			if content.InterimInputTranscription != nil {
				got = content.InterimInputTranscription.Text
			} else if content.InputTranscription != nil {
				got = content.InputTranscription.Text
			}
			if got != text {
				t.Fatalf("native segment %q, want %q", got, text)
			}
		}
		if err := s.Send(t.Context(), &gemini.LiveClientMessage{RealtimeInput: &gemini.LiveRealtimeInput{ActivityEnd: &gemini.LiveActivity{}}}); err != nil {
			t.Fatal(err)
		}
		event, err := s.Receive(t.Context())
		if err != nil || event.ServerContent == nil || event.ServerContent.InputTranscription == nil || event.ServerContent.InputTranscription.Text != " Hello." {
			t.Fatalf("native committed segment %+v: %v", event, err)
		}
		event, err = s.Receive(t.Context())
		if err != nil || event.ServerContent == nil || !event.ServerContent.GenerationComplete {
			t.Fatalf("native terminal frame %+v: %v", event, err)
		}
		cause := errors.New("caller ended native session")
		s.Abort(cause)
		if _, err := s.Receive(t.Context()); !errors.Is(err, cause) {
			t.Fatalf("abort cause %v", err)
		}
	})
}
