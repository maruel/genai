// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Tests Gemini Live against a scripted WebSocket peer, without credentials.

package gemini_test

import (
	"context"
	"encoding/json"
	"net/http"
	"strings"
	"testing"
	"time"

	"github.com/coder/websocket"

	"github.com/maruel/genai"
	"github.com/maruel/genai/internal/internaltest"
	"github.com/maruel/genai/internal/realtimetest"
	"github.com/maruel/genai/providers/gemini"
)

func liveClient(t *testing.T) *gemini.Client {
	return liveClientForModel(t, "gemini-3.8-live")
}

func liveClientForModel(t *testing.T, model string) *gemini.Client {
	c, err := gemini.New(t.Context(), genai.ProviderOptionAPIKey("test-live-key"), genai.ProviderOptionModel(model), genai.ProviderOptionModalities{genai.ModalityAudio})
	if err != nil {
		t.Fatal(err)
	}
	return c
}

// livePeer performs the real upgrade and setup handshake before running a script.
func livePeer(t *testing.T, script func(context.Context, *websocket.Conn, map[string]any)) string {
	u := realtimetest.Peer(t, func(r *http.Request) {
		if r.Header.Get("X-Goog-Api-Key") != "test-live-key" || r.URL.RawQuery != "" {
			t.Error("missing header authentication or unexpected query")
		}
	}, func(ctx context.Context, ws *websocket.Conn) {
		_, data, err := ws.Read(ctx)
		if err != nil {
			t.Error(err)
			return
		}
		var setup map[string]any
		if err := json.Unmarshal(data, &setup); err != nil {
			t.Error(err)
			return
		}
		realtimetest.Write(t, ctx, ws, `{"setupComplete":{"sessionId":"live-test"}}`)
		script(ctx, ws, setup)
	})
	return "ws" + strings.TrimPrefix(u, "http")
}

func TestRealtimeVoiceActivity(t *testing.T) {
	t.Parallel()
	endpoint := livePeer(t, func(ctx context.Context, ws *websocket.Conn, _ map[string]any) {
		realtimetest.Write(t, ctx, ws, `{"voiceActivity":{"type":"ACTIVITY_START","audioOffset":"0.1s"}}`)
		realtimetest.Write(t, ctx, ws, `{"voiceActivity":{"type":"ACTIVITY_END","audioOffset":"0.2s"}}`)
	})
	s, err := liveClient(t).ConnectRealtime(t.Context(), &gemini.RealtimeOption{Endpoint: endpoint})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = s.Close() })
	realtimetest.Receive(t, s, genai.RealtimeEventInputSpeechStart)
	realtimetest.Receive(t, s, genai.RealtimeEventInputSpeechEnd)
}

func TestRealtimeContextUpdate(t *testing.T) {
	sent := make(chan struct{})
	endpoint := livePeer(t, func(ctx context.Context, ws *websocket.Conn, setup map[string]any) {
		m := realtimetest.Read(t, ctx, ws)
		update, _ := m["contextUpdate"].(map[string]any)
		instr, _ := update["systemInstruction"].(map[string]any)
		parts, _ := instr["parts"].([]any)
		if len(parts) != 1 || internaltest.AssertType[map[string]any](t, parts[0])["text"] != "be terse" {
			t.Errorf("instructions %#v", m)
		}
		if _, ok := update["tools"]; ok {
			t.Errorf("instructions update touched tools: %#v", m)
		}

		m = realtimetest.Read(t, ctx, ws)
		update, _ = m["contextUpdate"].(map[string]any)
		tools, _ := update["tools"].(map[string]any)
		if len(tools) != 0 {
			t.Errorf("none did not clear tools: %#v", m)
		}

		m = realtimetest.Read(t, ctx, ws)
		update, _ = m["contextUpdate"].(map[string]any)
		tools, _ = update["tools"].(map[string]any)
		if len(tools) != 0 {
			t.Errorf("tool update escaped none: %#v", m)
		}

		m = realtimetest.Read(t, ctx, ws)
		update, _ = m["contextUpdate"].(map[string]any)
		tools, _ = update["tools"].(map[string]any)
		decls, _ := tools["tools"].([]any)
		if len(decls) != 1 || internaltest.AssertType[map[string]any](t, decls[0])["functionDeclarations"] == nil {
			t.Errorf("auto did not restore tools: %#v", m)
		}
		close(sent)
	})
	tool := genai.ToolDef{Name: "lookup", Description: "Lookup", InputSchemaOverride: json.RawMessage(`{"type":"object"}`)}
	s, err := liveClient(t).ConnectRealtime(t.Context(), &genai.RealtimeOptionTools{Tools: []genai.ToolDef{tool}}, &gemini.RealtimeOption{Endpoint: endpoint})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = s.Close() })
	instr := "be terse"
	none := genai.ToolCallNone
	auto := genai.ToolCallAny
	second := genai.ToolDef{Name: "second", Description: "Second", InputSchemaOverride: json.RawMessage(`{"type":"object"}`)}
	for _, in := range []genai.RealtimeInput{
		genai.RealtimeUpdate{Instructions: &instr},
		genai.RealtimeUpdate{ToolChoice: &none},
		genai.RealtimeUpdate{Tools: &[]genai.ToolDef{second}},
		genai.RealtimeUpdate{ToolChoice: &auto},
	} {
		if err := s.Send(t.Context(), in); err != nil {
			t.Fatal(err)
		}
	}
	required := genai.ToolCallRequired
	if err := s.Send(t.Context(), genai.RealtimeUpdate{ToolChoice: &required}); err == nil {
		t.Fatal("required tool choice accepted")
	}
	manual := genai.RealtimeTurnManual
	voice := "marin"
	for _, u := range []genai.RealtimeUpdate{{TurnDetection: &manual}, {Voice: &voice}} {
		if err := s.Send(t.Context(), u); err == nil {
			t.Fatalf("unsupported update accepted: %+v", u)
		}
	}
	select {
	case <-sent:
	case <-time.After(5 * time.Second):
		t.Fatal("timeout waiting for context updates")
	}
}

func TestRealtimeSendRaw(t *testing.T) {
	sent := make(chan struct{})
	endpoint := livePeer(t, func(ctx context.Context, ws *websocket.Conn, _ map[string]any) {
		m := realtimetest.Read(t, ctx, ws)
		input, _ := m["realtimeInput"].(map[string]any)
		if input["text"] != "raw" {
			t.Errorf("raw message %#v", m)
		}
		close(sent)
	})
	s, err := liveClient(t).ConnectRealtime(t.Context(), &gemini.RealtimeOption{Endpoint: endpoint})
	if err != nil {
		t.Fatal(err)
	}
	rs, ok := s.(*gemini.RealtimeSession)
	if !ok {
		t.Fatalf("session type %T", s)
	}
	t.Cleanup(func() { _ = s.Close() })
	if err := rs.SendRaw(t.Context(), nil); err == nil {
		t.Fatal("empty raw message accepted")
	}
	if err := rs.SendRaw(t.Context(), []byte(`{"realtimeInput":{"text":"raw"}}`)); err != nil {
		t.Fatal(err)
	}
	select {
	case <-sent:
	case <-time.After(5 * time.Second):
		t.Fatal("timeout waiting for raw message")
	}
}
