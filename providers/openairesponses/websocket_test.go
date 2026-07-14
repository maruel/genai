// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Tests for websocket.go

package openairesponses

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"

	"github.com/maruel/roundtrippers"
	"golang.org/x/net/websocket"

	"github.com/maruel/genai"
	"github.com/maruel/genai/base"
	"github.com/maruel/genai/websocketrecord"
)

func TestWSRequest(t *testing.T) {
	t.Run("valid", func(t *testing.T) {
		req := WSRequest{
			Type: "response.create",
			Response: Response{
				Model: "gpt-5.6-luna",
				Input: []Message{
					{
						Type: MessageMessage,
						Role: "user",
						Content: []Content{
							{Type: ContentInputText, Text: "Hello"},
						},
					},
				},
				PreviousResponseID: "resp_abc",
				Store:              false,
			},
		}
		data, err := json.Marshal(req)
		if err != nil {
			t.Fatal(err)
		}
		var m map[string]any
		if err := json.Unmarshal(data, &m); err != nil {
			t.Fatal(err)
		}
		if got := m["type"]; got != "response.create" {
			t.Errorf("type = %q, want %q", got, "response.create")
		}
		if got := m["model"]; got != "gpt-5.6-luna" {
			t.Errorf("model = %q, want %q", got, "gpt-5.6-luna")
		}
		if got := m["previous_response_id"]; got != "resp_abc" {
			t.Errorf("previous_response_id = %q, want %q", got, "resp_abc")
		}
		if _, ok := m["stream"]; ok {
			t.Error("stream field should not be present in WS request")
		}
		if _, ok := m["background"]; ok {
			t.Error("background field should not be present in WS request")
		}
	})

	t.Run("no_previous_id", func(t *testing.T) {
		req := WSRequest{
			Type: "response.create",
			Response: Response{
				Model: "gpt-5.6-luna",
			},
		}
		data, err := json.Marshal(req)
		if err != nil {
			t.Fatal(err)
		}
		var m map[string]any
		if err := json.Unmarshal(data, &m); err != nil {
			t.Fatal(err)
		}
		if _, ok := m["previous_response_id"]; ok {
			t.Error("previous_response_id should not be present when empty")
		}
	})
}

func TestWebSocketConn(t *testing.T) {
	newClient := func(model string) *Client {
		return &Client{
			impl: base.Provider[*ErrorResponse, *Response, *Response, ResponseStreamChunkResponse]{
				ProviderBase: base.ProviderBase[*ErrorResponse]{Model: model},
			},
		}
	}

	t.Run("Smoke", func(t *testing.T) {
		rec, err := websocketrecord.New("testdata/TestWebSocketConn/Smoke")
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() {
			if err := rec.Stop(); err != nil {
				t.Error(err)
			}
		})

		if !rec.IsReplay() {
			if os.Getenv("RECORD") != "all" || os.Getenv("OPENAI_API_KEY") == "" {
				t.Skip("set RECORD=all and OPENAI_API_KEY to record")
			}
			model := "gpt-5.6-luna"
			if v := os.Getenv("WS_MODEL"); v != "" {
				model = v
			}
			c, err := makeTestClient(t, model)
			if err != nil {
				t.Fatal(err)
			}
			wsCfg, err := buildWSConfig(c)
			if err != nil {
				t.Fatal(err)
			}
			conn, err := rec.DialContext(t.Context(), wsCfg)
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() {
				if err := conn.Close(); err != nil {
					t.Error(err)
				}
			})

			// Record 3 turns with raw messages, exercising the recorder.
			// Each turn sends a response.create and waits for the terminal event.
			var prevID string
			texts := []string{
				"Say hello in exactly 3 words.",
				"Now count from 1 to 3.",
				"Good, now say goodbye in exactly 2 words.",
			}
			for i, text := range texts {
				req := fmt.Sprintf(`{"type":"response.create","model":%q,"input":[{"type":"message","role":"user","content":[{"type":"input_text","text":%q}]}],"store":false%s}`,
					model, text, prevIDStr(prevID))
				if err := conn.Send(req); err != nil {
					t.Fatal(err)
				}
				prevID = ""
				for {
					var msg string
					if err := conn.Receive(&msg); err != nil {
						t.Fatal(err)
					}
					var evt struct {
						Type     string `json:"type"`
						Response struct {
							ID string `json:"id"`
						} `json:"response"`
					}
					if err := json.Unmarshal([]byte(msg), &evt); err != nil {
						t.Fatal(err)
					}
					if evt.Response.ID != "" {
						prevID = evt.Response.ID
					}
					if evt.Type == "response.completed" || evt.Type == "response.failed" {
						t.Logf("Turn %d: %s (id=%s)", i+1, evt.Type, prevID)
						break
					}
				}
			}
			return
		}

		// Replay: exercise the fixture with raw send/receive.
		wsCfg, err := websocket.NewConfig("ws://localhost:0", "http://localhost:0")
		if err != nil {
			t.Fatal(err)
		}
		conn, err := rec.DialContext(t.Context(), wsCfg)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() {
			if err := conn.Close(); err != nil {
				t.Error(err)
			}
		})

		for turn := range 3 {
			if err := conn.Send("request"); err != nil {
				t.Fatal(err)
			}
			for {
				var msg string
				if err := conn.Receive(&msg); err != nil {
					t.Fatal(err)
				}
				var evt struct {
					Type string `json:"type"`
				}
				if err := json.Unmarshal([]byte(msg), &evt); err != nil {
					t.Fatal(err)
				}
				if evt.Type == "response.completed" || evt.Type == "response.failed" {
					t.Logf("Replay turn %d: %s", turn+1, evt.Type)
					break
				}
			}
		}
	})

	t.Run("Smoke_Local", func(t *testing.T) {
		// Self-contained test with a local echo server. Verifies the full
		// record→replay flow without any network dependency.
		srv := startEchoServer()
		t.Cleanup(srv.Close)

		dir := t.TempDir()
		path := dir + "/smoke"

		// Record phase.
		rec, err := websocketrecord.New(path)
		if err != nil {
			t.Fatal(err)
		}
		wsCfg, err := websocket.NewConfig("ws"+srv.URL[4:], srv.URL)
		if err != nil {
			t.Fatal(err)
		}
		conn, err := rec.DialContext(t.Context(), wsCfg)
		if err != nil {
			t.Fatal(err)
		}
		// Simulate multi-turn: send request, receive "echoed" response.
		for i := range 3 {
			req := fmt.Sprintf("turn %d", i+1)
			if err := conn.Send(req); err != nil {
				t.Fatal(err)
			}
			var resp string
			if err := conn.Receive(&resp); err != nil {
				t.Fatal(err)
			}
			t.Logf("Turn %d: %s → %s", i+1, req, resp)
		}
		if err := conn.Close(); err != nil {
			t.Fatal(err)
		}
		if err := rec.Stop(); err != nil {
			t.Fatal(err)
		}

		// Replay phase.
		rec2, err := websocketrecord.New(path)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() {
			if err := rec2.Stop(); err != nil {
				t.Error(err)
			}
		})
		if !rec2.IsReplay() {
			t.Fatal("expected replay mode after recording")
		}
		wsCfg2, err := websocket.NewConfig("ws://localhost:0", "http://localhost:0")
		if err != nil {
			t.Fatal(err)
		}
		conn2, err := rec2.DialContext(t.Context(), wsCfg2)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() {
			if err := conn2.Close(); err != nil {
				t.Error(err)
			}
		})
		for i := range 3 {
			req := fmt.Sprintf("turn %d", i+1)
			if err := conn2.Send(req); err != nil {
				t.Fatal(err)
			}
			var resp string
			if err := conn2.Receive(&resp); err != nil {
				t.Fatal(err)
			}
			t.Logf("Replay turn %d: %s → %s", i+1, req, resp)
		}
	})

	t.Run("buildRequest", func(t *testing.T) {
		w := &WebSocketConn{client: newClient("gpt-5.6-luna")}
		msgs := genai.Messages{genai.NewTextMessage("Hello")}
		req, err := w.buildRequest(msgs)
		if err != nil {
			t.Fatal(err)
		}
		if req.Type != "response.create" {
			t.Errorf("Type = %q, want %q", req.Type, "response.create")
		}
		if req.Model != "gpt-5.6-luna" {
			t.Errorf("Model = %q, want %q", req.Model, "gpt-5.6-luna")
		}
		if req.PreviousResponseID != "" {
			t.Errorf("PreviousResponseID = %q, want empty", req.PreviousResponseID)
		}
	})

	t.Run("buildRequest_with_meta", func(t *testing.T) {
		w := &WebSocketConn{client: newClient("gpt-5.6-luna")}
		msgs := genai.Messages{
			genai.NewTextMessage("Hello"),
			{Replies: []genai.Reply{
				{Text: "Hi"},
				{Opaque: map[string]any{
					opaqueResponseID: "resp_prev",
					opaqueSentMsgs:   float64(2),
				}},
			}},
			genai.NewTextMessage("Continue"),
		}
		req, err := w.buildRequest(msgs)
		if err != nil {
			t.Fatal(err)
		}
		if req.PreviousResponseID != "resp_prev" {
			t.Errorf("PreviousResponseID = %q, want %q", req.PreviousResponseID, "resp_prev")
		}
		// Should only contain the delta (msg #2: the "Continue" message).
		if len(req.Input) != 1 {
			t.Errorf("got %d input messages, want 1 (delta only)", len(req.Input))
		}
	})

	t.Run("buildRequest_user_override", func(t *testing.T) {
		w := &WebSocketConn{client: newClient("gpt-5.6-luna")}
		msgs := genai.Messages{
			genai.NewTextMessage("Hello"),
			{Replies: []genai.Reply{{Opaque: map[string]any{
				opaqueResponseID: "resp_auto",
				opaqueSentMsgs:   float64(2),
			}}}},
			genai.NewTextMessage("Continue"),
		}
		req, err := w.buildRequest(msgs, &GenOptionText{PreviousResponseID: "resp_manual"})
		if err != nil {
			t.Fatal(err)
		}
		if req.PreviousResponseID != "resp_manual" {
			t.Errorf("PreviousResponseID = %q, want %q (user override)", req.PreviousResponseID, "resp_manual")
		}
	})

	t.Run("name", func(t *testing.T) {
		w := &WebSocketConn{client: &Client{}}
		if got := w.Name(); got != "openairesponses" {
			t.Errorf("Name() = %q, want %q", got, "openairesponses")
		}
	})

	t.Run("capabilities", func(t *testing.T) {
		w := &WebSocketConn{}
		caps := w.Capabilities()
		if caps.GenAsync {
			t.Error("should not report GenAsync")
		}
		if caps.Caching {
			t.Error("should not report Caching")
		}
	})

	t.Run("close", func(t *testing.T) {
		w := &WebSocketConn{}
		if err := w.Close(); err != nil {
			t.Errorf("Close() on nil connection: %v", err)
		}
	})

	t.Run("http_client", func(t *testing.T) {
		c := &Client{}
		w := &WebSocketConn{client: c}
		if got, want := w.HTTPClient(), c.HTTPClient(); got != want {
			t.Errorf("HTTPClient() = %v, want %v", got, want)
		}
	})
}

func makeTestClient(t *testing.T, model string) (*Client, error) {
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		apiKey = "<insert_api_key_here>"
	}
	opts := []genai.ProviderOption{
		genai.ProviderOptionAPIKey(apiKey),
		genai.ProviderOptionModel(model),
	}
	return New(t.Context(), opts...)
}

// buildWSConfig creates a websocket.Config for the OpenAI Responses API,
// replicating the logic in Client.WebSocket().
func buildWSConfig(c *Client) (*websocket.Config, error) {
	wsURL := strings.Replace(c.baseURL, "https://", "wss://", 1)
	wsURL = strings.Replace(wsURL, "http://", "ws://", 1)
	wsURL += "/responses"
	cfg, err := websocket.NewConfig(wsURL, wsURL)
	if err != nil {
		return nil, err
	}
	cfg.Header = http.Header{}
	cfg.Header.Set("OpenAI-Beta", "responses=v1")
	if h, ok := c.impl.Client.Transport.(*roundtrippers.Header); ok {
		for k, vs := range h.Header {
			for _, v := range vs {
				cfg.Header.Set(k, v)
			}
		}
	}
	return cfg, nil
}

func prevIDStr(id string) string {
	if id == "" {
		return ""
	}
	return fmt.Sprintf(`,"previous_response_id":%q`, id)
}

// startEchoServer starts a local WebSocket server that echoes received
// messages with a prefix, simulating a conversation for testing.
func startEchoServer() *httptest.Server {
	return httptest.NewServer(websocket.Handler(func(ws *websocket.Conn) {
		for {
			var msg string
			if err := websocket.Message.Receive(ws, &msg); err != nil {
				return
			}
			resp := "echo: " + strings.ToUpper(msg)
			if err := websocket.Message.Send(ws, resp); err != nil {
				return
			}
		}
	}))
}
