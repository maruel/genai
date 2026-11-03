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
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/coder/websocket"

	"github.com/maruel/genai"
	"github.com/maruel/genai/base"
	"github.com/maruel/genai/internal/internaltest"
	"github.com/maruel/genai/websocketrecord"
)

func TestWebSocketConn(t *testing.T) {
	t.Run("Embed", func(t *testing.T) {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.Method != http.MethodPost || r.URL.Path != "/embeddings" {
				t.Errorf("request %s %s", r.Method, r.URL.Path)
			}
			var in embeddingWireRequest
			if err := json.NewDecoder(r.Body).Decode(&in); err != nil {
				t.Error(err)
			}
			if in.Model != "text-embedding-3-small" || !slices.Equal(in.Input, []string{"hello"}) {
				t.Errorf("request %+v", in)
			}
			w.Header().Set("Content-Type", "application/json")
			if _, err := w.Write([]byte(`{"model":"text-embedding-3-small","data":[{"index":0,"embedding":"AAAAQAAAQMA="}],"usage":{"prompt_tokens":7,"total_tokens":7}}`)); err != nil {
				t.Error(err)
			}
		}))
		t.Cleanup(srv.Close)
		c, err := New(t.Context(), genai.ProviderOptionAPIKey("test"), genai.ProviderOptionModel("text-embedding-3-small"))
		if err != nil {
			t.Fatal(err)
		}
		internaltest.CleanupCloser(t, c)
		c.shared.BaseURL = srv.URL
		w := WebSocketConn{client: c}
		var p genai.Provider = &w
		if !p.Capabilities().Embed {
			t.Fatal("wrapper hid embedding implementation")
		}

		out, err := p.Embed(t.Context(), &genai.EmbeddingRequest{Inputs: []genai.Request{{Text: "hello"}}})
		if err != nil {
			t.Fatal(err)
		}
		if !slices.Equal(out.Embeddings[0], []float32{2, -3}) || out.Usage.InputTokens != 7 {
			t.Fatalf("response %+v", out)
		}
	})
	newClient := func(t *testing.T, model string) *Client {
		c := &Client{
			impl: base.Provider[*ErrorResponse, *Response, *Response, ResponseStreamChunkResponse]{
				ProviderBase: base.ProviderBase[*ErrorResponse]{Model: model},
			},
		}
		internaltest.CleanupCloser(t, c)
		return c
	}

	t.Run("WebSocket", func(t *testing.T) {
		path := filepath.Join(t.TempDir(), "test")
		rec, err := websocketrecord.New(path)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() {
			if err := rec.Close(); err != nil {
				t.Error(err)
			}
		})
		if err := rec.Start(startEchoServer(t)); err != nil {
			t.Fatal(err)
		}
		c, err := makeTestClient(t, "gpt-6-luna")
		if err != nil {
			t.Fatal(err)
		}
		c.baseURL = rec.URL()
		w, err := c.WebSocket(t.Context())
		if err != nil {
			t.Fatal(err)
		}
		if err := w.Close(); err != nil {
			t.Fatal(err)
		}
	})

	t.Run("Smoke", func(t *testing.T) {
		rec, err := websocketrecord.New("testdata/TestWebSocketConn/Smoke")
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() {
			var err error
			if t.Failed() {
				err = rec.Discard()
			} else {
				err = rec.Close()
			}
			if err != nil {
				t.Error(err)
			}
		})

		model := "gpt-6-luna"
		if v := os.Getenv("WS_MODEL"); v != "" {
			model = v
		}
		c, err := makeTestClient(t, model)
		if err != nil {
			t.Fatal(err)
		}
		target := ""
		if !rec.IsReplay() {
			if os.Getenv("RECORD") != "failure_only" || os.Getenv("OPENAI_API_KEY") == "" {
				t.Skip("set RECORD=failure_only and OPENAI_API_KEY to record")
			}
			target = "wss://api.openai.com/v1/responses"
		}
		if err := rec.Start(target); err != nil {
			t.Fatal(err)
		}
		conn, _, err := websocket.Dial(t.Context(), rec.WSURL(), &websocket.DialOptions{ //nolint:bodyclose // coder/websocket owns the response body.
			HTTPClient: &c.impl.Client,
			HTTPHeader: http.Header{"OpenAI-Beta": []string{"responses=v1"}},
		})
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { _ = conn.CloseNow() })

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
			if err := conn.Write(t.Context(), websocket.MessageText, []byte(req)); err != nil {
				t.Fatal(err)
			}
			prevID = ""
			for {
				typ, data, err := conn.Read(t.Context())
				if err != nil {
					t.Fatal(err)
				}
				if typ != websocket.MessageText {
					t.Fatalf("message type %v", typ)
				}
				var evt struct {
					Type     string `json:"type"`
					Response struct {
						ID string `json:"id"`
					} `json:"response"`
				}
				if err := json.Unmarshal(data, &evt); err != nil {
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
	})

	t.Run("Smoke_Local", func(t *testing.T) {
		// Self-contained test with a local echo server. Verifies the full
		// record→replay flow without any network dependency.
		target := startEchoServer(t)
		path := filepath.Join(t.TempDir(), "smoke")

		rec, err := websocketrecord.New(path)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() {
			if err := rec.Close(); err != nil {
				t.Error(err)
			}
		})
		if err := rec.Start(target); err != nil {
			t.Fatal(err)
		}
		conn := dialProxy(t, rec.WSURL())
		runEchoTurns(t, conn)
		_ = conn.CloseNow()
		if err := rec.Close(); err != nil {
			t.Fatal(err)
		}

		rec2, err := websocketrecord.New(path)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() {
			if err := rec2.Close(); err != nil {
				t.Error(err)
			}
		})
		if !rec2.IsReplay() {
			t.Fatal("expected replay mode after recording")
		}
		if err := rec2.Start(""); err != nil {
			t.Fatal(err)
		}
		conn2 := dialProxy(t, rec2.WSURL())
		runEchoTurns(t, conn2)
	})

	t.Run("buildRequest", func(t *testing.T) {
		w := &WebSocketConn{client: newClient(t, "gpt-6-luna")}
		msgs := genai.Messages{genai.NewTextMessage("Hello")}
		req, err := w.buildRequest(msgs)
		if err != nil {
			t.Fatal(err)
		}
		if req.Type != "response.create" {
			t.Errorf("Type = %q, want %q", req.Type, "response.create")
		}
		if req.Model != "gpt-6-luna" {
			t.Errorf("Model = %q, want %q", req.Model, "gpt-6-luna")
		}
		if req.PreviousResponseID != "" {
			t.Errorf("PreviousResponseID = %q, want empty", req.PreviousResponseID)
		}
	})

	t.Run("buildRequest_with_meta", func(t *testing.T) {
		w := &WebSocketConn{client: newClient(t, "gpt-6-luna")}
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
		w := &WebSocketConn{client: newClient(t, "gpt-6-luna")}
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
		c := &Client{}
		internaltest.CleanupCloser(t, c)
		w := &WebSocketConn{client: c}
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
		internaltest.CleanupCloser(t, c)
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
	c, err := New(t.Context(), opts...)
	if c != nil {
		internaltest.CleanupCloser(t, c)
	}
	return c, err
}

func prevIDStr(id string) string {
	if id == "" {
		return ""
	}
	return fmt.Sprintf(`,"previous_response_id":%q`, id)
}

// startEchoServer starts a local WebSocket server that echoes received
// messages with a prefix, simulating a conversation for testing.
func startEchoServer(t *testing.T) string {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		c, err := websocket.Accept(w, r, &websocket.AcceptOptions{InsecureSkipVerify: true})
		if err != nil {
			t.Error(err)
			return
		}
		defer func() { _ = c.CloseNow() }()
		for {
			typ, data, err := c.Read(r.Context())
			if err != nil {
				return
			}
			if err := c.Write(r.Context(), typ, []byte("echo: "+strings.ToUpper(string(data)))); err != nil {
				return
			}
		}
	}))
	t.Cleanup(srv.Close)
	return "ws" + strings.TrimPrefix(srv.URL, "http")
}

func dialProxy(t *testing.T, endpoint string) *websocket.Conn {
	c, _, err := websocket.Dial(t.Context(), endpoint, &websocket.DialOptions{}) //nolint:bodyclose // coder/websocket owns the response body.
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = c.CloseNow() })
	return c
}

func runEchoTurns(t *testing.T, c *websocket.Conn) {
	for i := range 3 {
		req := fmt.Sprintf("turn %d", i+1)
		if err := c.Write(t.Context(), websocket.MessageText, []byte(req)); err != nil {
			t.Fatal(err)
		}
		typ, data, err := c.Read(t.Context())
		if err != nil {
			t.Fatal(err)
		}
		want := "echo: " + strings.ToUpper(req)
		if typ != websocket.MessageText || string(data) != want {
			t.Fatalf("replay turn %d: %v %q, want %q", i+1, typ, data, want)
		}
		t.Logf("Turn %d: %s -> %s", i+1, req, data)
	}
}

type embeddingWireRequest struct {
	Model string
	Input []string
}

func TestWebSocketSendRaw(t *testing.T) {
	got := make(chan string, 1)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		ws, err := websocket.Accept(w, r, nil)
		if err != nil {
			t.Error(err)
			return
		}
		defer func() { _ = ws.CloseNow() }()
		typ, data, err := ws.Read(r.Context())
		if err != nil {
			t.Error(err)
			return
		}
		if typ != websocket.MessageText {
			t.Errorf("message type %v", typ)
			return
		}
		got <- string(data)
	}))
	t.Cleanup(srv.Close)
	c, err := makeTestClient(t, "gpt-6-luna")
	if err != nil {
		t.Fatal(err)
	}
	c.baseURL = srv.URL
	w, err := c.WebSocket(t.Context())
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = w.Close() })
	if err := w.SendRaw(t.Context(), nil); err == nil {
		t.Fatal("empty raw message accepted")
	}
	const message = `{"type":"response.create","response":{}}`
	if err := w.SendRaw(t.Context(), []byte(message)); err != nil {
		t.Fatal(err)
	}
	select {
	case v := <-got:
		if v != message {
			t.Fatalf("raw message %s", v)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("timeout waiting for raw message")
	}
}
