// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Tests for the websocketrecord package.

package websocketrecord

import (
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"golang.org/x/net/websocket"
)

func TestNew(t *testing.T) {
	t.Run("valid", func(t *testing.T) {
		t.Run("record_when_no_fixture", func(t *testing.T) {
			rec, err := New(filepath.Join(t.TempDir(), "test"))
			if err != nil {
				t.Fatal(err)
			}
			if rec.replay {
				t.Fatal("expected record mode")
			}
		})
		t.Run("replay_when_fixture_exists", func(t *testing.T) {
			dir := t.TempDir()
			path := filepath.Join(dir, "test")
			if err := os.WriteFile(path+".ndjson", []byte(`{"dir":">","msg":"hello"}`+"\n"), 0o644); err != nil {
				t.Fatal(err)
			}
			rec, err := New(path)
			if err != nil {
				t.Fatal(err)
			}
			if !rec.replay {
				t.Fatal("expected replay mode")
			}
		})
	})
	t.Run("error", func(t *testing.T) {
		t.Run("invalid_json", func(t *testing.T) {
			dir := t.TempDir()
			path := filepath.Join(dir, "test")
			if err := os.WriteFile(path+".ndjson", []byte("not json\n"), 0o644); err != nil {
				t.Fatal(err)
			}
			_, err := New(path)
			if err == nil {
				t.Fatal("expected error for invalid JSON")
			}
		})
	})
}

func TestRecorder(t *testing.T) {
	t.Run("valid", func(t *testing.T) {
		t.Run("record_and_replay", func(t *testing.T) {
			// Start a real WebSocket echo server.
			echoSrv := httptest.NewServer(websocket.Handler(func(ws *websocket.Conn) {
				for {
					var msg string
					if err := websocket.Message.Receive(ws, &msg); err != nil {
						return
					}
					if err := websocket.Message.Send(ws, strings.ToUpper(msg)); err != nil {
						return
					}
				}
			}))
			defer echoSrv.Close()

			dir := t.TempDir()
			path := filepath.Join(dir, "test")

			// Record phase.
			rec, err := New(path)
			if err != nil {
				t.Fatal(err)
			}
			cfg, err := websocket.NewConfig("ws"+echoSrv.URL[4:], echoSrv.URL)
			if err != nil {
				t.Fatal(err)
			}
			c, err := rec.DialContext(t.Context(), cfg)
			if err != nil {
				t.Fatal(err)
			}
			if err := c.Send("hello"); err != nil {
				t.Fatal(err)
			}
			var resp string
			if err := c.Receive(&resp); err != nil {
				t.Fatal(err)
			}
			if resp != "HELLO" {
				t.Fatalf("got %q, want %q", resp, "HELLO")
			}
			if err := c.Send("world"); err != nil {
				t.Fatal(err)
			}
			if err := c.Receive(&resp); err != nil {
				t.Fatal(err)
			}
			if resp != "WORLD" {
				t.Fatalf("got %q, want %q", resp, "WORLD")
			}
			if err := c.Close(); err != nil {
				t.Fatal(err)
			}
			if err := rec.Stop(); err != nil {
				t.Fatal(err)
			}

			// Verify fixture was written.
			data, err := os.ReadFile(path + ".ndjson")
			if err != nil {
				t.Fatal(err)
			}
			if !strings.Contains(string(data), `"hello"`) {
				t.Fatalf("fixture missing 'hello': %s", data)
			}

			// Replay phase.
			rec2, err := New(path)
			if err != nil {
				t.Fatal(err)
			}
			if !rec2.replay {
				t.Fatal("expected replay mode")
			}
			cfg2, err := websocket.NewConfig("ws://localhost:0", "http://localhost:0")
			if err != nil {
				t.Fatal(err)
			}
			c2, err := rec2.DialContext(t.Context(), cfg2)
			if err != nil {
				t.Fatal(err)
			}
			// During replay, the server sends back recorded responses.
			// We still need to send the requests.
			if err := c2.Send("hello"); err != nil {
				t.Fatal(err)
			}
			if err := c2.Receive(&resp); err != nil {
				t.Fatal(err)
			}
			if resp != "HELLO" {
				t.Fatalf("replay: got %q, want %q", resp, "HELLO")
			}
			if err := c2.Send("world"); err != nil {
				t.Fatal(err)
			}
			if err := c2.Receive(&resp); err != nil {
				t.Fatal(err)
			}
			if resp != "WORLD" {
				t.Fatalf("replay: got %q, want %q", resp, "WORLD")
			}
			if err := c2.Close(); err != nil {
				t.Fatal(err)
			}
			if err := rec2.Stop(); err != nil {
				t.Fatal(err)
			}
		})

		t.Run("replay_only", func(t *testing.T) {
			dir := t.TempDir()
			path := filepath.Join(dir, "test")
			// Pre-create fixture with known messages.
			fixture := `{"dir":">","msg":"ping"}
{"dir":"<","msg":"PONG"}
`
			if err := os.WriteFile(path+".ndjson", []byte(fixture), 0o644); err != nil {
				t.Fatal(err)
			}
			rec, err := New(path)
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() {
				if err := rec.Stop(); err != nil {
					t.Error(err)
				}
			})
			cfg, err := websocket.NewConfig("ws://localhost:0", "http://localhost:0")
			if err != nil {
				t.Fatal(err)
			}
			c, err := rec.DialContext(t.Context(), cfg)
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() {
				if err := c.Close(); err != nil {
					t.Error(err)
				}
			})
			// Send the request (replay server expects one request then sends responses).
			if err := c.Send("anything"); err != nil {
				t.Fatal(err)
			}
			var resp string
			if err := c.Receive(&resp); err != nil {
				t.Fatal(err)
			}
			if resp != "PONG" {
				t.Fatalf("got %q, want %q", resp, "PONG")
			}
		})

		t.Run("stop_idempotent", func(t *testing.T) {
			rec, err := New(filepath.Join(t.TempDir(), "test"))
			if err != nil {
				t.Fatal(err)
			}
			if err := rec.Stop(); err != nil {
				t.Fatal(err)
			}
			if err := rec.Stop(); err != nil {
				t.Fatal(err)
			}
		})
	})
}

func TestConn(t *testing.T) {
	t.Run("valid", func(t *testing.T) {
		t.Run("close_idempotent", func(t *testing.T) {
			// Start an echo server.
			echoSrv := httptest.NewServer(websocket.Handler(func(ws *websocket.Conn) {
				var msg string
				if err := websocket.Message.Receive(ws, &msg); err != nil {
					return
				}
				if err := websocket.Message.Send(ws, msg); err != nil {
					return
				}
			}))
			defer echoSrv.Close()

			cfg, err := websocket.NewConfig("ws"+echoSrv.URL[4:], echoSrv.URL)
			if err != nil {
				t.Fatal(err)
			}
			ws, err := cfg.DialContext(t.Context())
			if err != nil {
				t.Fatal(err)
			}
			c := &Conn{ws: ws}
			// Close twice should not panic.
			if err := c.Close(); err != nil {
				t.Fatal(err)
			}
			if err := c.Close(); err != nil {
				t.Fatal(err)
			}
		})

		t.Run("ws_returns_underlying", func(t *testing.T) {
			echoSrv := httptest.NewServer(websocket.Handler(func(ws *websocket.Conn) {
				var msg string
				if err := websocket.Message.Receive(ws, &msg); err != nil {
					return
				}
				if err := websocket.Message.Send(ws, msg); err != nil {
					return
				}
			}))
			defer echoSrv.Close()

			cfg, err := websocket.NewConfig("ws"+echoSrv.URL[4:], echoSrv.URL)
			if err != nil {
				t.Fatal(err)
			}
			ws, err := cfg.DialContext(t.Context())
			if err != nil {
				t.Fatal(err)
			}
			c := &Conn{ws: ws}
			t.Cleanup(func() {
				if err := c.Close(); err != nil {
					t.Error(err)
				}
			})
			if c.WS() != ws {
				t.Fatal("WS() should return the underlying connection")
			}
		})
	})
	t.Run("error", func(t *testing.T) {
		t.Run("send_on_closed", func(t *testing.T) {
			c := &Conn{}
			err := c.Send("test")
			if err == nil {
				t.Fatal("expected error sending on nil connection")
			}
		})
		t.Run("receive_on_closed", func(t *testing.T) {
			c := &Conn{}
			var msg string
			err := c.Receive(&msg)
			if err == nil {
				t.Fatal("expected error receiving on nil connection")
			}
		})
	})
}

func TestMessageRoundTrip(t *testing.T) {
	t.Run("valid", func(t *testing.T) {
		t.Run("json_messages", func(t *testing.T) {
			dir := t.TempDir()
			path := filepath.Join(dir, "test")

			// Record phase with JSON messages.
			echoSrv := httptest.NewServer(websocket.Handler(func(ws *websocket.Conn) {
				for {
					var msg string
					if err := websocket.Message.Receive(ws, &msg); err != nil {
						return
					}
					// Echo back with a wrapper.
					resp := `{"echo":"` + msg + `"}`
					if err := websocket.Message.Send(ws, resp); err != nil {
						return
					}
				}
			}))
			defer echoSrv.Close()

			rec, err := New(path)
			if err != nil {
				t.Fatal(err)
			}
			cfg, err := websocket.NewConfig("ws"+echoSrv.URL[4:], echoSrv.URL)
			if err != nil {
				t.Fatal(err)
			}
			c, err := rec.DialContext(t.Context(), cfg)
			if err != nil {
				t.Fatal(err)
			}

			requests := []string{`{"type":"a"}`, `{"type":"b"}`, `{"type":"c"}`}
			for _, req := range requests {
				if err := c.Send(req); err != nil {
					t.Fatal(err)
				}
				var resp string
				if err := c.Receive(&resp); err != nil {
					t.Fatal(err)
				}
				if resp != `{"echo":"`+req+`"}` {
					t.Fatalf("got %q, want %q", resp, `{"echo":"`+req+`"}`)
				}
			}
			if err := c.Close(); err != nil {
				t.Fatal(err)
			}
			if err := rec.Stop(); err != nil {
				t.Fatal(err)
			}

			// Replay.
			rec2, err := New(path)
			if err != nil {
				t.Fatal(err)
			}
			cfg2, err := websocket.NewConfig("ws://localhost:0", "http://localhost:0")
			if err != nil {
				t.Fatal(err)
			}
			c2, err := rec2.DialContext(t.Context(), cfg2)
			if err != nil {
				t.Fatal(err)
			}
			for _, req := range requests {
				if err := c2.Send(req); err != nil {
					t.Fatal(err)
				}
				var resp string
				if err := c2.Receive(&resp); err != nil {
					t.Fatal(err)
				}
				want := `{"echo":"` + req + `"}`
				if resp != want {
					t.Fatalf("replay: got %q, want %q", resp, want)
				}
			}
			if err := c2.Close(); err != nil {
				t.Fatal(err)
			}
			if err := rec2.Stop(); err != nil {
				t.Fatal(err)
			}
		})
	})
}
