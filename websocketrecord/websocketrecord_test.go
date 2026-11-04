// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Tests for the websocketrecord package.

package websocketrecord

import (
	"bytes"
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/coder/websocket"
)

func TestNew(t *testing.T) {
	t.Run("discard", func(t *testing.T) {
		path := filepath.Join(t.TempDir(), "test")
		rec := newRecorder(t, path, echoServer(t))
		recordMessages(t, rec.WSURL())
		if err := rec.Discard(); err != nil {
			t.Fatal(err)
		}
		if _, err := os.Stat(path + ".ndjson"); !errors.Is(err, os.ErrNotExist) {
			t.Fatalf("fixture remains: %v", err)
		}
		rec2, err := New(path)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() {
			if err := rec2.Close(); err != nil {
				t.Error(err)
			}
		})
		if rec2.IsReplay() {
			t.Fatal("discarded fixture replayed")
		}
	})
	t.Run("base URL keeps the dialed path", func(t *testing.T) {
		// The caller passes the base URL it gives the client, not the endpoint: the
		// proxy appends what the client dialed.
		var got []string
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			got = append(got, r.URL.Path)
			c, err := websocket.Accept(w, r, nil)
			if err != nil {
				return
			}
			defer func() { _ = c.CloseNow() }()
			ctx, cancel := context.WithCancel(r.Context())
			defer cancel()
			for {
				typ, data, err := c.Read(ctx)
				if err != nil {
					return
				}
				if err := c.Write(ctx, typ, data); err != nil {
					return
				}
			}
		}))
		t.Cleanup(srv.Close)
		rec := newRecorder(t, filepath.Join(t.TempDir(), "test"), srv.URL+"/v1")
		t.Cleanup(func() {
			if err := rec.Close(); err != nil {
				t.Error(err)
			}
		})
		recordMessages(t, rec.WSURL()+"/responses")
		if len(got) != 1 || got[0] != "/v1/responses" {
			t.Fatalf("upstream paths: %q", got)
		}
	})
	t.Run("handshake headers are replayed", func(t *testing.T) {
		path := filepath.Join(t.TempDir(), "test")
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("X-RateLimit-Remaining", "7")
			c, err := websocket.Accept(w, r, nil)
			if err != nil {
				return
			}
			defer func() { _ = c.CloseNow() }()
		}))
		t.Cleanup(srv.Close)
		rec, err := New(path)
		if err != nil {
			t.Fatal(err)
		}
		upstream := "ws" + strings.TrimPrefix(srv.URL, "http")
		if err := rec.Start(upstream); err != nil {
			t.Fatal(err)
		}
		assertHeader := func(t *testing.T, url string, want string) {
			t.Helper()
			ctx, cancel := context.WithTimeout(t.Context(), 10*time.Second)
			defer cancel()
			c, resp, err := websocket.Dial(ctx, "ws"+strings.TrimPrefix(url, "http"), nil) //nolint:bodyclose // coder/websocket owns the response body.
			if err != nil {
				t.Fatal(err)
			}
			_ = c.CloseNow()
			if got := resp.Header.Get("X-RateLimit-Remaining"); got != want {
				t.Fatalf("header %q, want %q", got, want)
			}
		}
		assertHeader(t, rec.URL(), "7")
		if err := rec.Close(); err != nil {
			t.Fatal(err)
		}
		replay, err := New(path)
		if err != nil {
			t.Fatal(err)
		}
		if err := replay.Start(upstream); err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() {
			if err := replay.Close(); err != nil {
				t.Error(err)
			}
		})
		assertHeader(t, replay.URL(), "7")
	})
	t.Run("empty fixture is kept", func(t *testing.T) {
		// A fixture is numbered by the call that created it, so a session that
		// exchanged no frame must leave an empty file: removing it shifts every
		// later number.
		path := filepath.Join(t.TempDir(), "test")
		rec, err := New(path)
		if err != nil {
			t.Fatal(err)
		}
		if err := rec.Start(echoServer(t)); err != nil {
			t.Fatal(err)
		}
		if err := rec.Close(); err != nil {
			t.Fatal(err)
		}
		if _, err := os.Stat(path + ".ndjson"); err != nil {
			t.Fatalf("empty fixture was removed: %v", err)
		}
	})
	t.Run("error", func(t *testing.T) {
		for name, data := range map[string]string{
			"invalid_json":      "not json\n",
			"invalid_direction": `{"dir":"?","msg":"x"}` + "\n",
			"invalid_type":      `{"dir":">","type":"other","msg":"x"}` + "\n",
			"text_with_data":    `{"dir":">","msg":"x","data":"AQ=="}` + "\n",
			"binary_with_text":  `{"dir":">","type":"binary","msg":"x"}` + "\n",
		} {
			t.Run(name, func(t *testing.T) {
				path := filepath.Join(t.TempDir(), "test")
				if err := os.WriteFile(path+".ndjson", []byte(data), 0o644); err != nil {
					t.Fatal(err)
				}
				if _, err := New(path); err == nil {
					t.Fatal("expected error")
				}
			})
		}
	})
}

func TestRecorder(t *testing.T) {
	t.Run("valid", func(t *testing.T) {
		t.Run("text_and_binary", func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "test")
			rec := newRecorder(t, path, echoServer(t))
			recordMessages(t, rec.WSURL())
			if err := rec.Close(); err != nil {
				t.Fatal(err)
			}

			rec2, err := New(path)
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() {
				if err := rec2.Close(); err != nil {
					t.Error(err)
				}
			})
			if !rec2.IsReplay() {
				t.Fatal("expected replay mode")
			}
			if err := rec2.Start(""); err != nil {
				t.Fatal(err)
			}
			replayMessages(t, rec2.WSURL())
		})
	})
	t.Run("error", func(t *testing.T) {
		t.Run("start_target", func(t *testing.T) {
			rec, err := New(filepath.Join(t.TempDir(), "test"))
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() {
				if err := rec.Close(); err != nil {
					t.Error(err)
				}
			})
			if rec.URL() != "" || rec.WSURL() != "" {
				t.Fatal("URL before Start")
			}
			for _, target := range []string{"", "ftp://example.com", "ws://", "example.com"} {
				if err := rec.Start(target); err == nil {
					t.Fatalf("accepted target %q", target)
				}
			}
		})
		t.Run("mismatch", func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "test")
			rec := newRecorder(t, path, echoServer(t))
			recordMessages(t, rec.WSURL())
			if err := rec.Close(); err != nil {
				t.Fatal(err)
			}

			rec2, err := New(path)
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() {
				if err := rec2.Close(); err != nil {
					t.Error(err)
				}
			})
			if err := rec2.Start(""); err != nil {
				t.Fatal(err)
			}
			c, _, err := websocket.Dial(t.Context(), rec2.WSURL(), &websocket.DialOptions{}) //nolint:bodyclose // coder/websocket owns the response body.
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() { _ = c.CloseNow() })
			if err := c.Write(t.Context(), websocket.MessageText, []byte("wrong")); err != nil {
				t.Fatal(err)
			}
			if _, _, err := c.Read(t.Context()); websocket.CloseStatus(err) != websocket.StatusPolicyViolation {
				t.Fatalf("Read() = %v, want policy violation", err)
			}
		})
	})
}

// newRecorder returns a recorder started towards target. Call Close or Discard
// once to stop it and write or drop the fixture.
func newRecorder(t *testing.T, path, target string) *Recorder {
	rec, err := New(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := rec.Start(target); err != nil {
		t.Fatal(err)
	}
	return rec
}

func recordMessages(t *testing.T, endpoint string) {
	c, _, err := websocket.Dial(t.Context(), endpoint, &websocket.DialOptions{}) //nolint:bodyclose // coder/websocket owns the response body.
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = c.CloseNow() })
	for _, msg := range []struct {
		typ  websocket.MessageType
		data []byte
	}{
		{websocket.MessageText, []byte("hello")},
		{websocket.MessageBinary, []byte{0, 1, 2, 255}},
	} {
		if err := c.Write(t.Context(), msg.typ, msg.data); err != nil {
			t.Fatal(err)
		}
		typ, data, err := c.Read(t.Context())
		if err != nil {
			t.Fatal(err)
		}
		if typ != msg.typ || !bytes.Equal(data, msg.data) {
			t.Fatalf("echo %v %v", typ, data)
		}
	}
	_ = c.CloseNow()
}

func replayMessages(t *testing.T, endpoint string) {
	c, _, err := websocket.Dial(t.Context(), endpoint, &websocket.DialOptions{}) //nolint:bodyclose // coder/websocket owns the response body.
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = c.CloseNow() })
	for _, msg := range []struct {
		typ  websocket.MessageType
		data []byte
	}{
		{websocket.MessageText, []byte("hello")},
		{websocket.MessageBinary, []byte{0, 1, 2, 255}},
	} {
		if err := c.Write(t.Context(), msg.typ, msg.data); err != nil {
			t.Fatal(err)
		}
		typ, data, err := c.Read(t.Context())
		if err != nil {
			t.Fatal(err)
		}
		if typ != msg.typ || !bytes.Equal(data, msg.data) {
			t.Fatalf("replay %v %v", typ, data)
		}
	}
}

func echoServer(t *testing.T) string {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		c, err := websocket.Accept(w, r, &websocket.AcceptOptions{InsecureSkipVerify: true})
		if err != nil {
			t.Error(err)
			return
		}
		defer func() { _ = c.CloseNow() }()
		ctx, cancel := context.WithCancel(r.Context())
		defer cancel()
		for {
			typ, data, err := c.Read(ctx)
			if err != nil {
				return
			}
			if err := c.Write(ctx, typ, data); err != nil {
				return
			}
		}
	}))
	t.Cleanup(srv.Close)
	return "ws" + strings.TrimPrefix(srv.URL, "http")
}
