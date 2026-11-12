// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Tests cleanup failures on actual WebSocket upgrades.

package realtime_test

import (
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/coder/websocket"
	"github.com/maruel/genai/internal/realtime"
)

type failingCloseTransport struct{ base http.RoundTripper }

func (t *failingCloseTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	resp, err := t.base.RoundTrip(req)
	if err == nil && resp.StatusCode == http.StatusSwitchingProtocols {
		stream, ok := resp.Body.(io.ReadWriteCloser)
		if !ok {
			return nil, errors.New("upgraded response body is not writable")
		}
		resp.Body = &failingCloseStream{ReadWriteCloser: stream}
	}
	return resp, err
}

type failingCloseStream struct{ io.ReadWriteCloser }

func (s *failingCloseStream) Close() error {
	return errors.Join(s.ReadWriteCloser.Close(), errors.New("secret socket cleanup failure"))
}

func TestConn(t *testing.T) {
	t.Run("Close", func(t *testing.T) {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			ws, err := websocket.Accept(w, r, nil)
			if err != nil {
				t.Error(err)
				return
			}
			t.Cleanup(func() { _ = ws.CloseNow() })
			_, _, _ = ws.Read(r.Context())
		}))
		t.Cleanup(srv.Close)
		hc := &http.Client{Transport: &failingCloseTransport{base: http.DefaultTransport}}
		c, err := realtime.Dial(t.Context(), "ws"+strings.TrimPrefix(srv.URL, "http"), hc, 16<<20, time.Hour, time.Second)
		if err != nil {
			t.Fatal(err)
		}
		if err := c.Close(); err == nil || strings.Contains(err.Error(), "secret") {
			t.Fatalf("cleanup error was dropped or exposed: %v", err)
		}
	})
	t.Run("Read", func(t *testing.T) {
		const reason = "Operation is not implemented, or supported, or enabled."
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			ws, err := websocket.Accept(w, r, nil)
			if err != nil {
				t.Error(err)
				return
			}
			_ = ws.Close(websocket.StatusPolicyViolation, reason)
		}))
		t.Cleanup(srv.Close)
		c, err := realtime.Dial(t.Context(), "ws"+strings.TrimPrefix(srv.URL, "http"), http.DefaultClient, 1<<10, 0, 0)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { _ = c.Close() })
		_, err = c.Read(t.Context())
		const want = `realtime WebSocket closed (status 1008): "` + reason + `"`
		if err == nil || err.Error() != want {
			t.Fatalf("Read() error = %v; want %s", err, want)
		}
	})
	t.Run("write limit", func(t *testing.T) {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			ws, err := websocket.Accept(w, r, nil)
			if err != nil {
				return
			}
			defer func() { _ = ws.CloseNow() }()
			for {
				if _, _, err := ws.Read(r.Context()); err != nil {
					return
				}
			}
		}))
		t.Cleanup(srv.Close)
		const limit = 8
		url := "ws" + strings.TrimPrefix(srv.URL, "http")
		if _, err := realtime.Dial(t.Context(), url, http.DefaultClient, 0, time.Hour, time.Second); err == nil {
			t.Fatal("non-positive message limit admitted")
		}
		c, err := realtime.Dial(t.Context(), url, http.DefaultClient, limit, 0, 0)
		if err != nil {
			t.Fatal(err)
		}
		defer func() {
			if err := c.Close(); err != nil {
				t.Error(err)
			}
		}()
		if err := c.Write(t.Context(), make([]byte, limit+1)); err != nil {
			t.Fatal(err)
		}
	})
}
