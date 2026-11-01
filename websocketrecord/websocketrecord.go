// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Package websocketrecord provides recording and replay of WebSocket message
// exchanges for provider smoke tests.
//
// It is the WebSocket analog of httprecord (HTTP) and subprocessrecord
// (subprocess I/O). Use it to record real WebSocket interactions and replay
// them in tests without network access.
//
// A Recorder starts a local proxy. In record mode, the proxy forwards messages
// to the target and records every data message. In replay mode, the proxy
// replays the recorded script and verifies each client message.
//
// Recording format is NDJSON where each line is a JSON object with direction,
// message type and content:
//
//	{"dir":">","msg":"{\"type\":\"request\"}"}
//	{"dir":"<","type":"binary","data":"AQID"}
//
// Text messages omit the type. Binary messages use base64 data.
package websocketrecord

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync"

	"github.com/coder/websocket"
)

// message is a single recorded WebSocket message.
type message struct {
	Dir  string `json:"dir"`
	Type string `json:"type,omitempty"`
	Msg  string `json:"msg,omitempty"`
	Data []byte `json:"data,omitempty"`
}

func (m *message) payload() []byte {
	if m.Type == "binary" {
		return m.Data
	}
	return []byte(m.Msg)
}

func (m *message) messageType() websocket.MessageType {
	if m.Type == "binary" {
		return websocket.MessageBinary
	}
	return websocket.MessageText
}

func (m *message) validate() error {
	if m.Dir != ">" && m.Dir != "<" {
		return fmt.Errorf("websocketrecord: invalid fixture direction %q", m.Dir)
	}
	switch m.Type {
	case "", "text":
		if m.Data != nil {
			return errors.New("websocketrecord: text fixture line contains data")
		}
	case "binary":
		if m.Msg != "" {
			return errors.New("websocketrecord: binary fixture line contains text")
		}
	default:
		return fmt.Errorf("websocketrecord: invalid fixture type %q", m.Type)
	}
	return nil
}

// Recorder records or replays WebSocket messages through a local proxy.
//
// Call Start to start the proxy, then point the client at URL or WSURL. Call
// Close when done.
type Recorder struct {
	fixture string
	replay  bool

	ctx    context.Context
	cancel context.CancelFunc

	mu     sync.Mutex
	target string
	srv    *httptest.Server
	lines  []message
	f      io.WriteCloser
	enc    *json.Encoder
}

// New creates a Recorder for the given path.
//
// The path should not include the ".ndjson" extension; it is appended
// automatically. If the fixture file exists, the recorder replays it.
// Otherwise it records.
func New(path string) (*Recorder, error) {
	fixture := path + ".ndjson"
	ctx, cancel := context.WithCancel(context.Background())
	r := &Recorder{fixture: fixture, ctx: ctx, cancel: cancel}
	_, err := os.Stat(fixture)
	switch {
	case err == nil:
		r.replay = true
		if err := r.load(); err != nil {
			cancel()
			return nil, err
		}
	case errors.Is(err, os.ErrNotExist):
	default:
		cancel()
		return nil, fmt.Errorf("websocketrecord: failed to stat fixture: %w", err)
	}
	return r, nil
}

// Start starts the local proxy. target is the full upstream WebSocket URL and
// is required in record mode. In replay mode target is ignored.
func (r *Recorder) Start(target string) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.srv != nil {
		return errors.New("websocketrecord: already started")
	}
	if r.ctx.Err() != nil {
		return errors.New("websocketrecord: closed")
	}
	if !r.replay {
		u, err := url.Parse(target)
		if err != nil || (u.Scheme != "ws" && u.Scheme != "wss") || u.Host == "" {
			return errors.New("websocketrecord: target must be an absolute ws or wss URL")
		}
		r.target = target
		if err := os.MkdirAll(filepath.Dir(r.fixture), 0o755); err != nil {
			return fmt.Errorf("websocketrecord: failed to create directory: %w", err)
		}
		f, err := os.Create(r.fixture)
		if err != nil {
			return fmt.Errorf("websocketrecord: failed to create fixture: %w", err)
		}
		r.f = f
		r.enc = json.NewEncoder(f)
	}
	r.srv = httptest.NewServer(http.HandlerFunc(r.handle))
	return nil
}

// URL returns the local HTTP proxy URL, or empty before Start.
func (r *Recorder) URL() string {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.srv == nil {
		return ""
	}
	return r.srv.URL
}

// WSURL returns the local WebSocket proxy URL, or empty before Start.
func (r *Recorder) WSURL() string {
	u := r.URL()
	if u == "" {
		return ""
	}
	return "ws" + strings.TrimPrefix(u, "http")
}

// IsReplay reports whether the recorder is in replay mode.
func (r *Recorder) IsReplay() bool {
	return r.replay
}

// Close stops the proxy and closes the fixture file.
func (r *Recorder) Close() error {
	r.mu.Lock()
	srv := r.srv
	r.srv = nil
	f := r.f
	r.f = nil
	r.enc = nil
	r.cancel()
	r.mu.Unlock()
	var errs []error
	if srv != nil {
		srv.Close()
	}
	if f != nil {
		errs = append(errs, f.Close())
	}
	return errors.Join(errs...)
}

func (r *Recorder) load() error {
	data, err := os.ReadFile(r.fixture)
	if err != nil {
		return fmt.Errorf("websocketrecord: failed to read fixture: %w", err)
	}
	sc := bufio.NewScanner(bytes.NewReader(data))
	for sc.Scan() {
		line := sc.Bytes()
		if len(line) == 0 {
			continue
		}
		var m message
		if err := json.Unmarshal(line, &m); err != nil {
			return fmt.Errorf("websocketrecord: invalid fixture line: %w", err)
		}
		if err := m.validate(); err != nil {
			return err
		}
		r.lines = append(r.lines, m)
	}
	if err := sc.Err(); err != nil {
		return fmt.Errorf("websocketrecord: failed to scan fixture: %w", err)
	}
	return nil
}

func (r *Recorder) handle(w http.ResponseWriter, req *http.Request) {
	subprotocols := parseSubprotocols(req.Header)
	c, err := websocket.Accept(w, req, &websocket.AcceptOptions{InsecureSkipVerify: true, Subprotocols: subprotocols})
	if err != nil {
		return
	}
	defer func() { _ = c.CloseNow() }()
	ctx, cancel := context.WithCancel(req.Context())
	defer cancel()
	stop := context.AfterFunc(r.ctx, cancel) //nolint:contextcheck // recorder Close must cancel active handlers.
	defer stop()
	if r.replay {
		r.replayConn(ctx, c)
		return
	}
	r.recordConn(ctx, c, req)
}

func (r *Recorder) recordConn(ctx context.Context, c *websocket.Conn, req *http.Request) {
	target, err := r.upstreamURL(req)
	if err != nil {
		_ = c.Close(websocket.StatusInternalError, "invalid target")
		return
	}
	u, _, err := websocket.Dial(ctx, target, &websocket.DialOptions{ //nolint:bodyclose // coder/websocket owns the response body.
		HTTPHeader:   forwardHeader(req.Header),
		Subprotocols: parseSubprotocols(req.Header),
	})
	if err != nil {
		_ = c.Close(websocket.StatusInternalError, "upstream dial failed")
		return
	}
	defer func() { _ = u.CloseNow() }()
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()
	errs := make(chan error, 2)
	go func() { errs <- r.relay(ctx, c, u, ">") }()
	go func() { errs <- r.relay(ctx, u, c, "<") }()
	<-errs
	cancel()
	<-errs
}

func (r *Recorder) relay(ctx context.Context, src, dst *websocket.Conn, dir string) error {
	for {
		typ, data, err := src.Read(ctx)
		if err != nil {
			return err
		}
		if err := r.record(dir, typ, data); err != nil {
			return err
		}
		if err := dst.Write(ctx, typ, data); err != nil {
			return err
		}
	}
}

func (r *Recorder) replayConn(ctx context.Context, c *websocket.Conn) {
	r.mu.Lock()
	lines := slices.Clone(r.lines)
	r.mu.Unlock()
	for i := range lines {
		m := &lines[i]
		switch m.Dir {
		case ">":
			typ, data, err := c.Read(ctx)
			if err != nil {
				return
			}
			if typ != m.messageType() || !bytes.Equal(data, m.payload()) {
				_ = c.Close(websocket.StatusPolicyViolation, "replay message mismatch")
				return
			}
		case "<":
			if err := c.Write(ctx, m.messageType(), m.payload()); err != nil {
				return
			}
		}
	}
	_ = c.Close(websocket.StatusNormalClosure, "")
}

func (r *Recorder) record(dir string, typ websocket.MessageType, data []byte) error {
	m := message{Dir: dir}
	if typ == websocket.MessageBinary {
		m.Type = "binary"
		m.Data = data
	} else {
		m.Msg = string(data)
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.enc == nil {
		return errors.New("websocketrecord: not recording")
	}
	return r.enc.Encode(&m)
}

func (r *Recorder) upstreamURL(req *http.Request) (string, error) {
	u, err := url.Parse(r.target)
	if err != nil {
		return "", err
	}
	q := u.Query()
	for k, vs := range req.URL.Query() {
		for _, v := range vs {
			q.Add(k, v)
		}
	}
	u.RawQuery = q.Encode()
	return u.String(), nil
}

func parseSubprotocols(h http.Header) []string {
	var out []string
	for _, v := range h.Values("Sec-WebSocket-Protocol") {
		for p := range strings.SplitSeq(v, ",") {
			if p = strings.TrimSpace(p); p != "" {
				out = append(out, p)
			}
		}
	}
	return out
}

func forwardHeader(src http.Header) http.Header {
	h := src.Clone()
	for _, k := range []string{
		"Connection", "Content-Length", "Host", "Origin", "Proxy-Connection",
		"Sec-WebSocket-Accept", "Sec-WebSocket-Extensions", "Sec-WebSocket-Key",
		"Sec-WebSocket-Protocol", "Sec-WebSocket-Version", "Upgrade",
	} {
		h.Del(k)
	}
	return h
}
