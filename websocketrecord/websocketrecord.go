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
// Recording format is NDJSON where each line is a JSON object with direction
// and message content:
//
//	{"dir":">","msg":"{\"type\":\"request\"}"}
//	{"dir":"<","msg":"{\"type\":\"response\"}"}
//
// In record mode, all sent and received messages are written to the fixture
// file. In replay mode, a local WebSocket server replays the recorded
// messages, matching request-response pairs from the recording.
package websocketrecord

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http/httptest"
	"os"
	"path/filepath"
	"sync"

	"golang.org/x/net/websocket"
)

// message is a single recorded WebSocket message.
type message struct {
	Dir string `json:"dir"`
	Msg string `json:"msg"`
}

// Recorder records or replays WebSocket message exchanges.
//
// Use New to create a recorder, then use its DialContext method to establish
// WebSocket connections. In record mode, the real server is dialed and
// messages are captured. In replay mode, a local server replays the fixture.
//
// Always call Stop() when done to release resources.
type Recorder struct {
	fixture string
	replay  bool

	mu        sync.Mutex
	f         io.WriteCloser
	lines     []message
	replaySrv *httptest.Server
}

// New creates a Recorder for the given path.
//
// The path should not include the ".ndjson" extension; it is appended
// automatically. If the fixture file already exists, the recorder replays it.
// Otherwise it records.
func New(path string) (*Recorder, error) {
	fixture := path + ".ndjson"
	r := &Recorder{fixture: fixture}
	if _, err := os.Stat(fixture); err == nil {
		r.replay = true
		data, err := os.ReadFile(fixture)
		if err != nil {
			return nil, fmt.Errorf("websocketrecord: failed to read fixture: %w", err)
		}
		br := bytesReader(data)
		sc := bufio.NewScanner(&br)
		for sc.Scan() {
			line := sc.Bytes()
			if len(line) == 0 {
				continue
			}
			var m message
			if err := json.Unmarshal(line, &m); err != nil {
				return nil, fmt.Errorf("websocketrecord: invalid fixture line: %w", err)
			}
			r.lines = append(r.lines, m)
		}
		if err := sc.Err(); err != nil {
			return nil, fmt.Errorf("websocketrecord: failed to scan fixture: %w", err)
		}
	}
	return r, nil
}

// Stop closes the recording file (record mode) or the replay server (replay
// mode).
//
// It is safe to call multiple times.
func (r *Recorder) Stop() error {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.replaySrv != nil {
		r.replaySrv.Close()
		r.replaySrv = nil
	}
	if r.f != nil {
		err := r.f.Close()
		r.f = nil
		return err
	}
	return nil
}

// IsReplay reports whether the recorder is in replay mode.
func (r *Recorder) IsReplay() bool {
	return r.replay
}

// DialContext dials a WebSocket connection with recording or replay support.
//
// In record mode it dials the real server and wraps the connection for
// recording. In replay mode it starts a local server that replays the fixture
// and connects to it.
//
// The returned Conn provides Send and Receive methods that must be used
// instead of websocket.Message.Send and websocket.Message.Receive.
func (r *Recorder) DialContext(ctx context.Context, cfg *websocket.Config) (*Conn, error) {
	if r.replay {
		return r.replayDial(ctx, cfg)
	}
	return r.recordDial(ctx, cfg)
}

// recordDial dials the real server and returns a recording-wrapped Conn.
func (r *Recorder) recordDial(ctx context.Context, cfg *websocket.Config) (*Conn, error) {
	ws, err := cfg.DialContext(ctx)
	if err != nil {
		return nil, err
	}
	c := &Conn{ws: ws, rec: r, replay: false}
	// Open the fixture file for writing.
	if err := os.MkdirAll(filepath.Dir(r.fixture), 0o755); err != nil {
		_ = ws.Close()
		return nil, fmt.Errorf("websocketrecord: failed to create directory: %w", err)
	}
	f, err := os.Create(r.fixture)
	if err != nil {
		_ = ws.Close()
		return nil, fmt.Errorf("websocketrecord: failed to create fixture: %w", err)
	}
	c.enc = json.NewEncoder(f)
	c.fixtureFile = f
	return c, nil
}

// replayDial starts a local WebSocket server and connects to it.
func (r *Recorder) replayDial(ctx context.Context, cfg *websocket.Config) (*Conn, error) {
	// Build a handler that replays recorded messages.
	r.mu.Lock()
	lines := make([]message, len(r.lines))
	copy(lines, r.lines)
	r.mu.Unlock()

	handler := func(ws *websocket.Conn) {
		defer func() { _ = ws.Close() }()
		// Track position in the recorded lines.
		pos := 0
		for pos < len(lines) {
			// Read one message from the client.
			var clientMsg string
			if err := websocket.Message.Receive(ws, &clientMsg); err != nil {
				return
			}
			// Advance past the matching sent message.
			if pos < len(lines) && lines[pos].Dir == ">" {
				pos++
			}
			// Replay all received messages until the next sent message.
			for pos < len(lines) && lines[pos].Dir == "<" {
				if err := websocket.Message.Send(ws, lines[pos].Msg); err != nil {
					return
				}
				pos++
			}
		}
	}

	srv := httptest.NewServer(websocket.Handler(handler))
	r.mu.Lock()
	r.replaySrv = srv
	r.mu.Unlock()

	// Build a config pointing to the local server.
	localCfg, err := websocket.NewConfig("ws"+srv.URL[4:], cfg.Origin.String())
	if err != nil {
		return nil, fmt.Errorf("websocketrecord: failed to create local config: %w", err)
	}
	localCfg.Header = cfg.Header
	localCfg.Protocol = cfg.Protocol
	localCfg.TlsConfig = cfg.TlsConfig

	ws, err := localCfg.DialContext(ctx)
	if err != nil {
		return nil, fmt.Errorf("websocketrecord: failed to dial local server: %w", err)
	}
	return &Conn{ws: ws, rec: r, replay: true}, nil
}

// Conn wraps a *websocket.Conn with message-level recording or replay.
//
// Use Send and Receive instead of websocket.Message.Send and
// websocket.Message.Receive to automatically record or replay messages.
type Conn struct {
	ws     *websocket.Conn
	rec    *Recorder
	replay bool

	mu          sync.Mutex
	enc         *json.Encoder
	fixtureFile io.WriteCloser
}

// Send sends a text message over the WebSocket connection.
//
// In record mode, the message is also written to the fixture file.
func (c *Conn) Send(data string) error {
	if c.ws == nil {
		return errors.New("websocketrecord: connection is closed")
	}
	if !c.replay {
		if err := c.recordOutgoing(data); err != nil {
			return err
		}
	}
	return websocket.Message.Send(c.ws, data)
}

// Receive receives a text message from the WebSocket connection.
//
// In record mode, the received message is also written to the fixture file.
func (c *Conn) Receive(msg *string) error {
	if c.ws == nil {
		return errors.New("websocketrecord: connection is closed")
	}
	if err := websocket.Message.Receive(c.ws, msg); err != nil {
		return err
	}
	if !c.replay && *msg != "" {
		if err := c.recordIncoming(*msg); err != nil {
			return err
		}
	}
	return nil
}

// Close closes the underlying WebSocket connection and the fixture file.
func (c *Conn) Close() error {
	c.mu.Lock()
	defer c.mu.Unlock()
	var errs []error
	if c.fixtureFile != nil {
		if err := c.fixtureFile.Close(); err != nil {
			errs = append(errs, err)
		}
		c.fixtureFile = nil
	}
	if c.ws != nil {
		if err := c.ws.Close(); err != nil {
			errs = append(errs, err)
		}
		c.ws = nil
	}
	return errors.Join(errs...)
}

// WS returns the underlying *websocket.Conn for advanced use cases.
func (c *Conn) WS() *websocket.Conn {
	return c.ws
}

func (c *Conn) recordOutgoing(data string) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.enc == nil {
		return nil
	}
	return c.enc.Encode(message{Dir: ">", Msg: data})
}

func (c *Conn) recordIncoming(data string) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.enc == nil {
		return nil
	}
	return c.enc.Encode(message{Dir: "<", Msg: data})
}

// bytesReader is a minimal io.Reader over a byte slice, avoiding
// the extra allocation of strings.NewReader.
type bytesReader []byte

func (b *bytesReader) Read(p []byte) (int, error) {
	if len(*b) == 0 {
		return 0, io.EOF
	}
	n := copy(p, *b)
	*b = (*b)[n:]
	return n, nil
}
