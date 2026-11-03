// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Package realtimetest provides scripted WebSocket peers, session checks and a
// fake session.
//
// It supports realtime provider tests.
package realtimetest

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"iter"
	"net"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"
	"time"

	"github.com/coder/websocket"

	"github.com/maruel/genai"
)

// Peer starts a local WebSocket server and returns its ws:// URL. check may be
// nil and runs before the upgrade. script runs for each accepted connection.
func Peer(t *testing.T, check func(*http.Request), script func(context.Context, *websocket.Conn)) string {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if check != nil {
			check(r)
		}
		ws, err := websocket.Accept(w, r, nil)
		if err != nil {
			t.Error(err)
			return
		}
		t.Cleanup(func() {
			if err := ws.CloseNow(); err != nil && !errors.Is(err, net.ErrClosed) {
				t.Error(err)
			}
		})
		ctx, cancel := context.WithTimeout(r.Context(), 5*time.Second)
		t.Cleanup(cancel)
		script(ctx, ws)
	}))
	t.Cleanup(srv.Close)
	return srv.URL
}

// Write sends one text message.
func Write(t *testing.T, ctx context.Context, ws *websocket.Conn, data string) {
	if err := ws.Write(ctx, websocket.MessageText, []byte(data)); err != nil {
		t.Error(err)
	}
}

// Read returns the next text message as a JSON object.
func Read(t *testing.T, ctx context.Context, ws *websocket.Conn) map[string]any {
	_, b, err := ws.Read(ctx)
	if err != nil {
		t.Error(err)
		return nil
	}
	var m map[string]any
	if err := json.Unmarshal(b, &m); err != nil {
		t.Error(err)
	}
	return m
}

// Close registers cleanup that closes s and reports the error.
func Close(t *testing.T, s genai.RealtimeSession) {
	t.Cleanup(func() {
		if err := s.Close(); err != nil {
			t.Error(err)
		}
	})
}

// Receive returns the next event and requires it to have kind k.
func Receive(t *testing.T, s genai.RealtimeSession, k genai.RealtimeEventKind) genai.RealtimeEvent {
	next, stop := iter.Pull2(s.Receive(t.Context()))
	defer stop()
	e, err, ok := next()
	if !ok {
		t.Fatal("realtime session ended")
	}
	if err != nil {
		t.Fatal(err)
	}
	if e.Kind != k {
		t.Fatalf("event %+v, want %s", e, k)
	}
	return e
}

// FakeSession is a scripted genai.RealtimeSession that records accepted inputs.
//
// Receive yields the events Reply queues for each accepted input and blocks
// otherwise. End ends the stream as a provider disconnect would, without closing
// the session. Close ends the stream and returns CloseErr. Call Close once. Send
// serializes concurrent calls, rejects input after End with io.EOF and after
// Close with io.ErrClosedPipe.
type FakeSession struct {
	// SessionInfo is returned by Info.
	SessionInfo genai.RealtimeSessionInfo
	// Reply scripts the events Send queues for one input. The turn argument is
	// the ordinal of the accepted input. A nil Reply queues no event.
	Reply func(turn int, in genai.RealtimeInput) []genai.RealtimeEvent
	// CloseErr is returned by Close.
	CloseErr error
	// Sent records the inputs Send accepted, in order.
	Sent []genai.RealtimeInput

	mu     sync.Mutex
	sendMu sync.Mutex
	queue  []genai.RealtimeEvent
	notify chan struct{}
	closed chan struct{}
	ended  chan struct{}
}

// NewFakeSession returns a FakeSession that reports info and queues no event.
func NewFakeSession(info genai.RealtimeSessionInfo) *FakeSession { //nolint:gocritic // hugeParam: callers build the info value they assert on.
	return &FakeSession{
		SessionInfo: info,
		notify:      make(chan struct{}, 1),
		closed:      make(chan struct{}),
		ended:       make(chan struct{}),
	}
}

// Close implements genai.RealtimeSession.
func (s *FakeSession) Close() error {
	close(s.closed)
	return s.CloseErr
}

// End ends the event stream without closing the session. Call it once.
func (s *FakeSession) End() {
	close(s.ended)
}

// Info implements genai.RealtimeSession.
func (s *FakeSession) Info() genai.RealtimeSessionInfo {
	return s.SessionInfo
}

// Receive implements genai.RealtimeSession.
func (s *FakeSession) Receive(ctx context.Context) iter.Seq2[genai.RealtimeEvent, error] {
	return func(yield func(genai.RealtimeEvent, error) bool) {
		for {
			s.mu.Lock()
			queued := len(s.queue) != 0
			var e genai.RealtimeEvent
			if queued {
				e = s.queue[0]
				s.queue = s.queue[1:]
			}
			s.mu.Unlock()
			if queued {
				if !yield(e, nil) {
					return
				}
				continue
			}
			select {
			case <-ctx.Done():
				yield(genai.RealtimeEvent{}, ctx.Err())
				return
			case <-s.closed:
				yield(genai.RealtimeEvent{}, io.EOF)
				return
			case <-s.ended:
				yield(genai.RealtimeEvent{}, io.EOF)
				return
			case <-s.notify:
			}
		}
	}
}

// Send implements genai.RealtimeSession.
func (s *FakeSession) Send(ctx context.Context, in genai.RealtimeInput) error {
	if err := in.Validate(); err != nil {
		return err
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	select {
	case <-s.closed:
		return io.ErrClosedPipe
	case <-s.ended:
		return io.EOF
	default:
	}
	// sendMu holds the recording and the queuing together so concurrent sends
	// stay serialized, as the session contract requires.
	s.sendMu.Lock()
	defer s.sendMu.Unlock()
	s.Sent = append(s.Sent, in)
	if s.Reply == nil {
		return nil
	}
	s.mu.Lock()
	s.queue = append(s.queue, s.Reply(len(s.Sent), in)...)
	s.mu.Unlock()
	select {
	case s.notify <- struct{}{}:
	default:
	}
	return nil
}

var _ genai.RealtimeSession = (*FakeSession)(nil)
