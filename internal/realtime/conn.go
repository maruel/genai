// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Package realtime provides cancellable WebSocket I/O with bounded reads for
// realtime providers.
package realtime

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/coder/websocket"
)

// Conn owns one connection, its lifetime cancellation, and a keepalive worker.
// The provider runs one Read loop; writes can run concurrently. Cancelling a
// per-call I/O context terminates the connection, as does lifetime cancellation.
type Conn struct {
	ws       *websocket.Conn
	ctx      context.Context
	cancel   context.CancelCauseFunc
	closed   chan struct{}
	closeErr error
	wg       sync.WaitGroup
}

// Dial opens a connection using the provider's HTTP authentication and transport.
// maxReadBytes bounds inbound messages. maxReadBytes, pingInterval and
// pingTimeout must be positive. Call Close once on success.
func Dial(ctx context.Context, url string, hc *http.Client, maxReadBytes int, pingInterval, pingTimeout time.Duration) (*Conn, error) {
	if maxReadBytes <= 0 {
		return nil, errors.New("realtime read limit must be positive")
	}
	// Never let an authentication wrapper reapply credentials to a redirected
	// origin. Copy the client so the caller's redirect policy remains unchanged.
	client := *hc
	client.CheckRedirect = func(req *http.Request, via []*http.Request) error {
		first := via[0].URL
		if req.URL.Scheme != first.Scheme || !strings.EqualFold(req.URL.Host, first.Host) {
			return errors.New("realtime WebSocket upgrade redirect changed origin")
		}
		if hc.CheckRedirect != nil {
			return hc.CheckRedirect(req, via)
		}
		if len(via) >= 10 {
			return errors.New("too many realtime WebSocket redirects")
		}
		return nil
	}
	// coder/websocket owns the response body, including failed upgrades.
	ws, _, err := websocket.Dial(ctx, url, &websocket.DialOptions{HTTPClient: &client}) //nolint:bodyclose // coder/websocket owns and closes upgrade response bodies.
	if err != nil {
		// Handshake errors may include URL credentials or response bodies.
		if ctx.Err() != nil {
			return nil, ctx.Err()
		}
		return nil, errors.New("realtime WebSocket handshake failed")
	}
	ws.SetReadLimit(int64(maxReadBytes))
	life, cancel := context.WithCancelCause(ctx)
	c := &Conn{ws: ws, ctx: life, cancel: cancel, closed: make(chan struct{})}
	context.AfterFunc(life, func() {
		if err := ws.CloseNow(); err != nil && !errors.Is(err, net.ErrClosed) && !errors.Is(err, io.EOF) {
			// A read/write failure may already have closed the socket. Unexpected
			// cleanup failures are retained without echoing peer-controlled text.
			c.closeErr = errors.New("realtime WebSocket cleanup failed")
		}
		close(c.closed)
	})
	c.wg.Go(func() {
		ticker := time.NewTicker(pingInterval)
		defer ticker.Stop()
		for {
			select {
			case <-life.Done():
				return
			case <-ticker.C:
				pingCtx, stop := context.WithTimeout(life, pingTimeout)
				err := ws.Ping(pingCtx)
				stop()
				if err != nil {
					if life.Err() == nil {
						cancel(errors.New("realtime WebSocket keepalive failed"))
					}
					return
				}
			}
		}
	})
	return c, nil
}

// Err returns the terminal cause, if any.
func (c *Conn) Err() error {
	return context.Cause(c.ctx)
}

// Done is closed when the connection terminates, including lifetime cancellation.
func (c *Conn) Done() <-chan struct{} {
	return c.ctx.Done()
}

// Read returns one owned data message. Providers decode its JSON payload.
// Gemini uses binary messages; OpenAI uses text messages. I/O errors terminate the session.
func (c *Conn) Read(ctx context.Context) ([]byte, error) {
	if err := c.Err(); err != nil {
		return nil, err
	}
	_, data, err := c.ws.Read(ctx)
	if err != nil {
		return nil, c.fail(ctx, err)
	}
	return data, nil
}

// WriteRaw writes one complete text message under sendMu. It rejects empty
// messages and leaves message validation to the caller. A failed write cannot
// be retried safely.
func WriteRaw(ctx context.Context, c *Conn, sendMu chan struct{}, data []byte) error {
	if len(data) == 0 {
		return errors.New("raw realtime message is empty")
	}
	select {
	case sendMu <- struct{}{}:
		defer func() { <-sendMu }()
	case <-ctx.Done():
		return ctx.Err()
	case <-c.Done():
		return c.Err()
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	if err := c.Err(); err != nil {
		return err
	}
	return c.Write(ctx, data)
}

// Write writes one text message. A failed write cannot be retried safely.
func (c *Conn) Write(ctx context.Context, data []byte) error {
	if err := c.Err(); err != nil {
		return err
	}
	if err := c.ws.Write(ctx, websocket.MessageText, data); err != nil {
		return c.fail(ctx, err)
	}
	return nil
}

// Abort terminates the connection because a recognized provider message is invalid.
func (c *Conn) Abort(err error) {
	c.cancel(err)
}

// Close stops I/O and waits for socket and worker cleanup.
func (c *Conn) Close() error {
	c.cancel(io.EOF)
	<-c.closed
	c.wg.Wait()
	return c.closeErr
}

func (c *Conn) fail(ctx context.Context, err error) error {
	if cause := c.Err(); cause != nil {
		return cause
	}
	switch {
	case ctx.Err() != nil:
		err = ctx.Err()
	case websocket.CloseStatus(err) == websocket.StatusNormalClosure || websocket.CloseStatus(err) == websocket.StatusGoingAway:
		err = io.EOF
	default:
		// Peer close reasons can echo credentials; preserve the category only.
		err = fmt.Errorf("realtime WebSocket I/O failed (close status %d)", websocket.CloseStatus(err))
	}
	c.cancel(err)
	return c.Err()
}
