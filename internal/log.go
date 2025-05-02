// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package internal

import (
	"bytes"
	"context"
	"errors"
	"io"
	"log/slog"
	"net/http"
	"time"

	"github.com/maruel/roundtrippers"
)

// Log is a http.RoundTripper that logs each request and response via slog.
// It defaults to slog.LevelInfo level unless an error is returned from the
// roundtripper, then the final log is logged at error level.
type Log struct {
	Transport http.RoundTripper
	L         *slog.Logger
	Level     slog.Level

	_ struct{}
}

// RoundTrip implements http.RoundTripper.
func (l *Log) RoundTrip(req *http.Request) (*http.Response, error) {
	ctx := req.Context()
	rid := req.Header.Get("X-Request-ID")
	if rid == "" {
		return nil, errors.New("roundtrippers.Log requires roundtrippers.RequestID")
	}
	ll := l.L.With("id", rid, "dur", elapsedTimeValue{start: time.Now()})
	ll.Log(ctx, l.Level, "http", "url", req.URL.String(), "method", req.Method, "Content-Encoding", req.Header.Get("Content-Encoding"), "body", body)
	resp, err := l.Transport.RoundTrip(req)
	if err != nil {
		ll.ErrorContext(ctx, "http", "err", err)
	} else {
		ce := resp.Header.Get("Content-Encoding")
		cl := resp.Header.Get("Content-Length")
		ct := resp.Header.Get("Content-Type")
		ll.Log(ctx, l.Level, "http", "status", resp.StatusCode, "Content-Encoding", ce, "Content-Length", cl, "Content-Type", ct)
		resp.Body = &logBody{body: resp.Body, ctx: ctx, l: ll, level: l.Level}
	}
	return resp, err
}

func (l *Log) Unwrap() http.RoundTripper {
	return l.Transport
}

//

type captureBody struct {
	body    io.ReadCloser
	resp    *http.Response
	c       chan<- roundtrippers.Record
	content *bytes.Buffer
	err     error
}

func (c *captureBody) Read(p []byte) (int, error) {
	n, err := c.body.Read(p)
	_, _ = c.content.Write(p[:n])
	if err != nil && err != io.EOF && c.err == nil {
		c.err = err
	}
	return n, err
}

func (c *captureBody) Close() error {
	err := c.body.Close()
	c.resp.Body = io.NopCloser(c.content)
	c.c <- roundtrippers.Record{Request: c.resp.Request, Response: c.resp, Err: c.err}
	return err
}

type logBody struct {
	body  io.ReadCloser
	ctx   context.Context
	l     *slog.Logger
	level slog.Level

	responseSize int64
	err          error
}

func (l *logBody) Read(p []byte) (int, error) {
	n, err := l.body.Read(p)
	if n > 0 {
		l.responseSize += int64(n)
	}
	if err != nil && err != io.EOF && l.err == nil {
		l.err = err
	}
	return n, err
}

func (l *logBody) Close() error {
	err := l.body.Close()
	if err != nil && l.err == nil {
		l.err = err
	}
	level := l.level
	if l.err != nil {
		level = slog.LevelError
	}
	l.l.Log(l.ctx, level, "http", "size", l.responseSize, "err", l.err)
	return err
}

type elapsedTimeValue struct {
	start time.Time
}

func (v elapsedTimeValue) LogValue() slog.Value {
	return slog.DurationValue(time.Since(v.start))
}
