// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Package internal is awesome sauce.
package internal

import (
	"bytes"
	"context"
	"crypto/rand"
	"encoding/base64"
	"io"
	"log/slog"
	"net/http"
	"time"
)

// TransportLog logs the requests and responses.
type TransportLog struct {
	R http.RoundTripper
}

func (t *TransportLog) RoundTrip(r *http.Request) (*http.Response, error) {
	ctx := r.Context()
	start := time.Now()
	ll := slog.Default().With("id", genID())
	ll.InfoContext(ctx, "http", "url", r.URL.String(), "method", r.Method, "Content-Encoding", r.Header.Get("Content-Encoding"))
	resp, err := t.R.RoundTrip(r)
	if err != nil {
		ll.ErrorContext(ctx, "http", "duration", time.Since(start), "err", err)
	} else {
		ce := resp.Header.Get("Content-Encoding")
		cl := resp.Header.Get("Content-Length")
		ct := resp.Header.Get("Content-Type")
		ll.InfoContext(ctx, "http", "duration", time.Since(start), "status", resp.StatusCode, "Content-Encoding", ce, "Content-Length", cl, "Content-Type", ct)
		resp.Body = &loggingBody{r: resp.Body, ctx: ctx, start: start, l: ll}
	}
	return resp, err
}

type loggingBody struct {
	r     io.ReadCloser
	ctx   context.Context
	start time.Time
	l     *slog.Logger

	responseSize    int64
	responseContent bytes.Buffer
	err             error
}

func (l *loggingBody) Read(p []byte) (int, error) {
	n, err := l.r.Read(p)
	if n > 0 {
		l.responseSize += int64(n)
		_, _ = l.responseContent.Write(p[:n])
	}
	if err != nil && err != io.EOF && l.err == nil {
		l.err = err
	}
	return n, err
}

func (l *loggingBody) Close() error {
	err := l.r.Close()
	if err != nil && l.err == nil {
		l.err = err
	}
	level := slog.LevelInfo
	if l.err != nil {
		level = slog.LevelError
	}
	// l.l.Log(l.ctx, level, "http", "duration", time.Since(l.start), "size", l.responseSize, "err", l.err)
	l.l.Log(l.ctx, level, "http", "duration", time.Since(l.start), "body", l.responseContent.String(), "err", l.err)
	return err
}

func genID() string {
	var bytes [12]byte
	rand.Read(bytes[:])
	return base64.RawURLEncoding.EncodeToString(bytes[:])
}
