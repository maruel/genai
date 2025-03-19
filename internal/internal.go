// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Package internal is awesome sauce.
package internal

import (
	"bytes"
	"context"
	"io"
	"log/slog"
	"net/http"
	"time"
)

// TransportHeaders add headers to the requests.
type TransportHeaders struct {
	R http.RoundTripper
	H map[string]string
}

func (t *TransportHeaders) RoundTrip(req *http.Request) (*http.Response, error) {
	req = req.Clone(req.Context())
	for k, v := range t.H {
		req.Header.Set(k, v)
	}
	return t.R.RoundTrip(req)
}

// TransportLog logs the requests and responses.
type TransportLog struct {
	R http.RoundTripper
}

func (t *TransportLog) RoundTrip(r *http.Request) (*http.Response, error) {
	start := time.Now()
	resp, err := t.R.RoundTrip(r)
	if resp != nil {
		resp.Body = &loggingBody{
			ReadCloser: resp.Body,
			req:        r,
			resp:       resp,
			start:      start,
			ctx:        r.Context(),
		}
	}
	return resp, err
}

type loggingBody struct {
	io.ReadCloser
	n     int64
	req   *http.Request
	resp  *http.Response
	start time.Time
	ctx   context.Context
}

func (l *loggingBody) Read(p []byte) (int, error) {
	n, err := l.ReadCloser.Read(p)
	if n > 0 {
		l.n += int64(n)
	}
	return n, err
}

func (l *loggingBody) Close() error {
	err := l.ReadCloser.Close()
	slog.InfoContext(l.ctx, "http", "url", l.req.URL, "status", l.resp.Status, "dur", time.Since(l.start), "size", l.n)
	return err
}

type capturingBody struct {
	io.ReadCloser
	B bytes.Buffer
}

func (b *capturingBody) Read(p []byte) (int, error) {
	n, err := b.ReadCloser.Read(p)
	if n > 0 {
		_, _ = b.B.Write(p[:n])
	}
	return n, err
}
