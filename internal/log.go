// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package internal

import (
	"io"
	"log/slog"
	"net/http"

	"github.com/maruel/roundtrippers"
)

// LogTransport logs the transport to help debugging.
func LogTransport(t http.RoundTripper) http.RoundTripper {
	ch := make(chan roundtrippers.Record, 1)
	go func() {
		for r := range ch {
			var reqb, respb []byte
			if r.Request.GetBody != nil {
				if b, _ := r.Request.GetBody(); b != nil {
					reqb, _ = io.ReadAll(b)
				}
			} else if b, ok := r.Request.Body.(io.ReadSeeker); ok {
				_, _ = b.Seek(0, io.SeekStart)
				reqb, _ = io.ReadAll(b)
			}
			if r.Response.Body != nil {
				respb, _ = io.ReadAll(r.Response.Body)
			}
			slog.InfoContext(r.Request.Context(), "log", "url", r.Request.URL.String(), "hdr", r.Request.Header, "post", reqb, "resp", respb)
		}
	}()
	return &roundtrippers.Capture{Transport: t, C: ch}
}
