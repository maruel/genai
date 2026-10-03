// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Tests for download error handling in the vid-txt_to_txt example.

package main

import (
	"io"
	"net/http"
	"strings"
	"testing"

	"github.com/maruel/genai/base"
)

func TestMainImpl(t *testing.T) {
	t.Run("downloadStatus", func(t *testing.T) {
		t.Setenv("GEMINI_API_KEY", "test-key")
		body := &downloadBody{Reader: strings.NewReader("not found")}
		prevTransport := base.DefaultTransport
		base.DefaultTransport = downloadTransport{body: body}
		t.Cleanup(func() { base.DefaultTransport = prevTransport })
		prev := http.DefaultClient
		http.DefaultClient = &http.Client{Transport: downloadTransport{body: body}}
		t.Cleanup(func() { http.DefaultClient = prev })
		if err := mainImpl(); err == nil || !strings.Contains(err.Error(), "unexpected HTTP status 404") {
			t.Errorf("mainImpl() = %v, want download status error", err)
		}
		if body.closes != 1 {
			t.Errorf("download body closed %d times, want 1", body.closes)
		}
	})
}

type downloadTransport struct {
	body *downloadBody
}

func (d downloadTransport) RoundTrip(r *http.Request) (*http.Response, error) {
	if r.URL.Host == "generativelanguage.googleapis.com" {
		body := io.NopCloser(strings.NewReader(`{"models":[{"name":"models/gemini-flash-lite-latest","supportedGenerationMethods":["generateContent"]}]}`))
		return &http.Response{StatusCode: http.StatusOK, Body: body, Header: http.Header{"Content-Type": {"application/json"}}, Request: r}, nil
	}
	return &http.Response{StatusCode: http.StatusNotFound, Body: d.body, Header: http.Header{}, Request: r}, nil
}

type downloadBody struct {
	*strings.Reader
	closes int
}

func (d *downloadBody) Close() error {
	d.closes++
	return nil
}
