// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Package httprecord provides safe HTTP recording logic for users that was to understand the API and do smoke
// tests.
//
// It is known to be safe from saving API keys and credentials.
package httprecord

import (
	"bytes"
	"net/http"
	"net/url"
	"regexp"
	"strings"
	"time"

	"gopkg.in/dnaeon/go-vcr.v4/pkg/cassette"
	"gopkg.in/dnaeon/go-vcr.v4/pkg/recorder"
)

// New starts an HTTP records and/or playback HTTP session.
//
// It drops from recording and ignores during playback:
//
//   - Host port number (llamacpp and ollama).
//   - HTTP request headers: "Authorization", "X-Api-Key", "X-Goog-Api-Key", "X-Key", "X-Request-Id".
//   - HTTP response header: "Anthropic-Organization-Id", "Date", "Request-Id", "Set-Cookie".
//   - Cloudflare's account ID in the URL path
//   - Empty recordings are not saved.
//
// Don't forget to call Stop()!
func New(path string, h http.RoundTripper, opts ...recorder.Option) (*recorder.Recorder, error) {
	args := []recorder.Option{
		recorder.WithHook(trimResponseHeaders, recorder.AfterCaptureHook),
		recorder.WithHook(trimRecordingCloudflare, recorder.AfterCaptureHook),
		recorder.WithHook(trimRecordingHostPort, recorder.AfterCaptureHook),
		recorder.WithSkipRequestLatency(true),
		recorder.WithRealTransport(h),
		recorder.WithMatcher(matchIgnorePort),
		recorder.WithFS(&skipEmptyFS{FS: cassette.NewDiskFS()}),
	}
	return recorder.New(path, append(args, opts...)...)
}

type skipEmptyFS struct {
	cassette.FS
}

func (c *skipEmptyFS) WriteFile(name string, data []byte) error {
	if bytes.Contains(data, []byte("interactions: []")) {
		// Do not save files without an interaction.
		return nil
	}
	return c.FS.WriteFile(name, data)
}

// Trimmers.

func trimResponseHeaders(i *cassette.Interaction) error {
	// Authentication via API keys.
	i.Request.Headers.Del("Authorization")
	i.Request.Headers.Del("X-Api-Key")
	i.Request.Headers.Del("X-Goog-Api-Key")
	i.Request.Headers.Del("X-Key")
	// OMG What are they thinking? This happens on HTTP 302 redirect when fetching Veo generated videos:
	i.Response.Headers.Del("X-Goog-Api-Key")
	// Noise.
	i.Request.Headers.Del("X-Request-Id")
	i.Response.Headers.Del("Date")
	i.Response.Headers.Del("Request-Id")
	// Remove this here since it also happens in openaicompatible.
	i.Response.Headers.Del("Anthropic-Organization-Id")
	// The cookie may be used for authentication?
	i.Response.Headers.Del("Set-Cookie")
	// Noise.
	i.Response.Duration = i.Response.Duration.Round(time.Millisecond)
	return nil
}

var reCloudflareAccount = regexp.MustCompile(`/accounts/[0-9a-fA-F]{32}/`)

func trimRecordingCloudflare(i *cassette.Interaction) error {
	// Zap the account ID from the URL path before saving.
	i.Request.URL = reCloudflareAccount.ReplaceAllString(i.Request.URL, "/accounts/ACCOUNT_ID/")
	return nil
}

// trimRecordingHostPort is a recorder.HookFunc to remove the port number when recording.
func trimRecordingHostPort(i *cassette.Interaction) error {
	i.Request.Host = strings.Split(i.Request.Host, ":")[0]
	u, err := url.Parse(i.Request.URL)
	if err != nil {
		return err
	}
	u.Host = strings.Split(u.Host, ":")[0]
	i.Request.URL = u.String()
	return nil
}

// Matchers.

// matchIgnorePort is a recorder.MatcherFunc that ignore the host port number. This is useful for locally
// hosted LLM providers like llamacpp and ollama.
func matchIgnorePort(r *http.Request, i cassette.Request) bool {
	r = r.Clone(r.Context())
	r.URL.Host = strings.Split(r.URL.Host, ":")[0]
	r.Host = strings.Split(r.Host, ":")[0]
	return matchCassetteCloudflare(r, i)
}

func matchCassetteCloudflare(r *http.Request, i cassette.Request) bool {
	r = r.Clone(r.Context())
	// When matching, ignore the account ID from the URL path.
	r.URL.Path = reCloudflareAccount.ReplaceAllString(r.URL.Path, "/accounts/ACCOUNT_ID/")
	return defaultMatcher(r, i)
}

var defaultMatcher = cassette.NewDefaultMatcher(cassette.WithIgnoreHeaders("Authorization", "X-Api-Key", "X-Goog-Api-Key", "X-Key", "X-Request-Id"))
