// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Package internaltest is awesome sauce for unit testign.
package internaltest

import (
	"net/http"
	"net/url"
	"os"
	"strings"
	"testing"
	"time"

	"gopkg.in/dnaeon/go-vcr.v4/pkg/cassette"
	"gopkg.in/dnaeon/go-vcr.v4/pkg/recorder"
)

// Record records and replays HTTP requests for unit testing.
//
// When the environment variable RECORD=1 is set, it forcibly re-record the
// cassettes and save in testdata/<testname>.yaml.
//
// It ignores the port number in the URL both for recording and playback so it
// works with local services like ollama and llama-server.
func Record(t *testing.T) *recorder.Recorder {
	m := cassette.NewDefaultMatcher()
	fnMatch := func(r *http.Request, i cassette.Request) bool {
		r = r.Clone(r.Context())
		r.URL.Host = strings.Split(r.URL.Host, ":")[0]
		r.Host = strings.Split(r.Host, ":")[0]
		i.Headers.Del("Authorization")
		i.Headers.Del("X-Api-Key")
		return m(r, i)
	}
	fnSave := func(i *cassette.Interaction) error {
		i.Request.Host = strings.Split(i.Request.Host, ":")[0]
		u, err := url.Parse(i.Request.URL)
		if err != nil {
			t.Fatal(err)
		}
		u.Host = strings.Split(u.Host, ":")[0]
		i.Request.URL = u.String()
		i.Request.Headers.Del("Authorization")
		i.Request.Headers.Del("X-Api-Key")
		i.Response.Headers.Del("Date")
		i.Response.Duration = i.Response.Duration.Round(time.Millisecond)
		return nil
	}
	mode := recorder.ModeRecordOnce
	if os.Getenv("RECORD") == "1" {
		mode = recorder.ModeRecordOnly
	}
	r, err := recorder.New(
		"testdata/"+strings.ReplaceAll(t.Name(), "/", "_"),
		recorder.WithMatcher(fnMatch),
		recorder.WithHook(fnSave, recorder.AfterCaptureHook),
		recorder.WithMode(mode),
		recorder.WithSkipRequestLatency(true))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := r.Stop(); err != nil {
			t.Error(err)
		}
	})
	return r
}
