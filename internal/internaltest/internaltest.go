// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Package internaltest is awesome sauce for unit testign.
package internaltest

import (
	"context"
	"flag"
	"fmt"
	"log/slog"
	"net/http"
	"net/url"
	"os"
	"strings"
	"testing"

	"github.com/maruel/genai/internal"
	"gopkg.in/dnaeon/go-vcr.v4/pkg/cassette"
	"gopkg.in/dnaeon/go-vcr.v4/pkg/recorder"
)

type Records struct {
	r *internal.Records
}

func NewRecords() *Records {
	rr, err := internal.NewRecords("testdata")
	if err != nil {
		panic(err)
	}
	rr.Signal("example")
	return &Records{r: rr}
}

func (r *Records) Close() int {
	filtered := false
	flag.Visit(func(f *flag.Flag) {
		if f.Name == "test.run" {
			filtered = true
		}
	})
	if filtered {
		return 0
	}
	if err := r.r.Close(); err != nil {
		fmt.Fprintf(os.Stderr, "%v\n", err)
		return 1
	}
	return 0
}

// Record records and replays HTTP requests for unit testing.
//
// When the environment variable RECORD=1 is set, it forcibly re-record the
// cassettes and save in testdata/<testname>.yaml.
//
// It ignores the port number in the URL both for recording and playback so it
// works with local services like ollama and llama-server.
func (r *Records) Record(t *testing.T, h http.RoundTripper, opts ...recorder.Option) *recorder.Recorder {
	rr, err := r.r.Record(t.Name(), h, opts...)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := rr.Stop(); err != nil {
			t.Error(err)
		}
	})
	return rr
}

// SaveIgnorePort is a recorder.HookFunc (with a testing.T).
func SaveIgnorePort(t *testing.T, i *cassette.Interaction) error {
	i.Request.Host = strings.Split(i.Request.Host, ":")[0]
	u, err := url.Parse(i.Request.URL)
	if err != nil {
		t.Fatal(err)
	}
	u.Host = strings.Split(u.Host, ":")[0]
	i.Request.URL = u.String()
	return nil
}

// MatchIgnorePort is a recorder.MatcherFunc that ignore the host port number. This is useful for locally
// hosted LLM providers like llamacpp and ollama.
func MatchIgnorePort(r *http.Request, i cassette.Request) bool {
	r = r.Clone(r.Context())
	r.URL.Host = strings.Split(r.URL.Host, ":")[0]
	r.Host = strings.Split(r.Host, ":")[0]
	return internal.DefaultMatcher(r, i)
}

//

// Log returns a slog.Logger that redirects to testing.TB.Log() and adds it to the Context.
func Log(tb testing.TB) (context.Context, *slog.Logger) {
	level := &slog.LevelVar{}
	flag.Visit(func(f *flag.Flag) {
		if f.Name == "test.v" {
			level.Set(slog.LevelDebug)
		}
	})
	l := slog.New(slog.NewTextHandler(&testWriter{t: tb}, &slog.HandlerOptions{
		Level: level,
		ReplaceAttr: func(groups []string, a slog.Attr) slog.Attr {
			switch a.Key {
			case "level":
				a.Key = "l"
				a.Value = slog.StringValue(a.Value.String()[:3])
			case "time":
				a = slog.Attr{}
			}
			return a
		},
	}))
	ctx := internal.WithLogger(tb.Context(), l)
	return ctx, l
}

// testWriter wraps t.Log() to implement io.Writer
type testWriter struct {
	t testing.TB
}

func (tw *testWriter) Write(p []byte) (n int, err error) {
	tw.t.Log(strings.TrimSpace(string(p)))
	return len(p), nil
}
