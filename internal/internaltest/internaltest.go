// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Package internaltest is awesome sauce for unit testing.
package internaltest

import (
	"context"
	"errors"
	"flag"
	"iter"
	"log/slog"
	"net/http"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/maruel/genai"
	"github.com/maruel/genai/internal"
	"github.com/maruel/genai/internal/myrecorder"
	"github.com/maruel/httpjson"
	"gopkg.in/dnaeon/go-vcr.v4/pkg/recorder"
)

type Records struct {
	Records *myrecorder.Records
}

func NewRecords() *Records {
	rr, err := myrecorder.NewRecords("testdata")
	if err != nil {
		panic(err)
	}
	if err := rr.Signal("example"); err != nil {
		panic(err.Error())
	}
	return &Records{Records: rr}
}

func (r *Records) Close() error {
	filtered := false
	flag.Visit(func(f *flag.Flag) {
		if f.Name == "test.run" {
			filtered = true
		}
	})
	if filtered {
		return nil
	}
	return r.Records.Close()
}

// Record records and replays HTTP requests for unit testing.
//
// When the environment variable RECORD=1 is set, it forcibly re-record the
// cassettes and save in testdata/<testname>.yaml.
//
// It ignores the port number in the URL both for recording and playback so it
// works with local services like ollama and llama-server.
func (r *Records) Record(t testing.TB, h http.RoundTripper, opts ...recorder.Option) *myrecorder.Recorder {
	return r.RecordWithName(t, t.Name(), h, opts...)
}

func (r *Records) RecordWithName(t testing.TB, name string, h http.RoundTripper, opts ...recorder.Option) *myrecorder.Recorder {
	rr, err := r.Records.Record(name, h, opts...)
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

// ValidateWordResponse validates that the response contains exactly one of the expected words.
func ValidateWordResponse(t testing.TB, resp genai.Result, want ...string) {
	got := resp.String()
	cleaned := strings.TrimRight(strings.TrimSpace(strings.ToLower(got)), ".!")
	if !slices.Contains(want, cleaned) {
		t.Helper()
		t.Fatalf("Expected %q, got %q", strings.Join(want, ", "), got)
	}
}

//

// Log returns a slog.Logger that redirects to testing.TB.Log() and adds it to the Context.
func Log(tb testing.TB) (context.Context, *slog.Logger) {
	level := &slog.LevelVar{}
	if *superVerbose {
		flag.Visit(func(f *flag.Flag) {
			if f.Name == "test.v" {
				level.Set(slog.LevelDebug)
			}
		})
	}
	l := slog.New(slog.NewTextHandler(&WriterToLog{T: tb}, &slog.HandlerOptions{
		AddSource: true,
		Level:     level,
		ReplaceAttr: func(groups []string, a slog.Attr) slog.Attr {
			switch a.Key {
			case "level":
				a.Key = "l"
				a.Value = slog.StringValue(a.Value.String()[:3])
			case "source":
				a.Key = "s"
				s := a.Value.Any().(*slog.Source)
				s.File = filepath.Base(s.File)
			case "time":
				a = slog.Attr{}
			}
			return a
		},
	}))
	ctx := internal.WithLogger(tb.Context(), l)
	return ctx, l
}

// LogFile returns a file to be used as log that is only saved when the test fails.
//
// This makes it possible to inspect the logs for debugging, yet keeps the test cacheable by "go test".
func LogFile(tb testing.TB, cache, name string) *os.File {
	p := filepath.Join(tb.TempDir(), name)
	l, err := os.Create(p)
	if err != nil {
		tb.Fatal(err)
	}
	tb.Cleanup(func() {
		if err2 := l.Close(); err2 != nil {
			tb.Error(err2)
		}
		if tb.Failed() {
			if err2 := os.Rename(p, filepath.Join(cache, name)); err2 != nil {
				tb.Error(err2)
			}
		}
	})
	return l
}

// InjectOptions injects options into the provider GenSync and GenStream calls.
type InjectOptions struct {
	genai.Provider
	Opts []genai.Options
}

func (i *InjectOptions) GenSync(ctx context.Context, msgs genai.Messages, opts ...genai.Options) (genai.Result, error) {
	return i.Provider.GenSync(ctx, msgs, append(opts, i.Opts...)...)
}

func (i *InjectOptions) GenStream(ctx context.Context, msgs genai.Messages, opts ...genai.Options) (iter.Seq[genai.ReplyFragment], func() (genai.Result, error)) {
	return i.Provider.GenStream(ctx, msgs, append(opts, i.Opts...)...)
}

func (i *InjectOptions) Unwrap() genai.Provider {
	return i.Provider
}

// HideHTTP500 is hides HTTP 500 errors from the reply.
type HideHTTP500 struct {
	genai.Provider
}

func (h *HideHTTP500) Unwrap() genai.Provider {
	return h.Provider
}

func (h *HideHTTP500) GenSync(ctx context.Context, msgs genai.Messages, opts ...genai.Options) (genai.Result, error) {
	resp, err := h.Provider.GenSync(ctx, msgs, opts...)
	if err != nil {
		var herr *httpjson.Error
		if errors.As(err, &herr) && herr.StatusCode == 500 {
			err = errors.New("hiding a HTTP 500 error")
		}
	}
	return resp, err
}

func (h *HideHTTP500) GenStream(ctx context.Context, msgs genai.Messages, opts ...genai.Options) (iter.Seq[genai.ReplyFragment], func() (genai.Result, error)) {
	fragments, finish := h.Provider.GenStream(ctx, msgs, opts...)
	return fragments, func() (genai.Result, error) {
		res, err := finish()
		if err != nil {
			var herr *httpjson.Error
			if errors.As(err, &herr) && herr.StatusCode == 500 {
				err = errors.New("hiding a HTTP 500 error")
			}
		}
		return res, err
	}
}

//

// WriterToLog wraps t.Log() to implement io.Writer
type WriterToLog struct {
	T testing.TB
}

func (tw *WriterToLog) Write(p []byte) (n int, err error) {
	// Sadly the log output is attributed to this line.
	tw.T.Log(strings.TrimSpace(string(p)))
	return len(p), nil
}

var superVerbose = flag.Bool("superv", false, "super verbose; enables internaltest.Log() to log more")
