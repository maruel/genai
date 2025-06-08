// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Package internaltest is awesome sauce for unit testign.
package internaltest

import (
	"flag"
	"fmt"
	"io/fs"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"testing"
	"time"

	"gopkg.in/dnaeon/go-vcr.v4/pkg/cassette"
	"gopkg.in/dnaeon/go-vcr.v4/pkg/recorder"
)

type Records struct {
	mu          sync.Mutex
	preexisting map[string]struct{}
	recorded    []string
}

func NewRecords() *Records {
	r := &Records{preexisting: make(map[string]struct{})}
	const root = "testdata" + string(os.PathSeparator)
	err := filepath.WalkDir(root, func(path string, d fs.DirEntry, err error) error {
		if err == nil && !d.IsDir() && strings.HasSuffix(path, ".yaml") {
			if p := path[len(root):]; p != "example.yaml" {
				r.preexisting[p] = struct{}{}
			}
		}
		return err
	})
	if os.IsNotExist(err) {
		return r
	}
	if err != nil {
		panic(err)
	}
	return r
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
	r.mu.Lock()
	defer r.mu.Unlock()
	for _, f := range r.recorded {
		delete(r.preexisting, f)
	}
	code := 0
	if len(r.preexisting) != 0 {
		code = 1
		println("Found orphaned recordings:")
		names := make([]string, 0, len(r.preexisting))
		for f := range r.preexisting {
			names = append(names, f)
		}
		sort.Strings(names)
		for _, f := range names {
			println(fmt.Sprintf("- %q", f))
		}
	}
	return code
}

func (r *Records) Signal(t *testing.T) string {
	name := strings.ReplaceAll(strings.ReplaceAll(t.Name(), "/", string(os.PathSeparator)), ":", "-")
	if d := filepath.Dir(name); d != "." {
		if err := os.MkdirAll(filepath.Join("testdata", d), 0o755); err != nil {
			t.Fatal(err)
		}
	}
	r.mu.Lock()
	r.recorded = append(r.recorded, name+".yaml")
	r.mu.Unlock()
	return name
}

// Record records and replays HTTP requests for unit testing.
//
// When the environment variable RECORD=1 is set, it forcibly re-record the
// cassettes and save in testdata/<testname>.yaml.
//
// It ignores the port number in the URL both for recording and playback so it
// works with local services like ollama and llama-server.
func (r *Records) Record(t *testing.T, h http.RoundTripper, opts ...recorder.Option) *recorder.Recorder {
	mode := recorder.ModeRecordOnce
	if os.Getenv("RECORD") == "1" {
		mode = recorder.ModeRecordOnly
	}
	args := []recorder.Option{
		recorder.WithHook(trimResponseHeaders, recorder.AfterCaptureHook),
		recorder.WithMode(mode),
		recorder.WithSkipRequestLatency(true),
		recorder.WithRealTransport(h),
		recorder.WithMatcher(DefaultMatcher),
	}
	rr, err := recorder.New("testdata/"+r.Signal(t), append(args, opts...)...)
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
	return DefaultMatcher(r, i)
}

// DefaultMatcher ignores authentication via API keys.
var DefaultMatcher = cassette.NewDefaultMatcher(cassette.WithIgnoreHeaders("Authorization", "X-Api-Key", "X-Key", "X-Request-Id"))

func trimResponseHeaders(i *cassette.Interaction) error {
	// Authentication via API keys.
	i.Request.Headers.Del("Authorization")
	i.Request.Headers.Del("X-Api-Key")
	i.Request.Headers.Del("X-Key")
	// Noise.
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
