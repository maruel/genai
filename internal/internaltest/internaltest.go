// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Package internaltest is awesome sauce for unit testign.
package internaltest

import (
	"fmt"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"github.com/maruel/genai"
	"gopkg.in/dnaeon/go-vcr.v4/pkg/cassette"
	"gopkg.in/dnaeon/go-vcr.v4/pkg/recorder"
)

type Records struct {
	mu          sync.Mutex
	preexisting map[string]struct{}
	recorded    []string
}

func NewRecords() *Records {
	r := &Records{
		preexisting: make(map[string]struct{}),
	}
	const prefix = "testdata/"
	m, err := filepath.Glob(prefix + "*.yaml")
	if err != nil {
		panic(err)
	}
	for _, f := range m {
		r.preexisting[f[len(prefix):]] = struct{}{}
	}
	return r
}

func (r *Records) Close() {
	r.mu.Lock()
	defer r.mu.Unlock()
	for _, f := range r.recorded {
		if _, ok := r.preexisting[f]; ok {
			delete(r.preexisting, f)
		} else {
			println(fmt.Sprintf("- Found orphaned recording %q", f))
		}
	}
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
		recorder.WithHook(func(i *cassette.Interaction) error { return cleanup(t, i) }, recorder.AfterCaptureHook),
		recorder.WithMode(mode),
		recorder.WithSkipRequestLatency(true),
		recorder.WithRealTransport(h),
	}
	name := strings.ReplaceAll(t.Name(), "/", "_")
	r.mu.Lock()
	r.recorded = append(r.recorded, name)
	r.mu.Unlock()
	rr, err := recorder.New("testdata/"+name, append(args, opts...)...)
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

// MatchIgnorePort is a recorder.MatcherFunc.
func MatchIgnorePort(r *http.Request, i cassette.Request) bool {
	r = r.Clone(r.Context())
	r.URL.Host = strings.Split(r.URL.Host, ":")[0]
	r.Host = strings.Split(r.Host, ":")[0]
	return defaultMatcher(r, i)
}

// AssertResponses ensures the responses we got match what we want.
func AssertResponses(t *testing.T, want, got genai.Messages) {
	if len(got) != len(want) {
		t.Errorf("Expected %d responses, got %d", len(want), len(got))
	}
	for i := range got {
		for j := range got[i].ToolCalls {
			if got[i].ToolCalls[j].ID != "" {
				got[i].ToolCalls[j].ID = strconv.Itoa(i + j + 1)
			}
		}
	}
	for i := range want {
		if diff := cmp.Diff(&want[i], &got[i]); diff != "" {
			t.Errorf("(+want), (-got):\n%s", diff)
		}
	}
	if t.Failed() {
		t.FailNow()
	}
}

//

func cleanup(t *testing.T, i *cassette.Interaction) error {
	if i.Request.Headers.Get("Authorization") != "" || i.Request.Headers.Get("X-Api-Key") != "" {
		t.Fatal("got unexpected token; get roundtrippers ordering")
	}
	// Noise.
	i.Response.Headers.Del("Date")
	// The cookie may be used for authentication?
	i.Response.Headers.Del("Set-Cookie")
	// Noise.
	i.Response.Duration = i.Response.Duration.Round(time.Millisecond)
	return nil
}

var defaultMatcher = cassette.NewDefaultMatcher()
