// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Package internaltest is awesome sauce for unit testign.
package internaltest

import (
	"net/http"
	"net/url"
	"os"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"github.com/maruel/genai"
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
func Record(t *testing.T, h http.RoundTripper, opts ...recorder.Option) *recorder.Recorder {
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
	r, err := recorder.New("testdata/"+strings.ReplaceAll(t.Name(), "/", "_"), append(args, opts...)...)
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

//

// ChatStream runs a  ChatStream and returns the concatenated response.
func ChatStream(t *testing.T, c genai.ChatProvider, msgs genai.Messages, opts *genai.ChatOptions) genai.Messages {
	ctx := t.Context()
	chunks := make(chan genai.MessageFragment)
	end := make(chan genai.Messages, 1)
	go func() {
		var pendingMsgs genai.Messages
		defer func() {
			end <- pendingMsgs
			close(end)
		}()
		for {
			select {
			case <-ctx.Done():
				return
			case pkt, ok := <-chunks:
				if !ok {
					return
				}
				var err2 error
				if pendingMsgs, err2 = pkt.Accumulate(pendingMsgs); err2 != nil {
					t.Error(err2)
					return
				}
			}
		}
	}()
	err := c.ChatStream(ctx, msgs, opts, chunks)
	close(chunks)
	responses := <-end
	t.Logf("Raw responses: %#v", responses)
	if err != nil {
		t.Fatal(err)
	}
	return responses
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
