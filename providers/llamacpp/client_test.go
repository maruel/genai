// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package llamacpp_test

import (
	"context"
	"net/http"
	"os"
	"strings"
	"sync"
	"testing"

	"github.com/maruel/genai"
	"github.com/maruel/genai/base"
	"github.com/maruel/genai/internal"
	"github.com/maruel/genai/internal/internaltest"
	"github.com/maruel/genai/providers/llamacpp"
	"gopkg.in/dnaeon/go-vcr.v4/pkg/cassette"
	"gopkg.in/dnaeon/go-vcr.v4/pkg/recorder"
)

func TestClient(t *testing.T) {
	s := lazyServer{t: t}

	t.Run("Scoreboard", func(t *testing.T) {
		internaltest.TestScoreboard(t, func(t *testing.T, m string) genai.ProviderGen { return s.getClient(t) }, nil)
	})
}

type lazyServer struct {
	t   *testing.T
	mu  sync.Mutex
	url string
}

func (l *lazyServer) shouldStart(t *testing.T) (string, func(http.RoundTripper) http.RoundTripper) {
	transport := testRecorder.Record(t, http.DefaultTransport,
		recorder.WithHook(func(i *cassette.Interaction) error { return internaltest.SaveIgnorePort(t, i) }, recorder.AfterCaptureHook),
		recorder.WithMatcher(internaltest.MatchIgnorePort),
	)
	wrapper := func(http.RoundTripper) http.RoundTripper { return transport }
	if !transport.IsNewCassette() {
		return "http://localhost:0", wrapper
	}
	name := "testdata/" + strings.ReplaceAll(t.Name(), "/", "_") + ".yaml"
	suffix := " (forced)"
	if os.Getenv("RECORD") != "1" {
		suffix = " (missing " + name + ")"
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	t.Cleanup(func() {
		if t.Failed() {
			t.Log("Removing record")
			_ = os.Remove(name)
		}
	})
	if l.url == "" {
		t.Log("Starting server" + suffix)
		// Use the context of the parent for server lifecycle management.
		parts := strings.Split(llamacpp.Scoreboard.Scenarios[0].Models[0], "/")
		srv, err := startServer(l.t.Context(), parts[0], parts[1], parts[2])
		if err != nil {
			t.Fatal(err)
		}
		l.url = srv.URL()
		l.t.Cleanup(func() {
			if err := srv.Close(); err != nil && err != context.Canceled {
				l.t.Error(err)
			}
		})
	} else {
		t.Log("Recording " + suffix)
	}
	return l.url, wrapper
}

func (l *lazyServer) getClient(t *testing.T) genai.ProviderGen {
	serverURL, wrapper := l.shouldStart(t)
	c, err := llamacpp.New(serverURL, nil, wrapper)
	if err != nil {
		t.Fatal(err)
	}
	return c
}

// This test doesn't require the server to start.
func TestClient_Preferred(t *testing.T) {
	data := []struct {
		name string
		want string
	}{
		{base.PreferredCheap, ""},
		{base.PreferredGood, ""},
		{base.PreferredSOTA, ""},
	}
	for _, line := range data {
		t.Run(line.name, func(t *testing.T) {
			c, err := llamacpp.New(line.name, nil, nil)
			if err != nil {
				t.Fatal(err)
			}
			if got := c.ModelID(); got != line.want {
				t.Fatalf("got model %q, want %q", got, line.want)
			}
		})
	}
}

var testRecorder *internaltest.Records

func TestMain(m *testing.M) {
	testRecorder = internaltest.NewRecords()
	code := m.Run()
	os.Exit(max(code, testRecorder.Close()))
}

func init() {
	internal.BeLenient = false
}
