// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package llamacpp_test

import (
	"context"
	"errors"
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
	"github.com/maruel/genai/scoreboard/scoreboardtest"
	"github.com/maruel/httpjson"
)

func TestClient(t *testing.T) {
	s := lazyServer{t: t}

	t.Run("Scoreboard", func(t *testing.T) {
		models := []scoreboardtest.Model{{Model: llamacpp.Scoreboard.Scenarios[0].Models[0]}}
		scoreboardtest.AssertScoreboard(t, func(t testing.TB, model scoreboardtest.Model, fn func(http.RoundTripper) http.RoundTripper) genai.Provider {
			serverURL := s.lazyStart(t)
			c, err := llamacpp.New(serverURL, nil, fn)
			if err != nil {
				t.Fatal(err)
			}
			return &hideHTTP500{c}
		}, models, testRecorder.Records)
	})
}

type hideHTTP500 struct {
	*llamacpp.Client
}

func (h *hideHTTP500) Unwrap() genai.Provider {
	return h.Client
}

func (h *hideHTTP500) ModelID() string {
	// Hack, it should be a separate class.
	return llamacpp.Scoreboard.Scenarios[0].Models[0]
}

func (h *hideHTTP500) GenSync(ctx context.Context, msgs genai.Messages, opts genai.Options) (genai.Result, error) {
	resp, err := h.Client.GenSync(ctx, msgs, opts)
	if err != nil {
		var herr *httpjson.Error
		if errors.As(err, &herr) && herr.StatusCode == 500 {
			return resp, errors.New("server is having a bad day")
		}
		return resp, err
	}
	return resp, err
}

func (h *hideHTTP500) GenStream(ctx context.Context, msgs genai.Messages, chunks chan<- genai.ContentFragment, opts genai.Options) (genai.Result, error) {
	resp, err := h.Client.GenStream(ctx, msgs, chunks, opts)
	if err != nil {
		var herr *httpjson.Error
		if errors.As(err, &herr) && herr.StatusCode == 500 {
			return resp, errors.New("server is having a bad day")
		}
		return resp, err
	}
	return resp, err
}

type lazyServer struct {
	t   *testing.T
	mu  sync.Mutex
	url string
}

func (l *lazyServer) lazyStart(t testing.TB) string {
	if os.Getenv("RECORD") != "1" && os.Getenv("CI") == "true" {
		return "http://localhost:0"
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	if l.url == "" {
		t.Log("Starting server")
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
	}
	return l.url
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
