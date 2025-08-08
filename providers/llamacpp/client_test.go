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
	"github.com/maruel/genai/scoreboard/scoreboardtest"
)

func TestClient(t *testing.T) {
	s := lazyServer{t: t}

	t.Run("ListModels", func(t *testing.T) {
		c, err := llamacpp.New(&genai.OptionsProvider{Remote: s.lazyStart(t), Model: base.NoModel}, func(h http.RoundTripper) http.RoundTripper {
			return testRecorder.Record(t, h)
		})
		if err != nil {
			t.Fatal(err)
		}
		genaiModels, err := c.ListModels(t.Context())
		if err != nil {
			t.Fatal(err)
		}
		if len(genaiModels) != 1 {
			t.Fatalf("unexpected: %#v", genaiModels)
		}
	})

	t.Run("Scoreboard", func(t *testing.T) {
		serverURL := s.lazyStart(t)
		c, err := llamacpp.New(&genai.OptionsProvider{Remote: serverURL, Model: base.NoModel}, func(h http.RoundTripper) http.RoundTripper {
			return testRecorder.Record(t, h)
		})
		if err != nil {
			t.Fatal(err)
		}
		var models []scoreboardtest.Model
		for _, sc := range c.Scoreboard().Scenarios {
			for _, id := range sc.Models {
				models = append(models, scoreboardtest.Model{Model: id})
			}
		}
		scoreboardtest.AssertScoreboard(t, func(t testing.TB, model scoreboardtest.Model, fn func(http.RoundTripper) http.RoundTripper) genai.Provider {
			c2, err2 := llamacpp.New(&genai.OptionsProvider{Remote: serverURL, Model: model.Model}, fn)
			if err2 != nil {
				t.Fatal(err2)
			}
			return c2
		}, models, testRecorder.Records)
	})

	// Run this at the end so there would be non-zero values.
	t.Run("Metrics", func(t *testing.T) {
		serverURL := s.lazyStart(t)
		c, err := llamacpp.New(&genai.OptionsProvider{Remote: serverURL, Model: base.NoModel}, func(h http.RoundTripper) http.RoundTripper {
			return testRecorder.Record(t, h)
		})
		if err != nil {
			t.Fatal(err)
		}
		m := llamacpp.Metrics{}
		if err := c.GetMetrics(t.Context(), &m); err != nil {
			t.Fatal(err)
		}
		t.Logf("Metrics: %+v", m)
	})
}

type lazyServer struct {
	t   testing.TB
	mu  sync.Mutex
	url string
}

func (l *lazyServer) lazyStart(t testing.TB) string {
	if os.Getenv("RECORD") != "1" && os.Getenv("CI") == "true" {
		return "http://localhost:0"
	}
	if url := os.Getenv("LLAMA_SERVER"); url != "" {
		return url
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	if l.url == "" {
		t.Log("Starting server")
		// Use the context of the parent for server lifecycle management.
		mains := strings.SplitN(llamacpp.Scoreboard.Scenarios[0].Models[0], "#", 2)
		parts := strings.Split(mains[0], "/")
		srv, err := startServer(l.t.Context(), parts[0], parts[1], parts[2], mains[1])
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
			c, err := llamacpp.New(&genai.OptionsProvider{Model: line.name}, nil)
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
