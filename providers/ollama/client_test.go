// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package ollama_test

import (
	"context"
	"log/slog"
	"net/http"
	"os"
	"slices"
	"strings"
	"sync"
	"testing"

	"github.com/maruel/genai"
	"github.com/maruel/genai/adapters"
	"github.com/maruel/genai/internal"
	"github.com/maruel/genai/internal/internaltest"
	"github.com/maruel/genai/providers/ollama"
	"github.com/maruel/genai/smoke/smoketest"
	"github.com/maruel/roundtrippers"
)

// Not implementing TestClient_AllModels since we need to preload Ollama models. Can be done later.

func TestClient(t *testing.T) {
	s := lazyServer{t: t}

	t.Run("Scoreboard", func(t *testing.T) {
		var models []smoketest.Model
		for _, sc := range ollama.Scoreboard().Scenarios {
			for _, m := range sc.Models {
				models = append(models, smoketest.Model{Model: m, Reason: sc.Reason})
			}
		}
		smoketest.Run(t, func(t testing.TB, model smoketest.Model, fn func(http.RoundTripper) http.RoundTripper) genai.Provider {
			serverURL := s.lazyStart(t)
			ctx, l := internaltest.Log(t)
			fnWithLog := func(h http.RoundTripper) http.RoundTripper {
				if fn != nil {
					h = fn(h)
				}
				return &roundtrippers.Log{
					Transport: h,
					Logger:    l,
					Level:     slog.LevelDebug,
				}
			}
			c, err := ollama.New(ctx, &genai.ProviderOptions{Remote: serverURL, Model: model.Model}, fnWithLog)
			if err != nil {
				t.Fatal(err)
			}
			if strings.HasPrefix(model.Model, "qwen") {
				if !model.Reason {
					t.Fatal("expected thinking")
				}
				// Check if it has predefined thinking tokens.
				for _, sc := range c.Scoreboard().Scenarios {
					if sc.Reason && slices.Contains(sc.Models, model.Model) {
						if sc.ReasoningTokenStart != "" && sc.ReasoningTokenEnd != "" {
							return &adapters.ProviderReasoning{
								Provider:            &adapters.ProviderAppend{Provider: c, Append: genai.Request{Text: "\n\n/think"}},
								ReasoningTokenStart: sc.ReasoningTokenStart,
								ReasoningTokenEnd:   sc.ReasoningTokenEnd,
							}
						}
						break
					}
				}
			}
			return c
		}, models, testRecorder.Records)
	})

	t.Run("errors", func(t *testing.T) {
		data := []internaltest.ProviderError{
			{
				Name: "bad model",
				Opts: genai.ProviderOptions{
					Model: "bad_model",
				},
				ErrGenSync:   "pull failed: http 500\npull model manifest: file does not exist",
				ErrGenStream: "pull failed: http 500\npull model manifest: file does not exist",
			},
		}
		f := func(t *testing.T, opts genai.ProviderOptions) (genai.Provider, error) {
			return s.getClient(t, opts.Model)
		}
		internaltest.TestClient_Provider_errors(t, f, data)
	})
}

type lazyServer struct {
	t   *testing.T
	mu  sync.Mutex
	url string
}

func (l *lazyServer) lazyStartWithRecord(t testing.TB) (string, func(http.RoundTripper) http.RoundTripper) {
	transport := testRecorder.Record(t, http.DefaultTransport)
	wrapper := func(http.RoundTripper) http.RoundTripper { return transport }
	if !transport.IsNewCassette() {
		return "http://localhost:0", wrapper
	}
	name := "testdata/" + strings.ReplaceAll(t.Name(), "/", "_") + ".yaml"
	t.Cleanup(func() {
		if t.Failed() {
			t.Log("Removing record")
			_ = os.Remove(name)
		}
	})
	return l.lazyStart(t), wrapper
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
		srv, err := startServer(l.t.Context())
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

func (l *lazyServer) getClient(t testing.TB, model string) (genai.Provider, error) {
	serverURL, wrapper := l.lazyStartWithRecord(t)
	return ollama.New(t.Context(), &genai.ProviderOptions{Remote: serverURL, Model: model}, wrapper)
}

// This test doesn't require the server to start.
func TestClient_Preferred(t *testing.T) {
	data := []struct {
		name string
		want string
	}{
		{genai.ModelCheap, "gemma3:1b"},
		{genai.ModelGood, "qwen3:30b"},
		{genai.ModelSOTA, "qwen3:32b"},
	}
	for _, line := range data {
		t.Run(line.name, func(t *testing.T) {
			c, err := ollama.New(t.Context(), &genai.ProviderOptions{Model: line.name}, nil)
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
