// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package ollama_test

import (
	"context"
	"errors"
	"iter"
	"log/slog"
	"net/http"
	"os"
	"strings"
	"sync"
	"testing"

	"github.com/maruel/genai"
	"github.com/maruel/genai/internal"
	"github.com/maruel/genai/internal/internaltest"
	"github.com/maruel/genai/providers/ollama"
	"github.com/maruel/genai/scoreboard"
	"github.com/maruel/genai/smoke/smoketest"
	"github.com/maruel/roundtrippers"
)

// Not implementing TestClient_AllModels since we need to preload Ollama models. Can be done later.

func TestClient(t *testing.T) {
	testRecorder := internaltest.NewRecords()
	t.Cleanup(func() {
		if err := testRecorder.Close(); err != nil {
			t.Error(err)
		}
	})

	s := lazyServer{t: t}

	t.Run("Capabilities", func(t *testing.T) {
		c, err := ollama.New(t.Context(), genai.ProviderOptionRemote(s.lazyStart(t)))
		if err != nil {
			t.Fatal(err)
		}
		internaltest.TestCapabilities(t, c)
	})

	t.Run("Scoreboard", func(t *testing.T) {
		scenarios := ollama.Scoreboard().Scenarios
		models := make([]scoreboard.Model, 0, len(scenarios))
		for _, sc := range scenarios {
			for _, m := range sc.Models {
				models = append(models, scoreboard.Model{Model: m, Reason: sc.Reason})
			}
		}
		smoketest.Run(t, func(t testing.TB, model scoreboard.Model, fn func(http.RoundTripper) http.RoundTripper) genai.Provider {
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
			opts := []genai.ProviderOption{genai.ProviderOptionRemote(s.lazyStart(t)), genai.ProviderOptionTransportWrapper(fnWithLog)}
			if model.Model != "" {
				opts = append(opts, genai.ProviderOptionModel(model.Model))
			}
			c, err := ollama.New(ctx, opts...)
			if err != nil {
				t.Fatal(err)
			}
			if strings.HasPrefix(model.Model, "qwen") {
				// Ollama v0.17.4+ auto-enables thinking for thinking-capable
				// models. Use the "think" API parameter to control it.
				if !model.Reason {
					return &ollamaThinkOff{Provider: c}
				}
			}
			return c
		}, models, testRecorder.Records)
	})

	// This test doesn't require the server to start.
	t.Run("Preferred", func(t *testing.T) {
		internaltest.TestPreferredModels(t, func(st *testing.T, model string, modality genai.Modality) (genai.Provider, error) {
			opts := []genai.ProviderOption{
				genai.ProviderOptionModalities{modality},
				genai.ProviderOptionRemote("http://localhost:66666"),
				genai.ProviderOptionTransportWrapper(func(h http.RoundTripper) http.RoundTripper {
					return testRecorder.Record(st, h)
				}),
			}
			if model != "" {
				opts = append(opts, genai.ProviderOptionModel(model))
			}
			return ollama.New(st.Context(), opts...)
		})
	})

	t.Run("TextOutputDocInput", func(t *testing.T) {
		internaltest.TestTextOutputDocInput(t, func(t *testing.T) genai.Provider {
			opts := []genai.ProviderOption{
				genai.ProviderOptionRemote(s.lazyStart(t)),
				genai.ModelCheap,
				genai.ProviderOptionTransportWrapper(func(h http.RoundTripper) http.RoundTripper {
					return testRecorder.Record(t, h)
				}),
			}
			c, err := ollama.New(t.Context(), opts...)
			if err != nil {
				t.Fatal(err)
			}
			return c
		})
	})

	t.Run("errors", func(t *testing.T) {
		data := []internaltest.ProviderError{
			{
				Name: "bad model",
				Opts: []genai.ProviderOption{
					genai.ProviderOptionModel("bad_model"),
				},
				ErrGenSync:   "pull failed: http 500\npull model manifest: file does not exist",
				ErrGenStream: "pull failed: http 500\npull model manifest: file does not exist",
			},
		}
		f := func(t *testing.T, opts ...genai.ProviderOption) (genai.Provider, error) {
			serverURL := ""
			transport := testRecorder.Record(t, http.DefaultTransport)
			wrapper := func(h http.RoundTripper) http.RoundTripper { return transport }
			if !transport.IsNewCassette() {
				serverURL = "http://localhost:0"
			} else {
				name := "testdata/" + strings.ReplaceAll(t.Name(), "/", "_") + ".yaml"
				t.Cleanup(func() {
					if t.Failed() {
						t.Log("Removing record")
						_ = os.Remove(name)
					}
				})
				serverURL = s.lazyStart(t)
			}
			opts = append(opts, genai.ProviderOptionRemote(serverURL), genai.ProviderOptionTransportWrapper(wrapper))
			return ollama.New(t.Context(), opts...)
		}
		internaltest.TestClientProviderErrors(t, f, data)
	})
}

type lazyServer struct {
	t   *testing.T
	mu  sync.Mutex
	url string
}

func (l *lazyServer) lazyStart(t testing.TB) string {
	// Skip server startup when not recording. The HTTP cassettes in testdata/
	// are replayed by the recording transport so the URL is never contacted.
	// Only RECORD=all needs a real server; RECORD=failure_only replays in the
	// parent and re-runs failures as a subprocess with RECORD=all.
	if os.Getenv("RECORD") != "all" {
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
			if err := srv.Close(); err != nil && !errors.Is(err, context.Canceled) {
				l.t.Error(err)
			}
		})
	}
	return l.url
}

// ollamaThinkOff wraps a provider to inject ReasoningEffortOff on every call,
// disabling thinking for models that default to it.
type ollamaThinkOff struct {
	genai.Provider
}

func (o *ollamaThinkOff) GenSync(ctx context.Context, msgs genai.Messages, opts ...genai.GenOption) (genai.Result, error) {
	return o.Provider.GenSync(ctx, msgs, append(opts, &ollama.GenOptionText{ReasoningEffort: ollama.ReasoningEffortOff})...)
}

func (o *ollamaThinkOff) GenStream(ctx context.Context, msgs genai.Messages, opts ...genai.GenOption) (iter.Seq[genai.Reply], func() (genai.Result, error)) {
	return o.Provider.GenStream(ctx, msgs, append(opts, &ollama.GenOptionText{ReasoningEffort: ollama.ReasoningEffortOff})...)
}

func (o *ollamaThinkOff) Unwrap() genai.Provider {
	return o.Provider
}

func init() {
	internal.BeLenient = false
}
