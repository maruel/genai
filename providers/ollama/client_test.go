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
		c, err := ollama.New(t.Context(), &genai.ProviderOptions{Remote: s.lazyStart(t), Model: genai.ModelNone}, nil)
		if err != nil {
			t.Fatal(err)
		}
		internaltest.TestCapabilities(t, c)
	})

	t.Run("Scoreboard", func(t *testing.T) {
		var models []scoreboard.Model
		for _, sc := range ollama.Scoreboard().Scenarios {
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
			c, err := ollama.New(ctx, &genai.ProviderOptions{Remote: s.lazyStart(t), Model: model.Model}, fnWithLog)
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

	// This test doesn't require the server to start.
	t.Run("Preferred", func(t *testing.T) {
		internaltest.TestPreferredModels(t, func(st *testing.T, model string, modality genai.Modality) (genai.Provider, error) {
			opts := genai.ProviderOptions{
				Model:            model,
				OutputModalities: genai.Modalities{modality},
				Remote:           "http://localhost:66666",
			}
			return ollama.New(st.Context(), &opts, func(h http.RoundTripper) http.RoundTripper {
				return testRecorder.Record(st, h)
			})
		})
	})

	t.Run("TextOutputDocInput", func(t *testing.T) {
		serverURL := s.lazyStart(t)
		internaltest.TestTextOutputDocInput(t, func(t *testing.T) genai.Provider {
			c, err := ollama.New(t.Context(), &genai.ProviderOptions{Remote: serverURL, Model: genai.ModelCheap}, nil)
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
				Opts: genai.ProviderOptions{
					Model: "bad_model",
				},
				ErrGenSync:   "pull failed: http 500\npull model manifest: file does not exist",
				ErrGenStream: "pull failed: http 500\npull model manifest: file does not exist",
			},
		}
		f := func(t *testing.T, opts genai.ProviderOptions) (genai.Provider, error) {
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
			opts.Remote = serverURL
			return ollama.New(t.Context(), &opts, wrapper)
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

func init() {
	internal.BeLenient = false
}
