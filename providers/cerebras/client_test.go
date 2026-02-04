// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package cerebras_test

import (
	"net/http"
	"os"
	"slices"
	"strings"
	"testing"

	"github.com/maruel/genai"
	"github.com/maruel/genai/adapters"
	"github.com/maruel/genai/internal"
	"github.com/maruel/genai/internal/internaltest"
	"github.com/maruel/genai/providers/cerebras"
	"github.com/maruel/genai/scoreboard"
	"github.com/maruel/genai/smoke/smoketest"
)

func getClientInner(t *testing.T, fn func(http.RoundTripper) http.RoundTripper, opts ...genai.ProviderOption) (genai.Provider, error) {
	hasAPIKey := false
	for _, opt := range opts {
		if _, ok := opt.(genai.ProviderOptionAPIKey); ok {
			hasAPIKey = true
			break
		}
	}
	if !hasAPIKey && os.Getenv("CEREBRAS_API_KEY") == "" {
		opts = append(opts, genai.ProviderOptionAPIKey("<insert_api_key_here>"))
	}
	// ctx, l := internaltest.Log(t)
	// return &roundtrippers.Log{Transport: r, Logger: l, Level: slog.LevelWarn}
	if fn != nil {
		opts = append([]genai.ProviderOption{genai.ProviderOptionTransportWrapper(fn)}, opts...)
	}
	return cerebras.New(t.Context(), opts...)
}

func TestClient(t *testing.T) {
	testRecorder := internaltest.NewRecords()
	t.Cleanup(func() {
		if err := testRecorder.Close(); err != nil {
			t.Error(err)
		}
	})
	cl, err2 := getClientInner(t, func(h http.RoundTripper) http.RoundTripper {
		return testRecorder.RecordWithName(t, t.Name()+"/Warmup", h)
	})
	if err2 != nil {
		t.Fatal(err2)
	}
	cachedModels, err2 := cl.ListModels(t.Context())
	if err2 != nil {
		t.Fatal(err2)
	}
	getClient := func(t *testing.T, m string) genai.Provider {
		t.Parallel()
		opts := []genai.ProviderOption{genai.ProviderOptionPreloadedModels(cachedModels)}
		if m != "" {
			opts = append(opts, genai.ProviderOptionModel(m))
		}
		ci, err := getClientInner(t, func(h http.RoundTripper) http.RoundTripper {
			return testRecorder.Record(t, h)
		}, opts...)
		if err != nil {
			t.Fatal(err)
		}
		return ci
	}

	t.Run("Capabilities", func(t *testing.T) {
		internaltest.TestCapabilities(t, getClient(t, ""))
	})

	t.Run("Scoreboard", func(t *testing.T) {
		c := getClient(t, "")
		genaiModels, err := c.ListModels(t.Context())
		if err != nil {
			t.Fatal(err)
		}
		scenarios := c.Scoreboard().Scenarios
		var models []scoreboard.Model
		for _, m := range genaiModels {
			id := m.GetID()
			thinking := false
			for _, sc := range scenarios {
				if slices.Contains(sc.Models, id) {
					thinking = sc.Reason
					break
				}
			}
			models = append(models, scoreboard.Model{Model: id, Reason: thinking})
		}
		getClientRT := func(t testing.TB, model scoreboard.Model, fn func(http.RoundTripper) http.RoundTripper) genai.Provider {
			provOpts := []genai.ProviderOption{
				genai.ProviderOptionPreloadedModels(cachedModels),
			}
			if model.Model != "" {
				provOpts = append(provOpts, genai.ProviderOptionModel(model.Model))
			}
			if os.Getenv("CEREBRAS_API_KEY") == "" {
				provOpts = append(provOpts, genai.ProviderOptionAPIKey("<insert_api_key_here>"))
			}
			ctx := t.Context()
			// ctx, l := internaltest.Log(t)
			fnWithLog := func(h http.RoundTripper) http.RoundTripper {
				if fn != nil {
					h = fn(h)
				}
				return h
				// return &roundtrippers.Log{Transport: h, Logger: l, Level: slog.LevelDebug}
			}
			if fn != nil {
				provOpts = append([]genai.ProviderOption{genai.ProviderOptionTransportWrapper(fnWithLog)}, provOpts...)
			}
			c, err2 := cerebras.New(ctx, provOpts...)
			if err2 != nil {
				t.Fatal(err2)
			}
			var p genai.Provider = c
			if model.Reason {
				// Check if it has predefined thinking tokens.
				for _, sc := range c.Scoreboard().Scenarios {
					if sc.Reason && slices.Contains(sc.Models, model.Model) {
						if sc.ReasoningTokenEnd != "" {
							// This is bad. We should have a better way to determine if a model needs to be prompted to think
							// (qwen-3-32b) or must not (qwen-3-235b-a22b-thinking-2507).
							if !strings.Contains(model.Model, "-thinking-") {
								p = &adapters.ProviderAppend{Provider: p, Append: genai.Request{Text: "\n\n/think"}}
							}
							return &adapters.ProviderReasoning{
								Provider:            p,
								ReasoningTokenStart: sc.ReasoningTokenStart,
								ReasoningTokenEnd:   sc.ReasoningTokenEnd,
							}
						}
						break
					}
				}
			}
			return p
		}
		smoketest.Run(t, getClientRT, models, testRecorder.Records)
	})

	t.Run("Preferred", func(t *testing.T) {
		internaltest.TestPreferredModels(t, func(st *testing.T, model string, modality genai.Modality) (genai.Provider, error) {
			opts := []genai.ProviderOption{
				genai.ProviderOptionModalities{modality},
				genai.ProviderOptionPreloadedModels(cachedModels),
			}
			if model != "" {
				opts = append(opts, genai.ProviderOptionModel(model))
			}
			return getClientInner(st, func(h http.RoundTripper) http.RoundTripper {
				return testRecorder.Record(st, h)
			}, opts...)
		})
	})

	t.Run("TextOutputDocInput", func(t *testing.T) {
		internaltest.TestTextOutputDocInput(t, func(t *testing.T) genai.Provider {
			return getClient(t, string(genai.ModelCheap))
		})
	})

	t.Run("errors", func(t *testing.T) {
		data := []internaltest.ProviderError{
			{
				Name: "bad apiKey",
				Opts: []genai.ProviderOption{
					genai.ProviderOptionAPIKey("bad apiKey"),
					genai.ProviderOptionModel("llama-3.1-8b"),
				},
				ErrGenSync:   "http 401\ninvalid_request_error/api_key/wrong_api_key: Wrong API Key\nget a new API key at https://cloud.cerebras.ai/platform/",
				ErrGenStream: "http 401\ninvalid_request_error/api_key/wrong_api_key: Wrong API Key\nget a new API key at https://cloud.cerebras.ai/platform/",
				ErrListModel: "http 401\ninvalid_request_error/api_key/wrong_api_key: Wrong API Key\nget a new API key at https://cloud.cerebras.ai/platform/",
			},
			{
				Name: "bad model",
				Opts: []genai.ProviderOption{
					genai.ProviderOptionModel("bad model"),
				},
				ErrGenSync:   "http 404\nnot_found_error/model/model_not_found: Model bad model does not exist or you do not have access to it.",
				ErrGenStream: "http 404\nnot_found_error/model/model_not_found: Model bad model does not exist or you do not have access to it.",
			},
		}
		f := func(t *testing.T, opts ...genai.ProviderOption) (genai.Provider, error) {
			opts = append(opts, genai.ProviderOptionModalities{genai.ModalityText})
			return getClientInner(t, func(h http.RoundTripper) http.RoundTripper {
				return testRecorder.Record(t, h)
			}, opts...)
		}
		internaltest.TestClientProviderErrors(t, f, data)
	})
}

func init() {
	internal.BeLenient = false
}
