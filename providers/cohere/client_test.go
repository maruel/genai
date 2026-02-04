// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package cohere_test

import (
	"net/http"
	"os"
	"slices"
	"strings"
	"testing"

	"github.com/maruel/genai"
	"github.com/maruel/genai/internal"
	"github.com/maruel/genai/internal/internaltest"
	"github.com/maruel/genai/providers/cohere"
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
	if !hasAPIKey && os.Getenv("COHERE_API_KEY") == "" {
		opts = append(opts, genai.ProviderOptionAPIKey("<insert_api_key_here>"))
	}
	if fn != nil {
		opts = append([]genai.ProviderOption{genai.ProviderOptionTransportWrapper(fn)}, opts...)
	}
	return cohere.New(t.Context(), opts...)
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
		genaiModels, err := getClient(t, "").ListModels(t.Context())
		if err != nil {
			t.Fatal(err)
		}
		var models []scoreboard.Model
		for _, m := range genaiModels {
			mdl := m.(*cohere.Model)
			if !slices.Contains(mdl.Endpoints, "chat") || strings.Contains(mdl.Name, "nightly") {
				continue
			}
			// All models "reason" when tool calling except light because it doesn't support tool calling.
			if !strings.Contains(mdl.Name, "light") {
				models = append(models, scoreboard.Model{Model: mdl.Name, Reason: true})
			} else {
				models = append(models, scoreboard.Model{Model: mdl.Name})
			}
		}
		getClientRT := func(t testing.TB, model scoreboard.Model, fn func(http.RoundTripper) http.RoundTripper) genai.Provider {
			provOpts := []genai.ProviderOption{
				genai.ProviderOptionPreloadedModels(cachedModels),
			}
			if model.Model != "" {
				provOpts = append(provOpts, genai.ProviderOptionModel(model.Model))
			}
			if os.Getenv("COHERE_API_KEY") == "" {
				provOpts = append(provOpts, genai.ProviderOptionAPIKey("<insert_api_key_here>"))
			}
			if fn != nil {
				provOpts = append([]genai.ProviderOption{genai.ProviderOptionTransportWrapper(fn)}, provOpts...)
			}
			c, err := cohere.New(t.Context(), provOpts...)
			if err != nil {
				t.Fatal(err)
			}
			return c
		}
		smoketest.Run(t, getClientRT, models, testRecorder.Records)
	})

	t.Run("Preferred", func(t *testing.T) {
		internaltest.TestPreferredModels(t, func(st *testing.T, model string, modality genai.Modality) (genai.Provider, error) {
			opts := []genai.ProviderOption{
				genai.ProviderOptionModalities(genai.Modalities{modality}),
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
			return getClient(t, genai.ModelCheap)
		})
	})

	t.Run("errors", func(t *testing.T) {
		data := []internaltest.ProviderError{
			{
				Name: "bad apiKey",
				Opts: []genai.ProviderOption{
					genai.ProviderOptionAPIKey("bad apiKey"),
					genai.ProviderOptionModel("command-r7b-12-2024"),
				},
				ErrGenSync:   "http 401\ninvalid api token\nget a new API key at https://dashboard.cohere.com/api-keys",
				ErrGenStream: "http 401\ninvalid api token\nget a new API key at https://dashboard.cohere.com/api-keys",
				ErrListModel: "http 401\ninvalid api token\nget a new API key at https://dashboard.cohere.com/api-keys",
			},
			{
				Name: "bad model",
				Opts: []genai.ProviderOption{
					genai.ProviderOptionModel("bad model"),
				},
				ErrGenSync:   "http 404\nmodel 'bad model' not found, make sure the correct model ID was used and that you have access to the model.",
				ErrGenStream: "http 404\nmodel 'bad model' not found, make sure the correct model ID was used and that you have access to the model.",
			},
		}
		f := func(t *testing.T, opts ...genai.ProviderOption) (genai.Provider, error) {
			opts = append(opts, genai.ProviderOptionModalities(genai.Modalities{genai.ModalityText}))
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
