// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package mistral_test

import (
	_ "embed"
	"net/http"
	"os"
	"strings"
	"testing"

	"github.com/maruel/genai"
	"github.com/maruel/genai/internal"
	"github.com/maruel/genai/internal/internaltest"
	"github.com/maruel/genai/providers/mistral"
	"github.com/maruel/genai/scoreboard"
	"github.com/maruel/genai/smoke/smoketest"
)

func getClientInner(t *testing.T, apiKey string, opts []genai.ProviderOption, fn func(http.RoundTripper) http.RoundTripper) (genai.Provider, error) {
	if apiKey == "" && os.Getenv("MISTRAL_API_KEY") == "" {
		apiKey = "<insert_api_key_here>"
	}
	if apiKey != "" {
		opts = append(opts, genai.ProviderOptionAPIKey(apiKey))
	}
	if fn != nil {
		opts = append(opts, genai.ProviderOptionTransportWrapper(fn))
	}
	return mistral.New(t.Context(), opts...)
}

func TestClient(t *testing.T) {
	testRecorder := internaltest.NewRecords()
	t.Cleanup(func() {
		if err := testRecorder.Close(); err != nil {
			t.Error(err)
		}
	})
	cl, err2 := getClientInner(t, "", nil, func(h http.RoundTripper) http.RoundTripper {
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
		ci, err := getClientInner(t, "", opts, func(h http.RoundTripper) http.RoundTripper {
			return testRecorder.Record(t, h)
		})
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
			models = append(models, scoreboard.Model{Model: m.GetID()})
		}
		getClientRT := func(t testing.TB, model scoreboard.Model, fn func(http.RoundTripper) http.RoundTripper) genai.Provider {
			opts := []genai.ProviderOption{genai.ProviderOptionPreloadedModels(cachedModels)}
			if model.Model != "" {
				opts = append(opts, genai.ProviderOptionModel(model.Model))
			}
			if os.Getenv("MISTRAL_API_KEY") == "" {
				opts = append(opts, genai.ProviderOptionAPIKey("<insert_api_key_here>"))
			}
			if fn != nil {
				opts = append(opts, genai.ProviderOptionTransportWrapper(fn))
			}
			c, err := mistral.New(t.Context(), opts...)
			if err != nil {
				t.Fatal(err)
			}
			if model.Reason {
				t.Fatal("implement me")
			}
			if strings.HasPrefix(model.Model, "voxtral") {
				// If anyone at Mistral reads this, please get your shit together.
				return &internaltest.HideHTTPCode{Provider: c, StatusCode: 500}
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
			return getClientInner(st, "", opts, func(h http.RoundTripper) http.RoundTripper {
				return testRecorder.Record(st, h)
			})
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
					genai.ProviderOptionModel("ministral-3b-latest"),
				},
				ErrGenSync:   "http 401\nUnauthorized\nget a new API key at https://console.mistral.ai/api-keys",
				ErrGenStream: "http 401\nUnauthorized\nget a new API key at https://console.mistral.ai/api-keys",
				ErrListModel: "http 401\nUnauthorized\nget a new API key at https://console.mistral.ai/api-keys",
			},
			{
				Name: "bad model",
				Opts: []genai.ProviderOption{
					genai.ProviderOptionModel("bad model"),
				},
				ErrGenSync:   "http 400\ninvalid_model: Invalid model: bad model",
				ErrGenStream: "http 400\ninvalid_model: Invalid model: bad model",
			},
		}
		f := func(t *testing.T, opts ...genai.ProviderOption) (genai.Provider, error) {
			opts = append(opts, genai.ProviderOptionModalities(genai.Modalities{genai.ModalityText}))
			return getClientInner(t, "", opts, func(h http.RoundTripper) http.RoundTripper {
				return testRecorder.Record(t, h)
			})
		}
		internaltest.TestClientProviderErrors(t, f, data)
	})
}

func init() {
	internal.BeLenient = false
}
