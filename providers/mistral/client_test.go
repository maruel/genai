// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package mistral_test

import (
	_ "embed"
	"net/http"
	"os"
	"strings"
	"sync"
	"testing"

	"github.com/maruel/genai"
	"github.com/maruel/genai/internal"
	"github.com/maruel/genai/internal/internaltest"
	"github.com/maruel/genai/internal/myrecorder"
	"github.com/maruel/genai/providers/mistral"
	"github.com/maruel/genai/smoke/smoketest"
)

func TestClient(t *testing.T) {
	t.Run("Scoreboard", func(t *testing.T) {
		genaiModels, err := getClient(t, genai.ModelNone).ListModels(t.Context())
		if err != nil {
			t.Fatal(err)
		}
		var models []smoketest.Model
		for _, m := range genaiModels {
			models = append(models, smoketest.Model{Model: m.GetID()})
		}
		smoketest.Run(t, getClientRT, models, testRecorder.Records)
	})

	t.Run("Preferred", func(t *testing.T) {
		data := []struct {
			name string
			want string
		}{
			{genai.ModelCheap, "mistral-tiny-latest"},
			{genai.ModelGood, "mistral-medium-latest"},
			{genai.ModelSOTA, "mistral-large-latest"},
		}
		for _, line := range data {
			t.Run(line.name, func(t *testing.T) {
				if got := getClient(t, line.name).ModelID(); got != line.want {
					t.Fatalf("got model %q, want %q", got, line.want)
				}
			})
		}
	})

	t.Run("errors", func(t *testing.T) {
		data := []internaltest.ProviderError{
			{
				Name: "bad apiKey",
				Opts: genai.ProviderOptions{
					APIKey: "bad apiKey",
					Model:  "ministral-3b-latest",
				},
				ErrGenSync:   "http 401\nUnauthorized\nget a new API key at https://console.mistral.ai/api-keys",
				ErrGenStream: "http 401\nUnauthorized\nget a new API key at https://console.mistral.ai/api-keys",
				ErrListModel: "http 401\nUnauthorized\nget a new API key at https://console.mistral.ai/api-keys",
			},
			{
				Name: "bad model",
				Opts: genai.ProviderOptions{
					Model: "bad model",
				},
				ErrGenSync:   "http 400\ninvalid_model: Invalid model: bad model",
				ErrGenStream: "http 400\ninvalid_model: Invalid model: bad model",
			},
		}
		f := func(t *testing.T, opts genai.ProviderOptions) (genai.Provider, error) {
			opts.OutputModalities = genai.Modalities{genai.ModalityText}
			return getClientInner(t, opts)
		}
		internaltest.TestClient_Provider_errors(t, f, data)
	})
}

func getClientRT(t testing.TB, model smoketest.Model, fn func(http.RoundTripper) http.RoundTripper) genai.Provider {
	apiKey := ""
	if os.Getenv("MISTRAL_API_KEY") == "" {
		apiKey = "<insert_api_key_here>"
	}
	opts := genai.ProviderOptions{
		APIKey:          apiKey,
		Model:           model.Model,
		PreloadedModels: loadCachedModelsList(t),
	}
	c, err := mistral.New(t.Context(), &opts, fn)
	if err != nil {
		t.Fatal(err)
	}
	if model.Reason {
		t.Fatal("implement me")
	}
	if strings.HasPrefix(model.Model, "voxtral") {
		// If anyone at Mistral reads this, please get your shit together.
		return &internaltest.HideHTTP500{Provider: c}
	}
	return c
}

func getClient(t *testing.T, m string) *mistral.Client {
	t.Parallel()
	opts := genai.ProviderOptions{
		Model:           m,
		PreloadedModels: loadCachedModelsList(t),
	}
	c, err := getClientInner(t, opts)
	if err != nil {
		t.Fatal(err)
	}
	return c
}

func getClientInner(t *testing.T, opts genai.ProviderOptions) (*mistral.Client, error) {
	if opts.APIKey == "" && os.Getenv("MISTRAL_API_KEY") == "" {
		opts.APIKey = "<insert_api_key_here>"
	}
	return mistral.New(t.Context(), &opts, func(h http.RoundTripper) http.RoundTripper { return testRecorder.Record(t, h) })
}

func loadCachedModelsList(t testing.TB) []genai.Model {
	doOnce.Do(func() {
		var r myrecorder.Recorder
		var err2 error
		ctx := t.Context()
		opts := genai.ProviderOptions{Model: genai.ModelNone}
		if os.Getenv("MISTRAL_API_KEY") == "" {
			opts.APIKey = "<insert_api_key_here>"
		}
		c, err := mistral.New(ctx, &opts, func(h http.RoundTripper) http.RoundTripper {
			r, err2 = testRecorder.Records.Record("WarmupCache", h)
			return r
		})
		if err != nil {
			t.Fatal(err)
		}
		if err2 != nil {
			t.Fatal(err2)
		}
		if cachedModels, err = c.ListModels(ctx); err != nil {
			t.Fatal(err)
		}
		if err = r.Stop(); err != nil {
			t.Fatal(err)
		}
	})
	return cachedModels
}

var doOnce sync.Once

var cachedModels []genai.Model

var testRecorder *internaltest.Records

func TestMain(m *testing.M) {
	testRecorder = internaltest.NewRecords()
	code := m.Run()
	os.Exit(max(code, testRecorder.Close()))
}

func init() {
	internal.BeLenient = false
}
