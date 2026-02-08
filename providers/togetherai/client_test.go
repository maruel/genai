// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package togetherai_test

import (
	"context"
	_ "embed"
	"net/http"
	"os"
	"slices"
	"strings"
	"testing"

	"github.com/maruel/genai"
	"github.com/maruel/genai/adapters"
	"github.com/maruel/genai/internal"
	"github.com/maruel/genai/internal/internaltest"
	"github.com/maruel/genai/providers/togetherai"
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
	if !hasAPIKey && os.Getenv("TOGETHER_API_KEY") == "" {
		opts = append(opts, genai.ProviderOptionAPIKey("<insert_api_key_here>"))
	}
	if fn != nil {
		opts = append([]genai.ProviderOption{genai.ProviderOptionTransportWrapper(fn)}, opts...)
	}
	return togetherai.New(t.Context(), opts...)
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
		models := make([]scoreboard.Model, 0, len(genaiModels))
		for _, m := range genaiModels {
			// TODO: This is weak.
			id := m.GetID()
			reason := (strings.Contains(id, "-Thinking") ||
				strings.HasPrefix(id, "openai/gpt") ||
				(strings.HasPrefix(id, "Qwen/Qwen3") && !strings.Contains(id, "Instruct")) ||
				strings.HasPrefix(id, "zai-org/GLM"))
			models = append(models, scoreboard.Model{Model: m.GetID(), Reason: reason})
		}
		getClientRT := func(t testing.TB, model scoreboard.Model, fn func(http.RoundTripper) http.RoundTripper) genai.Provider {
			provOpts := []genai.ProviderOption{
				genai.ProviderOptionPreloadedModels(cachedModels),
			}
			if model.Model != "" {
				provOpts = append(provOpts, genai.ProviderOptionModel(model.Model))
			}
			if os.Getenv("TOGETHER_API_KEY") == "" {
				provOpts = append(provOpts, genai.ProviderOptionAPIKey("<insert_api_key_here>"))
			}
			if fn != nil {
				provOpts = append([]genai.ProviderOption{genai.ProviderOptionTransportWrapper(fn)}, provOpts...)
			}
			c, err := togetherai.New(t.Context(), provOpts...)
			if err != nil {
				t.Fatal(err)
			}
			if model.Reason {
				for _, sc := range c.Scoreboard().Scenarios {
					if sc.ReasoningTokenEnd != "" && slices.Contains(sc.Models, model.Model) {
						return &adapters.ProviderReasoning{
							Provider:            c,
							ReasoningTokenStart: sc.ReasoningTokenStart,
							ReasoningTokenEnd:   sc.ReasoningTokenEnd,
						}
					}
				}
			}
			size := 256
			if strings.Contains(model.Model, "FLUX.2-dev") || strings.Contains(model.Model, "FLUX.2-max") {
				size = 512
			}
			// If anyone at Together.AI reads this, please get your shit together.
			return &smallImage{Provider: &internaltest.HideHTTPCode{Provider: c, StatusCode: 500}, size: size}
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
					genai.ProviderOptionModel("meta-llama/Llama-3.2-3B-Instruct-Turbo"),
				},
				ErrGenSync:   "http 401\ninvalid_api_key (invalid_request_error): Invalid API key provided. You can find your API key at https://api.together.ai/settings/api-keys.",
				ErrGenStream: "http 401\ninvalid_api_key (invalid_request_error): Invalid API key provided. You can find your API key at https://api.together.ai/settings/api-keys.",
				ErrListModel: "http 401\nUnauthorized\nget a new API key at https://api.together.ai/settings/api-keys",
			},
			{
				Name: "bad apiKey image",
				Opts: []genai.ProviderOption{
					genai.ProviderOptionAPIKey("bad apiKey"),
					genai.ProviderOptionModel("black-forest-labs/FLUX.1-schnell"),
					genai.ProviderOptionModalities{genai.ModalityImage},
				},
				ErrGenSync:   "http 401\ninvalid_api_key (invalid_request_error): Invalid API key provided. You can find your API key at https://api.together.ai/settings/api-keys.",
				ErrGenStream: "http 401\ninvalid_api_key (invalid_request_error): Invalid API key provided. You can find your API key at https://api.together.ai/settings/api-keys.",
			},
			{
				Name: "bad model",
				Opts: []genai.ProviderOption{
					genai.ProviderOptionModel("bad model"),
				},
				ErrGenSync:   "http 404\nmodel_not_available (invalid_request_error): Unable to access model bad model. Please visit https://api.together.ai/models to view the list of supported models.",
				ErrGenStream: "http 404\nmodel_not_available (invalid_request_error): Unable to access model bad model. Please visit https://api.together.ai/models to view the list of supported models.",
			},
			{
				Name: "bad model image",
				Opts: []genai.ProviderOption{
					genai.ProviderOptionModel("bad model"),
					genai.ProviderOptionModalities{genai.ModalityImage},
				},
				ErrGenSync:   "http 404\nmodel_not_available (invalid_request_error): Unable to access model bad model. Please visit https://api.together.ai/models to view the list of supported models.",
				ErrGenStream: "http 404\nmodel_not_available (invalid_request_error): Unable to access model bad model. Please visit https://api.together.ai/models to view the list of supported models.",
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

// smallImage speeds up image generation.
type smallImage struct {
	genai.Provider
	size int
}

func (h *smallImage) Unwrap() genai.Provider {
	return h.Provider
}

func (h *smallImage) GenSync(ctx context.Context, msgs genai.Messages, opts ...genai.GenOption) (genai.Result, error) {
	for i := range opts {
		if v, ok := opts[i].(*genai.GenOptionImage); ok {
			// Ask for a smaller size.
			n := *v
			n.Width = h.size
			n.Height = h.size
			opts[i] = &n
		}
	}
	return h.Provider.GenSync(ctx, msgs, opts...)
}

func init() {
	internal.BeLenient = false
}
