// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package togetherai_test

import (
	"context"
	_ "embed"
	"fmt"
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

func getClientInner(t *testing.T, opts genai.ProviderOptions, fn func(http.RoundTripper) http.RoundTripper) (genai.Provider, error) {
	if opts.APIKey == "" && os.Getenv("TOGETHER_API_KEY") == "" {
		opts.APIKey = "<insert_api_key_here>"
	}
	return togetherai.New(t.Context(), &opts, fn)
}

func TestClient(t *testing.T) {
	testRecorder := internaltest.NewRecords()
	t.Cleanup(func() {
		if err := testRecorder.Close(); err != nil {
			t.Error(err)
		}
	})
	cl, err2 := getClientInner(t, genai.ProviderOptions{Model: genai.ModelNone}, func(h http.RoundTripper) http.RoundTripper {
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
		opts := genai.ProviderOptions{Model: m, PreloadedModels: cachedModels}
		ci, err := getClientInner(t, opts, func(h http.RoundTripper) http.RoundTripper {
			return testRecorder.Record(t, h)
		})
		if err != nil {
			t.Fatal(err)
		}
		return ci
	}

	t.Run("Capabilities", func(t *testing.T) {
		internaltest.TestCapabilities(t, getClient(t, genai.ModelNone))
	})

	t.Run("Scoreboard", func(t *testing.T) {
		genaiModels, err := getClient(t, genai.ModelNone).ListModels(t.Context())
		if err != nil {
			t.Fatal(err)
		}
		var models []scoreboard.Model
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
			opts := genai.ProviderOptions{Model: model.Model, PreloadedModels: cachedModels}
			if os.Getenv("TOGETHER_API_KEY") == "" {
				opts.APIKey = "<insert_api_key_here>"
			}
			c, err := togetherai.New(t.Context(), &opts, fn)
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
		data := []struct {
			modality genai.Modality
			name     string
			want     string
		}{
			{genai.ModalityText, genai.ModelCheap, "openai/gpt-oss-20b"},
			{genai.ModalityText, genai.ModelGood, "Qwen/Qwen3-235B-A22B-fp8-tput"},
			{genai.ModalityText, genai.ModelSOTA, "Qwen/Qwen3-235B-A22B-Thinking-2507"},
			{genai.ModalityImage, genai.ModelCheap, "black-forest-labs/FLUX.1-schnell"},
			{genai.ModalityImage, genai.ModelGood, "black-forest-labs/FLUX.2-dev"},
			{genai.ModalityImage, genai.ModelSOTA, "black-forest-labs/FLUX.2-pro"},
		}
		for _, line := range data {
			t.Run(fmt.Sprintf("%s-%s", line.modality, line.name), func(t *testing.T) {
				opts := genai.ProviderOptions{
					Model:            line.name,
					OutputModalities: genai.Modalities{line.modality},
					PreloadedModels:  cachedModels,
				}
				c, err := getClientInner(t, opts, func(h http.RoundTripper) http.RoundTripper {
					return testRecorder.Record(t, h)
				})
				if err != nil {
					t.Fatal(err)
				}
				if got := c.ModelID(); got != line.want {
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
					Model:  "meta-llama/Llama-3.2-3B-Instruct-Turbo",
				},
				ErrGenSync:   "http 401\ninvalid_api_key (invalid_request_error): Invalid API key provided. You can find your API key at https://api.together.ai/settings/api-keys.",
				ErrGenStream: "http 401\ninvalid_api_key (invalid_request_error): Invalid API key provided. You can find your API key at https://api.together.ai/settings/api-keys.",
				ErrListModel: "http 401\nUnauthorized\nget a new API key at https://api.together.ai/settings/api-keys",
			},
			{
				Name: "bad apiKey image",
				Opts: genai.ProviderOptions{
					APIKey:           "bad apiKey",
					Model:            "black-forest-labs/FLUX.1-schnell",
					OutputModalities: genai.Modalities{genai.ModalityImage},
				},
				ErrGenSync:   "http 401\ninvalid_api_key (invalid_request_error): Invalid API key provided. You can find your API key at https://api.together.ai/settings/api-keys.",
				ErrGenStream: "http 401\ninvalid_api_key (invalid_request_error): Invalid API key provided. You can find your API key at https://api.together.ai/settings/api-keys.",
			},
			{
				Name: "bad model",
				Opts: genai.ProviderOptions{
					Model: "bad model",
				},
				ErrGenSync:   "http 404\nmodel_not_available (invalid_request_error): Unable to access model bad model. Please visit https://api.together.ai/models to view the list of supported models.",
				ErrGenStream: "http 404\nmodel_not_available (invalid_request_error): Unable to access model bad model. Please visit https://api.together.ai/models to view the list of supported models.",
			},
			{
				Name: "bad model image",
				Opts: genai.ProviderOptions{
					Model:            "bad model",
					OutputModalities: genai.Modalities{genai.ModalityImage},
				},
				ErrGenSync:   "http 404\nmodel_not_available (invalid_request_error): Unable to access model bad model. Please visit https://api.together.ai/models to view the list of supported models.",
				ErrGenStream: "http 404\nmodel_not_available (invalid_request_error): Unable to access model bad model. Please visit https://api.together.ai/models to view the list of supported models.",
			},
		}
		f := func(t *testing.T, opts genai.ProviderOptions) (genai.Provider, error) {
			opts.OutputModalities = genai.Modalities{genai.ModalityText}
			return getClientInner(t, opts, func(h http.RoundTripper) http.RoundTripper {
				return testRecorder.Record(t, h)
			})
		}
		internaltest.TestClient_Provider_errors(t, f, data)
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

func (h *smallImage) GenSync(ctx context.Context, msgs genai.Messages, opts ...genai.Options) (genai.Result, error) {
	for i := range opts {
		if v, ok := opts[i].(*genai.OptionsImage); ok {
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
