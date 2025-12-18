// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package huggingface_test

import (
	"net/http"
	"slices"
	"strings"
	"testing"

	"github.com/maruel/genai"
	"github.com/maruel/genai/adapters"
	"github.com/maruel/genai/internal"
	"github.com/maruel/genai/internal/internaltest"
	"github.com/maruel/genai/providers/huggingface"
	"github.com/maruel/genai/scoreboard"
	"github.com/maruel/genai/smoke/smoketest"
)

func getClientInner(t *testing.T, opts genai.ProviderOptions, fn func(http.RoundTripper) http.RoundTripper) (genai.Provider, error) {
	if opts.APIKey == "" {
		opts.APIKey = getAPIKeyTest(t)
	}
	return huggingface.New(t.Context(), &opts, fn)
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
		// We do not want to test thousands of models, so get the ones already in the scoreboard.
		sb := getClient(t, genai.ModelNone).Scoreboard()
		var models []scoreboard.Model
		for _, sc := range sb.Scenarios {
			for _, model := range sc.Models {
				models = append(models, scoreboard.Model{Model: model, Reason: sc.Reason})
			}
		}
		getClientRT := func(t testing.TB, model scoreboard.Model, fn func(http.RoundTripper) http.RoundTripper) genai.Provider {
			opts := genai.ProviderOptions{APIKey: getAPIKeyTest(t), Model: model.Model, PreloadedModels: cachedModels}
			c, err := huggingface.New(t.Context(), &opts, fn)
			if err != nil {
				t.Fatal(err)
			}
			if strings.HasPrefix(model.Model, "Qwen/Qwen3") {
				if !model.Reason {
					return &adapters.ProviderAppend{Provider: c, Append: genai.Request{Text: "\n\n/no_think"}}
				}
				// Check if it has predefined reasoning tokens.
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
			if strings.Contains(model.Model, "-1B-") {
				// It returns 502 on JSON
				return &internaltest.HideHTTPCode{Provider: c, StatusCode: 502}
			}
			return c
		}
		smoketest.Run(t, getClientRT, models, testRecorder.Records)
	})

	t.Run("Preferred", func(t *testing.T) {
		internaltest.TestPreferredModels(t, func(st *testing.T, model string, modality genai.Modality) (genai.Provider, error) {
			opts := genai.ProviderOptions{
				Model:            model,
				OutputModalities: genai.Modalities{modality},
				PreloadedModels:  cachedModels,
			}
			return getClientInner(st, opts, func(h http.RoundTripper) http.RoundTripper {
				return testRecorder.Record(st, h)
			})
		})
	})

	t.Run("errors", func(t *testing.T) {
		data := []internaltest.ProviderError{
			{
				Name: "bad apiKey",
				Opts: genai.ProviderOptions{
					APIKey: "bad apiKey",
					Model:  "Qwen/Qwen3-4B",
				},
				ErrGenSync:   "http 401\nInvalid credentials in Authorization header\nget a new API key at https://huggingface.co/settings/tokens",
				ErrGenStream: "http 401\nInvalid credentials in Authorization header\nget a new API key at https://huggingface.co/settings/tokens",
				ErrListModel: "http 401\nInvalid credentials in Authorization header\nget a new API key at https://huggingface.co/settings/tokens",
			},
			{
				Name: "bad model",
				Opts: genai.ProviderOptions{
					Model: "bad model",
				},
				ErrGenSync:   "http 400\ninvalid_request_error (model_not_found): model: The requested model 'bad model' does not exist.",
				ErrGenStream: "http 400\ninvalid_request_error (model_not_found): model: The requested model 'bad model' does not exist.",
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

func getAPIKeyTest(t testing.TB) string {
	apiKey, err := getAPIKey()
	if err != nil {
		t.Fatal(err)
	}
	return apiKey
}

func init() {
	internal.BeLenient = false
}
