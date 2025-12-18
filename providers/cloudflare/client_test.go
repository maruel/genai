// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package cloudflare_test

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
	"github.com/maruel/genai/providers/cloudflare"
	"github.com/maruel/genai/scoreboard"
	"github.com/maruel/genai/smoke/smoketest"
)

func getClientInner(t *testing.T, opts genai.ProviderOptions, fn func(http.RoundTripper) http.RoundTripper) (genai.Provider, error) {
	if opts.APIKey == "" && os.Getenv("CLOUDFLARE_API_KEY") == "" {
		opts.APIKey = "<insert_api_key_here>"
	}
	if opts.AccountID == "" && os.Getenv("CLOUDFLARE_ACCOUNT_ID") == "" {
		opts.AccountID = "ACCOUNT_ID"
	}
	return cloudflare.New(t.Context(), &opts, fn)
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
			id := m.GetID()
			models = append(models, scoreboard.Model{Model: id, Reason: strings.Contains(id, "deepseek-r1")})
		}
		getClientRT := func(t testing.TB, model scoreboard.Model, fn func(http.RoundTripper) http.RoundTripper) genai.Provider {
			opts := genai.ProviderOptions{Model: model.Model, PreloadedModels: cachedModels}
			if os.Getenv("CLOUDFLARE_API_KEY") == "" {
				opts.APIKey = "<insert_api_key_here>"
			}
			if os.Getenv("CLOUDFLARE_ACCOUNT_ID") == "" {
				opts.AccountID = "ACCOUNT_ID"
			}
			c, err := cloudflare.New(t.Context(), &opts, fn)
			if err != nil {
				t.Fatal(err)
			}
			if model.Reason {
				// Check if it has predefined thinking tokens.
				for _, sc := range c.Scoreboard().Scenarios {
					if sc.Reason && slices.Contains(sc.Models, model.Model) {
						if sc.ReasoningTokenEnd != "" {
							return &adapters.ProviderReasoning{
								Provider:            c,
								ReasoningTokenStart: sc.ReasoningTokenStart,
								ReasoningTokenEnd:   sc.ReasoningTokenEnd,
							}
						}
						break
					}
				}
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
					Model:  "@hf/nousresearch/hermes-2-pro-mistral-7b",
				},
				ErrGenSync:   "http 401\nAuthentication error\nget a new API key at https://dash.cloudflare.com/profile/api-tokens",
				ErrGenStream: "http 401\nAuthentication error\nget a new API key at https://dash.cloudflare.com/profile/api-tokens",
				ErrListModel: "http 400\nUnable to authenticate request",
			},
			{
				Name: "bad model",
				Opts: genai.ProviderOptions{
					Model: "bad model",
				},
				ErrGenSync:   "http 400\nNo route for that URI",
				ErrGenStream: "http 400\nNo route for that URI",
			},
		}
		f := func(t *testing.T, opts genai.ProviderOptions) (genai.Provider, error) {
			opts.OutputModalities = genai.Modalities{genai.ModalityText}
			return getClientInner(t, opts, func(h http.RoundTripper) http.RoundTripper {
				return testRecorder.Record(t, h)
			})
		}
		internaltest.TestClientProviderErrors(t, f, data)
	})
}

func init() {
	internal.BeLenient = false
}
