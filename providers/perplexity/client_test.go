// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package perplexity_test

import (
	"context"
	"iter"
	"net/http"
	"os"
	"slices"
	"strings"
	"testing"

	"github.com/maruel/genai"
	"github.com/maruel/genai/adapters"
	"github.com/maruel/genai/internal"
	"github.com/maruel/genai/internal/internaltest"
	"github.com/maruel/genai/providers/perplexity"
	"github.com/maruel/genai/scoreboard"
	"github.com/maruel/genai/smoke/smoketest"
	"github.com/maruel/roundtrippers"
)

func getClientInner(t *testing.T, apiKey string, opts []genai.ProviderOption, fn func(http.RoundTripper) http.RoundTripper) (genai.Provider, error) {
	if apiKey == "" && os.Getenv("PERPLEXITY_API_KEY") == "" {
		apiKey = "<insert_api_key_here>"
	}
	if apiKey != "" {
		opts = append(opts, genai.ProviderOptionAPIKey(apiKey))
	}
	if fn != nil {
		opts = append(opts, genai.ProviderOptionTransportWrapper(fn))
	}
	return perplexity.New(t.Context(), opts...)
}

func TestClient(t *testing.T) {
	testRecorder := internaltest.NewRecords()
	t.Cleanup(func() {
		if err := testRecorder.Close(); err != nil {
			t.Error(err)
		}
	})
	// Perplexity doesn't support listing models. See https://docs.perplexity.ai/api-reference
	getClient := func(t *testing.T, m string) genai.Provider {
		t.Parallel()
		var opts []genai.ProviderOption
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
		// Perplexity doesn't support listing models. See https://docs.perplexity.ai/api-reference
		sb := getClient(t, "").Scoreboard()
		var models []scoreboard.Model
		for _, sc := range sb.Scenarios {
			for _, model := range sc.Models {
				models = append(models, scoreboard.Model{Model: model, Reason: sc.Reason})
			}
		}
		getClientRT := func(t testing.TB, model scoreboard.Model, fn func(http.RoundTripper) http.RoundTripper) genai.Provider {
			var o []genai.ProviderOption
			if model.Model != "" {
				o = append(o, genai.ProviderOptionModel(model.Model))
			}
			if os.Getenv("PERPLEXITY_API_KEY") == "" {
				o = append(o, genai.ProviderOptionAPIKey("<insert_api_key_here>"))
			}
			c, err := perplexity.New(t.Context(), append([]genai.ProviderOption{genai.ProviderOptionTransportWrapper(func(h http.RoundTripper) http.RoundTripper {
				// Perplexity is quick to ban users. It first start with 429 and then Cloudflare blocks it with a
				// javascript challenge. It's extra dumb because it is an API endpoint.
				// https://docs.perplexity.ai/guides/usage-tiers
				qps := 48. / 60.
				if strings.Contains(model.Model, "deep") {
					// Assume Tier 0 with is 5 RPM. For this test to succeed, it must be run with "go test -timeout 1h"
					qps = 4.8 / 60.
				}
				h = &roundtrippers.Throttle{QPS: qps, Transport: h}
				if fn != nil {
					h = fn(h)
				}
				return h
			})}, o...)...)
			if err != nil {
				t.Fatal(err)
			}
			// Save on costs when running the smoke test.
			var p genai.Provider = &injectOptions{
				Provider: c,
				Opts: []genai.GenOptions{
					&genai.GenOptionsTools{WebSearch: false},
					&perplexity.GenOptions{DisableRelatedQuestions: true},
				},
			}
			if model.Reason {
				for _, sc := range c.Scoreboard().Scenarios {
					if sc.Reason && slices.Contains(sc.Models, model.Model) {
						if sc.ReasoningTokenStart != "" && sc.ReasoningTokenEnd != "" {
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
				genai.ProviderOptionModalities(genai.Modalities{modality}),
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
					genai.ProviderOptionModel("sonar"),
				},
				// It returns an HTML page...
				ErrGenSync:   "http 401\nget a new API key at https://www.perplexity.ai/settings/api",
				ErrGenStream: "http 401\nget a new API key at https://www.perplexity.ai/settings/api",
			},
			{
				Name: "bad model",
				Opts: []genai.ProviderOption{
					genai.ProviderOptionModel("bad model"),
				},
				ErrGenSync:   "http 400\ninvalid_model (400): Invalid model 'bad model'. Permitted models can be found in the documentation at https://docs.perplexity.ai/guides/model-cards.",
				ErrGenStream: "http 400\ninvalid_model (400): Invalid model 'bad model'. Permitted models can be found in the documentation at https://docs.perplexity.ai/guides/model-cards.",
			},
		}
		f := func(t *testing.T, opts ...genai.ProviderOption) (genai.Provider, error) {
			return getClientInner(t, "", opts, func(h http.RoundTripper) http.RoundTripper {
				return testRecorder.Record(t, h)
			})
		}
		internaltest.TestClientProviderErrors(t, f, data)
	})
}

// injectOptions generally inject the option unless "Quackiland" is in the last message.
type injectOptions struct {
	genai.Provider
	Opts []genai.GenOptions
}

func (i *injectOptions) Unwrap() genai.Provider {
	return i.Provider
}

func (i *injectOptions) GenSync(ctx context.Context, msgs genai.Messages, opts ...genai.GenOptions) (genai.Result, error) {
	if !slices.ContainsFunc(opts, func(o genai.GenOptions) bool {
		v, ok := o.(*genai.GenOptionsTools)
		return ok && v.WebSearch
	}) {
		opts = append(opts, i.Opts...)
	}
	return i.Provider.GenSync(ctx, msgs, opts...)
}

func (i *injectOptions) GenStream(ctx context.Context, msgs genai.Messages, opts ...genai.GenOptions) (iter.Seq[genai.Reply], func() (genai.Result, error)) {
	if !slices.ContainsFunc(opts, func(o genai.GenOptions) bool {
		v, ok := o.(*genai.GenOptionsTools)
		return ok && v.WebSearch
	}) {
		opts = append(opts, i.Opts...)
	}
	return i.Provider.GenStream(ctx, msgs, opts...)
}

func init() {
	internal.BeLenient = false
}
