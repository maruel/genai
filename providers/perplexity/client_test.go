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
	"github.com/maruel/genai/smoke/smoketest"
	"github.com/maruel/roundtrippers"
)

func getClientInner(t *testing.T, opts genai.ProviderOptions, fn func(http.RoundTripper) http.RoundTripper) (genai.Provider, error) {
	if opts.APIKey == "" && os.Getenv("PERPLEXITY_API_KEY") == "" {
		opts.APIKey = "<insert_api_key_here>"
	}
	return perplexity.New(t.Context(), &opts, fn)
}

func TestClient(t *testing.T) {
	testRecorder := internaltest.NewRecords()
	t.Cleanup(func() {
		if err := testRecorder.Close(); err != nil {
			t.Error(err)
		}
	})
	getClient := func(t *testing.T, m string) genai.Provider {
		t.Parallel()
		opts := genai.ProviderOptions{Model: m}
		ci, err := getClientInner(t, opts, func(h http.RoundTripper) http.RoundTripper {
			return testRecorder.Record(t, h)
		})
		if err != nil {
			t.Fatal(err)
		}
		return ci
	}

	t.Run("Scoreboard", func(t *testing.T) {
		// Perplexity doesn't support listing models. See https://docs.perplexity.ai/api-reference
		sb := getClient(t, genai.ModelNone).Scoreboard()
		var models []smoketest.Model
		for _, sc := range sb.Scenarios {
			for _, model := range sc.Models {
				models = append(models, smoketest.Model{Model: model, Reason: sc.Reason})
			}
		}
		getClientRT := func(t testing.TB, model smoketest.Model, fn func(http.RoundTripper) http.RoundTripper) genai.Provider {
			opts := genai.ProviderOptions{Model: model.Model}
			if os.Getenv("PERPLEXITY_API_KEY") == "" {
				opts.APIKey = "<insert_api_key_here>"
			}
			c, err := perplexity.New(t.Context(), &opts, func(h http.RoundTripper) http.RoundTripper {
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
			})
			if err != nil {
				t.Fatal(err)
			}
			// Save on costs when running the smoke test.
			var p genai.Provider = &injectOptions{
				Provider: c,
				Opts: []genai.Options{
					&genai.OptionsTools{WebSearch: false},
					&perplexity.Options{DisableRelatedQuestions: true},
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
		data := []struct {
			name string
			want string
		}{
			{genai.ModelCheap, "sonar"},
			{genai.ModelGood, "sonar-pro"},
			{genai.ModelSOTA, "sonar-reasoning-pro"},
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
					Model:  "sonar",
				},
				// It returns an HTML page...
				ErrGenSync:   "http 401\nget a new API key at https://www.perplexity.ai/settings/api",
				ErrGenStream: "http 401\nget a new API key at https://www.perplexity.ai/settings/api",
			},
			{
				Name: "bad model",
				Opts: genai.ProviderOptions{
					Model: "bad model",
				},
				ErrGenSync:   "http 400\ninvalid_model (400): Invalid model 'bad model'. Permitted models can be found in the documentation at https://docs.perplexity.ai/guides/model-cards.",
				ErrGenStream: "http 400\ninvalid_model (400): Invalid model 'bad model'. Permitted models can be found in the documentation at https://docs.perplexity.ai/guides/model-cards.",
			},
		}
		f := func(t *testing.T, opts genai.ProviderOptions) (genai.Provider, error) {
			return getClientInner(t, opts, func(h http.RoundTripper) http.RoundTripper {
				return testRecorder.Record(t, h)
			})
		}
		internaltest.TestClient_Provider_errors(t, f, data)
	})
}

// injectOptions generally inject the option unless "Quackiland" is in the last message.
type injectOptions struct {
	genai.Provider
	Opts []genai.Options
}

func (i *injectOptions) Unwrap() genai.Provider {
	return i.Provider
}

func (i *injectOptions) GenSync(ctx context.Context, msgs genai.Messages, opts ...genai.Options) (genai.Result, error) {
	if !slices.ContainsFunc(opts, func(o genai.Options) bool {
		v, ok := o.(*genai.OptionsTools)
		return ok && v.WebSearch
	}) {
		opts = append(opts, i.Opts...)
	}
	return i.Provider.GenSync(ctx, msgs, opts...)
}

func (i *injectOptions) GenStream(ctx context.Context, msgs genai.Messages, opts ...genai.Options) (iter.Seq[genai.ReplyFragment], func() (genai.Result, error)) {
	if !slices.ContainsFunc(opts, func(o genai.Options) bool {
		v, ok := o.(*genai.OptionsTools)
		return ok && v.WebSearch
	}) {
		opts = append(opts, i.Opts...)
	}
	return i.Provider.GenStream(ctx, msgs, opts...)
}

func init() {
	internal.BeLenient = false
}
