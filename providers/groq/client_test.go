// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Tests for the Groq provider client.

package groq_test

import (
	"context"
	_ "embed"
	"iter"
	"net/http"
	"slices"
	"strings"
	"testing"

	"github.com/maruel/genai"
	"github.com/maruel/genai/internal"
	"github.com/maruel/genai/internal/internaltest"
	"github.com/maruel/genai/providers/groq"
	"github.com/maruel/genai/scoreboard"
	"github.com/maruel/genai/smoke/smoketest"
)

func getClientInner(t *testing.T, opts []genai.ProviderOption, fn func(http.RoundTripper) http.RoundTripper) (genai.Provider, error) {
	if !slices.ContainsFunc(opts, func(o genai.ProviderOption) bool { _, ok := o.(genai.ProviderOptionAPIKey); return ok }) {
		key := internaltest.GetEnv("GROQ_API_KEY")
		if key == "" {
			key = "<insert_api_key_here>"
		}
		opts = append(opts, genai.ProviderOptionAPIKey(key))
	}
	if fn != nil {
		opts = append(opts, genai.ProviderOptionTransportWrapper(fn))
	}
	c, err := groq.New(t.Context(), opts...)
	if c != nil {
		internaltest.CleanupCloser(t, c)
	}
	return c, err
}

func TestNew(t *testing.T) {
	t.Run("SOTA", func(t *testing.T) {
		t.Run("error", func(t *testing.T) {
			cl, err := groq.New(t.Context(), genai.ProviderOptionAPIKey("test-key"), genai.ModelSOTA,
				genai.ProviderOptionPreloadedModels{&groq.Model{ID: "openai/gpt-oss-120b"}},
			)
			if cl != nil {
				internaltest.CleanupCloser(t, cl)
			}
			if err == nil || !strings.Contains(err.Error(), "failed to find a model automatically") {
				t.Fatalf("unexpected error: %v", err)
			}
		})
	})
}

func TestGenOption(t *testing.T) {
	t.Run("Validate", func(t *testing.T) {
		t.Run("valid", func(t *testing.T) {
			t.Run("ReasoningFormat", func(t *testing.T) {
				for _, v := range []groq.ReasoningFormat{"", "hidden", "parsed", "raw"} {
					t.Run(string(v), func(t *testing.T) {
						o := groq.GenOption{ReasoningFormat: v}
						if err := o.Validate(); err != nil {
							t.Fatal(err)
						}
					})
				}
			})
			t.Run("ServiceTier", func(t *testing.T) {
				for _, v := range []groq.ServiceTier{"", "auto", "flex", "on_demand", "performance"} {
					t.Run(string(v), func(t *testing.T) {
						o := groq.GenOption{ServiceTier: v}
						if err := o.Validate(); err != nil {
							t.Fatal(err)
						}
					})
				}
			})
		})
		t.Run("error", func(t *testing.T) {
			for _, tc := range []struct {
				name string
				opt  groq.GenOption
				want string
			}{
				{name: "ReasoningEffort", opt: groq.GenOption{ReasoningEffort: "invalid"}, want: "invalid reasoning effort"},
				{name: "ReasoningFormat", opt: groq.GenOption{ReasoningFormat: "invalid"}, want: "invalid reasoning format"},
				{name: "ServiceTier", opt: groq.GenOption{ServiceTier: "invalid"}, want: "invalid service tier"},
			} {
				t.Run(tc.name, func(t *testing.T) {
					if err := tc.opt.Validate(); err == nil || !strings.Contains(err.Error(), tc.want) {
						t.Fatalf("unexpected error: %v", err)
					}
				})
			}
		})
	})
}

func TestClient(t *testing.T) {
	testRecorder := internaltest.NewRecords()
	t.Cleanup(func() {
		if err := testRecorder.Close(); err != nil {
			t.Error(err)
		}
	})
	cl, err2 := getClientInner(t, nil, func(h http.RoundTripper) http.RoundTripper {
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
		ci, err := getClientInner(t, opts, func(h http.RoundTripper) http.RoundTripper {
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
		c := getClient(t, "")
		genaiModels, err := c.ListModels(t.Context())
		if err != nil {
			t.Fatal(err)
		}
		scenarios := c.Scoreboard().Scenarios
		models := make([]scoreboard.Model, 0, len(genaiModels))
		for _, m := range genaiModels {
			id := m.GetID()
			found := false
			for _, sc := range scenarios {
				if slices.Contains(sc.Models, id) {
					models = append(models, scoreboard.Model{Model: id, Reason: sc.Reason})
					found = true
				}
			}
			if !found {
				models = append(models, scoreboard.Model{Model: id})
			}
		}
		getClientRT := func(t testing.TB, model scoreboard.Model, fn func(http.RoundTripper) http.RoundTripper) genai.Provider {
			opts := []genai.ProviderOption{genai.ProviderOptionPreloadedModels(cachedModels)}
			if model.Model != "" {
				opts = append(opts, genai.ProviderOptionModel(model.Model))
			}
			key := internaltest.GetEnv("GROQ_API_KEY")
			if key == "" {
				key = "<insert_api_key_here>"
			}
			opts = append(opts, genai.ProviderOptionAPIKey(key))
			if fn != nil {
				opts = append(opts, genai.ProviderOptionTransportWrapper(fn))
			}
			cl, err := groq.New(t.Context(), opts...)
			if cl != nil {
				internaltest.CleanupCloser(t, cl)
			}
			if err != nil {
				t.Fatal(err)
			}
			if strings.HasPrefix(model.Model, "qwen/") {
				o := groq.GenOption{}
				if model.Reason {
					o.ReasoningEffort = groq.ReasoningEffortMedium
					o.ReasoningFormat = groq.ReasoningFormatParsed
				}
				return &handleGroqReasoning{Provider: cl, opt: o}
			}
			return cl
		}
		smoketest.Run(t, getClientRT, models, testRecorder.Records, nil)
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
			return getClientInner(st, opts, func(h http.RoundTripper) http.RoundTripper {
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
					genai.ProviderOptionModel("llama-3.1-8b-instant"),
				},
				ErrGenSync:   "http 401\ninvalid_api_key (invalid_request_error): Invalid API Key\nget a new API key at https://console.groq.com/keys",
				ErrGenStream: "http 401\ninvalid_api_key (invalid_request_error): Invalid API Key\nget a new API key at https://console.groq.com/keys",
				ErrListModel: "http 401\ninvalid_api_key (invalid_request_error): Invalid API Key\nget a new API key at https://console.groq.com/keys",
			},
			{
				Name: "bad model",
				Opts: []genai.ProviderOption{
					genai.ProviderOptionModel("bad model"),
				},
				ErrGenSync:   "http 404\nmodel_not_found (invalid_request_error): The model `bad model` does not exist or you do not have access to it.",
				ErrGenStream: "http 404\nmodel_not_found (invalid_request_error): The model `bad model` does not exist or you do not have access to it.",
			},
		}
		f := func(t *testing.T, opts ...genai.ProviderOption) (genai.Provider, error) {
			opts = append(opts, genai.ProviderOptionModalities{genai.ModalityText})
			return getClientInner(t, opts, func(h http.RoundTripper) http.RoundTripper {
				return testRecorder.Record(t, h)
			})
		}
		internaltest.TestClientProviderErrors(t, f, data)
	})
}

type handleGroqReasoning struct {
	genai.Provider
	opt groq.GenOption
}

func (h *handleGroqReasoning) GenSync(ctx context.Context, msgs genai.Messages, opts ...genai.GenOption) (genai.Result, error) {
	opts = append(opts, &h.opt)
	return h.Provider.GenSync(ctx, msgs, opts...)
}

func (h *handleGroqReasoning) GenStream(ctx context.Context, msgs genai.Messages, opts ...genai.GenOption) (iter.Seq[genai.Reply], func() (genai.Result, error)) {
	opts = append(opts, &h.opt)
	return h.Provider.GenStream(ctx, msgs, opts...)
}

func (h *handleGroqReasoning) Unwrap() genai.Provider {
	return h.Provider
}

func init() {
	internal.BeLenient = false
}
