// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package groq_test

import (
	"context"
	_ "embed"
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
	"github.com/maruel/genai/providers/groq"
	"github.com/maruel/genai/scoreboard"
	"github.com/maruel/genai/smoke/smoketest"
)

func getClientInner(t *testing.T, opts genai.ProviderOptions, fn func(http.RoundTripper) http.RoundTripper) (genai.Provider, error) {
	if opts.APIKey == "" && os.Getenv("GROQ_API_KEY") == "" {
		opts.APIKey = "<insert_api_key_here>"
	}
	return groq.New(t.Context(), &opts, fn)
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
		c := getClient(t, genai.ModelNone)
		genaiModels, err := c.ListModels(t.Context())
		if err != nil {
			t.Fatal(err)
		}
		scenarios := c.Scoreboard().Scenarios
		var models []scoreboard.Model
		for _, m := range genaiModels {
			id := m.GetID()
			reason := false
			for _, sc := range scenarios {
				if slices.Contains(sc.Models, id) {
					reason = sc.Reason
					break
				}
			}
			models = append(models, scoreboard.Model{Model: id, Reason: reason})
		}
		getClientRT := func(t testing.TB, model scoreboard.Model, fn func(http.RoundTripper) http.RoundTripper) genai.Provider {
			opts := genai.ProviderOptions{Model: model.Model, PreloadedModels: cachedModels}
			if os.Getenv("GROQ_API_KEY") == "" {
				opts.APIKey = "<insert_api_key_here>"
			}
			cl, err := groq.New(t.Context(), &opts, fn)
			if err != nil {
				t.Fatal(err)
			}
			var c genai.Provider = cl
			if strings.HasPrefix(model.Model, "qwen/") && model.Reason {
				c = &adapters.ProviderAppend{Provider: c, Append: genai.Request{Text: "\n\n/think"}}
			}
			// Groq models with native reasoning support (groq/compound, etc.) already return reasoning in a
			// separate field. Don't wrap with ProviderReasoning which expects reasoning embedded in text.
			// Only apply ProviderReasoning to models that need text-based reasoning extraction.
			if model.Reason && strings.HasPrefix(model.Model, "qwen/") {
				return &handleGroqReasoning{Provider: c}
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

	t.Run("TextOutputDocInput", func(t *testing.T) {
		internaltest.TestTextOutputDocInput(t, func(t *testing.T) genai.Provider {
			return getClient(t, genai.ModelCheap)
		})
	})

	t.Run("errors", func(t *testing.T) {
		data := []internaltest.ProviderError{
			{
				Name: "bad apiKey",
				Opts: genai.ProviderOptions{
					APIKey: "bad apiKey",
					Model:  "llama-3.1-8b-instant",
				},
				ErrGenSync:   "http 401\ninvalid_api_key (invalid_request_error): Invalid API Key\nget a new API key at https://console.groq.com/keys",
				ErrGenStream: "http 401\ninvalid_api_key (invalid_request_error): Invalid API Key\nget a new API key at https://console.groq.com/keys",
				ErrListModel: "http 401\ninvalid_api_key (invalid_request_error): Invalid API Key\nget a new API key at https://console.groq.com/keys",
			},
			{
				Name: "bad model",
				Opts: genai.ProviderOptions{
					Model: "bad model",
				},
				ErrGenSync:   "http 404\nmodel_not_found (invalid_request_error): The model `bad model` does not exist or you do not have access to it.",
				ErrGenStream: "http 404\nmodel_not_found (invalid_request_error): The model `bad model` does not exist or you do not have access to it.",
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

type handleGroqReasoning struct {
	genai.Provider
}

func (h *handleGroqReasoning) GenSync(ctx context.Context, msgs genai.Messages, opts ...genai.Options) (genai.Result, error) {
	for _, opt := range opts {
		if o, ok := opt.(*genai.OptionsTools); ok && len(o.Tools) != 0 {
			opts = append(opts, &groq.Options{ReasoningFormat: groq.ReasoningFormatParsed})
			return h.Provider.GenSync(ctx, msgs, opts...)
		}
		if o, ok := opt.(*genai.OptionsText); ok && (o.DecodeAs != nil || o.ReplyAsJSON) {
			opts = append(opts, &groq.Options{ReasoningFormat: groq.ReasoningFormatParsed})
			return h.Provider.GenSync(ctx, msgs, opts...)
		}
	}
	c := adapters.ProviderReasoning{Provider: h.Provider, ReasoningTokenStart: "<think>", ReasoningTokenEnd: "\n</think>\n"}
	return c.GenSync(ctx, msgs, opts...)
}

func (h *handleGroqReasoning) GenStream(ctx context.Context, msgs genai.Messages, opts ...genai.Options) (iter.Seq[genai.Reply], func() (genai.Result, error)) {
	for _, opt := range opts {
		if o, ok := opt.(*genai.OptionsTools); ok && len(o.Tools) != 0 {
			opts = append(opts, &groq.Options{ReasoningFormat: groq.ReasoningFormatParsed})
			return h.Provider.GenStream(ctx, msgs, opts...)
		}
		if o, ok := opt.(*genai.OptionsText); ok && (o.DecodeAs != nil || o.ReplyAsJSON) {
			opts = append(opts, &groq.Options{ReasoningFormat: groq.ReasoningFormatParsed})
			return h.Provider.GenStream(ctx, msgs, opts...)
		}
	}
	c := adapters.ProviderReasoning{Provider: h.Provider, ReasoningTokenStart: "<think>", ReasoningTokenEnd: "\n</think>\n"}
	return c.GenStream(ctx, msgs, opts...)
}

func (h *handleGroqReasoning) Unwrap() genai.Provider {
	return h.Provider
}

func init() {
	internal.BeLenient = false
}
