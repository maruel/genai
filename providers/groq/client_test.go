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

	t.Run("Scoreboard", func(t *testing.T) {
		c := getClient(t, genai.ModelNone)
		genaiModels, err := c.ListModels(t.Context())
		if err != nil {
			t.Fatal(err)
		}
		scenarios := c.Scoreboard().Scenarios
		var models []smoketest.Model
		for _, m := range genaiModels {
			id := m.GetID()
			reason := false
			for _, sc := range scenarios {
				if slices.Contains(sc.Models, id) {
					t.Logf("%s: %t", id, sc.Reason)
					reason = sc.Reason
					break
				}
			}
			models = append(models, smoketest.Model{Model: id, Reason: reason})
		}
		getClientRT := func(t testing.TB, model smoketest.Model, fn func(http.RoundTripper) http.RoundTripper) genai.Provider {
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
			// OpenAI must not enable the ReasoningFormat flag.
			if model.Reason && !strings.HasPrefix(model.Model, "openai/") {
				return &handleGroqReasoning{Provider: c}
			}
			return c
		}
		smoketest.Run(t, getClientRT, models, testRecorder.Records)
	})

	t.Run("Preferred", func(t *testing.T) {
		data := []struct {
			name string
			want string
		}{
			{genai.ModelCheap, "llama-3.1-8b-instant"},
			{genai.ModelGood, "meta-llama/llama-4-maverick-17b-128e-instruct"},
			{genai.ModelSOTA, "qwen/qwen3-32b"},
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
		internaltest.TestClient_Provider_errors(t, f, data)
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

func (h *handleGroqReasoning) GenStream(ctx context.Context, msgs genai.Messages, opts ...genai.Options) (iter.Seq[genai.ReplyFragment], func() (genai.Result, error)) {
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
