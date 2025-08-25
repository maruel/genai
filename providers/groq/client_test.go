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
	"sync"
	"testing"

	"github.com/maruel/genai"
	"github.com/maruel/genai/adapters"
	"github.com/maruel/genai/internal"
	"github.com/maruel/genai/internal/internaltest"
	"github.com/maruel/genai/internal/myrecorder"
	"github.com/maruel/genai/providers/groq"
	"github.com/maruel/genai/scoreboard/scoreboardtest"
)

func getClientRT(t testing.TB, model scoreboardtest.Model, fn func(http.RoundTripper) http.RoundTripper) genai.Provider {
	apiKey := ""
	if os.Getenv("GROQ_API_KEY") == "" {
		apiKey = "<insert_api_key_here>"
	}
	opts := genai.ProviderOptions{
		APIKey:          apiKey,
		Model:           model.Model,
		PreloadedModels: loadCachedModelsList(t),
	}
	cl, err := groq.New(t.Context(), &opts, fn)
	if err != nil {
		t.Fatal(err)
	}
	var c genai.Provider = cl
	if strings.HasPrefix(model.Model, "qwen/") && model.Thinking {
		c = &adapters.ProviderAppend{Provider: c, Append: genai.Request{Text: "\n\n/think"}}
	}
	if model.Thinking {
		return &handleGroqReasoning{Provider: c}
	}
	return c
}

func TestClient_Scoreboard(t *testing.T) {
	c := getClient(t, genai.ModelNone)
	genaiModels, err := c.ListModels(t.Context())
	if err != nil {
		t.Fatal(err)
	}
	scenarios := c.Scoreboard().Scenarios
	var models []scoreboardtest.Model
	for _, m := range genaiModels {
		id := m.GetID()
		thinking := false
		for _, sc := range scenarios {
			if slices.Contains(sc.Models, id) {
				t.Logf("%s: %t", id, sc.Thinking)
				thinking = sc.Thinking
				break
			}
		}
		models = append(models, scoreboardtest.Model{Model: id, Thinking: thinking})
	}
	scoreboardtest.AssertScoreboard(t, getClientRT, models, testRecorder.Records)
}

type handleGroqReasoning struct {
	genai.Provider
}

func (h *handleGroqReasoning) GenSync(ctx context.Context, msgs genai.Messages, opts ...genai.Options) (genai.Result, error) {
	for _, opt := range opts {
		if o, ok := opt.(*genai.OptionsText); ok && len(o.Tools) != 0 || o.DecodeAs != nil || o.ReplyAsJSON {
			opts = append(opts, &groq.Options{ReasoningFormat: groq.ReasoningFormatParsed})
			return h.Provider.GenSync(ctx, msgs, opts...)
		}
	}
	c := adapters.ProviderThinking{Provider: h.Provider, ThinkingTokenStart: "<think>", ThinkingTokenEnd: "\n</think>\n"}
	return c.GenSync(ctx, msgs, opts...)
}

func (h *handleGroqReasoning) GenStream(ctx context.Context, msgs genai.Messages, opts ...genai.Options) (iter.Seq[genai.ReplyFragment], func() (genai.Result, error)) {
	for _, opt := range opts {
		if o, ok := opt.(*genai.OptionsText); ok && len(o.Tools) != 0 || o.DecodeAs != nil || o.ReplyAsJSON {
			opts = append(opts, &groq.Options{ReasoningFormat: groq.ReasoningFormatParsed})
			return h.Provider.GenStream(ctx, msgs, opts...)
		}
	}
	c := adapters.ProviderThinking{Provider: h.Provider, ThinkingTokenStart: "<think>", ThinkingTokenEnd: "\n</think>\n"}
	return c.GenStream(ctx, msgs, opts...)
}

func (h *handleGroqReasoning) Unwrap() genai.Provider {
	return h.Provider
}

func TestClient_Preferred(t *testing.T) {
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
}

func TestClient_Provider_errors(t *testing.T) {
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
		return getClientInner(t, opts)
	}
	internaltest.TestClient_Provider_errors(t, f, data)
}

func getClient(t *testing.T, m string) *groq.Client {
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

func getClientInner(t *testing.T, opts genai.ProviderOptions) (*groq.Client, error) {
	if opts.APIKey == "" && os.Getenv("GROQ_API_KEY") == "" {
		opts.APIKey = "<insert_api_key_here>"
	}
	return groq.New(t.Context(), &opts, func(h http.RoundTripper) http.RoundTripper { return testRecorder.Record(t, h) })
}

func loadCachedModelsList(t testing.TB) []genai.Model {
	doOnce.Do(func() {
		var r myrecorder.Recorder
		var err2 error
		ctx := t.Context()
		opts := genai.ProviderOptions{Model: genai.ModelNone}
		if os.Getenv("GROQ_API_KEY") == "" {
			opts.APIKey = "<insert_api_key_here>"
		}
		c, err := groq.New(ctx, &opts, func(h http.RoundTripper) http.RoundTripper {
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
