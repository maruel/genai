// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package cerebras_test

import (
	"log/slog"
	"net/http"
	"os"
	"slices"
	"sync"
	"testing"

	"github.com/maruel/genai"
	"github.com/maruel/genai/adapters"
	"github.com/maruel/genai/internal"
	"github.com/maruel/genai/internal/internaltest"
	"github.com/maruel/genai/internal/myrecorder"
	"github.com/maruel/genai/providers/cerebras"
	"github.com/maruel/genai/smoke/smoketest"
	"github.com/maruel/roundtrippers"
)

func getClientRT(t testing.TB, model smoketest.Model, fn func(http.RoundTripper) http.RoundTripper) genai.Provider {
	apiKey := ""
	if os.Getenv("CEREBRAS_API_KEY") == "" {
		apiKey = "<insert_api_key_here>"
	}
	ctx, l := internaltest.Log(t)
	fnWithLog := func(h http.RoundTripper) http.RoundTripper {
		if fn != nil {
			h = fn(h)
		}
		return &roundtrippers.Log{
			Transport: h,
			Logger:    l,
			Level:     slog.LevelDebug,
		}
	}
	opts := genai.ProviderOptions{
		APIKey:          apiKey,
		Model:           model.Model,
		PreloadedModels: loadCachedModelsList(t),
	}
	c, err2 := cerebras.New(ctx, &opts, fnWithLog)
	if err2 != nil {
		t.Fatal(err2)
	}
	if model.Thinking {
		// Check if it has predefined thinking tokens.
		for _, sc := range c.Scoreboard().Scenarios {
			if sc.Thinking && slices.Contains(sc.Models, model.Model) {
				if sc.ThinkingTokenStart != "" && sc.ThinkingTokenEnd != "" {
					return &adapters.ProviderThinking{
						Provider:           &adapters.ProviderAppend{Provider: c, Append: genai.Request{Text: "\n\n/think"}},
						ThinkingTokenStart: sc.ThinkingTokenStart,
						ThinkingTokenEnd:   sc.ThinkingTokenEnd,
					}
				}
				break
			}
		}
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
	var models []smoketest.Model
	for _, m := range genaiModels {
		id := m.GetID()
		thinking := false
		for _, sc := range scenarios {
			if slices.Contains(sc.Models, id) {
				thinking = sc.Thinking
				break
			}
		}
		models = append(models, smoketest.Model{Model: id, Thinking: thinking})
	}
	smoketest.Run(t, getClientRT, models, testRecorder.Records)
}

func TestClient_Preferred(t *testing.T) {
	data := []struct {
		name string
		want string
	}{
		{genai.ModelCheap, "llama3.1-8b"},
		{genai.ModelGood, "llama-4-maverick-17b-128e-instruct"},
		{genai.ModelSOTA, "qwen-3-235b-a22b-instruct-2507"},
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
				Model:  "llama-3.1-8b",
			},
			ErrGenSync:   "http 401\ninvalid_request_error/api_key/wrong_api_key: Wrong API Key\nget a new API key at https://cloud.cerebras.ai/platform/",
			ErrGenStream: "http 401\ninvalid_request_error/api_key/wrong_api_key: Wrong API Key\nget a new API key at https://cloud.cerebras.ai/platform/",
			ErrListModel: "http 401\ninvalid_request_error/api_key/wrong_api_key: Wrong API Key\nget a new API key at https://cloud.cerebras.ai/platform/",
		},
		{
			Name: "bad model",
			Opts: genai.ProviderOptions{
				Model: "bad model",
			},
			ErrGenSync:   "http 404\nnot_found_error/model/model_not_found: Model bad model does not exist or you do not have access to it.",
			ErrGenStream: "http 404\nnot_found_error/model/model_not_found: Model bad model does not exist or you do not have access to it.",
		},
	}
	f := func(t *testing.T, opts genai.ProviderOptions) (genai.Provider, error) {
		opts.OutputModalities = genai.Modalities{genai.ModalityText}
		return getClientInner(t, opts)
	}
	internaltest.TestClient_Provider_errors(t, f, data)
}

func getClient(t *testing.T, m string) *cerebras.Client {
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

func getClientInner(t *testing.T, opts genai.ProviderOptions) (*cerebras.Client, error) {
	if opts.APIKey == "" && os.Getenv("CEREBRAS_API_KEY") == "" {
		opts.APIKey = "<insert_api_key_here>"
	}
	ctx, l := internaltest.Log(t)
	return cerebras.New(ctx, &opts, func(h http.RoundTripper) http.RoundTripper {
		r := testRecorder.Record(t, h)
		return &roundtrippers.Log{
			Transport: r,
			Logger:    l,
			Level:     slog.LevelWarn,
		}
	})
}

func loadCachedModelsList(t testing.TB) []genai.Model {
	doOnce.Do(func() {
		var r myrecorder.Recorder
		var err2 error
		ctx := t.Context()
		opts := genai.ProviderOptions{Model: genai.ModelNone}
		if os.Getenv("CEREBRAS_API_KEY") == "" {
			opts.APIKey = "<insert_api_key_here>"
		}
		c, err := cerebras.New(ctx, &opts, func(h http.RoundTripper) http.RoundTripper {
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
