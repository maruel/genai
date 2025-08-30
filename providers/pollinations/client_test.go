// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package pollinations_test

import (
	"context"
	_ "embed"
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
	"github.com/maruel/genai/providers/pollinations"
	"github.com/maruel/genai/smoke/smoketest"
)

func TestClient(t *testing.T) {
	t.Run("Scoreboard", func(t *testing.T) {
		models := loadCachedModelsList(t)
		var sbModels []smoketest.Model
		for _, m := range models {
			id := m.GetID()
			sbModels = append(sbModels, smoketest.Model{Model: id, Reason: id == "deepseek-reasoning"})
		}
		smoketest.Run(t, getClientRT, sbModels, testRecorder.Records)
	})

	t.Run("Preferred", func(t *testing.T) {
		data := []struct {
			name string
			want string
		}{
			{genai.ModelCheap, "llamascout"},
			{genai.ModelGood, "openai-large"},
			{genai.ModelSOTA, "deepseek-reasoning"},
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
				Name: "bad model",
				Opts: genai.ProviderOptions{
					Model: "bad model",
				},
				ErrGenSync:   "model \"bad model\" not supported by pollinations",
				ErrGenStream: "model \"bad model\" not supported by pollinations",
			},
			{
				Name: "bad model image",
				Opts: genai.ProviderOptions{
					Model:            "bad model",
					OutputModalities: genai.Modalities{genai.ModalityImage},
				},
				ErrGenSync:   "model \"bad model\" not supported by pollinations",
				ErrGenStream: "model \"bad model\" not supported by pollinations",
			},
		}
		f := func(t *testing.T, opts genai.ProviderOptions) (genai.Provider, error) {
			opts.APIKey = "genai-unittests"
			opts.OutputModalities = genai.Modalities{genai.ModalityText}
			return getClientInner(t, opts)
		}
		internaltest.TestClient_Provider_errors(t, f, data)
	})
}

func getClientRT(t testing.TB, model smoketest.Model, fn func(http.RoundTripper) http.RoundTripper) genai.Provider {
	opts := genai.ProviderOptions{
		APIKey:          "genai-unittests",
		Model:           model.Model,
		PreloadedModels: loadCachedModelsList(t),
	}
	c, err := pollinations.New(t.Context(), &opts, fn)
	if err != nil {
		t.Fatal(err)
	}
	c2 := &smallImage{Provider: &internaltest.HideHTTP500{Provider: c}}
	if model.Reason {
		for _, sc := range c.Scoreboard().Scenarios {
			if sc.Reason && slices.Contains(sc.Models, model.Model) {
				if sc.ReasoningTokenStart != "" && sc.ReasoningTokenEnd != "" {
					return &adapters.ProviderReasoning{
						Provider:            c2,
						ReasoningTokenStart: sc.ReasoningTokenStart,
						ReasoningTokenEnd:   sc.ReasoningTokenEnd,
					}
				}
				break
			}
		}
	}
	return c2
}

// smallImage speeds up image generation.
type smallImage struct {
	genai.Provider
}

func (h *smallImage) Unwrap() genai.Provider {
	return h.Provider
}

func (h *smallImage) GenSync(ctx context.Context, msgs genai.Messages, opts ...genai.Options) (genai.Result, error) {
	for i := range opts {
		if v, ok := opts[i].(*genai.OptionsImage); ok {
			// Ask for a smaller size.
			n := *v
			n.Width = 256
			n.Height = 256
			opts[i] = &n
		}
	}
	return h.Provider.GenSync(ctx, msgs, opts...)
}

func getClient(t *testing.T, m string) *pollinations.Client {
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

func getClientInner(t *testing.T, opts genai.ProviderOptions) (*pollinations.Client, error) {
	if opts.APIKey == "" {
		opts.APIKey = "genai-unittests"
	}
	return pollinations.New(t.Context(), &opts, func(h http.RoundTripper) http.RoundTripper {
		return testRecorder.Record(t, h)
	})
}

func loadCachedModelsList(t testing.TB) []genai.Model {
	doOnce.Do(func() {
		var r *myrecorder.Recorder
		var err2 error
		ctx := t.Context()
		opts := genai.ProviderOptions{APIKey: "genai-unittests", Model: genai.ModelNone}
		c, err := pollinations.New(ctx, &opts, func(h http.RoundTripper) http.RoundTripper {
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
