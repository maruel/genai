// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package openairesponses_test

import (
	"fmt"
	"net/http"
	"os"
	"slices"
	"strings"
	"sync"
	"testing"

	"github.com/maruel/genai"
	"github.com/maruel/genai/internal"
	"github.com/maruel/genai/internal/internaltest"
	"github.com/maruel/genai/internal/myrecorder"
	"github.com/maruel/genai/providers/openai/openairesponses"
	"github.com/maruel/genai/smoke/smoketest"
)

func TestClient(t *testing.T) {
	t.Run("Scoreboard", func(t *testing.T) {
		genaiModels, err := getClient(t, genai.ModelNone).ListModels(t.Context())
		if err != nil {
			t.Fatal(err)
		}
		var models []smoketest.Model
		for _, m := range genaiModels {
			id := m.GetID()
			models = append(models, smoketest.Model{Model: id, Reason: strings.HasPrefix(id, "o") && !strings.Contains(id, "moderation")})
		}
		smoketest.Run(t, getClientRT, models, testRecorder.Records)
	})

	t.Run("Batch", func(t *testing.T) {
		t.Skip("implement")
		/*
			// This is a tricky test since batch operations can take up to 24h to complete.
			ctx := t.Context()
			c := getClient(t, "gpt-3.5-turbo")
			// Using an extremely old cheap model that nobody uses helps a lot on reducing the latency, I got it to work
			// within a few minutes.
			msgs := genai.Messages{genai.NewTextMessage("Tell a joke in 10 words")}
			job, err := c.GenAsync(ctx, msgs)
			if err != nil {
				t.Fatal(err)
			}
			// TODO: Detect when recording and sleep only in this case.
			is_recording := os.Getenv("RECORD") == "1"
			for {
				res, err := c.PokeResult(ctx, job)
				if err != nil {
					t.Fatal(err)
				}
				if res.FinishReason == genai.Pending {
					if is_recording {
						t.Logf("Waiting...")
						time.Sleep(time.Second)
					}
					continue
				}
				if res.InputTokens == 0 || res.OutputTokens == 0 {
					t.Error("expected usage")
				}
				if res.FinishReason != genai.FinishedStop {
					t.Errorf("finish reason: %s", res.FinishReason)
				}
				if s := res.String(); len(s) < 15 {
					t.Errorf("not enough text: %q", s)
				}
				break
			}
		*/
	})

	t.Run("WebSearch", func(t *testing.T) {
		// See https://platform.openai.com/docs/guides/tools-web-search
		opts := genai.OptionsTools{WebSearch: true, Force: genai.ToolCallRequired}
		t.Run("GenSync", func(t *testing.T) {
			c := getClient(t, "gpt-4o-mini")
			msgs := genai.Messages{genai.NewTextMessage("Search the web to determine what's the full name of the IANA organization")}
			res, err := c.GenSync(t.Context(), msgs, &opts)
			if err != nil {
				t.Fatal(err)
			}
			if !slices.ContainsFunc(res.Replies, func(r genai.Reply) bool {
				return r.Citations != nil
			}) {
				t.Logf("%+v", res)
				t.Fatal("no citations")
			}
		})
		t.Run("GenStream", func(t *testing.T) {
			c := getClient(t, "gpt-4o-mini")
			msgs := genai.Messages{genai.NewTextMessage("Search the web to determine what's the full name of the IANA organization")}
			fragments, finish := c.GenStream(t.Context(), msgs, &opts)
			hasCitation := false
			for f := range fragments {
				if !f.Citation.IsZero() {
					hasCitation = true
				}
				// t.Logf("%+v", f)
			}
			res, err := finish()
			if err != nil {
				t.Fatal(err)
			}
			if !hasCitation {
				t.Logf("%+v", res)
				t.Fatal("no citations")
			}
		})
	})

	t.Run("Preferred", func(t *testing.T) {
		data := []struct {
			modality genai.Modality
			name     string
			want     string
		}{
			{genai.ModalityText, genai.ModelCheap, "gpt-4.1-nano"},
			{genai.ModalityText, genai.ModelGood, "gpt-5-mini"},
			{genai.ModalityText, genai.ModelSOTA, "o3-pro"},
			{genai.ModalityImage, genai.ModelCheap, "dall-e-3"},
			{genai.ModalityImage, genai.ModelGood, "gpt-image-1"},
			{genai.ModalityImage, genai.ModelSOTA, "gpt-image-1"},
		}
		for _, line := range data {
			t.Run(line.name, func(t *testing.T) {
				t.Run(fmt.Sprintf("%s-%s", line.modality, line.name), func(t *testing.T) {
					opts := genai.ProviderOptions{
						Model:            line.name,
						OutputModalities: genai.Modalities{line.modality},
						PreloadedModels:  loadCachedModelsList(t),
					}
					c, err := getClientInner(t, opts)
					if err != nil {
						t.Fatal(err)
					}
					if got := c.ModelID(); got != line.want {
						t.Fatalf("got model %q, want %q", got, line.want)
					}
				})
			})
		}
	})

	t.Run("errors", func(t *testing.T) {
		data := []internaltest.ProviderError{
			{
				Name: "bad apiKey",
				Opts: genai.ProviderOptions{
					APIKey: "bad apiKey",
					Model:  "gpt-4.1-nano",
				},
				ErrGenSync:   "http 401\nIncorrect API key provided: bad apiKey. You can find your API key at https://platform.openai.com/account/api-keys. (type: invalid_request_error, code: invalid_api_key)",
				ErrGenStream: "http 401\nIncorrect API key provided: bad apiKey. You can find your API key at https://platform.openai.com/account/api-keys. (type: invalid_request_error, code: invalid_api_key)",
				ErrListModel: "http 401\nIncorrect API key provided: bad apiKey. You can find your API key at https://platform.openai.com/account/api-keys. (type: invalid_request_error, code: invalid_api_key)",
			},
			{
				Name: "bad apiKey image",
				Opts: genai.ProviderOptions{
					APIKey:           "bad apiKey",
					Model:            "gpt-image-1",
					OutputModalities: genai.Modalities{genai.ModalityImage},
				},
				ErrGenSync:   "http 401\nIncorrect API key provided: bad apiKey. You can find your API key at https://platform.openai.com/account/api-keys. (type: invalid_request_error, code: invalid_api_key)",
				ErrGenStream: "http 401\nIncorrect API key provided: bad apiKey. You can find your API key at https://platform.openai.com/account/api-keys. (type: invalid_request_error, code: invalid_api_key)",
			},
			{
				Name: "bad model",
				Opts: genai.ProviderOptions{
					Model: "bad model",
				},
				ErrGenSync:   "http 400\nThe requested model 'bad model' does not exist. (type: invalid_request_error, code: model_not_found)",
				ErrGenStream: "http 400\nThe requested model 'bad model' does not exist. (type: invalid_request_error, code: model_not_found)",
			},
			{
				Name: "bad model image",
				Opts: genai.ProviderOptions{
					Model:            "bad model",
					OutputModalities: genai.Modalities{genai.ModalityImage},
				},
				ErrGenSync:   "http 400\nInvalid value: 'bad model'. Supported values are: 'gpt-image-1', 'gpt-image-1-io', 'gpt-image-0721-mini-alpha', 'dall-e-2', and 'dall-e-3'. (type: invalid_request_error, code: invalid_value)",
				ErrGenStream: "http 400\nInvalid value: 'bad model'. Supported values are: 'gpt-image-1', 'gpt-image-1-io', 'gpt-image-0721-mini-alpha', 'dall-e-2', and 'dall-e-3'. (type: invalid_request_error, code: invalid_value)",
			},
		}
		f := func(t *testing.T, opts genai.ProviderOptions) (genai.Provider, error) {
			return getClientInner(t, opts)
		}
		internaltest.TestClient_Provider_errors(t, f, data)
	})
}

func getClientRT(t testing.TB, model smoketest.Model, fn func(http.RoundTripper) http.RoundTripper) genai.Provider {
	apiKey := ""
	if os.Getenv("OPENAI_API_KEY") == "" {
		apiKey = "<insert_api_key_here>"
	}
	opts := genai.ProviderOptions{
		APIKey:          apiKey,
		Model:           model.Model,
		PreloadedModels: loadCachedModelsList(t),
	}
	c, err := openairesponses.New(t.Context(), &opts, fn)
	if err != nil {
		t.Fatal(err)
	}
	if model.Reason {
		return &internaltest.InjectOptions{
			Provider: c,
			Opts: []genai.Options{
				&openairesponses.OptionsText{
					// This will lead to spurious HTTP 500 but it is 25% of the cost.
					ServiceTier:     openairesponses.ServiceTierFlex,
					ReasoningEffort: openairesponses.ReasoningEffortLow,
				},
			},
		}
	}
	if slices.Equal(c.OutputModalities(), []genai.Modality{genai.ModalityText}) {
		// See https://platform.openai.com/docs/guides/flex-processing
		if id := c.ModelID(); id == "o3" || id == "o4-mini" || strings.HasPrefix(id, "gpt-5") {
			return &internaltest.InjectOptions{
				Provider: c,
				Opts: []genai.Options{
					&openairesponses.OptionsText{
						// This will lead to spurious HTTP 500 but it is 25% of the cost.
						ServiceTier: openairesponses.ServiceTierFlex,
					},
				},
			}
		}
	}
	return c
}

func getClient(t *testing.T, m string) *openairesponses.Client {
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

func getClientInner(t *testing.T, opts genai.ProviderOptions) (*openairesponses.Client, error) {
	if opts.APIKey == "" && os.Getenv("OPENAI_API_KEY") == "" {
		opts.APIKey = "<insert_api_key_here>"
	}
	return openairesponses.New(t.Context(), &opts, func(h http.RoundTripper) http.RoundTripper {
		return testRecorder.Record(t, h)
	})
}

func loadCachedModelsList(t testing.TB) []genai.Model {
	doOnce.Do(func() {
		var r myrecorder.Recorder
		var err2 error
		ctx := t.Context()
		opts := genai.ProviderOptions{Model: genai.ModelNone}
		if os.Getenv("OPENAI_API_KEY") == "" {
			opts.APIKey = "<insert_api_key_here>"
		}
		c, err := openairesponses.New(ctx, &opts, func(h http.RoundTripper) http.RoundTripper {
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
