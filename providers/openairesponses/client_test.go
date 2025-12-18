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
	"testing"

	"github.com/maruel/genai"
	"github.com/maruel/genai/internal"
	"github.com/maruel/genai/internal/internaltest"
	"github.com/maruel/genai/providers/openairesponses"
	"github.com/maruel/genai/scoreboard"
	"github.com/maruel/genai/smoke/smoketest"
)

func getClientInner(t *testing.T, opts genai.ProviderOptions, fn func(http.RoundTripper) http.RoundTripper) (genai.Provider, error) {
	if opts.APIKey == "" && os.Getenv("OPENAI_API_KEY") == "" {
		opts.APIKey = "<insert_api_key_here>"
	}
	return openairesponses.New(t.Context(), &opts, fn)
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
			reason := (strings.HasPrefix(id, "gpt-5") || strings.HasPrefix(id, "o")) && !strings.Contains(id, "moderation")
			models = append(models, scoreboard.Model{Model: id, Reason: reason})
		}
		getClientRT := func(t testing.TB, model scoreboard.Model, fn func(http.RoundTripper) http.RoundTripper) genai.Provider {
			opts := genai.ProviderOptions{Model: model.Model, PreloadedModels: cachedModels}
			if os.Getenv("OPENAI_API_KEY") == "" {
				opts.APIKey = "<insert_api_key_here>"
			}
			c, err := openairesponses.New(t.Context(), &opts, fn)
			if err != nil {
				t.Fatal(err)
			}
			// This will lead to spurious HTTP 500 but it is 25% of the cost.
			tier := openairesponses.ServiceTierFlex
			res := openairesponses.ReasoningEffortLow
			if strings.Contains(model.Model, "-chat-latest") {
				// Flex and Low are not supported.
				tier = openairesponses.ServiceTierDefault
				res = openairesponses.ReasoningEffortMedium
			}
			if model.Reason {
				return &internaltest.InjectOptions{
					Provider: c,
					Opts: []genai.Options{
						&openairesponses.OptionsText{
							ReasoningEffort: res,
							ServiceTier:     tier,
						},
					},
				}
			}
			if slices.Equal(c.OutputModalities(), []genai.Modality{genai.ModalityText}) {
				// See https://platform.openai.com/docs/guides/flex-processing
				if id := c.ModelID(); id == "o3" || id == "o4-mini" || strings.HasPrefix(id, "gpt-5") {
					return &internaltest.InjectOptions{
						Provider: c,
						Opts:     []genai.Options{&openairesponses.OptionsText{ServiceTier: tier}},
					}
				}
			}
			return c
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

	t.Run("Preferred", func(t *testing.T) {
		data := []struct {
			modality genai.Modality
			name     string
			want     string
		}{
			{genai.ModalityText, genai.ModelCheap, "gpt-5-nano"},
			{genai.ModalityText, genai.ModelGood, "gpt-5-mini"},
			{genai.ModalityText, genai.ModelSOTA, "gpt-5.2-chat-latest"},
			{genai.ModalityImage, genai.ModelCheap, "dall-e-3"},
			{genai.ModalityImage, genai.ModelGood, "gpt-image-1-mini"},
			{genai.ModalityImage, genai.ModelSOTA, "gpt-image-1"},
			{genai.ModalityVideo, genai.ModelCheap, "sora-2"},
			{genai.ModalityVideo, genai.ModelGood, "sora-2"},
			{genai.ModalityVideo, genai.ModelSOTA, "sora-2-pro"},
		}
		// Note: Audio test is in errors section since OpenAI Responses API doesn't support it

		for _, line := range data {
			t.Run(line.name, func(t *testing.T) {
				t.Run(fmt.Sprintf("%s-%s", line.modality, line.name), func(t *testing.T) {
					opts := genai.ProviderOptions{
						Model:            line.name,
						OutputModalities: genai.Modalities{line.modality},
						PreloadedModels:  cachedModels,
					}
					c, err := getClientInner(t, opts, func(h http.RoundTripper) http.RoundTripper {
						return testRecorder.Record(t, h)
					})
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
			/* TODO:
			{
				Name: "audio not supported",
				Opts: genai.ProviderOptions{
					Model:            genai.ModelGood,
					OutputModalities: genai.Modalities{genai.ModalityAudio},
				},
				ErrGenSync:   "OpenAI Responses API does not support audio output as of December 2025; see https://platform.openai.com/docs/guides/audio",
				ErrGenStream: "OpenAI Responses API does not support audio output as of December 2025; see https://platform.openai.com/docs/guides/audio",
			},
			*/
		}
		f := func(t *testing.T, opts genai.ProviderOptions) (genai.Provider, error) {
			return getClientInner(t, opts, func(h http.RoundTripper) http.RoundTripper {
				return testRecorder.Record(t, h)
			})
		}
		internaltest.TestClient_Provider_errors(t, f, data)
	})
}

func init() {
	internal.BeLenient = false
}
