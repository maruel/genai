// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package openairesponses_test

import (
	"encoding/json"
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

func getClientInner(t *testing.T, fn func(http.RoundTripper) http.RoundTripper, opts ...genai.ProviderOption) (genai.Provider, error) {
	hasAPIKey := false
	for _, opt := range opts {
		if _, ok := opt.(genai.ProviderOptionAPIKey); ok {
			hasAPIKey = true
			break
		}
	}
	if !hasAPIKey && os.Getenv("OPENAI_API_KEY") == "" {
		opts = append(opts, genai.ProviderOptionAPIKey("<insert_api_key_here>"))
	}
	if fn != nil {
		opts = append([]genai.ProviderOption{genai.ProviderOptionTransportWrapper(fn)}, opts...)
	}
	return openairesponses.New(t.Context(), opts...)
}

func TestClient(t *testing.T) {
	testRecorder := internaltest.NewRecords()
	t.Cleanup(func() {
		if err := testRecorder.Close(); err != nil {
			t.Error(err)
		}
	})
	cl, err2 := getClientInner(t, func(h http.RoundTripper) http.RoundTripper {
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
		ci, err := getClientInner(t, func(h http.RoundTripper) http.RoundTripper {
			return testRecorder.Record(t, h)
		}, opts...)
		if err != nil {
			t.Fatal(err)
		}
		return ci
	}

	t.Run("Capabilities", func(t *testing.T) {
		internaltest.TestCapabilities(t, getClient(t, ""))
	})

	t.Run("Scoreboard", func(t *testing.T) {
		genaiModels, err := getClient(t, "").ListModels(t.Context())
		if err != nil {
			t.Fatal(err)
		}
		models := make([]scoreboard.Model, 0, len(genaiModels))
		for _, m := range genaiModels {
			id := m.GetID()
			reason := (strings.HasPrefix(id, "gpt-5") || strings.HasPrefix(id, "o")) && !strings.Contains(id, "moderation")
			models = append(models, scoreboard.Model{Model: id, Reason: reason})
		}
		getClientRT := func(t testing.TB, model scoreboard.Model, fn func(http.RoundTripper) http.RoundTripper) genai.Provider {
			opts := []genai.ProviderOption{genai.ProviderOptionPreloadedModels(cachedModels)}
			if model.Model != "" {
				opts = append(opts, genai.ProviderOptionModel(model.Model))
			}
			if os.Getenv("OPENAI_API_KEY") == "" {
				opts = append(opts, genai.ProviderOptionAPIKey("<insert_api_key_here>"))
			}
			if fn != nil {
				opts = append([]genai.ProviderOption{genai.ProviderOptionTransportWrapper(fn)}, opts...)
			}
			c, err := openairesponses.New(t.Context(), opts...)
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
					Opts: []genai.GenOption{
						&openairesponses.GenOptionText{
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
						Opts:     []genai.GenOption{&openairesponses.GenOptionText{ServiceTier: tier}},
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
		internaltest.TestPreferredModels(t, func(st *testing.T, model string, modality genai.Modality) (genai.Provider, error) {
			opts := []genai.ProviderOption{
				genai.ProviderOptionModalities{modality},
				genai.ProviderOptionPreloadedModels(cachedModels),
			}
			if model != "" {
				opts = append(opts, genai.ProviderOptionModel(model))
			}
			return getClientInner(st, func(h http.RoundTripper) http.RoundTripper {
				return testRecorder.Record(st, h)
			}, opts...)
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
					genai.ProviderOptionModel("gpt-4.1-nano"),
				},
				ErrGenSync:   "http 401\nIncorrect API key provided: bad apiKey. You can find your API key at https://platform.openai.com/account/api-keys. (type: invalid_request_error, code: invalid_api_key)",
				ErrGenStream: "http 401\nIncorrect API key provided: bad apiKey. You can find your API key at https://platform.openai.com/account/api-keys. (type: invalid_request_error, code: invalid_api_key)",
				ErrListModel: "http 401\nIncorrect API key provided: bad apiKey. You can find your API key at https://platform.openai.com/account/api-keys. (type: invalid_request_error, code: invalid_api_key)",
			},
			{
				Name: "bad apiKey image",
				Opts: []genai.ProviderOption{
					genai.ProviderOptionAPIKey("bad apiKey"),
					genai.ProviderOptionModel("gpt-image-1"),
					genai.ProviderOptionModalities{genai.ModalityImage},
				},
				ErrGenSync:   "http 401\nIncorrect API key provided: bad apiKey. You can find your API key at https://platform.openai.com/account/api-keys. (type: invalid_request_error, code: invalid_api_key)",
				ErrGenStream: "http 401\nIncorrect API key provided: bad apiKey. You can find your API key at https://platform.openai.com/account/api-keys. (type: invalid_request_error, code: invalid_api_key)",
			},
			{
				Name: "bad model",
				Opts: []genai.ProviderOption{
					genai.ProviderOptionModel("bad model"),
				},
				ErrGenSync:   "http 400\nThe requested model 'bad model' does not exist. (type: invalid_request_error, code: model_not_found)",
				ErrGenStream: "http 400\nThe requested model 'bad model' does not exist. (type: invalid_request_error, code: model_not_found)",
			},
			{
				Name: "bad model image",
				Opts: []genai.ProviderOption{
					genai.ProviderOptionModel("bad model"),
					genai.ProviderOptionModalities{genai.ModalityImage},
				},
				ErrGenSync:   "http 400\nInvalid value: 'bad model'. Supported values are: 'gpt-image-1', 'gpt-image-1-io', 'gpt-image-0721-mini-alpha', 'dall-e-2', and 'dall-e-3'. (type: invalid_request_error, code: invalid_value)",
				ErrGenStream: "http 400\nInvalid value: 'bad model'. Supported values are: 'gpt-image-1', 'gpt-image-1-io', 'gpt-image-0721-mini-alpha', 'dall-e-2', and 'dall-e-3'. (type: invalid_request_error, code: invalid_value)",
			},
			/* TODO:
			{
				Name: "audio not supported",
				Opts: []genai.ProviderOption{
					genai.ModelGood,
					genai.ProviderOptionModalities{genai.ModalityAudio},
				},
				ErrGenSync:   "OpenAI Responses API does not support audio output as of December 2025; see https://platform.openai.com/docs/guides/audio",
				ErrGenStream: "OpenAI Responses API does not support audio output as of December 2025; see https://platform.openai.com/docs/guides/audio",
			},
			*/
		}
		f := func(t *testing.T, opts ...genai.ProviderOption) (genai.Provider, error) {
			return getClientInner(t, func(h http.RoundTripper) http.RoundTripper {
				return testRecorder.Record(t, h)
			}, opts...)
		}
		internaltest.TestClientProviderErrors(t, f, data)
	})
}

func TestPreviousResponseID(t *testing.T) {
	msgs := genai.Messages{genai.NewTextMessage("hello")}
	t.Run("wired", func(t *testing.T) {
		var req openairesponses.Response
		if err := req.Init(msgs, "gpt-4.1-nano", &openairesponses.GenOptionText{PreviousResponseID: "resp_abc123"}); err != nil {
			t.Fatal(err)
		}
		if req.PreviousResponseID != "resp_abc123" {
			t.Errorf("PreviousResponseID = %q, want %q", req.PreviousResponseID, "resp_abc123")
		}
		b, err := json.Marshal(&req)
		if err != nil {
			t.Fatal(err)
		}
		got := string(b)
		if !strings.Contains(got, `"previous_response_id":"resp_abc123"`) {
			t.Errorf("JSON missing previous_response_id: %s", got)
		}
	})
	t.Run("empty", func(t *testing.T) {
		var req openairesponses.Response
		if err := req.Init(msgs, "gpt-4.1-nano"); err != nil {
			t.Fatal(err)
		}
		if req.PreviousResponseID != "" {
			t.Errorf("PreviousResponseID = %q, want empty", req.PreviousResponseID)
		}
		b, err := json.Marshal(&req)
		if err != nil {
			t.Fatal(err)
		}
		got := string(b)
		if strings.Contains(got, "previous_response_id") {
			t.Errorf("JSON should omit empty previous_response_id: %s", got)
		}
	})
}

func init() {
	internal.BeLenient = false
}
