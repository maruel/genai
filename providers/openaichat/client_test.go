// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package openaichat_test

import (
	"context"
	_ "embed"
	"iter"
	"net/http"
	"os"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/maruel/genai"
	"github.com/maruel/genai/internal"
	"github.com/maruel/genai/internal/internaltest"
	"github.com/maruel/genai/providers/openaichat"
	"github.com/maruel/genai/scoreboard"
	"github.com/maruel/genai/smoke/smoketest"
)

func getClientInner(t *testing.T, opts genai.ProviderOptions, fn func(http.RoundTripper) http.RoundTripper) (genai.Provider, error) {
	if opts.APIKey == "" && os.Getenv("OPENAI_API_KEY") == "" {
		opts.APIKey = "<insert_api_key_here>"
	}
	return openaichat.New(t.Context(), &opts, fn)
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
		internaltest.TestCapabilities(t, getClient(t, "gpt-4.1"))
	})

	t.Run("GenAsync-Text", func(t *testing.T) {
		internaltest.TestCapabilitiesGenAsync(t, getClient(t, genai.ModelCheap))
	})

	t.Run("Caching-Text", func(t *testing.T) {
		internaltest.TestCapabilitiesCaching(t, getClient(t, genai.ModelCheap))
	})

	t.Run("Scoreboard", func(t *testing.T) {
		genaiModels, err := getClient(t, genai.ModelNone).ListModels(t.Context())
		if err != nil {
			t.Fatal(err)
		}
		var models []scoreboard.Model
		for _, m := range genaiModels {
			id := m.GetID()
			models = append(models, scoreboard.Model{Model: id, Reason: strings.HasPrefix(id, "o") && !strings.Contains(id, "moderation")})
		}
		getClientRT := func(t testing.TB, model scoreboard.Model, fn func(http.RoundTripper) http.RoundTripper) genai.Provider {
			opts := genai.ProviderOptions{Model: model.Model, PreloadedModels: cachedModels}
			if os.Getenv("OPENAI_API_KEY") == "" {
				opts.APIKey = "<insert_api_key_here>"
			}
			c, err := openaichat.New(t.Context(), &opts, fn)
			if err != nil {
				t.Fatal(err)
			}
			// This will lead to spurious HTTP 500 but it is 25% of the cost.
			tier := openaichat.ServiceTierFlex
			if strings.Contains(model.Model, "-chat-latest") {
				tier = openaichat.ServiceTierDefault
			}
			if model.Reason {
				return &injectReasoning{
					Provider: &internaltest.InjectOptions{
						Provider: c,
						Opts: []genai.Options{
							&openaichat.OptionsText{
								ReasoningEffort: openaichat.ReasoningEffortLow,
								ServiceTier:     tier,
							},
						},
					},
				}
			}
			if slices.Equal(c.OutputModalities(), []genai.Modality{genai.ModalityText}) {
				// See https://platform.openai.com/docs/guides/flex-processing
				if id := c.ModelID(); id == "o3" || id == "o4-mini" || strings.HasPrefix(id, "gpt-5") {
					return &internaltest.InjectOptions{
						Provider: c,
						Opts:     []genai.Options{&openaichat.OptionsText{ServiceTier: tier}},
					}
				}
			}
			return c
		}
		smoketest.Run(t, getClientRT, models, testRecorder.Records)
	})

	t.Run("Batch", func(t *testing.T) {
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
		isRecording := os.Getenv("RECORD") == "1"
		for {
			res, err := c.PokeResult(ctx, job)
			if err != nil {
				t.Fatal(err)
			}
			if res.Usage.FinishReason == genai.Pending {
				if isRecording {
					t.Logf("Waiting...")
					time.Sleep(time.Second)
				}
				continue
			}
			if res.Usage.InputTokens == 0 || res.Usage.OutputTokens == 0 {
				t.Error("expected usage")
			}
			if res.Usage.FinishReason != genai.FinishedStop {
				t.Errorf("finish reason: %s", res.Usage.FinishReason)
			}
			if s := res.String(); len(s) < 15 {
				t.Errorf("not enough text: %q", s)
			}
			break
		}
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

	t.Run("errors", func(t *testing.T) {
		data := []internaltest.ProviderError{
			{
				Name: "bad apiKey",
				Opts: genai.ProviderOptions{
					APIKey: "bad apiKey",
					Model:  "gpt-4.1-nano",
				},
				ErrGenSync:   "http 401\ninvalid_request_error/invalid_api_key: Incorrect API key provided: bad apiKey. You can find your API key at https://platform.openai.com/account/api-keys.",
				ErrGenStream: "http 401\ninvalid_request_error/invalid_api_key: Incorrect API key provided: bad apiKey. You can find your API key at https://platform.openai.com/account/api-keys.",
				ErrListModel: "http 401\ninvalid_request_error/invalid_api_key: Incorrect API key provided: bad apiKey. You can find your API key at https://platform.openai.com/account/api-keys.",
			},
			{
				Name: "bad apiKey image",
				Opts: genai.ProviderOptions{
					APIKey:           "bad apiKey",
					Model:            "gpt-image-1",
					OutputModalities: genai.Modalities{genai.ModalityImage},
				},
				ErrGenSync:   "http 401\ninvalid_request_error/invalid_api_key: Incorrect API key provided: bad apiKey. You can find your API key at https://platform.openai.com/account/api-keys.",
				ErrGenStream: "http 401\ninvalid_request_error/invalid_api_key: Incorrect API key provided: bad apiKey. You can find your API key at https://platform.openai.com/account/api-keys.",
			},
			{
				Name: "bad model",
				Opts: genai.ProviderOptions{
					Model: "bad model",
				},
				ErrGenSync:   "http 400\ninvalid_request_error: invalid model ID",
				ErrGenStream: "http 400\ninvalid_request_error: invalid model ID",
			},
			{
				Name: "bad model image",
				Opts: genai.ProviderOptions{
					Model:            "bad model",
					OutputModalities: genai.Modalities{genai.ModalityImage},
				},
				ErrGenSync:   "http 400\ninvalid_request_error/invalid_value for \"model\": Invalid value: 'bad model'. Supported values are: 'gpt-image-1', 'gpt-image-1-io', 'gpt-image-0721-mini-alpha', 'dall-e-2', and 'dall-e-3'.",
				ErrGenStream: "http 400\ninvalid_request_error/invalid_value for \"model\": Invalid value: 'bad model'. Supported values are: 'gpt-image-1', 'gpt-image-1-io', 'gpt-image-0721-mini-alpha', 'dall-e-2', and 'dall-e-3'.",
			},
		}
		f := func(t *testing.T, opts genai.ProviderOptions) (genai.Provider, error) {
			return getClientInner(t, opts, func(h http.RoundTripper) http.RoundTripper {
				return testRecorder.Record(t, h)
			})
		}
		internaltest.TestClientProviderErrors(t, f, data)
	})
}

// OpenAI returns the count of reasoning tokens but never return them. Duh. This messes up the scoreboard so
// inject fake reasoning whitespace.
type injectReasoning struct {
	genai.Provider
}

func (i *injectReasoning) GenSync(ctx context.Context, msgs genai.Messages, opts ...genai.Options) (genai.Result, error) {
	res, err := i.Provider.GenSync(ctx, msgs, opts...)
	if res.Usage.ReasoningTokens > 0 {
		res.Replies = append(res.Replies, genai.Reply{Reasoning: "\n"})
	}
	return res, err
}

func (i *injectReasoning) GenStream(ctx context.Context, msgs genai.Messages, opts ...genai.Options) (iter.Seq[genai.Reply], func() (genai.Result, error)) {
	res, err := i.Provider.GenStream(ctx, msgs, opts...)
	return res, err
}

func init() {
	internal.BeLenient = false
}
