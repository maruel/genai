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
	"github.com/maruel/genai/providers/openai/openaichat"
	"github.com/maruel/genai/providers/openai/openairesponses"
	"github.com/maruel/genai/scoreboard/scoreboardtest"
)

func getClientRT(t testing.TB, model scoreboardtest.Model, fn func(http.RoundTripper) http.RoundTripper) genai.Provider {
	apiKey := ""
	if os.Getenv("OPENAI_API_KEY") == "" {
		apiKey = "<insert_api_key_here>"
	}
	c, err := openairesponses.New(t.Context(), &genai.ProviderOptions{APIKey: apiKey, Model: model.Model}, fn)
	if err != nil {
		t.Fatal(err)
	}
	if model.Thinking {
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
					&openaichat.OptionsText{
						// This will lead to spurious HTTP 500 but it is 25% of the cost.
						ServiceTier: openaichat.ServiceTierFlex,
					},
				},
			}
		}
	}
	return c
}

func TestClient_Scoreboard(t *testing.T) {
	genaiModels, err := getClient(t, genai.ModelNone).ListModels(t.Context())
	if err != nil {
		t.Fatal(err)
	}
	var models []scoreboardtest.Model
	for _, m := range genaiModels {
		id := m.GetID()
		models = append(models, scoreboardtest.Model{Model: id, Thinking: strings.HasPrefix(id, "o") && !strings.Contains(id, "moderation")})
	}
	scoreboardtest.AssertScoreboard(t, getClientRT, models, testRecorder.Records)
}

/*
// This is a tricky test since batch operations can take up to 24h to complete.
func TestClient_Batch(t *testing.T) {
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
}
*/

func TestClient_Preferred(t *testing.T) {
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
				opts := genai.ProviderOptions{Model: line.name, OutputModalities: genai.Modalities{line.modality}}
				c, err := getClientInner(t, &opts)
				if err != nil {
					t.Fatal(err)
				}
				if got := c.ModelID(); got != line.want {
					t.Fatalf("got model %q, want %q", got, line.want)
				}
			})
		})
	}
}

func TestClient_Provider_errors(t *testing.T) {
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
			ErrGenSync:   "http 400\nThe requested model 'bad model' does not exist. (type: invalid_request_error, code: model_not_found)",
			ErrGenStream: "http 400\nThe requested model 'bad model' does not exist. (type: invalid_request_error, code: model_not_found)",
		},
	}
	f := func(t *testing.T, opts genai.ProviderOptions) (genai.Provider, error) {
		return getClientInner(t, &genai.ProviderOptions{APIKey: opts.APIKey, Model: opts.Model})
	}
	internaltest.TestClient_Provider_errors(t, f, data)
}

func getClient(t *testing.T, m string) *openairesponses.Client {
	t.Parallel()
	c, err := getClientInner(t, &genai.ProviderOptions{Model: m})
	if err != nil {
		t.Fatal(err)
	}
	return c
}

func getClientInner(t *testing.T, opts *genai.ProviderOptions) (*openairesponses.Client, error) {
	o := *opts
	if o.APIKey == "" && os.Getenv("OPENAI_API_KEY") == "" {
		o.APIKey = "<insert_api_key_here>"
	}
	return openairesponses.New(t.Context(), &o, func(h http.RoundTripper) http.RoundTripper { return testRecorder.Record(t, h) })
}

var testRecorder *internaltest.Records

func TestMain(m *testing.M) {
	testRecorder = internaltest.NewRecords()
	code := m.Run()
	os.Exit(max(code, testRecorder.Close()))
}

func init() {
	internal.BeLenient = false
}
