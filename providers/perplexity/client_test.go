// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package perplexity_test

import (
	"context"
	"iter"
	"net/http"
	"os"
	"strings"
	"testing"

	"github.com/maruel/genai"
	"github.com/maruel/genai/adapters"
	"github.com/maruel/genai/internal"
	"github.com/maruel/genai/internal/internaltest"
	"github.com/maruel/genai/providers/perplexity"
	"github.com/maruel/genai/scoreboard/scoreboardtest"
	"github.com/maruel/roundtrippers"
)

func getClientRT(t testing.TB, model scoreboardtest.Model, fn func(http.RoundTripper) http.RoundTripper) genai.Provider {
	apiKey := ""
	if os.Getenv("PERPLEXITY_API_KEY") == "" {
		apiKey = "<insert_api_key_here>"
	}
	c, err := perplexity.New(t.Context(), &genai.ProviderOptions{APIKey: apiKey, Model: model.Model}, func(h http.RoundTripper) http.RoundTripper {
		// Perplexity is quick to ban users. It first start with 429 and then Cloudflare blocks it with a
		// javascript challenge. It's extra dumb because it is an API endpoint.
		// https://docs.perplexity.ai/guides/usage-tiers
		qps := 48. / 60.
		if strings.Contains(model.Model, "deep") {
			// Assume Tier 0 with is 5 RPM. For this test to succeed, it must be run with "go test -timeout 1h"
			qps = 4.8 / 60.
		}
		h = &roundtrippers.Throttle{QPS: qps, Transport: h}
		if fn != nil {
			h = fn(h)
		}
		return h
	})
	if err != nil {
		t.Fatal(err)
	}
	// Save on costs when running the smoke test.
	var p genai.Provider = &injectOptions{
		Provider: c,
		Opts:     []genai.Options{&perplexity.Options{DisableSearch: true, DisableRelatedQuestions: true}},
	}
	if model.Thinking {
		p = &adapters.ProviderThinking{Provider: p, ThinkingTokenStart: "<think>", ThinkingTokenEnd: "</think>"}
	}
	return p
}

// injectOptions generally inject the option unless "Quackiland" is in the last message.
type injectOptions struct {
	genai.Provider
	Opts []genai.Options
}

func (i *injectOptions) Unwrap() genai.Provider {
	return i.Provider
}

func (i *injectOptions) GenSync(ctx context.Context, msgs genai.Messages, opts ...genai.Options) (genai.Result, error) {
	if !strings.Contains(msgs[len(msgs)-1].Requests[len(msgs[0].Requests)-1].Text, "Quackiland") {
		opts = append(opts, i.Opts...)
	}
	return i.Provider.GenSync(ctx, msgs, opts...)
}

func (i *injectOptions) GenStream(ctx context.Context, msgs genai.Messages, opts ...genai.Options) (iter.Seq[genai.ReplyFragment], func() (genai.Result, error)) {
	if !strings.Contains(msgs[len(msgs)-1].Requests[len(msgs[0].Requests)-1].Text, "Quackiland") {
		opts = append(opts, i.Opts...)
	}
	return i.Provider.GenStream(ctx, msgs, opts...)
}

func TestClient_Scoreboard(t *testing.T) {
	// Perplexity doesn't support listing models. See https://docs.perplexity.ai/api-reference
	sb := getClient(t, genai.ModelNone).Scoreboard()
	var models []scoreboardtest.Model
	for _, sc := range sb.Scenarios {
		for _, model := range sc.Models {
			models = append(models, scoreboardtest.Model{Model: model, Thinking: sc.Thinking})
		}
	}
	scoreboardtest.AssertScoreboard(t, getClientRT, models, testRecorder.Records)
}

func TestClient_Preferred(t *testing.T) {
	data := []struct {
		name string
		want string
	}{
		{genai.ModelCheap, "sonar"},
		{genai.ModelGood, "sonar-pro"},
		{genai.ModelSOTA, "sonar-reasoning-pro"},
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
				Model:  "sonar",
			},
			// It returns an HTML page...
			ErrGenSync:   "http 401\nget a new API key at https://www.perplexity.ai/settings/api",
			ErrGenStream: "http 401\nget a new API key at https://www.perplexity.ai/settings/api",
		},
		{
			Name: "bad model",
			Opts: genai.ProviderOptions{
				Model: "bad model",
			},
			ErrGenSync:   "http 400\ninvalid_model (400): Invalid model 'bad model'. Permitted models can be found in the documentation at https://docs.perplexity.ai/guides/model-cards.",
			ErrGenStream: "http 400\ninvalid_model (400): Invalid model 'bad model'. Permitted models can be found in the documentation at https://docs.perplexity.ai/guides/model-cards.",
		},
	}
	f := func(t *testing.T, opts genai.ProviderOptions) (genai.Provider, error) {
		return getClientInner(t, opts.APIKey, opts.Model)
	}
	internaltest.TestClient_Provider_errors(t, f, data)
}

func getClient(t *testing.T, m string) *perplexity.Client {
	t.Parallel()
	c, err := getClientInner(t, "", m)
	if err != nil {
		t.Fatal(err)
	}
	return c
}

func getClientInner(t *testing.T, apiKey, m string) (*perplexity.Client, error) {
	if apiKey == "" && os.Getenv("PERPLEXITY_API_KEY") == "" {
		apiKey = "<insert_api_key_here>"
	}
	return perplexity.New(t.Context(), &genai.ProviderOptions{APIKey: apiKey, Model: m}, func(h http.RoundTripper) http.RoundTripper { return testRecorder.Record(t, h) })
}

var testRecorder *internaltest.Records

func TestMain(m *testing.M) {
	testRecorder = internaltest.NewRecords()
	code := m.Run()
	testRecorder.Close()
	os.Exit(code)
}

func init() {
	internal.BeLenient = false
}
