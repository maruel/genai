// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package perplexity_test

import (
	"net/http"
	"os"
	"testing"

	"github.com/maruel/genai"
	"github.com/maruel/genai/adapters"
	"github.com/maruel/genai/base"
	"github.com/maruel/genai/internal"
	"github.com/maruel/genai/internal/internaltest"
	"github.com/maruel/genai/providers/perplexity"
)

func TestClient_Scoreboard(t *testing.T) {
	internaltest.TestScoreboard(t, func(t *testing.T, m string) genai.ProviderGen {
		c := getClient(t, m)
		if m == "r1-1776" || m == "sonar-reasoning" {
			return &adapters.ProviderGenThinking{ProviderGen: c, TagName: "think"}
		}
		return c
	}, nil)
}

func TestClient_Preferred(t *testing.T) {
	data := []struct {
		name string
		want string
	}{
		{base.PreferredCheap, "sonar"},
		{base.PreferredGood, "sonar-pro"},
		{base.PreferredSOTA, "sonar-reasoning-pro"},
	}
	for _, line := range data {
		t.Run(line.name, func(t *testing.T) {
			if got := getClient(t, line.name).ModelID(); got != line.want {
				t.Fatalf("got model %q, want %q", got, line.want)
			}
		})
	}
}

func TestClient_ProviderGen_errors(t *testing.T) {
	data := []internaltest.ProviderGenError{
		{
			Name:         "bad apiKey",
			ApiKey:       "bad apiKey",
			Model:        "sonar",
			ErrGenSync:   "http 401: Unauthorized. You can get a new API key at https://www.perplexity.ai/settings/api",
			ErrGenStream: "http 401: Unauthorized. You can get a new API key at https://www.perplexity.ai/settings/api",
		},
		{
			Name:         "bad model",
			Model:        "bad model",
			ErrGenSync:   "http 400: error Invalid model 'bad model'. Permitted models can be found in the documentation at https://docs.perplexity.ai/guides/model-cards.",
			ErrGenStream: "http 400: error Invalid model 'bad model'. Permitted models can be found in the documentation at https://docs.perplexity.ai/guides/model-cards.",
		},
	}
	f := func(t *testing.T, apiKey, model string) genai.ProviderGen {
		return getClientInner(t, apiKey, model)
	}
	internaltest.TestClient_ProviderGen_errors(t, f, data)
}

func getClient(t *testing.T, m string) *perplexity.Client {
	testRecorder.Signal(t)
	t.Parallel()
	return getClientInner(t, "", m)
}

func getClientInner(t *testing.T, apiKey, m string) *perplexity.Client {
	if apiKey == "" && os.Getenv("PERPLEXITY_API_KEY") == "" {
		apiKey = "<insert_api_key_here>"
	}
	c, err := perplexity.New(apiKey, m, func(h http.RoundTripper) http.RoundTripper { return testRecorder.Record(t, h) })
	if err != nil {
		t.Fatal(err)
	}
	return c
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
