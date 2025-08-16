// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package cohere_test

import (
	"net/http"
	"os"
	"testing"

	"github.com/maruel/genai"
	"github.com/maruel/genai/internal"
	"github.com/maruel/genai/internal/internaltest"
	"github.com/maruel/genai/providers/cohere"
	"github.com/maruel/genai/scoreboard/scoreboardtest"
)

func getClientRT(t testing.TB, model scoreboardtest.Model, fn func(http.RoundTripper) http.RoundTripper) genai.Provider {
	apiKey := ""
	if os.Getenv("COHERE_API_KEY") == "" {
		apiKey = "<insert_api_key_here>"
	}
	c, err := cohere.New(&genai.OptionsProvider{APIKey: apiKey, Model: model.Model}, fn)
	if err != nil {
		t.Fatal(err)
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
		// Hack.
		if id == "c4ai-aya-vision-8b" || id == "command-r7b-12-2024" {
			models = append(models, scoreboardtest.Model{Model: id, Thinking: true})
		} else {
			models = append(models, scoreboardtest.Model{Model: id})
		}
	}
	scoreboardtest.AssertScoreboard(t, getClientRT, models, testRecorder.Records)
}

func TestClient_Preferred(t *testing.T) {
	data := []struct {
		name string
		want string
	}{
		{genai.ModelCheap, "command-light"},
		{genai.ModelGood, "command-r7b-12-2024"},
		{genai.ModelSOTA, "command-a-03-2025"},
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
			Name:         "bad apiKey",
			APIKey:       "bad apiKey",
			Model:        "command-r7b-12-2024",
			ErrGenSync:   "http 401\ninvalid api token\nget a new API key at https://dashboard.cohere.com/api-keys",
			ErrGenStream: "http 401\ninvalid api token\nget a new API key at https://dashboard.cohere.com/api-keys",
			ErrListModel: "http 401\ninvalid api token\nget a new API key at https://dashboard.cohere.com/api-keys",
		},
		{
			Name:         "bad model",
			Model:        "bad model",
			ErrGenSync:   "http 404\nmodel 'bad model' not found, make sure the correct model ID was used and that you have access to the model.",
			ErrGenStream: "http 404\nmodel 'bad model' not found, make sure the correct model ID was used and that you have access to the model.",
		},
	}
	f := func(t *testing.T, apiKey, model string) (genai.Provider, error) {
		return getClientInner(t, apiKey, model)
	}
	internaltest.TestClient_Provider_errors(t, f, data)
}

func getClient(t *testing.T, m string) *cohere.Client {
	t.Parallel()
	c, err := getClientInner(t, "", m)
	if err != nil {
		t.Fatal(err)
	}
	return c
}

func getClientInner(t *testing.T, apiKey, m string) (*cohere.Client, error) {
	if apiKey == "" && os.Getenv("COHERE_API_KEY") == "" {
		apiKey = "<insert_api_key_here>"
	}
	return cohere.New(&genai.OptionsProvider{APIKey: apiKey, Model: m}, func(h http.RoundTripper) http.RoundTripper { return testRecorder.Record(t, h) })
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
