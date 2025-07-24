// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package cerebras_test

import (
	"net/http"
	"os"
	"strings"
	"testing"

	"github.com/maruel/genai"
	"github.com/maruel/genai/adapters"
	"github.com/maruel/genai/base"
	"github.com/maruel/genai/internal"
	"github.com/maruel/genai/internal/internaltest"
	"github.com/maruel/genai/providers/cerebras"
	"github.com/maruel/genai/scoreboard/scoreboardtest"
)

func getClientRT(t testing.TB, model string, fn func(http.RoundTripper) http.RoundTripper) genai.Provider {
	apiKey := ""
	if os.Getenv("CEREBRAS_API_KEY") == "" {
		apiKey = "<insert_api_key_here>"
	}
	c, err2 := cerebras.New(apiKey, model, fn)
	if err2 != nil {
		t.Fatal(err2)
	}
	if strings.HasPrefix(model, "qwen") {
		return &adapters.ProviderGenThinking{
			ProviderGen: &adapters.ProviderGenAppend{
				ProviderGen: c,
				Append:      genai.NewTextMessage(genai.User, "/think"),
			},
			TagName: "think",
		}
	}
	return c
}

func TestClient_Scoreboard(t *testing.T) {
	models, err := getClient(t, "").ListModels(t.Context())
	if err != nil {
		t.Fatal(err)
	}
	scoreboardtest.TestClient_Scoreboard(t, getClientRT, models, testRecorder.Records)
}

func TestClient_Preferred(t *testing.T) {
	data := []struct {
		name string
		want string
	}{
		{base.PreferredCheap, "llama3.1-8b"},
		{base.PreferredGood, "llama-4-scout-17b-16e-instruct"},
		{base.PreferredSOTA, "qwen-3-32b"},
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
			Model:        "llama-3.1-8b",
			ErrGenSync:   "http 401: error invalid_request_error/api_key/wrong_api_key: Wrong API Key. You can get a new API key at https://cloud.cerebras.ai/platform/",
			ErrGenStream: "http 401: error invalid_request_error/api_key/wrong_api_key: Wrong API Key. You can get a new API key at https://cloud.cerebras.ai/platform/",
			ErrListModel: "http 401: error invalid_request_error/api_key/wrong_api_key: Wrong API Key. You can get a new API key at https://cloud.cerebras.ai/platform/",
		},
		{
			Name:         "bad model",
			Model:        "bad model",
			ErrGenSync:   "http 404: error not_found_error/model/model_not_found: Model bad model does not exist or you do not have access to it.",
			ErrGenStream: "http 404: error not_found_error/model/model_not_found: Model bad model does not exist or you do not have access to it.",
		},
	}
	f := func(t *testing.T, apiKey, model string) genai.Provider {
		return getClientInner(t, apiKey, model)
	}
	internaltest.TestClient_Provider_errors(t, f, data)
}

func getClient(t *testing.T, m string) *cerebras.Client {
	t.Parallel()
	return getClientInner(t, "", m)
}

func getClientInner(t *testing.T, apiKey, m string) *cerebras.Client {
	if apiKey == "" && os.Getenv("CEREBRAS_API_KEY") == "" {
		apiKey = "<insert_api_key_here>"
	}
	c, err := cerebras.New(apiKey, m, func(h http.RoundTripper) http.RoundTripper { return testRecorder.Record(t, h) })
	if err != nil {
		t.Fatal(err)
	}
	return c
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
