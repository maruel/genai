// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package deepseek_test

import (
	"net/http"
	"os"
	"testing"

	"github.com/maruel/genai"
	"github.com/maruel/genai/base"
	"github.com/maruel/genai/internal"
	"github.com/maruel/genai/internal/internaltest"
	"github.com/maruel/genai/providers/deepseek"
	"github.com/maruel/genai/scoreboard/scoreboardtest"
)

func getClientRT(t testing.TB, model string, fn func(http.RoundTripper) http.RoundTripper) genai.Provider {
	apiKey := ""
	if os.Getenv("DEEPSEEK_API_KEY") == "" {
		apiKey = "<insert_api_key_here>"
	}
	c, err := deepseek.New(apiKey, model, fn)
	if err != nil {
		t.Fatal(err)
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
		{base.PreferredCheap, "deepseek-chat"},
		{base.PreferredGood, "deepseek-reasoner"},
		{base.PreferredSOTA, "deepseek-reasoner"},
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
			Model:        "deepseek-chat",
			ErrGenSync:   "http 401: error authentication_error: Authentication Fails, Your api key: ****iKey is invalid. You can get a new API key at https://platform.deepseek.com/api_keys",
			ErrGenStream: "http 401: error authentication_error: Authentication Fails, Your api key: ****iKey is invalid. You can get a new API key at https://platform.deepseek.com/api_keys",
			ErrListModel: "http 401: error authentication_error: Authentication Fails, Your api key: ****iKey is invalid. You can get a new API key at https://platform.deepseek.com/api_keys",
		},
		{
			Name:         "bad model",
			Model:        "bad model",
			ErrGenSync:   "http 400: error invalid_request_error: Model Not Exist",
			ErrGenStream: "http 400: error invalid_request_error: Model Not Exist",
		},
	}
	f := func(t *testing.T, apiKey, model string) genai.Provider {
		return getClientInner(t, apiKey, model)
	}
	internaltest.TestClient_Provider_errors(t, f, data)
}

func getClient(t *testing.T, m string) *deepseek.Client {
	t.Parallel()
	return getClientInner(t, "", m)
}

func getClientInner(t *testing.T, apiKey, m string) *deepseek.Client {
	if apiKey == "" && os.Getenv("DEEPSEEK_API_KEY") == "" {
		apiKey = "<insert_api_key_here>"
	}
	c, err := deepseek.New(apiKey, m, func(h http.RoundTripper) http.RoundTripper { return testRecorder.Record(t, h) })
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
