// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package anthropic_test

import (
	_ "embed"
	"os"
	"testing"

	"github.com/maruel/genai"
	"github.com/maruel/genai/anthropic"
	"github.com/maruel/genai/internal"
	"github.com/maruel/genai/internal/internaltest"
)

func TestClient_Scoreboard(t *testing.T) {
	internaltest.TestScoreboard(t, func(t *testing.T, m string) genai.ProviderGen { return getClient(t, m) }, nil)
}

func TestClient_ProviderGen_errors(t *testing.T) {
	data := []internaltest.ProviderGenError{
		{
			Name:          "bad apiKey",
			ApiKey:        "bad apiKey",
			Model:         "claude-3-haiku-20240307",
			ErrChat:       "http 401: error authentication_error: invalid x-api-key. You can get a new API key at https://console.anthropic.com/settings/keys",
			ErrChatStream: "http 401: error authentication_error: invalid x-api-key. You can get a new API key at https://console.anthropic.com/settings/keys",
		},
		{
			Name:          "bad model",
			Model:         "bad model",
			ErrChat:       "http 404: error not_found_error: model: bad model",
			ErrChatStream: "http 404: error not_found_error: model: bad model",
		},
	}
	f := func(t *testing.T, apiKey, model string) genai.ProviderGen {
		return getClientInner(t, apiKey, model)
	}
	internaltest.TestClient_ProviderGen_errors(t, f, data)
}

func TestClient_ProviderModel_errors(t *testing.T) {
	data := []internaltest.ProviderModelError{
		{
			Name:   "bad apiKey",
			ApiKey: "badApiKey",
			Err:    "http 401: error authentication_error: invalid x-api-key. You can get a new API key at https://console.anthropic.com/settings/keys",
		},
	}
	f := func(t *testing.T, apiKey string) genai.ProviderModel {
		return getClientInner(t, apiKey, "")
	}
	internaltest.TestClient_ProviderModel_errors(t, f, data)
}

func getClient(t *testing.T, m string) *anthropic.Client {
	testRecorder.Signal(t)
	t.Parallel()
	return getClientInner(t, "", m)
}

func getClientInner(t *testing.T, apiKey, m string) *anthropic.Client {
	if apiKey == "" && os.Getenv("ANTHROPIC_API_KEY") == "" {
		apiKey = "<insert_api_key_here>"
	}
	c, err := anthropic.New(apiKey, m, nil)
	if err != nil {
		t.Fatal(err)
	}
	c.ClientJSON.Client.Transport = testRecorder.Record(t, c.ClientJSON.Client.Transport)
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
