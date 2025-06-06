// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package deepseek_test

import (
	"os"
	"testing"

	"github.com/maruel/genai"
	"github.com/maruel/genai/internal"
	"github.com/maruel/genai/internal/internaltest"
	"github.com/maruel/genai/providers/deepseek"
)

func TestClient_Scoreboard(t *testing.T) {
	internaltest.TestScoreboard(t, func(t *testing.T, m string) genai.ProviderGen { return getClient(t, m) }, nil)
}

func TestClient_ProviderGen_errors(t *testing.T) {
	data := []internaltest.ProviderGenError{
		{
			Name:         "bad apiKey",
			ApiKey:       "bad apiKey",
			Model:        "deepseek-chat",
			ErrGenSync:   "http 401: error authentication_error: Authentication Fails, Your api key: ****iKey is invalid. You can get a new API key at https://platform.deepseek.com/api_keys",
			ErrGenStream: "http 401: error authentication_error: Authentication Fails, Your api key: ****iKey is invalid. You can get a new API key at https://platform.deepseek.com/api_keys",
		},
		{
			Name:         "bad model",
			Model:        "bad model",
			ErrGenSync:   "http 400: error invalid_request_error: Model Not Exist",
			ErrGenStream: "http 400: error invalid_request_error: Model Not Exist",
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
			Err:    "http 401: error authentication_error: Authentication Fails, Your api key: ****iKey is invalid. You can get a new API key at https://platform.deepseek.com/api_keys",
		},
	}
	f := func(t *testing.T, apiKey string) genai.ProviderModel {
		return getClientInner(t, apiKey, "")
	}
	internaltest.TestClient_ProviderModel_errors(t, f, data)
}

func getClient(t *testing.T, m string) *deepseek.Client {
	testRecorder.Signal(t)
	t.Parallel()
	return getClientInner(t, "", m)
}

func getClientInner(t *testing.T, apiKey, m string) *deepseek.Client {
	if apiKey == "" && os.Getenv("DEEPSEEK_API_KEY") == "" {
		apiKey = "<insert_api_key_here>"
	}
	c, err := deepseek.New(apiKey, m, nil)
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
