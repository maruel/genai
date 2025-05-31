// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package mistral_test

import (
	_ "embed"
	"os"
	"testing"

	"github.com/maruel/genai"
	"github.com/maruel/genai/internal"
	"github.com/maruel/genai/internal/internaltest"
	"github.com/maruel/genai/mistral"
)

func TestClient_Scoreboard(t *testing.T) {
	internaltest.TestScoreboard(t, func(t *testing.T, m string) genai.ProviderChat { return getClient(t, m) }, nil)
}

var testCases = &internaltest.TestCases{
	Default: internaltest.Settings{
		GetClient: func(t *testing.T, m string) genai.ProviderChat { return getClient(t, m) },
		Model:     "mistral-small-latest",
	},
}

func TestClient_Chat_tool_use_position_bias(t *testing.T) {
	testCases.TestChatToolUsePositionBias(t, &internaltest.Settings{Model: "ministral-3b-latest"}, false)
}

func TestClient_ProviderChat_errors(t *testing.T) {
	data := []internaltest.ProviderChatError{
		{
			Name:          "bad apiKey",
			ApiKey:        "bad apiKey",
			Model:         "ministral-3b-2410",
			ErrChat:       "http 401: error Unauthorized. You can get a new API key at https://console.mistral.ai/api-keys",
			ErrChatStream: "http 401: error Unauthorized. You can get a new API key at https://console.mistral.ai/api-keys",
		},
		{
			Name:          "bad model",
			Model:         "bad model",
			ErrChat:       "http 400: error invalid_model: Invalid model: bad model",
			ErrChatStream: "http 400: error invalid_model: Invalid model: bad model",
		},
	}
	f := func(t *testing.T, apiKey, model string) genai.ProviderChat {
		return getClientInner(t, apiKey, model)
	}
	internaltest.TestClient_ProviderChat_errors(t, f, data)
}

func TestClient_ProviderModel_errors(t *testing.T) {
	data := []internaltest.ProviderModelError{
		{
			Name:   "bad apiKey",
			ApiKey: "badApiKey",
			Err:    "http 401: error Unauthorized. You can get a new API key at https://console.mistral.ai/api-keys",
		},
	}
	f := func(t *testing.T, apiKey string) genai.ProviderModel {
		return getClientInner(t, apiKey, "")
	}
	internaltest.TestClient_ProviderModel_errors(t, f, data)
}

func getClient(t *testing.T, m string) *mistral.Client {
	testRecorder.Signal(t)
	t.Parallel()
	return getClientInner(t, "", m)
}

func getClientInner(t *testing.T, apiKey, m string) *mistral.Client {
	if apiKey == "" && os.Getenv("MISTRAL_API_KEY") == "" {
		apiKey = "<insert_api_key_here>"
	}
	c, err := mistral.New(apiKey, m, nil)
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
