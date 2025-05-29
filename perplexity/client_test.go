// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package perplexity_test

import (
	"os"
	"testing"

	"github.com/maruel/genai"
	"github.com/maruel/genai/internal"
	"github.com/maruel/genai/internal/internaltest"
	"github.com/maruel/genai/perplexity"
)

var testCases = &internaltest.TestCases{
	Default: internaltest.Settings{
		GetClient: func(t *testing.T, m string) genai.ChatProvider { return getClient(t, m) },
		Model:     "r1-1776",
	},
}

// Not implementing TestClient_Chat_allModels since perplexity has no ListModels API.

func TestClient_Chat_simple(t *testing.T) {
	testCases.TestChatSimple_simple(t, &internaltest.Settings{
		GetClient: func(t *testing.T, m string) genai.ChatProvider {
			return &genai.ChatProviderThinking{ChatProvider: getClient(t, m), TagName: "think"}
		},
	})
}

func TestClient_ChatStream_simple(t *testing.T) {
	testCases.TestChatStream_simple(t, &internaltest.Settings{
		GetClient: func(t *testing.T, m string) genai.ChatProvider {
			return &genai.ChatProviderThinking{ChatProvider: getClient(t, m), TagName: "think"}
		},
	})
}

func TestClient_max_tokens(t *testing.T) {
	testCases.TestChatMaxTokens(t, nil)
}

func TestClient_stop_sequence(t *testing.T) {
	t.Skip("Way too verbose")
	testCases.TestChatStopSequence(t, nil)
}

func TestClient_ChatProvider_errors(t *testing.T) {
	data := []internaltest.ChatProviderError{
		{
			Name:          "bad apiKey",
			ApiKey:        "bad apiKey",
			Model:         "sonar",
			ErrChat:       "http 401: Unauthorized. You can get a new API key at https://www.perplexity.ai/settings/api",
			ErrChatStream: "http 401: Unauthorized. You can get a new API key at https://www.perplexity.ai/settings/api",
		},
		{
			Name:          "bad model",
			Model:         "bad model",
			ErrChat:       "http 400: error Invalid model 'bad model'. Permitted models can be found in the documentation at https://docs.perplexity.ai/guides/model-cards.",
			ErrChatStream: "http 400: error Invalid model 'bad model'. Permitted models can be found in the documentation at https://docs.perplexity.ai/guides/model-cards.",
		},
	}
	f := func(t *testing.T, apiKey, model string) genai.ChatProvider {
		return getClientInner(t, apiKey, model)
	}
	internaltest.TestClient_ChatProvider_errors(t, f, data)
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
	c, err := perplexity.New(apiKey, m, nil)
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
	testRecorder.Close()
	os.Exit(code)
}

func init() {
	internal.BeLenient = false
}
