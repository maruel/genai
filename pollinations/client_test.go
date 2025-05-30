// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package pollinations_test

import (
	_ "embed"
	"os"
	"testing"

	"github.com/maruel/genai"
	"github.com/maruel/genai/internal"
	"github.com/maruel/genai/internal/internaltest"
	"github.com/maruel/genai/pollinations"
)

func TestClient_Scoreboard(t *testing.T) {
	internaltest.TestScoreboard(t, func(t *testing.T, m string) genai.ChatProvider { return getClient(t, m) }, nil)
}

var testCases = &internaltest.TestCases{
	Default: internaltest.Settings{
		GetClient: func(t *testing.T, m string) genai.ChatProvider { return getClient(t, m) },
		Model:     "llama-scout",
	},
}

func TestClient_Chat_tool_use_position_bias(t *testing.T) {
	t.Run("Chat", func(t *testing.T) {
		testCases.TestChatToolUsePositionBiasCore(t, nil, false, false)
	})
	t.Run("ChatStream", func(t *testing.T) {
		testCases.TestChatToolUsePositionBiasCore(t, &internaltest.Settings{UsageIsBroken: true}, false, true)
	})
}

func TestClient_ChatProvider_errors(t *testing.T) {
	t.Skip("TODO")
	data := []internaltest.ChatProviderError{
		{
			Name:          "bad apiKey",
			ApiKey:        "bad apiKey",
			Model:         "llama3-8b-8192",
			ErrChat:       "http 401: error invalid_api_key (invalid_request_error): Invalid API Key. You can get a new API key at https://console.groq.com/keys",
			ErrChatStream: "http 401: error invalid_api_key (invalid_request_error): Invalid API Key. You can get a new API key at https://console.groq.com/keys",
		},
		{
			Name:          "bad model",
			Model:         "bad model",
			ErrChat:       "http 404: error model_not_found (invalid_request_error): The model `bad model` does not exist or you do not have access to it.",
			ErrChatStream: "http 404: error model_not_found (invalid_request_error): The model `bad model` does not exist or you do not have access to it.",
		},
	}
	f := func(t *testing.T, apiKey, model string) genai.ChatProvider {
		return getClientInner(t, model)
	}
	internaltest.TestClient_ChatProvider_errors(t, f, data)
}

func getClient(t *testing.T, m string) *pollinations.Client {
	testRecorder.Signal(t)
	t.Parallel()
	return getClientInner(t, m)
}

func getClientInner(t *testing.T, m string) *pollinations.Client {
	c, err := pollinations.New("genai-unittests", m, nil)
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
