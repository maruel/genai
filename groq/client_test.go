// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package groq_test

import (
	_ "embed"
	"os"
	"strings"
	"testing"

	"github.com/maruel/genai"
	"github.com/maruel/genai/groq"
	"github.com/maruel/genai/internal"
	"github.com/maruel/genai/internal/internaltest"
)

var testCases = &internaltest.TestCases{
	Default: internaltest.Settings{
		GetClient: func(t *testing.T, m string) genai.ChatProvider { return getClient(t, m) },
		Model:     "llama3-8b-8192",
	},
}

func TestClient_Chat_allModels(t *testing.T) {
	testCases.TestChatAllModels(
		t,
		func(m genai.Model) bool {
			id := m.GetID()
			// Groq doesn't provide model metadata, so guess based on the name.
			return !(strings.Contains(id, "tts") || strings.Contains(id, "whisper") || strings.Contains(id, "llama-guard") || id == "mistral-saba-24b")
		})
}

func TestClient_Chat_thinking(t *testing.T) {
	testCases.TestChatThinking(t, &internaltest.Settings{Model: "qwen-qwq-32b"})
}

func TestClient_Chat_simple(t *testing.T) {
	testCases.TestChatSimple_simple(t, &internaltest.Settings{Model: "llama-3.1-8b-instant"})
}

func TestClient_ChatStream_simple(t *testing.T) {
	testCases.TestChatStream_simple(t, &internaltest.Settings{Model: "llama-3.1-8b-instant"})
}

func TestClient_max_tokens(t *testing.T) {
	testCases.TestChatMaxTokens(t, nil)
}

func TestClient_stop_sequence(t *testing.T) {
	testCases.TestChatStopSequence(t, nil)
}

func TestClient_Chat_vision_jPG_inline(t *testing.T) {
	testCases.TestChatVisionJPGInline(t, &internaltest.Settings{Model: "meta-llama/llama-4-scout-17b-16e-instruct"})
}

func TestClient_Chat_jSON(t *testing.T) {
	testCases.TestChatJSON(t, nil)
}

func TestClient_Chat_jSON_schema(t *testing.T) {
	t.Skip("Currently broken. To be investigated. See https://discord.com/channels/1207099205563457597/1207101178631159830/1371897729395064832")
	testCases.TestChatJSONSchema(t, &internaltest.Settings{Model: "gemma2-9b-it"})
}

func TestClient_Chat_tool_use_reply(t *testing.T) {
	testCases.TestChatToolUseReply(t, nil)
}

func TestClient_Chat_tool_use_position_bias(t *testing.T) {
	testCases.TestChatToolUsePositionBias(t, nil, false)
}

func TestClient_ChatProvider_errors(t *testing.T) {
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
		return getClientInner(t, apiKey, model)
	}
	internaltest.TestClient_ChatProvider_errors(t, f, data)
}

func TestClient_ModelProvider_errors(t *testing.T) {
	data := []internaltest.ModelProviderError{
		{
			Name:   "bad apiKey",
			ApiKey: "badApiKey",
			Err:    "http 401: error invalid_api_key (invalid_request_error): Invalid API Key. You can get a new API key at https://console.groq.com/keys",
		},
	}
	f := func(t *testing.T, apiKey string) genai.ModelProvider {
		return getClientInner(t, apiKey, "")
	}
	internaltest.TestClient_ModelProvider_errors(t, f, data)
}

func getClient(t *testing.T, m string) *groq.Client {
	testRecorder.Signal(t)
	t.Parallel()
	return getClientInner(t, "", m)
}

func getClientInner(t *testing.T, apiKey, m string) *groq.Client {
	if apiKey == "" && os.Getenv("GROQ_API_KEY") == "" {
		apiKey = "<insert_api_key_here>"
	}
	c, err := groq.New(apiKey, m, nil)
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
