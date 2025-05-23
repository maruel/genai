// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package cerebras_test

import (
	"os"
	"testing"

	"github.com/maruel/genai"
	"github.com/maruel/genai/cerebras"
	"github.com/maruel/genai/internal"
	"github.com/maruel/genai/internal/internaltest"
)

var testCases = &internaltest.TestCases{
	Default: internaltest.Settings{
		GetClient: func(t *testing.T, m string) genai.ChatProvider { return getClient(t, m) },
		Model:     "llama-3.1-8b",
	},
}

func TestClient_Chat_allModels(t *testing.T) {
	testCases.TestChatAllModels(t, func(model genai.Model) bool {
		// Skip it because it requires explicit processing and it's tested at TestClient_Chat_thinking below.
		return model.GetID() != "qwen-3-32b"
	})
}

func TestClient_Chat_thinking(t *testing.T) {
	testCases.TestChatThinking(t, &internaltest.Settings{
		GetClient: func(t *testing.T, m string) genai.ChatProvider {
			return &genai.ChatProviderThinking{Provider: getClient(t, m), TagName: "think"}
		},
		Model: "qwen-3-32b",
	})
}

func TestClient_ChatStream(t *testing.T) {
	testCases.TestChatStream(t, nil)
}

func TestClient_Chat_vision_jPG_inline(t *testing.T) {
	t.Skip("Implement multi-content messages")
	testCases.TestChatVisionJPGInline(t, &internaltest.Settings{Model: "llama-4-scout-17b-16e-instruct"})
}

func TestClient_Chat_jSON(t *testing.T) {
	testCases.TestChatJSON(t, nil)
}

func TestClient_Chat_jSON_schema(t *testing.T) {
	testCases.TestChatJSONSchema(t, nil)
}

func TestClient_Chat_tool_use_reply(t *testing.T) {
	testCases.TestChatToolUseReply(t, &internaltest.Settings{
		GetClient: func(t *testing.T, m string) genai.ChatProvider {
			return &genai.ChatProviderThinking{Provider: getClient(t, m), TagName: "think"}
		},
		Model: "qwen-3-32b",
	})
}

func TestClient_Chat_tool_use_position_bias(t *testing.T) {
	testCases.TestChatToolUsePositionBias(t, nil, false)
}

func TestClient_ChatProvider_errors(t *testing.T) {
	data := []internaltest.ChatProviderError{
		{
			Name:          "bad apiKey",
			ApiKey:        "bad apiKey",
			Model:         "llama-3.1-8b",
			ErrChat:       "http 401: error invalid_request_error/api_key/wrong_api_key: Wrong API Key. You can get a new API key at https://cloud.cerebras.ai/platform/",
			ErrChatStream: "http 401: error invalid_request_error/api_key/wrong_api_key: Wrong API Key. You can get a new API key at https://cloud.cerebras.ai/platform/",
		},
		{
			Name:          "bad model",
			Model:         "bad model",
			ErrChat:       "http 404: error not_found_error/model/model_not_found: Model bad model does not exist or you do not have access to it.",
			ErrChatStream: "http 404: error not_found_error/model/model_not_found: Model bad model does not exist or you do not have access to it.",
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
			Err:    "http 401: error invalid_request_error/api_key/wrong_api_key: Wrong API Key. You can get a new API key at https://cloud.cerebras.ai/platform/",
		},
	}
	f := func(t *testing.T, apiKey string) genai.ModelProvider {
		return getClientInner(t, apiKey, "")
	}
	internaltest.TestClient_ModelProvider_errors(t, f, data)
}

func getClient(t *testing.T, m string) *cerebras.Client {
	testRecorder.Signal(t)
	t.Parallel()
	return getClientInner(t, "", m)
}

func getClientInner(t *testing.T, apiKey, m string) *cerebras.Client {
	if apiKey == "" && os.Getenv("CEREBRAS_API_KEY") == "" {
		apiKey = "<insert_api_key_here>"
	}
	c, err := cerebras.New(apiKey, m)
	if err != nil {
		t.Fatal(err)
	}
	c.Client.Client.Transport = testRecorder.Record(t, c.Client.Client.Transport)
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
