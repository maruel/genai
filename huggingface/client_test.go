// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package huggingface_test

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/maruel/genai"
	"github.com/maruel/genai/huggingface"
	"github.com/maruel/genai/internal"
	"github.com/maruel/genai/internal/internaltest"
)

var testCases = &internaltest.TestCases{
	Default: internaltest.Settings{
		GetClient: func(t *testing.T, m string) genai.ChatProvider { return getClient(t, m) },
		Model:     "meta-llama/Llama-3.3-70B-Instruct",
	},
}

func TestClient_Chat_allModels(t *testing.T) {
	testCases.TestChatAllModels(
		t,
		func(m genai.Model) bool {
			model := m.(*huggingface.Model)
			if model.PipelineTag != "text-generation" {
				return false
			}
			id := model.ID
			return id == "meta-llama/Llama-3.3-70B-Instruct"
		})
}

func TestClient_Chat_thinking(t *testing.T) {
	t.Skip(`would need to split manually "\n</think>\n\n"`)
	testCases.TestChatThinking(t, &internaltest.Settings{Model: "Qwen/QwQ-32B"})
}

func TestClient_ChatStream(t *testing.T) {
	// TODO: Figure out why smaller models fail.
	testCases.TestChatStream(t, nil)
}

func TestClient_Chat_jSON(t *testing.T) {
	t.Skip(`{"error":"Input validation error: grammar is not supported","error_type":"validation"}`)
	testCases.TestChatJSON(t, nil)
}

func TestClient_Chat_jSON_schema(t *testing.T) {
	testCases.TestChatJSONSchema(t, nil)
}

func TestClient_Chat_tool_use_reply(t *testing.T) {
	testCases.TestChatToolUseReply(t, nil)
}

func TestClient_Chat_tool_use_position_bias(t *testing.T) {
	// TODO: Figure out why smaller models fail.
	t.Run("Chat", func(t *testing.T) {
		testCases.TestChatToolUsePositionBiasCore(t, nil, false, false)
	})
	t.Run("ChatStream", func(t *testing.T) {
		testCases.TestChatToolUsePositionBiasCore(t, &internaltest.Settings{FinishReasonIsBroken: true}, false, true)
	})
}

func TestClient_ChatProvider_errors(t *testing.T) {
	data := []internaltest.ChatProviderError{
		{
			Name:          "bad apiKey",
			ApiKey:        "bad apiKey",
			Model:         "Qwen/Qwen3-4B",
			ErrChat:       "http 401: Unauthorized",
			ErrChatStream: "http 401: Unauthorized",
		},
		{
			Name:          "bad model",
			Model:         "bad model",
			ErrChat:       "http 404: Not Found",
			ErrChatStream: "http 404: Not Found",
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
			Err:    "http 401: error: Invalid credentials in Authorization header. You can get a new API key at https://huggingface.co/settings/tokens",
		},
	}
	f := func(t *testing.T, apiKey string) genai.ModelProvider {
		return getClientInner(t, apiKey, "")
	}
	internaltest.TestClient_ModelProvider_errors(t, f, data)
}

func getClient(t *testing.T, m string) *huggingface.Client {
	testRecorder.Signal(t)
	t.Parallel()
	return getClientInner(t, "", m)
}

func getClientInner(t *testing.T, apiKey, m string) *huggingface.Client {
	if apiKey == "" && os.Getenv("HUGGINGFACE_API_KEY") == "" {
		// Fallback to loading from the python client's cache.
		h, err := os.UserHomeDir()
		if err != nil {
			t.Fatal("can't find home directory")
		}
		if _, err := os.Stat(filepath.Join(h, ".cache", "huggingface", "token")); err != nil {
			apiKey = "<insert_api_key_here>"
		}
	}
	c, err := huggingface.New(apiKey, m)
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
