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

func TestClient_Scoreboard(t *testing.T) {
	internaltest.TestScoreboard(t, func(t *testing.T, m string) genai.ProviderChat {
		c := getClient(t, m)
		if m == "Qwen/QwQ-32B" {
			return &genai.ProviderChatThinking{ProviderChat: c, TagName: "think", SkipJSON: true}
		}
		return c
	}, nil)
}

var testCases = &internaltest.TestCases{
	Default: internaltest.Settings{
		GetClient: func(t *testing.T, m string) genai.ProviderChat { return getClient(t, m) },
		Model:     "meta-llama/Llama-3.3-70B-Instruct",
	},
}

func TestClient_Chat_tool_use_position_bias(t *testing.T) {
	t.Run("Chat", func(t *testing.T) {
		testCases.TestChatToolUsePositionBiasCore(t, nil, false, false)
	})
	t.Run("ChatStream", func(t *testing.T) {
		testCases.TestChatToolUsePositionBiasCore(t, &internaltest.Settings{FinishReasonIsBroken: true}, false, true)
	})
}

func TestClient_ProviderChat_errors(t *testing.T) {
	data := []internaltest.ProviderChatError{
		{
			Name:          "bad apiKey",
			ApiKey:        "bad apiKey",
			Model:         "Qwen/Qwen3-4B",
			ErrChat:       "http 401: Unauthorized. You can get a new API key at https://huggingface.co/settings/tokens",
			ErrChatStream: "http 401: Unauthorized. You can get a new API key at https://huggingface.co/settings/tokens",
		},
		{
			Name:          "bad model",
			Model:         "bad model",
			ErrChat:       "http 404: Not Found",
			ErrChatStream: "http 404: Not Found",
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
			Err:    "http 401: error Invalid credentials in Authorization header. You can get a new API key at https://huggingface.co/settings/tokens",
		},
	}
	f := func(t *testing.T, apiKey string) genai.ProviderModel {
		return getClientInner(t, apiKey, "")
	}
	internaltest.TestClient_ProviderModel_errors(t, f, data)
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
	c, err := huggingface.New(apiKey, m, nil)
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
