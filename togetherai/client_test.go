// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package togetherai_test

import (
	_ "embed"
	"os"
	"strings"
	"testing"

	"github.com/maruel/genai"
	"github.com/maruel/genai/internal"
	"github.com/maruel/genai/internal/internaltest"
	"github.com/maruel/genai/togetherai"
)

func TestClient_Scoreboard(t *testing.T) {
	f := func(m genai.Model) bool {
		model := m.(*togetherai.Model)
		if model.ID == "arcee-ai/maestro-reasoning" || // Requires CoT processing.
			model.ID == "google/gemma-2b-it" || // Doesn't follow instruction.
			model.ID == "deepseek-ai/DeepSeek-V3-p-dp" || // Causes HTTP 503.
			model.ID == "meta-llama/Llama-3.3-70B-Instruct-Turbo" || // rate_limit even if been a while.
			model.ID == "togethercomputer/Refuel-Llm-V2-Small" || // Fails because Seed option.
			strings.HasPrefix(model.ID, "deepseek-ai/DeepSeek-R1") || // Requires CoT processing.
			strings.HasPrefix(model.ID, "perplexity-ai/r1-") || // Requires CoT processing.
			strings.HasPrefix(model.ID, "Qwen/QwQ-32B") || // Requires CoT processing.
			strings.HasPrefix(model.ID, "Qwen/Qwen3-235B-A22B-") || // Requires CoT processing.
			strings.HasPrefix(model.ID, "togethercomputer/MoA-1") { // Causes HTTP 500.
			return false
		}
		return model.Type == "chat"
	}
	internaltest.TestScoreboard(t, func(t *testing.T, m string) genai.ProviderChat { return getClient(t, m) }, f)
}

var testCases = &internaltest.TestCases{
	Default: internaltest.Settings{
		GetClient: func(t *testing.T, m string) genai.ProviderChat { return getClient(t, m) },
		Model:     "meta-llama/Llama-4-Scout-17B-16E-Instruct",
	},
}

func TestClient_Chat_tool_use_position_bias(t *testing.T) {
	testCases.TestChatToolUsePositionBias(t, &internaltest.Settings{Model: "Qwen/Qwen2.5-7B-Instruct-Turbo"}, false)
}

func TestClient_ProviderChat_errors(t *testing.T) {
	data := []internaltest.ProviderChatError{
		{
			Name:          "bad apiKey",
			ApiKey:        "bad apiKey",
			Model:         "meta-llama/Llama-3.2-3B-Instruct-Turbo",
			ErrChat:       "http 401: error invalid_api_key (invalid_request_error): Invalid API key provided. You can find your API key at https://api.together.xyz/settings/api-keys.",
			ErrChatStream: "http 401: error invalid_api_key (invalid_request_error): Invalid API key provided. You can find your API key at https://api.together.xyz/settings/api-keys.",
		},
		{
			Name:          "bad model",
			Model:         "bad model",
			ErrChat:       "http 404: error model_not_available (invalid_request_error): Unable to access model bad model. Please visit https://api.together.ai/models to view the list of supported models.",
			ErrChatStream: "http 404: error model_not_available (invalid_request_error): Unable to access model bad model. Please visit https://api.together.ai/models to view the list of supported models.",
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
			Err:    "http 401: error (): Unauthorized. You can get a new API key at https://api.together.xyz/settings/api-keys",
		},
	}
	f := func(t *testing.T, apiKey string) genai.ProviderModel {
		return getClientInner(t, apiKey, "")
	}
	internaltest.TestClient_ProviderModel_errors(t, f, data)
}

func getClient(t *testing.T, m string) *togetherai.Client {
	testRecorder.Signal(t)
	t.Parallel()
	return getClientInner(t, "", m)
}

func getClientInner(t *testing.T, apiKey, m string) *togetherai.Client {
	if apiKey == "" && os.Getenv("TOGETHER_API_KEY") == "" {
		apiKey = "<insert_api_key_here>"
	}
	c, err := togetherai.New(apiKey, m, nil)
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
