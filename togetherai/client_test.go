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

var testCases = &internaltest.TestCases{
	Default: internaltest.Settings{
		GetClient: func(t *testing.T, m string) genai.ChatProvider { return getClient(t, m) },
		Model:     "meta-llama/Llama-4-Scout-17B-16E-Instruct",
	},
}

func TestClient_Chat_allModels(t *testing.T) {
	testCases.TestChatAllModels(
		t,
		func(m genai.Model) bool {
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
		})
}

func TestClient_Chat_thinking(t *testing.T) {
	t.Skip(`bugged when using streaming; https://discord.com/channels/1082503318624022589/1082503319165083700/1373293708438405170`)
	testCases.TestChatThinking(t, &internaltest.Settings{
		GetClient: func(t *testing.T, m string) genai.ChatProvider {
			return &genai.ThinkingChatProvider{Provider: getClient(t, m), TagName: "think"}
		},
		Model: "Qwen/Qwen3-235B-A22B-fp8-tput",
	})
}

// google/gemma-3-4b-it and google/gemma-3-27b-it are not available without a dedicated endpoint. We must
// select one of the Serverless at https://api.together.ai/models.

func TestClient_ChatStream(t *testing.T) {
	testCases.TestChatStream(t, &internaltest.Settings{Model: "meta-llama/Llama-3.2-3B-Instruct-Turbo"})
}

func TestClient_Chat_vision_jPG_inline(t *testing.T) {
	testCases.TestChatVisionJPGInline(t, nil)
}

func TestClient_Chat_jSON(t *testing.T) {
	testCases.TestChatJSON(t, nil)
}

func TestClient_Chat_jSON_schema(t *testing.T) {
	testCases.TestChatJSONSchema(t, nil)
}

func TestClient_Chat_video(t *testing.T) {
	c := getClient(t, "Qwen/Qwen2.5-VL-72B-Instruct")
	f, err := os.Open("testdata/animation.mp4")
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	// TogetherAI seems to require separate messages for text and images.
	msgs := genai.Messages{
		genai.NewTextMessage(genai.User, "What is the word? Reply with exactly and only one word."),
		{Role: genai.User, Contents: []genai.Content{{Document: f}}},
	}
	opts := genai.ChatOptions{
		Seed:        1,
		Temperature: 0.01,
		MaxTokens:   50,
	}
	resp, err := c.Chat(t.Context(), msgs, &opts)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("Raw response: %#v", resp)
	if resp.InputTokens != 3025 || resp.OutputTokens != 4 {
		t.Logf("Unexpected tokens usage: %v", resp.Usage)
	}
	if len(resp.Contents) != 1 {
		t.Fatal("Unexpected response")
	}
	if saw := strings.ToLower(resp.Contents[0].Text); saw != "banana" {
		t.Fatal(saw)
	}
}

func TestClient_Chat_tool_use_reply(t *testing.T) {
	testCases.TestChatToolUseReply(t, nil)
}

func TestClient_Chat_tool_use_position_bias(t *testing.T) {
	testCases.TestChatToolUsePositionBias(t, &internaltest.Settings{Model: "Qwen/Qwen2.5-7B-Instruct-Turbo"}, false)
}

func TestClient_ChatProvider_errors(t *testing.T) {
	data := []internaltest.ChatProviderError{
		{
			Name:   "bad apiKey",
			ApiKey: "bad apiKey",
			Model:  "meta-llama/Llama-3.2-3B-Instruct-Turbo",
			// ErrChat:       "TODO",
			ErrChatStream: "server error: error invalid_api_key (invalid_request_error): Invalid API key provided. You can find your API key at https://api.together.xyz/settings/api-keys.",
		},
		{
			Name:  "bad model",
			Model: "bad model",
			// ErrChat:       "TODO",
			ErrChatStream: "server error: error model_not_available (invalid_request_error): Unable to access model bad model. Please visit https://api.together.ai/models to view the list of supported models.",
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
			Err:    "json: cannot unmarshal object into Go value of type []togetherai.Model\nhttp 401\n{\"error\":{\"message\":\"Unauthorized\"}}",
		},
	}
	f := func(t *testing.T, apiKey string) genai.ModelProvider {
		return getClientInner(t, apiKey, "")
	}
	internaltest.TestClient_ModelProvider_errors(t, f, data)
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
	c, err := togetherai.New(apiKey, m)
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
