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
	GetClient: func(t *testing.T, m string) genai.ChatProvider { return getClient(t, m) },
	Default: internaltest.Settings{
		Model: "meta-llama/Llama-4-Scout-17B-16E-Instruct",
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
	t.Skip(`would need to split manually "\n</think>\n\n"`)
	testCases.TestChatThinking(t, &internaltest.Settings{Model: "Qwen/Qwen3-235B-A22B-fp8-tput"})
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

func TestClient_Chat_tool_use(t *testing.T) {
	testCases.TestChatToolUseCountry(t, &internaltest.Settings{Model: "Qwen/Qwen2.5-7B-Instruct-Turbo"})
}

func getClient(t *testing.T, m string) *togetherai.Client {
	testRecorder.Signal(t)
	t.Parallel()
	apiKey := ""
	if os.Getenv("TOGETHER_API_KEY") == "" {
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
