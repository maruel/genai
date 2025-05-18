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
	testCases.TestChatAllModels(t, nil)
}

func TestClient_Chat_thinking(t *testing.T) {
	testCases.TestChatThinking(t, &internaltest.Settings{
		GetClient: func(t *testing.T, m string) genai.ChatProvider {
			return &genai.ThinkingChatProvider{Provider: getClient(t, m), TagName: "think"}
		},
		Model: "qwen-3-32b",
	})
}

func TestClient_ChatStream(t *testing.T) {
	testCases.TestChatStream(t, nil)
}

func TestClient_Chat_vision_jPG_inline(t *testing.T) {
	t.Skip("Implement multi-content messages")
	testCases.TestChatVisionJPGInline(t, &internaltest.Settings{Model: "meta-llama/llama-4-scout-17b-16e-instruct"})
}

func TestClient_Chat_jSON(t *testing.T) {
	testCases.TestChatJSON(t, nil)
}

func TestClient_Chat_jSON_schema(t *testing.T) {
	testCases.TestChatJSONSchema(t, nil)
}

func TestClient_Chat_tool_use_position_bias(t *testing.T) {
	testCases.TestChatToolUsePositionBias(t, nil)
}

func getClient(t *testing.T, m string) *cerebras.Client {
	testRecorder.Signal(t)
	t.Parallel()
	apiKey := ""
	if os.Getenv("CEREBRAS_API_KEY") == "" {
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
