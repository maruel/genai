// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package mistral_test

import (
	_ "embed"
	"os"
	"testing"

	"github.com/maruel/genai"
	"github.com/maruel/genai/internal"
	"github.com/maruel/genai/internal/internaltest"
	"github.com/maruel/genai/mistral"
)

var testCases = &internaltest.TestCases{
	Default: internaltest.Settings{
		GetClient: func(t *testing.T, m string) genai.ChatProvider { return getClient(t, m) },
		Model:     "mistral-small-latest",
	},
}

func TestClient_Chat_allModels(t *testing.T) {
	testCases.TestChatAllModels(
		t,
		func(m genai.Model) bool {
			model := m.(*mistral.Model)
			if !model.Capabilities.CompletionChat {
				return false
			}
			id := model.ID
			return id == "mistral-medium-2505"
		})
}

func TestClient_ChatStream(t *testing.T) {
	testCases.TestChatStream(t, &internaltest.Settings{Model: "ministral-3b-latest"})
}

func TestClient_Chat_jSON(t *testing.T) {
	testCases.TestChatJSON(t, nil)
}

func TestClient_Chat_jSON_schema(t *testing.T) {
	testCases.TestChatJSONSchema(t, nil)
}

func TestClient_Chat_vision_jPG_inline(t *testing.T) {
	testCases.TestChatVisionJPGInline(t, nil)
}

func TestClient_Chat_vision_pDF_uRL(t *testing.T) {
	// Mistral does not support inline PDF.
	testCases.TestChatVisionPDFURL(t, nil)
}

func TestClient_Chat_tool_use_reply(t *testing.T) {
	testCases.TestChatToolUseReply(t, nil)
}

func TestClient_Chat_tool_use_position_bias(t *testing.T) {
	testCases.TestChatToolUsePositionBias(t, &internaltest.Settings{Model: "ministral-3b-latest"}, false)
}

func getClient(t *testing.T, m string) *mistral.Client {
	testRecorder.Signal(t)
	t.Parallel()
	apiKey := ""
	if os.Getenv("MISTRAL_API_KEY") == "" {
		apiKey = "<insert_api_key_here>"
	}
	c, err := mistral.New(apiKey, m)
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
