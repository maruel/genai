// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package mistral_test

import (
	_ "embed"
	"os"
	"strings"
	"testing"

	"github.com/maruel/genai"
	"github.com/maruel/genai/internal"
	"github.com/maruel/genai/internal/internaltest"
	"github.com/maruel/genai/mistral"
)

func TestClient_Chat_allModels(t *testing.T) {
	internaltest.TestChatAllModels(
		t,
		func(t *testing.T, m string) genai.ChatProvider { return getClient(t, m) },
		func(m genai.Model) bool {
			model := m.(*mistral.Model)
			if !model.Capabilities.CompletionChat {
				return false
			}
			id := model.ID
			return id == "mistral-medium-2505"
		})
}

func TestClient_Chat_vision_and_JSON(t *testing.T) {
	internaltest.TestChatVisionJSON(t, func(t *testing.T) genai.ChatProvider { return getClient(t, "mistral-small-latest") })
}

func TestClient_Chat_vision_jPG_inline(t *testing.T) {
	internaltest.TestChatVisionJPGInline(t, func(t *testing.T) genai.ChatProvider { return getClient(t, "mistral-small-latest") })
}

func TestClient_Chat_vision_pDF_uRL(t *testing.T) {
	// Mistral does not support inline PDF.
	internaltest.TestChatVisionPDFURL(t, func(t *testing.T) genai.ChatProvider { return getClient(t, "mistral-small-latest") })
}

func TestClient_Chat_tool_use(t *testing.T) {
	internaltest.TestChatToolUseCountry(t, func(t *testing.T) genai.ChatProvider { return getClient(t, "ministral-3b-latest") })
}

func TestClient_ChatStream(t *testing.T) {
	msgs := genai.Messages{
		genai.NewTextMessage(genai.User, "Say hello. Use only one word."),
	}
	opts := genai.ChatOptions{
		Seed:        1,
		Temperature: 0.01,
		MaxTokens:   50,
	}
	responses := internaltest.ChatStream(t, func(t *testing.T) genai.ChatProvider { return getClient(t, "ministral-3b-latest") }, msgs, &opts)
	if len(responses) != 1 {
		t.Fatal("Unexpected response")
	}
	resp := responses[0]
	if len(resp.Contents) != 1 {
		t.Fatal("Unexpected response")
	}
	// Normalize some of the variance. Obviously many models will still fail this test.
	if got := strings.TrimRight(strings.TrimSpace(strings.ToLower(resp.Contents[0].Text)), ".!"); got != "hello" {
		t.Fatal(got)
	}
}

func getClient(t *testing.T, m string) *mistral.Client {
	testRecorder.Signal(t)
	if os.Getenv("MISTRAL_API_KEY") == "" {
		t.Skip("MISTRAL_API_KEY not set")
	}
	t.Parallel()
	c, err := mistral.New("", m)
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
