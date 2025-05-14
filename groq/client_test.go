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
	GetClient: func(t *testing.T, m string) genai.ChatProvider { return getClient(t, m) },
	Default: internaltest.Settings{
		Model: "llama3-8b-8192",
	},
}

func TestClient_Chat_allModels(t *testing.T) {
	testCases.TestChatAllModels(
		t,
		func(m genai.Model) bool {
			id := m.GetID()
			// Groq doesn't provide model metadata, so guess based on the name.
			return !(strings.Contains(id, "tts") || strings.Contains(id, "whisper") || strings.HasPrefix(id, "llama-guard") || id == "mistral-saba-24b")
		})
}

func TestClient_Chat_thinking(t *testing.T) {
	t.Skip("Currently broken. To be investigated. See https://discord.com/channels/1207099205563457597/1207101178631159830/1372301381612081163")
	testCases.TestChatThinking(t, &internaltest.Settings{Model: "qwen-qwq-32b"})
}

func TestClient_ChatStream(t *testing.T) {
	testCases.TestChatStream(t, &internaltest.Settings{Model: "llama-3.1-8b-instant"})
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

func TestClient_Chat_tool_use(t *testing.T) {
	testCases.TestChatToolUseCountry(t, nil)
}

func getClient(t *testing.T, m string) *groq.Client {
	testRecorder.Signal(t)
	if os.Getenv("GROQ_API_KEY") == "" {
		t.Skip("GROQ_API_KEY not set")
	}
	t.Parallel()
	c, err := groq.New("", m)
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
