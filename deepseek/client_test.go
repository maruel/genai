// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package deepseek_test

import (
	"os"
	"testing"

	"github.com/maruel/genai"
	"github.com/maruel/genai/deepseek"
	"github.com/maruel/genai/internal"
	"github.com/maruel/genai/internal/internaltest"
)

var testCases = &internaltest.TestCases{
	GetClient:    func(t *testing.T, m string) genai.ChatProvider { return getClient(t, m) },
	DefaultModel: "deepseek-chat",
}

func TestClient_Chat_allModels(t *testing.T) {
	testCases.TestChatAllModels(t, nil)
}

func TestClient_Chat_thinking(t *testing.T) {
	testCases.TestChatThinking(t, "deepseek-reasoner")
}

func TestClient_ChatStream(t *testing.T) {
	testCases.TestChatStream(t, "", true)
}

func TestClient_Chat_jSON(t *testing.T) {
	t.Skip("Deep seek struggle to follow the requested JSON schema in the prompt. To be investigated.")
	testCases.TestChatJSON(t, "", true)
}

func TestClient_Chat_tool_use(t *testing.T) {
	testCases.TestChatToolUseCountry(t, "", true)
}

func getClient(t *testing.T, m string) *deepseek.Client {
	testRecorder.Signal(t)
	if os.Getenv("DEEPSEEK_API_KEY") == "" {
		t.Skip("DEEPSEEK_API_KEY not set")
	}
	t.Parallel()
	c, err := deepseek.New("", m)
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
