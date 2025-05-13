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

func TestClient_Chat_allModels(t *testing.T) {
	internaltest.TestChatAllModels(t, func(t *testing.T, m string) genai.ChatProvider { return getClient(t, m) }, nil)
}

func TestClient_ChatStream(t *testing.T) {
	internaltest.TestChatStream(t, func(t *testing.T) genai.ChatProvider { return getClient(t, "llama-3.1-8b") }, true)
}

func TestClient_Chat_vision_jPG_inline(t *testing.T) {
	t.Skip("Implement multi-content messages")
	internaltest.TestChatVisionJPGInline(t, func(t *testing.T) genai.ChatProvider {
		return getClient(t, "meta-llama/llama-4-scout-17b-16e-instruct")
	})
}

func TestClient_Chat_jSON(t *testing.T) {
	internaltest.TestChatJSON(t, func(t *testing.T) genai.ChatProvider { return getClient(t, "llama-3.1-8b") }, true)
}

func TestClient_Chat_jSON_schema(t *testing.T) {
	internaltest.TestChatJSONSchema(t, func(t *testing.T) genai.ChatProvider { return getClient(t, "llama-3.1-8b") }, true)
}

func TestClient_Chat_tool_use(t *testing.T) {
	internaltest.TestChatToolUseCountry(t, func(t *testing.T) genai.ChatProvider { return getClient(t, "llama-3.1-8b") }, true)
}

func getClient(t *testing.T, m string) *cerebras.Client {
	testRecorder.Signal(t)
	if os.Getenv("CEREBRAS_API_KEY") == "" {
		t.Skip("CEREBRAS_API_KEY not set")
	}
	t.Parallel()
	c, err := cerebras.New("", m)
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
