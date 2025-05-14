// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package anthropic_test

import (
	_ "embed"
	"os"
	"testing"

	"github.com/maruel/genai"
	"github.com/maruel/genai/anthropic"
	"github.com/maruel/genai/internal"
	"github.com/maruel/genai/internal/internaltest"
)

var testCases = &internaltest.TestCases{
	GetClient:    func(t *testing.T, m string) genai.ChatProvider { return getClient(t, m) },
	DefaultModel: "claude-3-haiku-20240307",
}

func TestClient_Chat_allModels(t *testing.T) {
	testCases.TestChatAllModels(t, nil)
}

func TestClient_Chat_thinking(t *testing.T) {
	// https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking
	testCases.TestChatThinking(t, "claude-3-7-sonnet-20250219")
}

func TestClient_ChatStream(t *testing.T) {
	testCases.TestChatStream(t, "", true)
}

func TestClient_Chat_vision_jPG_inline(t *testing.T) {
	// Using very small model for testing. As of March 2025,
	// claude-3-haiku-20240307 is 0.20$/1.25$ while claude-3-5-haiku-20241022 is
	// 0.80$/4.00$. 3.0 supports images, 3.5 supports PDFs.
	// https://docs.anthropic.com/en/docs/about-claude/models/all-models
	testCases.TestChatVisionJPGInline(t, "")
}

func TestClient_Chat_vision_pDF_inline(t *testing.T) {
	// 3.0 doesn't support PDFs.
	testCases.TestChatVisionPDFInline(t, "claude-3-5-haiku-20241022")
}

func TestClient_Chat_vision_pDF_uRL(t *testing.T) {
	testCases.TestChatVisionPDFURL(t, "claude-3-5-haiku-20241022")
}

func TestClient_Chat_tool_use(t *testing.T) {
	testCases.TestChatToolUseCountry(t, "", true)
}

func getClient(t *testing.T, m string) *anthropic.Client {
	testRecorder.Signal(t)
	if os.Getenv("ANTHROPIC_API_KEY") == "" {
		t.Skip("ANTHROPIC_API_KEY not set")
	}
	t.Parallel()
	c, err := anthropic.New("", m)
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
