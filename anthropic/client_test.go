// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package anthropic_test

import (
	_ "embed"
	"os"
	"strings"
	"testing"

	"github.com/maruel/genai"
	"github.com/maruel/genai/anthropic"
	"github.com/maruel/genai/internal"
	"github.com/maruel/genai/internal/internaltest"
)

func TestClient_Chat_allModels(t *testing.T) {
	internaltest.TestChatAllModels(t, func(t *testing.T, m string) genai.ChatProvider { return getClient(t, m) }, nil)
}

func TestClient_Chat_vision_jPG(t *testing.T) {
	// Using very small model for testing. As of March 2025,
	// claude-3-haiku-20240307 is 0.20$/1.25$ while claude-3-5-haiku-20241022 is
	// 0.80$/4.00$. 3.0 supports images, 3.5 supports PDFs.
	// https://docs.anthropic.com/en/docs/about-claude/models/all-models
	internaltest.TestChatVisionJPG(t, func(t *testing.T) genai.ChatProvider { return getClient(t, "claude-3-haiku-20240307") })
}

func TestClient_Chat_vision_pdf(t *testing.T) {
	// 3.0 doesn't support PDFs.
	internaltest.TestChatVisionPDFInline(t, func(t *testing.T) genai.ChatProvider { return getClient(t, "claude-3-5-haiku-20241022") })
}

func TestClient_Chat_vision_pDF_uRL(t *testing.T) {
	internaltest.TestChatVisionPDFURL(t, func(t *testing.T) genai.ChatProvider { return getClient(t, "claude-3-5-haiku-20241022") })
}

func TestClient_Chat_tool_use(t *testing.T) {
	internaltest.TestChatToolUseCountry(t, func(t *testing.T) genai.ChatProvider { return getClient(t, "claude-3-haiku-20240307") })
}

func TestClient_ChatStream(t *testing.T) {
	msgs := genai.Messages{
		genai.NewTextMessage(genai.User, "Say hello. Use only one word."),
	}
	opts := genai.ChatOptions{
		Temperature: 0.01,
		MaxTokens:   50,
	}
	responses := internaltest.ChatStream(t, func(t *testing.T) genai.ChatProvider { return getClient(t, "claude-3-haiku-20240307") }, msgs, &opts)
	if len(responses) != 1 {
		t.Fatal("Unexpected response")
	}
	resp := responses[0]
	if len(resp.Contents) != 1 {
		t.Fatal("Unexpected response")
	}
	// Normalize some of the variance. Obviously many models will still fail this test.
	got := strings.TrimRight(strings.TrimSpace(strings.ToLower(resp.Contents[0].Text)), ".!")
	if got != "hello" {
		t.Fatal(got)
	}
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
