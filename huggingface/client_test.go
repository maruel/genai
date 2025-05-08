// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package huggingface_test

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/maruel/genai"
	"github.com/maruel/genai/huggingface"
	"github.com/maruel/genai/internal"
	"github.com/maruel/genai/internal/internaltest"
)

func TestClient_Chat(t *testing.T) {
	// TODO: Figure out why smaller models fail.
	c := getClient(t, "meta-llama/Llama-3.3-70B-Instruct")
	msgs := genai.Messages{
		genai.NewTextMessage(genai.User, "Say hello. Use only one word."),
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
	if resp.InputTokens != 43 || resp.OutputTokens != 3 {
		t.Logf("Unexpected tokens usage: %v", resp.Usage)
	}
	if len(resp.Contents) != 1 {
		t.Fatal("Unexpected response")
	}
	// Normalize some of the variance. Obviously many models will still fail this test.
	if got := strings.TrimRight(strings.TrimSpace(strings.ToLower(resp.Contents[0].Text)), ".!"); got != "hello" {
		t.Fatal(got)
	}
}

func TestClient_Chat_tool_use(t *testing.T) {
	// TODO: Figure out why smaller models fail.
	internaltest.TestChatToolUseCountry(t, func(t *testing.T) genai.ChatProvider { return getClient(t, "meta-llama/Llama-3.3-70B-Instruct") })
}

func TestClient_Chat_vision_and_JSON(t *testing.T) {
	t.Skip("llama 3.3 70B returns false instead of true as structured JSON (!?!)")
	opts := genai.ChatOptions{
		Seed:        1,
		Temperature: 0.01,
		MaxTokens:   50,
	}
	internaltest.TestChatVisionJSON(t, func(t *testing.T) genai.ChatProvider { return getClient(t, "meta-llama/Llama-3.3-70B-Instruct") }, &opts)
}

func TestClient_ChatStream(t *testing.T) {
	// TODO: Figure out why smaller models fail.
	msgs := genai.Messages{
		genai.NewTextMessage(genai.User, "Say hello. Use only one word."),
	}
	opts := genai.ChatOptions{
		Seed:        1,
		Temperature: 0.01,
		MaxTokens:   50,
	}
	responses := internaltest.ChatStream(t, func(t *testing.T) genai.ChatProvider { return getClient(t, "meta-llama/Llama-3.3-70B-Instruct") }, msgs, &opts)
	// TODO: handle t.Logf("HF currently tend to return spurious HTTP 422: %s", err)
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

func getClient(t *testing.T, m string) *huggingface.Client {
	testRecorder.Signal(t)
	if os.Getenv("HUGGINGFACE_API_KEY") == "" {
		// Fallback to loading from the python client's cache.
		h, err := os.UserHomeDir()
		if err != nil {
			t.Fatal("can't find home directory")
		}
		if _, err := os.Stat(filepath.Join(h, ".cache", "huggingface", "token")); err != nil {
			t.Skip("HUGGINGFACE_API_KEY not set and can't find ~/.cache/huggingface/token")
		}
	}
	t.Parallel()
	c, err := huggingface.New("", m)
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
