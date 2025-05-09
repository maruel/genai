// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package cohere_test

import (
	"os"
	"strings"
	"testing"

	"github.com/maruel/genai"
	"github.com/maruel/genai/cohere"
	"github.com/maruel/genai/internal"
	"github.com/maruel/genai/internal/internaltest"
)

func TestClient_AllModels(t *testing.T) {
	internaltest.TestAllModels(
		t,
		func(t *testing.T, m string) genai.ChatProvider { return getClient(t, m) },
		func(m genai.Model) bool {
			id := m.GetID()
			return strings.HasPrefix(id, "command-")
		})
}

func TestClient_Chat_jSON(t *testing.T) {
	c := getClient(t, "command-r-08-2024")
	msgs := genai.Messages{
		genai.NewTextMessage(genai.User, "Is a circle round? Reply as JSON."),
	}
	var got struct {
		Round bool `json:"round"`
	}
	opts := genai.ChatOptions{
		Seed:        1,
		Temperature: 0.01,
		MaxTokens:   50,
		DecodeAs:    got,
	}
	resp, err := c.Chat(t.Context(), msgs, &opts)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("Raw response: %#v", resp)
	if resp.InputTokens != 220 || resp.OutputTokens != 9 {
		t.Logf("Unexpected tokens usage: %v", resp.Usage)
	}
	if len(resp.Contents) != 1 {
		t.Fatalf("Unexpected response")
	}
	if err := resp.Contents[0].Decode(&got); err != nil {
		t.Fatal(err)
	}
	if !got.Round {
		t.Fatal("unexpected")
	}
}

func TestClient_Chat_tool_use(t *testing.T) {
	internaltest.TestChatToolUseCountry(t, func(t *testing.T) genai.ChatProvider { return getClient(t, "command-r-08-2024") })
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
	responses := internaltest.ChatStream(t, func(t *testing.T) genai.ChatProvider { return getClient(t, "command-r7b-12-2024") }, msgs, &opts)
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

func getClient(t *testing.T, m string) *cohere.Client {
	testRecorder.Signal(t)
	if os.Getenv("COHERE_API_KEY") == "" {
		t.Skip("COHERE_API_KEY not set")
	}
	t.Parallel()
	c, err := cohere.New("", m)
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
