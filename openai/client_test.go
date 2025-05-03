// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package openai_test

import (
	_ "embed"
	"os"
	"strings"
	"testing"

	"github.com/maruel/genai"
	"github.com/maruel/genai/internal"
	"github.com/maruel/genai/internal/internaltest"
	"github.com/maruel/genai/openai"
)

const model = "gpt-4.1-nano"

func TestClient_Chat_vision_and_JSON(t *testing.T) {
	c := getClient(t, model)
	opts := genai.ChatOptions{
		Seed:        1,
		Temperature: 0.01,
		MaxTokens:   50,
	}
	internaltest.ChatVisionJSON(t, func(t *testing.T) genai.ChatProvider { return c }, &opts)
}

func TestClient_Chat_audio(t *testing.T) {
	c := getClient(t, "gpt-4o-audio-preview")
	f, err := os.Open("testdata/mystery_word.mp3")
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	msgs := genai.Messages{
		{Role: genai.User, Contents: []genai.Content{{Document: f}}},
		genai.NewTextMessage(genai.User, "What is the word said? Reply with only the word."),
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
	if resp.InputTokens != 31 || resp.OutputTokens != 1 {
		t.Logf("Unexpected tokens usage: %v", resp.Usage)
	}
	if len(resp.Contents) != 1 {
		t.Fatal("Unexpected response")
	}
	if heard := strings.ToLower(resp.Contents[0].Text); heard != "orange" {
		t.Fatal(heard)
	}
}

func TestClient_Chat_pDF(t *testing.T) {
	c := getClient(t, model)
	f, err := os.Open("testdata/hidden_word.pdf")
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	msgs := genai.Messages{
		{
			Role: genai.User,
			Contents: []genai.Content{
				{Text: "What is the hidden word? Reply with only the word."},
				{Document: f},
			},
		},
	}
	opts := genai.ChatOptions{
		Temperature: 0.01,
		MaxTokens:   50,
	}
	resp, err := c.Chat(t.Context(), msgs, &opts)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("Raw response: %#v", resp)
	if resp.InputTokens != 238 || resp.OutputTokens != 2 {
		t.Logf("Unexpected tokens usage: %v", resp.Usage)
	}
	if len(resp.Contents) != 1 {
		t.Fatal("Unexpected response")
	}
	if got := strings.TrimSpace(strings.ToLower(resp.Contents[0].Text)); got != "orange" {
		t.Fatal(got)
	}
}

func TestClient_Chat_tool_use(t *testing.T) {
	c := getClient(t, model)
	opts := genai.ChatOptions{
		Seed:        1,
		Temperature: 0.01,
		MaxTokens:   200,
	}
	internaltest.ChatToolUseCountry(t, func(t *testing.T) genai.ChatProvider { return c }, &opts)
}

func TestClient_ChatStream(t *testing.T) {
	c := getClient(t, model)
	msgs := genai.Messages{
		genai.NewTextMessage(genai.User, "Say hello. Use only one word."),
	}
	opts := genai.ChatOptions{
		Seed:        1,
		Temperature: 0.01,
		MaxTokens:   50,
	}
	responses := internaltest.ChatStream(t, func(t *testing.T) genai.ChatProvider { return c }, msgs, &opts)
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

func getClient(t *testing.T, m string) *openai.Client {
	if os.Getenv("OPENAI_API_KEY") == "" {
		t.Skip("OPENAI_API_KEY not set")
	}
	t.Parallel()
	c, err := openai.New("", m)
	if err != nil {
		t.Fatal(err)
	}
	c.Client.Client.Transport = internaltest.Record(t, c.Client.Client.Transport)
	return c
}

func init() {
	internal.BeLenient = false
}
