// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package togetherai_test

import (
	"bytes"
	_ "embed"
	"os"
	"strings"
	"testing"

	"github.com/maruel/genai"
	"github.com/maruel/genai/internal"
	"github.com/maruel/genai/internal/internaltest"
	"github.com/maruel/genai/togetherai"
)

func TestClient_Chat_vision_and_JSON(t *testing.T) {
	c := getClient(t, "meta-llama/Llama-Vision-Free")
	// TogetherAI seems to require separate messages for text and images.
	msgs := genai.Messages{
		{
			Role: genai.User,
			Contents: []genai.Content{
				{Filename: "banana.jpg", Document: bytes.NewReader(bananaJpg)},
			},
		},
		genai.NewTextMessage(genai.User, "Is it a banana? Reply as JSON with the form {\"banana\": false} or {\"banana\": true}."),
	}
	var got struct {
		Banana bool `json:"banana"`
	}
	opts := genai.ChatOptions{
		Seed:        1,
		Temperature: 0.01,
		MaxTokens:   50,
		DecodeAs:    &got,
	}
	resp, err := c.Chat(t.Context(), msgs, &opts)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("Raw response: %#v", resp)
	if resp.InputTokens != 33 || resp.OutputTokens != 6 {
		t.Logf("Unexpected tokens usage: %v", resp.Usage)
	}
	if len(resp.Contents) != 1 {
		t.Fatal("Unexpected response")
	}
	if err := resp.Contents[0].Decode(&got); err != nil {
		t.Fatal(err)
	}
	if !got.Banana {
		t.Fatal(got.Banana)
	}
}

func TestClient_Chat_video(t *testing.T) {
	c := getClient(t, "Qwen/Qwen2.5-VL-72B-Instruct")
	f, err := os.Open("testdata/animation.mp4")
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	// TogetherAI seems to require separate messages for text and images.
	msgs := genai.Messages{
		genai.NewTextMessage(genai.User, "What is the word? Reply with exactly and only one word."),
		{Role: genai.User, Contents: []genai.Content{{Document: f}}},
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
	if resp.InputTokens != 3025 || resp.OutputTokens != 4 {
		t.Logf("Unexpected tokens usage: %v", resp.Usage)
	}
	if len(resp.Contents) != 1 {
		t.Fatal("Unexpected response")
	}
	if saw := strings.ToLower(resp.Contents[0].Text); saw != "banana" {
		t.Fatal(saw)
	}
}

func TestClient_Chat_tool_use(t *testing.T) {
	c := getClient(t, "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free")
	opts := genai.ChatOptions{
		Seed:        1,
		Temperature: 0.01,
		MaxTokens:   50,
	}
	internaltest.ChatToolUseCountry(t, c, &opts)
}

func TestClient_ChatStream(t *testing.T) {
	c := getClient(t, "meta-llama/Llama-3.2-3B-Instruct-Turbo")
	msgs := genai.Messages{
		genai.NewTextMessage(genai.User, "Say hello. Use only one word."),
	}
	opts := genai.ChatOptions{
		Seed:        1,
		Temperature: 0.01,
		MaxTokens:   50,
	}
	responses := internaltest.ChatStream(t, c, msgs, &opts)
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

func getClient(t *testing.T, m string) *togetherai.Client {
	if os.Getenv("TOGETHER_API_KEY") == "" {
		t.Skip("TOGETHER_API_KEY not set")
	}
	t.Parallel()
	c, err := togetherai.New("", m)
	if err != nil {
		t.Fatal(err)
	}
	c.Client.Client.Transport = internaltest.Record(t, c.Client.Client.Transport)
	return c
}

func init() {
	internal.BeLenient = false
}
