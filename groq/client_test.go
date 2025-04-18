// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package groq_test

import (
	"bytes"
	_ "embed"
	"os"
	"strings"
	"testing"

	"github.com/maruel/genai"
	"github.com/maruel/genai/groq"
	"github.com/maruel/genai/internal"
	"github.com/maruel/genai/internal/internaltest"
)

func TestClient_Chat_vision_and_JSON(t *testing.T) {
	c := getClient(t, "meta-llama/llama-4-scout-17b-16e-instruct")
	msgs := genai.Messages{
		{
			Role: genai.User,
			Contents: []genai.Content{
				{Text: "Is it a banana? Reply as JSON with the form {\"banana\": false} or {\"banana\": true}."},
				{Filename: "banana.jpg", Document: bytes.NewReader(bananaJpg)},
			},
		},
	}
	opts := genai.ChatOptions{
		Seed:        1,
		Temperature: 0.01,
		MaxTokens:   50,
		ReplyAsJSON: true,
	}
	resp, err := c.Chat(t.Context(), msgs, &opts)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("Raw response: %#v", resp)
	if resp.InputTokens != 59 || resp.OutputTokens != 8 {
		t.Logf("Unexpected tokens usage: %v", resp.Usage)
	}
	if len(resp.Contents) != 1 {
		t.Fatal("Unexpected response")
	}
	var got struct {
		Banana bool `json:"banana"`
	}
	if err := resp.Contents[0].Decode(&got); err != nil {
		t.Fatal(err)
	}
	if !got.Banana {
		t.Fatal("unexpected")
	}
}

func TestClient_Chat_tool_use(t *testing.T) {
	c := getClient(t, "llama-3.1-8b-instant")
	msgs := genai.Messages{
		genai.NewTextMessage(genai.User, "I wonder if Canada is a better country than the US? Call the tool best_country to tell me which country is the best one."),
	}
	var got struct {
		Country string `json:"country" jsonschema:"enum=Canada,enum=USA"`
	}
	opts := genai.ChatOptions{
		Seed:        1,
		Temperature: 0.01,
		MaxTokens:   50,
		Tools: []genai.ToolDef{
			{
				Name:        "best_country",
				Description: "A tool to determine the best country",
				InputsAs:    &got,
			},
		},
	}
	resp, err := c.Chat(t.Context(), msgs, &opts)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("Raw response: %#v", resp)
	if resp.InputTokens != 286 || resp.OutputTokens != 11 {
		t.Logf("Unexpected tokens usage: %v", resp.Usage)
	}
	if len(resp.ToolCalls) != 1 || resp.ToolCalls[0].Name != "best_country" {
		t.Fatal("Unexpected response")
	}
	if err := resp.ToolCalls[0].Decode(&got); err != nil {
		t.Fatal(err)
	}
	if got.Country != "Canada" {
		t.Fatal(got.Country)
	}
}

func TestClient_ChatStream(t *testing.T) {
	c := getClient(t, "llama-3.1-8b-instant")
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

func getClient(t *testing.T, m string) *groq.Client {
	if os.Getenv("GROQ_API_KEY") == "" {
		t.Skip("GROQ_API_KEY not set")
	}
	c, err := groq.New("", m)
	if err != nil {
		t.Fatal(err)
	}
	c.Client.Client.Transport = internaltest.Record(t, c.Client.Client.Transport)
	return c
}

func init() {
	internal.BeLenient = false
}
