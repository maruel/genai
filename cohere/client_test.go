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
	c := getClient(t, "command-r-08-2024")
	msgs := genai.Messages{
		genai.NewTextMessage(genai.User, "I wonder if Canada is a better country than the US? Call the tool best_country to tell me which country is the best one."),
	}
	var got struct {
		Country string `json:"country" jsonschema:"enum=Canada,enum=USA"`
	}
	opts := genai.ChatOptions{
		Seed:        1,
		Temperature: 0.01,
		MaxTokens:   200,
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
	if resp.InputTokens != 925 || resp.OutputTokens != 96 {
		t.Logf("Unexpected tokens usage: %v", resp.Usage)
	}
	// Warning: when the model is undecided, it call both.
	if len(resp.ToolCalls) == 0 || resp.ToolCalls[0].Name != "best_country" {
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
	c := getClient(t, "command-r7b-12-2024")
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

func getClient(t *testing.T, m string) *cohere.Client {
	if os.Getenv("COHERE_API_KEY") == "" {
		t.Skip("COHERE_API_KEY not set")
	}
	c, err := cohere.New("", m)
	if err != nil {
		t.Fatal(err)
	}
	c.Client.Client.Transport = internaltest.Record(t, c.Client.Client.Transport)
	return c
}

func init() {
	internal.BeLenient = false
}
