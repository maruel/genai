// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package openai_test

import (
	"bytes"
	_ "embed"
	"fmt"
	"os"
	"strings"
	"testing"

	"github.com/maruel/genai"
	"github.com/maruel/genai/internal"
	"github.com/maruel/genai/internal/internaltest"
	"github.com/maruel/genai/openai"
)

func TestClient_Chat_vision_and_JSON(t *testing.T) {
	c := getClient(t, model)
	msgs := genai.Messages{
		{
			Role: genai.User,
			Contents: []genai.Content{
				{Text: "Is it a banana? Reply as JSON."},
				{Filename: "banana.jpg", Document: bytes.NewReader(bananaJpg)},
			},
		},
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
	if resp.InputTokens != 295 || resp.OutputTokens != 6 {
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
	if resp.InputTokens != 78 || resp.OutputTokens != 15 {
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
	c := getClient(t, model)
	ctx := t.Context()
	msgs := genai.Messages{
		genai.NewTextMessage(genai.User, "Say hello. Use only one word."),
	}
	opts := genai.ChatOptions{
		Seed:        1,
		Temperature: 0.01,
		MaxTokens:   50,
	}
	chunks := make(chan genai.MessageFragment)
	end := make(chan genai.Message, 10)
	go func() {
		var pendingMsgs genai.Messages
		defer func() {
			for _, m := range pendingMsgs {
				end <- m
			}
			close(end)
		}()
		for {
			select {
			case <-ctx.Done():
				return
			case pkt, ok := <-chunks:
				if !ok {
					return
				}
				var err error
				if pendingMsgs, err = pkt.Accumulate(pendingMsgs); err != nil {
					end <- genai.NewTextMessage(genai.Assistant, fmt.Sprintf("Error: %v", err))
					return
				}
			}
		}
	}()
	err := c.ChatStream(ctx, msgs, &opts, chunks)
	close(chunks)
	var responses genai.Messages
	for m := range end {
		responses = append(responses, m)
	}
	t.Logf("Raw responses: %#v", responses)
	if err != nil {
		t.Fatal(err)
	}
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
