// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package anthropic_test

import (
	"bytes"
	_ "embed"
	"errors"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/maruel/genai"
	"github.com/maruel/genai/anthropic"
	"github.com/maruel/genai/internal"
	"github.com/maruel/genai/internal/internaltest"
	"github.com/maruel/httpjson"
)

func TestClient_Chat_vision(t *testing.T) {
	// Using very small model for testing. As of March 2025,
	// claude-3-haiku-20240307 is 0.20$/1.25$ while claude-3-5-haiku-20241022 is
	// 0.80$/4.00$. 3.0 supports images, 3.5 supports PDFs.
	// https://docs.anthropic.com/en/docs/about-claude/models/all-models
	c := getClient(t, "claude-3-haiku-20240307")
	msgs := genai.Messages{
		{
			Role: genai.User,
			Contents: []genai.Content{
				{Text: "Is it a banana? Reply with only one word."},
				{Filename: "banana.jpg", Document: bytes.NewReader(bananaJpg)},
			},
		},
	}
	opts := genai.ChatOptions{
		Temperature: 0.01,
		MaxTokens:   50,
	}
	for i := range 3 {
		resp, err := c.Chat(t.Context(), msgs, &opts)
		if err != nil {
			var herr *httpjson.Error
			// See https://docs.anthropic.com/en/api/errors#http-errors
			if errors.As(err, &herr) && herr.StatusCode == 529 && i != 2 {
				t.Log("retrying after 2s")
				time.Sleep(2 * time.Second)
				continue
			}
			t.Fatal(err)
		}
		t.Logf("Raw response: %#v", resp)
		if resp.InputTokens != 237 || resp.OutputTokens != 5 {
			t.Logf("Unexpected tokens usage: %v", resp.Usage)
		}
		if len(resp.Contents) != 1 {
			t.Fatal("Unexpected response")
		}
		// Normalize some of the variance. Obviously many models will still fail this test.
		txt := strings.TrimRight(strings.TrimSpace(strings.ToLower(resp.Contents[0].Text)), ".!")
		if txt != "yes" {
			t.Fatal(txt)
		}
		return
	}
	t.Fatal("too many retries")
}

func TestClient_Chat_pdf(t *testing.T) {
	// 3.0 doesn't support PDFs.
	c := getClient(t, "claude-3-5-haiku-20241022")
	f, err := os.Open("testdata/hidden_word.pdf")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err2 := f.Close(); err2 != nil {
			t.Fatal(err2)
		}
	})
	msgs := genai.Messages{
		{
			Role: genai.User,
			Contents: []genai.Content{
				{Text: "What is the word? Reply with only the word."},
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
	if resp.InputTokens != 1628 || resp.OutputTokens != 4 {
		t.Logf("Unexpected tokens usage: %v", resp.Usage)
	}
	if len(resp.Contents) != 1 {
		t.Fatal("Unexpected response")
	}
	if strings.ToLower(resp.Contents[0].Text) != "orange" {
		t.Fatal(resp.Contents[0].Text)
	}
}

func TestClient_Chat_tool_use(t *testing.T) {
	c := getClient(t, "claude-3-haiku-20240307")
	msgs := genai.Messages{
		genai.NewTextMessage(genai.User, "I wonder if Canada is a better country than the US? Call the tool best_country to tell me which country is the best one."),
	}
	var got struct {
		Country string `json:"country" jsonschema:"enum=Canada,enum=USA"`
	}
	opts := genai.ChatOptions{
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
	if resp.InputTokens != 483 || resp.OutputTokens != 38 {
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
	c := getClient(t, "claude-3-haiku-20240307")
	msgs := genai.Messages{
		genai.NewTextMessage(genai.User, "Say hello. Use only one word."),
	}
	opts := genai.ChatOptions{
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
	got := strings.TrimRight(strings.TrimSpace(strings.ToLower(resp.Contents[0].Text)), ".!")
	if got != "hello" {
		t.Fatal(got)
	}
}

func getClient(t *testing.T, m string) *anthropic.Client {
	if os.Getenv("ANTHROPIC_API_KEY") == "" {
		t.Skip("ANTHROPIC_API_KEY not set")
	}
	c, err := anthropic.New("", m)
	if err != nil {
		t.Fatal(err)
	}
	c.Client.Client.Transport = internaltest.Record(t, c.Client.Client.Transport)
	return c
}

func init() {
	internal.BeLenient = false
}
