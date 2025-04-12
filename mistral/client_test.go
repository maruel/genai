// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package mistral_test

import (
	"bytes"
	_ "embed"
	"errors"
	"fmt"
	"net/http"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/maruel/genai"
	"github.com/maruel/genai/internal"
	"github.com/maruel/genai/internal/internaltest"
	"github.com/maruel/genai/mistral"
	"github.com/maruel/httpjson"
)

func TestClient_Chat_vision_and_JSON(t *testing.T) {
	c := getClient(t, "mistral-small-latest")
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
	var err error
	var resp genai.ChatResult
	for i := range 3 {
		// Mistral has a very good rate limiting implementation.
		if resp, err = c.Chat(t.Context(), msgs, &opts); err != nil && i != 2 {
			var herr *httpjson.Error
			if errors.As(err, &herr) {
				if herr.StatusCode == http.StatusTooManyRequests {
					fmt.Fprintf(os.Stderr, "Rate limited, waiting 2s\n")
					time.Sleep(2 * time.Second)
					continue
				}
			}
			t.Fatal(err)
		}
		break
	}
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("Raw response: %#v", resp)
	if resp.InputTokens != 67 || resp.OutputTokens != 9 {
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

func TestClient_Chat_pDF(t *testing.T) {
	c := getClient(t, "mistral-small-latest")
	msgs := genai.Messages{
		{
			Role: genai.User,
			Contents: []genai.Content{
				{Text: "What is the word? Reply with only the word."},
				{URL: "https://raw.githubusercontent.com/maruel/genai/refs/heads/main/mistral/testdata/hidden_word.pdf"},
			},
		},
	}
	opts := genai.ChatOptions{
		Temperature: 0.01,
		MaxTokens:   50,
	}
	var err error
	var resp genai.ChatResult
	for i := range 3 {
		// Mistral has a very good rate limiting implementation.
		if resp, err = c.Chat(t.Context(), msgs, &opts); err != nil && i != 2 {
			var herr *httpjson.Error
			if errors.As(err, &herr) {
				if herr.StatusCode == http.StatusTooManyRequests {
					fmt.Fprintf(os.Stderr, "Rate limited, waiting 2s\n")
					time.Sleep(2 * time.Second)
					continue
				}
			}
			t.Fatal(err)
		}
		break
	}
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("Raw response: %#v", resp)
	// Mistral is super efficient with tokens for PDFs.
	if resp.InputTokens != 28 || resp.OutputTokens != 1 {
		t.Logf("Unexpected tokens usage: %v", resp.Usage)
	}
	if len(resp.Contents) != 1 {
		t.Fatal("Unexpected response")
	}
	if got := strings.ToLower(resp.Contents[0].Text); got != "orange" {
		t.Fatal(got)
	}
}

func TestClient_Chat_tool_use(t *testing.T) {
	c := getClient(t, "ministral-3b-latest")
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
	var err error
	var resp genai.ChatResult
	for i := range 3 {
		// Mistral has a very good rate limiting implementation.
		if resp, err = c.Chat(t.Context(), msgs, &opts); err != nil && i != 2 {
			var herr *httpjson.Error
			if errors.As(err, &herr) {
				if herr.StatusCode == http.StatusTooManyRequests {
					fmt.Fprintf(os.Stderr, "Rate limited, waiting 2s\n")
					time.Sleep(2 * time.Second)
					continue
				}
			}
			t.Fatal(err)
		}
		break
	}
	t.Logf("Raw response: %#v", resp)
	if resp.InputTokens != 129 || resp.OutputTokens != 19 {
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
	c := getClient(t, "ministral-3b-latest")
	ctx := t.Context()
	msgs := genai.Messages{
		genai.NewTextMessage(genai.User, "Say hello. Use only one word."),
	}
	opts := genai.ChatOptions{
		Seed:        1,
		Temperature: 0.01,
		MaxTokens:   50,
	}
	for i := range 3 {
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
		if err != nil && i != 2 {
			// Mistral has a very good rate limiting implementation.
			var herr *httpjson.Error
			if errors.As(err, &herr) {
				if herr.StatusCode == http.StatusTooManyRequests {
					fmt.Fprintf(os.Stderr, "Rate limited, waiting 2s\n")
					time.Sleep(2 * time.Second)
					continue
				}
			}
			t.Fatal(err)
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
		return
	}
}

func getClient(t *testing.T, m string) *mistral.Client {
	if os.Getenv("MISTRAL_API_KEY") == "" {
		t.Skip("MISTRAL_API_KEY not set")
	}
	c, err := mistral.New("", m)
	if err != nil {
		t.Fatal(err)
	}
	c.Client.Client.Transport = internaltest.Record(t, c.Client.Client.Transport)
	return c
}

func init() {
	internal.BeLenient = false
}
