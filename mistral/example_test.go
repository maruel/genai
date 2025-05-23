// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package mistral_test

import (
	"bytes"
	"context"
	_ "embed"
	"fmt"
	"log"
	"os"
	"strings"

	"github.com/maruel/genai"
	"github.com/maruel/genai/mistral"
)

func ExampleClient_Chat_vision_and_JSON() {
	// Require a model which has the "vision" capability.
	// https://docs.mistral.ai/capabilities/vision/
	c, err := mistral.New("", "mistral-small-latest")
	if err != nil {
		log.Fatal(err)
	}
	bananaJpg, err := os.ReadFile("banana.jpg")
	if err != nil {
		log.Fatal(err)
	}
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
	resp, err := c.Chat(context.Background(), msgs, &opts)
	if err != nil {
		log.Fatal(err)
	}
	log.Printf("Raw response: %#v", resp)
	if err := resp.Decode(&got); err != nil {
		log.Fatal(err)
	}
	fmt.Printf("Banana: %v\n", got.Banana)
	// This would Output: Banana: true
}

func ExampleClient_Chat_pDF() {
	// Require a model which has the "OCR" or the "Document understanding"
	// capability. There's a subtle difference between the two; from what I
	// understand, the document understanding will only parse the text, while the
	// OCR will try to understand the pictures.
	// https://docs.mistral.ai/capabilities/document/
	// https://docs.mistral.ai/capabilities/vision/
	c, err := mistral.New("", "mistral-small-latest")
	if err != nil {
		log.Fatal(err)
	}
	msgs := genai.Messages{
		{
			Role: genai.User,
			Contents: []genai.Content{
				{Text: "What is the word? Reply with only the word."},
				{URL: "https://raw.githubusercontent.com/maruel/genai/refs/heads/main/internal/internaltest/testdata/hidden_word.pdf"},
			},
		},
	}
	opts := genai.ChatOptions{
		Temperature: 0.01,
		MaxTokens:   50,
	}
	resp, err := c.Chat(context.Background(), msgs, &opts)
	if err != nil {
		log.Fatal(err)
	}
	log.Printf("Raw response: %#v", resp)
	fmt.Printf("Hidden word in PDF: %v\n", strings.ToLower(resp.AsText()))
	// This would Output: Hidden word in PDF: orange
}

func ExampleClient_Chat_tool_use() {
	// This example shows LLM positional bias. It will always return the first country listed.

	// Require a model which has the tool capability. See
	// https://docs.mistral.ai/capabilities/function_calling/
	c, err := mistral.New("", "ministral-3b-latest")
	if err != nil {
		log.Fatal(err)
	}
	msgs := genai.Messages{
		genai.NewTextMessage(genai.User, "I wonder if Canada is a better country than the US? Call the tool best_country to tell me which country is the best one."),
	}
	type got struct {
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
				Callback: func(g *got) string {
					return g.Country
				},
			},
		},
	}
	resp, err := c.Chat(context.Background(), msgs, &opts)
	if err != nil {
		log.Fatal(err)
	}
	log.Printf("Raw response: %#v", resp)
	if len(resp.ToolCalls) != 1 || resp.ToolCalls[0].Name != "best_country" {
		log.Fatal("Unexpected response")
	}
	res, err := resp.ToolCalls[0].Call(opts.Tools)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("Best: %v\n", res)
	// This would Output: Best: Canada
}

func ExampleClient_ChatStream() {
	// Using very small model for testing.
	// See https://docs.mistral.ai/getting-started/models/models_overview/
	c, err := mistral.New("", "ministral-3b-latest")
	if err != nil {
		log.Fatal(err)
	}
	ctx := context.Background()
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
				var err2 error
				if pendingMsgs, err2 = pkt.Accumulate(pendingMsgs); err2 != nil {
					end <- genai.NewTextMessage(genai.Assistant, fmt.Sprintf("Error: %v", err2))
					return
				}
			}
		}
	}()
	_, err = c.ChatStream(ctx, msgs, &opts, chunks)
	close(chunks)
	var responses genai.Messages
	for m := range end {
		responses = append(responses, m)
	}
	if err != nil {
		log.Fatal(err)
	}
	log.Printf("Raw responses: %#v", responses)
	if err != nil {
		log.Fatal(err)
	}
	if len(responses) != 1 {
		log.Fatal("Unexpected response")
	}
	resp := responses[0]
	// Normalize some of the variance. Obviously many models will still fail this test.
	fmt.Printf("Response: %s\n", strings.TrimRight(strings.TrimSpace(strings.ToLower(resp.AsText())), ".!"))
	// This would Output: Response: hello
}
