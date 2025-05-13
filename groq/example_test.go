// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package groq_test

import (
	"bytes"
	"context"
	_ "embed"
	"fmt"
	"log"
	"os"
	"strings"

	"github.com/maruel/genai"
	"github.com/maruel/genai/groq"
)

func ExampleClient_Chat_vision_and_JSON() {
	// We must select a model that supports vision *and* JSON mode (not
	// necessarily tool use).
	// See "JSON Mode with Images" at https://console.groq.com/docs/vision
	c, err := groq.New("", "meta-llama/llama-4-scout-17b-16e-instruct")
	if err != nil {
		log.Fatal(err)
	}
	bananaJpg, err := os.ReadFile("banana.pdf")
	if err != nil {
		log.Fatal(err)
	}
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
	resp, err := c.Chat(context.Background(), msgs, &opts)
	if err != nil {
		log.Fatal(err)
	}
	log.Printf("Raw response: %#v", resp)
	if len(resp.Contents) != 1 {
		log.Fatal("Unexpected response")
	}
	var got struct {
		Banana bool `json:"banana"`
	}
	if err := resp.Contents[0].Decode(&got); err != nil {
		log.Fatal(err)
	}
	fmt.Printf("Banana: %v\n", got.Banana)
	// This would Output: Banana: true
}

func ExampleClient_Chat_tool_use() {
	// We must select a model that supports tool use. Use the smallest one.
	// See https://console.groq.com/docs/tool-use
	c, err := groq.New("", "llama-3.1-8b-instant")
	if err != nil {
		log.Fatal(err)
	}
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
	resp, err := c.Chat(context.Background(), msgs, &opts)
	if err != nil {
		log.Fatal(err)
	}
	log.Printf("Raw response: %#v", resp)
	if len(resp.ToolCalls) != 1 || resp.ToolCalls[0].Name != "best_country" {
		log.Fatal("Unexpected response")
	}
	if err := resp.ToolCalls[0].Decode(&got); err != nil {
		log.Fatal(err)
	}
	fmt.Printf("Best: %v\n", got.Country)
	// This would Output: Best: Canada
}

func ExampleClient_ChatStream() {
	// Using very small model for testing.
	// See https://console.groq.com/docs/models
	c, err := groq.New("", "llama-3.1-8b-instant")
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
	log.Printf("Raw responses: %#v", responses)
	if err != nil {
		log.Fatal(err)
	}
	if len(responses) != 1 {
		log.Fatal("Unexpected response")
	}
	resp := responses[0]
	if len(resp.Contents) != 1 {
		log.Fatal("Unexpected response")
	}
	// Normalize some of the variance. Obviously many models will still fail this test.
	fmt.Printf("Response: %s\n", strings.TrimRight(strings.TrimSpace(strings.ToLower(resp.Contents[0].Text)), ".!"))
	// This would Output: Response: hello
}

func ExampleClient_ListModels() {
	c, err := groq.New("", "")
	if err != nil {
		fmt.Fprintf(os.Stderr, "Couldn't connect: %v\n", err)
		return
	}
	models, err := c.ListModels(context.Background())
	if err != nil {
		fmt.Printf("Failed to get models: %v\n", err)
		return
	}
	for _, model := range models {
		// The list of models will change over time. Print them to stderr so the
		// test doesn't capture them.
		fmt.Fprintf(os.Stderr, "- %s\n", model)
	}
}
