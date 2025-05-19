// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package cohere_test

import (
	"context"
	"fmt"
	"log"
	"strings"

	"github.com/maruel/genai"
	"github.com/maruel/genai/cohere"
)

func ExampleClient_Chat_jSON() {
	// We need to use a model that supports structured output.
	// https://docs.cohere.com/v2/docs/structured-outputs
	c, err := cohere.New("", "command-r-08-2024")
	if err != nil {
		log.Fatal(err)
	}
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
	resp, err := c.Chat(context.Background(), msgs, &opts)
	if err != nil {
		log.Fatal(err)
	}
	log.Printf("Raw response: %#v", resp)
	if err := resp.Decode(&got); err != nil {
		log.Fatal(err)
	}
	fmt.Printf("Round: %v\n", got.Round)
	// This would Output: Round: true
}

func ExampleClient_Chat_tool_use() {
	// This example shows LLM positional bias. It will always return the first country listed.

	// We need to use a model that supports structured output.
	// https://docs.cohere.com/v2/docs/structured-outputs
	c, err := cohere.New("", "command-r-08-2024")
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
	if resp.InputTokens != 925 || resp.OutputTokens != 96 {
		log.Printf("Unexpected tokens usage: %v", resp.Usage)
	}
	// Warning: when the model is undecided, it call both.
	if len(resp.ToolCalls) == 0 || resp.ToolCalls[0].Name != "best_country" {
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
	// See https://docs.cohere.com/v2/docs/models
	c, err := cohere.New("", "command-r7b-12-2024")
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
	// Normalize some of the variance. Obviously many models will still fail this test.
	fmt.Printf("Response: %s\n", strings.TrimRight(strings.TrimSpace(strings.ToLower(resp.AsText())), ".!"))
	// This would Output: Response: hello
}

func ExampleClient_ListModels() {
	c, err := cohere.New("", "")
	if err != nil {
		fmt.Printf("Couldn't connect: %v\n", err)
		return
	}
	models, err := c.ListModels(context.Background())
	if err != nil {
		fmt.Printf("Failed to get models: %v\n", err)
		return
	}
	for _, model := range models {
		fmt.Printf("- %s\n", model)
	}
}
