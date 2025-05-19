// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package togetherai_test

import (
	"bytes"
	"context"
	_ "embed"
	"fmt"
	"log"
	"os"
	"strings"

	"github.com/maruel/genai"
	"github.com/maruel/genai/togetherai"
)

func ExampleClient_Chat_vision_and_JSON() {
	// We must select a model that supports vision *and* JSON mode (not
	// necessarily tool use).
	// Warning: looks like this model doesn't support JSON schema.
	// https://docs.together.ai/docs/serverless-models#vision-models
	c, err := togetherai.New("", "meta-llama/Llama-Vision-Free")
	if err != nil {
		log.Fatal(err)
	}
	bananaJpg, err := os.ReadFile("banana.jpg")
	if err != nil {
		log.Fatal(err)
	}
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

func ExampleClient_Chat_video() {
	// We must select a model that supports video.
	// https://docs.together.ai/docs/serverless-models#vision-models
	//
	// 2025-03-19: TogetherAI removed Qwen/Qwen2.5-VL-72B-Instruct and
	// Qwen/Qwen2-VL-72B-Instruct cannot process videos.
	c, err := togetherai.New("", "Qwen/Qwen2.5-VL-72B-Instruct")
	if err != nil {
		log.Fatal(err)
	}
	f, err := os.Open("testdata/animation.mp4")
	if err != nil {
		log.Fatal(err)
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
	resp, err := c.Chat(context.Background(), msgs, &opts)
	if err != nil {
		log.Fatal(err)
	}
	log.Printf("Raw response: %#v", resp)
	fmt.Printf("Saw: %v\n", strings.ToLower(resp.AsText()))
	// This would Output: Saw: banana
}

func ExampleClient_Chat_tool_use() {
	// This example shows LLM positional bias. It will always return the first country listed.

	c, err := togetherai.New("", "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free")
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
		MaxTokens:   50,
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
	c, err := togetherai.New("", "meta-llama/Llama-3.2-3B-Instruct-Turbo")
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
	c, err := togetherai.New("", "")
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
