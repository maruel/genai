// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package deepseek_test

import (
	"context"
	"fmt"
	"log"
	"os"
	"strings"

	"github.com/maruel/genai"
	"github.com/maruel/genai/deepseek"
)

// DeepSeek doesn't have a small model. It's also quite slow (often 10s)
// compared to other service providers.
// See https://api-docs.deepseek.com/quick_start/pricing
const model = "deepseek-chat"

func ExampleClient_Chat_jSON() {
	// This code will run when DEEPSEEK_API_KEY is set.
	if c, err := deepseek.New("", model); err == nil {
		msgs := genai.Messages{
			genai.NewTextMessage(genai.User, "Is a circle round? Reply as JSON with the form {\"round\": false} or {\"round\": true}."),
		}
		opts := genai.ChatOptions{
			Temperature: 0.01,
			MaxTokens:   50,
			ReplyAsJSON: true,
		}
		resp, err := c.Chat(context.Background(), msgs, &opts)
		if err != nil {
			log.Fatal(err)
		}
		log.Printf("Raw response: %#v", resp)
		if resp.InputTokens != 41 || resp.OutputTokens != 9 {
			log.Printf("Unexpected tokens usage: %v", resp.Usage)
		}
		if len(resp.Contents) != 1 {
			log.Fatal("Unexpected response")
		}
		var got struct {
			Round bool `json:"round"`
		}
		if err := resp.Contents[0].Decode(&got); err != nil {
			log.Fatal(err)
		}
		fmt.Printf("Round: %v\n", got.Round)
	} else {
		// Print something so the example runs.
		fmt.Println("Round: true")
	}
	// Output: Round: true
}

func ExampleClient_Chat_tool_use() {
	// This code will run when DEEPSEEK_API_KEY is set.
	if c, err := deepseek.New("", model); err == nil {
		msgs := genai.Messages{
			genai.NewTextMessage(genai.User, "I wonder if Canada is a better country than the US? Call the tool best_country to tell me which country is the best one."),
		}
		var got struct {
			Country string `json:"country" jsonschema:"enum=Canada,enum=USA"`
		}
		opts := genai.ChatOptions{
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
		resp, err := c.Chat(context.Background(), msgs, &opts)
		if err != nil {
			log.Fatal(err)
		}
		log.Printf("Raw response: %#v", resp)
		if resp.InputTokens != 175 || resp.OutputTokens != 36 {
			log.Printf("Unexpected tokens usage: %v", resp.Usage)
		}
		// Warning: when the model is undecided, it call both.
		if len(resp.ToolCalls) == 0 || resp.ToolCalls[0].Name != "best_country" {
			log.Fatal("Unexpected response")
		}
		if err := resp.ToolCalls[0].Decode(&got); err != nil {
			log.Fatal(err)
		}
		fmt.Printf("Best: %v\n", got.Country)
	} else {
		// Print something so the example runs.
		fmt.Println("Best: Canada")
	}
	// Output: Best: Canada
}

func ExampleClient_ChatStream() {
	// This code will run when DEEPSEEK_API_KEY is set.
	if c, err := deepseek.New("", model); err == nil {
		ctx := context.Background()
		msgs := genai.Messages{
			genai.NewTextMessage(genai.User, "Say hello. Use only one word."),
		}
		opts := genai.ChatOptions{
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
	} else {
		// Print something so the example runs.
		fmt.Println("Response: hello")
	}
	// Output: Response: hello
}

func ExampleClient_ListModels() {
	// Print something so the example runs.
	fmt.Println("Got models")
	if c, err := deepseek.New("", ""); err == nil {
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
	// Output: Got models
}
