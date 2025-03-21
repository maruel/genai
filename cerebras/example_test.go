// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package cerebras_test

import (
	"context"
	"fmt"
	"log"
	"os"

	"github.com/maruel/genai"
	"github.com/maruel/genai/cerebras"
)

func ExampleClient_Chat_jSON() {
	// This code will run when CEREBRAS_API_KEY is set.
	//
	// As of March 2025, you can try it out for free.
	//
	// Cerebras supports a limited set of models which you can see on the drop
	// down of https://inference.cerebras.ai/
	if c, err := cerebras.New("", "llama-3.1-8b"); err == nil {
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
		if resp.InputTokens != 173 || resp.OutputTokens != 6 {
			log.Printf("Unexpected tokens usage: %v", resp.Usage)
		}
		if len(resp.Contents) != 1 {
			log.Fatal("Unexpected response")
		}
		if err := resp.Contents[0].Decode(&got); err != nil {
			log.Fatal(err)
		}
		fmt.Printf("Round: %v\n", got.Round)
	} else {
		// Print something so the example runs.
		fmt.Println("Round: true")
	}
	// // Output: Round: true
}

func ExampleClient_ChatStream_tool_use() {
	// This code will run when CEREBRAS_API_KEY is set.
	//
	// As of March 2025, you can try it out for free.
	//
	// Cerebras supports a limited set of models which you can see on the drop
	// down of https://inference.cerebras.ai/
	if c, err := cerebras.New("", "llama-3.1-8b"); err == nil {
		ctx := context.Background()
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
			log.Fatal("Unexpected responses")
		}
		resp := responses[0]
		if len(resp.ToolCalls) != 1 || resp.ToolCalls[0].Name != "best_country" {
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
	// // Output: Best: Canada
}

func ExampleClient_ListModels() {
	// Print something so the example runs.
	fmt.Println("Got models")
	if c, err := cerebras.New("", ""); err == nil {
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
