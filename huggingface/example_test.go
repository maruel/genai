// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package huggingface_test

import (
	"context"
	"fmt"
	"log"
	"strings"

	"github.com/maruel/genai/genaiapi"
	"github.com/maruel/genai/huggingface"
)

func ExampleClient_Completion() {
	// This code will run when HUGGINGFACE_API_KEY is set or ~/.cache/huggingface/token exists.
	// As of March 2025, you can try it out for free.
	// See https://huggingface.co/models?inference=warm&sort=trending
	// Eventually use one that supports structured output.
	if c, err := huggingface.New("", "meta-llama/Llama-3.2-1B-Instruct"); err == nil {
		msgs := genaiapi.Messages{
			genaiapi.NewTextMessage(genaiapi.User, "Say hello. Use only one word."),
		}
		opts := genaiapi.CompletionOptions{
			Seed:        1,
			Temperature: 0.01,
			MaxTokens:   50,
		}
		resp, err := c.Completion(context.Background(), msgs, &opts)
		if err != nil {
			log.Fatal(err)
		}
		log.Printf("Raw response: %#v", resp)
		if resp.InputTokens != 43 || resp.OutputTokens != 3 {
			log.Printf("Unexpected tokens usage: %v", resp.Usage)
		}
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

func ExampleClient_Completion_tool_use() {
	// This code will run when HUGGINGFACE_API_KEY is set or ~/.cache/huggingface/token exists.
	// As of March 2025, you can try it out for free.
	// See https://huggingface.co/models?inference=warm&sort=trending
	// Eventually use one that supports structured output.
	if c, err := huggingface.New("", "meta-llama/Llama-3.2-3B-Instruct"); err == nil {
		msgs := genaiapi.Messages{
			genaiapi.NewTextMessage(genaiapi.User, "I wonder if Canada is a better country than the US? Call the tool best_country to tell me which country is the best one."),
		}
		var got struct {
			Country string `json:"country" jsonschema:"enum=Canada,enum=USA"`
		}
		opts := genaiapi.CompletionOptions{
			Seed:        1,
			Temperature: 0.01,
			MaxTokens:   200,
			Tools: []genaiapi.ToolDef{
				{
					Name:        "best_country",
					Description: "A tool to determine the best country",
					InputsAs:    &got,
				},
			},
		}
		resp, err := c.Completion(context.Background(), msgs, &opts)
		if err != nil {
			log.Fatal(err)
		}
		log.Printf("Raw response: %#v", resp)
		if resp.InputTokens != 293 || resp.OutputTokens != 17 {
			log.Printf("Unexpected tokens usage: %v", resp.Usage)
		}
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
	// Output: Best: Canada
}

func ExampleClient_CompletionStream() {
	// This code will run when HUGGINGFACE_API_KEY is set or ~/.cache/huggingface/token exists.
	// As of March 2025, you can try it out for free.
	// See https://huggingface.co/models?inference=warm&sort=trending
	// Eventually use one that supports structured output.
	if c, err := huggingface.New("", "meta-llama/Llama-3.2-1B-Instruct"); err == nil {
		ctx := context.Background()
		msgs := genaiapi.Messages{
			genaiapi.NewTextMessage(genaiapi.User, "Say hello. Use only one word."),
		}
		opts := genaiapi.CompletionOptions{
			Seed:        1,
			Temperature: 0.01,
			MaxTokens:   50,
		}
		chunks := make(chan genaiapi.MessageFragment)
		end := make(chan genaiapi.Message, 10)
		go func() {
			var msgs genaiapi.Messages
			defer func() {
				for _, m := range msgs {
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
					if msgs, err = pkt.Accumulate(msgs); err != nil {
						end <- genaiapi.NewTextMessage(genaiapi.Assistant, fmt.Sprintf("Error: %v", err))
						return
					}
				}
			}
		}()
		err := c.CompletionStream(ctx, msgs, &opts, chunks)
		close(chunks)
		var responses genaiapi.Messages
		for m := range end {
			responses = append(responses, m)
		}
		log.Printf("Raw responses: %#v", responses)
		if err != nil {
			if len(responses) == 0 {
				log.Fatal(err)
			}
			log.Printf("HF currently tend to return spurious HTTP 422: %s", err)
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
	if c, err := huggingface.New("", ""); err == nil {
		models, err := c.ListModels(context.Background())
		if err != nil {
			fmt.Printf("Failed to get models: %v\n", err)
			return
		}
		// Warning: Hugginface hosts a lot of models!
		for _, model := range models {
			fmt.Printf("- %s\n", model)
		}
	}
}
