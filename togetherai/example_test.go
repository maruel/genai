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
	"path/filepath"
	"strings"

	"github.com/maruel/genai"
	"github.com/maruel/genai/togetherai"
)

// See the 1kib banana jpg online at
// https://github.com/maruel/genai/blob/main/togetherai/testdata/banana.jpg
//
//go:embed testdata/banana.jpg
var bananaJpg []byte

func ExampleClient_Completion_vision_and_JSON() {
	// This code will run when TOGETHERAI_API_KEY is set.
	//
	// As of March 2025, you can try it out for free.
	//
	// We must select a model that supports vision *and* JSON mode (not
	// necessarily tool use).
	// Warning: looks like this model doesn't support JSON schema.
	// https://docs.together.ai/docs/serverless-models#vision-models
	if c, err := togetherai.New("", "meta-llama/Llama-Vision-Free"); err == nil {
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
		opts := genai.CompletionOptions{
			Seed:        1,
			Temperature: 0.01,
			MaxTokens:   50,
			DecodeAs:    &got,
		}
		resp, err := c.Completion(context.Background(), msgs, &opts)
		if err != nil {
			log.Fatal(err)
		}
		log.Printf("Raw response: %#v", resp)
		if resp.InputTokens != 33 || resp.OutputTokens != 6 {
			log.Printf("Unexpected tokens usage: %v", resp.Usage)
		}
		if len(resp.Contents) != 1 {
			log.Fatal("Unexpected response")
		}
		if err := resp.Contents[0].Decode(&got); err != nil {
			log.Fatal(err)
		}
		fmt.Printf("Banana: %v\n", got.Banana)
	} else {
		// Print something so the example runs.
		fmt.Println("Banana: true")
	}
	// Output: Banana: true
}

func ExampleClient_Completion_video() {
	// This code will run when TOGETHERAI_API_KEY is set.
	//
	// As of March 2025, you can try it out for free.
	//
	// We must select a model that supports video.
	// https://docs.together.ai/docs/serverless-models#vision-models
	//
	// 2025-03-19: TogetherAI removed Qwen/Qwen2.5-VL-72B-Instruct and
	// Qwen/Qwen2-VL-72B-Instruct cannot process videos.
	if c, err := togetherai.New("", "Qwen/Qwen2.5-VL-72B-Instruct"); err == nil && false {
		f, err := os.Open("testdata/animation.mp4")
		if err != nil {
			log.Fatal(err)
		}
		defer f.Close()
		// TogetherAI seems to require separate messages for text and images.
		msgs := genai.Messages{
			genai.NewTextMessage(genai.User, "What is the word? Reply with exactly and only one word."),
			{
				Role: genai.User,
				Contents: []genai.Content{
					{Filename: filepath.Base(f.Name()), Document: f},
				},
			},
		}
		opts := genai.CompletionOptions{
			Seed:        1,
			Temperature: 0.01,
			MaxTokens:   50,
		}
		resp, err := c.Completion(context.Background(), msgs, &opts)
		if err != nil {
			log.Fatal(err)
		}
		log.Printf("Raw response: %#v", resp)
		if resp.InputTokens != 3025 || resp.OutputTokens != 4 {
			log.Printf("Unexpected tokens usage: %v", resp.Usage)
		}
		if len(resp.Contents) != 1 {
			log.Fatal("Unexpected response")
		}
		fmt.Printf("Saw: %v\n", strings.ToLower(resp.Contents[0].Text))
	} else {
		// Print something so the example runs.
		fmt.Println("Saw: banana")
	}
	// Output: Saw: banana
}

func ExampleClient_Completion_tool_use() {
	// This code will run when TOGETHER_API_KEY is set.
	//
	// As of March 2025, you can try it out for free.
	//
	// You must select a model that supports tool use.
	if c, err := togetherai.New("", "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"); err == nil {
		msgs := genai.Messages{
			genai.NewTextMessage(genai.User, "I wonder if Canada is a better country than the US? Call the tool best_country to tell me which country is the best one."),
		}
		var got struct {
			Country string `json:"country" jsonschema:"enum=Canada,enum=USA"`
		}
		opts := genai.CompletionOptions{
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
		resp, err := c.Completion(context.Background(), msgs, &opts)
		if err != nil {
			log.Fatal(err)
		}
		log.Printf("Raw response: %#v", resp)
		if resp.InputTokens != 340 || resp.OutputTokens != 15 {
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
	// This code will run when TOGETHERAI_API_KEY is set.
	// See https://api.together.ai/models
	if c, err := togetherai.New("", "meta-llama/Llama-3.2-3B-Instruct-Turbo"); err == nil {
		ctx := context.Background()
		msgs := genai.Messages{
			genai.NewTextMessage(genai.User, "Say hello. Use only one word."),
		}
		opts := genai.CompletionOptions{
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
					if pendingMsgs, err = pkt.Accumulate(pendingMsgs); err != nil {
						end <- genai.NewTextMessage(genai.Assistant, fmt.Sprintf("Error: %v", err))
						return
					}
				}
			}
		}()
		err := c.CompletionStream(ctx, msgs, &opts, chunks)
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
	if c, err := togetherai.New("", ""); err == nil {
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
