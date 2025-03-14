// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package huggingface_test

import (
	"context"
	"fmt"
	"log"
	"os"
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
		msgs := []genaiapi.Message{
			{
				Role: genaiapi.User,
				Type: genaiapi.Text,
				Text: "Say hello. Use only one word.",
			},
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
		// Print to stderr so the test doesn't capture it.
		fmt.Fprintf(os.Stderr, "Raw response: %#v\n", resp)
		// Normalize some of the variance. Obviously many models will still fail this test.
		txt := strings.TrimRight(strings.TrimSpace(strings.ToLower(resp.Text)), ".!")
		fmt.Printf("Response: %s\n", txt)
		if resp.InputTokens < 10 || resp.OutputTokens < 2 {
			log.Fatalf("Missing usage token")
		}
	} else {
		// Print something so the example runs.
		fmt.Println("Response: hello")
	}
	// Output: Response: hello
}

func ExampleClient_CompletionStream() {
	// This code will run when HUGGINGFACE_API_KEY is set or ~/.cache/huggingface/token exists.
	// As of March 2025, you can try it out for free.
	// See https://huggingface.co/models?inference=warm&sort=trending
	// Eventually use one that supports structured output.
	if c, err := huggingface.New("", "meta-llama/Llama-3.2-1B-Instruct"); err == nil {
		ctx := context.Background()
		msgs := []genaiapi.Message{
			{
				Role: genaiapi.User,
				Type: genaiapi.Text,
				Text: "Say hello. Use only one word.",
			},
		}
		chunks := make(chan genaiapi.MessageChunk)
		end := make(chan string)
		go func() {
			resp := ""
			for {
				select {
				case <-ctx.Done():
					end <- resp
					return
				case w, ok := <-chunks:
					if !ok {
						end <- resp
						return
					}
					if w.Type != genaiapi.Text {
						end <- fmt.Sprintf("Got %q; Unexpected type: %v", resp, w.Type)
						return
					}
					resp += w.Text
				}
			}
		}()
		opts := genaiapi.CompletionOptions{
			Temperature: 0.01,
			MaxTokens:   50,
		}
		err := c.CompletionStream(ctx, msgs, &opts, chunks)
		close(chunks)
		response := <-end
		if err != nil {
			log.Fatal(err)
		}
		// Normalize some of the variance. Obviously many models will still fail this test.
		response = strings.TrimRight(strings.TrimSpace(strings.ToLower(response)), ".!")
		fmt.Printf("Response: %s\n", response)
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
