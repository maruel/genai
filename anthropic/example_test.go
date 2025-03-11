// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package anthropic_test

import (
	"context"
	"fmt"
	"log"
	"os"

	"github.com/maruel/genai/anthropic"
	"github.com/maruel/genai/genaiapi"
)

var (
	key = os.Getenv("ANTHROPIC_API_KEY")
	// Using very small model for testing.
	// https://docs.anthropic.com/en/docs/about-claude/models/all-models
	model = "claude-3-haiku-20240307"
)

func ExampleClient_Completion() {
	if key != "" {
		c := anthropic.Client{ApiKey: key, Model: model}
		msgs := []genaiapi.Message{
			{
				Role:    genaiapi.User,
				Type:    genaiapi.Text,
				Content: "Say hello. Use only one word.",
			},
		}
		opts := genaiapi.CompletionOptions{MaxTokens: 4096}
		resp, err := c.Completion(context.Background(), msgs, &opts)
		if err != nil {
			log.Fatal(err)
		}
		if len(resp) < 2 || len(resp) > 100 {
			log.Fatalf("Unexpected response: %s", resp)
		}
	}
	// Print something so the example runs.
	fmt.Println("Hello, world!")
	// Output: Hello, world!
}

func ExampleClient_CompletionStream() {
	if key != "" {
		c := anthropic.Client{ApiKey: key, Model: model}
		ctx := context.Background()
		msgs := []genaiapi.Message{
			{
				Role:    genaiapi.User,
				Type:    genaiapi.Text,
				Content: "Say hello. Use only one word.",
			},
		}
		words := make(chan string, 10)
		end := make(chan struct{})
		go func() {
			resp := ""
			for {
				select {
				case <-ctx.Done():
					goto end
				case w, ok := <-words:
					if !ok {
						goto end
					}
					resp += w
				}
			}
		end:
			close(end)
			if len(resp) < 2 || len(resp) > 100 {
				log.Printf("Unexpected response: %s", resp)
			}
		}()
		opts := genaiapi.CompletionOptions{MaxTokens: 4096}
		err := c.CompletionStream(ctx, msgs, &opts, words)
		close(words)
		<-end
		if err != nil {
			log.Fatal(err)
		}
	}
	// Print something so the example runs.
	fmt.Println("Hello, world!")
	// Output: Hello, world!
}
