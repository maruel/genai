// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package togetherai_test

import (
	"context"
	"fmt"
	"log"
	"strings"

	"github.com/maruel/genai/genaiapi"
	"github.com/maruel/genai/internal/togetherai"
)

// Not all models seem to work with the initial key. They just return with
// invalid_request_error which is quite odd.
// See https://api.together.ai/models
var model = "Qwen/Qwen2.5-7B-Instruct-Turbo"

func ExampleClient_Completion() {
	// This code will run when TOGETHERAI_API_KEY is set.
	if c, err := togetherai.New("", model); err == nil {
		msgs := []genaiapi.Message{
			{
				Role: genaiapi.User,
				Type: genaiapi.Text,
				Text: "Say hello. Use only one word.",
			},
		}
		resp, err := c.Completion(context.Background(), msgs, &genaiapi.CompletionOptions{})
		if err != nil {
			log.Fatal(err)
		}
		txt := resp.Text
		if len(txt) < 2 || len(txt) > 100 {
			log.Fatalf("Unexpected response: %s", txt)
		}
	}
	// Print something so the example runs.
	fmt.Println("Hello, world!")
	// Output: Hello, world!
}

func ExampleClient_CompletionStream() {
	// This code will run when TOGETHERAI_API_KEY is set.
	if c, err := togetherai.New("", model); err == nil {
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
			Seed:        1,
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
