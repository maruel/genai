// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package huggingface_test

import (
	"context"
	"fmt"
	"log"

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
		log.Printf("Response: %#v", resp)
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
		err := c.CompletionStream(ctx, msgs, &genaiapi.CompletionOptions{}, words)
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
