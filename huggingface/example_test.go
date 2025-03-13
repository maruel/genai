// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package huggingface_test

import (
	"context"
	"fmt"
	"log"
	"os"
	"path/filepath"

	"github.com/maruel/genai/genaiapi"
	"github.com/maruel/genai/huggingface"
)

var (
	shouldRun = func() bool {
		h, err := os.UserHomeDir()
		if err != nil {
			return false
		}
		// TODO: Windows.
		_, err = os.ReadFile(filepath.Join(h, ".cache", "huggingface", "token"))
		return err == nil
	}()
)

func ExampleClient_Completion() {
	if shouldRun {
		// See https://huggingface.co/models?inference=warm&sort=trending
		// Eventually use one that supports structured output.
		c := huggingface.Client{Model: "meta-llama/Llama-3.2-1B-Instruct"}
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
	if shouldRun {
		// Using very small model for testing.
		// See https://huggingface.co/models?inference=warm&sort=trending
		c := huggingface.Client{Model: "meta-llama/Llama-3.2-1B-Instruct"}
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
