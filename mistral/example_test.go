// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package mistral_test

import (
	"context"
	"fmt"
	"log"
	"os"

	"github.com/maruel/genai/genaiapi"
	"github.com/maruel/genai/mistral"
)

var (
	key = os.Getenv("MISTRAL_API_KEY")
	// Using very small model for testing.
	// See https://docs.mistral.ai/getting-started/models/models_overview/
	model = "ministral-3b-latest"
)

func ExampleClient_Completion() {
	if key != "" {
		c := mistral.Client{ApiKey: key, Model: model}
		msgs := []genaiapi.Message{
			{Role: genaiapi.User, Content: "Say hello. Use only one word."},
		}
		resp, err := c.Completion(context.Background(), msgs, &genaiapi.CompletionOptions{})
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
		c := mistral.Client{ApiKey: key, Model: model}
		ctx := context.Background()
		msgs := []genaiapi.Message{
			{Role: genaiapi.User, Content: "Say hello. Use only one word."},
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
