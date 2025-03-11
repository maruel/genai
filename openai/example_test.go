// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package openai_test

import (
	"context"
	_ "embed"
	"fmt"
	"log"
	"os"

	"github.com/maruel/genai/genaiapi"
	"github.com/maruel/genai/openai"
)

//go:embed testdata/banana.jpg
var bananaJpg []byte

var (
	key = os.Getenv("OPENAI_API_KEY")
	// Using small model for testing.
	// See https://platform.openai.com/docs/models
	model = "gpt-4o-mini"
)

func ExampleClient_Completion() {
	if key != "" {
		c := openai.Client{ApiKey: key, Model: model}
		msgs := []genaiapi.Message{
			{
				Role:     genaiapi.User,
				Type:     genaiapi.Document,
				Inline:   true,
				MimeType: "image/jpeg",
				// OpenAI requires higher quality image than Gemini. See
				// ../gemini/testdata/banana.jpg to compare.
				Data: bananaJpg,
			},
			{
				Role: genaiapi.User,
				Type: genaiapi.Text,
				Text: "Is it a banana? Reply with only one word.",
			},
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
		c := openai.Client{ApiKey: key, Model: model}
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
		opts := genaiapi.CompletionOptions{}
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
