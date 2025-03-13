// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package gemini_test

import (
	"bytes"
	"context"
	_ "embed"
	"fmt"
	"log"

	"github.com/maruel/genai/gemini"
	"github.com/maruel/genai/genaiapi"
)

//go:embed testdata/banana.jpg
var bananaJpg []byte

// Using small model for testing.
// See https://ai.google.dev/gemini-api/docs/models/gemini?hl=en
var model = "gemini-2.0-flash-lite"

func ExampleClient_Completion() {
	// This code will run when GEMINI_API_KEY is set.
	// As of March 2025, you can try it out for free.
	if c, err := gemini.New("", model); err == nil {
		msgs := []genaiapi.Message{
			{
				Role: genaiapi.User,
				Type: genaiapi.Document,
				// Gemini supports highly compressed jpg.
				Filename: "banana.jpg",
				Document: bytes.NewReader(bananaJpg),
			},
			{
				Role: genaiapi.User,
				Type: genaiapi.Text,
				Text: "Is it a banana? Reply with only one word.",
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
	// This code will run when GEMINI_API_KEY is set.
	// As of March 2025, you can try it out for free.
	if c, err := gemini.New("", model); err == nil {
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
