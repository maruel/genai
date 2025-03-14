// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package anthropic_test

import (
	"bytes"
	"context"
	_ "embed"
	"fmt"
	"log"
	"strings"

	"github.com/maruel/genai/anthropic"
	"github.com/maruel/genai/genaiapi"
)

//go:embed testdata/banana.jpg
var bananaJpg []byte

// Using very small model for testing. As of March 2025,
// claude-3-haiku-20240307 is 0.20$/1.25$ while claude-3-5-haiku-20241022 is
// 0.80$/4.00$.
// https://docs.anthropic.com/en/docs/about-claude/models/all-models
var model = "claude-3-haiku-20240307"

func ExampleClient_Completion() {
	// This code will run when ANTHROPIC_API_KEY is set.
	if c, err := anthropic.New("", model); err == nil {
		msgs := []genaiapi.Message{
			{
				Role:     genaiapi.User,
				Type:     genaiapi.Document,
				Filename: "banana.jpg",
				// Anthropic requires higher quality image than Gemini or Mistral. See
				// ../gemini/testdata/banana.jpg to compare.
				Document: bytes.NewReader(bananaJpg),
			},
			{
				Role: genaiapi.User,
				Type: genaiapi.Text,
				Text: "Is it a banana? Reply with only one word.",
			},
		}
		opts := genaiapi.CompletionOptions{
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
	// This code will run when ANTHROPIC_API_KEY is set.
	if c, err := anthropic.New("", model); err == nil {
		ctx := context.Background()
		msgs := []genaiapi.Message{
			{
				Role: genaiapi.User,
				Type: genaiapi.Text,
				Text: "Say hello. Use only one word.",
			},
		}
		chunks := make(chan genaiapi.MessageChunk, 10)
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
