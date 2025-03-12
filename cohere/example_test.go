// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package cohere_test

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"strings"

	"github.com/maruel/genai/cohere"
	"github.com/maruel/genai/genaiapi"
)

var (
	key = os.Getenv("COHERE_API_KEY")
	// Using very small model for testing.
	// See https://docs.cohere.com/v2/docs/models
	model = "command-r7b-12-2024"
)

func ExampleClient_Completion() {
	if key != "" {
		c := cohere.Client{ApiKey: key, Model: model}
		msgs := []genaiapi.Message{
			{
				Role: genaiapi.User,
				Type: genaiapi.Text,
				Text: "Is a circle round? Reply as JSON with the form {\"round\": false} or {\"round\": true}.",
			},
		}
		opts := genaiapi.CompletionOptions{
			Seed:        1,
			Temperature: 0.01,
			MaxTokens:   50,
			ReplyAsJSON: true,
		}
		resp, err := c.Completion(context.Background(), msgs, &opts)
		if err != nil {
			log.Fatal(err)
		}
		log.Printf("Response: %#v", resp)
		var expected struct {
			Round bool `json:"round"`
		}
		d := json.NewDecoder(strings.NewReader(resp.Text))
		d.DisallowUnknownFields()
		if err := d.Decode(&expected); err != nil {
			log.Fatalf("Failed to decode JSON: %v", err)
		}
		fmt.Printf("Round: %v\n", expected.Round)
	} else {
		// Print something so the example runs.
		fmt.Println("Round: true")
	}
	// Output: Round: true
}

func ExampleClient_CompletionStream() {
	if key != "" {
		c := cohere.Client{ApiKey: key, Model: model}
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
