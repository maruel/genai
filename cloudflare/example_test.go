// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package cloudflare_test

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"strings"

	"github.com/maruel/genai/cloudflare"
	"github.com/maruel/genai/genaiapi"
)

var (
	accountID = os.Getenv("CLOUDFLARE_ACCOUNT_ID")
	key       = os.Getenv("CLOUDFLARE_API_KEY")
)

func ExampleClient_Completion() {
	if accountID != "" && key != "" {
		// We need to use a model that supports structured output.
		c := cloudflare.Client{AccountID: accountID, ApiKey: key, Model: "@hf/nousresearch/hermes-2-pro-mistral-7b"}
		msgs := []genaiapi.Message{
			{
				Role: genaiapi.User,
				Type: genaiapi.Text,
				Text: "Is a circle round? Reply as JSON.",
			},
		}
		opts := genaiapi.CompletionOptions{
			Seed:        1,
			Temperature: 0.01,
			MaxTokens:   50,
			ReplyAsJSON: true,
			JSONSchema: genaiapi.JSONSchema{
				Type: "object",
				Properties: map[string]genaiapi.JSONSchema{
					"round": {
						Type: "boolean",
					},
				},
				Required: []string{"round"},
			},
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
	if accountID != "" && key != "" {
		// Using very small model for testing.
		// See https://developers.cloudflare.com/workers-ai/models/
		c := cloudflare.Client{AccountID: accountID, ApiKey: key, Model: "@cf/meta/llama-3.2-3b-instruct"}
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
