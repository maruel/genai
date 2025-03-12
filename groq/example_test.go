// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package groq_test

import (
	"bytes"
	"context"
	_ "embed"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"strings"

	"github.com/maruel/genai/genaiapi"
	"github.com/maruel/genai/groq"
)

//go:embed testdata/banana.jpg
var bananaJpg []byte

var key = os.Getenv("GROQ_API_KEY")

func ExampleClient_Completion() {
	if key != "" {
		// We must select a model that supports vision.
		// See https://console.groq.com/docs/vision
		c := groq.Client{ApiKey: key, Model: "llama-3.2-11b-vision-preview"}
		msgs := []genaiapi.Message{
			{
				Role:     genaiapi.User,
				Type:     genaiapi.Document,
				Filename: "banana.jpg",
				// Groq requires higher quality image than Gemini or Mistral. See
				// ../gemini/testdata/banana.jpg to compare.
				Document: bytes.NewReader(bananaJpg),
			},
			{
				Role: genaiapi.User,
				Type: genaiapi.Text,
				Text: "Is it a banana? Reply as JSON with the form {\"banana\": false} or {\"banana\": true}.",
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
			Banana bool `json:"banana"`
		}
		d := json.NewDecoder(strings.NewReader(resp.Text))
		d.DisallowUnknownFields()
		if err := d.Decode(&expected); err != nil {
			log.Fatalf("Failed to decode JSON: %v", err)
		}
		fmt.Printf("Banana: %v\n", expected.Banana)
	} else {
		// Print something so the example runs.
		fmt.Println("Banana: true")
	}
	// Output: Banana: true
}

func ExampleClient_CompletionStream() {
	if key != "" {
		// Using very small model for testing.
		// See https://console.groq.com/docs/models
		c := groq.Client{ApiKey: key, Model: "llama-3.2-1b-preview"}
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
