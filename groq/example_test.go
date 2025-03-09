// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package groq_test

import (
	"context"
	"fmt"
	"log"
	"os"

	"github.com/maruel/genai/genaiapi"
	"github.com/maruel/genai/groq"
)

var (
	key = os.Getenv("GROQ_API_KEY")
	// Using very small model for testing.
	// See https://console.groq.com/docs/models
	model = "llama-3.2-1b-preview"
)

func ExampleClient_Completion() {
	if key != "" {
		c := groq.Client{ApiKey: key, Model: model}
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
