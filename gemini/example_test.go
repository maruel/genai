// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package gemini_test

import (
	"context"
	"fmt"
	"log"
	"os"

	"github.com/maruel/genai"
	"github.com/maruel/genai/gemini"
)

func ExampleClient_Completion() {
	if key := os.Getenv("GEMINI_API_KEY"); key != "" {
		// Using very small model for testing.
		// See https://ai.google.dev/gemini-api/docs/models/gemini?hl=en
		c := gemini.Client{
			ApiKey: key,
			Model:  "gemini-2.0-flash-lite",
		}
		ctx := context.Background()
		msgs := []genai.Message{
			{Role: genai.User, Content: "Say hello. Use only one word."},
		}
		resp, err := c.Completion(ctx, msgs, 0, 0, 0)
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
