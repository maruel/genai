// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package openai_test

import (
	"context"
	"fmt"
	"log"
	"os"

	"github.com/maruel/genai"
	"github.com/maruel/genai/openai"
)

func ExampleClient_Completion() {
	if key := os.Getenv("OPENAI_API_KEY"); key != "" {
		// Using very small model for testing.
		// See https://platform.openai.com/docs/models
		c := openai.Client{
			ApiKey: key,
			Model:  "gpt-4o-mini",
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
