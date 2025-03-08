// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package anthropic_test

import (
	"context"
	"fmt"
	"log"
	"os"

	"github.com/maruel/genai/anthropic"
	"github.com/maruel/genai/genaiapi"
)

func ExampleClient_Completion() {
	if key := os.Getenv("ANTHROPIC_API_KEY"); key != "" {
		// Using very small model for testing.
		// https://docs.anthropic.com/en/docs/about-claude/models/all-models
		c := anthropic.Client{
			ApiKey: key,
			Model:  "claude-3-haiku-20240307",
		}
		ctx := context.Background()
		msgs := []genaiapi.Message{
			{Role: genaiapi.User, Content: "Say hello. Use only one word."},
		}
		resp, err := c.Completion(ctx, msgs, 4096, 0, 0)
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
