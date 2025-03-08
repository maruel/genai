// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package cohere_test

import (
	"context"
	"fmt"
	"log"
	"os"

	"github.com/maruel/genai/cohere"
	"github.com/maruel/genai/genaiapi"
)

func ExampleClient_Completion() {
	if key := os.Getenv("COHERE_API_KEY"); key != "" {
		// Using very small model for testing.
		// See https://docs.cohere.com/v2/docs/models
		c := cohere.Client{
			ApiKey: key,
			Model:  "command-r7b-12-2024",
		}
		ctx := context.Background()
		msgs := []genaiapi.Message{
			{Role: genaiapi.User, Content: "Say hello. Use only one word."},
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
