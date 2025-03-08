// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package deepseek_test

import (
	"context"
	"fmt"
	"log"
	"os"

	"github.com/maruel/genai/deepseek"
	"github.com/maruel/genai/genaiapi"
)

func ExampleClient_Completion() {
	if key := os.Getenv("DEEPSEEK_API_KEY"); key != "" {
		// DeepSeek doesn't have a small model. It's also quite slow (often 10s)
		// compared to other service providers.
		// See https://api-docs.deepseek.com/quick_start/pricing
		c := deepseek.Client{
			ApiKey: key,
			Model:  "deepseek-chat",
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
