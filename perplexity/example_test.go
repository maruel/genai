// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package perplexity_test

import (
	"context"
	"fmt"
	"log"
	"strings"

	"github.com/maruel/genai/genaiapi"
	"github.com/maruel/genai/perplexity"
)

func ExampleClient_Completion() {
	// This code will run when PERPLEXITY_API_KEY is set.
	if c, err := perplexity.New(""); err == nil {
		msgs := genaiapi.Messages{
			genaiapi.NewTextMessage(genaiapi.User, "Say hello. Use only one word."),
		}
		opts := genaiapi.CompletionOptions{
			Temperature: 0.01,
			MaxTokens:   50,
		}
		resp, err := c.Completion(context.Background(), msgs, &opts)
		if err != nil {
			log.Fatal(err)
		}
		log.Printf("Raw response: %#v", resp)
		if resp.InputTokens != 8 || resp.OutputTokens != 3 {
			log.Printf("Unexpected tokens usage: %v", resp.Usage)
		}
		if len(resp.Contents) != 1 {
			log.Fatal("Unexpected response")
		}
		// Normalize some of the variance. Obviously many models will still fail this test.
		fmt.Printf("Response: %s\n", strings.TrimRight(strings.TrimSpace(strings.ToLower(resp.Contents[0].Text)), ".!"))
	} else {
		// Print something so the example runs.
		fmt.Println("Response: hello")
	}
	// Output: Response: hello
}

func ExampleClient_CompletionStream() {
	// This code will run when PERPLEXITY_API_KEY is set.
	if c, err := perplexity.New(""); err == nil {
		ctx := context.Background()
		msgs := genaiapi.Messages{
			genaiapi.NewTextMessage(genaiapi.User, "Say hello. Use only one word."),
		}
		opts := genaiapi.CompletionOptions{
			Temperature: 0.01,
			MaxTokens:   50,
		}
		chunks := make(chan genaiapi.MessageFragment)
		end := make(chan genaiapi.Message, 10)
		go func() {
			var msgs genaiapi.Messages
			defer func() {
				for _, m := range msgs {
					end <- m
				}
				close(end)
			}()
			for {
				select {
				case <-ctx.Done():
					return
				case pkt, ok := <-chunks:
					if !ok {
						return
					}
					if msgs, err = pkt.Accumulate(msgs); err != nil {
						end <- genaiapi.NewTextMessage(genaiapi.Assistant, fmt.Sprintf("Error: %v", err))
						return
					}
				}
			}
		}()
		err := c.CompletionStream(ctx, msgs, &opts, chunks)
		close(chunks)
		var responses genaiapi.Messages
		for m := range end {
			responses = append(responses, m)
		}
		log.Printf("Raw responses: %#v", responses)
		if err != nil {
			log.Fatal(err)
		}
		if len(responses) != 1 {
			log.Fatal("Unexpected response")
		}
		resp := responses[0]
		if len(resp.Contents) != 1 {
			log.Fatal("Unexpected response")
		}
		// Normalize some of the variance. Obviously many models will still fail this test.
		fmt.Printf("Response: %s\n", strings.TrimRight(strings.TrimSpace(strings.ToLower(resp.Contents[0].Text)), ".!"))
	} else {
		// Print something so the example runs.
		fmt.Println("Response: hello")
	}
	// Output: Response: hello
}
