// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package cohere_test

import (
	"context"
	"fmt"
	"log"
	"os"
	"strings"

	"github.com/maruel/genai/cohere"
	"github.com/maruel/genai/genaiapi"
)

func ExampleClient_Completion_jSON() {
	// This code will run when COHERE_API_KEY is set.
	// As of March 2025, you can try it out for free with limitations on which
	// functionalities are available.
	// We need to use a model that supports structured output.
	// https://docs.cohere.com/v2/docs/structured-outputs
	if c, err := cohere.New("", "command-r-08-2024"); err == nil {
		msgs := genaiapi.Messages{
			genaiapi.NewTextMessage(genaiapi.User, "Is a circle round? Reply as JSON."),
		}
		var got struct {
			Round bool `json:"round"`
		}
		opts := genaiapi.CompletionOptions{
			Seed:        1,
			Temperature: 0.01,
			MaxTokens:   50,
			DecodeAs:    got,
		}
		resp, err := c.Completion(context.Background(), msgs, &opts)
		if err != nil {
			log.Fatal(err)
		}
		if resp.Role != genaiapi.Assistant || resp.Type != genaiapi.Text {
			log.Fatalf("Unexpected response: %#v", resp)
		}
		// Print to stderr so the test doesn't capture it.
		fmt.Fprintf(os.Stderr, "Raw response: %#v\n", resp)
		if err := resp.Decode(&got); err != nil {
			log.Fatal(err)
		}
		fmt.Printf("Round: %v\n", got.Round)
		if resp.InputTokens < 100 || resp.OutputTokens < 2 {
			log.Fatalf("Missing usage token")
		}
	} else {
		// Print something so the example runs.
		fmt.Println("Round: true")
	}
	// Output: Round: true
}

func ExampleClient_Completion_tool_use() {
	// This code will run when COHERE_API_KEY is set.
	// As of March 2025, you can try it out for free with limitations on which
	// functionalities are available.
	// We need to use a model that supports structured output.
	// https://docs.cohere.com/v2/docs/structured-outputs
	if c, err := cohere.New("", "command-r-08-2024"); err == nil {
		msgs := genaiapi.Messages{
			genaiapi.NewTextMessage(genaiapi.User, "I wonder if Canada is a better country than the US? Call the tool best_country to tell me which country is the best one."),
		}
		var got struct {
			Country string `json:"country" jsonschema:"enum=Canada,enum=USA"`
		}
		opts := genaiapi.CompletionOptions{
			Seed:        1,
			Temperature: 0.01,
			MaxTokens:   200,
			Tools: []genaiapi.ToolDef{
				{
					Name:        "best_country",
					Description: "A tool to determine the best country",
					InputsAs:    &got,
				},
			},
		}
		resp, err := c.Completion(context.Background(), msgs, &opts)
		if err != nil {
			log.Fatal(err)
		}
		if resp.Role != genaiapi.Assistant || resp.Type != genaiapi.ToolCalls {
			log.Fatalf("Unexpected response: %#v", resp)
		}
		log.Printf("Response: %#v", resp)
		// Warning: when the model is undecided, it call both.
		if len(resp.ToolCalls) == 0 || resp.ToolCalls[0].Name != "best_country" {
			log.Fatal("Expected at least one best_country tool call")
		}
		if err := resp.ToolCalls[0].Decode(&got); err != nil {
			log.Fatal(err)
		}
		fmt.Printf("Best: %v\n", got.Country)
	} else {
		// Print something so the example runs.
		fmt.Println("Best: Canada")
	}
	// Output: Best: Canada
}

func ExampleClient_CompletionStream() {
	// This code will run when COHERE_API_KEY is set.
	// As of March 2025, you can try it out for free with limitations on which
	// functionalities are available.
	// Using very small model for testing.
	// See https://docs.cohere.com/v2/docs/models
	if c, err := cohere.New("", "command-r7b-12-2024"); err == nil {
		ctx := context.Background()
		msgs := genaiapi.Messages{
			genaiapi.NewTextMessage(genaiapi.User, "Say hello. Use only one word."),
		}
		chunks := make(chan genaiapi.MessageFragment)
		end := make(chan string)
		go func() {
			resp := ""
			for {
				select {
				case <-ctx.Done():
					end <- resp
					return
				case w, ok := <-chunks:
					if !ok {
						end <- resp
						return
					}
					if w.Type != genaiapi.Text {
						end <- fmt.Sprintf("Got %q; Unexpected type: %v", resp, w.Type)
						return
					}
					resp += w.TextFragment
				}
			}
		}()
		opts := genaiapi.CompletionOptions{
			Seed:        1,
			Temperature: 0.01,
			MaxTokens:   50,
		}
		err := c.CompletionStream(ctx, msgs, &opts, chunks)
		close(chunks)
		response := <-end
		if err != nil {
			log.Fatal(err)
		}
		// Normalize some of the variance. Obviously many models will still fail this test.
		response = strings.TrimRight(strings.TrimSpace(strings.ToLower(response)), ".!")
		fmt.Printf("Response: %s\n", response)
	} else {
		// Print something so the example runs.
		fmt.Println("Response: hello")
	}
	// Output: Response: hello
}

func ExampleClient_ListModels() {
	// Print something so the example runs.
	fmt.Println("Got models")
	if c, err := cohere.New("", ""); err == nil {
		models, err := c.ListModels(context.Background())
		if err != nil {
			fmt.Printf("Failed to get models: %v\n", err)
			return
		}
		for _, model := range models {
			// The list of models will change over time. Print them to stderr so the
			// test doesn't capture them.
			fmt.Fprintf(os.Stderr, "- %s\n", model)
		}
	}
	// Output: Got models
}
