// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package cerebras_test

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"strings"

	"github.com/invopop/jsonschema"
	"github.com/maruel/genai/cerebras"
	"github.com/maruel/genai/genaiapi"
)

func ExampleClient_Completion_jSONSchema() {
	// This code will run when CEREBRAS_API_KEY is set.
	// As of March 2025, you can try it out for free.
	// Cerebras supports a limited set of models.
	if c, err := cerebras.New("", "llama3.1-8b"); err == nil {
		msgs := []genaiapi.Message{
			{
				Role: genaiapi.User,
				Type: genaiapi.Text,
				Text: "Is a circle round? Reply as JSON.",
			},
		}
		var expected struct {
			Round bool `json:"round"`
		}
		opts := genaiapi.CompletionOptions{
			Seed:        1,
			Temperature: 0.01,
			MaxTokens:   50,
			JSONSchema:  jsonschema.Reflect(expected),
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
		d := json.NewDecoder(strings.NewReader(resp.Text))
		d.DisallowUnknownFields()
		if err := d.Decode(&expected); err != nil {
			log.Fatalf("Failed to decode JSON: %v", err)
		}
		fmt.Printf("Round: %v\n", expected.Round)
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
	// This code will run when CEREBRAS_API_KEY is set.
	// As of March 2025, you can try it out for free.
	// Cerebras supports a limited set of models.
	if c, err := cerebras.New("", "llama-3.1-8b"); err == nil {
		msgs := []genaiapi.Message{
			{
				Role: genaiapi.User,
				Type: genaiapi.Text,
				Text: "I wonder if Canada is a better country than the US? Call the tool best_country to tell me which country is the best one.",
			},
		}
		var expected struct {
			Country string `json:"country"`
		}
		param := jsonschema.Reflect(expected)
		param.Properties.Value("country").Enum = []any{"Canada", "US"}
		opts := genaiapi.CompletionOptions{
			Seed:        1,
			Temperature: 0.01,
			MaxTokens:   50,
			Tools: []genaiapi.ToolDef{
				{
					Name:        "best_country",
					Description: "A tool to determine the best country",
					Parameters:  param,
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
		// Warning: there's a bug where it returns two identical tool calls. To verify.
		if len(resp.ToolCalls) == 0 || resp.ToolCalls[0].Name != "best_country" {
			log.Fatal("Expected 1 best_country tool call")
		}
		d := json.NewDecoder(strings.NewReader(resp.ToolCalls[0].Arguments))
		d.DisallowUnknownFields()
		if err := d.Decode(&expected); err != nil {
			log.Fatalf("Failed to decode %q as JSON: %v", resp.ToolCalls[0].Arguments, err)
		}
		fmt.Printf("Best: %v\n", expected.Country)
	} else {
		// Print something so the example runs.
		fmt.Println("Best: Canada")
	}
	// Output: Best: Canada
}

func ExampleClient_CompletionStream() {
	// This code will run when CEREBRAS_API_KEY is set.
	// As of March 2025, you can try it out for free.
	// Cerebras supports a limited set of models.
	if c, err := cerebras.New("", "llama3.1-8b"); err == nil {
		ctx := context.Background()
		msgs := []genaiapi.Message{
			{
				Role: genaiapi.User,
				Type: genaiapi.Text,
				Text: "Say hello. Use only one word.",
			},
		}
		chunks := make(chan genaiapi.MessageChunk)
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
					resp += w.Text
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
	if c, err := cerebras.New("", ""); err == nil {
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
