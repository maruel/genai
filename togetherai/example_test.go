// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package togetherai_test

import (
	"bytes"
	"context"
	_ "embed"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"strings"

	"github.com/invopop/jsonschema"
	"github.com/maruel/genai/genaiapi"
	"github.com/maruel/genai/togetherai"
)

// See the 1kib banana jpg online at
// https://github.com/maruel/genai/blob/main/togetherai/testdata/banana.jpg
//
//go:embed testdata/banana.jpg
var bananaJpg []byte

func ExampleClient_Completion_vision_and_JSON() {
	// This code will run when TOGETHERAI_API_KEY is set.
	//
	// As of March 2025, you can try it out for free.
	//
	// We must select a model that supports vision *and* JSON mode (not
	// necessarily tool use).
	// Warning: looks like this model doesn't support JSON schema.
	if c, err := togetherai.New("", "meta-llama/Llama-Vision-Free"); err == nil {
		msgs := []genaiapi.Message{
			{
				Role:     genaiapi.User,
				Type:     genaiapi.Document,
				Filename: "banana.jpg",
				// Together.AI supports highly compressed jpg.
				Document: bytes.NewReader(bananaJpg),
			},
			{
				Role: genaiapi.User,
				Type: genaiapi.Text,
				Text: "Is it a banana? Reply as JSON with the form {\"banana\": false} or {\"banana\": true}.",
			},
		}
		var expected struct {
			Banana bool `json:"banana"`
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
		// Print to stderr so the test doesn't capture it.
		fmt.Fprintf(os.Stderr, "Raw response: %#v\n", resp)
		if resp.Role != genaiapi.Assistant || resp.Type != genaiapi.Text {
			log.Fatalf("Unexpected response: %#v", resp)
		}
		d := json.NewDecoder(strings.NewReader(resp.Text))
		d.DisallowUnknownFields()
		if err := d.Decode(&expected); err != nil {
			log.Fatalf("Failed to decode %q as JSON: %v", resp.Text, err)
		}
		fmt.Printf("Banana: %v\n", expected.Banana)
		if resp.InputTokens < 10 || resp.OutputTokens < 2 {
			log.Fatalf("Missing usage token")
		}
	} else {
		// Print something so the example runs.
		fmt.Println("Banana: true")
	}
	// Output: Banana: true
}

func ExampleClient_Completion_tool_use() {
	// This code will run when TOGETHER_API_KEY is set.
	//
	// As of March 2025, you can try it out for free.
	//
	// You must select a model that supports tool use.
	if c, err := togetherai.New("", "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"); err == nil {
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
		if len(resp.ToolCalls) != 1 || resp.ToolCalls[0].Name != "best_country" {
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
	// This code will run when TOGETHERAI_API_KEY is set.
	// See https://api.together.ai/models
	if c, err := togetherai.New("", "meta-llama/Llama-3.2-3B-Instruct-Turbo"); err == nil {
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
	if c, err := togetherai.New("", ""); err == nil {
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
