// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package gemini_test

import (
	"bytes"
	"context"
	_ "embed"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strings"

	"github.com/maruel/genai/gemini"
	"github.com/maruel/genai/genaiapi"
)

// See the 1kib banana jpg online at
// https://github.com/maruel/genai/blob/main/gemini/testdata/banana.jpg
//
//go:embed testdata/banana.jpg
var bananaJpg []byte

// Using small model for testing.
// See https://ai.google.dev/gemini-api/docs/models/gemini?hl=en
var model = "gemini-2.0-flash-lite"

func ExampleClient_Completion_vision_and_JSON() {
	// This code will run when GEMINI_API_KEY is set.
	// As of March 2025, you can try it out for free.
	if c, err := gemini.New("", model); err == nil {
		msgs := genaiapi.Messages{
			{
				Role: genaiapi.User,
				Type: genaiapi.Document,
				// Gemini supports highly compressed jpg.
				Filename: "banana.jpg",
				Document: bytes.NewReader(bananaJpg),
			},
			genaiapi.NewTextMessage(genaiapi.User, "Say hello. Use only one word."),
		}
		var got struct {
			Banana bool `json:"banana"`
		}
		opts := genaiapi.CompletionOptions{
			Seed:        1,
			Temperature: 0.01,
			MaxTokens:   50,
			DecodeAs:    &got,
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
		fmt.Printf("Banana: %v\n", got.Banana)
		if resp.InputTokens < 100 || resp.OutputTokens < 2 {
			log.Fatalf("Missing usage token")
		}
	} else {
		// Print something so the example runs.
		fmt.Println("Banana: true")
	}
	// Output: Banana: true
}

func ExampleClient_Completion_pDF() {
	// This code will run when GEMINI_API_KEY is set.
	// As of March 2025, you can try it out for free.
	if c, err := gemini.New("", model); err == nil {
		f, err := os.Open("testdata/hidden_word.pdf")
		if err != nil {
			log.Fatal(err)
		}
		defer f.Close()
		msgs := genaiapi.Messages{
			{
				Role:     genaiapi.User,
				Type:     genaiapi.Document,
				Filename: filepath.Base(f.Name()),
				Document: f,
			},
			genaiapi.NewTextMessage(genaiapi.User, "What is the word? Reply with only the word."),
		}
		opts := genaiapi.CompletionOptions{
			Seed:        1,
			Temperature: 0.01,
			MaxTokens:   50,
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
		fmt.Printf("Hidden word in PDF: %v\n", resp.Text)
		if resp.InputTokens < 100 || resp.OutputTokens < 2 {
			log.Fatalf("Missing usage token")
		}
	} else {
		// Print something so the example runs.
		fmt.Println("Hidden word in PDF: orange")
	}
	// Output: Hidden word in PDF: orange
}

func ExampleClient_Completion_audio() {
	// This code will run when GEMINI_API_KEY is set.
	// As of March 2025, you can try it out for free.
	if c, err := gemini.New("", model); err == nil {
		f, err := os.Open("testdata/mystery_word.opus")
		if err != nil {
			log.Fatal(err)
		}
		defer f.Close()
		msgs := genaiapi.Messages{
			{
				Role:     genaiapi.User,
				Type:     genaiapi.Document,
				Filename: filepath.Base(f.Name()),
				Document: f,
			},
			genaiapi.NewTextMessage(genaiapi.User, "What is the word said? Reply with only the word."),
		}
		opts := genaiapi.CompletionOptions{
			Seed:        1,
			Temperature: 0.01,
			MaxTokens:   50,
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
		fmt.Printf("Said: %v\n", strings.TrimRight(strings.ToLower(resp.Text), "."))
		if resp.InputTokens < 10 || resp.OutputTokens < 2 {
			log.Fatalf("Missing usage token")
		}
	} else {
		// Print something so the example runs.
		fmt.Println("Said: orange")
	}
	// Output: Said: orange
}

func ExampleClient_Completion_tool_use() {
	// This code will run when GEMINI_API_KEY is set.
	// As of March 2025, you can try it out for free.
	if c, err := gemini.New("", model); err == nil {
		f, err := os.Open("testdata/animation.mp4")
		if err != nil {
			log.Fatal(err)
		}
		defer f.Close()
		msgs := genaiapi.Messages{
			{
				Role:     genaiapi.User,
				Type:     genaiapi.Document,
				Filename: filepath.Base(f.Name()),
				Document: f,
			},
			genaiapi.NewTextMessage(genaiapi.User, "What is the word? Call the tool hidden_word to tell me what word you saw."),
		}
		var got struct {
			Word string `json:"word" jsonschema:"enum=Orange,enum=Banana,enum=Apple"`
		}
		opts := genaiapi.CompletionOptions{
			Seed:        1,
			Temperature: 0.01,
			MaxTokens:   50,
			Tools: []genaiapi.ToolDef{
				{
					Name:        "hidden_word",
					Description: "A tool to state what word was seen in the video.",
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
		// Warning: there's a bug where it returns two identical tool calls. To verify.
		if len(resp.ToolCalls) == 0 || resp.ToolCalls[0].Name != "hidden_word" {
			log.Fatal("Expected 1 best_country tool call")
		}
		if err := resp.ToolCalls[0].Decode(&got); err != nil {
			log.Fatal(err)
		}
		fmt.Printf("Saw: %v\n", strings.ToLower(got.Word))
	} else {
		// Print something so the example runs.
		fmt.Println("Saw: banana")
	}
	// Output: Saw: banana
}

func ExampleClient_CompletionStream() {
	// This code will run when GEMINI_API_KEY is set.
	// As of March 2025, you can try it out for free.
	if c, err := gemini.New("", model); err == nil {
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
	if c, err := gemini.New("", ""); err == nil {
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
