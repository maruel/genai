// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package anthropic_test

import (
	"bytes"
	"context"
	_ "embed"
	"errors"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/maruel/genai/anthropic"
	"github.com/maruel/genai/genaiapi"
	"github.com/maruel/httpjson"
)

// See the 3kib banana jpg online at
// https://github.com/maruel/genai/blob/main/anthropic/testdata/banana.jpg
//
//go:embed testdata/banana.jpg
var bananaJpg []byte

// Using very small model for testing. As of March 2025,
// claude-3-haiku-20240307 is 0.20$/1.25$ while claude-3-5-haiku-20241022 is
// 0.80$/4.00$. 3.0 supports images, 3.5 supports PDFs.
// https://docs.anthropic.com/en/docs/about-claude/models/all-models
var model = "claude-3-haiku-20240307"

func ExampleClient_Completion_vision() {
	// This code will run when ANTHROPIC_API_KEY is set.
	if c, err := anthropic.New("", model); err == nil {
		msgs := []genaiapi.Message{
			{
				Role:     genaiapi.User,
				Type:     genaiapi.Document,
				Filename: "banana.jpg",
				// Anthropic requires higher quality image than Gemini or Mistral. See
				// ../gemini/testdata/banana.jpg to compare.
				Document: bytes.NewReader(bananaJpg),
			},
			{
				Role: genaiapi.User,
				Type: genaiapi.Text,
				Text: "Is it a banana? Reply with only one word.",
			},
		}
		opts := genaiapi.CompletionOptions{
			Temperature: 0.01,
			MaxTokens:   50,
		}
		for i := range 3 {
			resp, err := c.Completion(context.Background(), msgs, &opts)
			if err != nil {
				var herr *httpjson.Error
				// See https://docs.anthropic.com/en/api/errors#http-errors
				if errors.As(err, &herr) && herr.StatusCode == 529 && i != 2 {
					log.Printf("retrying after 2s")
					time.Sleep(2 * time.Second)
					continue
				}
				log.Fatal(err)
			}
			// Print to stderr so the test doesn't capture it.
			fmt.Fprintf(os.Stderr, "Raw response: %#v\n", resp)
			if resp.Role != genaiapi.Assistant || resp.Type != genaiapi.Text {
				log.Fatalf("Unexpected response: %#v", resp)
			}
			// Normalize some of the variance. Obviously many models will still fail this test.
			txt := strings.TrimRight(strings.TrimSpace(strings.ToLower(resp.Text)), ".!")
			fmt.Printf("Response: %s\n", txt)
			if resp.InputTokens < 100 || resp.OutputTokens < 2 {
				log.Fatalf("Missing usage token")
			}
			break
		}
	} else {
		// Print something so the example runs.
		fmt.Println("Response: yes")
	}
	// Output: Response: yes
}

func ExampleClient_Completion_pDF() {
	// This code will run when ANTHROPIC_API_KEY is set.
	if c, err := anthropic.New("", "claude-3-5-haiku-20241022"); err == nil {
		f, err := os.Open("testdata/hidden_word.pdf")
		if err != nil {
			log.Fatal(err)
		}
		defer f.Close()
		msgs := []genaiapi.Message{
			{
				Role:     genaiapi.User,
				Type:     genaiapi.Document,
				Filename: filepath.Base(f.Name()),
				Document: f,
			},
			{
				Role: genaiapi.User,
				Type: genaiapi.Text,
				Text: "What is the hidden word? Reply with only the word.",
			},
		}
		opts := genaiapi.CompletionOptions{
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

func ExampleClient_Completion_tool_use() {
	// This code will run when ANTHROPIC_API_KEY is set.
	// Claude 3.5 is required for tool use. ? "claude-3-5-haiku-20241022"
	if c, err := anthropic.New("", model); err == nil {
		msgs := []genaiapi.Message{
			{
				Role: genaiapi.User,
				Type: genaiapi.Text,
				Text: "I wonder if Canada is a better country than the US? Call the tool best_country to tell me which country is the best one.",
			},
		}
		var got struct {
			Country string `json:"country" jsonschema:"enum=Canada,enum=USA"`
		}
		opts := genaiapi.CompletionOptions{
			Temperature: 0.01,
			MaxTokens:   50,
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
		if len(resp.ToolCalls) != 1 || resp.ToolCalls[0].Name != "best_country" {
			log.Fatal("Expected 1 best_country tool call")
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
	// This code will run when ANTHROPIC_API_KEY is set.
	if c, err := anthropic.New("", model); err == nil {
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
	if c, err := anthropic.New("", ""); err == nil {
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
