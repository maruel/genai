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
	"strings"
	"time"

	"github.com/maruel/genai"
	"github.com/maruel/genai/anthropic"
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

func ExampleClient_Chat_vision() {
	// This code will run when ANTHROPIC_API_KEY is set.
	if c, err := anthropic.New("", model); err == nil {
		msgs := genai.Messages{
			{
				Role: genai.User,
				Contents: []genai.Content{
					{Text: "Is it a banana? Reply with only one word."},
					{Filename: "banana.jpg", Document: bytes.NewReader(bananaJpg)},
				},
			},
		}
		opts := genai.ChatOptions{
			Temperature: 0.01,
			MaxTokens:   50,
		}
		for i := range 3 {
			resp, err := c.Chat(context.Background(), msgs, &opts)
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
			log.Printf("Raw response: %#v", resp)
			if resp.InputTokens != 237 || resp.OutputTokens != 5 {
				log.Printf("Unexpected tokens usage: %v", resp.Usage)
			}
			if len(resp.Contents) != 1 {
				log.Fatal("Unexpected response")
			}
			// Normalize some of the variance. Obviously many models will still fail this test.
			txt := strings.TrimRight(strings.TrimSpace(strings.ToLower(resp.Contents[0].Text)), ".!")
			fmt.Printf("Response: %s\n", txt)
			break
		}
	} else {
		// Print something so the example runs.
		fmt.Println("Response: yes")
	}
	// Output: Response: yes
}

func ExampleClient_Chat_pDF() {
	// This code will run when ANTHROPIC_API_KEY is set.
	if c, err := anthropic.New("", "claude-3-5-haiku-20241022"); err == nil {
		f, err := os.Open("testdata/hidden_word.pdf")
		if err != nil {
			log.Fatal(err)
		}
		defer f.Close()
		msgs := genai.Messages{
			{
				Role: genai.User,
				Contents: []genai.Content{
					{Text: "What is the word? Reply with only the word."},
					{Document: f},
				},
			},
		}
		opts := genai.ChatOptions{
			Temperature: 0.01,
			MaxTokens:   50,
		}
		resp, err := c.Chat(context.Background(), msgs, &opts)
		if err != nil {
			log.Fatal(err)
		}
		log.Printf("Raw response: %#v", resp)
		if resp.InputTokens != 1628 || resp.OutputTokens != 4 {
			log.Printf("Unexpected tokens usage: %v", resp.Usage)
		}
		if len(resp.Contents) != 1 {
			log.Fatal("Unexpected response")
		}
		fmt.Printf("Hidden word in PDF: %v\n", strings.ToLower(resp.Contents[0].Text))
	} else {
		// Print something so the example runs.
		fmt.Println("Hidden word in PDF: orange")
	}
	// Output: Hidden word in PDF: orange
}

func ExampleClient_Chat_tool_use() {
	// This code will run when ANTHROPIC_API_KEY is set.
	// Claude 3.5 is required for tool use. ? "claude-3-5-haiku-20241022"
	if c, err := anthropic.New("", model); err == nil {
		msgs := genai.Messages{
			genai.NewTextMessage(genai.User, "I wonder if Canada is a better country than the US? Call the tool best_country to tell me which country is the best one."),
		}
		var got struct {
			Country string `json:"country" jsonschema:"enum=Canada,enum=USA"`
		}
		opts := genai.ChatOptions{
			Temperature: 0.01,
			MaxTokens:   50,
			Tools: []genai.ToolDef{
				{
					Name:        "best_country",
					Description: "A tool to determine the best country",
					InputsAs:    &got,
				},
			},
		}
		resp, err := c.Chat(context.Background(), msgs, &opts)
		if err != nil {
			log.Fatal(err)
		}
		log.Printf("Raw response: %#v", resp)
		if resp.InputTokens != 483 || resp.OutputTokens != 38 {
			log.Printf("Unexpected tokens usage: %v", resp.Usage)
		}
		if len(resp.ToolCalls) != 1 || resp.ToolCalls[0].Name != "best_country" {
			log.Fatal("Unexpected response")
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

func ExampleClient_ChatStream() {
	// This code will run when ANTHROPIC_API_KEY is set.
	if c, err := anthropic.New("", model); err == nil {
		ctx := context.Background()
		msgs := genai.Messages{
			genai.NewTextMessage(genai.User, "Say hello. Use only one word."),
		}
		opts := genai.ChatOptions{
			Temperature: 0.01,
			MaxTokens:   50,
		}
		chunks := make(chan genai.MessageFragment)
		end := make(chan genai.Message, 10)
		go func() {
			var pendingMsgs genai.Messages
			defer func() {
				for _, m := range pendingMsgs {
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
					if pendingMsgs, err = pkt.Accumulate(pendingMsgs); err != nil {
						end <- genai.NewTextMessage(genai.Assistant, fmt.Sprintf("Error: %v", err))
						return
					}
				}
			}
		}()
		err := c.ChatStream(ctx, msgs, &opts, chunks)
		close(chunks)
		var responses genai.Messages
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
