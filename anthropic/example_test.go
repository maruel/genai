// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package anthropic_test

import (
	"bytes"
	"context"
	_ "embed"
	"fmt"
	"log"
	"os"
	"strings"

	"github.com/maruel/genai"
	"github.com/maruel/genai/anthropic"
)

// See the 3kib banana jpg online at
// https://github.com/maruel/genai/blob/main/anthropic/testdata/banana.jpg
//
//go:embed testdata/banana.jpg
var bananaJpg []byte

func ExampleClient_Chat_vision() {
	// Using very small model for testing. As of March 2025,
	// claude-3-haiku-20240307 is 0.20$/1.25$ while claude-3-5-haiku-20241022 is
	// 0.80$/4.00$. 3.0 supports images, 3.5 supports PDFs.
	// https://docs.anthropic.com/en/docs/about-claude/models/all-models
	c, err := anthropic.New("", "claude-3-haiku-20240307")
	if err != nil {
		log.Fatal(err)
	}
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
	resp, err := c.Chat(context.Background(), msgs, &opts)
	if err != nil {
		log.Fatal(err)
	}
	log.Printf("Raw response: %#v", resp)
	if len(resp.Contents) != 1 {
		log.Fatal("Unexpected response")
	}
	// Normalize some of the variance. Obviously many models will still fail this test.
	txt := strings.TrimRight(strings.TrimSpace(strings.ToLower(resp.Contents[0].Text)), ".!")
	fmt.Printf("Response: %s\n", txt)
	// This would Output: Response: yes
}

func ExampleClient_Chat_pDF() {
	// Claude 3.5 is required for PDF input.
	c, err := anthropic.New("", "claude-3-5-haiku-20241022")
	if err != nil {
		log.Fatal(err)
	}
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
	if len(resp.Contents) != 1 {
		log.Fatal("Unexpected response")
	}
	fmt.Printf("Hidden word in PDF: %v\n", strings.ToLower(resp.Contents[0].Text))
	// This would Output: Hidden word in PDF: orange
}

func ExampleClient_Chat_tool_use() {
	// Claude 3.5 is required for tool use.
	c, err := anthropic.New("", "claude-3-5-haiku-20241022")
	if err != nil {
		log.Fatal(err)
	}
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
	if len(resp.ToolCalls) != 1 || resp.ToolCalls[0].Name != "best_country" {
		log.Fatal("Unexpected response")
	}
	if err := resp.ToolCalls[0].Decode(&got); err != nil {
		log.Fatal(err)
	}
	fmt.Printf("Best: %v\n", got.Country)
	// This would Output: Best: Canada
}

func ExampleClient_ChatStream() {
	c, err := anthropic.New("", "claude-3-haiku-20240307")
	if err != nil {
		log.Fatal(err)
	}
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
				var err2 error
				if pendingMsgs, err2 = pkt.Accumulate(pendingMsgs); err2 != nil {
					end <- genai.NewTextMessage(genai.Assistant, fmt.Sprintf("Error: %v", err2))
					return
				}
			}
		}
	}()
	err = c.ChatStream(ctx, msgs, &opts, chunks)
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
	// This would Output: Response: hello
}

func ExampleClient_ListModels() {
	c, err := anthropic.New("", "")
	if err != nil {
		fmt.Printf("Couldn't connect: %v\n", err)
		return
	}
	models, err := c.ListModels(context.Background())
	if err != nil {
		fmt.Printf("Failed to get models: %v\n", err)
		return
	}
	for _, model := range models {
		fmt.Printf("- %s\n", model)
	}
}
