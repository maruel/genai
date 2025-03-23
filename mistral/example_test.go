// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package mistral_test

import (
	"bytes"
	"context"
	_ "embed"
	"errors"
	"fmt"
	"log"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/maruel/genai"
	"github.com/maruel/genai/mistral"
	"github.com/maruel/httpjson"
)

// See the 1kib banana jpg online at
// https://github.com/maruel/genai/blob/main/mistral/testdata/banana.jpg
//
//go:embed testdata/banana.jpg
var bananaJpg []byte

func ExampleClient_Chat_vision_and_JSON() {
	// Require a model which has the "vision" capability.
	// https://docs.mistral.ai/capabilities/vision/
	c, err := mistral.New("", "mistral-small-latest")
	if err != nil {
		log.Fatal(err)
	}
	msgs := genai.Messages{
		{
			Role: genai.User,
			Contents: []genai.Content{
				{Text: "Is it a banana? Reply as JSON."},
				{Filename: "banana.jpg", Document: bytes.NewReader(bananaJpg)},
			},
		},
	}
	var got struct {
		Banana bool `json:"banana"`
	}
	opts := genai.ChatOptions{
		Seed:        1,
		Temperature: 0.01,
		MaxTokens:   50,
		DecodeAs:    &got,
	}
	var resp genai.ChatResult
	for i := range 3 {
		// Mistral has a very good rate limiting implementation.
		if resp, err = c.Chat(context.Background(), msgs, &opts); err != nil && i != 2 {
			var herr *httpjson.Error
			if errors.As(err, &herr) {
				if herr.StatusCode == http.StatusTooManyRequests {
					fmt.Fprintf(os.Stderr, "Rate limited, waiting 2s\n")
					time.Sleep(2 * time.Second)
					continue
				}
			}
			log.Fatal(err)
		}
		break
	}
	if err != nil {
		log.Fatal(err)
	}
	log.Printf("Raw response: %#v", resp)
	if len(resp.Contents) != 1 {
		log.Fatal("Unexpected response")
	}
	if err := resp.Contents[0].Decode(&got); err != nil {
		log.Fatal(err)
	}
	fmt.Printf("Banana: %v\n", got.Banana)
	// This would Output: Banana: true
}

func ExampleClient_Chat_pDF() {
	// Require a model which has the "OCR" or the "Document understanding"
	// capability. There's a subtle difference between the two; from what I
	// understand, the document understanding will only parse the text, while the
	// OCR will try to understand the pictures.
	// https://docs.mistral.ai/capabilities/document/
	// https://docs.mistral.ai/capabilities/vision/
	c, err := mistral.New("", "mistral-small-latest")
	if err != nil {
		log.Fatal(err)
	}
	msgs := genai.Messages{
		{
			Role: genai.User,
			Contents: []genai.Content{
				{Text: "What is the word? Reply with only the word."},
				{URL: "https://raw.githubusercontent.com/maruel/genai/refs/heads/main/mistral/testdata/hidden_word.pdf"},
			},
		},
	}
	opts := genai.ChatOptions{
		Temperature: 0.01,
		MaxTokens:   50,
	}
	var resp genai.ChatResult
	for i := range 3 {
		// Mistral has a very good rate limiting implementation.
		if resp, err = c.Chat(context.Background(), msgs, &opts); err != nil && i != 2 {
			var herr *httpjson.Error
			if errors.As(err, &herr) {
				if herr.StatusCode == http.StatusTooManyRequests {
					fmt.Fprintf(os.Stderr, "Rate limited, waiting 2s\n")
					time.Sleep(2 * time.Second)
					continue
				}
			}
			log.Fatal(err)
		}
		break
	}
	if err != nil {
		log.Fatal(err)
	}
	log.Printf("Raw response: %#v", resp)
	// Mistral is super efficient with tokens for PDFs.
	if len(resp.Contents) != 1 {
		log.Fatal("Unexpected response")
	}
	fmt.Printf("Hidden word in PDF: %v\n", strings.ToLower(resp.Contents[0].Text))
	// This would Output: Hidden word in PDF: orange
}

func ExampleClient_Chat_tool_use() {
	// Require a model which has the tool capability. See
	// https://docs.mistral.ai/capabilities/function_calling/
	c, err := mistral.New("", "ministral-3b-latest")
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
		Seed:        1,
		Temperature: 0.01,
		MaxTokens:   200,
		Tools: []genai.ToolDef{
			{
				Name:        "best_country",
				Description: "A tool to determine the best country",
				InputsAs:    &got,
			},
		},
	}
	var resp genai.ChatResult
	for i := range 3 {
		// Mistral has a very good rate limiting implementation.
		if resp, err = c.Chat(context.Background(), msgs, &opts); err != nil && i != 2 {
			var herr *httpjson.Error
			if errors.As(err, &herr) {
				if herr.StatusCode == http.StatusTooManyRequests {
					fmt.Fprintf(os.Stderr, "Rate limited, waiting 2s\n")
					time.Sleep(2 * time.Second)
					continue
				}
			}
			log.Fatal(err)
		}
		break
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
	// Using very small model for testing.
	// See https://docs.mistral.ai/getting-started/models/models_overview/
	c, err := mistral.New("", "ministral-3b-latest")
	if err != nil {
		log.Fatal(err)
	}
	ctx := context.Background()
	msgs := genai.Messages{
		genai.NewTextMessage(genai.User, "Say hello. Use only one word."),
	}
	opts := genai.ChatOptions{
		Seed:        1,
		Temperature: 0.01,
		MaxTokens:   50,
	}
	for i := range 3 {
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
		err := c.ChatStream(ctx, msgs, &opts, chunks)
		close(chunks)
		var responses genai.Messages
		for m := range end {
			responses = append(responses, m)
		}
		if err != nil && i != 2 {
			// Mistral has a very good rate limiting implementation.
			var herr *httpjson.Error
			if errors.As(err, &herr) {
				if herr.StatusCode == http.StatusTooManyRequests {
					fmt.Fprintf(os.Stderr, "Rate limited, waiting 2s\n")
					time.Sleep(2 * time.Second)
					continue
				}
			}
			log.Fatal(err)
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
		break
	}
	// This would Output: Response: hello
}

func ExampleClient_ListModels() {
	// Print something so the example runs.
	fmt.Println("Got models")
	c, err := mistral.New("", "")
	if err != nil {
		fmt.Fprintf(os.Stderr, "Couldn't connect: %v\n", err)
		return
	}
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
	// Output: Got models
}
