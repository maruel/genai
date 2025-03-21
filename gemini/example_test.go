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
	"strings"

	"github.com/maruel/genai"
	"github.com/maruel/genai/gemini"
)

// See the 1kib banana jpg online at
// https://github.com/maruel/genai/blob/main/gemini/testdata/banana.jpg
//
//go:embed testdata/banana.jpg
var bananaJpg []byte

// Using small model for testing.
// See https://ai.google.dev/gemini-api/docs/models/gemini?hl=en
const model = "gemini-2.0-flash-lite"

func ExampleClient_Chat_vision_and_JSON() {
	// This code will run when GEMINI_API_KEY is set.
	// As of March 2025, you can try it out for free.
	if c, err := gemini.New("", model); err == nil {
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
		resp, err := c.Chat(context.Background(), msgs, &opts)
		if err != nil {
			log.Fatal(err)
		}
		log.Printf("Raw response: %#v", resp)
		if resp.InputTokens != 268 || resp.OutputTokens != 9 {
			log.Printf("Unexpected tokens usage: %v", resp.Usage)
		}
		if len(resp.Contents) != 1 {
			log.Fatal("Unexpected response")
		}
		if err := resp.Contents[0].Decode(&got); err != nil {
			log.Fatal(err)
		}
		fmt.Printf("Banana: %v\n", got.Banana)
	} else {
		// Print something so the example runs.
		fmt.Println("Banana: true")
	}
	// Output: Banana: true
}

func ExampleClient_Chat_pDF() {
	// This code will run when GEMINI_API_KEY is set.
	// As of March 2025, you can try it out for free.
	if c, err := gemini.New("", model); err == nil {
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
			Seed:        1,
			Temperature: 0.01,
			MaxTokens:   50,
		}
		resp, err := c.Chat(context.Background(), msgs, &opts)
		if err != nil {
			log.Fatal(err)
		}
		log.Printf("Raw response: %#v", resp)
		if resp.InputTokens != 1301 || resp.OutputTokens != 2 {
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

func ExampleClient_Chat_audio() {
	// This code will run when GEMINI_API_KEY is set.
	// As of March 2025, you can try it out for free.
	if c, err := gemini.New("", model); err == nil {
		f, err := os.Open("testdata/mystery_word.opus")
		if err != nil {
			log.Fatal(err)
		}
		defer f.Close()
		msgs := genai.Messages{
			{
				Role: genai.User,
				Contents: []genai.Content{
					{Text: "What is the word said? Reply with only the word."},
					{Document: f},
				},
			},
		}
		opts := genai.ChatOptions{
			Seed:        1,
			Temperature: 0.01,
			MaxTokens:   50,
		}
		resp, err := c.Chat(context.Background(), msgs, &opts)
		if err != nil {
			log.Fatal(err)
		}
		log.Printf("Raw response: %#v", resp)
		if resp.InputTokens != 12 || resp.OutputTokens != 2 {
			log.Printf("Unexpected tokens usage: %v", resp.Usage)
		}
		if len(resp.Contents) != 1 {
			log.Fatal("Unexpected response")
		}
		fmt.Printf("Heard: %v\n", strings.TrimRight(strings.ToLower(resp.Contents[0].Text), "."))
	} else {
		// Print something so the example runs.
		fmt.Println("Heard: orange")
	}
	// Output: Heard: orange
}

func ExampleClient_Chat_tool_use() {
	// This code will run when GEMINI_API_KEY is set.
	// As of March 2025, you can try it out for free.
	if c, err := gemini.New("", model); err == nil {
		f, err := os.Open("testdata/animation.mp4")
		if err != nil {
			log.Fatal(err)
		}
		defer f.Close()
		msgs := genai.Messages{
			{
				Role: genai.User,
				Contents: []genai.Content{
					{Text: "What is the word? Call the tool hidden_word to tell me what word you saw."},
					{Document: f},
				},
			},
		}
		var got struct {
			Word string `json:"word" jsonschema:"enum=Orange,enum=Banana,enum=Apple"`
		}
		opts := genai.ChatOptions{
			Seed:        1,
			Temperature: 0.01,
			MaxTokens:   50,
			Tools: []genai.ToolDef{
				{
					Name:        "hidden_word",
					Description: "A tool to state what word was seen in the video.",
					InputsAs:    &got,
				},
			},
		}
		resp, err := c.Chat(context.Background(), msgs, &opts)
		if err != nil {
			log.Fatal(err)
		}
		log.Printf("Raw response: %#v", resp)
		if resp.InputTokens != 1079 || resp.OutputTokens != 5 {
			log.Printf("Unexpected tokens usage: %v", resp.Usage)
		}
		// Warning: there's a bug where it returns two identical tool calls. To verify.
		if len(resp.ToolCalls) == 0 || resp.ToolCalls[0].Name != "hidden_word" {
			log.Fatal("Unexpected response")
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

func ExampleClient_ChatStream() {
	// This code will run when GEMINI_API_KEY is set.
	// As of March 2025, you can try it out for free.
	if c, err := gemini.New("", model); err == nil {
		ctx := context.Background()
		msgs := genai.Messages{
			genai.NewTextMessage(genai.User, "Say hello. Use only one word."),
		}
		opts := genai.ChatOptions{
			Seed:        1,
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
			log.Fatal("Unexpected responses")
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
