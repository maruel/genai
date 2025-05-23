// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package cloudflare_test

import (
	"context"
	"fmt"
	"log"
	"strings"

	"github.com/maruel/genai"
	"github.com/maruel/genai/cloudflare"
)

func ExampleClient_Chat_jSON() {
	// We need to use a model that supports structured output.
	c, err := cloudflare.New("", "", "@hf/nousresearch/hermes-2-pro-mistral-7b")
	if err != nil {
		log.Fatal(err)
	}
	msgs := genai.Messages{
		genai.NewTextMessage(genai.User, "Is a circle round? Reply as JSON."),
	}
	var got struct {
		Round bool `json:"round"`
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
	if resp.InputTokens != 0 || resp.OutputTokens != 0 {
		log.Printf("Did cloudflare finally start filling the usage fields?")
	}
	if err := resp.Decode(&got); err != nil {
		log.Fatal(err)
	}
	fmt.Printf("Round: %v\n", got.Round)
	// This would Output: Round: true
}

func ExampleClient_ChatStream() {
	// Using very small model for testing.
	// See https://developers.cloudflare.com/workers-ai/models/
	c, err := cloudflare.New("", "", "@cf/meta/llama-3.2-3b-instruct")
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
	_, err = c.ChatStream(ctx, msgs, &opts, chunks)
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
	// Normalize some of the variance. Obviously many models will still fail this test.
	fmt.Printf("Response: %s\n", strings.TrimRight(strings.TrimSpace(strings.ToLower(resp.AsText())), ".!"))
	// This would Output: Response: hello
}
