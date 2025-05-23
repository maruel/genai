// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package deepseek_test

import (
	"context"
	"fmt"
	"log"

	"github.com/maruel/genai"
	"github.com/maruel/genai/deepseek"
)

func ExampleClient_Chat_jSON() {
	// DeepSeek doesn't have a small model. It's also quite slow (often 10s)
	// compared to other service providers.
	// See https://api-docs.deepseek.com/quick_start/pricing
	c, err := deepseek.New("", "deepseek-chat")
	if err != nil {
		log.Fatal(err)
	}
	msgs := genai.Messages{
		genai.NewTextMessage(genai.User, "Is a circle round? Reply as JSON with the form {\"round\": false} or {\"round\": true}."),
	}
	opts := genai.ChatOptions{
		Temperature: 0.01,
		MaxTokens:   50,
		ReplyAsJSON: true,
	}
	resp, err := c.Chat(context.Background(), msgs, &opts)
	if err != nil {
		log.Fatal(err)
	}
	log.Printf("Raw response: %#v", resp)
	var got struct {
		Round bool `json:"round"`
	}
	if err := resp.Decode(&got); err != nil {
		log.Fatal(err)
	}
	fmt.Printf("Round: %v\n", got.Round)
	// This would Output: Round: true
}
