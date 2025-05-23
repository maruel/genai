// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package togetherai_test

import (
	"context"
	_ "embed"
	"fmt"
	"log"
	"os"
	"strings"

	"github.com/maruel/genai"
	"github.com/maruel/genai/togetherai"
)

func ExampleClient_Chat_video() {
	// We must select a model that supports video.
	// https://docs.together.ai/docs/serverless-models#vision-models
	//
	// 2025-03-19: TogetherAI removed Qwen/Qwen2.5-VL-72B-Instruct and
	// Qwen/Qwen2-VL-72B-Instruct cannot process videos.
	c, err := togetherai.New("", "Qwen/Qwen2.5-VL-72B-Instruct")
	if err != nil {
		log.Fatal(err)
	}
	f, err := os.Open("testdata/animation.mp4")
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()
	// TogetherAI seems to require separate messages for text and images.
	msgs := genai.Messages{
		genai.NewTextMessage(genai.User, "What is the word? Reply with exactly and only one word."),
		{Role: genai.User, Contents: []genai.Content{{Document: f}}},
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
	fmt.Printf("Saw: %v\n", strings.ToLower(resp.AsText()))
	// This would Output: Saw: banana
}
