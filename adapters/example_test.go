// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package adapters_test

import (
	"context"
	"fmt"
	"log"
	"os"
	"time"

	"github.com/maruel/genai"
	"github.com/maruel/genai/adapters"
	"github.com/maruel/genai/providers/gemini"
)

func ExampleGenSyncWithToolCallLoop() {
	// Supported by Anthropic, Cerebras, Cloudflare, Cohere, DeepSeek, Gemini, Groq, HuggingFace, Mistral,
	// Ollama, OpenAI, TogetherAI.

	// Using a free small model for testing.
	// See https://ai.google.dev/gemini-api/docs/models/gemini?hl=en
	ctx := context.Background()
	c, err := gemini.New(ctx, &genai.ProviderOptions{Model: "gemini-2.5-flash"}, nil)
	if err != nil {
		log.Fatal(err)
	}
	msgs := genai.Messages{genai.NewTextMessage("What season are we in?")}
	opts := genai.OptionsText{
		// GetTodayClockTime returns the current time and day in a format that the LLM
		// can understand. It includes the weekend.
		Tools: []genai.ToolDef{GetTodayClockTime},
		// Force the LLM to do a tool call first.
		ToolCallRequest: genai.ToolCallRequired,
	}
	newMsgs, _, err := adapters.GenSyncWithToolCallLoop(ctx, c, msgs, &opts)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("%s\n", newMsgs[len(newMsgs)-1].String())
}

func ExampleGenStreamWithToolCallLoop() {
	// Supported by Anthropic, Cerebras, Cloudflare, Cohere, DeepSeek, Gemini, Groq, HuggingFace, Mistral,
	// Ollama, OpenAI, TogetherAI.

	// Using a free small model for testing.
	// See https://ai.google.dev/gemini-api/docs/models/gemini?hl=en
	ctx := context.Background()
	c, err := gemini.New(ctx, &genai.ProviderOptions{Model: "gemini-2.5-flash"}, nil)
	if err != nil {
		log.Fatal(err)
	}
	msgs := genai.Messages{genai.NewTextMessage("What season are we in?")}
	opts := genai.OptionsText{
		// GetTodayClockTime returns the current time and day in a format that the LLM
		// can understand. It includes the weekend.
		Tools: []genai.ToolDef{GetTodayClockTime},
		// Force the LLM to do a tool call first.
		ToolCallRequest: genai.ToolCallRequired,
	}
	chunks := make(chan genai.ReplyFragment)
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()
	go func() {
		for {
			select {
			case <-ctx.Done():
				return
			case fragment, ok := <-chunks:
				if !ok {
					return
				}
				_, _ = os.Stdout.WriteString(fragment.TextFragment)
			}
		}
	}()
	_, _, err = adapters.GenStreamWithToolCallLoop(ctx, c, msgs, chunks, &opts)
	if err != nil {
		log.Fatal(err)
	}
}

var GetTodayClockTime = genai.ToolDef{
	Name:        "get_today_date_current_clock_time",
	Description: "Get the current clock time and today's date.",
	Callback: func(ctx context.Context, e *empty) (string, error) {
		return time.Now().Format("Monday 2006-01-02 15:04:05"), nil
	},
}

type empty struct{}
