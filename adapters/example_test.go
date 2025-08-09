// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package adapters_test

import (
	"context"
	"fmt"
	"log"
	"os"

	"github.com/maruel/genai"
	"github.com/maruel/genai/adapters"
	"github.com/maruel/genai/genaitools"
	"github.com/maruel/genai/providers/gemini"
)

func ExampleGenSyncWithToolCallLoop() {
	// Supported by Anthropic, Cerebras, Cloudflare, Cohere, DeepSeek, Gemini, Groq, HuggingFace, Mistral,
	// Ollama, OpenAI, TogetherAI.

	// Using a free small model for testing.
	// See https://ai.google.dev/gemini-api/docs/models/gemini?hl=en
	c, err := gemini.New(&genai.OptionsProvider{Model: "gemini-2.0-flash"}, nil)
	if err != nil {
		log.Fatal(err)
	}
	msgs := genai.Messages{
		genai.NewTextMessage("What is 3214 + 5632? Leverage the tool available to you to tell me the answer. Do not explain. Be terse. Include only the answer."),
	}
	opts := genai.OptionsText{
		Tools: []genai.ToolDef{genaitools.Arithmetic},
		// Force the LLM to do a tool call first.
		ToolCallRequest: genai.ToolCallRequired,
	}
	newMsgs, _, err := adapters.GenSyncWithToolCallLoop(context.Background(), c, msgs, &opts)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("%s\n", newMsgs[len(newMsgs)-1].AsText())
	// Remove this comment line to run the example.
	// Output:
	// 8846
}

func ExampleGenStreamWithToolCallLoop() {
	// Supported by Anthropic, Cerebras, Cloudflare, Cohere, DeepSeek, Gemini, Groq, HuggingFace, Mistral,
	// Ollama, OpenAI, TogetherAI.

	// Using a free small model for testing.
	// See https://ai.google.dev/gemini-api/docs/models/gemini?hl=en
	c, err := gemini.New(&genai.OptionsProvider{Model: "gemini-2.0-flash"}, nil)
	if err != nil {
		log.Fatal(err)
	}
	msgs := genai.Messages{
		genai.NewTextMessage("What is 3214 + 5632? Leverage the tool available to you to tell me the answer. Do not explain. Be terse. Include only the answer."),
	}
	opts := genai.OptionsText{
		Tools: []genai.ToolDef{genaitools.Arithmetic},
		// Force the LLM to do a tool call first.
		ToolCallRequest: genai.ToolCallRequired,
	}
	chunks := make(chan genai.ContentFragment)
	ctx, cancel := context.WithCancel(context.Background())
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
	// Remove this comment line to run the example.
	// Output:
	// 8846
}
