// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package genai_test

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"log"
	"os"
	"strings"

	"github.com/maruel/genai"
	"github.com/maruel/genai/anthropic"
	"github.com/maruel/genai/cerebras"
	"github.com/maruel/genai/cloudflare"
	"github.com/maruel/genai/cohere"
	"github.com/maruel/genai/deepseek"
	"github.com/maruel/genai/gemini"
	"github.com/maruel/genai/groq"
	"github.com/maruel/genai/huggingface"
	"github.com/maruel/genai/llamacpp"
	"github.com/maruel/genai/mistral"
	"github.com/maruel/genai/openai"
	"github.com/maruel/genai/perplexity"
	"github.com/maruel/genai/togetherai"
)

func ExampleModelProvider_all() {
	modelProviders := map[string]genai.ModelProvider{}
	if c, err := anthropic.New("", ""); err == nil {
		modelProviders["anthropic"] = c
	}
	if c, err := cerebras.New("", ""); err == nil {
		modelProviders["cerebras"] = c
	}
	if c, err := cloudflare.New("", "", ""); err == nil {
		modelProviders["cloudflare"] = c
	}
	if c, err := cohere.New("", ""); err == nil {
		modelProviders["cohere"] = c
	}
	if c, err := deepseek.New("", ""); err == nil {
		modelProviders["deepseek"] = c
	}
	if c, err := gemini.New("", ""); err == nil {
		modelProviders["gemini"] = c
	}
	if c, err := groq.New("", ""); err == nil {
		modelProviders["groq"] = c
	}
	if c, err := huggingface.New("", ""); err == nil {
		modelProviders["huggingface"] = c
	}
	// llamapcpp doesn't implement ModelProvider.
	if c, err := mistral.New("", ""); err == nil {
		modelProviders["mistral"] = c
	}
	if c, err := openai.New("", ""); err == nil {
		modelProviders["openai"] = c
	}
	// perplexity doesn't implement ModelProvider.
	if c, err := togetherai.New("", ""); err == nil {
		modelProviders["togetherai"] = c
	}

	for name, p := range modelProviders {
		models, err := p.ListModels(context.Background())
		fmt.Printf("%s:\n", name)
		if err != nil {
			fmt.Printf("  Failed to get models: %v\n", err)
		}
		for _, model := range models {
			fmt.Printf("- %s\n", model)
		}
	}
}

func ExampleChatProvider_all() {
	chatProviders := map[string]genai.ChatProvider{}
	// https://docs.anthropic.com/en/docs/about-claude/models/all-models
	if c, err := anthropic.New("", "claude-3-7-sonnet-latest"); err == nil {
		chatProviders["anthropic"] = c
	}
	// https://inference-docs.cerebras.ai/api-reference/models
	if c, err := cerebras.New("", "llama-3.3-70b"); err == nil {
		chatProviders["cerebras"] = c
	}
	// https://developers.cloudflare.com/workers-ai/models/
	if c, err := cloudflare.New("", "", "@cf/deepseek-ai/deepseek-r1-distill-qwen-32b"); err == nil {
		chatProviders["cloudflare"] = c
	}
	// https://docs.cohere.com/v2/docs/models
	if c, err := cohere.New("", "command-r-plus"); err == nil {
		chatProviders["cohere"] = c
	}
	// https://api-docs.deepseek.com/quick_start/pricing
	if c, err := deepseek.New("", "deepseek-reasoner"); err == nil {
		chatProviders["deepseek"] = c
	}
	// https://ai.google.dev/gemini-api/docs/models/gemini
	if c, err := gemini.New("", "gemini-2.0-flash"); err == nil {
		chatProviders["gemini"] = c
	}
	// https://console.groq.com/docs/models
	if c, err := groq.New("", "qwen-qwq-32b"); err == nil {
		chatProviders["groq"] = c
	}
	// https://huggingface.co/models?inference=warm&sort=trending
	if c, err := huggingface.New("", "Qwen/QwQ-32B"); err == nil {
		chatProviders["huggingface"] = c
	}
	if false {
		// See llamacpp/llamacppsrv to see how to run a local server.
		if c, err := llamacpp.New("http://localhost:8080", nil); err == nil {
			chatProviders["llamacpp"] = c
		}
	}
	// https://docs.mistral.ai/getting-started/models/models_overview/
	if c, err := mistral.New("", "mistral-large-latest"); err == nil {
		chatProviders["mistral"] = c
	}
	// https://platform.openai.com/docs/api-reference/models
	if c, err := openai.New("", "o3-mini"); err == nil {
		chatProviders["openai"] = c
	}
	// https://docs.perplexity.ai/models/model-cards
	if c, err := perplexity.New("", "sonar"); err == nil {
		chatProviders["perplexity"] = c
	}
	// https://api.together.ai/models
	if c, err := togetherai.New("", "deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free"); err == nil {
		chatProviders["togetherai"] = c
	}

	for name, provider := range chatProviders {
		msgs := genai.Messages{
			genai.NewTextMessage(genai.User, "Tell a story in 10 words."),
		}
		// Include options with some unsupported features to demonstrate UnsupportedContinuableError
		opts := &genai.ChatOptions{
			TopK:      50, // Not all providers support this
			MaxTokens: 512,
		}
		response, err := provider.Chat(context.Background(), msgs, opts)
		if err != nil {
			if uce, ok := err.(*genai.UnsupportedContinuableError); ok {
				fmt.Printf("- %s (ignored args: %s): %v\n", name, strings.Join(uce.Unsupported, ","), response)
			} else {
				fmt.Printf("- %s: %v\n", name, err)
			}
		} else {
			fmt.Printf("- %s: %v\n", name, response)
		}
	}
}

func ExampleChatProvider_chat_vision_and_JSON_schema() {
	// Supported by Anthropic (except for JSON), Gemini, Groq, Mistral, Ollama, OpenAI, TogetherAI.

	// Using a free small model for testing.
	// See https://ai.google.dev/gemini-api/docs/models/gemini?hl=en
	c, err := gemini.New("", "gemini-2.0-flash-lite")
	if err != nil {
		log.Fatal(err)
	}
	bananaJpg, err := os.ReadFile("internal/internaltest/testdata/banana.jpg")
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
		Seed:      1,
		MaxTokens: 50,
		DecodeAs:  &got,
	}
	resp, err := c.Chat(context.Background(), msgs, &opts)
	if err != nil {
		log.Fatal(err)
	}
	log.Printf("Raw response: %#v", resp)
	if err := resp.Decode(&got); err != nil {
		log.Fatal(err)
	}
	fmt.Printf("Banana: %v\n", got.Banana)
	// This would Output: Banana: true
}

func ExampleChatProvider_chat_pdf() {
	// Supported by Anthropic, Gemini, Mistral, OpenAI.

	// Using a free small model for testing.
	// See https://ai.google.dev/gemini-api/docs/models/gemini?hl=en
	c, err := gemini.New("", "gemini-2.0-flash-lite")
	if err != nil {
		log.Fatal(err)
	}
	f, err := os.Open("internal/internaltest/testdata/hidden_word.pdf")
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
		Seed:      1,
		MaxTokens: 50,
	}
	resp, err := c.Chat(context.Background(), msgs, &opts)
	if err != nil {
		log.Fatal(err)
	}
	log.Printf("Raw response: %#v", resp)
	fmt.Printf("Hidden word in PDF: %v\n", strings.ToLower(resp.AsText()))
	// This would Output: Hidden word in PDF: orange
}

func ExampleChatProvider_chat_audio() {
	c, err := gemini.New("", "gemini-2.0-flash-lite")
	if err != nil {
		log.Fatal(err)
	}
	f, err := os.Open("internal/internaltest/testdata/mystery_word.opus")
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
		Seed:      1,
		MaxTokens: 50,
	}
	resp, err := c.Chat(context.Background(), msgs, &opts)
	if err != nil {
		log.Fatal(err)
	}
	log.Printf("Raw response: %#v", resp)
	fmt.Printf("Heard: %v\n", strings.TrimRight(strings.ToLower(resp.AsText()), "."))
	// This would Output: Heard: orange
}

func ExampleChatProvider_chat_tool_use() {
	c, err := gemini.New("", "gemini-2.0-flash")
	if err != nil {
		log.Fatal(err)
	}
	f, err := os.Open("internal/internaltest/testdata/animation.mp4")
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
	type Word struct {
		Word string `json:"word" jsonschema:"enum=Orange,enum=Banana,enum=Apple"`
	}
	opts := genai.ChatOptions{
		Seed: 1,
		Tools: []genai.ToolDef{
			{
				Name:        "hidden_word",
				Description: "A tool to state what word was seen in the video.",
				Callback: func(got *Word) (string, error) {
					w := strings.ToLower(got.Word)
					fmt.Printf("Saw: %q\n", w)
					if w == "banana" {
						return "That's correct!", nil
					}
					return "", errors.New("That's incorrect.")
				},
			},
		},
	}
	resp, err := c.Chat(context.Background(), msgs, &opts)
	if err != nil {
		log.Fatal(err)
	}
	log.Printf("Raw response: %#v", resp)
	// Warning: there's a bug where it returns two identical tool calls. To verify.
	if len(resp.ToolCalls) == 0 || resp.ToolCalls[0].Name != "hidden_word" {
		log.Fatal("Unexpected response")
	}
	res, err := resp.ToolCalls[0].Call(opts.Tools)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("%s\n", res)
	// Remove this comment line to run the example.
	// Output:
	// Saw: "banana"
	// That's correct!
}

func ExampleChatWithToolCallLoop() {
	c, err := gemini.New("", "gemini-2.0-flash")
	if err != nil {
		log.Fatal(err)
	}
	f, err := os.Open("internal/internaltest/testdata/animation.mp4")
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
	type Word struct {
		Word string `json:"word" jsonschema:"enum=Orange,enum=Banana,enum=Apple"`
	}
	opts := genai.ChatOptions{
		Seed: 1,
		Tools: []genai.ToolDef{
			{
				Name:        "hidden_word",
				Description: "A tool to state what word was seen in the video.",
				Callback: func(got *Word) (string, error) {
					w := strings.ToLower(got.Word)
					fmt.Printf("Saw: %q\n", w)
					if w == "banana" {
						return "That's correct!", nil
					}
					return "", errors.New("That's incorrect.")
				},
			},
		},
	}
	newMsgs, _, err := genai.ChatWithToolCallLoop(context.Background(), c, msgs, &opts)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("%s\n", newMsgs[len(newMsgs)-1].AsText())
	// Remove this comment line to run the example.
	// Output:
	// Saw: "banana"
	// That's correct!
}

func ExampleChatStreamWithToolCallLoop() {
	c, err := gemini.New("", "gemini-2.0-flash")
	if err != nil {
		log.Fatal(err)
	}
	f, err := os.Open("internal/internaltest/testdata/animation.mp4")
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
	type Word struct {
		Word string `json:"word" jsonschema:"enum=Orange,enum=Banana,enum=Apple"`
	}
	opts := genai.ChatOptions{
		Seed: 1,
		Tools: []genai.ToolDef{
			{
				Name:        "hidden_word",
				Description: "A tool to state what word was seen in the video.",
				Callback: func(got *Word) (string, error) {
					w := strings.ToLower(got.Word)
					fmt.Printf("Saw: %q\n", w)
					if w == "banana" {
						return "That's correct!", nil
					}
					return "", errors.New("That's incorrect.")
				},
			},
		},
	}
	chunks := make(chan genai.MessageFragment)
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
				// Print text fragments as they arrive
				if fragment.TextFragment != "" {
					fmt.Printf("Fragment: %s", fragment.TextFragment)
				}
				// Ignore all tool calls here, they are handled transparently!
			}
		}
	}()
	newMsgs, _, err := genai.ChatStreamWithToolCallLoop(ctx, c, msgs, &opts, chunks)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("Final response: %s\n", newMsgs[len(newMsgs)-1].AsText())
	// Remove this comment line to run the example.
	// Output:
	// Saw: "banana"
	// Final response: That's correct!
}

func ExampleChatProvider_ChatStream() {
	c, err := gemini.New("", "gemini-2.0-flash-lite")
	if err != nil {
		log.Fatal(err)
	}
	ctx := context.Background()
	msgs := genai.Messages{
		genai.NewTextMessage(genai.User, "Say hello. Use only one word."),
	}
	opts := genai.ChatOptions{
		Seed:      1,
		MaxTokens: 50,
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
		log.Fatal("Unexpected responses")
	}
	resp := responses[0]
	// Normalize some of the variance. Obviously many models will still fail this test.
	fmt.Printf("Response: %s\n", strings.TrimRight(strings.TrimSpace(strings.ToLower(resp.AsText())), ".!"))
	// This would Output: Response: hello
}

func ExampleToolCall_Call() {
	// Define a tool that adds two numbers
	type math struct {
		A int `json:"a"`
		B int `json:"b"`
	}
	tool := genai.ToolDef{
		Name:        "add",
		Description: "Add two numbers together",
		Callback: func(input *math) (string, error) {
			return fmt.Sprintf("%d + %d = %d", input.A, input.B, input.A+input.B), nil
		},
	}

	// Create a tool call that would come from an LLM
	toolCall := genai.ToolCall{
		ID:        "call1",
		Name:      "add",
		Arguments: `{"a": 5, "b": 3}`,
	}

	// Invoke the tool with the arguments
	result, err := toolCall.Call([]genai.ToolDef{tool})
	if err != nil {
		fmt.Printf("Error calling tool: %v\n", err)
		return
	}

	// Print the result
	fmt.Println(result)
	// Output: 5 + 3 = 8
}
