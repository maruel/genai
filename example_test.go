// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package genai_test

import (
	"bytes"
	"context"
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
	"github.com/maruel/genai/genaitools"
	"github.com/maruel/genai/groq"
	"github.com/maruel/genai/huggingface"
	"github.com/maruel/genai/llamacpp"
	"github.com/maruel/genai/mistral"
	"github.com/maruel/genai/openai"
	"github.com/maruel/genai/perplexity"
	"github.com/maruel/genai/pollinations"
	"github.com/maruel/genai/togetherai"
	"golang.org/x/sync/errgroup"
)

func ExampleProviderModel_all() {
	modelProviders := map[string]genai.ProviderModel{}
	if c, err := anthropic.New("", "", nil); err == nil {
		modelProviders["anthropic"] = c
	}
	if c, err := cerebras.New("", "", nil); err == nil {
		modelProviders["cerebras"] = c
	}
	if c, err := cloudflare.New("", "", "", nil); err == nil {
		modelProviders["cloudflare"] = c
	}
	if c, err := cohere.New("", "", nil); err == nil {
		modelProviders["cohere"] = c
	}
	if c, err := deepseek.New("", "", nil); err == nil {
		modelProviders["deepseek"] = c
	}
	if c, err := gemini.New("", "", nil); err == nil {
		modelProviders["gemini"] = c
	}
	if c, err := groq.New("", "", nil); err == nil {
		modelProviders["groq"] = c
	}
	if c, err := huggingface.New("", "", nil); err == nil {
		modelProviders["huggingface"] = c
	}
	// llamapcpp doesn't implement ProviderModel.
	if c, err := mistral.New("", "", nil); err == nil {
		modelProviders["mistral"] = c
	}
	if c, err := openai.New("", "", nil); err == nil {
		modelProviders["openai"] = c
	}
	if c, err := pollinations.New("", "", nil); err == nil {
		modelProviders["pollinations"] = c
	}
	// perplexity doesn't implement ProviderModel.
	if c, err := togetherai.New("", "", nil); err == nil {
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

func ExampleProviderGen_all() {
	chatProviders := map[string]genai.ProviderGen{}
	// https://docs.anthropic.com/en/docs/about-claude/models/all-models
	if c, err := anthropic.New("", "claude-3-7-sonnet-latest", nil); err == nil {
		chatProviders["anthropic"] = c
	}
	// https://inference-docs.cerebras.ai/api-reference/models
	if c, err := cerebras.New("", "llama-3.3-70b", nil); err == nil {
		chatProviders["cerebras"] = c
	}
	// https://developers.cloudflare.com/workers-ai/models/
	if c, err := cloudflare.New("", "", "@cf/deepseek-ai/deepseek-r1-distill-qwen-32b", nil); err == nil {
		chatProviders["cloudflare"] = c
	}
	// https://docs.cohere.com/v2/docs/models
	if c, err := cohere.New("", "command-r-plus", nil); err == nil {
		chatProviders["cohere"] = c
	}
	// https://api-docs.deepseek.com/quick_start/pricing
	if c, err := deepseek.New("", "deepseek-reasoner", nil); err == nil {
		chatProviders["deepseek"] = c
	}
	// https://ai.google.dev/gemini-api/docs/models/gemini
	if c, err := gemini.New("", "gemini-2.0-flash", nil); err == nil {
		chatProviders["gemini"] = c
	}
	// https://console.groq.com/docs/models
	if c, err := groq.New("", "qwen-qwq-32b", nil); err == nil {
		chatProviders["groq"] = c
	}
	// https://huggingface.co/models?inference=warm&sort=trending
	if c, err := huggingface.New("", "Qwen/QwQ-32B", nil); err == nil {
		chatProviders["huggingface"] = c
	}
	if false {
		// See llamacpp/llamacppsrv to see how to run a local server.
		if c, err := llamacpp.New("http://localhost:8080", nil, nil); err == nil {
			chatProviders["llamacpp"] = c
		}
	}
	// https://docs.mistral.ai/getting-started/models/models_overview/
	if c, err := mistral.New("", "mistral-large-latest", nil); err == nil {
		chatProviders["mistral"] = c
	}
	// https://platform.openai.com/docs/api-reference/models
	if c, err := openai.New("", "o3-mini", nil); err == nil {
		chatProviders["openai"] = c
	}
	// https://docs.perplexity.ai/models/model-cards
	if c, err := perplexity.New("", "sonar", nil); err == nil {
		chatProviders["perplexity"] = c
	}
	// https://text.pollinations.ai/models
	if c, err := pollinations.New("", "qwen-coder", nil); err == nil {
		chatProviders["pollinations"] = c
	}
	// https://api.together.ai/models
	if c, err := togetherai.New("", "deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free", nil); err == nil {
		chatProviders["togetherai"] = c
	}

	for name, provider := range chatProviders {
		msgs := genai.Messages{
			genai.NewTextMessage(genai.User, "Tell a story in 10 words."),
		}
		// Include options with some unsupported features to demonstrate UnsupportedContinuableError
		opts := &genai.TextOptions{
			TopK:      50, // Not all providers support this
			MaxTokens: 512,
		}
		response, err := provider.GenSync(context.Background(), msgs, opts)
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

func ExampleProviderGen_chat_vision() {
	// Supported by Anthropic, Gemini, Groq, Mistral, Ollama, OpenAI, TogetherAI.

	// Using a free small model for testing.
	// See https://ai.google.dev/gemini-api/docs/models/gemini?hl=en
	c, err := gemini.New("", "gemini-2.0-flash-lite", nil)
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
				{Text: "Is it a banana? Reply with only the word yes or no."},
				{Filename: "banana.jpg", Document: bytes.NewReader(bananaJpg)},
			},
		},
	}
	resp, err := c.GenSync(context.Background(), msgs, nil)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("Banana: %v\n", resp.AsText())
	// This would Output: Banana: yes
}

func ExampleClient_GenSync_jSON() {
	// Supported by Cerebras, Cloudflare, Cohere, DeepSeek, Gemini, Groq, HuggingFace, Mistral, Ollama, OpenAI, TogetherAI.

	// Using a free small model for testing.
	// See https://ai.google.dev/gemini-api/docs/models/gemini?hl=en
	c, err := gemini.New("", "gemini-2.0-flash-lite", nil)
	if err == nil {
		log.Fatal(err)
	}
	msgs := genai.Messages{
		genai.NewTextMessage(genai.User, "Is a circle round? Reply as JSON with the form {\"round\": false} or {\"round\": true}."),
	}
	opts := genai.TextOptions{ReplyAsJSON: true}
	resp, err := c.GenSync(context.Background(), msgs, &opts)
	if err != nil {
		log.Fatal(err)
	}
	got := map[string]any{}
	if err := resp.Decode(&got); err != nil {
		log.Fatal(err)
	}
	fmt.Printf("Round: %v\n", got["round"])
	// This would Output: Round: true
}

func ExampleClient_GenSync_jSON_schema() {
	// Supported by Cerebras, Cloudflare, Cohere, Gemini, Groq, HuggingFace, Mistral, Ollama, OpenAI, Perplexity, TogetherAI.

	// Using a free small model for testing.
	// See https://ai.google.dev/gemini-api/docs/models/gemini?hl=en
	c, err := gemini.New("", "gemini-2.0-flash-lite", nil)
	if err == nil {
		log.Fatal(err)
	}
	msgs := genai.Messages{
		genai.NewTextMessage(genai.User, "Is a circle round? Reply as JSON."),
	}
	var got struct {
		Round bool `json:"round"`
	}
	opts := genai.TextOptions{DecodeAs: got}
	resp, err := c.GenSync(context.Background(), msgs, &opts)
	if err != nil {
		log.Fatal(err)
	}
	if err := resp.Decode(&got); err != nil {
		log.Fatal(err)
	}
	fmt.Printf("Round: %v\n", got.Round)
	// This would Output: Round: true
}

func ExampleProviderGen_chat_pdf() {
	// Supported by Anthropic, Gemini, Mistral, OpenAI.

	// Using a free small model for testing.
	// See https://ai.google.dev/gemini-api/docs/models/gemini?hl=en
	c, err := gemini.New("", "gemini-2.0-flash-lite", nil)
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
	resp, err := c.GenSync(context.Background(), msgs, nil)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("Hidden word in PDF: %v\n", strings.ToLower(resp.AsText()))
	// This would Output: Hidden word in PDF: orange
}

func ExampleProviderGen_chat_audio() {
	// Supported by Gemini, OpenAI.

	// Using a free small model for testing.
	// See https://ai.google.dev/gemini-api/docs/models/gemini?hl=en
	c, err := gemini.New("", "gemini-2.0-flash-lite", nil)
	if err != nil {
		log.Fatal(err)
	}
	f, err := os.Open("internal/internaltest/testdata/mystery_word.mp3")
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
	resp, err := c.GenSync(context.Background(), msgs, nil)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("Heard: %v\n", strings.TrimRight(strings.ToLower(resp.AsText()), "."))
	// This would Output: Heard: orange
}

func ExampleProviderGen_chat_video() {
	// Supported by Gemini, TogetherAI.

	// Using a free small model for testing.
	// See https://ai.google.dev/gemini-api/docs/models/gemini?hl=en
	c, err := gemini.New("", "gemini-2.0-flash", nil)
	if err != nil {
		log.Fatal(err)
	}
	f, err := os.Open("internal/internaltest/testdata/animation.mp4")
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()
	// TogetherAI seems to require separate messages for text and images.
	msgs := genai.Messages{
		genai.NewTextMessage(genai.User, "What is the word? Reply with exactly and only one word."),
		{Role: genai.User, Contents: []genai.Content{{Document: f}}},
	}
	resp, err := c.GenSync(context.Background(), msgs, nil)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("Saw: %v\n", strings.ToLower(resp.AsText()))
	// This would Output: Saw: banana
}

func ExampleChatWithToolCallLoop() {
	// Supported by Anthropic, Cerebras, Cloudflare, Cohere, DeepSeek, Gemini, Groq, HuggingFace, Mistral,
	// Ollama, OpenAI, TogetherAI.

	// Using a free small model for testing.
	// See https://ai.google.dev/gemini-api/docs/models/gemini?hl=en
	c, err := gemini.New("", "gemini-2.0-flash", nil)
	if err != nil {
		log.Fatal(err)
	}
	msgs := genai.Messages{
		genai.NewTextMessage(genai.User, "What is 3214 + 5632? Leverage the tool available to you to tell me the answer. Do not explain. Be terse. Include only the answer."),
	}
	opts := genai.TextOptions{
		Tools: []genai.ToolDef{genaitools.Arithmetic},
		// Force the LLM to do a tool call first.
		ToolCallRequest: genai.ToolCallRequired,
	}
	newMsgs, _, err := genai.ChatWithToolCallLoop(context.Background(), c, msgs, &opts)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("%s\n", newMsgs[len(newMsgs)-1].AsText())
	// Remove this comment line to run the example.
	// Output:
	// 8846
}

func ExampleChatStreamWithToolCallLoop() {
	// Supported by Anthropic, Cerebras, Cloudflare, Cohere, DeepSeek, Gemini, Groq, HuggingFace, Mistral,
	// Ollama, OpenAI, TogetherAI.

	// Using a free small model for testing.
	// See https://ai.google.dev/gemini-api/docs/models/gemini?hl=en
	c, err := gemini.New("", "gemini-2.0-flash", nil)
	if err != nil {
		log.Fatal(err)
	}
	msgs := genai.Messages{
		genai.NewTextMessage(genai.User, "What is 3214 + 5632? Leverage the tool available to you to tell me the answer. Do not explain. Be terse. Include only the answer."),
	}
	opts := genai.TextOptions{
		Tools: []genai.ToolDef{genaitools.Arithmetic},
		// Force the LLM to do a tool call first.
		ToolCallRequest: genai.ToolCallRequired,
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
				_, _ = os.Stdout.WriteString(fragment.TextFragment)
			}
		}
	}()
	_, _, err = genai.ChatStreamWithToolCallLoop(ctx, c, msgs, &opts, chunks)
	if err != nil {
		log.Fatal(err)
	}
	// Remove this comment line to run the example.
	// Output:
	// 8846
}

func ExampleProviderGen_GenStream() {
	// Supported by all providers.

	// Using a free small model for testing.
	// See https://ai.google.dev/gemini-api/docs/models/gemini?hl=en
	c, err := gemini.New("", "gemini-2.0-flash-lite", nil)
	if err != nil {
		log.Fatal(err)
	}
	ctx := context.Background()
	msgs := genai.Messages{
		genai.NewTextMessage(genai.User, "Say hello. Use only one word."),
	}
	opts := genai.TextOptions{
		Seed:      1,
		MaxTokens: 50,
	}
	chunks := make(chan genai.MessageFragment)
	eg := errgroup.Group{}
	eg.Go(func() error {
		for {
			select {
			case <-ctx.Done():
				return nil
			case pkt, ok := <-chunks:
				if !ok {
					return nil
				}
				if _, err2 := os.Stdout.WriteString(pkt.TextFragment); err2 != nil {
					return err2
				}
			}
		}
	})
	_, err = c.GenStream(ctx, msgs, &opts, chunks)
	close(chunks)
	_ = eg.Wait()
	if err != nil {
		log.Fatal(err)
	}
	// This would Output: Response: hello
}
