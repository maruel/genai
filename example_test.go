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
	"github.com/maruel/genai/providers/gemini"
	"golang.org/x/sync/errgroup"
)

func ExampleProviderGen_genSync_vision() {
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
	opts := genai.OptionsText{ReplyAsJSON: true}
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
	opts := genai.OptionsText{DecodeAs: got}
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

func ExampleProviderGen_genSync_pdf() {
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

func ExampleProviderGen_genSync_audio() {
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

func ExampleProviderGen_genSync_video() {
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
	opts := genai.OptionsText{
		Seed:      1,
		MaxTokens: 50,
	}
	chunks := make(chan genai.ContentFragment)
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
	_, err = c.GenStream(ctx, msgs, chunks, &opts)
	close(chunks)
	_ = eg.Wait()
	if err != nil {
		log.Fatal(err)
	}
	// This would Output: Response: hello
}
