// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package genai_test

import (
	"bytes"
	"context"
	"fmt"
	"log"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/maruel/genai"
	"github.com/maruel/genai/adapters"
	"github.com/maruel/genai/providers/anthropic"
	"github.com/maruel/genai/providers/gemini"
	"github.com/maruel/roundtrippers"
)

func ExampleProvider_genSync_vision() {
	// Supported by Anthropic, Gemini, Groq, Mistral, Ollama, OpenAI, TogetherAI.

	// Using a free small model for testing.
	// See https://ai.google.dev/gemini-api/docs/models/gemini?hl=en
	ctx := context.Background()
	c, err := gemini.New(ctx, genai.ProviderOptionModel("gemini-2.5-flash-lite"))
	if err != nil {
		log.Fatal(err)
	}
	bananaJpg, err := os.ReadFile("internal/internaltest/testdata/banana.jpg")
	if err != nil {
		log.Fatal(err)
	}
	msgs := genai.Messages{
		{
			Requests: []genai.Request{
				{Text: "Is it a banana? Reply with only the word yes or no."},
				{Doc: genai.Doc{Filename: "banana.jpg", Src: bytes.NewReader(bananaJpg)}},
			},
		},
	}
	resp, err := c.GenSync(ctx, msgs)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("Banana: %v\n", resp.String())
	// This would Output: Banana: yes
}

func ExampleClient_GenSync_jSON() {
	// Supported by Cerebras, Cloudflare, Cohere, DeepSeek, Gemini, Groq, HuggingFace, Mistral, Ollama, OpenAI, TogetherAI.

	// Using a free small model for testing.
	// See https://ai.google.dev/gemini-api/docs/models/gemini?hl=en
	ctx := context.Background()
	c, err := gemini.New(ctx, genai.ProviderOptionModel("gemini-2.5-flash-lite"))
	if err == nil {
		log.Fatal(err)
	}
	msgs := genai.Messages{
		genai.NewTextMessage("Is a circle round? Reply as JSON with the form {\"round\": false} or {\"round\": true}."),
	}
	opts := genai.GenOptionText{ReplyAsJSON: true}
	resp, err := c.GenSync(ctx, msgs, &opts)
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
	ctx := context.Background()
	c, err := gemini.New(ctx, genai.ProviderOptionModel("gemini-2.5-flash-lite"))
	if err == nil {
		log.Fatal(err)
	}
	msgs := genai.Messages{
		genai.NewTextMessage("Is a circle round? Reply as JSON."),
	}
	var got struct {
		Round bool `json:"round"`
	}
	opts := genai.GenOptionText{DecodeAs: got}
	resp, err := c.GenSync(ctx, msgs, &opts)
	if err != nil {
		log.Fatal(err)
	}
	if err := resp.Decode(&got); err != nil {
		log.Fatal(err)
	}
	fmt.Printf("Round: %v\n", got.Round)
	// This would Output: Round: true
}

func ExampleProvider_genSync_pdf() {
	// Supported by Anthropic, Gemini, Mistral, OpenAI.

	// Using a free small model for testing.
	// See https://ai.google.dev/gemini-api/docs/models/gemini?hl=en
	ctx := context.Background()
	c, err := gemini.New(ctx, genai.ProviderOptionModel("gemini-2.5-flash-lite"))
	if err != nil {
		log.Fatal(err)
	}
	f, err := os.Open("internal/internaltest/testdata/hidden_word.pdf")
	if err != nil {
		log.Fatal(err)
	}
	defer func() { _ = f.Close() }()
	msgs := genai.Messages{
		{
			Requests: []genai.Request{
				{Text: "What is the word? Reply with only the word."},
				{Doc: genai.Doc{Src: f}},
			},
		},
	}
	resp, err := c.GenSync(ctx, msgs)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("Hidden word in PDF: %v\n", strings.ToLower(resp.String()))
	// This would Output: Hidden word in PDF: orange
}

func ExampleProvider_genSync_audio() {
	// Supported by Gemini, OpenAI.

	// Using a free small model for testing.
	// See https://ai.google.dev/gemini-api/docs/models/gemini?hl=en
	ctx := context.Background()
	c, err := gemini.New(ctx, genai.ProviderOptionModel("gemini-2.5-flash-lite"))
	if err != nil {
		log.Fatal(err)
	}
	f, err := os.Open("internal/internaltest/testdata/mystery_word.mp3")
	if err != nil {
		log.Fatal(err)
	}
	defer func() { _ = f.Close() }()
	msgs := genai.Messages{
		{
			Requests: []genai.Request{
				{Text: "What is the word said? Reply with only the word."},
				{Doc: genai.Doc{Src: f}},
			},
		},
	}
	resp, err := c.GenSync(ctx, msgs)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("Heard: %v\n", strings.TrimRight(strings.ToLower(resp.String()), "."))
	// This would Output: Heard: orange
}

func ExampleProvider_genSync_video() {
	// Supported by Gemini, TogetherAI.

	// Using a free small model for testing.
	// See https://ai.google.dev/gemini-api/docs/models/gemini?hl=en
	ctx := context.Background()
	c, err := gemini.New(ctx, genai.ProviderOptionModel("gemini-2.5-flash"))
	if err != nil {
		log.Fatal(err)
	}
	f, err := os.Open("internal/internaltest/testdata/animation.mp4")
	if err != nil {
		log.Fatal(err)
	}
	defer func() { _ = f.Close() }()
	// TogetherAI seems to require separate messages for text and images.
	msgs := genai.Messages{
		genai.NewTextMessage("What is the word? Reply with exactly and only one word."),
		{Requests: []genai.Request{{Doc: genai.Doc{Src: f}}}},
	}
	resp, err := c.GenSync(ctx, msgs)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("Saw: %v\n", strings.ToLower(resp.String()))
	// This would Output: Saw: banana
}

func ExampleProvider_GenStream() {
	// Supported by all providers.

	// Using a free small model for testing.
	// See https://ai.google.dev/gemini-api/docs/models/gemini?hl=en
	ctx := context.Background()
	c, err := gemini.New(ctx, genai.ProviderOptionModel("gemini-2.5-flash-lite"))
	if err != nil {
		log.Fatal(err)
	}
	msgs := genai.Messages{
		genai.NewTextMessage("Say hello. Use only one word."),
	}
	opts := genai.GenOptionText{
		MaxTokens: 50,
	}
	fragments, finish := c.GenStream(ctx, msgs, &opts, genai.GenOptionSeed(1))
	for f := range fragments {
		_, _ = os.Stdout.WriteString(f.Text)
	}
	if _, err := finish(); err != nil {
		log.Fatal(err)
	}
	// This would Output: Response: hello
}

func Example_genSyncWithToolCallLoop_with_custom_HTTP_Header() {
	// Modified version of the example in package adapters, with a custom header.
	//
	// As of June 2025, interleaved thinking can be enabled with a custom header.
	// https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking#interleaved-thinking
	wrapper := func(h http.RoundTripper) http.RoundTripper {
		return &roundtrippers.Header{
			Transport: h,
			Header:    http.Header{"anthropic-beta": []string{"interleaved-thinking-2025-05-14"}},
		}
	}
	ctx := context.Background()
	c, err := anthropic.New(ctx, genai.ProviderOptionModel("claude-sonnet-4-20250514"), genai.ProviderOptionTransportWrapper(wrapper))
	if err != nil {
		log.Fatal(err)
	}
	msgs := genai.Messages{genai.NewTextMessage("What season is Montréal currently in?")}
	opts := genai.GenOptionTools{
		Tools: []genai.ToolDef{locationClockTime},
		// Force the LLM to do a tool call first.
		Force: genai.ToolCallRequired,
	}
	newMsgs, _, err := adapters.GenSyncWithToolCallLoop(ctx, c, msgs, &opts)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("%s\n", newMsgs[len(newMsgs)-1].String())
}

var locationClockTime = genai.ToolDef{
	Name:        "get_today_date_current_clock_time",
	Description: "Get the current clock time and today's date.",
	Callback: func(ctx context.Context, e *location) (string, error) {
		if e.Location != "Montréal" {
			return "ask again with Montréal", nil
		}
		return time.Now().Format("Monday 2006-01-02 15:04:05"), nil
	},
}

type location struct {
	Location string `json:"location" json_description:"Location to ask the current time in"`
}
