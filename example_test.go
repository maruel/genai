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
	"github.com/maruel/genai/genaitools"
	"github.com/maruel/genai/providers/anthropic"
	"github.com/maruel/genai/providers/gemini"
	"github.com/maruel/genai/providers/groq"
	"github.com/maruel/roundtrippers"
	"golang.org/x/sync/errgroup"
	"gopkg.in/dnaeon/go-vcr.v4/pkg/cassette"
	"gopkg.in/dnaeon/go-vcr.v4/pkg/recorder"
)

func ExampleProviderGen_genSync_vision() {
	// Supported by Anthropic, Gemini, Groq, Mistral, Ollama, OpenAI, TogetherAI.

	// Using a free small model for testing.
	// See https://ai.google.dev/gemini-api/docs/models/gemini?hl=en
	c, err := gemini.New(&genai.OptionsProvider{Model: "gemini-2.0-flash-lite"}, nil)
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
	c, err := gemini.New(&genai.OptionsProvider{Model: "gemini-2.0-flash-lite"}, nil)
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
	c, err := gemini.New(&genai.OptionsProvider{Model: "gemini-2.0-flash-lite"}, nil)
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
	c, err := gemini.New(&genai.OptionsProvider{Model: "gemini-2.0-flash-lite"}, nil)
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
	c, err := gemini.New(&genai.OptionsProvider{Model: "gemini-2.0-flash-lite"}, nil)
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
	c, err := gemini.New(&genai.OptionsProvider{Model: "gemini-2.0-flash"}, nil)
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
	c, err := gemini.New(&genai.OptionsProvider{Model: "gemini-2.0-flash-lite"}, nil)
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

func ExampleProvider_hTTP_record() {
	// Example to do HTTP recording and playback for smoke testing.
	// The example recording is in testdata/example.yaml.
	//
	// WARNING: Many providers use a slightly different way to send the tokens. Examples include Anthropic,
	// Cloudflare and Gemini. Make sure to do a test recording first and confirm no key is saved. See the
	// ExampleProvider_hTTP_record example for each provider to learn how to do a safe HTTP record.
	var rr *recorder.Recorder
	defer func() {
		// In a smoke test, use t.Cleanup().
		if rr != nil {
			if err := rr.Stop(); err != nil {
				log.Printf("Failed saving recordings: %v", err)
			}
		}
	}()

	wrapper := func(h http.RoundTripper) http.RoundTripper {
		// Simple trick to force recording via an environment variable.
		mode := recorder.ModeRecordOnce
		if os.Getenv("RECORD") == "1" {
			mode = recorder.ModeRecordOnly
		}
		// Remove API key when matching the request, so the playback doesn't need to have access to the API key.
		// See the corresponding provider example as each provider has its own way to set the API key.
		m := cassette.NewDefaultMatcher(cassette.WithIgnoreHeaders("Authorization", "X-Request-Id"))
		var err error
		rr, err = recorder.New("testdata/example",
			recorder.WithHook(trimResponseHeaders, recorder.AfterCaptureHook),
			recorder.WithMode(mode),
			recorder.WithSkipRequestLatency(true),
			recorder.WithRealTransport(h),
			recorder.WithMatcher(m),
		)
		if err != nil {
			log.Fatal(err)
		}
		return rr
	}
	// When playing back the smoke test, no API key is needed. Insert a fake API key.
	apiKey := ""
	if os.Getenv("GROQ_API_KEY") == "" {
		apiKey = "<insert_api_key_here>"
	}
	c, err := groq.New(&genai.OptionsProvider{APIKey: apiKey}, wrapper)
	if err != nil {
		log.Fatal(err)
	}
	models, err := c.ListModels(context.Background())
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("Found %d models\n", len(models))
	// Output:
	// Found 22 models
}

// trimResponseHeaders trims API key and noise from the recording.
func trimResponseHeaders(i *cassette.Interaction) error {
	// Do not save the API key in the recording.
	i.Request.Headers.Del("Authorization")
	i.Request.Headers.Del("X-Request-Id")
	i.Response.Headers.Del("Set-Cookie")
	// Reduce noise.
	i.Response.Headers.Del("Date")
	i.Response.Headers.Del("X-Request-Id")
	i.Response.Duration = i.Response.Duration.Round(time.Millisecond)
	return nil
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
	c, err := anthropic.New(&genai.OptionsProvider{Model: "claude-sonnet-4-20250514"}, wrapper)
	if err != nil {
		log.Fatal(err)
	}
	msgs := genai.Messages{
		genai.NewTextMessage(genai.User, "What is 3214 + 5632? Leverage the tool available to you to tell me the answer. Do not explain. Be terse. Include only the answer."),
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
