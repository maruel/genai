// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package internaltest

import (
	"bytes"
	"context"
	_ "embed"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/maruel/genai"
)

// ChatProviderFactory is what a Java developer would write.
type ChatProviderFactory func(t *testing.T) genai.ChatProvider

// ModelChatProviderFactory is what a Java developer would write.
type ModelChatProviderFactory func(t *testing.T, model string) genai.ChatProvider

func TestChatThinking(t *testing.T, factory ChatProviderFactory) {
	ctx := t.Context()
	c := factory(t)
	msgs := genai.Messages{
		genai.NewTextMessage(genai.User, "Say hello. Use only one word. Say only hello."),
	}
	// I believe most thinking models do not like Temperature to be set.
	opts := genai.ChatOptions{
		MaxTokens:      2000,
		Seed:           1,
		ThinkingBudget: 1900,
	}
	resp, err := c.Chat(ctx, msgs, &opts)
	if uce, ok := err.(*genai.UnsupportedContinuableError); ok {
		t.Log(uce)
	} else if err != nil {
		t.Fatal(err)
	}
	t.Logf("Raw response: %#v", resp)
	testUsage(t, &resp.Usage)
	if len(resp.Contents) != 1 {
		t.Fatal("Unexpected response")
	}
	// Normalize some of the variance. Obviously many models will still fail this test.
	got := strings.TrimRight(strings.TrimSpace(strings.ToLower(resp.Contents[0].Text)), ".!")
	if got != "hello" {
		t.Fatal(got)
	}
}

// TestChatStream makes sure ChatStream() works.
func TestChatStream(t *testing.T, factory ChatProviderFactory, hasUsage bool) {
	c := factory(t)
	ctx := t.Context()
	chunks := make(chan genai.MessageFragment)
	end := make(chan genai.Messages, 1)
	go func() {
		var pendingMsgs genai.Messages
		defer func() {
			end <- pendingMsgs
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
				// Gemini API has both FinishReason and TextFragment in the final message.
				if pkt.FinishReason == "" && pkt.TextFragment == "" {
					t.Errorf("Must have at least one FinishReason or Text: %#v", pkt)
				}
				var err2 error
				if pendingMsgs, err2 = pkt.Accumulate(pendingMsgs); err2 != nil {
					t.Error(err2)
					return
				}
				if pkt.FinishReason != "" && pkt.FinishReason != "stop" {
					t.Errorf("Unexpected FinishReason: %q", pkt.FinishReason)
				}
			}
		}
	}()
	msgs := genai.Messages{
		genai.NewTextMessage(genai.User, "Say hello. Use only one word."),
	}
	opts := genai.ChatOptions{
		Temperature: 0.01,
		MaxTokens:   50,
		Seed:        1,
	}
	usage, err := c.ChatStream(ctx, msgs, &opts, chunks)
	if uce, ok := err.(*genai.UnsupportedContinuableError); ok {
		t.Log(uce)
	} else if err != nil {
		t.Error(err)
	}
	close(chunks)
	// Hugginface, OpenAI do not set tokens when streaming as shown in testdata/TestClient_ChatStream.yaml
	if hasUsage {
		testUsage(t, &usage)
	}
	responses := <-end
	t.Logf("Raw responses: %#v", responses)
	if len(responses) != 1 {
		t.Fatal("Unexpected response")
	}
	resp := responses[0]
	if len(resp.Contents) != 1 {
		t.Fatal("Unexpected response")
	}
	// Normalize some of the variance. Obviously many models will still fail this test.
	got := strings.TrimRight(strings.TrimSpace(strings.ToLower(resp.Contents[0].Text)), ".!")
	if got != "hello" {
		t.Fatal(got)
	}
}

// TestChatAllModels says hello with all models.
func TestChatAllModels(t *testing.T, factory ModelChatProviderFactory, filter func(model genai.Model) bool) {
	ctx := t.Context()
	l := factory(t, "").(genai.ModelProvider)
	models, err := l.ListModels(ctx)
	if err != nil {
		t.Fatal(err)
	}
	msgs := genai.Messages{
		genai.NewTextMessage(genai.User, "Say hello. Use only one word. Say only hello."),
	}
	// MaxTokens has to be long because of some thinking models (e.g. qwen-qwq-32b and
	// deepseek-r1-distill-llama-70b) cannot have thinking disabled.
	opts := genai.ChatOptions{
		Temperature: 0.1,
		MaxTokens:   1000,
		Seed:        1,
	}
	for _, m := range models {
		id := m.GetID()
		if filter != nil && !filter(m) {
			continue
		}
		t.Run(id, func(t *testing.T) {
			resp, err := factory(t, id).Chat(ctx, msgs, &opts)
			if uce, ok := err.(*genai.UnsupportedContinuableError); ok {
				t.Log(uce)
			} else if err != nil {
				t.Fatal(err)
			}
			t.Logf("Raw response: %#v", resp)
			testUsage(t, &resp.Usage)
			if len(resp.Contents) != 1 {
				t.Fatal("Unexpected response")
			}
			// Normalize some of the variance. Obviously many models will still fail this test.
			got := strings.TrimRight(strings.TrimSpace(strings.ToLower(resp.Contents[0].Text)), ".!")
			if got != "hello" {
				t.Fatal(got)
			}
		})
	}
}

// Tool

// TestChatToolUseCountry runs a Chat with tool use and verifies that the tools are called correctly.
// It runs subtests for both Chat and ChatStream methods.
func TestChatToolUseCountry(t *testing.T, factory ChatProviderFactory, hasUsage bool) {
	t.Run("Chat", func(t *testing.T) {
		chatToolUseCountryCore(t, factory, hasUsage, false)
	})
	t.Run("ChatStream", func(t *testing.T) {
		t.Skip("TODO")
		chatToolUseCountryCore(t, factory, hasUsage, true)
	})
}

// chatToolUseCountryCore runs a Chat or ChatStream with tool use and verifies that the tools are called correctly.
// The useStream parameter determines whether to use Chat or ChatStream.
// It returns the response for further validation.
func chatToolUseCountryCore(t *testing.T, factory ChatProviderFactory, hasUsage bool, useStream bool) {
	ctx := t.Context()
	t.Run("Canada", func(t *testing.T) {
		c := factory(t)
		msgs := genai.Messages{
			genai.NewTextMessage(genai.User, "I wonder if Canada is a better country than the USA? Call the tool best_country to tell me which country is the best one."),
		}
		var got struct {
			Country string `json:"country" jsonschema:"enum=Canada,enum=USA"`
		}
		opts := genai.ChatOptions{
			Temperature: 0.01,
			MaxTokens:   200,
			Tools: []genai.ToolDef{
				{
					Name:        "best_country",
					Description: "A tool to determine the best country",
					InputsAs:    &got,
				},
			},
		}
		var resp genai.ChatResult
		if useStream {
			resp = processChatStream(t, ctx, c, msgs, &opts)
		} else {
			resp = processChat(t, ctx, c, msgs, &opts)
		}
		// I'm disappointed by Cloudflare.
		if hasUsage {
			testUsage(t, &resp.Usage)
		}

		// Warning: when the model is undecided, it call both.
		// Check for tool calls
		want := "best_country"
		if len(resp.ToolCalls) == 0 || resp.ToolCalls[0].Name != want {
			t.Fatalf("Expected tool call to %s, got: %v", want, resp.ToolCalls)
		}
		if err := resp.ToolCalls[0].Decode(&got); err != nil {
			t.Fatal(err)
		}
		if got.Country != "Canada" {
			t.Fatal(got.Country)
		}
		// TODO: Follow up!
	})
	t.Run("USA", func(t *testing.T) {
		c := factory(t)
		msgs := genai.Messages{
			genai.NewTextMessage(genai.User, "I wonder if the USA is a better country than Canada? Call the tool best_country to tell me which country is the best one."),
		}
		var got struct {
			Country string `json:"country" jsonschema:"enum=USA,enum=Canada"`
		}
		opts := genai.ChatOptions{
			Temperature: 0.01,
			MaxTokens:   200,
			Seed:        1,
			Tools: []genai.ToolDef{
				{
					Name:        "best_country",
					Description: "A tool to determine the best country",
					InputsAs:    &got,
				},
			},
		}
		var resp genai.ChatResult
		if useStream {
			resp = processChatStream(t, ctx, c, msgs, &opts)
		} else {
			resp = processChat(t, ctx, c, msgs, &opts)
		}
		if hasUsage {
			testUsage(t, &resp.Usage)
		}
		// Warning: when the model is undecided, it call both.
		// Check for tool calls
		want := "best_country"
		if len(resp.ToolCalls) == 0 || resp.ToolCalls[0].Name != want {
			t.Fatalf("Expected tool call to %s, got: %v", want, resp.ToolCalls)
		}
		if err := resp.ToolCalls[0].Decode(&got); err != nil {
			t.Fatal(err)
		}
		if got.Country != "USA" {
			t.Fatal(got.Country)
		}
		// TODO: Follow up!
	})
}

// Multi-modal (audio, image, video)

// TestChatVisionJPGInline runs a Chat with vision capabilities and verifies that the model correctly identifies a
// banana image.
func TestChatVisionJPGInline(t *testing.T, factory ChatProviderFactory) {
	c := factory(t)
	ctx := t.Context()
	msgs := genai.Messages{
		{
			Role: genai.User,
			Contents: []genai.Content{
				{Text: "Is it a banana? Reply with only one word."},
				{Filename: "banana.jpg", Document: bytes.NewReader(bananaJpg)},
			},
		},
	}
	opts := genai.ChatOptions{
		Temperature: 0.01,
		MaxTokens:   200,
		Seed:        1,
	}
	resp, err := c.Chat(ctx, msgs, &opts)
	if uce, ok := err.(*genai.UnsupportedContinuableError); ok {
		t.Log(uce)
	} else if err != nil {
		t.Fatal(err)
	}
	t.Logf("Raw response: %#v", resp)
	testUsage(t, &resp.Usage)
	if len(resp.Contents) != 1 {
		t.Fatal("Unexpected response")
	}
	// Normalize some of the variance. Obviously many models will still fail this test.
	txt := strings.TrimRight(strings.TrimSpace(strings.ToLower(resp.Contents[0].Text)), ".!")
	if txt != "yes" {
		t.Fatal(txt)
	}
}

func TestChatVisionPDFInline(t *testing.T, factory ChatProviderFactory) {
	c := factory(t)
	// Path with the assumption it's run from "//<provider>/".
	f, err := os.Open("../internal/internaltest/testdata/hidden_word.pdf")
	if err != nil {
		t.Fatal(err)
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
		Seed:        1,
		Temperature: 0.01,
		MaxTokens:   50,
	}
	resp, err := c.Chat(t.Context(), msgs, &opts)
	if uce, ok := err.(*genai.UnsupportedContinuableError); ok {
		t.Log(uce)
	} else if err != nil {
		t.Fatal(err)
	}
	t.Logf("Raw response: %#v", resp)
	testUsage(t, &resp.Usage)
	if resp.InputTokens != 1301 || resp.OutputTokens != 2 {
		t.Logf("Unexpected tokens usage: %v", resp.Usage)
	}
	if len(resp.Contents) != 1 {
		t.Fatal("Unexpected response")
	}
	if got := strings.TrimSpace(strings.ToLower(resp.Contents[0].Text)); got != "orange" {
		t.Fatal(got)
	}
}

func TestChatVisionPDFURL(t *testing.T, factory ChatProviderFactory) {
	c := factory(t)
	msgs := genai.Messages{
		{
			Role: genai.User,
			Contents: []genai.Content{
				{Text: "What is the word? Reply with only the word."},
				{URL: "https://raw.githubusercontent.com/maruel/genai/refs/heads/main/internal/internaltest/testdata/hidden_word.pdf"},
			},
		},
	}
	opts := genai.ChatOptions{
		Seed:        1,
		Temperature: 0.01,
		MaxTokens:   50,
	}
	resp, err := c.Chat(t.Context(), msgs, &opts)
	if uce, ok := err.(*genai.UnsupportedContinuableError); ok {
		t.Log(uce)
	} else if err != nil {
		t.Fatal(err)
	}
	t.Logf("Raw response: %#v", resp)
	testUsage(t, &resp.Usage)
	if resp.InputTokens != 1301 || resp.OutputTokens != 2 {
		t.Logf("Unexpected tokens usage: %v", resp.Usage)
	}
	if len(resp.Contents) != 1 {
		t.Fatal("Unexpected response")
	}
	if got := strings.TrimSpace(strings.ToLower(resp.Contents[0].Text)); got != "orange" {
		t.Fatal(got)
	}
}

func TestChatAudioMP3Inline(t *testing.T, factory ChatProviderFactory) {
	testChatAudioInline(t, factory, "mystery_word.mp3")
}

func TestChatAudioOpusInline(t *testing.T, factory ChatProviderFactory) {
	testChatAudioInline(t, factory, "mystery_word.opus")
}

func testChatAudioInline(t *testing.T, factory ChatProviderFactory, filename string) {
	c := factory(t)
	// Path with the assumption it's run from "//<provider>/".
	f, err := os.Open(filepath.Join("..", "internal", "internaltest", "testdata", filename))
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	msgs := genai.Messages{
		{Role: genai.User, Contents: []genai.Content{{Document: f}}},
		genai.NewTextMessage(genai.User, "What is the word said? Reply with only the word."),
	}
	opts := genai.ChatOptions{
		Seed:        1,
		Temperature: 0.01,
		MaxTokens:   50,
	}
	resp, err := c.Chat(t.Context(), msgs, &opts)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("Raw response: %#v", resp)
	testUsage(t, &resp.Usage)
	if len(resp.Contents) != 1 {
		t.Fatal("Unexpected response")
	}
	if heard := strings.TrimRight(strings.ToLower(resp.Contents[0].Text), "."); heard != "orange" {
		t.Fatal(heard)
	}
}

func TestChatVideoMP4Inline(t *testing.T, factory ChatProviderFactory) {
	c := factory(t)
	// Path with the assumption it's run from "//<provider>/".
	f, err := os.Open(filepath.Join("..", "internal", "internaltest", "testdata", "animation.mp4"))
	if err != nil {
		t.Fatal(err)
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
		Seed:        1,
		Temperature: 0.01,
		MaxTokens:   50,
	}
	resp, err := c.Chat(t.Context(), msgs, &opts)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("Raw response: %#v", resp)
	testUsage(t, &resp.Usage)
	if len(resp.Contents) != 1 {
		t.Fatal("Unexpected response")
	}
	if saw := strings.TrimSpace(strings.ToLower(resp.Contents[0].Text)); saw != "banana" {
		t.Fatal(saw)
	}
}

// JSON

// TestChatJSON runs a Chat verifying that the model correctly outputs JSON.
func TestChatJSON(t *testing.T, factory ChatProviderFactory, hasUsage bool) {
	c := factory(t)
	ctx := t.Context()
	msgs := genai.Messages{
		{
			Role: genai.User,
			Contents: []genai.Content{
				{Text: `Is a banana a fruit? Do not include an explanation. Reply ONLY as JSON according to the provided schema: {"is_fruit": bool}.`},
			},
		},
	}
	opts := genai.ChatOptions{
		Temperature: 0.01,
		MaxTokens:   200,
		Seed:        1,
		ReplyAsJSON: true,
	}
	resp, err := c.Chat(ctx, msgs, &opts)
	if uce, ok := err.(*genai.UnsupportedContinuableError); ok {
		t.Log(uce)
	} else if err != nil {
		t.Fatal(err)
	}
	t.Logf("Raw response: %#v", resp)
	// I'm disappointed by Cloudflare.
	if hasUsage {
		testUsage(t, &resp.Usage)
	}
	if len(resp.Contents) != 1 {
		t.Fatal("Unexpected response")
	}
	got := map[string]any{}
	if err := resp.Contents[0].Decode(&got); err != nil {
		// Gemini returns a list of map. Tolerate that too.
		got2 := []map[string]any{}
		if err := resp.Contents[0].Decode(&got2); err != nil {
			t.Fatal(err)
		}
		if len(got2) != 1 {
			t.Fatal(got2)
		}
		got = got2[0]
	}
	val, ok := got["is_fruit"]
	if !ok {
		t.Fatal(got)
	}
	// Accept both strings and bool.
	switch v := val.(type) {
	case bool:
		if !v {
			t.Fatal(got)
		}
	case string:
		if v != "true" {
			t.Fatal(got)
		}
	default:
		t.Fatal(got)
	}
}

// TestChatJSONSchema runs a Chat verifying that the model correctly outputs JSON according to a schema.
func TestChatJSONSchema(t *testing.T, factory ChatProviderFactory, hasUsage bool) {
	c := factory(t)
	ctx := t.Context()
	msgs := genai.Messages{
		{
			Role: genai.User,
			Contents: []genai.Content{
				{Text: "Is a banana a fruit? Reply as JSON according to the provided schema."},
			},
		},
	}
	// TODO: Test optional vs required, enum, bool, int, etc.
	var got struct {
		IsFruit bool `json:"is_fruit" jsonschema_description:"True  if the answer is that it is a fruit, false otherwise"`
	}
	opts := genai.ChatOptions{
		Temperature: 0.01,
		MaxTokens:   200,
		Seed:        1,
		DecodeAs:    &got,
	}
	resp, err := c.Chat(ctx, msgs, &opts)
	if uce, ok := err.(*genai.UnsupportedContinuableError); ok {
		t.Log(uce)
	} else if err != nil {
		t.Fatal(err)
	}
	t.Logf("Raw response: %#v", resp)
	// I'm disappointed by Cloudflare.
	if hasUsage {
		testUsage(t, &resp.Usage)
	}
	if len(resp.Contents) != 1 {
		t.Fatal("Unexpected response")
	}
	if err := resp.Contents[0].Decode(&got); err != nil {
		t.Fatal(err)
	}
	if !got.IsFruit {
		t.Fatal(got.IsFruit)
	}
}

//

// See the 3kib banana jpg online at
// https://github.com/maruel/genai/blob/main/openai/testdata/banana.jpg
//
//go:embed testdata/banana.jpg
var bananaJpg []byte

// processChat runs Chat and returns the result.
func processChat(t *testing.T, ctx context.Context, c genai.ChatProvider, msgs genai.Messages, opts *genai.ChatOptions) genai.ChatResult {
	resp, err := c.Chat(ctx, msgs, opts)
	if uce, ok := err.(*genai.UnsupportedContinuableError); ok {
		t.Log(uce)
	} else if err != nil {
		t.Fatal(err)
	}
	t.Logf("Raw response: %#v", resp)
	return resp
}

// processChatStream runs ChatStream and returns the result.
func processChatStream(t *testing.T, ctx context.Context, c genai.ChatProvider, msgs genai.Messages, opts *genai.ChatOptions) genai.ChatResult {
	chunks := make(chan genai.MessageFragment)
	end := make(chan genai.Messages, 1)
	go func() {
		var pendingMsgs genai.Messages
		defer func() {
			end <- pendingMsgs
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
					t.Error(err2)
					return
				}
			}
		}
	}()
	usage, err := c.ChatStream(ctx, msgs, opts, chunks)
	close(chunks)
	responses := <-end
	t.Logf("Raw responses: %#v", responses)
	testUsage(t, &usage)
	if uce, ok := err.(*genai.UnsupportedContinuableError); ok {
		t.Log(uce)
	} else if err != nil {
		t.Fatal(err)
	}
	if len(responses) == 0 {
		t.Fatal("No response received")
	}
	if len(responses) > 1 {
		t.Fatalf("Multiple responses received: %#v", responses)
	}
	resp := genai.ChatResult{Message: responses[len(responses)-1]}
	t.Logf("Raw response: %#v", resp)
	return resp
}

func testUsage(t *testing.T, u *genai.Usage) {
	if u.InputTokens == 0 {
		t.Error("expected Usage.InputTokens to be set")
	}
	if u.OutputTokens == 0 {
		t.Error("expected Usage.OutputTokens to be set")
	}
}
