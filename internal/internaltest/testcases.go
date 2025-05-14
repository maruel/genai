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

type Settings struct {
	// Model is the model to use when none is specified.
	Model string
	// Options return a genai.ChatOptions or one of the model specific options.
	Options func(opts *genai.ChatOptions) genai.Validatable
	// UsageIsBroken is true if the provider doesn't report usage properly, i.e. it is buggy.
	UsageIsBroken bool
}

// TestCases contains shared test cases that can be reused across providers.
type TestCases struct {
	// GetClient is a factory function that returns a chat provider for a specific model.
	GetClient ModelChatProviderFactory
	// Default is the default settings to use.
	Default Settings
}

func (tc *TestCases) getClient(t *testing.T, override *Settings) genai.ChatProvider {
	model := tc.Default.Model
	if override != nil && override.Model != "" {
		model = override.Model
	}
	return tc.GetClient(t, model)
}

func (tc *TestCases) getOptions(opts *genai.ChatOptions, override *Settings) genai.Validatable {
	if override != nil && override.Options != nil {
		return override.Options(opts)
	}
	if tc.Default.Options != nil {
		return tc.Default.Options(opts)
	}
	return opts
}

func (tc *TestCases) usageIsBroken(override *Settings) bool {
	if override != nil && override.UsageIsBroken {
		return true
	}
	return tc.Default.UsageIsBroken
}

// TestChatThinking runs a test for the thinking feature of a chat model.
func (tc *TestCases) TestChatThinking(t *testing.T, override *Settings) {
	msgs := genai.Messages{genai.NewTextMessage(genai.User, "Say hello. Use only one word. Say only hello.")}
	// I believe most thinking models do not like Temperature to be set.
	opts := tc.getOptions(&genai.ChatOptions{MaxTokens: 2000, Seed: 1}, override)
	c := tc.getClient(t, override)
	usageIsBroken := tc.usageIsBroken(override)
	resp := tc.testChat(t, msgs, c, opts, usageIsBroken)
	validateSingleWordResponse(t, resp, "hello")
	msgs = append(msgs, resp.Message, genai.NewTextMessage(genai.User, "Say the same word again. Use only one word."))
	resp = tc.testChat(t, msgs, c, opts, usageIsBroken)
	validateSingleWordResponse(t, resp, "hello")
}

// TestChatStream makes sure ChatStream() works.
func (tc *TestCases) TestChatStream(t *testing.T, override *Settings) {
	msgs := genai.Messages{genai.NewTextMessage(genai.User, "Say hello. Use only one word.")}
	opts := genai.ChatOptions{Temperature: 0.01, MaxTokens: 50, Seed: 1}
	resp := tc.testChatStreamHelper(t, msgs, override, opts)
	validateSingleWordResponse(t, resp, "hello")
}

// TestChatAllModels says hello with all models.
func (tc *TestCases) TestChatAllModels(t *testing.T, filter func(model genai.Model) bool) {
	ctx := t.Context()
	l := tc.GetClient(t, "").(genai.ModelProvider)
	models, err := l.ListModels(ctx)
	if err != nil {
		t.Fatal(err)
	}

	// MaxTokens has to be long because of some thinking models (e.g. qwen-qwq-32b and
	// deepseek-r1-distill-llama-70b) cannot have thinking disabled.
	baseOpts := &genai.ChatOptions{Temperature: 0.1, Seed: 1, MaxTokens: 1000}
	opts := tc.getOptions(baseOpts, nil)

	for _, m := range models {
		id := m.GetID()
		if filter != nil && !filter(m) {
			continue
		}
		t.Run(id, func(t *testing.T) {
			t.Helper()
			c := tc.GetClient(t, id)
			msgs := genai.Messages{
				genai.NewTextMessage(genai.User, "Say hello. Use only one word. Say only hello."),
			}
			resp, err := c.Chat(ctx, msgs, opts)
			if uce, ok := err.(*genai.UnsupportedContinuableError); ok {
				t.Log(uce)
			} else if err != nil {
				t.Fatal(err)
			}
			t.Logf("Raw response: %#v", resp)
			testUsage(t, &resp.Usage, false)
			validateSingleWordResponse(t, resp, "hello")
		})
	}
}

// Tool

// TestChatToolUseCountry runs a Chat with tool use and verifies that the tools are called correctly.
// It runs subtests for both Chat and ChatStream methods.
func (tc *TestCases) TestChatToolUseCountry(t *testing.T, override *Settings) {
	t.Run("Chat", func(t *testing.T) {
		tc.chatToolUseCountryCore(t, override, false)
	})
	t.Run("ChatStream", func(t *testing.T) {
		t.Skip("TODO")
		tc.chatToolUseCountryCore(t, override, true)
	})
}

// chatToolUseCountryCore runs a Chat or ChatStream with tool use and verifies that the tools are called correctly.
// The useStream parameter determines whether to use Chat or ChatStream.
// It returns the response for further validation.
func (tc *TestCases) chatToolUseCountryCore(t *testing.T, override *Settings, useStream bool) {
	ctx := t.Context()
	t.Run("Canada", func(t *testing.T) {
		c := tc.getClient(t, override)
		msgs := genai.Messages{
			genai.NewTextMessage(genai.User, "I wonder if Canada is a better country than the USA? Call the tool best_country to tell me which country is the best one."),
		}
		var got struct {
			Country string `json:"country" jsonschema:"enum=Canada,enum=USA"`
		}
		opts := tc.getOptions(&genai.ChatOptions{
			MaxTokens: 200,
			Seed:      1,
			Tools: []genai.ToolDef{
				{
					Name:        "best_country",
					Description: "A tool to determine the best country",
					InputsAs:    &got,
				},
			},
			ToolCallRequired: true,
		}, override)
		var resp genai.ChatResult
		if useStream {
			resp = processChatStream(t, ctx, c, msgs, opts)
		} else {
			resp = processChat(t, ctx, c, msgs, opts)
		}
		testUsage(t, &resp.Usage, tc.usageIsBroken(override))

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
		c := tc.getClient(t, override)
		msgs := genai.Messages{
			genai.NewTextMessage(genai.User, "I wonder if the USA is a better country than Canada? Call the tool best_country to tell me which country is the best one."),
		}
		var got struct {
			Country string `json:"country" jsonschema:"enum=USA,enum=Canada"`
		}
		opts := tc.getOptions(&genai.ChatOptions{
			MaxTokens: 200,
			Seed:      1,
			Tools: []genai.ToolDef{
				{
					Name:        "best_country",
					Description: "A tool to determine the best country",
					InputsAs:    &got,
				},
			},
			ToolCallRequired: true,
		}, override)
		var resp genai.ChatResult
		if useStream {
			resp = processChatStream(t, ctx, c, msgs, opts)
		} else {
			resp = processChat(t, ctx, c, msgs, opts)
		}
		testUsage(t, &resp.Usage, tc.usageIsBroken(override))

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
func (tc *TestCases) TestChatVisionJPGInline(t *testing.T, override *Settings) {
	msgs := genai.Messages{
		{
			Role: genai.User,
			Contents: []genai.Content{
				{Text: "Is it a banana? Reply with only one word."},
				{Filename: "banana.jpg", Document: bytes.NewReader(bananaJpg)},
			},
		},
	}
	opts := genai.ChatOptions{Temperature: 0.01, MaxTokens: 200, Seed: 1}
	resp := tc.testChatVisionHelper(t, msgs, override, opts)
	if len(resp.Contents) != 1 {
		t.Fatal("Unexpected response")
	}
	// Normalize some of the variance. Obviously many models will still fail this test.
	txt := strings.TrimRight(strings.TrimSpace(strings.ToLower(resp.Contents[0].Text)), ".!")
	if txt != "yes" {
		t.Fatal(txt)
	}
}

func (tc *TestCases) TestChatVisionPDFInline(t *testing.T, override *Settings) {
	c := tc.getClient(t, override)
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
	opts := tc.getOptions(&genai.ChatOptions{Temperature: 0.01, MaxTokens: 50, Seed: 1}, override)
	resp, err := c.Chat(t.Context(), msgs, opts)
	if uce, ok := err.(*genai.UnsupportedContinuableError); ok {
		t.Log(uce)
	} else if err != nil {
		t.Fatal(err)
	}
	t.Logf("Raw response: %#v", resp)
	testUsage(t, &resp.Usage, tc.usageIsBroken(override))
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

func (tc *TestCases) TestChatVisionPDFURL(t *testing.T, override *Settings) {
	c := tc.getClient(t, override)
	msgs := genai.Messages{
		{
			Role: genai.User,
			Contents: []genai.Content{
				{Text: "What is the word? Reply with only the word."},
				{URL: "https://raw.githubusercontent.com/maruel/genai/refs/heads/main/internal/internaltest/testdata/hidden_word.pdf"},
			},
		},
	}
	opts := tc.getOptions(&genai.ChatOptions{Temperature: 0.01, MaxTokens: 50, Seed: 1}, override)
	resp, err := c.Chat(t.Context(), msgs, opts)
	if uce, ok := err.(*genai.UnsupportedContinuableError); ok {
		t.Log(uce)
	} else if err != nil {
		t.Fatal(err)
	}
	t.Logf("Raw response: %#v", resp)
	testUsage(t, &resp.Usage, tc.usageIsBroken(override))
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

func (tc *TestCases) TestChatAudioMP3Inline(t *testing.T, override *Settings) {
	tc.testChatAudioInline(t, override, "mystery_word.mp3")
}

func (tc *TestCases) TestChatAudioOpusInline(t *testing.T, override *Settings) {
	tc.testChatAudioInline(t, override, "mystery_word.opus")
}

func (tc *TestCases) testChatAudioInline(t *testing.T, override *Settings, filename string) {
	c := tc.getClient(t, override)
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
	opts := tc.getOptions(&genai.ChatOptions{Temperature: 0.01, MaxTokens: 50, Seed: 1}, override)
	resp, err := c.Chat(t.Context(), msgs, opts)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("Raw response: %#v", resp)
	testUsage(t, &resp.Usage, tc.usageIsBroken(override))
	if len(resp.Contents) != 1 {
		t.Fatal("Unexpected response")
	}
	if heard := strings.TrimRight(strings.ToLower(resp.Contents[0].Text), "."); heard != "orange" {
		t.Fatal(heard)
	}
}

func (tc *TestCases) TestChatVideoMP4Inline(t *testing.T, override *Settings) {
	c := tc.getClient(t, override)
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
	opts := tc.getOptions(&genai.ChatOptions{Temperature: 0.01, MaxTokens: 50, Seed: 1}, override)
	resp, err := c.Chat(t.Context(), msgs, opts)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("Raw response: %#v", resp)
	testUsage(t, &resp.Usage, tc.usageIsBroken(override))
	if len(resp.Contents) != 1 {
		t.Fatal("Unexpected response")
	}
	if saw := strings.TrimSpace(strings.ToLower(resp.Contents[0].Text)); saw != "banana" {
		t.Fatal(saw)
	}
}

// JSON

// TestChatJSON runs a Chat verifying that the model correctly outputs JSON.
func (tc *TestCases) TestChatJSON(t *testing.T, override *Settings) {
	msgs := genai.Messages{
		genai.NewTextMessage(genai.User, `Is a banana a fruit? Do not include an explanation. Reply ONLY as JSON according to the provided schema: {"is_fruit": bool}.`),
	}
	opts := genai.ChatOptions{Temperature: 0.1, MaxTokens: 200, Seed: 1, ReplyAsJSON: true}
	resp := tc.testChatHelper(t, msgs, override, opts)
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
func (tc *TestCases) TestChatJSONSchema(t *testing.T, override *Settings) {
	// TODO: Test optional vs required, enum, bool, int, etc.
	var got struct {
		IsFruit bool `json:"is_fruit" jsonschema_description:"True if the answer is that it is a fruit, false otherwise"`
	}
	msgs := genai.Messages{genai.NewTextMessage(genai.User, "Is a banana a fruit? Reply as JSON according to the provided schema.")}
	opts := genai.ChatOptions{Temperature: 0.1, MaxTokens: 200, Seed: 1, DecodeAs: &got}
	resp := tc.testChatHelper(t, msgs, override, opts)
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
func processChat(t *testing.T, ctx context.Context, c genai.ChatProvider, msgs genai.Messages, opts genai.Validatable) genai.ChatResult {
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
func processChatStream(t *testing.T, ctx context.Context, c genai.ChatProvider, msgs genai.Messages, opts genai.Validatable) genai.ChatResult {
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
	resp := genai.ChatResult{Message: responses[len(responses)-1], Usage: usage}
	t.Logf("Raw response: %#v", resp)
	return resp
}

func testUsage(t *testing.T, u *genai.Usage, usageIsBroken bool) {
	if usageIsBroken {
		if u.InputTokens != 0 {
			t.Error("expected Usage.InputTokens to be zero")
		}
		if u.InputCachedTokens != 0 {
			t.Error("expected Usage.OutputTokens to be zero")
		}
		if u.OutputTokens != 0 {
			t.Error("expected Usage.OutputTokens to be zero")
		}
	} else {
		if u.InputTokens == 0 {
			t.Error("expected Usage.InputTokens to be set")
		}
		if u.OutputTokens == 0 {
			t.Error("expected Usage.OutputTokens to be set")
		}
	}
}

func (tc *TestCases) testChatHelper(t *testing.T, msgs genai.Messages, override *Settings, opts genai.ChatOptions) genai.ChatResult {
	return tc.testChat(t, msgs, tc.getClient(t, override), tc.getOptions(&opts, override), tc.usageIsBroken(override))
}

func (tc *TestCases) testChat(t *testing.T, msgs genai.Messages, c genai.ChatProvider, opts genai.Validatable, usageIsBroken bool) genai.ChatResult {
	ctx := t.Context()
	resp, err := c.Chat(ctx, msgs, opts)
	if uce, ok := err.(*genai.UnsupportedContinuableError); ok {
		t.Log(uce)
	} else if err != nil {
		t.Fatal(err)
	}
	t.Logf("Raw response: %#v", resp)
	testUsage(t, &resp.Usage, usageIsBroken)
	if len(resp.Contents) != 1 {
		t.Fatal("Unexpected response")
	}
	return resp
}

// validateSingleWordResponse validates that the response contains exactly one of the expected words.
func validateSingleWordResponse(t *testing.T, resp genai.ChatResult, want string) {
	if len(resp.Contents) != 1 {
		t.Fatal("Unexpected response")
	}
	if got := strings.TrimRight(strings.TrimSpace(strings.ToLower(resp.Contents[0].Text)), ".!"); want != got {
		t.Fatalf("Expected %q, got %q", want, got)
	}
}

// testChatStreamHelper is a general helper function to run ChatStream tests with consistent patterns.
func (tc *TestCases) testChatStreamHelper(t *testing.T, msgs genai.Messages, override *Settings, opts genai.ChatOptions) genai.ChatResult {
	c := tc.getClient(t, override)
	ctx := t.Context()
	resp := processChatStream(t, ctx, c, msgs, tc.getOptions(&opts, override))
	testUsage(t, &resp.Usage, tc.usageIsBroken(override))
	return resp
}

// testChatVisionHelper is a general helper function for vision tests.
func (tc *TestCases) testChatVisionHelper(t *testing.T, msgs genai.Messages, override *Settings, opts genai.ChatOptions) genai.ChatResult {
	c := tc.getClient(t, override)
	ctx := t.Context()
	resp, err := c.Chat(ctx, msgs, tc.getOptions(&opts, override))
	if uce, ok := err.(*genai.UnsupportedContinuableError); ok {
		t.Log(uce)
	} else if err != nil {
		t.Fatal(err)
	}
	t.Logf("Raw response: %#v", resp)
	testUsage(t, &resp.Usage, tc.usageIsBroken(override))
	return resp
}
