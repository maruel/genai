// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package internaltest

import (
	"bytes"
	"context"
	_ "embed"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/maruel/genai"
	"golang.org/x/sync/errgroup"
)

// ChatProviderFactory is what a Java developer would write.
type ChatProviderFactory func(t *testing.T) genai.ChatProvider

// ModelChatProviderFactory is what a Java developer would write.
type ModelChatProviderFactory func(t *testing.T, model string) genai.ChatProvider

type Settings struct {
	// GetClient is a factory function that returns a chat provider for a specific model.
	GetClient ModelChatProviderFactory
	// Model is the model to use when none is specified.
	Model string
	// Options return a genai.ChatOptions or one of the model specific options.
	Options func(opts *genai.ChatOptions) genai.Validatable
	// UsageIsBroken is true if the provider doesn't report usage properly, i.e. it is buggy.
	UsageIsBroken bool
	// FinishReasonIsBroken is true if the provider doesn't report finish reason properly, i.e. it is buggy.
	FinishReasonIsBroken bool
}

// TestCases contains shared test cases that can be reused across providers.
type TestCases struct {
	// Default is the default settings to use.
	Default Settings
}

func (tc *TestCases) getClient(t *testing.T, override *Settings) genai.ChatProvider {
	model := tc.Default.Model
	gc := tc.Default.GetClient
	if override != nil {
		if override.Model != "" {
			model = override.Model
		}
		if override.GetClient != nil {
			gc = override.GetClient
		}
	}
	return gc(t, model)
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

func (tc *TestCases) finishReasonIsBroken(override *Settings) bool {
	if override != nil && override.FinishReasonIsBroken {
		return true
	}
	return tc.Default.FinishReasonIsBroken
}

// TestChatThinking runs a test for the thinking feature of a chat model.
func (tc *TestCases) TestChatThinking(t *testing.T, override *Settings) {
	t.Run("Chat", func(t *testing.T) {
		msgs := genai.Messages{genai.NewTextMessage(genai.User, "Say hello. Use only one word. Say only hello.")}
		// I believe most thinking models do not like Temperature to be set.
		opts := tc.getOptions(&genai.ChatOptions{MaxTokens: 2000, Seed: 1}, override)
		c := tc.getClient(t, override)
		usageIsBroken := tc.usageIsBroken(override)
		finishReasonIsBroken := tc.finishReasonIsBroken(override)
		resp := tc.testChat(t, msgs, c, opts, usageIsBroken, finishReasonIsBroken)
		ValidateSingleWordResponse(t, resp, "hello")
		msgs = append(msgs, resp.Message, genai.NewTextMessage(genai.User, "Say the same word again. Use only one word."))
		resp = tc.testChat(t, msgs, c, opts, usageIsBroken, finishReasonIsBroken)
		ValidateSingleWordResponse(t, resp, "hello")
	})
	t.Run("ChatStream", func(t *testing.T) {
		msgs := genai.Messages{genai.NewTextMessage(genai.User, "Say hello. Use only one word. Say only hello.")}
		// I believe most thinking models do not like Temperature to be set.
		opts := tc.getOptions(&genai.ChatOptions{MaxTokens: 2000, Seed: 1}, override)
		c := tc.getClient(t, override)
		usageIsBroken := tc.usageIsBroken(override)
		finishReasonIsBroken := tc.finishReasonIsBroken(override)
		resp := tc.testChatStream(t, msgs, c, opts, usageIsBroken, finishReasonIsBroken)
		ValidateSingleWordResponse(t, resp, "hello")
		msgs = append(msgs, resp.Message, genai.NewTextMessage(genai.User, "Say the same word again. Use only one word."))
		resp = tc.testChatStream(t, msgs, c, opts, usageIsBroken, finishReasonIsBroken)
		ValidateSingleWordResponse(t, resp, "hello")
	})
}

// TestChatSimple_simple makes sure Chat() works. Useful to restart a recording to determine if new fields were
// added.
func (tc *TestCases) TestChatSimple_simple(t *testing.T, override *Settings) {
	msgs := genai.Messages{genai.NewTextMessage(genai.User, "Say hello. Use only one word.")}
	opts := genai.ChatOptions{Temperature: 0.01, MaxTokens: 2000, Seed: 1}
	resp := tc.TestChatHelper(t, msgs, override, opts)
	ValidateSingleWordResponse(t, resp, "hello")
}

// TestChatStream_simple makes sure ChatStream() works. Useful to restart a recording to determine if new fields were
// added.
func (tc *TestCases) TestChatStream_simple(t *testing.T, override *Settings) {
	msgs := genai.Messages{genai.NewTextMessage(genai.User, "Say hello. Use only one word.")}
	opts := genai.ChatOptions{Temperature: 0.01, MaxTokens: 2000, Seed: 1}
	resp := tc.testChatStreamHelper(t, msgs, override, opts)
	ValidateSingleWordResponse(t, resp, "hello")
}

// TestChatAllModels says hello with all models.
func (tc *TestCases) TestChatAllModels(t *testing.T, filter func(model genai.Model) bool) {
	ctx := t.Context()
	l := tc.getClient(t, nil).(genai.ModelProvider)
	models, err := l.ListModels(ctx)
	if err != nil {
		t.Fatal(err)
	}

	// MaxTokens has to be long because of some thinking models (e.g. qwen-qwq-32b and
	// deepseek-r1-distill-llama-70b) cannot have thinking disabled.
	opts := genai.ChatOptions{Temperature: 0.1, Seed: 1, MaxTokens: 1000}

	for _, m := range models {
		id := m.GetID()
		if filter != nil && !filter(m) {
			continue
		}
		t.Run(id, func(t *testing.T) {
			msgs := genai.Messages{genai.NewTextMessage(genai.User, "Say hello. Use only one word. Say only hello.")}
			resp := tc.TestChatHelper(t, msgs, &Settings{Model: id}, opts)
			ValidateSingleWordResponse(t, resp, "hello")
		})
	}
}

// Tool

// TestChatToolUseReply confirms tool use fully works.
func (tc *TestCases) TestChatToolUseReply(t *testing.T, override *Settings) {
	ctx := t.Context()
	msgs := genai.Messages{
		genai.NewTextMessage(genai.User, "Use the square_root tool to calculate the square root of 132413 and reply with only the result. Do not give an explanation."),
	}
	type got struct {
		Number json.Number `json:"number"`
	}
	opts := genai.ChatOptions{
		SystemPrompt: "You are an helpful assistant that is very succinct. You only reply to the user's request with no additional information. Use the tools at your disposal and return their result as-is.",
		// Must be long enough for thinking models.
		MaxTokens: 4096,
		Seed:      1,
		Tools: []genai.ToolDef{
			{
				Name:        "square_root",
				Description: "Calculates and return the square root of a number",
				Callback: func(ctx context.Context, g *got) (string, error) {
					i, err := g.Number.Int64()
					if err != nil {
						return "", fmt.Errorf("wanted 132413 as an int, got %q: %w", g.Number, err)
					}
					if i != 132413 {
						return "", fmt.Errorf("wanted 132413 as an int, got %s", g.Number)
					}
					return fmt.Sprintf("%.2f", math.Sqrt(float64(i))), nil
				},
			},
		},
		// For this test, we want to make sure the tool is called.
		ToolCallRequest: genai.ToolCallRequired,
	}
	c := tc.getClient(t, override)
	resp := tc.testChat(t, msgs, c, tc.getOptions(&opts, override), tc.usageIsBroken(override), tc.finishReasonIsBroken(override))
	want := "square_root"
	if len(resp.ToolCalls) == 0 || resp.ToolCalls[0].Name != want {
		t.Fatalf("Expected tool call to %s, got: %v", want, resp.ToolCalls)
	}
	// Don't forget to add the tool call request first before the reply.
	msgs = append(msgs, resp.Message)
	msg, err := resp.DoToolCalls(ctx, opts.Tools)
	if err != nil {
		t.Fatal(err)
	}
	if !msg.IsZero() {
		// Don't forget to add the tool call request first before the reply.
		msgs = append(msgs, msg)
	} else {
		t.Fatal("unexpected zero message")
	}
	// Important!
	opts.ToolCallRequest = genai.ToolCallNone
	resp = tc.testChat(t, msgs, c, tc.getOptions(&opts, override), tc.usageIsBroken(override), tc.finishReasonIsBroken(override))
	// This is very annoying, llama4 is not following instructions.

	ValidateSingleWordResponse(t, resp, "363.89")
}

// TestChatToolUsePositionBias confirms that LLMs are position biased.
//
// Presented with a choice where they can't easily chose, they will always select the first item presented
// (!). In this case, this is a country name.
func (tc *TestCases) TestChatToolUsePositionBias(t *testing.T, override *Settings, allowOther bool) {
	t.Run("Chat", func(t *testing.T) {
		tc.TestChatToolUsePositionBiasCore(t, override, allowOther, false)
	})
	t.Run("ChatStream", func(t *testing.T) {
		tc.TestChatToolUsePositionBiasCore(t, override, allowOther, true)
	})
}

// TestChatToolUsePositionBiasCore runs a Chat or ChatStream with tool use and verifies that the tools are called correctly.
// The useStream parameter determines whether to use Chat or ChatStream.
// It returns the response for further validation.
func (tc *TestCases) TestChatToolUsePositionBiasCore(t *testing.T, override *Settings, allowOther, useStream bool) {
	type gotCanadaFirst struct {
		Country string `json:"country" jsonschema:"enum=Canada,enum=USA"`
	}
	type gotUSAFirst struct {
		Country string `json:"country" jsonschema:"enum=USA,enum=Canada"`
	}
	data := []struct {
		callback        any
		countrySelected string
		country1        string
		country2        string
	}{
		{
			func(ctx context.Context, g *gotCanadaFirst) (string, error) {
				return g.Country, nil
			},
			"Canada", "Canada", "the USA",
		},
		{
			func(ctx context.Context, g *gotUSAFirst) (string, error) {
				return g.Country, nil
			},
			"USA", "the USA", "Canada",
		},
	}
	for _, line := range data {
		t.Run(line.countrySelected, func(t *testing.T) {
			ctx := t.Context()
			msgs := genai.Messages{
				genai.NewTextMessage(genai.User, fmt.Sprintf("I wonder if %s is a better country than %s? Call the tool best_country to tell me which country is the best one.", line.country1, line.country2)),
			}
			opts := genai.ChatOptions{
				// Must be long enough for thinking models.
				MaxTokens: 4096,
				Seed:      1,
				Tools: []genai.ToolDef{
					{
						Name:        "best_country",
						Description: "A tool to determine the best country",
						Callback:    line.callback,
					},
				},
				ToolCallRequest: genai.ToolCallRequired,
			}
			var resp genai.ChatResult
			if useStream {
				resp = tc.testChatStreamHelper(t, msgs, override, opts)
			} else {
				resp = tc.TestChatHelper(t, msgs, override, opts)
			}
			want := "best_country"
			if len(resp.ToolCalls) == 0 || resp.ToolCalls[0].Name != want {
				t.Fatalf("Expected tool call to %s, got: %v", want, resp.ToolCalls)
			}
			res, err := resp.ToolCalls[0].Call(ctx, opts.Tools)
			if err != nil {
				t.Fatal(err)
			}
			if !allowOther && res != line.countrySelected {
				t.Fatal(res)
			}
		})
	}
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
	resp := tc.TestChatHelper(t, msgs, override, opts)
	// Normalize some of the variance. Obviously many models will still fail this test.
	txt := strings.TrimRight(strings.TrimSpace(strings.ToLower(resp.AsText())), ".!")
	if txt != "yes" {
		t.Fatal(txt)
	}
}

func (tc *TestCases) TestChatVisionPDFInline(t *testing.T, override *Settings) {
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
	opts := genai.ChatOptions{Temperature: 0.01, MaxTokens: 50, Seed: 1}
	resp := tc.TestChatHelper(t, msgs, override, opts)
	if got := strings.TrimSpace(strings.ToLower(resp.AsText())); got != "orange" {
		t.Fatal(got)
	}
}

func (tc *TestCases) TestChatVisionPDFURL(t *testing.T, override *Settings) {
	msgs := genai.Messages{
		{
			Role: genai.User,
			Contents: []genai.Content{
				{Text: "What is the word? Reply with only the word."},
				{URL: "https://raw.githubusercontent.com/maruel/genai/refs/heads/main/internal/internaltest/testdata/hidden_word.pdf"},
			},
		},
	}
	opts := genai.ChatOptions{Temperature: 0.01, MaxTokens: 50, Seed: 1}
	resp := tc.TestChatHelper(t, msgs, override, opts)
	if got := strings.TrimSpace(strings.ToLower(resp.AsText())); got != "orange" {
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
	opts := genai.ChatOptions{Temperature: 0.01, MaxTokens: 50, Seed: 1}
	resp := tc.TestChatHelper(t, msgs, override, opts)
	if heard := strings.TrimRight(strings.ToLower(resp.AsText()), "."); heard != "orange" {
		t.Fatal(heard)
	}
}

func (tc *TestCases) TestChatVideoMP4Inline(t *testing.T, override *Settings) {
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
	opts := genai.ChatOptions{Temperature: 0.01, MaxTokens: 50, Seed: 1}
	resp := tc.TestChatHelper(t, msgs, override, opts)
	if saw := strings.TrimSpace(strings.ToLower(resp.AsText())); saw != "banana" {
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
	resp := tc.TestChatHelper(t, msgs, override, opts)
	got := map[string]any{}
	if err := resp.Decode(&got); err != nil {
		// Gemini returns a list of map. Tolerate that too.
		got2 := []map[string]any{}
		if err := resp.Decode(&got2); err != nil {
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
	resp := tc.TestChatHelper(t, msgs, override, opts)
	if err := resp.Decode(&got); err != nil {
		t.Fatal(err)
	}
	if !got.IsFruit {
		t.Fatal(got.IsFruit)
	}
}

func (tc *TestCases) TestChatHelper(t *testing.T, msgs genai.Messages, override *Settings, opts genai.ChatOptions) genai.ChatResult {
	return tc.testChat(t, msgs, tc.getClient(t, override), tc.getOptions(&opts, override),
		tc.usageIsBroken(override), tc.finishReasonIsBroken(override))
}

func (tc *TestCases) testChat(t *testing.T, msgs genai.Messages, c genai.ChatProvider, opts genai.Validatable, usageIsBroken, finishReasonIsBroken bool) genai.ChatResult {
	ctx := t.Context()
	resp, err := c.Chat(ctx, msgs, opts)
	if uce, ok := err.(*genai.UnsupportedContinuableError); ok {
		t.Log(uce)
	} else if err != nil {
		t.Fatal(err)
	}
	t.Logf("Raw response: %#v", resp)
	testUsage(t, &resp.Usage, usageIsBroken, finishReasonIsBroken)
	if len(resp.Contents) == 0 && len(resp.ToolCalls) == 0 {
		t.Fatal("missing response")
	}
	return resp
}

func (tc *TestCases) testChatStreamHelper(t *testing.T, msgs genai.Messages, override *Settings, opts genai.ChatOptions) genai.ChatResult {
	return tc.testChatStream(t, msgs, tc.getClient(t, override), tc.getOptions(&opts, override),
		tc.usageIsBroken(override), tc.finishReasonIsBroken(override))
}

func (tc *TestCases) testChatStream(t *testing.T, msgs genai.Messages, c genai.ChatProvider, opts genai.Validatable, usageIsBroken, finishReasonIsBroken bool) genai.ChatResult {
	ctx := t.Context()
	chunks := make(chan genai.MessageFragment)
	// Assert that the message returned is the same as the one we accumulated.
	accumulated := genai.Message{}
	eg := errgroup.Group{}
	eg.Go(func() error {
		defer func() {
			for range chunks {
			}
		}()
		for {
			select {
			case <-ctx.Done():
				return nil
			case pkt, ok := <-chunks:
				if !ok {
					return nil
				}
				t.Logf("Packet: %#v", pkt)
				if err2 := accumulated.Accumulate(pkt); err2 != nil {
					return err2
				}
			}
		}
	})
	resp, err := c.ChatStream(ctx, msgs, opts, chunks)
	close(chunks)
	if err3 := eg.Wait(); err3 != nil {
		t.Fatal(err3)
	}
	t.Logf("Raw response: %#v", resp.Message)
	if uce, ok := err.(*genai.UnsupportedContinuableError); ok {
		t.Log(uce)
	} else if err != nil {
		t.Fatal(err)
	}
	if diff := cmp.Diff(&resp.Message, &accumulated); diff != "" {
		t.Errorf("(-result), (+accumulated):\n%s", diff)
	}
	testUsage(t, &resp.Usage, usageIsBroken, finishReasonIsBroken)
	return resp
}

//

type ChatProviderError struct {
	Name          string
	ApiKey        string
	Model         string
	ErrChat       string
	ErrChatStream string
}

func TestClient_ChatProvider_errors(t *testing.T, getClient func(t *testing.T, apiKey, model string) genai.ChatProvider, lines []ChatProviderError) {
	for _, line := range lines {
		t.Run(line.Name, func(t *testing.T) {
			msgs := genai.Messages{genai.NewTextMessage(genai.User, "Tell a short joke.")}
			if line.ErrChat != "" {
				t.Run("Chat", func(t *testing.T) {
					c := getClient(t, line.ApiKey, line.Model)
					_, err := c.Chat(t.Context(), msgs, &genai.ChatOptions{})
					if err == nil {
						t.Fatal("expected error")
					} else if _, ok := err.(*genai.UnsupportedContinuableError); ok {
						t.Fatal("should not be continuable")
					} else if got := err.Error(); got != line.ErrChat {
						t.Fatalf("Unexpected error.\nwant: %q\ngot : %q", line.ErrChat, got)
					}
				})
			}
			if line.ErrChatStream != "" {
				t.Run("ChatStream", func(t *testing.T) {
					c := getClient(t, line.ApiKey, line.Model)
					ch := make(chan genai.MessageFragment, 1)
					_, err := c.ChatStream(t.Context(), msgs, &genai.ChatOptions{}, ch)
					if err == nil {
						t.Fatal("expected error")
					} else if _, ok := err.(*genai.UnsupportedContinuableError); ok {
						t.Fatal("should not be continuable")
					} else if got := err.Error(); got != line.ErrChatStream {
						t.Fatalf("Unexpected error.\nwant: %q\ngot : %q", line.ErrChatStream, got)
					}
					select {
					case pkt := <-ch:
						t.Fatal(pkt)
					default:
					}
				})
			}
		})
	}
}

type ModelProviderError struct {
	Name   string
	ApiKey string
	Err    string
}

func TestClient_ModelProvider_errors(t *testing.T, getClient func(t *testing.T, apiKey string) genai.ModelProvider, lines []ModelProviderError) {
	for _, line := range lines {
		t.Run(line.Name, func(t *testing.T) {
			t.Run("ListModels", func(t *testing.T) {
				c := getClient(t, line.ApiKey)
				_, err := c.ListModels(t.Context())
				if err == nil {
					t.Fatal("expected error")
				} else if got := err.Error(); got != line.Err {
					t.Fatalf("Unexpected error.\nwant: %q\ngot : %q", line.Err, got)
				}
			})
		})
	}
}

//

// See the 3kib banana jpg online at
// https://github.com/maruel/genai/blob/main/openai/testdata/banana.jpg
//
//go:embed testdata/banana.jpg
var bananaJpg []byte

func testUsage(t *testing.T, u *genai.Usage, usageIsBroken, finishReasonIsBroken bool) {
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
	if finishReasonIsBroken {
		if u.FinishReason != "" {
			t.Fatalf("expected no FinishReason, got %q", u.FinishReason)
		}
	} else {
		if u.FinishReason == "" {
			t.Fatal("expected FinishReason")
		}
	}
}

// ValidateSingleWordResponse validates that the response contains exactly one of the expected words.
func ValidateSingleWordResponse(t *testing.T, resp genai.ChatResult, want string) {
	s := resp.AsText()
	if got := strings.TrimRight(strings.TrimSpace(strings.ToLower(s)), ".!"); want != got {
		t.Fatalf("Expected %q, got %q", want, s)
	}
}
