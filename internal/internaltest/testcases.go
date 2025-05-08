// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package internaltest

import (
	"bytes"
	"context"
	_ "embed"
	"strings"
	"testing"

	"github.com/maruel/genai"
)

// ChatProviderFactory is what a Java developer would write.
type ChatProviderFactory func(t *testing.T) genai.ChatProvider

// ChatStream runs a ChatStream and returns the concatenated response.
func ChatStream(t *testing.T, factory ChatProviderFactory, msgs genai.Messages, opts *genai.ChatOptions) genai.Messages {
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
	err := c.ChatStream(ctx, msgs, opts, chunks)
	close(chunks)
	responses := <-end
	t.Logf("Raw responses: %#v", responses)
	if err != nil {
		t.Fatal(err)
	}
	return responses
}

// TestChatToolUseCountry runs a Chat with tool use and verifies that the tools are called correctly.
// It runs subtests for both Chat and ChatStream methods.
func TestChatToolUseCountry(t *testing.T, factory ChatProviderFactory) {
	t.Run("Chat", func(t *testing.T) {
		chatToolUseCountryCore(t, factory, false)
	})
	t.Run("ChatStream", func(t *testing.T) {
		t.Skip("TODO")
		chatToolUseCountryCore(t, factory, true)
	})
}

// processChat runs either Chat or ChatStream based on the useStream parameter
// and returns the result.
func processChat(t *testing.T, ctx context.Context, c genai.ChatProvider, msgs genai.Messages, opts *genai.ChatOptions, useStream bool) genai.ChatResult {
	var resp genai.ChatResult
	var err error
	if useStream {
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
		err = c.ChatStream(ctx, msgs, opts, chunks)
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
		resp = genai.ChatResult{Message: responses[len(responses)-1]}
	} else {
		resp, err = c.Chat(ctx, msgs, opts)
		if uce, ok := err.(*genai.UnsupportedContinuableError); ok {
			t.Log(uce)
		} else if err != nil {
			t.Fatal(err)
		}
	}
	t.Logf("Raw response: %#v", resp)
	return resp
}

// chatToolUseCountryCore runs a Chat or ChatStream with tool use and verifies that the tools are called correctly.
// The useStream parameter determines whether to use Chat or ChatStream.
// It returns the response for further validation.
func chatToolUseCountryCore(t *testing.T, factory ChatProviderFactory, useStream bool) {
	c := factory(t)
	ctx := t.Context()
	t.Run("Canada", func(t *testing.T) {
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
		resp := processChat(t, ctx, c, msgs, &opts, useStream)

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
	})
	t.Run("USA", func(t *testing.T) {
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
		resp := processChat(t, ctx, c, msgs, &opts, useStream)

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
	})
}

// TestChatVisionText runs a Chat with vision capabilities and verifies that the model correctly identifies a
// banana image.
func TestChatVisionText(t *testing.T, factory ChatProviderFactory) {
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
	if len(resp.Contents) != 1 {
		t.Fatal("Unexpected response")
	}
	// Normalize some of the variance. Obviously many models will still fail this test.
	txt := strings.TrimRight(strings.TrimSpace(strings.ToLower(resp.Contents[0].Text)), ".!")
	if txt != "yes" {
		t.Fatal(txt)
	}
}

// TestChatVisionJSON runs a Chat with vision capabilities and verifies that the model correctly identifies a
// banana image. It enforces JSON schema.
func TestChatVisionJSON(t *testing.T, factory ChatProviderFactory) {
	c := factory(t)
	ctx := t.Context()
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
	if len(resp.Contents) != 1 {
		t.Fatal("Unexpected response")
	}
	if err := resp.Contents[0].Decode(&got); err != nil {
		t.Fatal(err)
	}
	if !got.Banana {
		t.Fatal(got.Banana)
	}
}

//

// See the 3kib banana jpg online at
// https://github.com/maruel/genai/blob/main/openai/testdata/banana.jpg
//
//go:embed testdata/banana.jpg
var bananaJpg []byte
