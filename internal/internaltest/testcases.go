// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package internaltest

import (
	"context"
	_ "embed"
	"fmt"
	"slices"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/maruel/genai"
	"golang.org/x/sync/errgroup"
)

// ProviderModelChatFactory is what a Java developer would write.
type ProviderModelChatFactory func(t *testing.T, model string) genai.ProviderChat

type Settings struct {
	// GetClient is a factory function that returns a chat provider for a specific model.
	GetClient ProviderModelChatFactory
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

func (tc *TestCases) getClient(t *testing.T, override *Settings) genai.ProviderChat {
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

// Tool

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
				resp = tc.testChatStreamHelper(t, msgs, override, opts, genai.FinishedToolCalls)
			} else {
				resp = tc.testChatHelper(t, msgs, override, opts, genai.FinishedToolCalls)
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

func (tc *TestCases) testChatHelper(t *testing.T, msgs genai.Messages, override *Settings, opts genai.ChatOptions, f genai.FinishReason) genai.ChatResult {
	if tc.finishReasonIsBroken(override) {
		f = ""
	}
	return tc.testChat(t, msgs, tc.getClient(t, override), tc.getOptions(&opts, override), tc.usageIsBroken(override), f)
}

func (tc *TestCases) testChat(t *testing.T, msgs genai.Messages, c genai.ProviderChat, opts genai.Validatable, usageIsBroken bool, f genai.FinishReason) genai.ChatResult {
	ctx := t.Context()
	resp, err := c.Chat(ctx, msgs, opts)
	if uce, ok := err.(*genai.UnsupportedContinuableError); ok {
		t.Log(uce)
	} else if err != nil {
		t.Fatal(err)
	}
	t.Logf("Raw response: %#v", resp)
	testUsage(t, &resp.Usage, usageIsBroken, f)
	if len(resp.Contents) == 0 && len(resp.ToolCalls) == 0 {
		t.Fatal("missing response")
	}
	return resp
}

func (tc *TestCases) testChatStreamHelper(t *testing.T, msgs genai.Messages, override *Settings, opts genai.ChatOptions, f genai.FinishReason) genai.ChatResult {
	if tc.finishReasonIsBroken(override) {
		f = ""
	}
	return tc.testChatStream(t, msgs, tc.getClient(t, override), tc.getOptions(&opts, override), tc.usageIsBroken(override), f)
}

func (tc *TestCases) testChatStream(t *testing.T, msgs genai.Messages, c genai.ProviderChat, opts genai.Validatable, usageIsBroken bool, f genai.FinishReason) genai.ChatResult {
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
	testUsage(t, &resp.Usage, usageIsBroken, f)
	return resp
}

//

type ProviderChatError struct {
	Name          string
	ApiKey        string
	Model         string
	ErrChat       string
	ErrChatStream string
}

func TestClient_ProviderChat_errors(t *testing.T, getClient func(t *testing.T, apiKey, model string) genai.ProviderChat, lines []ProviderChatError) {
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

type ProviderModelError struct {
	Name   string
	ApiKey string
	Err    string
}

func TestClient_ProviderModel_errors(t *testing.T, getClient func(t *testing.T, apiKey string) genai.ProviderModel, lines []ProviderModelError) {
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

func testUsage(t *testing.T, u *genai.Usage, usageIsBroken bool, f genai.FinishReason) {
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
	if u.FinishReason != f {
		// TODO: llamacpp returns "eos" instead of "stop" when ending a stream. I need to find a way to improve
		// the test validation.
		if f != "" {
			t.Fatalf("expected FinishReason %q, got %q", f, u.FinishReason)
		}
	}
}

// ValidateSingleWordResponse validates that the response contains exactly one of the expected words.
func ValidateSingleWordResponse(t *testing.T, resp genai.ChatResult, want ...string) {
	got := resp.AsText()
	cleaned := strings.TrimRight(strings.TrimSpace(strings.ToLower(got)), ".!")
	if !slices.Contains(want, cleaned) {
		t.Fatalf("Expected %q, got %q", strings.Join(want, ", "), got)
	}
}
