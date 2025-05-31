// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package internaltest

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/maruel/genai"
	"golang.org/x/sync/errgroup"
)

// ProviderChatModalityFactory is what a Java developer would write.
//
// It permits selecting the most cost effective model for the requested modalities. For example a very cheap
// model would be used for text-in, text-out test but use a slightly more expensive one for image-in, text-out
// case.
type ProviderChatModalityFactory func(t *testing.T, model string) genai.ProviderChat

func TestScoreboard(t *testing.T, g ProviderChatModalityFactory, filter func(model genai.Model) bool) {
	var modelsSeen []string
	var ps genai.ProviderScoreboard
	{
		baseC := g(t, "")
		if u, ok := baseC.(genai.ProviderUnwrap); ok {
			baseC = u.Unwrap().(genai.ProviderChat)
		}
		ps = baseC.(genai.ProviderScoreboard)
	}
	scenarios := ps.Scoreboard().Scenarios
	if len(scenarios) == 0 {
		t.Fatal("expected at least one scenario")
	}
	for _, s := range scenarios {
		in := slices.Clone(s.In)
		slices.Sort(in)
		out := slices.Clone(s.Out)
		slices.Sort(out)
		// Only test the first model but acknowledge them all.
		modelsSeen = append(modelsSeen, s.Models...)
		t.Run(fmt.Sprintf("%s_%s_%s", in, out, s.Models[0]), func(t *testing.T) {
			// TODO: Investigate if this could happen?
			if s.Chat.Inline != s.ChatStream.Inline {
				t.Fatal("Inconsistent Inline")
			}
			if s.Chat.URL != s.ChatStream.URL {
				t.Fatal("Inconsistent URL")
			}
			data := []struct {
				stream bool
				name   string
				f      *genai.Functionality
			}{
				{false, "Chat", &s.Chat},
				{true, "ChatStream", &s.ChatStream},
			}
			for _, line := range data {
				t.Run(line.name, func(t *testing.T) {
					ran := false
					// Do not run text only output cases when output modality includes non-text. Most model fails in
					// this case.
					// TODO: Some models are fine with this, we need to improve genai.Functionality.
					if slices.Equal(out, genai.Modalities{genai.ModalityText}) {
						// Hack: audio input models tend to not work well with text-only input.
						if slices.Contains(in, genai.ModalityText) && !slices.Contains(in, genai.ModalityAudio) {
							t.Run("text", func(t *testing.T) {
								testTextFunctionalities(t, g, s.Models[0], line.f, line.stream)
								// Text only mode.
								if slices.Equal(in, genai.Modalities{genai.ModalityText}) {
									if line.f.URL {
										t.Fatalf("Do not set URL for a text-only model")
									}
								}
							})
							ran = true
						}
						if slices.Contains(in, genai.ModalityImage) {
							t.Run("vision", func(t *testing.T) {
								testVisionFunctionalities(t, g, s.Models[0], line.f, line.stream)
							})
							ran = true
						}
						if slices.Contains(in, genai.ModalityPDF) {
							t.Run("pdf", func(t *testing.T) {
								testPDFFunctionalities(t, g, s.Models[0], line.f, line.stream)
							})
							ran = true
						}
						if slices.Contains(in, genai.ModalityAudio) {
							t.Run("audio", func(t *testing.T) {
								testAudioFunctionalities(t, g, s.Models[0], line.f, line.stream)
							})
							ran = true
						}
						if slices.Contains(in, genai.ModalityVideo) {
							t.Run("video", func(t *testing.T) {
								testVideoFunctionalities(t, g, s.Models[0], line.f, line.stream)
							})
							ran = true
						}
					}
					if slices.Contains(out, genai.ModalityImage) {
						t.Run("imagegen", func(t *testing.T) {
							testImageGenFunctionalities(t, g, s.Models[0], line.f, line.stream)
						})
						ran = true
					}
					if slices.Contains(out, genai.ModalityAudio) {
						t.Run("audiogen", func(t *testing.T) {
							testAudioGenFunctionalities(t, g, s.Models[0], line.f, line.stream)
						})
						ran = true
					}
					if !ran {
						t.Fatal("implement test case for this modalities combination")
					}
				})
			}
		})
	}

	slices.SortFunc(modelsSeen, func(a, b string) int {
		return strings.Compare(strings.ToLower(a), strings.ToLower(b))
	})
	if l := findDuplicates(modelsSeen); len(l) != 0 {
		t.Errorf("Duplicate models found in scorecard: %s", strings.Join(l, ","))
	}

	t.Run("ListModels", func(t *testing.T) {
		baseC := g(t, "")
		if u, ok := baseC.(genai.ProviderUnwrap); ok {
			baseC = u.Unwrap().(genai.ProviderChat)
		}
		ls, ok := baseC.(genai.ProviderModel)
		if !ok {
			if filter != nil {
				t.Fatal("filter function is provided but provider doesn't implement ProviderModel")
			}
			t.Skip("provider doesn't implement ProviderModel")
		}
		knownModels, err := ls.ListModels(t.Context())
		if err != nil {
			t.Fatal(err)
		}
		ids := make([]string, 0, len(knownModels))
		for i := range knownModels {
			id := knownModels[i].GetID()
			if filter != nil && !filter(knownModels[i]) {
				continue
			}
			ids = append(ids, id)
		}
		slices.SortFunc(ids, func(a, b string) int {
			return strings.Compare(strings.ToLower(a), strings.ToLower(b))
		})
		if l := findMissing(modelsSeen, ids); len(l) != 0 {
			// Work around broken list models w.r.t. to the best models. For example the best model on togetherai as
			// of Mai 2025 is "meta-llama/Llama-3.3-70B-Instruct-Turbo" but it is not reported by the endpoint.
			// Since we are testing the first model for each scenarios, do not alert on these.
			for i := 0; i < len(l); i++ {
				for s := range scenarios {
					if l[i] == scenarios[s].Models[0] {
						copy(l[i:], l[i+1:])
						l = l[:len(l)-1]
						i--
						break
					}
				}
			}
			if len(l) != 0 {
				t.Errorf("Found unexpected models:\n%s", strings.Join(l, "\n"))
			}
		}
		// Some providers (like Huggingface) have a ton of models.
		if l := findMissing(ids, modelsSeen); len(l) != 0 && len(l) < 100 {
			t.Logf("Found new models:\n%s", strings.Join(l, "\n"))
		}
	})
}

func testTextFunctionalities(t *testing.T, g ProviderChatModalityFactory, model string, f *genai.Functionality, stream bool) {
	defaultFR := genai.FinishedStop
	if !f.ReportFinishReason {
		defaultFR = ""
	}
	t.Run("Simple", func(t *testing.T) {
		msgs := genai.Messages{genai.NewTextMessage(genai.User, "Say hello. Use only one word.")}
		resp, err := run(t, g(t, model), msgs, nil, stream)
		if !basicCheck(t, err, true) {
			return
		}
		testUsage(t, &resp.Usage, !f.ReportTokenUsage, defaultFR)
		hasThinking := false
		for i := range resp.Contents {
			if resp.Contents[i].Thinking != "" {
				hasThinking = true
				break
			}
		}
		if f.Thinking {
			if !hasThinking {
				t.Errorf("expected thinking, didn't find any")
			}
		} else {
			if hasThinking {
				t.Errorf("unexpected thinking: %#v", resp.Message)
			}
		}
		// Some models are really bad at instruction following.
		ValidateSingleWordResponse(t, resp, "hello", "hey", "hi")
	})

	t.Run("MaxTokens", func(t *testing.T) {
		msgs := genai.Messages{genai.NewTextMessage(genai.User, "Tell a joke in 10 words")}
		// Give enough token so the <think> token can be emitted plus another word. MaxTokens:2 could cause
		// problems and it's not a value that is expected to be used in practice for this use case.
		resp, err := run(t, g(t, model), msgs, &genai.ChatOptions{MaxTokens: 3}, stream)
		// MaxTokens can fail in two ways:
		// - Chat() or ChatRequest() return an irrecoverable error.
		// - The length is not enforced.
		if !basicCheckAcceptUnexpectedSuccess(t, err, true) {
			return
		}
		fr := genai.FinishedLength
		if !f.MaxTokens {
			fr = genai.FinishedStop
		}
		if !f.ReportFinishReason {
			fr = ""
		}
		testUsage(t, &resp.Usage, !f.ReportTokenUsage, fr)
		if len(resp.AsText()) > 15 {
			if f.MaxTokens {
				// Deepseek counts "\"Parallel lines" as 3 tokens (!)
				t.Fatalf("Expected less than 15 letters, got %q", resp.AsText())
			}
		} else if !f.MaxTokens {
			t.Fatal("unexpected short answer")
		}
	})

	t.Run("Stop", func(t *testing.T) {
		msgs := genai.Messages{genai.NewTextMessage(genai.User, "Talk about Canada in 10 words. Start with: Canada is")}
		resp, err := run(t, g(t, model), msgs, &genai.ChatOptions{Stop: []string{"is"}}, stream)
		// Stop can fail in two ways:
		// - Chat() or ChatRequest() return an irrecoverable error.
		// - The Stop words to not stop generation.
		if !basicCheckAcceptUnexpectedSuccess(t, err, f.StopSequence) {
			return
		}
		fr := genai.FinishedStopSequence
		if !f.StopSequence {
			fr = genai.FinishedStop
		}
		if !f.ReportFinishReason {
			fr = ""
		}
		testUsage(t, &resp.Usage, !f.ReportTokenUsage, fr)
		// Special case the thinking token because explicit CoT models like Qwen3 will always restart the user's
		// question.
		if s := resp.AsText(); len(s) > 12 && !strings.HasPrefix(s, "<think") {
			// This is very unfortunate: Huggingface serves a version of the model that never prefixes with <think>.
			if !strings.Contains(strings.ToLower(model), "qwq") && f.StopSequence {
				t.Fatalf("Expected less than 12 letters, got %q", resp.AsText())
			}
		} else if !f.StopSequence {
			t.Fatal("unexpected short answer")
		}
	})

	t.Run("JSON", func(t *testing.T) {
		msgs := genai.Messages{
			genai.NewTextMessage(genai.User, `Is a banana a fruit? Do not include an explanation. Reply ONLY as JSON according to the provided schema: {"is_fruit": bool}.`),
		}
		resp, err := run(t, g(t, model), msgs, &genai.ChatOptions{ReplyAsJSON: true}, stream)
		if !basicCheckAcceptUnexpectedSuccess(t, err, f.JSON) {
			return
		}
		testUsage(t, &resp.Usage, !f.ReportTokenUsage, defaultFR)
		got := map[string]any{}
		if err := resp.Decode(&got); err != nil {
			if !f.JSON {
				return
			}
			// Gemini returns a list of map. Tolerate that too.
			got2 := []map[string]any{}
			if err := resp.Decode(&got2); err != nil {
				t.Fatal(err)
			}
			if len(got2) != 1 {
				t.Fatal(got2)
			}
			got = got2[0]
		} else if !f.JSON {
			t.Fatal("unexpected success")
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
	})

	t.Run("JSONSchema", func(t *testing.T) {
		// TODO: Test optional vs required, enum, bool, int, etc.
		var got struct {
			IsFruit bool `json:"is_fruit" jsonschema_description:"True if the answer is that it is a fruit, false otherwise"`
		}
		msgs := genai.Messages{genai.NewTextMessage(genai.User, "Is a banana a fruit? Reply as JSON according to the provided schema.")}
		resp, err := run(t, g(t, model), msgs, &genai.ChatOptions{DecodeAs: &got}, stream)
		if !basicCheck(t, err, f.JSONSchema) {
			return
		}
		testUsage(t, &resp.Usage, !f.ReportTokenUsage, defaultFR)
		if err := resp.Decode(&got); err != nil {
			t.Fatal(err)
		}
		if !got.IsFruit {
			t.Fatal(got.IsFruit)
		}
	})

	t.Run("Tools", func(t *testing.T) {
		if f.UnbiasedTool && f.Tools != genai.True {
			t.Fatal("UnbiasedTool required Tools to be True")
		}
		msgs := genai.Messages{
			genai.NewTextMessage(genai.User, "Use the square_root tool to calculate the square root of 132413 and reply with only the result. Do not give an explanation."),
		}
		type got struct {
			Number json.Number `json:"number"`
		}
		opts := genai.ChatOptions{
			SystemPrompt: "You are an helpful assistant that is very succinct. You only reply to the user's request with no additional information. Use the tools at your disposal and return their result as-is.",
			Tools: []genai.ToolDef{
				{
					Name:        "square_root",
					Description: "Calculates and return the square root of a number",
					Callback: func(ctx context.Context, g *got) (string, error) {
						i, err := g.Number.Int64()
						if err != nil {
							err = fmt.Errorf("wanted 132413 as an int, got %q: %w", g.Number, err)
							t.Error(err)
							return "", err
						}
						if i != 132413 {
							err = fmt.Errorf("wanted 132413 as an int, got %s", g.Number)
							t.Error(err)
							return "", err
						}
						return fmt.Sprintf("%.2f", math.Sqrt(float64(i))), nil
					},
				},
			},
			// For this test, we want to make sure the tool is called.
			ToolCallRequest: genai.ToolCallRequired,
		}
		c := g(t, model)
		fr := genai.FinishedToolCalls
		if !f.ReportFinishReason {
			fr = ""
		}
		resp, err := run(t, c, msgs, &opts, stream)
		if !basicCheckAcceptUnexpectedSuccess(t, err, f.Tools == genai.True) {
			return
		}
		if f.Tools != genai.True {
			if resp.FinishReason == genai.FinishedStop || resp.FinishReason == genai.FinishedLength {
				return
			}
			if f.Tools == genai.False {
				t.Fatalf("unexpected success: %s", resp.FinishReason)
			}
		}
		testUsage(t, &resp.Usage, !f.ReportTokenUsage, fr)
		want := "square_root"
		if len(resp.ToolCalls) == 0 || resp.ToolCalls[0].Name != want {
			if f.Tools == genai.Flaky {
				return
			}
			t.Fatalf("Expected tool call to %s, got: %v", want, resp.ToolCalls)
		}
		// Don't forget to add the tool call request first before the reply.
		msgs = append(msgs, resp.Message)
		msg, err := resp.DoToolCalls(t.Context(), opts.Tools)
		if err != nil {
			t.Fatal(err)
		}
		if msg.IsZero() {
			t.Fatal("unexpected zero message")
		}
		// Don't forget to add the tool call request first before the reply.
		msgs = append(msgs, msg)
		// Important!
		opts.ToolCallRequest = genai.ToolCallNone
		if f.ReportFinishReason {
			fr = genai.FinishedStop
		}
		resp, err = run(t, c, msgs, &opts, stream)
		if !basicCheck(t, err, f.Tools != genai.False) {
			return
		}
		testUsage(t, &resp.Usage, !f.ReportTokenUsage, fr)
		s := resp.AsText()
		want = "363.89"
		if got := strings.TrimRight(strings.TrimSpace(strings.ToLower(s)), ".!"); want != got {
			// I don't know why but llama-4-scout loves to round the number!
			if got != "364" {
				if f.Tools == genai.Flaky {
					return
				}
				t.Fatalf("Expected %q, got %q", want, s)
			}
		}
	})
}

func testVisionFunctionalities(t *testing.T, g ProviderChatModalityFactory, model string, f *genai.Functionality, stream bool) {
	defaultFR := genai.FinishedStop
	if !f.ReportFinishReason {
		defaultFR = ""
	}
	filename := "banana.jpg"
	prompt := "Is it a banana? Reply with only one word."
	t.Run("Inline", func(t *testing.T) {
		msgs := genai.Messages{
			{
				Role: genai.User,
				Contents: []genai.Content{
					{Text: prompt},
					{Filename: filename, Document: bytes.NewReader(bananaJpg)},
				},
			},
		}
		resp, err := run(t, g(t, model), msgs, nil, stream)
		if !basicCheck(t, err, f.Inline) {
			return
		}
		testUsage(t, &resp.Usage, !f.ReportTokenUsage, defaultFR)
		ValidateSingleWordResponse(t, resp, "yes")
	})

	t.Run("URL", func(t *testing.T) {
		msgs := genai.Messages{
			{
				Role: genai.User,
				Contents: []genai.Content{
					{Text: prompt},
					{URL: "https://raw.githubusercontent.com/maruel/genai/refs/heads/main/internal/internaltest/testdata/" + filename},
				},
			},
		}
		resp, err := run(t, g(t, model), msgs, nil, stream)
		if !basicCheck(t, err, f.URL) {
			return
		}
		testUsage(t, &resp.Usage, !f.ReportTokenUsage, defaultFR)
		ValidateSingleWordResponse(t, resp, "yes")
	})
}

func testAudioFunctionalities(t *testing.T, g ProviderChatModalityFactory, model string, f *genai.Functionality, stream bool) {
	defaultFR := genai.FinishedStop
	if !f.ReportFinishReason {
		defaultFR = ""
	}
	// Gemini supports Opus too.
	filename := "mystery_word.mp3"
	prompt := "What is the word said? Reply with only the word."
	t.Run("Inline", func(t *testing.T) {
		// Path with the assumption it's run from "//<provider>/".
		ff, err := os.Open(filepath.Join("..", "internal", "internaltest", "testdata", filename))
		if err != nil {
			t.Fatal(err)
		}
		defer ff.Close()
		msgs := genai.Messages{{Role: genai.User, Contents: []genai.Content{{Text: prompt}, {Document: ff}}}}
		resp, err := run(t, g(t, model), msgs, nil, stream)
		if !basicCheck(t, err, f.Inline) {
			return
		}
		testUsage(t, &resp.Usage, !f.ReportTokenUsage, defaultFR)
		ValidateSingleWordResponse(t, resp, "orange")
	})

	t.Run("URL", func(t *testing.T) {
		msgs := genai.Messages{
			{
				Role: genai.User,
				Contents: []genai.Content{
					{Text: prompt},
					{URL: "https://raw.githubusercontent.com/maruel/genai/refs/heads/main/internal/internaltest/testdata/" + filename},
				},
			},
		}
		resp, err := run(t, g(t, model), msgs, nil, stream)
		if !basicCheck(t, err, f.URL) {
			return
		}
		testUsage(t, &resp.Usage, !f.ReportTokenUsage, defaultFR)
		ValidateSingleWordResponse(t, resp, "orange")
	})
}

func testVideoFunctionalities(t *testing.T, g ProviderChatModalityFactory, model string, f *genai.Functionality, stream bool) {
	defaultFR := genai.FinishedStop
	if !f.ReportFinishReason {
		defaultFR = ""
	}
	filename := "animation.mp4"
	prompt := "What is the word? Reply with only the word."
	t.Run("Inline", func(t *testing.T) {
		// Path with the assumption it's run from "//<provider>/".
		ff, err := os.Open(filepath.Join("..", "internal", "internaltest", "testdata", filename))
		if err != nil {
			t.Fatal(err)
		}
		defer ff.Close()
		msgs := genai.Messages{{Role: genai.User, Contents: []genai.Content{{Text: prompt}, {Document: ff}}}}
		resp, err := run(t, g(t, model), msgs, nil, stream)
		if !basicCheck(t, err, f.Inline) {
			return
		}
		testUsage(t, &resp.Usage, !f.ReportTokenUsage, defaultFR)
		ValidateSingleWordResponse(t, resp, "banana")
	})

	t.Run("URL", func(t *testing.T) {
		msgs := genai.Messages{
			{
				Role: genai.User,
				Contents: []genai.Content{
					{Text: prompt},
					{URL: "https://raw.githubusercontent.com/maruel/genai/refs/heads/main/internal/internaltest/testdata/" + filename},
				},
			},
		}
		resp, err := run(t, g(t, model), msgs, nil, stream)
		if !basicCheck(t, err, f.URL) {
			return
		}
		testUsage(t, &resp.Usage, !f.ReportTokenUsage, defaultFR)
		ValidateSingleWordResponse(t, resp, "banana")
	})
}

func testPDFFunctionalities(t *testing.T, g ProviderChatModalityFactory, model string, f *genai.Functionality, stream bool) {
	defaultFR := genai.FinishedStop
	if !f.ReportFinishReason {
		defaultFR = ""
	}
	filename := "hidden_word.pdf"
	prompt := "What is the word? Reply with only the word."
	t.Run("Inline", func(t *testing.T) {
		// Path with the assumption it's run from "//<provider>/".
		ff, err := os.Open(filepath.Join("..", "internal", "internaltest", "testdata", filename))
		if err != nil {
			t.Fatal(err)
		}
		defer ff.Close()
		msgs := genai.Messages{{Role: genai.User, Contents: []genai.Content{{Text: prompt}, {Document: ff}}}}
		resp, err := run(t, g(t, model), msgs, nil, stream)
		if !basicCheck(t, err, f.Inline) {
			return
		}
		testUsage(t, &resp.Usage, !f.ReportTokenUsage, defaultFR)
		ValidateSingleWordResponse(t, resp, "orange")
	})

	t.Run("URL", func(t *testing.T) {
		msgs := genai.Messages{
			{
				Role: genai.User,
				Contents: []genai.Content{
					{Text: prompt},
					{URL: "https://raw.githubusercontent.com/maruel/genai/refs/heads/main/internal/internaltest/testdata/" + filename},
				},
			},
		}
		resp, err := run(t, g(t, model), msgs, nil, stream)
		if !basicCheck(t, err, f.URL) {
			return
		}
		testUsage(t, &resp.Usage, !f.ReportTokenUsage, defaultFR)
		ValidateSingleWordResponse(t, resp, "orange")
	})
}

func testImageGenFunctionalities(t *testing.T, g ProviderChatModalityFactory, model string, f *genai.Functionality, stream bool) {
	prompt := `A doodle animation on a white background of Cartoonish shiba inu with brown fur and a white belly, happily eating a pink ice-cream cone, subtle tail wag. Subtle motion but nothing else moves.`
	const style = `Simple, vibrant, varied-colored doodle/hand-drawn sketch`
	contents := `Generate one square, white-background doodle with smooth, vibrantly colored image depicting ` + prompt + `.

*Mandatory Requirements (Compacted):**

**Style:** ` + style + `.
**Background:** Plain solid white (no background colors/elements). Absolutely no black background.
**Content & Motion:** Clearly depict **` + prompt + `** action with colored, moving subject (no static images). If there's an action specified, it should be the main difference between frames.
**Format:** Square image (1:1 aspect ratio).
**Cropping:** Absolutely no black bars/letterboxing; colorful doodle fully visible against white.
**Output:** Actual image file for a smooth, colorful doodle-style image on a white background.`
	defaultFR := genai.FinishedStop
	if !f.ReportFinishReason {
		defaultFR = ""
	}
	msgs := genai.Messages{genai.NewTextMessage(genai.User, contents)}
	resp, err := run(t, g(t, model), msgs, nil, stream)
	if !basicCheck(t, err, f.Inline) {
		return
	}
	testUsage(t, &resp.Usage, !f.ReportTokenUsage, defaultFR)
	if len(resp.Contents) == 0 {
		t.Fatal("expected content")
	}
	if len(resp.Contents) != 1 {
		t.Fatalf("expected one content, got %d", len(resp.Contents))
	}
	if resp.Contents[0].Filename != "content.png" {
		t.Fatalf("expected one image, got %#v", resp.Contents[0])
	}
	// It can have text, images or both.
}

func testAudioGenFunctionalities(t *testing.T, g ProviderChatModalityFactory, model string, f *genai.Functionality, stream bool) {
	defaultFR := genai.FinishedStop
	if !f.ReportFinishReason {
		defaultFR = ""
	}
	msgs := genai.Messages{genai.NewTextMessage(genai.User, "Say hi. Just say this word, nothing else.")}
	resp, err := run(t, g(t, model), msgs, nil, stream)
	if !basicCheck(t, err, f.Inline) {
		return
	}
	testUsage(t, &resp.Usage, !f.ReportTokenUsage, defaultFR)
	if len(resp.Contents) == 0 {
		t.Fatal("expected content")
	}
	if len(resp.Contents) != 1 {
		t.Fatalf("expected one content, got %d", len(resp.Contents))
	}
	if resp.Contents[0].Filename != "sound.wav" {
		t.Fatalf("expected one image, got %#v", resp.Contents[0])
	}
}

func run(t *testing.T, c genai.ProviderChat, msgs genai.Messages, opts genai.Validatable, stream bool) (genai.ChatResult, error) {
	ctx := t.Context()
	if !stream {
		resp, err := c.Chat(ctx, msgs, opts)
		// Uncomment to diagnose issues:
		// t.Logf("Response: %v", resp.Message)
		return resp, err
	}
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
				// Uncomment to diagnose issues:
				// t.Logf("Packet: %s", pkt.GoString())
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
	// Uncomment to diagnose issues:
	// t.Logf("Response: %v", resp.Message)
	if diff := cmp.Diff(&resp.Message, &accumulated); diff != "" {
		t.Errorf("(-result), (+accumulated):\n%s", diff)
	}
	return resp, err
}

// basicCheck returns true if the test should continue.
func basicCheck(t *testing.T, err error, expectedSuccess bool) bool {
	if uce, ok := err.(*genai.UnsupportedContinuableError); ok {
		t.Log(uce)
		err = nil
	}
	if err != nil {
		if !expectedSuccess {
			// Skip the remainder.
			return false
		}
		t.Helper()
		t.Fatalf("unexpected failure: %s", err.Error())
	}
	if !expectedSuccess {
		t.Helper()
		t.Error("unexpected success")
	}
	return true
}

// basicCheckAcceptUnexpectedSuccess returns true if the test should continue.
func basicCheckAcceptUnexpectedSuccess(t *testing.T, err error, expectedSuccess bool) bool {
	if uce, ok := err.(*genai.UnsupportedContinuableError); ok {
		t.Log(uce)
		err = nil
	}
	if err != nil {
		if !expectedSuccess {
			// Skip the remainder.
			return false
		}
		t.Helper()
		t.Fatalf("unexpected failure: %s", err.Error())
	}
	return true
}

func findDuplicates(s []string) []string {
	var out []string
	for i := range len(s) - 1 {
		if s[i] == s[i+1] {
			if len(out) == 0 || out[len(out)-1] != s[i] {
				out = append(out, s[i])
			}
		}
	}
	return out
}

func findMissing(want, got []string) []string {
	m := map[string]struct{}{}
	for _, item := range got {
		m[item] = struct{}{}
	}
	var result []string
	for _, item := range want {
		if _, ok := m[item]; !ok {
			result = append(result, item)
		}
	}
	return result
}
