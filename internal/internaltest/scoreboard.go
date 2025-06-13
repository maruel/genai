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
	"net/url"
	"os"
	"path/filepath"
	"slices"
	"sort"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/maruel/genai"
	"golang.org/x/sync/errgroup"
)

// ProviderGenModalityFactory is what a Java developer would write.
//
// It permits selecting the most cost effective model for the requested modalities. For example a very cheap
// model would be used for text-in, text-out test but use a slightly more expensive one for image-in, text-out
// case.
type ProviderGenModalityFactory func(t *testing.T, model string) genai.ProviderGen

func TestScoreboard(t *testing.T, g ProviderGenModalityFactory, filter func(model genai.Model) bool) {
	var modelsSeen []string
	var ps genai.ProviderScoreboard
	{
		baseC := g(t, "")
		if u, ok := baseC.(genai.ProviderUnwrap); ok {
			baseC = u.Unwrap().(genai.ProviderGen)
		}
		ps = baseC.(genai.ProviderScoreboard)
	}
	scenarios := ps.Scoreboard().Scenarios
	if len(scenarios) == 0 {
		t.Fatal("expected at least one scenario")
	}
	for _, s := range scenarios {
		// Extract and sort modalities from capability maps
		var in []string
		for modality := range s.In {
			in = append(in, string(modality))
		}
		slices.Sort(in)
		var out []string
		for modality := range s.Out {
			out = append(out, string(modality))
		}
		slices.Sort(out)
		// Only test the first model but acknowledge them all.
		modelsSeen = append(modelsSeen, s.Models...)
		t.Run(fmt.Sprintf("%s_%s_%s", strings.Join(in, ","), strings.Join(out, ","), s.Models[0]), func(t *testing.T) {
			testModalities(t, s.In, s.Out)
			if s.GenSync != nil {
				t.Run("GenSync", func(t *testing.T) {
					testGen(t, g, s.Models[0], s.In, s.Out, s.GenSync, false)
				})
			}
			if s.GenStream != nil {
				t.Run("GenStream", func(t *testing.T) {
					testGen(t, g, s.Models[0], s.In, s.Out, s.GenStream, true)
				})
			}
			if s.GenDoc != nil {
				t.Run("GenDoc", func(t *testing.T) {
					testGenDoc(t, g, s.Models[0], s.In, s.Out, s.GenDoc)
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
			baseC = u.Unwrap().(genai.ProviderGen)
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

func testModalities(t *testing.T, in, out map[genai.Modality]genai.ModalCapability) {
	if c, ok := in[genai.ModalityText]; ok {
		if !c.Inline {
			t.Fatalf("input text modality requires inline")
		}
		if c.URL {
			t.Fatalf("input text modality doesn't support URL")
		}
	}
	if c, ok := out[genai.ModalityText]; ok {
		if !c.Inline {
			t.Fatalf("output text modality requires inline")
		}
		if c.URL {
			t.Fatalf("output text modality doesn't support URL")
		}
	}
	for modality, c := range in {
		if !c.Inline && !c.URL {
			t.Fatalf("input modality %s requires at least one of inline or URL", modality)
		}
	}
	for modality, c := range out {
		if !c.Inline && !c.URL {
			t.Fatalf("output modality %s requires at least one of inline or URL", modality)
		}
	}
}

func testGen(t *testing.T, g ProviderGenModalityFactory, model string, in, out map[genai.Modality]genai.ModalCapability, f *genai.FunctionalityText, stream bool) {
	ran := false
	// Text only output.
	// Do not run text only output cases when output modality includes non-text. Most model fails in
	// this case.
	// TODO: Some models are fine with this, we need to improve genai.Functionality.
	if len(out) == 1 && out[genai.ModalityText].Inline {
		// Hack: audio input models tend to not work well with text-only input.
		_, hasText := in[genai.ModalityText]
		_, hasAudio := in[genai.ModalityAudio]
		if hasText && !hasAudio {
			t.Run("text", func(t *testing.T) {
				testTextFunctionalities(t, g, model, f, stream)
			})
			ran = true
		}
		if _, hasImage := in[genai.ModalityImage]; hasImage {
			t.Run("vision", func(t *testing.T) {
				testVisionFunctionalities(t, g, model, in, out, f, stream)
			})
			ran = true
		}
		if _, hasPDF := in[genai.ModalityPDF]; hasPDF {
			t.Run("pdf", func(t *testing.T) {
				testPDFFunctionalities(t, g, model, in, out, f, stream)
			})
			ran = true
		}
		if hasAudio {
			t.Run("audio", func(t *testing.T) {
				testAudioSTTFunctionalities(t, g, model, in, out, f, stream)
			})
			ran = true
		}
		if _, hasVideo := in[genai.ModalityVideo]; hasVideo {
			t.Run("video", func(t *testing.T) {
				testVideoFunctionalities(t, g, model, in, out, f, stream)
			})
			ran = true
		}
	}

	// Multi-modal output.
	if _, hasImage := out[genai.ModalityImage]; hasImage {
		// Text+Image output, e.g. Gemini.
		t.Run("imagegen", func(t *testing.T) {
			testChatImageGenFunctionalities(t, g, model, in, out, f, stream)
		})
		ran = true
	}
	if _, hasAudio := out[genai.ModalityAudio]; hasAudio {
		t.Run("audiogen", func(t *testing.T) {
			t.Skip("TODO: it currently assumes GenDoc.")
			// testAudioGenFunctionalities(t, g, model, in, out, f, stream)
		})
		ran = true
	}
	if !ran {
		t.Fatal("implement test case for this modalities combination")
	}
}

func testTextFunctionalities(t *testing.T, g ProviderGenModalityFactory, model string, f *genai.FunctionalityText, stream bool) {
	defaultFR := genai.FinishedStop
	if f.BrokenFinishReason {
		defaultFR = ""
	}
	t.Run("Simple", func(t *testing.T) {
		msgs := genai.Messages{genai.NewTextMessage(genai.User, "Say hello. Use only one word.")}
		opts := genai.OptionsText{Seed: 42}
		resp, err := runGenText(t, g(t, model), msgs, &opts, stream)
		if !basicCheck(t, err, true) {
			return
		}
		if uce, ok := err.(*genai.UnsupportedContinuableError); ok {
			if f.Seed && slices.Contains(uce.Unsupported, "Seed") {
				t.Fatal("unexpected error: scoreboard reports seed is supported but it's not")
			}
		} else if !f.Seed {
			t.Fatal("unexpected success: scoreboard reports seed is not supported but it is")
		}
		testUsage(t, &resp.Usage, f.BrokenTokenUsage, defaultFR)
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
		// Some models are really bad at instruction following, and perplexity is not even trying.
		if !strings.Contains(model, "sonar") {
			ValidateWordResponse(t, resp, "hello", "hey", "hi")
		}
	})

	t.Run("MaxTokens", func(t *testing.T) {
		msgs := genai.Messages{genai.NewTextMessage(genai.User, "Tell a joke in 10 words")}
		// Give enough token so the <think> token can be emitted plus another word. MaxTokens:2 could cause
		// problems and it's not a value that is expected to be used in practice for this use case.
		resp, err := runGenText(t, g(t, model), msgs, &genai.OptionsText{MaxTokens: 3}, stream)
		// MaxTokens can fail:
		// - GenSync() or GenStream() return an irrecoverable error.
		// - The length is not enforced.
		// - Pollations processes the thinking but only when MaxTokens is set.
		if !basicCheckAcceptUnexpectedSuccess(t, err, !f.NoMaxTokens) {
			return
		}
		fr := genai.FinishedLength
		if f.NoMaxTokens {
			fr = genai.FinishedStop
		}
		if f.BrokenFinishReason {
			fr = ""
		}
		testUsage(t, &resp.Usage, f.BrokenTokenUsage, fr)
		// Deepseek counts "\"Parallel lines" as 3 tokens (!)
		// Llama-4-Maverick counts ""Why couldn't the" as 3 tokens.
		if len(resp.AsText()) > 16 {
			if !f.NoMaxTokens {
				t.Fatalf("Expected less than 15 letters, got %q", resp.AsText())
			}
		} else if f.NoMaxTokens {
			t.Fatal("unexpected short answer")
		}
	})

	t.Run("Stop", func(t *testing.T) {
		msgs := genai.Messages{genai.NewTextMessage(genai.User, "Talk about Canada in 10 words. Start with: Canada is")}
		resp, err := runGenText(t, g(t, model), msgs, &genai.OptionsText{Stop: []string{"is"}}, stream)
		// Stop can fail in two ways:
		// - GenSync() or GenStream() return an irrecoverable error.
		// - The Stop words to not stop generation.
		if !basicCheckAcceptUnexpectedSuccess(t, err, !f.NoStopSequence) {
			return
		}
		fr := genai.FinishedStopSequence
		if f.NoStopSequence {
			fr = genai.FinishedStop
		}
		if f.BrokenFinishReason {
			fr = ""
		}
		testUsage(t, &resp.Usage, f.BrokenTokenUsage, fr)
		// Special case the thinking token because explicit CoT models like Qwen3 will always restart the user's
		// question.
		if s := resp.AsText(); len(s) > 12 && !strings.HasPrefix(s, "<think") {
			// This is very unfortunate: Huggingface serves a version of the model that never prefixes with <think>.
			if !strings.Contains(strings.ToLower(model), "qwq") && !f.NoStopSequence {
				t.Fatalf("Expected less than 12 letters, got %q", resp.AsText())
			}
		} else if f.NoStopSequence {
			t.Fatal("unexpected short answer")
		}
	})

	t.Run("JSON", func(t *testing.T) {
		msgs := genai.Messages{
			genai.NewTextMessage(genai.User, `Is a banana a fruit? Do not include an explanation. Reply ONLY as JSON according to the provided schema: {"is_fruit": bool}.`),
		}
		resp, err := runGenText(t, g(t, model), msgs, &genai.OptionsText{ReplyAsJSON: true}, stream)
		if !basicCheckAcceptUnexpectedSuccess(t, err, f.JSON) {
			return
		}
		testUsage(t, &resp.Usage, f.BrokenTokenUsage, defaultFR)
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
		resp, err := runGenText(t, g(t, model), msgs, &genai.OptionsText{DecodeAs: &got}, stream)
		if !basicCheckAcceptUnexpectedSuccess(t, err, f.JSONSchema) {
			return
		}
		testUsage(t, &resp.Usage, f.BrokenTokenUsage, defaultFR)
		if err := resp.Decode(&got); err != nil {
			if !f.JSONSchema {
				return
			}
			t.Fatal(err)
		} else if !f.JSONSchema {
			t.Fatal("unexpected success")
		}
		if !got.IsFruit {
			t.Fatal(got.IsFruit)
		}
	})

	t.Run("Tools", func(t *testing.T) {
		msgs := genai.Messages{
			genai.NewTextMessage(genai.User, "Use the square_root tool to calculate the square root of 132413 and reply with only the result. Do not give an explanation."),
		}
		type got struct {
			Number json.Number `json:"number"`
		}
		opts := genai.OptionsText{
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
		if f.BrokenFinishReason {
			fr = ""
		}
		resp, err := runGenText(t, c, msgs, &opts, stream)
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
		testUsage(t, &resp.Usage, f.BrokenTokenUsage, fr)
		want := opts.Tools[0].Name
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
		// Important! We do not any any follow up tool call.
		opts.ToolCallRequest = genai.ToolCallNone
		if !f.BrokenFinishReason {
			fr = genai.FinishedStop
		}
		resp, err = runGenText(t, c, msgs, &opts, stream)
		if !basicCheck(t, err, f.Tools != genai.False) {
			return
		}
		testUsage(t, &resp.Usage, f.BrokenTokenUsage, fr)
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

	t.Run("ToolsBiased", func(t *testing.T) {
		if f.BiasedTool != genai.False && f.Tools == genai.False {
			t.Fatal("BiasedTool required Tools to be True or Flaky")
		}
		if f.IndecisiveTool != genai.False && f.Tools == genai.False {
			t.Fatal("IndecisiveTool required Tools to be True or Flaky")
		}
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
				// This test case has the same challenge as "Tools" above.
				msgs := genai.Messages{
					genai.NewTextMessage(genai.User, fmt.Sprintf("I wonder if %s is a better country than %s? Call the tool best_country to tell me which country is the best one.", line.country1, line.country2)),
				}
				opts := genai.OptionsText{
					Tools: []genai.ToolDef{
						{
							Name:        "best_country",
							Description: "A tool to determine the best country",
							Callback:    line.callback,
						},
					},
					ToolCallRequest: genai.ToolCallRequired,
				}
				c := g(t, model)
				fr := genai.FinishedToolCalls
				if f.BrokenFinishReason {
					fr = ""
				}
				resp, err := runGenText(t, c, msgs, &opts, stream)
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
				testUsage(t, &resp.Usage, f.BrokenTokenUsage, fr)
				if len(resp.ToolCalls) == 0 {
					if f.Tools != genai.True {
						return
					}
				} else {
					if f.Tools == genai.False {
						t.Fatal("unexpected tool call when not supported")
					}
				}
				want := opts.Tools[0].Name
				for i := range resp.ToolCalls {
					if resp.ToolCalls[i].Name != want {
						t.Fatalf("Expected tool call to %s, got: %v", want, resp.ToolCalls)
					}
				}
				switch len(resp.ToolCalls) {
				case 0:
					t.Fatal("expected at least one tool call")
				case 1:
					if f.IndecisiveTool == genai.True {
						t.Fatal("scenario marked as indecisive but it's not")
					}
					res, err := resp.ToolCalls[0].Call(t.Context(), opts.Tools)
					if err != nil {
						t.Fatal(err)
					}
					switch f.BiasedTool {
					case genai.True:
						// It must be the first.
						if res != line.countrySelected {
							t.Fatalf("Expected bias towards the first proposed item %q, got %q", line.countrySelected, res)
						}
					case genai.Flaky:
						// A good example is "gemini-2.0-flash-lite", it's slightly biased but not consistently.
					case genai.False:
						// It must be the other.
						if res == line.countrySelected {
							t.Fatalf("Expected not %q, got %q", line.countrySelected, res)
						}
					default:
					}
				case 2:
					if f.IndecisiveTool == genai.False {
						t.Fatal("got indecisive result but it's not marked as indecisive")
					}
					var countries []string
					for _, tc := range resp.ToolCalls {
						res, err := tc.Call(t.Context(), opts.Tools)
						if err != nil {
							t.Fatal(err)
						}
						countries = append(countries, res)
					}
					sort.Strings(countries)
					if !slices.Equal(countries, []string{"Canada", "USA"}) {
						t.Fatalf("while indecisive, unexpected countries: %s", countries)
					}
				default:
					t.Fatalf("had too many tool calls: %#v", resp.ToolCalls)
				}
			})
		}
	})

	t.Run("Citations", func(t *testing.T) {
		msgs := genai.Messages{
			genai.Message{
				Role: genai.User,
				Contents: []genai.Content{
					{
						Document: strings.NewReader("The capital of Quackiland is Quack. The Big Canard Statue is located in Quack."),
						Filename: "capital_info.txt",
					},
					{
						Text: "What is the capital of Quackiland?",
					},
				},
			},
		}
		want := "Quackiland"
		// Do a one off for perplexity since it can only search on live web results.
		isPerplexity := strings.Contains(model, "sonar")
		if isPerplexity {
			msgs = genai.Messages{genai.NewTextMessage(genai.User, "What is the capital of France?")}
			want = "Paris"
		}
		resp, err := runGenText(t, g(t, model), msgs, nil, stream)
		if !basicCheckAcceptUnexpectedSuccess(t, err, f.Citations) {
			return
		}
		testUsage(t, &resp.Usage, f.BrokenTokenUsage, defaultFR)
		if s := resp.AsText(); !strings.Contains(s, want) && f.Citations {
			// Small models may not return the string as-is.
			t.Errorf("expected %s, got: %q", want, s)
		}
		foundCitations := false
		for _, content := range resp.Contents {
			if len(content.Citations) > 0 {
				foundCitations = true
				t.Logf("Found %d citations in content", len(content.Citations))
				for i, citation := range content.Citations {
					t.Logf("Citation %d: text=%q, type=%q, start=%d, end=%d", i, citation.Text, citation.Type, citation.StartIndex, citation.EndIndex)
					if len(citation.Sources) > 0 {
						t.Logf("  Sources: %+v", citation.Sources)
					}
				}
			}
		}
		if !foundCitations {
			if f.Citations {
				t.Fatalf("expected citations in response, but found none")
			}
		} else {
			if !f.Citations {
				t.Fatalf("unexpected citations in response")
			}
		}
	})
}

func testVisionFunctionalities(t *testing.T, g ProviderGenModalityFactory, model string, in, out map[genai.Modality]genai.ModalCapability, f *genai.FunctionalityText, stream bool) {
	defaultFR := genai.FinishedStop
	if f.BrokenFinishReason {
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
		resp, err := runGenText(t, g(t, model), msgs, nil, stream)
		if !basicCheck(t, err, in[genai.ModalityImage].Inline) {
			return
		}
		testUsage(t, &resp.Usage, f.BrokenTokenUsage, defaultFR)
		ValidateWordResponse(t, resp, "yes")
	})

	t.Run("URL", func(t *testing.T) {
		msgs := genai.Messages{
			{
				Role: genai.User,
				Contents: []genai.Content{
					{Text: prompt},
					{URL: "https://raw.githubusercontent.com/maruel/genai/refs/heads/main/internal/internaltest/testdata/" + url.PathEscape(filename)},
				},
			},
		}
		resp, err := runGenText(t, g(t, model), msgs, nil, stream)
		if !basicCheck(t, err, in[genai.ModalityImage].URL) {
			return
		}
		testUsage(t, &resp.Usage, f.BrokenTokenUsage, defaultFR)
		ValidateWordResponse(t, resp, "yes")
	})
}

func testAudioSTTFunctionalities(t *testing.T, g ProviderGenModalityFactory, model string, in, out map[genai.Modality]genai.ModalCapability, f *genai.FunctionalityText, stream bool) {
	defaultFR := genai.FinishedStop
	if f.BrokenFinishReason {
		defaultFR = ""
	}
	// Gemini supports Opus too.
	filename := "mystery_word.mp3"
	prompt := "What is the word said? Reply with only the word."
	t.Run("Inline", func(t *testing.T) {
		// Path with the assumption it's run from "//providers/<provider>/".
		ff, err := os.Open(filepath.Join("..", "..", "internal", "internaltest", "testdata", filename))
		if err != nil {
			t.Fatal(err)
		}
		defer ff.Close()
		msgs := genai.Messages{{Role: genai.User, Contents: []genai.Content{{Text: prompt}, {Document: ff}}}}
		resp, err := runGenText(t, g(t, model), msgs, nil, stream)
		if !basicCheck(t, err, in[genai.ModalityAudio].Inline) {
			return
		}
		testUsage(t, &resp.Usage, f.BrokenTokenUsage, defaultFR)
		ValidateWordResponse(t, resp, "orange")
	})

	t.Run("URL", func(t *testing.T) {
		msgs := genai.Messages{
			{
				Role: genai.User,
				Contents: []genai.Content{
					{Text: prompt},
					{URL: "https://raw.githubusercontent.com/maruel/genai/refs/heads/main/internal/internaltest/testdata/" + url.PathEscape(filename)},
				},
			},
		}
		resp, err := runGenText(t, g(t, model), msgs, nil, stream)
		if !basicCheck(t, err, in[genai.ModalityAudio].URL) {
			return
		}
		testUsage(t, &resp.Usage, f.BrokenTokenUsage, defaultFR)
		ValidateWordResponse(t, resp, "orange")
	})
}

func testVideoFunctionalities(t *testing.T, g ProviderGenModalityFactory, model string, in, out map[genai.Modality]genai.ModalCapability, f *genai.FunctionalityText, stream bool) {
	defaultFR := genai.FinishedStop
	if f.BrokenFinishReason {
		defaultFR = ""
	}
	filename := "animation.mp4"
	prompt := "What is the word? Reply with only the word."
	t.Run("Inline", func(t *testing.T) {
		// Path with the assumption it's run from "//providers/<provider>/".
		ff, err := os.Open(filepath.Join("..", "..", "internal", "internaltest", "testdata", filename))
		if err != nil {
			t.Fatal(err)
		}
		defer ff.Close()
		msgs := genai.Messages{{Role: genai.User, Contents: []genai.Content{{Text: prompt}, {Document: ff}}}}
		resp, err := runGenText(t, g(t, model), msgs, nil, stream)
		if !basicCheck(t, err, in[genai.ModalityVideo].Inline) {
			return
		}
		testUsage(t, &resp.Usage, f.BrokenTokenUsage, defaultFR)
		ValidateWordResponse(t, resp, "banana")
	})

	t.Run("URL", func(t *testing.T) {
		msgs := genai.Messages{
			{
				Role: genai.User,
				Contents: []genai.Content{
					{Text: prompt},
					{URL: "https://raw.githubusercontent.com/maruel/genai/refs/heads/main/internal/internaltest/testdata/" + url.PathEscape(filename)},
				},
			},
		}
		resp, err := runGenText(t, g(t, model), msgs, nil, stream)
		if !basicCheck(t, err, in[genai.ModalityVideo].URL) {
			return
		}
		testUsage(t, &resp.Usage, f.BrokenTokenUsage, defaultFR)
		ValidateWordResponse(t, resp, "banana")
	})
}

func testPDFFunctionalities(t *testing.T, g ProviderGenModalityFactory, model string, in, out map[genai.Modality]genai.ModalCapability, f *genai.FunctionalityText, stream bool) {
	defaultFR := genai.FinishedStop
	if f.BrokenFinishReason {
		defaultFR = ""
	}
	filename := "hidden_word.pdf"
	prompt := "What is the word? Reply with only the word."
	t.Run("Inline", func(t *testing.T) {
		// Path with the assumption it's run from "//providers/<provider>/".
		ff, err := os.Open(filepath.Join("..", "..", "internal", "internaltest", "testdata", filename))
		if err != nil {
			t.Fatal(err)
		}
		defer ff.Close()
		msgs := genai.Messages{{Role: genai.User, Contents: []genai.Content{{Text: prompt}, {Document: ff}}}}
		resp, err := runGenText(t, g(t, model), msgs, nil, stream)
		if !basicCheck(t, err, in[genai.ModalityPDF].Inline) {
			return
		}
		testUsage(t, &resp.Usage, f.BrokenTokenUsage, defaultFR)
		ValidateWordResponse(t, resp, "orange")
	})

	t.Run("URL", func(t *testing.T) {
		msgs := genai.Messages{
			{
				Role: genai.User,
				Contents: []genai.Content{
					{Text: prompt},
					{URL: "https://raw.githubusercontent.com/maruel/genai/refs/heads/main/internal/internaltest/testdata/" + url.PathEscape(filename)},
				},
			},
		}
		resp, err := runGenText(t, g(t, model), msgs, nil, stream)
		if !basicCheck(t, err, in[genai.ModalityPDF].URL) {
			return
		}
		testUsage(t, &resp.Usage, f.BrokenTokenUsage, defaultFR)
		ValidateWordResponse(t, resp, "orange")
	})
}

func testChatImageGenFunctionalities(t *testing.T, g ProviderGenModalityFactory, model string, in, out map[genai.Modality]genai.ModalCapability, f *genai.FunctionalityText, stream bool) {
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
	if f.BrokenFinishReason {
		defaultFR = ""
	}
	msgs := genai.Messages{genai.NewTextMessage(genai.User, contents)}
	resp, err := runGenText(t, g(t, model), msgs, nil, stream)
	if !basicCheck(t, err, hasAnyModalityWithInline(out) || hasAnyModalityWithURL(out)) {
		return
	}
	testUsage(t, &resp.Usage, f.BrokenTokenUsage, defaultFR)
	if len(resp.Contents) == 0 {
		t.Fatal("expected content")
	}
	if len(resp.Contents) != 1 {
		t.Fatalf("expected one content, got %d", len(resp.Contents))
	}
	if resp.Contents[0].Filename != "content.png" && resp.Contents[0].Filename != "content.jpg" {
		t.Fatalf("expected one image, got %#v", resp.Contents[0])
	}
	// It can have text, images or both.
	// TODO: This is brittle.
	if hasAnyModalityWithInline(out) {
		if resp.Contents[0].Document == nil {
			t.Fatal("expected inline output")
		}
	}
	if hasAnyModalityWithURL(out) {
		if resp.Contents[0].URL == "" {
			t.Fatal("expected URL output")
		}
	}
}

//

func testGenDoc(t *testing.T, g ProviderGenModalityFactory, model string, in, out map[genai.Modality]genai.ModalCapability, f *genai.FunctionalityDoc) {
	ran := false
	if len(out) == 1 {
		if _, hasImage := out[genai.ModalityImage]; hasImage {
			// Image only output, e.g. OpenAI, Pollinations, TogetherAI.
			// Only run once, no need to run twice.
			t.Run("imagegen", func(t *testing.T) {
				testImageGenFunctionalities(t, g, model, in, out, f)
			})
			ran = true
		}
	}
	if _, hasAudio := out[genai.ModalityAudio]; hasAudio {
		t.Run("audiogen", func(t *testing.T) {
			testAudioGenFunctionalities(t, g, model, in, out, f)
		})
		ran = true
	}
	if !ran {
		t.Fatal("implement test case for this modalities combination")
	}
}

func testImageGenFunctionalities(t *testing.T, g ProviderGenModalityFactory, model string, in, out map[genai.Modality]genai.ModalCapability, f *genai.FunctionalityDoc) {
	defaultFR := genai.FinishedStop
	if f.BrokenFinishReason {
		defaultFR = ""
	}
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
	msg := genai.NewTextMessage(genai.User, contents)
	c := g(t, model).(genai.ProviderGenDoc)
	// TODO: options for non-text to test Seed.
	resp, err := c.GenDoc(t.Context(), msg, nil)
	if !basicCheck(t, err, hasAnyModalityWithInline(out) || hasAnyModalityWithURL(out)) {
		return
	}
	testUsage(t, &resp.Usage, f.BrokenTokenUsage, defaultFR)
	if len(resp.Contents) == 0 {
		t.Fatal("expected content")
	}
	if len(resp.Contents) != 1 {
		t.Fatalf("expected one content, got %d", len(resp.Contents))
	}
	if resp.Contents[0].Filename != "content.png" && resp.Contents[0].Filename != "content.jpg" {
		t.Fatalf("expected one image, got %#v", resp.Contents[0])
	}
	if hasAnyModalityWithInline(out) {
		if resp.Contents[0].Document == nil {
			t.Fatal("expected inline output")
		}
	}
	if hasAnyModalityWithURL(out) {
		if resp.Contents[0].URL == "" {
			t.Fatal("expected URL output")
		}
	}
}

func testAudioGenFunctionalities(t *testing.T, g ProviderGenModalityFactory, model string, in, out map[genai.Modality]genai.ModalCapability, f *genai.FunctionalityDoc) {
	defaultFR := genai.FinishedStop
	if f.BrokenFinishReason {
		defaultFR = ""
	}
	msg := genai.NewTextMessage(genai.User, "Say hi. Just say this word, nothing else.")
	c := g(t, model).(genai.ProviderGenDoc)
	// TODO: options for non-text to test Seed.
	resp, err := c.GenDoc(t.Context(), msg, nil)
	if !basicCheck(t, err, hasAnyModalityWithInline(out) || hasAnyModalityWithURL(out)) {
		return
	}
	testUsage(t, &resp.Usage, f.BrokenTokenUsage, defaultFR)
	if len(resp.Contents) == 0 {
		t.Fatal("expected content")
	}
	if len(resp.Contents) != 1 {
		t.Fatalf("expected one content, got %d", len(resp.Contents))
	}
	if resp.Contents[0].Filename != "sound.wav" {
		t.Fatalf("expected one image, got %#v", resp.Contents[0])
	}
	if hasAnyModalityWithInline(out) {
		if resp.Contents[0].Document == nil {
			t.Fatal("expected inline output")
		}
	}
	if hasAnyModalityWithURL(out) {
		if resp.Contents[0].URL == "" {
			t.Fatal("expected URL output")
		}
	}
}

//

func runGenText(t *testing.T, c genai.ProviderGen, msgs genai.Messages, opts genai.Options, stream bool) (genai.Result, error) {
	ctx := t.Context()
	if !stream {
		resp, err := c.GenSync(ctx, msgs, opts)
		// Uncomment to diagnose issues:
		// t.Logf("Response: %v", resp.Message)
		return resp, err
	}
	chunks := make(chan genai.ContentFragment)
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
	resp, err := c.GenStream(ctx, msgs, chunks, opts)
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

// ValidateWordResponse validates that the response contains exactly one of the expected words.
func ValidateWordResponse(t *testing.T, resp genai.Result, want ...string) {
	got := resp.AsText()
	cleaned := strings.TrimRight(strings.TrimSpace(strings.ToLower(got)), ".!")
	if !slices.Contains(want, cleaned) {
		t.Helper()
		t.Fatalf("Expected %q, got %q", strings.Join(want, ", "), got)
	}
}

// See the 3kib banana jpg online at
// https://github.com/maruel/genai/blob/main/openai/testdata/banana.jpg
//
//go:embed testdata/banana.jpg
var bananaJpg []byte

func hasAnyModalityWithInline(capabilities map[genai.Modality]genai.ModalCapability) bool {
	for _, c := range capabilities {
		if c.Inline {
			return true
		}
	}
	return false
}

func hasAnyModalityWithURL(capabilities map[genai.Modality]genai.ModalCapability) bool {
	for _, c := range capabilities {
		if c.URL {
			return true
		}
	}
	return false
}
