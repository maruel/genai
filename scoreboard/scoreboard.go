// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Package scoreboard creates a scoreboard based on a provider instance.
package scoreboard

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"maps"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strings"
	"sync"

	"github.com/maruel/genai"
	"github.com/maruel/genai/internal"
	"golang.org/x/sync/errgroup"
	"gopkg.in/dnaeon/go-vcr.v4/pkg/cassette"
)

// ProviderFactory is a function that returns a provider instance. The name represents the sub-test name. This
// may be used for HTTP recording and replays.
type ProviderFactory func(name string) genai.Provider

// CreateScenario calculates the supported Scenario for the given provider and its current model.
//
// ProviderFactory must be concurrent safe.
func CreateScenario(ctx context.Context, pf ProviderFactory) (genai.Scenario, genai.Usage, error) {
	usage := genai.Usage{}
	c := pf("")
	m := c.ModelID()
	if m == "" {
		return genai.Scenario{}, usage, errors.New("provider must have a model")
	}
	mu := sync.Mutex{}
	result := genai.Scenario{
		Models: []string{m},
		In:     map[genai.Modality]genai.ModalCapability{},
		Out:    map[genai.Modality]genai.ModalCapability{},
	}

	eg, ctx := errgroup.WithContext(ctx)
	if _, ok := c.(genai.ProviderGen); ok {
		eg.Go(func() error {
			// ctx2 := internal.WithLogger(ctx, internal.Logger(ctx).With("fn", "GenSync"))
			in, out, f, usageGen, err := exerciseGenCommon(ctx, pf, false, "GenSync")
			mu.Lock()
			usage.Add(usageGen)
			result.In = mergeModalities(result.In, in)
			result.Out = mergeModalities(result.Out, out)
			result.GenSync = f
			mu.Unlock()
			if err != nil {
				return fmt.Errorf("failed with GenSync: %w", err)
			}
			return nil
		})
		eg.Go(func() error {
			in, out, f, usageGen, err := exerciseGenCommon(ctx, pf, true, "GenStream")
			mu.Lock()
			usage.Add(usageGen)
			result.In = mergeModalities(result.In, in)
			result.Out = mergeModalities(result.Out, out)
			result.GenStream = f
			mu.Unlock()
			if err != nil {
				return fmt.Errorf("failed with GenStream: %w", err)
			}
			return nil
		})
	}

	if _, ok := c.(genai.ProviderGenDoc); ok {
		eg.Go(func() error {
			outDoc, usageDoc, err := exerciseGenDoc(ctx, pf)
			mu.Lock()
			usage.Add(usageDoc)
			result.In = mergeModalities(result.In, outDoc.In)
			result.Out = mergeModalities(result.Out, outDoc.Out)
			result.GenDoc = outDoc.GenDoc
			mu.Unlock()
			if err != nil {
				return fmt.Errorf("failed with GenDoc: %w", err)
			}
			return nil
		})
	}

	err := eg.Wait()
	return result, usage, err
}

// genai.ProviderGen

func exerciseGenCommon(ctx context.Context, pf ProviderFactory, isStream bool, name string) (map[genai.Modality]genai.ModalCapability, map[genai.Modality]genai.ModalCapability, *genai.FunctionalityText, genai.Usage, error) {
	in := map[genai.Modality]genai.ModalCapability{}
	out := map[genai.Modality]genai.ModalCapability{}
	usage := genai.Usage{}
	// Make sure simple text generation works, otherwise there's no point.
	msgs := genai.Messages{genai.NewTextMessage(genai.User, "Say hello. Use only one word.")}
	resp, err := callGen(ctx, pf, name+"Text", msgs, nil, isStream, &usage)
	if err != nil {
		return in, out, nil, usage, fmt.Errorf("basic check failed: %w", err)
	}
	in[genai.ModalityText] = genai.ModalCapability{Inline: true}
	out[genai.ModalityText] = genai.ModalCapability{Inline: true}
	f := &genai.FunctionalityText{}
	for _, c := range resp.Contents {
		if c.Thinking != "" {
			f.Thinking = true
		}
	}
	if resp.InputTokens == 0 || resp.OutputTokens == 0 {
		f.BrokenTokenUsage = genai.True
	}
	if expectedFR := genai.FinishedStop; resp.FinishReason != expectedFR {
		internal.Logger(ctx).DebugContext(ctx, "Text", "issue", "finish reason", "expected", expectedFR, "got", resp.FinishReason)
		f.BrokenFinishReason = true
	}

	// Seed
	msgs = genai.Messages{genai.NewTextMessage(genai.User, "Say hello. Use only one word.")}
	resp, err = callGen(ctx, pf, name+"Seed", msgs, &genai.OptionsText{Seed: 42}, isStream, &usage)
	if errors.Is(err, cassette.ErrInteractionNotFound) {
		return in, out, f, usage, err
	}
	f.Seed = err == nil

	// MaxTokens
	msgs = genai.Messages{genai.NewTextMessage(genai.User, "Explain the theory of relativity in great details.")}
	resp, err = callGen(ctx, pf, name+"MaxTokens", msgs, &genai.OptionsText{MaxTokens: 16}, isStream, &usage)
	if errors.Is(err, cassette.ErrInteractionNotFound) {
		return in, out, f, usage, err
	}
	f.NoMaxTokens = err != nil || strings.Count(resp.AsText(), " ")+1 > 20
	if !f.NoMaxTokens && (resp.InputTokens == 0 || resp.OutputTokens == 0) {
		f.BrokenTokenUsage = genai.True
	}
	expectedFR := genai.FinishedLength
	if f.NoMaxTokens {
		expectedFR = genai.FinishedStop
	}
	if resp.FinishReason != expectedFR {
		internal.Logger(ctx).DebugContext(ctx, "MaxTokens", "issue", "finish reason", "expected", expectedFR, "got", resp.FinishReason)
		f.BrokenFinishReason = true
	}

	// Stop
	msgs = genai.Messages{genai.NewTextMessage(genai.User, "Talk about Canada in great details. Start with: Canada is")}
	resp, err = callGen(ctx, pf, name+"Stop", msgs, &genai.OptionsText{Stop: []string{"is"}}, isStream, &usage)
	if errors.Is(err, cassette.ErrInteractionNotFound) {
		return in, out, f, usage, err
	}
	f.NoStopSequence = err != nil || strings.Count(resp.AsText(), " ")+1 > 20
	if !f.NoStopSequence && (resp.InputTokens == 0 || resp.OutputTokens == 0) {
		f.BrokenTokenUsage = genai.True
	}
	expectedFR = genai.FinishedStopSequence
	if f.NoStopSequence {
		expectedFR = genai.FinishedStop
	}
	if resp.FinishReason != expectedFR {
		internal.Logger(ctx).DebugContext(ctx, "Stop", "issue", "finish reason", "expected", expectedFR, "got", resp.FinishReason)
		f.BrokenFinishReason = true
	}

	// JSON
	msgs = genai.Messages{genai.NewTextMessage(genai.User, `Is a banana a fruit? Do not include an explanation. Reply ONLY as JSON according to the provided schema: {"is_fruit": bool}.`)}
	resp, err = callGen(ctx, pf, name+"JSON", msgs, &genai.OptionsText{ReplyAsJSON: true}, isStream, &usage)
	if errors.Is(err, cassette.ErrInteractionNotFound) {
		return in, out, f, usage, err
	}
	if err == nil {
		var data map[string]any
		// We could check for "is_fruit". In practice the fact that it's JSON is good enough to have the flag set.
		f.JSON = resp.Decode(&data) == nil
	}
	if f.JSON && (resp.InputTokens == 0 || resp.OutputTokens == 0) {
		f.BrokenTokenUsage = genai.True
	}
	if expectedFR = genai.FinishedStop; f.JSON && resp.FinishReason != expectedFR {
		internal.Logger(ctx).DebugContext(ctx, "JSON", "issue", "finish reason", "expected", expectedFR, "got", resp.FinishReason)
		f.BrokenFinishReason = true
	}

	// JSONSchema
	msgs = genai.Messages{genai.NewTextMessage(genai.User, `Is a banana a fruit? Do not include an explanation. Reply ONLY as JSON.`)}
	type schema struct {
		IsFruit bool `json:"is_fruit"`
	}
	resp, err = callGen(ctx, pf, name+"JSONSchema", msgs, &genai.OptionsText{DecodeAs: &schema{}}, isStream, &usage)
	if errors.Is(err, cassette.ErrInteractionNotFound) {
		return in, out, f, usage, err
	}
	if err == nil {
		data := schema{}
		f.JSONSchema = resp.Decode(&data) == nil && data.IsFruit
	}
	if f.JSONSchema && (resp.InputTokens == 0 || resp.OutputTokens == 0) {
		f.BrokenTokenUsage = genai.True
	}
	if expectedFR = genai.FinishedStop; f.JSONSchema && resp.FinishReason != expectedFR {
		internal.Logger(ctx).DebugContext(ctx, "JSONSchema", "issue", "finish reason", "expected", expectedFR, "got", resp.FinishReason)
		f.BrokenFinishReason = true
	}

	if err = exerciseGenTools(ctx, pf, f, isStream, name+"Tools", &usage); err != nil {
		return in, out, f, usage, err
	}

	// Citations
	msgs = genai.Messages{
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
	resp, err = callGen(ctx, pf, name+"Citations", msgs, nil, isStream, &usage)
	if errors.Is(err, cassette.ErrInteractionNotFound) {
		return in, out, f, usage, err
	}
	if err == nil {
		for _, content := range resp.Contents {
			if len(content.Citations) > 0 {
				f.Citations = true
				break
			}
		}
	}
	if f.Citations && (resp.InputTokens == 0 || resp.OutputTokens == 0) {
		f.BrokenTokenUsage = genai.True
	}
	if expectedFR = genai.FinishedStop; f.Citations && resp.FinishReason != expectedFR {
		internal.Logger(ctx).DebugContext(ctx, "Citations", "issue", "finish reason", "expected", expectedFR, "got", resp.FinishReason)
		f.BrokenFinishReason = true
	}

	root, err := getGitRootPath()
	if err != nil {
		return in, out, f, usage, err
	}
	m, err := exerciseGenPDFInput(ctx, root, pf, f, isStream, name+"PDF", &usage)
	if m != nil {
		in[genai.ModalityPDF] = *m
	}
	if err != nil {
		return in, out, f, usage, err
	}
	m, err = exerciseGenImageInput(ctx, root, pf, f, isStream, name+"Image", &usage)
	if m != nil {
		in[genai.ModalityImage] = *m
	}
	if err != nil {
		return in, out, f, usage, err
	}
	m, err = exerciseGenAudioInput(ctx, root, pf, f, isStream, name+"Audio", &usage)
	if m != nil {
		in[genai.ModalityAudio] = *m
	}
	if err != nil {
		return in, out, f, usage, err
	}
	m, err = exerciseGenVideoInput(ctx, root, pf, f, isStream, name+"Video", &usage)
	if m != nil {
		in[genai.ModalityVideo] = *m
	}
	if err != nil {
		return in, out, f, usage, err
	}
	return in, out, f, usage, nil
}

func exerciseGenPDFInput(ctx context.Context, root string, pf ProviderFactory, f *genai.FunctionalityText, isStream bool, name string, usage *genai.Usage) (*genai.ModalCapability, error) {
	pdfFile, err := os.Open(filepath.Join(root, "internal", "internaltest", "testdata", "hidden_word.pdf"))
	if err != nil {
		return nil, fmt.Errorf("failed to open PDF file: %w", err)
	}
	defer pdfFile.Close()

	msgs := genai.Messages{
		genai.Message{
			Role: genai.User,
			Contents: []genai.Content{
				{Text: "What is the word? Reply with only the word."},
				{Document: pdfFile, Filename: "hidden_word.pdf"},
			},
		},
	}
	var m *genai.ModalCapability
	if resp, err := callGen(ctx, pf, name+"Inline", msgs, nil, isStream, usage); err == nil {
		// TODO: Verify each "PDF" (really, document) formats and related friends.
		m = &genai.ModalCapability{}
		m.Inline = true
		m.SupportedFormats = []string{"application/pdf", "application/x-javascript", "text/javascript", "application/x-python", "text/x-python", "text/plain", "text/html", "text/css", "text/markdown"}
		if resp.InputTokens == 0 || resp.OutputTokens == 0 {
			f.BrokenTokenUsage = genai.True
		}
		if expectedFR := genai.FinishedStop; resp.FinishReason != expectedFR {
			internal.Logger(ctx).DebugContext(ctx, "PDFInline", "issue", "finish reason", "expected", expectedFR, "got", resp.FinishReason)
			f.BrokenFinishReason = true
		}
	} else if errors.Is(err, cassette.ErrInteractionNotFound) {
		return m, err
	}
	msgs = genai.Messages{
		genai.Message{
			Role: genai.User,
			Contents: []genai.Content{
				{Text: "What is the word? Reply with only the word."},
				{URL: "https://raw.githubusercontent.com/maruel/genai/refs/heads/main/internal/internaltest/testdata/hidden_word.pdf"},
			},
		},
	}
	if resp, err := callGen(ctx, pf, name+"URL", msgs, nil, isStream, usage); err == nil {
		if m == nil {
			m = &genai.ModalCapability{}
		}
		m.URL = true
		if len(m.SupportedFormats) == 0 {
			// TODO Same as above for PDFs. Make sure they match.
			m.SupportedFormats = []string{"application/pdf", "application/x-javascript", "text/javascript", "application/x-python", "text/x-python", "text/plain", "text/html", "text/css", "text/markdown"}
		}
		if resp.InputTokens == 0 || resp.OutputTokens == 0 {
			f.BrokenTokenUsage = genai.True
		}
		if expectedFR := genai.FinishedStop; resp.FinishReason != expectedFR {
			internal.Logger(ctx).DebugContext(ctx, "PDFURL", "issue", "finish reason", "expected", expectedFR, "got", resp.FinishReason)
			f.BrokenFinishReason = true
		}
	} else if errors.Is(err, cassette.ErrInteractionNotFound) {
		return m, err
	}
	return m, nil
}

func exerciseGenImageInput(ctx context.Context, root string, pf ProviderFactory, f *genai.FunctionalityText, isStream bool, name string, usage *genai.Usage) (*genai.ModalCapability, error) {
	imgFile, err := os.Open(filepath.Join(root, "internal", "internaltest", "testdata", "banana.jpg"))
	if err != nil {
		return nil, fmt.Errorf("failed to open image file: %w", err)
	}
	defer imgFile.Close()

	msgs := genai.Messages{
		genai.Message{
			Role: genai.User,
			Contents: []genai.Content{
				{Text: "Is it a banana? Reply with only one word."},
				{Document: imgFile, Filename: "banana.jpg"},
			},
		},
	}
	var m *genai.ModalCapability
	if resp, err := callGen(ctx, pf, name+"Inline", msgs, nil, isStream, usage); err == nil {
		// TODO: Test each of the image format. We need to generate the files first.
		m = &genai.ModalCapability{}
		m.Inline = true
		m.SupportedFormats = []string{"image/png", "image/jpeg", "image/gif", "image/webp"}
		if resp.InputTokens == 0 || resp.OutputTokens == 0 {
			f.BrokenTokenUsage = genai.True
		}
		if expectedFR := genai.FinishedStop; resp.FinishReason != expectedFR {
			internal.Logger(ctx).DebugContext(ctx, "ImageInline", "issue", "finish reason", "expected", expectedFR, "got", resp.FinishReason)
			f.BrokenFinishReason = true
		}
	} else if errors.Is(err, cassette.ErrInteractionNotFound) {
		return m, err
	}
	msgs = genai.Messages{
		genai.Message{
			Role: genai.User,
			Contents: []genai.Content{
				{Text: "Is it a banana? Reply with only one word."},
				{URL: "https://raw.githubusercontent.com/maruel/genai/refs/heads/main/internal/internaltest/testdata/banana.jpg"},
			},
		},
	}
	if resp, err := callGen(ctx, pf, name+"URL", msgs, nil, isStream, usage); err == nil {
		if m == nil {
			m = &genai.ModalCapability{}
		}
		m.URL = true
		if len(m.SupportedFormats) == 0 {
			// TODO Same as above for images. Make sure they match.
			m.SupportedFormats = []string{"image/png", "image/jpeg", "image/gif", "image/webp"}
		}
		if resp.InputTokens == 0 || resp.OutputTokens == 0 {
			f.BrokenTokenUsage = genai.True
		}
		if expectedFR := genai.FinishedStop; resp.FinishReason != expectedFR {
			internal.Logger(ctx).DebugContext(ctx, "ImageURL", "issue", "finish reason", "expected", expectedFR, "got", resp.FinishReason)
			f.BrokenFinishReason = true
		}
	} else if errors.Is(err, cassette.ErrInteractionNotFound) {
		return m, err
	}
	return m, nil
}

func exerciseGenAudioInput(ctx context.Context, root string, pf ProviderFactory, f *genai.FunctionalityText, isStream bool, name string, usage *genai.Usage) (*genai.ModalCapability, error) {
	audioFile, err := os.Open(filepath.Join(root, "internal", "internaltest", "testdata", "mystery_word.mp3"))
	if err != nil {
		return nil, fmt.Errorf("failed to open audio file: %w", err)
	}
	defer audioFile.Close()

	msgs := genai.Messages{
		genai.Message{
			Role: genai.User,
			Contents: []genai.Content{
				{Text: "What is the word said? Reply with only the word."},
				{Document: audioFile, Filename: "mystery_word.mp3"},
			},
		},
	}
	var m *genai.ModalCapability
	if resp, err := callGen(ctx, pf, name+"Inline", msgs, nil, isStream, usage); err == nil {
		// TODO: Test each of the audio format. We need to generate the files first.
		m = &genai.ModalCapability{}
		m.Inline = true
		m.SupportedFormats = []string{"audio/wav", "audio/mp3", "audio/aiff", "audio/aac", "audio/ogg", "audio/flac"}
		if resp.InputTokens == 0 || resp.OutputTokens == 0 {
			f.BrokenTokenUsage = genai.True
		}
		if expectedFR := genai.FinishedStop; resp.FinishReason != expectedFR {
			internal.Logger(ctx).DebugContext(ctx, "AudioInline", "issue", "finish reason", "expected", expectedFR, "got", resp.FinishReason)
			f.BrokenFinishReason = true
		}
	} else if errors.Is(err, cassette.ErrInteractionNotFound) {
		return m, err
	}
	msgs = genai.Messages{
		genai.Message{
			Role: genai.User,
			Contents: []genai.Content{
				{Text: "What is the word said? Reply with only the word."},
				{URL: "https://raw.githubusercontent.com/maruel/genai/refs/heads/main/internal/internaltest/testdata/mystery_word.mp3"},
			},
		},
	}
	if resp, err := callGen(ctx, pf, name+"URL", msgs, nil, isStream, usage); err == nil {
		if m == nil {
			m = &genai.ModalCapability{}
		}
		m.URL = true
		if len(m.SupportedFormats) == 0 {
			// TODO: Confirm audio format.
			m.SupportedFormats = []string{"audio/wav", "audio/mp3", "audio/aiff", "audio/aac", "audio/ogg", "audio/flac"}
		}
		if resp.InputTokens == 0 || resp.OutputTokens == 0 {
			f.BrokenTokenUsage = genai.True
		}
		if expectedFR := genai.FinishedStop; resp.FinishReason != expectedFR {
			internal.Logger(ctx).DebugContext(ctx, "AudioURL", "issue", "finish reason", "expected", expectedFR, "got", resp.FinishReason)
			f.BrokenFinishReason = true
		}
	} else if errors.Is(err, cassette.ErrInteractionNotFound) {
		return m, err
	}
	return m, nil
}

func exerciseGenVideoInput(ctx context.Context, root string, pf ProviderFactory, f *genai.FunctionalityText, isStream bool, name string, usage *genai.Usage) (*genai.ModalCapability, error) {
	videoFile, err := os.Open(filepath.Join(root, "internal", "internaltest", "testdata", "animation.mp4"))
	if err != nil {
		return nil, fmt.Errorf("failed to open video file: %w", err)
	}
	defer videoFile.Close()

	msgs := genai.Messages{
		genai.Message{
			Role: genai.User,
			Contents: []genai.Content{
				{Text: "What is the word? Reply with only the word."},
				{Document: videoFile, Filename: "animation.mp4"},
			},
		},
	}
	var m *genai.ModalCapability
	if resp, err := callGen(ctx, pf, name+"Inline", msgs, nil, isStream, usage); err == nil {
		// TODO: Verify each video format.
		m = &genai.ModalCapability{}
		m.Inline = true
		m.SupportedFormats = []string{"video/mp4", "video/mpeg", "video/mov", "video/avi", "video/x-flv", "video/mpg", "video/webm", "video/wmv", "video/3gpp"}
		if resp.InputTokens == 0 || resp.OutputTokens == 0 {
			f.BrokenTokenUsage = genai.True
		}
		if expectedFR := genai.FinishedStop; resp.FinishReason != expectedFR {
			internal.Logger(ctx).DebugContext(ctx, "VideoInline", "issue", "finish reason", "expected", expectedFR, "got", resp.FinishReason)
			f.BrokenFinishReason = true
		}
	} else if errors.Is(err, cassette.ErrInteractionNotFound) {
		return m, err
	}
	msgs = genai.Messages{
		genai.Message{
			Role: genai.User,
			Contents: []genai.Content{
				{Text: "What is the word? Reply with only the word."},
				{URL: "https://raw.githubusercontent.com/maruel/genai/refs/heads/main/internal/internaltest/testdata/animation.mp4"},
			},
		},
	}
	if resp, err := callGen(ctx, pf, name+"URL", msgs, nil, isStream, usage); err == nil {
		if m == nil {
			m = &genai.ModalCapability{}
		}
		m.URL = true
		if len(m.SupportedFormats) == 0 {
			// TODO: Confirm video format.
			m.SupportedFormats = []string{"video/mp4", "video/mpeg", "video/mov", "video/avi", "video/x-flv", "video/mpg", "video/webm", "video/wmv", "video/3gpp"}
		}
		if resp.InputTokens == 0 || resp.OutputTokens == 0 {
			f.BrokenTokenUsage = genai.True
		}
		if expectedFR := genai.FinishedStop; resp.FinishReason != expectedFR {
			internal.Logger(ctx).DebugContext(ctx, "VideoURL", "issue", "finish reason", "expected", expectedFR, "got", resp.FinishReason)
			f.BrokenFinishReason = true
		}
	} else if errors.Is(err, cassette.ErrInteractionNotFound) {
		return m, err
	}
	return m, nil
}

func exerciseGenTools(ctx context.Context, pf ProviderFactory, f *genai.FunctionalityText, isStream bool, name string, usage *genai.Usage) error {
	msgs := genai.Messages{genai.NewTextMessage(genai.User, "Use the square_root tool to calculate the square root of 132413 and reply with only the result. Do not give an explanation.")}
	type got struct {
		Number json.Number `json:"number"`
	}
	optsTools := genai.OptionsText{
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
		ToolCallRequest: genai.ToolCallRequired,
	}
	resp, err := callGen(ctx, pf, name+"SquareRoot", msgs, &optsTools, isStream, usage)
	if errors.Is(err, cassette.ErrInteractionNotFound) {
		return err
	}
	if err != nil || len(resp.ToolCalls) == 0 {
		// Tools are not supported, no need to do the rest.
		f.Tools = genai.False
		f.BiasedTool = genai.False
		f.IndecisiveTool = genai.False
		return nil
	}
	f.Tools = genai.True
	if resp.InputTokens == 0 || resp.OutputTokens == 0 {
		f.BrokenTokenUsage = genai.True
	}
	// The finish reason for tool calls is genai.FinishedToolCalls
	if expectedFR := genai.FinishedToolCalls; resp.FinishReason != expectedFR {
		internal.Logger(ctx).DebugContext(ctx, "SquareRoot", "issue", "finish reason", "expected", expectedFR, "got", resp.FinishReason)
		f.BrokenFinishReason = true
	}

	// BiasedTool and IndecisiveTool
	type gotCanadaFirst struct {
		Country string `json:"country" jsonschema:"enum=Canada,enum=USA"`
	}
	type gotUSAFirst struct {
		Country string `json:"country" jsonschema:"enum=USA,enum=Canada"`
	}
	data := [...]struct {
		callback        any
		countrySelected string // The country that should be selected if biased
		prompt          string
	}{
		{
			callback:        func(ctx context.Context, g *gotCanadaFirst) (string, error) { return g.Country, nil },
			countrySelected: "Canada",
			prompt:          "I wonder if Canada is a better country than the USA? Call the tool best_country to tell me which country is the best one.",
		},
		{
			callback:        func(ctx context.Context, g *gotUSAFirst) (string, error) { return g.Country, nil },
			countrySelected: "USA",
			prompt:          "I wonder if the USA is a better country than Canada? Call the tool best_country to tell me which country is the best one.",
		},
	}
	var biasedResults [len(data)]bool
	indecisiveOccurred := false
	for i, line := range data {
		opts := genai.OptionsText{
			Tools: []genai.ToolDef{
				{
					Name:        "best_country",
					Description: "A tool to specify the best country",
					Callback:    line.callback,
				},
			},
			ToolCallRequest: genai.ToolCallRequired,
		}

		check := name + fmt.Sprintf("ToolBias-%s", line.countrySelected)
		resp, err := callGen(ctx, pf, check, genai.Messages{genai.NewTextMessage(genai.User, line.prompt)}, &opts, isStream, usage)
		if errors.Is(err, cassette.ErrInteractionNotFound) {
			return err
		}
		if err != nil {
			// If there's an error, it means the tool call failed.
			// This might indicate flaky tool support.
			f.Tools = genai.Flaky
			continue // Skip to next test case
		}
		if len(resp.ToolCalls) == 0 {
			// No tool call, even though ToolCallRequired was set.
			// This also indicates flaky tool support.
			f.Tools = genai.Flaky
			continue
		}
		if resp.InputTokens == 0 || resp.OutputTokens == 0 {
			f.BrokenTokenUsage = genai.True
		}
		if expectedFR := genai.FinishedToolCalls; resp.FinishReason != expectedFR {
			internal.Logger(ctx).DebugContext(ctx, check, "issue", "finish reason", "expected", expectedFR, "got", resp.FinishReason)
			f.BrokenFinishReason = true
		}
		if len(resp.ToolCalls) == 1 {
			res, err := resp.ToolCalls[0].Call(context.Background(), opts.Tools)
			if err != nil {
				// Error during tool execution. That shouldn't happen.
				return fmt.Errorf("tool call failed: %w", err)
			}
			biasedResults[i] = res == line.countrySelected
		} else if len(resp.ToolCalls) == 2 {
			indecisiveOccurred = true
			var countries []string
			for _, tc := range resp.ToolCalls {
				res, err := tc.Call(context.Background(), opts.Tools)
				if err != nil {
					f.Tools = genai.Flaky
					continue
				}
				countries = append(countries, res)
			}
			// Verify countries if indecisive.
			slices.Sort(countries)
			if !slices.Equal(countries, []string{"Canada", "USA"}) {
				// This is an unexpected result for indecisive.
				f.Tools = genai.Flaky // Mark overall tools as flaky if indecisive result is not as expected
			}
		} else {
			// More than 2 tool calls, unexpected.
			f.Tools = genai.Flaky
			continue
		}
	}

	if indecisiveOccurred {
		f.IndecisiveTool = genai.True
	} else {
		f.IndecisiveTool = genai.False
	}

	if biasedResults[0] == biasedResults[1] {
		if biasedResults[0] {
			f.BiasedTool = genai.True
		} else {
			f.BiasedTool = genai.False
		}
	} else {
		f.BiasedTool = genai.Flaky
	}
	return nil
}

func callGen(ctx context.Context, pf ProviderFactory, name string, msgs genai.Messages, opts genai.Options, isStream bool, usage *genai.Usage) (genai.Result, error) {
	c := pf(name).(genai.ProviderGen)
	var err error
	var resp genai.Result
	if isStream {
		chunks := make(chan genai.ContentFragment)
		done := make(chan struct{})
		go func() {
			defer close(done)
			for {
				select {
				case <-done:
					return
				case _, ok := <-chunks:
					if !ok {
						return
					}
				}
			}
		}()
		resp, err = c.GenStream(ctx, msgs, chunks, opts)
		close(chunks)
		<-done
	} else {
		resp, err = c.GenSync(ctx, msgs, opts)
	}
	if err != nil {
		return resp, fmt.Errorf("%s: %w", name, err)
	}
	usage.InputTokens += resp.InputTokens
	usage.InputCachedTokens += resp.InputCachedTokens
	usage.OutputTokens += resp.OutputTokens
	return resp, err
}

// genai.ProviderGenDoc

// exerciseGenDoc exercises the ProviderGenDoc interface if available.
func exerciseGenDoc(ctx context.Context, pf ProviderFactory) (genai.Scenario, genai.Usage, error) {
	name := "GenDoc"
	out := genai.Scenario{
		In:  map[genai.Modality]genai.ModalCapability{},
		Out: map[genai.Modality]genai.ModalCapability{},
	}
	usage := genai.Usage{}
	out.GenDoc = &genai.FunctionalityDoc{}
	if err := exerciseGenDocImage(ctx, pf, name+"Image", &out, &usage); err != nil {
		return out, usage, err
	}
	if err := exerciseGenDocAudio(ctx, pf, name+"Audio", &out, &usage); err != nil {
		return out, usage, err
	}
	return out, usage, nil
}

func exerciseGenDocImage(ctx context.Context, pf ProviderFactory, name string, out *genai.Scenario, usage *genai.Usage) error {
	c := pf(name).(genai.ProviderGenDoc)
	promptImage := `A doodle animation on a white background of Cartoonish shiba inu with brown fur and a white belly, happily eating a pink ice-cream cone, subtle tail wag. Subtle motion but nothing else moves.`
	contentsImage := `Generate one square, white-background doodle with smooth, vibrantly colored image depicting ` + promptImage + `.

*Mandatory Requirements (Compacted):**

**Style:** Simple, vibrant, varied-colored doodle/hand-drawn sketch.
**Background:** Plain solid white (no background colors/elements). Absolutely no black background.
**Content & Motion:** Clearly depict **` + promptImage + `** action with colored, moving subject (no static images). If there's an action specified, it should be the main difference between frames.
**Format:** Square image (1:1 aspect ratio).
**Cropping:** Absolutely no black bars/letterboxing; colorful doodle fully visible against white.
**Output:** Actual image file for a smooth, colorful doodle-style image on a white background.`
	msg := genai.NewTextMessage(genai.User, contentsImage)
	resp, err := c.GenDoc(ctx, msg, nil)
	usage.InputTokens += resp.InputTokens
	usage.InputCachedTokens += resp.InputCachedTokens
	usage.OutputTokens += resp.OutputTokens
	if err == nil {
		if len(resp.Contents) > 0 && (resp.Contents[0].Filename == "content.png" || resp.Contents[0].Filename == "content.jpg") {
			v := out.In[genai.ModalityText]
			v.Inline = true
			out.In[genai.ModalityText] = v
			v = out.Out[genai.ModalityImage]
			// TODO: Detect if image generation is inline or URL.
			v.Inline = true
			out.Out[genai.ModalityImage] = v
			if resp.InputTokens == 0 || resp.OutputTokens == 0 {
				out.GenDoc.BrokenTokenUsage = genai.True
			}
			if expectedFR := genai.FinishedStop; resp.FinishReason != expectedFR {
				internal.Logger(ctx).DebugContext(ctx, "GenDocImage", "issue", "finish reason", "expected", expectedFR, "got", resp.FinishReason)
				out.GenDoc.BrokenFinishReason = true
			}
		}
	}
	if errors.Is(err, cassette.ErrInteractionNotFound) {
		return err
	}
	return nil
}

func exerciseGenDocAudio(ctx context.Context, pf ProviderFactory, name string, out *genai.Scenario, usage *genai.Usage) error {
	c := pf(name).(genai.ProviderGenDoc)
	msg := genai.NewTextMessage(genai.User, "Say hi. Just say this word, nothing else.")
	resp, err := c.GenDoc(ctx, msg, nil)
	usage.InputTokens += resp.InputTokens
	usage.InputCachedTokens += resp.InputCachedTokens
	usage.OutputTokens += resp.OutputTokens
	if err == nil {
		if len(resp.Contents) > 0 && resp.Contents[0].Filename == "sound.wav" {
			v := out.In[genai.ModalityText]
			v.Inline = true
			out.In[genai.ModalityText] = v
			v = out.Out[genai.ModalityAudio]
			// TODO: Detect if audio generation is inline or URL.
			v.Inline = true
			out.Out[genai.ModalityAudio] = v
			if resp.InputTokens == 0 || resp.OutputTokens == 0 {
				out.GenDoc.BrokenTokenUsage = genai.True
			}
			if expectedFR := genai.FinishedStop; resp.FinishReason != expectedFR {
				internal.Logger(ctx).DebugContext(ctx, "GenDocAudio", "issue", "finish reason", "expected", expectedFR, "got", resp.FinishReason)
				out.GenDoc.BrokenFinishReason = true
			}
		}
	}
	if errors.Is(err, cassette.ErrInteractionNotFound) {
		return err
	}
	return nil
}

func mergeModalities(m1, m2 map[genai.Modality]genai.ModalCapability) map[genai.Modality]genai.ModalCapability {
	out := maps.Clone(m1)
	for k, v2 := range m2 {
		if v1, ok := out[k]; ok {
			v1.Inline = v1.Inline || v2.Inline
			v1.URL = v1.URL || v2.URL
			v1.SupportedFormats = mergeSortedUnique(v1.SupportedFormats, v2.SupportedFormats)
			out[k] = v1
			continue
		}
		out[k] = v2
	}
	return out
}

func mergeSortedUnique(s1, s2 []string) []string {
	combined := append(s1, s2...)
	slices.Sort(combined)
	return slices.Compact(combined)
}

var (
	gitRootOnce sync.Once
	gitRootPath string
)

func getGitRootPath() (string, error) {
	var err error
	gitRootOnce.Do(func() {
		cmd := exec.Command("git", "rev-parse", "--show-toplevel")
		output, err2 := cmd.Output()
		if err2 != nil {
			err = err2
			return
		}
		path := strings.TrimSpace(string(output))
		// Ensure the path is absolute and clean
		absPath, err2 := filepath.Abs(path)
		if err2 != nil {
			err = err2
			return
		}
		gitRootPath = absPath
	})
	return gitRootPath, err
}
