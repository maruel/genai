// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Package scoreboard creates a scoreboard based on a provider instance.
package scoreboard

import (
	"bytes"
	"context"
	"embed"
	"encoding/json"
	"errors"
	"fmt"
	"maps"
	"math"
	"mime"
	"path/filepath"
	"slices"
	"strings"
	"sync"

	"github.com/maruel/genai"
	"github.com/maruel/genai/internal"
	"github.com/maruel/httpjson"
	"golang.org/x/sync/errgroup"
	"gopkg.in/dnaeon/go-vcr.v4/pkg/cassette"
)

//go:embed testdata/*
var testdataFiles embed.FS

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
			ctx2 := internal.WithLogger(ctx, internal.Logger(ctx).With("fn", "GenSync"))
			in, out, f, usageGen, err := exerciseGenCommon(ctx2, pf, false, "GenSync-")
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
			ctx2 := internal.WithLogger(ctx, internal.Logger(ctx).With("fn", "GenStream"))
			in, out, f, usageGen, err := exerciseGenCommon(ctx2, pf, true, "GenStream-")
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
			ctx2 := internal.WithLogger(ctx, internal.Logger(ctx).With("fn", "GenDoc"))
			outDoc, usageDoc, err := exerciseGenDoc(ctx2, pf)
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

func exerciseGenCommon(ctx context.Context, pf ProviderFactory, isStream bool, prefix string) (map[genai.Modality]genai.ModalCapability, map[genai.Modality]genai.ModalCapability, *genai.FunctionalityText, genai.Usage, error) {
	in := map[genai.Modality]genai.ModalCapability{}
	out := map[genai.Modality]genai.ModalCapability{}
	usage := genai.Usage{}
	// Make sure simple text generation works, otherwise there's no point.
	msgs := genai.Messages{genai.NewTextMessage(genai.User, "Say hello. Use only one word.")}
	resp, err := callGen(ctx, pf, prefix+"Text", msgs, nil, isStream, &usage)
	if err != nil {
		internal.Logger(ctx).DebugContext(ctx, "Text", "err", err)
		// It happens when the model is audio gen only.
		if !isBadError(err) {
			err = nil
		}
		return in, out, nil, usage, err
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
	resp, err = callGen(ctx, pf, prefix+"Seed", msgs, &genai.OptionsText{Seed: 42}, isStream, &usage)
	if isBadError(err) {
		return in, out, f, usage, err
	}
	f.Seed = err == nil

	// MaxTokens
	msgs = genai.Messages{genai.NewTextMessage(genai.User, "Explain the theory of relativity in great details.")}
	resp, err = callGen(ctx, pf, prefix+"MaxTokens", msgs, &genai.OptionsText{MaxTokens: 16}, isStream, &usage)
	if isBadError(err) {
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
	resp, err = callGen(ctx, pf, prefix+"Stop", msgs, &genai.OptionsText{Stop: []string{"is"}}, isStream, &usage)
	if isBadError(err) {
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
	resp, err = callGen(ctx, pf, prefix+"JSON", msgs, &genai.OptionsText{ReplyAsJSON: true}, isStream, &usage)
	if isBadError(err) {
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
	resp, err = callGen(ctx, pf, prefix+"JSONSchema", msgs, &genai.OptionsText{DecodeAs: &schema{}}, isStream, &usage)
	if isBadError(err) {
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

	if err = exerciseGenTools(ctx, pf, f, isStream, prefix+"Tools-", &usage); err != nil {
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
	resp, err = callGen(ctx, pf, prefix+"Citations", msgs, nil, isStream, &usage)
	if isBadError(err) {
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

	m, err := exerciseGenPDFInput(ctx, pf, f, isStream, prefix+"PDF-", &usage)
	if m != nil {
		in[genai.ModalityPDF] = *m
	}
	if err != nil {
		return in, out, f, usage, err
	}
	m, err = exerciseGenImageInput(ctx, pf, f, isStream, prefix+"Image-", &usage)
	if m != nil {
		in[genai.ModalityImage] = *m
	}
	if err != nil {
		return in, out, f, usage, err
	}
	m, err = exerciseGenAudioInput(ctx, pf, f, isStream, prefix+"Audio-", &usage)
	if m != nil {
		in[genai.ModalityAudio] = *m
	}
	if err != nil {
		return in, out, f, usage, err
	}
	m, err = exerciseGenVideoInput(ctx, pf, f, isStream, prefix+"Video-", &usage)
	if m != nil {
		in[genai.ModalityVideo] = *m
	}
	if err != nil {
		return in, out, f, usage, err
	}
	return in, out, f, usage, nil
}

func exerciseGenPDFInput(ctx context.Context, pf ProviderFactory, f *genai.FunctionalityText, isStream bool, prefix string, usage *genai.Usage) (*genai.ModalCapability, error) {
	var m *genai.ModalCapability
	want := "orange"
	for _, format := range []extMime{{"pdf", "application/pdf"}} {
		data, err := testdataFiles.ReadFile("testdata/document." + format.ext)
		if err != nil {
			return nil, fmt.Errorf("failed to open input file: %w", err)
		}
		msgs := genai.Messages{genai.Message{Role: genai.User, Contents: []genai.Content{
			{Text: "What is the word? Reply with only the word."},
			{Document: bytes.NewReader(data), Filename: "document." + format.ext},
		}}}
		name := prefix + format.ext + "-Inline"
		if err = exerciseModal(ctx, pf, f, isStream, name, usage, msgs, want); err == nil {
			if m == nil {
				m = &genai.ModalCapability{}
			}
			m.Inline = true
			if !slices.Contains(m.SupportedFormats, format.mime) {
				m.SupportedFormats = append(m.SupportedFormats, format.mime)
			}
		} else if isBadError(err) {
			return m, err
		} else {
			internal.Logger(ctx).DebugContext(ctx, name, "err", err)
		}
		msgs[0].Contents[1] = genai.Content{URL: rootURL + "document." + format.ext}
		name = prefix + format.ext + "-URL"
		if err = exerciseModal(ctx, pf, f, isStream, name, usage, msgs, want); err == nil {
			if m == nil {
				m = &genai.ModalCapability{}
			}
			m.URL = true
			if !slices.Contains(m.SupportedFormats, format.mime) {
				m.SupportedFormats = append(m.SupportedFormats, format.mime)
			}
		} else if isBadError(err) {
			return m, err
		} else {
			internal.Logger(ctx).DebugContext(ctx, name, "err", err)
		}
	}
	return m, nil
}

func exerciseGenImageInput(ctx context.Context, pf ProviderFactory, f *genai.FunctionalityText, isStream bool, prefix string, usage *genai.Usage) (*genai.ModalCapability, error) {
	var m *genai.ModalCapability
	want := "banana"
	for _, format := range []extMime{
		{"gif", "image/gif"},
		// TODO: {"heic", "image/heic"},
		{"jpg", "image/jpeg"},
		{"png", "image/png"},
		{"webp", "image/webp"},
	} {
		data, err := testdataFiles.ReadFile("testdata/image." + format.ext)
		if err != nil {
			return nil, fmt.Errorf("failed to open input file: %w", err)
		}
		msgs := genai.Messages{genai.Message{Role: genai.User, Contents: []genai.Content{
			{Text: "What fruit is it? Reply with only one word."},
			{Document: bytes.NewReader(data), Filename: "image." + format.ext},
		}}}
		name := prefix + format.ext + "-Inline"
		if err = exerciseModal(ctx, pf, f, isStream, name, usage, msgs, want); err == nil {
			if m == nil {
				m = &genai.ModalCapability{}
			}
			m.Inline = true
			if !slices.Contains(m.SupportedFormats, format.mime) {
				m.SupportedFormats = append(m.SupportedFormats, format.mime)
			}
		} else if isBadError(err) {
			return m, err
		} else {
			internal.Logger(ctx).DebugContext(ctx, name, "err", err)
		}
		msgs[0].Contents[1] = genai.Content{URL: rootURL + "image." + format.ext}
		name = prefix + format.ext + "-URL"
		if err = exerciseModal(ctx, pf, f, isStream, name, usage, msgs, want); err == nil {
			if m == nil {
				m = &genai.ModalCapability{}
			}
			m.URL = true
			if !slices.Contains(m.SupportedFormats, format.mime) {
				m.SupportedFormats = append(m.SupportedFormats, format.mime)
			}
		} else if isBadError(err) {
			return m, err
		} else {
			internal.Logger(ctx).DebugContext(ctx, name, "err", err)
		}
	}
	return m, nil
}

func exerciseGenAudioInput(ctx context.Context, pf ProviderFactory, f *genai.FunctionalityText, isStream bool, prefix string, usage *genai.Usage) (*genai.ModalCapability, error) {
	var m *genai.ModalCapability
	want := "orange"
	for _, format := range []extMime{
		{"aac", "audio/aac"},
		{"flac", "audio/flac"},
		{"mp3", "audio/mp3"},
		{"ogg", "audio/ogg"},
		{"wav", "audio/wav"},
	} {
		data, err := testdataFiles.ReadFile("testdata/audio." + format.ext)
		if err != nil {
			return nil, fmt.Errorf("failed to open input file: %w", err)
		}
		msgs := genai.Messages{genai.Message{Role: genai.User, Contents: []genai.Content{
			{Text: "What is the word said? Reply with only the word."},
			{Document: bytes.NewReader(data), Filename: "audio." + format.ext},
		}}}
		name := prefix + format.ext + "-Inline"
		if err = exerciseModal(ctx, pf, f, isStream, name, usage, msgs, want); err == nil {
			if m == nil {
				m = &genai.ModalCapability{}
			}
			m.Inline = true
			if !slices.Contains(m.SupportedFormats, format.mime) {
				m.SupportedFormats = append(m.SupportedFormats, format.mime)
			}
		} else if isBadError(err) {
			return m, err
		} else {
			internal.Logger(ctx).DebugContext(ctx, name, "err", err)
		}
		msgs[0].Contents[1] = genai.Content{URL: rootURL + "audio." + format.ext}
		name = prefix + format.ext + "-URL"
		if err = exerciseModal(ctx, pf, f, isStream, name, usage, msgs, want); err == nil {
			if m == nil {
				m = &genai.ModalCapability{}
			}
			m.URL = true
			if !slices.Contains(m.SupportedFormats, format.mime) {
				m.SupportedFormats = append(m.SupportedFormats, format.mime)
			}
		} else if isBadError(err) {
			return m, err
		} else {
			internal.Logger(ctx).DebugContext(ctx, name, "err", err)
		}
	}
	return m, nil
}

func exerciseGenVideoInput(ctx context.Context, pf ProviderFactory, f *genai.FunctionalityText, isStream bool, prefix string, usage *genai.Usage) (*genai.ModalCapability, error) {
	var m *genai.ModalCapability
	want := "orange"
	for _, format := range []extMime{
		{"mp4", "video/mp4"},
		{"webm", "video/webm"},
	} {
		data, err := testdataFiles.ReadFile("testdata/video." + format.ext)
		if err != nil {
			return nil, fmt.Errorf("failed to open input file: %w", err)
		}
		msgs := genai.Messages{genai.Message{Role: genai.User, Contents: []genai.Content{
			{Text: "What is the word said? Reply with only the word."},
			{Document: bytes.NewReader(data), Filename: "video." + format.ext},
		}}}
		name := prefix + format.ext + "-Inline"
		if err = exerciseModal(ctx, pf, f, isStream, name, usage, msgs, want); err == nil {
			if m == nil {
				m = &genai.ModalCapability{}
			}
			m.Inline = true
			if !slices.Contains(m.SupportedFormats, format.mime) {
				m.SupportedFormats = append(m.SupportedFormats, format.mime)
			}
		} else if isBadError(err) {
			return m, err
		} else {
			internal.Logger(ctx).DebugContext(ctx, name, "err", err)
		}
		msgs[0].Contents[1] = genai.Content{URL: rootURL + "video." + format.ext}
		name = prefix + format.ext + "-URL"
		if err = exerciseModal(ctx, pf, f, isStream, name, usage, msgs, want); err == nil {
			if m == nil {
				m = &genai.ModalCapability{}
			}
			m.URL = true
			if !slices.Contains(m.SupportedFormats, format.mime) {
				m.SupportedFormats = append(m.SupportedFormats, format.mime)
			}
		} else if isBadError(err) {
			return m, err
		} else {
			internal.Logger(ctx).DebugContext(ctx, name, "err", err)
		}
	}
	return m, nil
}

type extMime struct {
	ext  string
	mime string
}

func exerciseModal(ctx context.Context, pf ProviderFactory, f *genai.FunctionalityText, isStream bool, name string, usage *genai.Usage, msgs genai.Messages, want string) error {
	resp, err := callGen(ctx, pf, name, msgs, nil, isStream, usage)
	if err == nil {
		got := strings.ToLower(strings.TrimRight(strings.TrimSpace(resp.AsText()), "."))
		if want != "" && got != want {
			return fmt.Errorf("got %q, want %q", got, want)
		}
		if resp.InputTokens == 0 || resp.OutputTokens == 0 {
			f.BrokenTokenUsage = genai.True
		}
		if expectedFR := genai.FinishedStop; resp.FinishReason != expectedFR {
			internal.Logger(ctx).DebugContext(ctx, name, "issue", "finish reason", "expected", expectedFR, "got", resp.FinishReason)
			f.BrokenFinishReason = true
		}
	}
	return err
}

func exerciseGenTools(ctx context.Context, pf ProviderFactory, f *genai.FunctionalityText, isStream bool, prefix string, usage *genai.Usage) error {
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
	resp, err := callGen(ctx, pf, prefix+"SquareRoot", msgs, &optsTools, isStream, usage)
	if isBadError(err) {
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

		check := prefix + fmt.Sprintf("ToolBias-%s", line.countrySelected)
		resp, err := callGen(ctx, pf, check, genai.Messages{genai.NewTextMessage(genai.User, line.prompt)}, &opts, isStream, usage)
		if isBadError(err) {
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
	prefix := "GenDoc-"
	out := genai.Scenario{
		In:     map[genai.Modality]genai.ModalCapability{},
		Out:    map[genai.Modality]genai.ModalCapability{},
		GenDoc: &genai.FunctionalityDoc{},
	}
	usage := genai.Usage{}
	if err := exerciseGenDocImage(ctx, pf, prefix+"Image", &out, &usage); err != nil {
		return out, usage, err
	}
	if err := exerciseGenDocAudio(ctx, pf, prefix+"Audio", &out, &usage); err != nil {
		return out, usage, err
	}
	if len(out.In) == 0 || len(out.Out) == 0 {
		out.GenDoc = nil
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
	resp, err := c.GenDoc(ctx, msg, &genai.OptionsImage{})
	usage.InputTokens += resp.InputTokens
	usage.InputCachedTokens += resp.InputCachedTokens
	usage.OutputTokens += resp.OutputTokens
	if err == nil {
		if len(resp.Contents) == 0 {
			return fmt.Errorf("%s: no content", name)
		}
		c := resp.Contents[0]
		fn := c.GetFilename()
		if fn == "" {
			return fmt.Errorf("%s: no content filename", name)
		}
		out.In[genai.ModalityText] = genai.ModalCapability{Inline: true}
		v := out.Out[genai.ModalityImage]
		if resp.Contents[0].URL == "" {
			v.URL = true
		} else {
			v.Inline = true
		}
		v.SupportedFormats = append(v.SupportedFormats, mime.TypeByExtension(filepath.Ext(fn)))
		out.Out[genai.ModalityImage] = v
		if resp.InputTokens == 0 || resp.OutputTokens == 0 {
			out.GenDoc.BrokenTokenUsage = genai.True
		}
		if expectedFR := genai.FinishedStop; resp.FinishReason != expectedFR {
			internal.Logger(ctx).DebugContext(ctx, "GenDocImage", "issue", "finish reason", "expected", expectedFR, "got", resp.FinishReason)
			out.GenDoc.BrokenFinishReason = true
		}
	}
	if isBadError(err) {
		return err
	}
	return nil
}

func exerciseGenDocAudio(ctx context.Context, pf ProviderFactory, name string, out *genai.Scenario, usage *genai.Usage) error {
	c := pf(name).(genai.ProviderGenDoc)
	msg := genai.NewTextMessage(genai.User, "Say hi. Just say this word, nothing else.")
	resp, err := c.GenDoc(ctx, msg, &genai.OptionsAudio{})
	usage.InputTokens += resp.InputTokens
	usage.InputCachedTokens += resp.InputCachedTokens
	usage.OutputTokens += resp.OutputTokens
	if err == nil {
		if len(resp.Contents) == 0 {
			return fmt.Errorf("%s: no content", name)
		}
		c := resp.Contents[0]
		fn := c.GetFilename()
		if fn == "" {
			return fmt.Errorf("%s: no content filename", name)
		}
		out.In[genai.ModalityText] = genai.ModalCapability{Inline: true}
		out.Out[genai.ModalityAudio] = genai.ModalCapability{Inline: true}
		v := out.Out[genai.ModalityAudio]
		if resp.Contents[0].URL == "" {
			v.URL = true
		} else {
			v.Inline = true
		}
		v.SupportedFormats = append(v.SupportedFormats, mime.TypeByExtension(filepath.Ext(fn)))
		out.Out[genai.ModalityAudio] = v
		if resp.InputTokens == 0 || resp.OutputTokens == 0 {
			out.GenDoc.BrokenTokenUsage = genai.True
		}
		if expectedFR := genai.FinishedStop; resp.FinishReason != expectedFR {
			internal.Logger(ctx).DebugContext(ctx, "GenDocAudio", "issue", "finish reason", "expected", expectedFR, "got", resp.FinishReason)
			out.GenDoc.BrokenFinishReason = true
		}
	}
	if isBadError(err) {
		return err
	}
	return nil
}

const rootURL = "https://raw.githubusercontent.com/maruel/genai/refs/heads/main/scoreboard/testdata/"

func isBadError(err error) bool {
	var uerr *httpjson.UnknownFieldError
	if errors.Is(err, cassette.ErrInteractionNotFound) || errors.As(err, &uerr) {
		return true
	}
	// Tolerate HTTP 400 as when a model is passed something that it doesn't accept, e.g. sending audio input to
	// a text-only model, the provider often respond with 400. 500s should not be tolerated at all.
	var herr *httpjson.Error
	return errors.As(err, &herr) && herr.StatusCode >= 500
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
