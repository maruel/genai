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
	"math"
	"slices"
	"strings"

	"github.com/maruel/genai"
	"github.com/maruel/genai/internal"
	"gopkg.in/dnaeon/go-vcr.v4/pkg/cassette"
)

// ProviderFactory is a function that returns a provider instance. The name represents the sub-test name. This
// may be used for HTTP recording and replays.
type ProviderFactory func(name string) genai.Provider

// CreateScenario calculates the supported Scenario for the given provider and its current model.
func CreateScenario(ctx context.Context, pf ProviderFactory) (genai.Scenario, error) {
	c := pf("CreateScenario")
	m := c.ModelID()
	if m == "" {
		return genai.Scenario{}, errors.New("provider must have a model")
	}
	// TODO: We may still want the caller to tell which modalities are supported?
	out := genai.Scenario{
		Models: []string{m},
		In:     map[genai.Modality]genai.ModalCapability{},
		Out:    map[genai.Modality]genai.ModalCapability{},
	}
	if err := exerciseGen(ctx, pf, &out); err != nil {
		return out, err
	}
	if err := exerciseGenDoc(ctx, pf, &out); err != nil {
		return out, err
	}
	return out, nil
}

// genai.ProviderGen

// exerciseGen exercises the ProviderGen interface if available.
func exerciseGen(ctx context.Context, pf ProviderFactory, out *genai.Scenario) error {
	if _, ok := pf("Gen").(genai.ProviderGen); !ok {
		return nil
	}
	// TODO: Probe other modalities to detect if supported.
	// Test text-in, text-out.
	out.In[genai.ModalityText] = genai.ModalCapability{Inline: true}
	out.Out[genai.ModalityText] = genai.ModalCapability{Inline: true}

	// Test GenSync.
	fSync := genai.FunctionalityText{}
	ctx2 := internal.WithLogger(ctx, internal.Logger(ctx).With("fn", "GenSync"))
	if err := exerciseGenCommon(ctx2, pf, &fSync, false, "GenSync"); err != nil {
		return fmt.Errorf("failed with GenSync: %s", err)
	}
	out.GenSync = &fSync

	// Test GenStream.
	fStream := genai.FunctionalityText{}
	ctx2 = internal.WithLogger(ctx, internal.Logger(ctx).With("fn", "GenStream"))
	if err := exerciseGenCommon(ctx2, pf, &fStream, true, "GenStream"); err != nil {
		// It's ok for streaming to not be supported.
		out.GenStream = nil
		return fmt.Errorf("failed with GenAsync: %s", err)
	} else {
		out.GenStream = &fStream
	}
	return nil
}

func exerciseGenCommon(ctx context.Context, pf ProviderFactory, f *genai.FunctionalityText, isStream bool, name string) error {
	// Make sure simple text generation works, otherwise there's no point.
	msgs := genai.Messages{genai.NewTextMessage(genai.User, "Say hello. Use only one word.")}
	resp, err := callGen(ctx, pf, name+"Text", msgs, nil, isStream)
	if err != nil {
		return fmt.Errorf("basic check failed: %w", err)
	}
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
	_, err = callGen(ctx, pf, name+"Seed", msgs, &genai.OptionsText{Seed: 42}, isStream)
	if errors.Is(err, cassette.ErrInteractionNotFound) {
		return err
	}
	f.Seed = err == nil

	// MaxTokens
	msgs = genai.Messages{genai.NewTextMessage(genai.User, "Explain the theory of relativity in great details.")}
	resp, err = callGen(ctx, pf, name+"MaxTokens", msgs, &genai.OptionsText{MaxTokens: 16}, isStream)
	if errors.Is(err, cassette.ErrInteractionNotFound) {
		return err
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
	msgs = genai.Messages{genai.NewTextMessage(genai.User, "Talk about Canada in 10 words. Start with: Canada is")}
	resp, err = callGen(ctx, pf, name+"Stop", msgs, &genai.OptionsText{Stop: []string{"is"}}, isStream)
	if errors.Is(err, cassette.ErrInteractionNotFound) {
		return err
	}
	f.NoStopSequence = err != nil || len(resp.AsText()) > 12
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
	resp, err = callGen(ctx, pf, name+"JSON", msgs, &genai.OptionsText{ReplyAsJSON: true}, isStream)
	if errors.Is(err, cassette.ErrInteractionNotFound) {
		return err
	}
	if err == nil {
		var data map[string]any
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
	msgs = genai.Messages{genai.NewTextMessage(genai.User, `Responds with a JSON object with a float "number" and a bool "is_fruit".`)}
	type schema struct {
		Number  float64 `json:"number"`
		IsFruit bool    `json:"is_fruit"`
	}
	resp, err = callGen(ctx, pf, name+"JSONSchema", msgs, &genai.OptionsText{DecodeAs: &schema{}}, isStream)
	if errors.Is(err, cassette.ErrInteractionNotFound) {
		return err
	}
	if err == nil {
		var data map[string]any
		f.JSONSchema = resp.Decode(&data) == nil
	}
	if f.JSONSchema && (resp.InputTokens == 0 || resp.OutputTokens == 0) {
		f.BrokenTokenUsage = genai.True
	}
	if expectedFR = genai.FinishedStop; f.JSONSchema && resp.FinishReason != expectedFR {
		internal.Logger(ctx).DebugContext(ctx, "JSONSchema", "issue", "finish reason", "expected", expectedFR, "got", resp.FinishReason)
		f.BrokenFinishReason = true
	}

	if err = exerciseGenTools(ctx, pf, f, isStream, name+"Tools"); err != nil {
		return err
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
	resp, err = callGen(ctx, pf, name+"Citations", msgs, nil, isStream)
	if errors.Is(err, cassette.ErrInteractionNotFound) {
		return err
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
	return nil
}

func exerciseGenTools(ctx context.Context, pf ProviderFactory, f *genai.FunctionalityText, isStream bool, name string) error {
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
	resp, err := callGen(ctx, pf, name+"SquareRoot", msgs, &optsTools, isStream)
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
		resp, err := callGen(ctx, pf, check, genai.Messages{genai.NewTextMessage(genai.User, line.prompt)}, &opts, isStream)
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

func callGen(ctx context.Context, pf ProviderFactory, name string, msgs genai.Messages, opts genai.Options, isStream bool) (genai.Result, error) {
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
	return resp, err
}

// genai.ProviderGenDoc

// exerciseGenDoc exercises the ProviderGenDoc interface if available.
func exerciseGenDoc(ctx context.Context, pf ProviderFactory, out *genai.Scenario) error {
	if _, ok := pf("GenDoc").(genai.ProviderGenDoc); !ok {
		// Cerebras does not implement ProviderGenDoc, so we set GenDoc to nil.
		out.GenDoc = nil
		return nil
	}
	// TODO: implement actual GenDoc exercise
	out.GenDoc = nil
	return nil
}
