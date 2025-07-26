// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package scoreboard

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"slices"

	"github.com/maruel/genai"
	"github.com/maruel/genai/internal"
)

func exerciseGenTools(ctx context.Context, pf ProviderFactory, f *genai.FunctionalityText, isStream bool, prefix string, usage *genai.Usage) error {
	msgs := genai.Messages{genai.NewTextMessage(genai.User, "Use the square_root tool to calculate the square root of 132413 and reply with only the result. Do not give an explanation.")}
	type got struct {
		Number json.Number `json:"number" jsonschema:"type=number"`
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
	// TODO: Try a second time with ToolCallRequest to ToolCallAny.
	// TODO: Do not consider a single tool call failure a failure, try a few times.
	// TODO: Run adapters.GenSyncWithToolCallLoop or adapters.GenStreamWithToolCallLoop
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
		internal.Logger(ctx).DebugContext(ctx, "SquareRoot", "issue", "token usage")
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
			internal.Logger(ctx).DebugContext(ctx, check, "issue", "token usage")
			f.BrokenTokenUsage = genai.True
		}
		if expectedFR := genai.FinishedToolCalls; resp.FinishReason != expectedFR {
			internal.Logger(ctx).DebugContext(ctx, check, "issue", "finish reason", "expected", expectedFR, "got", resp.FinishReason)
			f.BrokenFinishReason = true
		}
		if len(resp.ToolCalls) == 1 {
			res, err := resp.ToolCalls[0].Call(context.Background(), opts.Tools)
			if err != nil {
				// Error during tool execution. This only happens if the json schema is not followed. For example
				// I've seen on Huggingface using "country1" and "country2", aka being indecisive with a single
				// function call.
				f.Tools = genai.Flaky
				continue
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
