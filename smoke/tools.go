// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package smoke

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
	"github.com/maruel/genai/scoreboard"
)

func exerciseGenTools(ctx context.Context, cs *callState, f *scoreboard.Functionality, prefix string) error {
	msgs := genai.Messages{genai.NewTextMessage("Use the square_root tool to calculate the square root of 132413 and reply with only the result. Do not give an explanation.")}
	type got struct {
		Number json.Number `json:"number" jsonschema:"type=number"`
	}
	optsTools := genai.OptionsTools{
		Tools: []genai.ToolDef{
			{
				Name:        "square_root",
				Description: "Calculates and return the square root of a number",
				Callback: func(ctx context.Context, g *got) (string, error) {
					i, err := g.Number.Int64()
					if err != nil {
						// Cohere appends ".0"
						f, err := g.Number.Float64()
						if err != nil {
							return "", fmt.Errorf("wanted 132413 as an int, got %q: %w", g.Number, err)
						}
						i = int64(f)
					}
					if i != 132413 {
						return "", fmt.Errorf("wanted 132413 as an int, got %s", g.Number)
					}
					return fmt.Sprintf("%.2f", math.Sqrt(float64(i))), nil
				},
			},
		},
		Force: genai.ToolCallRequired,
	}
	// TODO: Do not consider a single tool call failure a failure, try a few times.
	resp, err := cs.callGen(ctx, prefix+"SquareRoot-1", msgs, &optsTools)
	if isBadError(ctx, err) {
		internal.Logger(ctx).DebugContext(ctx, "SquareRoot-1", "err", err)
		return err
	}
	internal.Logger(ctx).DebugContext(ctx, "SquareRoot-1", "resp", resp)
	flaky := false
	f.ToolCallRequired = true
	var uerr *genai.UnsupportedContinuableError
	if errors.As(err, &uerr) {
		// Cheap trick to make sure the error is not wrapped. Figure out if there's another way!
		if strings.HasPrefix(err.Error(), "unsupported options: ") {
			if slices.Contains(uerr.Unsupported, "OptionsTools.Force") {
				// Do not mark the test as flaky since it worked. Remember about ToolCallRequired not being supported
				// though.
				f.ToolCallRequired = false
				err = nil
			}
		}
	}
	hasCalls := slices.ContainsFunc(resp.Replies, func(r genai.Reply) bool { return !r.ToolCall.IsZero() })
	if !hasCalls || err != nil {
		// It is not necessarily flaky if the client returned an error, it's often that ToolCallRequired is not
		// supported. But if the client didn't report an error and there are no tool calls, that's bad and it's
		// really flaky.
		flaky = err == nil
		f.ToolCallRequired = false
		internal.Logger(ctx).DebugContext(ctx, "SquareRoot-1", "err", err, "msg", "trying toolany")
		// Try a second time without forcing a tool call.
		optsTools.Force = genai.ToolCallAny
		resp, err = cs.callGen(ctx, prefix+"SquareRoot-1-any", msgs, &optsTools)
		if isBadError(ctx, err) {
			internal.Logger(ctx).DebugContext(ctx, "SquareRoot-1-any", "err", err)
			return err
		}
		hasCalls = slices.ContainsFunc(resp.Replies, func(r genai.Reply) bool { return !r.ToolCall.IsZero() })
	}

	if err != nil || !hasCalls {
		internal.Logger(ctx).DebugContext(ctx, "SquareRoot", "err", err)
		// Tools are not supported, no need to do the rest.
		f.Tools = scoreboard.False
		f.ToolsBiased = scoreboard.False
		f.ToolsIndecisive = scoreboard.False
		f.ToolCallRequired = false
		return nil
	}

	msgs = append(msgs, resp.Message)
	tr, err := resp.DoToolCalls(ctx, optsTools.Tools)
	if err != nil {
		internal.Logger(ctx).DebugContext(ctx, "SquareRoot-1 (do calls)", "err", err)
		f.Tools = scoreboard.False
		f.ToolsBiased = scoreboard.False
		f.ToolsIndecisive = scoreboard.False
		f.ToolCallRequired = false
		return nil
	}
	if tr.IsZero() {
		f.Tools = scoreboard.False
		f.ToolsBiased = scoreboard.False
		f.ToolsIndecisive = scoreboard.False
		f.ToolCallRequired = false
		return fmt.Errorf("expected tool call to return a result or an error")
	}
	msgs = append(msgs, tr)
	optsTools.Force = genai.ToolCallNone

	resp, err = cs.callGen(ctx, prefix+"SquareRoot-2", msgs, &optsTools)
	if isBadError(ctx, err) {
		internal.Logger(ctx).DebugContext(ctx, "SquareRoot-2", "err", err)
		return err
	}
	if err != nil || slices.ContainsFunc(resp.Replies, func(r genai.Reply) bool { return !r.ToolCall.IsZero() }) {
		internal.Logger(ctx).DebugContext(ctx, "SquareRoot-2", "err", err)
		f.Tools = scoreboard.Flaky
		f.ToolsBiased = scoreboard.False
		f.ToolsIndecisive = scoreboard.False
		return nil
	}

	if flaky {
		f.Tools = scoreboard.Flaky
	} else {
		f.Tools = scoreboard.True
	}
	if resp.Usage.InputTokens == 0 || resp.Usage.OutputTokens == 0 {
		if f.ReportTokenUsage != scoreboard.False {
			internal.Logger(ctx).DebugContext(ctx, "SquareRoot", "issue", "token usage")
			f.ReportTokenUsage = scoreboard.Flaky
		}
	}
	// The finish reason for tool calls is genai.FinishedToolCalls
	if expectedFR := genai.FinishedStop; resp.Usage.FinishReason != expectedFR {
		if f.ReportTokenUsage != scoreboard.False {
			internal.Logger(ctx).DebugContext(ctx, "SquareRoot", "issue", "finish reason", "expected", expectedFR, "got", resp.Usage.FinishReason)
			f.ReportFinishReason = scoreboard.Flaky
		}
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
		opts := genai.OptionsTools{
			Tools: []genai.ToolDef{
				{
					Name:        "best_country",
					Description: "A tool to specify the best country",
					Callback:    line.callback,
				},
			},
		}
		if f.ToolCallRequired {
			// Some providers like cerebras and togetherai absolutely do not support this flag. So do not use it
			// unless it's supported.
			opts.Force = genai.ToolCallRequired
		}

		check := prefix + fmt.Sprintf("ToolBias-%s", line.countrySelected)
		resp, err := cs.callGen(ctx, check, genai.Messages{genai.NewTextMessage(line.prompt)}, &opts)
		if isBadError(ctx, err) {
			return err
		}
		if err != nil {
			// If there's an error, it means the tool call failed.
			// This might indicate flaky tool support.
			f.Tools = scoreboard.Flaky
			continue // Skip to next test case
		}
		if !slices.ContainsFunc(resp.Replies, func(r genai.Reply) bool { return !r.ToolCall.IsZero() }) {
			// No tool call, even though ToolCallRequired was set.
			// This also indicates flaky tool support.
			f.Tools = scoreboard.Flaky
			continue
		}
		if resp.Usage.InputTokens == 0 || resp.Usage.OutputTokens == 0 {
			if f.ReportTokenUsage != scoreboard.False {
				internal.Logger(ctx).DebugContext(ctx, check, "issue", "token usage")
				f.ReportTokenUsage = scoreboard.Flaky
			}
		}
		if expectedFR := genai.FinishedToolCalls; resp.Usage.FinishReason != expectedFR {
			if f.ReportTokenUsage != scoreboard.False {
				internal.Logger(ctx).DebugContext(ctx, check, "issue", "finish reason", "expected", expectedFR, "got", resp.Usage.FinishReason)
				f.ReportFinishReason = scoreboard.Flaky
			}
		}
		toolCalls := 0
		for _, r := range resp.Replies {
			if !r.ToolCall.IsZero() {
				toolCalls++
			}
		}
		if toolCalls == 1 {
			for j := range resp.Replies {
				if resp.Replies[j].ToolCall.IsZero() {
					continue
				}
				res, err := resp.Replies[j].ToolCall.Call(ctx, opts.Tools)
				if err != nil {
					// Error during tool execution. This only happens if the json schema is not followed. For example
					// I've seen on Huggingface using "country1" and "country2", aka being indecisive with a single
					// function call.
					f.Tools = scoreboard.Flaky
					continue
				}
				biasedResults[i] = res == line.countrySelected
			}
		} else if toolCalls == 2 {
			indecisiveOccurred = true
			var countries []string
			for j := range resp.Replies {
				if resp.Replies[j].ToolCall.IsZero() {
					continue
				}
				res, err := resp.Replies[j].ToolCall.Call(ctx, opts.Tools)
				if err != nil {
					f.Tools = scoreboard.Flaky
					continue
				}
				countries = append(countries, res)
			}
			// Verify countries if indecisive.
			slices.Sort(countries)
			if !slices.Equal(countries, []string{"Canada", "USA"}) {
				// This is an unexpected result for indecisive.
				f.Tools = scoreboard.Flaky // Mark overall tools as flaky if indecisive result is not as expected
			}
		} else {
			// More than 2 tool calls, unexpected.
			f.Tools = scoreboard.Flaky
			continue
		}
	}

	if indecisiveOccurred {
		f.ToolsIndecisive = scoreboard.True
	} else {
		f.ToolsIndecisive = scoreboard.False
	}

	if biasedResults[0] == biasedResults[1] {
		if biasedResults[0] {
			f.ToolsBiased = scoreboard.True
		} else {
			f.ToolsBiased = scoreboard.False
		}
	} else {
		f.ToolsBiased = scoreboard.Flaky
	}
	return nil
}
