// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package adapters_test

import (
	"context"
	"fmt"
	"testing"

	"github.com/maruel/genai"
	"github.com/maruel/genai/adapters"
)

// TestGenStreamWithToolCallLoop tests the GenStreamWithToolCallLoop function.
func TestGenStreamWithToolCallLoop(t *testing.T) {
	provider := &mockProviderGenStream{
		streamResponses: []streamResponse{
			{
				fragments: []genai.ContentFragment{
					{TextFragment: "I'll help you calculate that. "},
					{TextFragment: "Let me use the calculator tool."},
					{ToolCall: genai.ToolCall{ID: "1", Name: "calculator", Arguments: `{"a": 5, "b": 3, "operation": "add"}`}},
				},
				usage: genai.Usage{InputTokens: 10, OutputTokens: 20},
			},
			{
				fragments: []genai.ContentFragment{
					{TextFragment: "The result of 5 + 3 is 8."},
				},
				usage: genai.Usage{InputTokens: 15, OutputTokens: 10},
			},
		},
	}
	msgs := genai.Messages{genai.NewTextMessage("Calculate 5 + 3")}
	type CalculatorArgs struct {
		A         int    `json:"a"`
		B         int    `json:"b"`
		Operation string `json:"operation"`
	}
	opts := &genai.OptionsText{
		Tools: []genai.ToolDef{
			{
				Name:        "calculator",
				Description: "A simple calculator",
				Callback: func(ctx context.Context, args *CalculatorArgs) (string, error) {
					switch args.Operation {
					case "add":
						return fmt.Sprintf("%d", args.A+args.B), nil
					default:
						return "", fmt.Errorf("unsupported operation: %s", args.Operation)
					}
				},
			},
		},
	}
	chunks := make(chan genai.ContentFragment)
	var frags []genai.ContentFragment
	ctx := t.Context()
	go func() error {
		for {
			select {
			case <-ctx.Done():
				return ctx.Err()
			case fragment, ok := <-chunks:
				if !ok {
					return nil
				}
				frags = append(frags, fragment)
			}
		}
	}()

	respMsgs, usage, err := adapters.GenStreamWithToolCallLoop(ctx, provider, msgs, chunks, opts)
	close(chunks)
	if err != nil {
		t.Fatalf("GenStreamWithToolCallLoop returned an error: %v", err)
	}
	// Verify we got the expected number of messages
	if len(respMsgs) != 3 { // original + 1 LLM response + 1 tool result
		t.Fatalf("Expected 3 messages, got %d", len(respMsgs))
	}
	t.Logf("Messages: %+v", respMsgs)
	expectedUsage := genai.Usage{InputTokens: 25, OutputTokens: 30}
	if usage.InputTokens != expectedUsage.InputTokens || usage.OutputTokens != expectedUsage.OutputTokens {
		t.Fatalf("Expected usage %+v, got %+v", expectedUsage, usage)
	}
	// Verify we received all fragments
	if len(frags) != 4 { // 2 text fragments + 1 tool call + 1 text fragment
		t.Fatalf("Expected 4 fragments, got %d", len(frags))
	}
}
