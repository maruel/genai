// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Tests for Together AI provider DTOs.

package togetherai_test

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/maruel/genai/providers/togetherai"
)

func TestChatResponseToResult(t *testing.T) {
	tests := []struct {
		name             string
		content          string
		reasoning        string
		reasoningContent string
		text             string
		wantReasoning    string
	}{
		{
			name:             "reasoning content takes precedence",
			content:          `"answer"`,
			reasoning:        "ignored",
			reasoningContent: "thought",
			text:             "answer",
			wantReasoning:    "thought",
		},
		{
			name:          "reasoning",
			content:       `[{"type":"text","text":"answer"}]`,
			reasoning:     "thought",
			text:          "answer",
			wantReasoning: "thought",
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			input := `{"choices":[{"message":{"role":"assistant","content":` + test.content + `,"reasoning":"` + test.reasoning + `","reasoning_content":"` + test.reasoningContent + `"}}]}`
			var response togetherai.ChatResponse
			if err := json.Unmarshal([]byte(input), &response); err != nil {
				t.Fatal(err)
			}
			if got := response.Choices[0].Message.Content[0].Text; got != test.text {
				t.Errorf("decoded content = %q, want %q", got, test.text)
			}
			result, err := response.ToResult()
			if err != nil {
				t.Fatal(err)
			}
			if got := result.Message.Replies[0].Text; got != test.text {
				t.Errorf("reply text = %q, want %q", got, test.text)
			}
			if got := result.Message.Replies[1].Reasoning; got != test.wantReasoning {
				t.Errorf("reply reasoning = %q, want %q", got, test.wantReasoning)
			}
			message, err := json.Marshal(response.Choices[0].Message)
			if err != nil {
				t.Fatal(err)
			}
			if strings.Contains(string(message), `"reasoning_content"`) {
				t.Errorf("marshaled message = %s, contains reasoning_content", message)
			}
		})
	}
}
