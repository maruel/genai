// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Tests for Together AI provider DTOs.

package togetherai_test

import (
	"encoding/json"
	"errors"
	"strings"
	"testing"

	"github.com/maruel/genai"
	"github.com/maruel/genai/base"
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

func TestChatRequest(t *testing.T) {
	t.Run("thinking", func(t *testing.T) {
		for _, tc := range []struct {
			name     string
			model    string
			level    genai.EffortLevel
			effort   togetherai.ReasoningEffort
			enabled  *bool
			template *bool
		}{
			{name: "default", model: "openai/gpt-oss-120b"},
			{name: "oss_off", model: "openai/gpt-oss-120b", level: genai.EffortOff, effort: togetherai.ReasoningEffortLow},
			{name: "oss_medium", model: "openai/gpt-oss-120b", level: genai.EffortMedium, effort: togetherai.ReasoningEffortMedium},
			{name: "oss_high", model: "openai/gpt-oss-120b", level: genai.EffortHigh, effort: togetherai.ReasoningEffortHigh},
			{name: "deepseek_off", model: "deepseek-ai/DeepSeek-V4-Pro-0813", level: genai.EffortOff, enabled: new(false)},
			{name: "deepseek_low", model: "deepseek-ai/DeepSeek-V4-Pro-0813", level: genai.EffortLow, effort: togetherai.ReasoningEffortHigh, enabled: new(true)},
			{name: "deepseek_max", model: "deepseek-ai/DeepSeek-V4-Pro-0813", level: genai.EffortMax, effort: togetherai.ReasoningEffortMax, enabled: new(true)},
			{name: "qwen_off", model: "Qwen/Qwen3.5-9B", level: genai.EffortOff, template: new(false)},
			{name: "qwen_on", model: "Qwen/Qwen3.5-9B", level: genai.EffortLow, template: new(true)},
			{name: "glm_off", model: "zai-org/GLM-5.2", level: genai.EffortOff, enabled: new(false)},
			{name: "glm_on", model: "zai-org/GLM-5.2", level: genai.EffortHigh, enabled: new(true)},
		} {
			t.Run(tc.name, func(t *testing.T) {
				var r togetherai.ChatRequest
				if err := r.Init(genai.Messages{genai.NewTextMessage("hello")}, tc.model, tc.level); err != nil {
					t.Fatal(err)
				}
				if r.ReasoningEffort != tc.effort {
					t.Errorf("effort = %q, want %q", r.ReasoningEffort, tc.effort)
				}
				if tc.enabled == nil {
					if r.Reasoning != nil {
						t.Errorf("unexpected reasoning: %#v", r.Reasoning)
					}
				} else if r.Reasoning == nil || r.Reasoning.Enabled != *tc.enabled {
					t.Errorf("reasoning = %#v, want enabled=%v", r.Reasoning, *tc.enabled)
				}
				if tc.template == nil {
					if r.ChatTemplateKwargs != nil {
						t.Errorf("unexpected template: %#v", r.ChatTemplateKwargs)
					}
				} else if r.ChatTemplateKwargs == nil || r.ChatTemplateKwargs.EnableThinking == nil || *r.ChatTemplateKwargs.EnableThinking != *tc.template {
					t.Errorf("template = %#v, want thinking=%v", r.ChatTemplateKwargs, *tc.template)
				}
			})
		}
	})
	t.Run("unsupported", func(t *testing.T) {
		var r togetherai.ChatRequest
		err := r.Init(genai.Messages{genai.NewTextMessage("hello")}, "meta-llama/Llama-3.3-70B-Instruct-Turbo", genai.EffortHigh)
		if _, ok := errors.AsType[*base.ErrNotSupported](err); !ok {
			t.Fatalf("got %v, want unsupported thinking", err)
		}
	})
}

func TestMessage(t *testing.T) {
	t.Run("native_reasoning_roundtrip", func(t *testing.T) {
		var m togetherai.Message
		if err := json.Unmarshal([]byte(`{"role":"assistant","content":"answer","reasoning":"first","reasoning_content":"second"}`), &m); err != nil {
			t.Fatal(err)
		}
		var out genai.Message
		if err := m.To(&out); err != nil {
			t.Fatal(err)
		}
		var got togetherai.Message
		if err := got.From(&out); err != nil {
			t.Fatal(err)
		}
		if got.Reasoning != "first" || got.ReasoningContent != "second" {
			t.Errorf("native reasoning lost: %#v", got)
		}
	})
	t.Run("visible_reasoning", func(t *testing.T) {
		var m togetherai.Message
		if err := m.From(&genai.Message{Replies: []genai.Reply{{Reasoning: "thought"}, {Text: "answer"}}}); err != nil {
			t.Fatal(err)
		}
		if m.Reasoning != "thought" {
			t.Errorf("reasoning=%q", m.Reasoning)
		}
	})
}
