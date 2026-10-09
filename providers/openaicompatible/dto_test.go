// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Tests for OpenAI-compatible tool calling wire types.

package openaicompatible_test

import (
	"encoding/json"
	"slices"
	"testing"

	"github.com/maruel/genai"
	"github.com/maruel/genai/providers/openaicompatible"
)

func TestChatRequest(t *testing.T) {
	t.Run("Init", func(t *testing.T) {
		t.Run("valid", func(t *testing.T) {
			for _, tc := range []struct {
				force genai.ToolCallRequest
				want  string
			}{{genai.ToolCallAny, "auto"}, {genai.ToolCallRequired, "required"}, {genai.ToolCallNone, "none"}} {
				t.Run(tc.want, func(t *testing.T) {
					msgs := genai.Messages{
						genai.NewTextMessage("Weather?"),
						{Replies: []genai.Reply{{Text: "Checking."}, {ToolCall: genai.ToolCall{ID: "a", Name: "weather", Arguments: `{}`}}, {ToolCall: genai.ToolCall{ID: "b", Name: "weather", Arguments: `{}`}}}},
						{ToolCallResults: []genai.ToolCallResult{{ID: "a", Name: "weather", Result: "sunny"}, {ID: "b", Name: "weather", Result: "rainy"}}},
					}
					var req openaicompatible.ChatRequest
					opts := &genai.GenOptionTools{Force: tc.force, Tools: []genai.ToolDef{{Name: "weather", Description: "Get weather", InputSchemaOverride: genai.JSONSchema(`{"type":"object"}`)}}}
					if err := req.Init(msgs, "model", opts, &genai.GenOptionText{SystemPrompt: "Help."}); err != nil {
						t.Fatal(err)
					}
					b, err := json.Marshal(&req)
					if err != nil {
						t.Fatal(err)
					}
					var wire struct {
						ToolChoice string `json:"tool_choice"`
						Tools      []struct {
							Type     string
							Function map[string]json.RawMessage
						}
						Messages []struct{ Role, Content string }
					}
					var raw map[string]json.RawMessage
					if err := json.Unmarshal(b, &raw); err != nil {
						t.Fatal(err)
					}
					if err := json.Unmarshal(b, &wire); err != nil {
						t.Fatal(err)
					}
					if wire.ToolChoice != tc.want || len(wire.Tools) != 1 || wire.Tools[0].Type != "function" {
						t.Fatalf("request: %s", b)
					}
					if string(wire.Tools[0].Function["parameters"]) != `{"type":"object"}` {
						t.Fatalf("schema: %s", b)
					}
					if _, ok := wire.Tools[0].Function["strict"]; ok {
						t.Fatalf("strict is not portable: %s", b)
					}
					if _, ok := raw["parallel_tool_calls"]; ok {
						t.Fatalf("parallel flag is not portable: %s", b)
					}
					if len(req.Messages) != 5 || req.Messages[2].Role != "assistant" || len(req.Messages[2].ToolCalls) != 2 {
						t.Fatalf("messages: %s", b)
					}
					if req.Messages[3].Role != "tool" || req.Messages[3].ToolCallID != "a" || req.Messages[4].ToolCallID != "b" {
						t.Fatalf("results: %s", b)
					}
					if wire.Messages[2].Content != "Checking." || wire.Messages[3].Content != "sunny" {
						t.Fatalf("content: %s", b)
					}
				})
			}
		})
		t.Run("error", func(t *testing.T) {
			var req openaicompatible.ChatRequest
			if err := req.Init(genai.Messages{genai.NewTextMessage("Hello")}, "", &genai.GenOptionTools{Force: genai.ToolCallRequired}); err == nil {
				t.Fatal("expected invalid tool options")
			}
		})
	})
}

func TestChatResponse(t *testing.T) {
	t.Run("ToResult", func(t *testing.T) {
		t.Run("error", func(t *testing.T) {
			for name, body := range map[string]string{
				"trailing_json":   `{"choices":[{"message":{"tool_calls":[{"function":{"name":"weather","arguments":"{}{}"}}]}}]}`,
				"unknown_content": `{"choices":[{"message":{"content":[{"type":"image","text":"lost"}]}}]}`,
			} {
				t.Run(name, func(t *testing.T) {
					var resp openaicompatible.ChatResponse
					if err := json.Unmarshal([]byte(body), &resp); err != nil {
						t.Fatal(err)
					}
					if _, err := resp.ToResult(); err == nil {
						t.Fatal("expected conversion error")
					}
				})
			}
		})
		t.Run("valid", func(t *testing.T) {
			for _, body := range []string{
				`{"choices":[{"message":{"role":"assistant","content":null,"tool_calls":[{"id":"a","type":"function","function":{"name":"weather","arguments":"{}","extension":true}}]},"finish_reason":"tool_calls"}]}`,
				`{"message":{"role":"assistant","tool_calls":[{"function":{"name":"weather","arguments":"{}"}}]},"finish_reason":"stop"}`,
				`{"role":"assistant","tool_calls":[{"function":{"name":"weather","arguments":"{}"}}]}`,
				`{"choices":[{"message":{"content":"","tool_calls":[{"function":{"name":"weather","arguments":"{}"}}]},"finish_reason":"stop"}]}`,
				`{"choices":[{"message":{"tool_calls":[{"function":{"name":"weather","arguments":""}}]},"finish_reason":"tool_calls"}]}`,
			} {
				var resp openaicompatible.ChatResponse
				if err := json.Unmarshal([]byte(body), &resp); err != nil {
					t.Fatal(err)
				}
				res, err := resp.ToResult()
				if err != nil {
					t.Fatal(err)
				}
				if len(res.Replies) != 1 || res.Replies[0].ToolCall.Name != "weather" || res.Replies[0].ToolCall.Arguments != `{}` || res.Usage.FinishReason != genai.FinishedToolCalls {
					t.Fatalf("result: %#v", res)
				}
			}
		})
	})
}

func TestProcessStream(t *testing.T) {
	t.Run("error", func(t *testing.T) {
		for name, body := range map[string]string{
			"invalid_arguments": `{"choices":[{"delta":{"tool_calls":[{"function":{"name":"weather","arguments":"{"}}]}}]}`,
			"multiple_choices":  `{"choices":[{},{}]}`,
			"unexpected_role":   `{"choices":[{"delta":{"role":"user"}}]}`,
			"unknown_content":   `{"choices":[{"delta":{"content":[{"type":"image","text":"lost"}]}}]}`,
		} {
			t.Run(name, func(t *testing.T) {
				var chunk openaicompatible.ChatStreamChunkResponse
				if err := json.Unmarshal([]byte(body), &chunk); err != nil {
					t.Fatal(err)
				}
				fragments, finish := openaicompatible.ProcessStream(slices.Values([]openaicompatible.ChatStreamChunkResponse{chunk}))
				for range fragments {
					t.Error("unexpected fragment")
				}
				if _, _, err := finish(); err == nil {
					t.Fatal("expected stream error")
				}
			})
		}
	})
	t.Run("valid", func(t *testing.T) {
		for _, tc := range []struct {
			name    string
			packets []string
			want    []genai.ToolCall
		}{
			{"repeated_id", []string{
				`{"choices":[{"delta":{"tool_calls":[{"id":"a","function":{"name":"weather","arguments":"{"}}]}}]}`,
				`{"choices":[{"delta":{"tool_calls":[{"id":"a","function":{"name":"weather","arguments":"}"}}]},"finish_reason":"tool_calls"}]}`,
			}, []genai.ToolCall{{ID: "a", Name: "weather", Arguments: `{}`}}},
			{"serial_calls", []string{
				`{"choices":[{"delta":{"tool_calls":[{"id":"a","function":{"name":"weather","arguments":"{"}}]}}]}`,
				`{"choices":[{"delta":{"tool_calls":[{"function":{"arguments":"}"}}]}}]}`,
				`{"choices":[{"delta":{"tool_calls":[{"id":"b","function":{"name":"weather","arguments":"{}"}}]}}]}`,
			}, []genai.ToolCall{{ID: "a", Name: "weather", Arguments: `{}`}, {ID: "b", Name: "weather", Arguments: `{}`}}},
			{"whole_replaces_delta", []string{
				`{"choices":[{"delta":{"tool_calls":[{"id":"a","function":{"name":"weather","arguments":"{"}}]}}]}`,
				`{"choices":[{"tool_calls":[{"id":"a","function":{"name":"weather","arguments":"{}"}}],"finish_reason":"stop"}]}`,
			}, []genai.ToolCall{{ID: "a", Name: "weather", Arguments: `{}`}}},
			{"choice_calls", []string{`{"choices":[{"delta":{},"tool_calls":[{"id":"a","function":{"name":"weather","arguments":"{}"}}],"finish_reason":"function_call"}]}`}, []genai.ToolCall{{ID: "a", Name: "weather", Arguments: `{}`}}},
			{"message_envelope", []string{`{"delta":{"message":{"role":"assistant","tool_calls":[{"function":{"name":"weather","arguments":"{}"}}]}},"finish_reason":"stop"}`}, []genai.ToolCall{{Name: "weather", Arguments: `{}`}}},
		} {
			t.Run(tc.name, func(t *testing.T) {
				chunks := make([]openaicompatible.ChatStreamChunkResponse, len(tc.packets))
				for i, p := range tc.packets {
					if err := json.Unmarshal([]byte(p), &chunks[i]); err != nil {
						t.Fatal(err)
					}
				}
				fragments, finish := openaicompatible.ProcessStream(slices.Values(chunks))
				var got []genai.ToolCall
				for f := range fragments {
					if !f.ToolCall.IsZero() {
						got = append(got, f.ToolCall)
					}
				}
				u, _, err := finish()
				if err != nil {
					t.Fatal(err)
				}
				if !slices.EqualFunc(got, tc.want, func(a, b genai.ToolCall) bool { return a.ID == b.ID && a.Name == b.Name && a.Arguments == b.Arguments }) {
					t.Fatalf("calls: %#v; want %#v", got, tc.want)
				}
				if u.FinishReason != genai.FinishedToolCalls {
					t.Fatalf("finish: %q", u.FinishReason)
				}
			})
		}
	})
}
