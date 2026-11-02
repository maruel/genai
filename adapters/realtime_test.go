// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Tests for the realtime tool-call adapter.

package adapters_test

import (
	"context"
	"reflect"
	"strings"
	"testing"

	"github.com/maruel/genai"
	"github.com/maruel/genai/adapters"
	"github.com/maruel/genai/internal/realtimetest"
)

func TestRealtimeWithToolCallLoop(t *testing.T) {
	info := genai.RealtimeSessionInfo{Capabilities: genai.RealtimeCapabilities{ToolResults: true}}
	s := realtimetest.NewFakeSession(info)
	t.Cleanup(func() {
		if err := s.Close(); err != nil {
			t.Error(err)
		}
	})
	oldTools := []genai.ToolDef{{Name: "old", Description: "Old", Callback: func(context.Context, *struct{}) (string, error) { return "old", nil }}}
	newTools := []genai.ToolDef{{Name: "lookup", Description: "Lookup", Callback: func(_ context.Context, in *struct {
		Query string `json:"query"`
	}) (string, error) {
		return "result for " + in.Query, nil
	}}}
	s.Reply = func(_ int, in genai.RealtimeInput) []genai.RealtimeEvent {
		switch v := in.(type) {
		case genai.RealtimeUpdate:
			return nil
		case genai.RealtimeInputContent:
			if v.ToolResults != nil {
				return []genai.RealtimeEvent{
					{Kind: genai.RealtimeEventContentText, ResponseID: "resp_2", Content: genai.RealtimeContent{Text: "done"}},
					{Kind: genai.RealtimeEventGenerationDone, ResponseID: "resp_2"},
					{Kind: genai.RealtimeEventTurnDone, ResponseID: "resp_2"},
				}
			}
			return []genai.RealtimeEvent{
				{Kind: genai.RealtimeEventContentToolCalls, ResponseID: "resp_1", Content: genai.RealtimeContent{ToolCalls: []genai.ToolCall{{ID: "call_1", Name: "lookup", Arguments: `{"query":"violet seven"}`}}}},
				{Kind: genai.RealtimeEventGenerationDone, ResponseID: "resp_1"},
			}
		default:
			return nil
		}
	}
	loop := adapters.RealtimeWithToolCallLoop(s, oldTools)
	newToolsCopy := newTools
	if err := loop.Send(t.Context(), genai.RealtimeUpdate{Tools: &newToolsCopy}); err != nil {
		t.Fatal(err)
	}
	if err := loop.Send(t.Context(), genai.RealtimeInputContent{Text: "look up", Generate: true}); err != nil {
		t.Fatal(err)
	}
	var kinds []genai.RealtimeEventKind
	for e, err := range loop.Receive(t.Context()) {
		if err != nil {
			t.Fatal(err)
		}
		kinds = append(kinds, e.Kind)
		if e.Kind == genai.RealtimeEventTurnDone {
			break
		}
	}
	if !reflect.DeepEqual(kinds, []genai.RealtimeEventKind{genai.RealtimeEventContentToolCalls, genai.RealtimeEventGenerationDone, genai.RealtimeEventContentText, genai.RealtimeEventGenerationDone, genai.RealtimeEventTurnDone}) {
		t.Fatalf("events: %v", kinds)
	}
	if len(s.Sent) != 3 {
		t.Fatalf("sent: %#v", s.Sent)
	}
	got, ok := s.Sent[2].(genai.RealtimeInputContent)
	if !ok || !got.Generate || len(got.ToolResults) != 1 || got.ToolResults[0].Result != "result for violet seven" {
		t.Fatalf("tool results: %#v", s.Sent[2])
	}
}

func TestRealtimeWithToolCallLoopMissingCallback(t *testing.T) {
	s := realtimetest.NewFakeSession(genai.RealtimeSessionInfo{Capabilities: genai.RealtimeCapabilities{ToolResults: true}})
	t.Cleanup(func() {
		if err := s.Close(); err != nil {
			t.Error(err)
		}
	})
	s.Reply = func(_ int, in genai.RealtimeInput) []genai.RealtimeEvent {
		if v, ok := in.(genai.RealtimeInputContent); ok && v.Text != "" {
			return []genai.RealtimeEvent{{Kind: genai.RealtimeEventContentToolCalls, Content: genai.RealtimeContent{ToolCalls: []genai.ToolCall{{ID: "call_1", Name: "lookup", Arguments: `{}`}}}}}
		}
		return nil
	}
	loop := adapters.RealtimeWithToolCallLoop(s, []genai.ToolDef{{Name: "lookup", Description: "Lookup", InputSchemaOverride: genai.JSONSchema(`{"type":"object"}`)}})
	if err := loop.Send(t.Context(), genai.RealtimeInputContent{Text: "look up", Generate: true}); err != nil {
		t.Fatal(err)
	}
	gotCall := false
	for e, err := range loop.Receive(t.Context()) {
		if err != nil {
			if !strings.Contains(err.Error(), "has no callback") {
				t.Fatalf("missing callback: %v", err)
			}
			if !gotCall {
				t.Fatal("tool call event was not yielded")
			}
			return
		}
		if e.Kind == genai.RealtimeEventContentToolCalls {
			gotCall = true
		}
	}
	t.Fatal("receive ended without an error")
}
