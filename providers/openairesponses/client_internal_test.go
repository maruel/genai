// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Tests for client_internal.go

package openairesponses

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"

	"github.com/maruel/genai"
	"github.com/maruel/genai/base"
	"github.com/maruel/genai/internal"
	"github.com/maruel/genai/internal/internaltest"
)

func TestResponse(t *testing.T) {
	t.Run("accessPrograms", func(t *testing.T) {
		for _, tc := range []struct {
			name string
			raw  string
			want CyberAccessProgram
		}{
			{name: "red", raw: `{"cyber":"daybreak_red"}`, want: CyberAccessProgramDaybreakRed},
		} {
			t.Run(tc.name, func(t *testing.T) {
				var got Response
				if err := internal.UnmarshalJSON([]byte(`{"access_programs":`+tc.raw+`}`), &got); err != nil {
					t.Fatal(err)
				}
				if tc.want == "" {
					if got.AccessPrograms != nil {
						t.Fatalf("access programs = %+v, want nil", got.AccessPrograms)
					}
				} else if got.AccessPrograms == nil || got.AccessPrograms.Cyber != tc.want {
					t.Fatalf("access programs = %+v, want %q", got.AccessPrograms, tc.want)
				}
			})
		}
	})
}

func TestResponseChronology(t *testing.T) {
	msgs := genai.Messages{
		genai.NewTextMessage("first"),
		genai.NewTextMessage("correction"),
		{Replies: []genai.Reply{
			{Reasoning: "output-only summary"},
			{Text: "before"},
			{ToolCall: genai.ToolCall{ID: "A", Name: "status", Arguments: `{"task":3}`}},
			{Text: "between"},
			{ToolCall: genai.ToolCall{ID: "B", Name: "status", Arguments: `{"task":7}`}},
			{Text: "after"},
		}},
		genai.NewTextMessage("intervening"),
		{ToolCallResults: []genai.ToolCallResult{{ID: "B", Name: "status", Result: "waiting"}}},
		{ToolCallResults: []genai.ToolCallResult{{ID: "A", Name: "status", Result: "running"}}},
	}
	t.Run("full_history", func(t *testing.T) {
		before, err := json.Marshal(msgs)
		if err != nil {
			t.Fatal(err)
		}
		var req Response
		if err := req.Init(msgs, "test"); err != nil {
			t.Fatal(err)
		}
		if len(req.Input) != 10 {
			t.Fatalf("input count=%d want=10", len(req.Input))
		}
		for _, tc := range []struct {
			idx        int
			text, role string
		}{
			{0, "first", "user"}, {1, "correction", "user"}, {2, "before", "assistant"},
			{4, "between", "assistant"}, {6, "after", "assistant"}, {7, "intervening", "user"},
		} {
			m := &req.Input[tc.idx]
			if m.Role != tc.role || len(m.Content) != 1 || m.Content[0].Text != tc.text {
				t.Fatalf("input[%d]=%+v", tc.idx, m)
			}
		}
		if req.Input[3].CallID != "A" || req.Input[3].Name != "status" || req.Input[3].Arguments != `{"task":3}` || req.Input[5].CallID != "B" || req.Input[5].Arguments != `{"task":7}` {
			t.Fatalf("calls reordered or changed: %+v", req.Input)
		}
		if req.Input[8].CallID != "B" || req.Input[8].Output != "waiting" || req.Input[9].CallID != "A" || req.Input[9].Output != "running" {
			t.Fatalf("results changed: %+v", req.Input)
		}
		c := &Client{impl: base.Provider[*ErrorResponse, *Response, *Response, ResponseStreamChunkResponse]{ProviderBase: base.ProviderBase[*ErrorResponse]{Model: "test"}}}
		internaltest.CleanupCloser(t, c)
		w := &WebSocketConn{client: c}
		ws, err := w.buildRequest(msgs)
		if err != nil {
			t.Fatal(err)
		}
		if ws.PreviousResponseID != "" {
			t.Fatalf("invented previous response ID: %q", ws.PreviousResponseID)
		}
		if diff := cmp.Diff(req.Input, ws.Input); diff != "" {
			t.Fatal(diff)
		}
		after, err := json.Marshal(msgs)
		if err != nil {
			t.Fatal(err)
		}
		if !bytes.Equal(before, after) {
			t.Fatal("mutated input")
		}
	})
	t.Run("bookkeeping_only", func(t *testing.T) {
		in := genai.Messages{{Replies: []genai.Reply{{Text: "text", Opaque: map[string]any{opaqueResponseID: "resp", opaqueSentMsgs: float64(0)}}, emitMeta("resp", 0)}}}
		got := deltaMessages(in, 0)
		if len(got) != 1 || len(got[0].Replies) != 1 || got[0].Replies[0].Opaque != nil {
			t.Fatalf("got=%+v", got)
		}
		if len(in[0].Replies[0].Opaque) != 2 || len(in[0].Replies) != 2 {
			t.Fatal("mutated bookkeeping")
		}
	})
	t.Run("unsupported_metadata", func(t *testing.T) {
		for _, reply := range []genai.Reply{
			{Text: "text"}, {Reasoning: "summary"},
			{ToolCall: genai.ToolCall{ID: "A", Name: "status", Arguments: `{}`}}, {},
		} {
			reply.Opaque = map[string]any{opaqueResponseID: "resp", opaqueSentMsgs: float64(0), "signature": []byte("preserve")}
			in := genai.Messages{{Replies: []genai.Reply{reply}}}
			got := deltaMessages(in, 0)
			if len(got) != 1 || len(got[0].Replies[0].Opaque) != 1 || len(in[0].Replies[0].Opaque) != 3 {
				t.Fatal("lost metadata or mutated input")
			}
			var req Response
			if err := req.Init(got, "test"); err == nil {
				t.Fatal("silently accepted unsupported metadata")
			}
		}
	})
}

func TestResponseStructuralValidationBeforeDelta(t *testing.T) {
	msgs := genai.Messages{
		{Requests: []genai.Request{{Text: "invalid union"}}, Replies: []genai.Reply{{Text: "assistant"}}},
		{Replies: []genai.Reply{emitMeta("resp_previous", 1)}},
		genai.NewTextMessage("new valid delta"),
	}
	c := &Client{}
	internaltest.CleanupCloser(t, c)
	if _, err := c.GenSync(t.Context(), msgs); err == nil || !strings.Contains(err.Error(), "exactly one") {
		t.Fatalf("GenSync error=%v", err)
	}
	fragments, finish := c.GenStream(t.Context(), msgs)
	for range fragments {
	}
	if _, err := finish(); err == nil || !strings.Contains(err.Error(), "exactly one") {
		t.Fatalf("GenStream error=%v", err)
	}
	w := &WebSocketConn{client: c}
	if _, err := w.buildRequest(msgs); err == nil || !strings.Contains(err.Error(), "exactly one") {
		t.Fatalf("WebSocket error=%v", err)
	}
}
