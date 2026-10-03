// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Tests for client_internal.go

package openairesponses

import (
	"testing"

	"github.com/maruel/genai"
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
			{name: "null", raw: `null`},
			{name: "standard", raw: `{"cyber":"standard"}`, want: CyberAccessProgramStandard},
			{name: "blue", raw: `{"cyber":"daybreak_blue"}`, want: CyberAccessProgramDaybreakBlue},
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

func TestFindPrevMeta(t *testing.T) {
	t.Run("found", func(t *testing.T) {
		msgs := genai.Messages{
			genai.NewTextMessage("Hello"),
			{Replies: []genai.Reply{
				{Text: "Hi"},
				{Opaque: map[string]any{
					opaqueResponseID: "resp_abc",
					opaqueSentMsgs:   float64(3),
				}},
			}},
		}
		sentMsgs, respID := findPrevMeta(msgs)
		if respID != "resp_abc" {
			t.Errorf("respID = %q, want %q", respID, "resp_abc")
		}
		if sentMsgs != 3 {
			t.Errorf("sentMsgs = %d, want 3", sentMsgs)
		}
	})

	t.Run("not_found", func(t *testing.T) {
		msgs := genai.Messages{genai.NewTextMessage("Hello")}
		sentMsgs, respID := findPrevMeta(msgs)
		if respID != "" {
			t.Errorf("respID = %q, want empty", respID)
		}
		if sentMsgs != 0 {
			t.Errorf("sentMsgs = %d, want 0", sentMsgs)
		}
	})
}

func TestPrepareDelta(t *testing.T) {
	const inputMsgCount = 1
	msgs := genai.Messages{
		genai.NewTextMessage("What is the weather?"),
		{Replies: []genai.Reply{
			{ToolCall: genai.ToolCall{ID: "call_123", Name: "get_weather", Arguments: `{"city":"Paris"}`}},
			emitMeta("resp_123", inputMsgCount),
		}},
		{ToolCallResults: []genai.ToolCallResult{{
			ID:     "call_123",
			Name:   "get_weather",
			Result: "sunny",
		}}},
	}

	c := &Client{}
	internaltest.CleanupCloser(t, c)
	got, respID := c.prepareDelta(msgs, nil)
	if respID != "resp_123" {
		t.Errorf("response ID = %q, want %q", respID, "resp_123")
	}
	if len(got) != 1 {
		t.Fatalf("got %d messages, want only the tool result", len(got))
	}
	if len(got[0].ToolCallResults) != 1 {
		t.Errorf("got %#v, want one tool result", got[0])
	}
}

func TestStreamWithRespID(t *testing.T) {
	t.Run("completed", func(t *testing.T) {
		events := []ResponseStreamChunkResponse{
			{Type: ResponseCreated, Response: Response{ID: "resp_123"}},
			{Type: ResponseOutputTextDelta, Delta: "Hello"},
			{Type: ResponseCompleted, Response: Response{ID: "resp_456"}},
		}
		src := func(yield func(ResponseStreamChunkResponse) bool) {
			for _, e := range events {
				if !yield(e) {
					return
				}
			}
		}
		var respID string
		filtered := streamWithRespID(src, &respID)
		var got []ResponseStreamChunkResponse
		for pkt := range filtered {
			got = append(got, pkt)
		}
		if len(got) != 3 {
			t.Fatalf("got %d events, want 3", len(got))
		}
		if respID != "resp_456" {
			t.Errorf("respID = %q, want %q", respID, "resp_456")
		}
	})

	t.Run("failed", func(t *testing.T) {
		events := []ResponseStreamChunkResponse{
			{Type: ResponseCreated, Response: Response{ID: "resp_123"}},
			{Type: ResponseFailed, Response: Response{ID: "resp_789"}},
		}
		src := func(yield func(ResponseStreamChunkResponse) bool) {
			for _, e := range events {
				if !yield(e) {
					return
				}
			}
		}
		var respID string
		filtered := streamWithRespID(src, &respID)
		for range filtered {
		}
		if respID != "resp_789" {
			t.Errorf("respID = %q, want %q", respID, "resp_789")
		}
	})
}
