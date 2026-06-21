// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Tests for client_internal.go

package openairesponses

import (
	"testing"

	"github.com/maruel/genai"
)

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
