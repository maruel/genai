// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Tests for the message utility package.

package msgutil

import (
	"bytes"
	"testing"

	"github.com/maruel/genai"
)

func TestExtractOpaqueID(t *testing.T) {
	t.Run("found", func(t *testing.T) {
		msgs := genai.Messages{
			genai.NewTextMessage("hi"),
			{Replies: []genai.Reply{
				{Text: "Hello"},
				{Opaque: map[string]any{"session_id": "abc-123"}},
			}},
		}
		if got := ExtractOpaqueID(msgs, "session_id"); got != "abc-123" {
			t.Errorf("got %q, want %q", got, "abc-123")
		}
	})
	t.Run("not_found", func(t *testing.T) {
		msgs := genai.Messages{genai.NewTextMessage("hi")}
		if got := ExtractOpaqueID(msgs, "session_id"); got != "" {
			t.Errorf("got %q, want empty", got)
		}
	})
	t.Run("last_wins", func(t *testing.T) {
		msgs := genai.Messages{
			{Replies: []genai.Reply{{Opaque: map[string]any{"k": "old"}}}},
			{Replies: []genai.Reply{{Opaque: map[string]any{"k": "new"}}}},
		}
		if got := ExtractOpaqueID(msgs, "k"); got != "new" {
			t.Errorf("got %q, want %q", got, "new")
		}
	})
	t.Run("wrong_type", func(t *testing.T) {
		msgs := genai.Messages{
			{Replies: []genai.Reply{{Opaque: map[string]any{"k": 42}}}},
		}
		if got := ExtractOpaqueID(msgs, "k"); got != "" {
			t.Errorf("got %q, want empty", got)
		}
	})
}

func TestLastUserMsg(t *testing.T) {
	t.Run("valid", func(t *testing.T) {
		msgs := genai.Messages{genai.NewTextMessage("hello world")}
		got, err := LastUserMsg(msgs)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if len(got.Requests) == 0 || got.Requests[0].Text != "hello world" {
			t.Errorf("got %v, want text %q", got.Requests, "hello world")
		}
	})
	t.Run("no_user_message", func(t *testing.T) {
		msgs := genai.Messages{{Replies: []genai.Reply{{Text: "hi"}}}}
		if _, err := LastUserMsg(msgs); err == nil {
			t.Fatal("expected error for no user message")
		}
	})
	t.Run("last_user_wins", func(t *testing.T) {
		msgs := genai.Messages{
			genai.NewTextMessage("first"),
			{Replies: []genai.Reply{{Text: "response"}}},
			genai.NewTextMessage("second"),
		}
		got, err := LastUserMsg(msgs)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if len(got.Requests) == 0 || got.Requests[0].Text != "second" {
			t.Errorf("got %v, want text %q", got.Requests, "second")
		}
	})
	t.Run("empty_requests", func(t *testing.T) {
		msgs := genai.Messages{{Requests: nil}}
		if _, err := LastUserMsg(msgs); err == nil {
			t.Fatal("expected error for empty requests")
		}
	})
}

func TestWriteNDJSON(t *testing.T) {
	t.Run("valid", func(t *testing.T) {
		var buf bytes.Buffer
		if err := WriteNDJSON(&buf, map[string]string{"a": "b"}); err != nil {
			t.Fatal(err)
		}
		if got := buf.String(); got != "{\"a\":\"b\"}\n" {
			t.Errorf("got %q", got)
		}
	})
	t.Run("error", func(t *testing.T) {
		if err := WriteNDJSON(&bytes.Buffer{}, func() {}); err == nil {
			t.Fatal("expected marshal error")
		}
	})
}

func TestCheckExtraArgs(t *testing.T) {
	reserved := []string{"output-format", "p"}
	t.Run("valid", func(t *testing.T) {
		for _, args := range [][]string{
			nil,
			{"--add-dir", "/tmp"},
			{"--model=x", "value-p"},
			{"-c", "p=1"},
		} {
			if err := CheckExtraArgs(args, reserved...); err != nil {
				t.Errorf("%q: %v", args, err)
			}
		}
	})
	t.Run("error", func(t *testing.T) {
		for _, args := range [][]string{
			{""},
			{"-p"},
			{"--p=x"},
			{"--output-format", "json"},
			{"-output-format=json"},
		} {
			if err := CheckExtraArgs(args, reserved...); err == nil {
				t.Errorf("%q: expected error", args)
			}
		}
	})
}

func TestRejectModelMarker(t *testing.T) {
	t.Run("valid", func(t *testing.T) {
		for _, m := range []genai.ProviderOptionModel{"", "sonnet"} {
			if err := RejectModelMarker(m); err != nil {
				t.Errorf("%q: %v", m, err)
			}
		}
	})
	t.Run("error", func(t *testing.T) {
		for _, m := range []genai.ProviderOptionModel{genai.ModelCheap, genai.ModelGood, genai.ModelSOTA} {
			if err := RejectModelMarker(m); err == nil {
				t.Errorf("%q: expected error", m)
			}
		}
	})
}
