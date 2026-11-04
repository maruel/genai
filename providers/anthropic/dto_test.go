// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Tests for Anthropic response DTOs.

package anthropic_test

import (
	"encoding/json"
	"slices"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/maruel/genai"
	"github.com/maruel/genai/providers/anthropic"
)

func TestContent(t *testing.T) {
	t.Run("FromReply", func(t *testing.T) {
		t.Run("thinking_replay", func(t *testing.T) {
			native := anthropic.Content{Type: anthropic.ContentThinking, Thinking: "signed reasoning", Signature: []byte("signed")}
			replies, err := native.To()
			if err != nil {
				t.Fatal(err)
			}
			r := genai.Result{Message: genai.Message{Replies: replies}}
			b, err := json.Marshal(r.Message)
			if err != nil {
				t.Fatal(err)
			}
			if err := json.Unmarshal(b, &r.Message); err != nil {
				t.Fatal(err)
			}
			if len(r.Replies) != 1 || r.Replies[0].Reasoning != "signed reasoning" {
				t.Fatalf("hidden reply: %+v", r.Replies)
			}
			var got anthropic.Content
			if _, err := got.FromReply(&r.Replies[0]); err != nil {
				t.Fatal(err)
			}
			if diff := cmp.Diff(native, got); diff != "" {
				t.Fatal(diff)
			}
		})
		t.Run("thinking_signature", func(t *testing.T) {
			in := genai.Reply{Reasoning: "thinking", Opaque: map[string]any{"signature": []byte("signed")}}
			var got anthropic.Content
			skip, err := got.FromReply(&in)
			if err != nil {
				t.Fatal(err)
			}
			if skip {
				t.Fatal("discarded signed thinking")
			}
			want := anthropic.Content{Type: anthropic.ContentThinking, Thinking: "thinking", Signature: []byte("signed")}
			if diff := cmp.Diff(want, got); diff != "" {
				t.Fatal(diff)
			}
		})
		t.Run("error", func(t *testing.T) {
			for _, v := range []any{"%%%", true} {
				var got anthropic.Content
				if _, err := got.FromReply(&genai.Reply{Reasoning: "check", Opaque: map[string]any{"signature": v}}); err == nil {
					t.Fatal("accepted invalid signature")
				}
			}
		})
	})
}

func TestRedactedThinkingReplay(t *testing.T) {
	want := []anthropic.Content{
		{Type: anthropic.ContentThinking, Thinking: "check", Signature: []byte("sig")},
		{Type: anthropic.ContentRedactedThinking, Data: "encrypted"},
	}
	for _, mode := range []string{"sync", "stream"} {
		t.Run(mode, func(t *testing.T) {
			var msg genai.Message
			if mode == "sync" {
				for i := range want {
					replies, err := want[i].To()
					if err != nil {
						t.Fatal(err)
					}
					msg.Replies = append(msg.Replies, replies...)
				}
			} else {
				chunks := make([]anthropic.ChatStreamChunkResponse, 0, 5)
				for _, body := range []string{
					`{"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":"check"}}`,
					`{"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"c2ln"}}`,
					`{"type":"content_block_stop","index":0}`,
					`{"type":"content_block_start","index":1,"content_block":{"type":"redacted_thinking","data":"encrypted"}}`,
					`{"type":"content_block_stop","index":1}`,
				} {
					var c anthropic.ChatStreamChunkResponse
					if err := json.Unmarshal([]byte(body), &c); err != nil {
						t.Fatal(err)
					}
					chunks = append(chunks, c)
				}
				seq, finish := anthropic.ProcessStream(slices.Values(chunks))
				for f := range seq {
					if err := msg.Accumulate(&f); err != nil {
						t.Fatal(err)
					}
				}
				if _, _, err := finish(); err != nil {
					t.Fatal(err)
				}
			}
			b, err := json.Marshal(msg)
			if err != nil {
				t.Fatal(err)
			}
			if err := json.Unmarshal(b, &msg); err != nil {
				t.Fatal(err)
			}
			got := make([]anthropic.Content, len(msg.Replies))
			for i := range msg.Replies {
				if _, err := got[i].FromReply(&msg.Replies[i]); err != nil {
					t.Fatal(err)
				}
			}
			if diff := cmp.Diff(want, got); diff != "" {
				t.Fatal(diff)
			}
		})
	}
}
