// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Tests for Anthropic response DTOs.

package anthropic_test

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/maruel/genai"
	"github.com/maruel/genai/providers/anthropic"
)

func TestContent(t *testing.T) {
	t.Run("FromReply", func(t *testing.T) {
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
	})
}

func TestChatResponse(t *testing.T) {
	t.Run("Diagnostics", func(t *testing.T) {
		for _, tc := range []struct {
			name        string
			raw         string
			diagnostics bool
			cacheMiss   bool
		}{
			{"absent", `null`, false, false},
			{"pending", `{"cache_miss_reason":null}`, true, false},
			{"cacheMiss", `{"cache_miss_reason":{"type":"messages_changed","cache_missed_input_tokens":42}}`, true, true},
		} {
			t.Run(tc.name, func(t *testing.T) {
				var r anthropic.ChatResponse
				d := json.NewDecoder(strings.NewReader(`{"diagnostics":` + tc.raw + `}`))
				d.DisallowUnknownFields()
				if err := d.Decode(&r); err != nil {
					t.Fatal(err)
				}
				if (r.Diagnostics != nil) != tc.diagnostics {
					t.Fatalf("diagnostics presence: got %v, want %v", r.Diagnostics != nil, tc.diagnostics)
				}
				if r.Diagnostics == nil {
					return
				}
				if (r.Diagnostics.CacheMissReason != nil) != tc.cacheMiss {
					t.Fatalf("cache miss presence: got %v, want %v", r.Diagnostics.CacheMissReason != nil, tc.cacheMiss)
				}
				if tc.cacheMiss && (r.Diagnostics.CacheMissReason.Type != "messages_changed" || r.Diagnostics.CacheMissReason.CacheMissedInputTokens != 42) {
					t.Fatalf("unexpected cache miss: %+v", r.Diagnostics.CacheMissReason)
				}
			})
		}
	})
}

func TestChatStreamChunkResponse(t *testing.T) {
	var r anthropic.ChatStreamChunkResponse
	d := json.NewDecoder(strings.NewReader(`{"type":"message_start","message":{"diagnostics":{"cache_miss_reason":{"type":"messages_changed","cache_missed_input_tokens":42}}}}`))
	d.DisallowUnknownFields()
	if err := d.Decode(&r); err != nil {
		t.Fatal(err)
	}
	if r.Message.Diagnostics == nil || r.Message.Diagnostics.CacheMissReason == nil || r.Message.Diagnostics.CacheMissReason.CacheMissedInputTokens != 42 {
		t.Fatalf("unexpected diagnostics: %+v", r.Message.Diagnostics)
	}
}

func TestModelsResponse(t *testing.T) {
	var r anthropic.ModelsResponse
	d := json.NewDecoder(strings.NewReader(`{"data":[{"id":"claude-sonnet-5-5","line":"sonnet","created_at":"2026-09-28T00:00:00Z"}]}`))
	d.DisallowUnknownFields()
	if err := d.Decode(&r); err != nil {
		t.Fatal(err)
	}
	if len(r.Data) != 1 || r.Data[0].Line != "sonnet" || r.Data[0].CreatedAt.IsZero() {
		t.Fatalf("unexpected models: %+v", r.Data)
	}
}
