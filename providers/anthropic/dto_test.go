// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Tests for Anthropic response DTOs.

package anthropic_test

import (
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
