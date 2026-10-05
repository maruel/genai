// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Tests for Perplexity assistant document conversion.

package perplexity_test

import (
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"

	"github.com/maruel/genai"
	"github.com/maruel/genai/providers/perplexity"
)

func TestChatRequest(t *testing.T) {
	t.Run("Init", func(t *testing.T) {
		t.Run("assistant_document", func(t *testing.T) {
			msgs := genai.Messages{
				genai.NewTextMessage("user"),
				{Replies: []genai.Reply{{Doc: genai.Doc{Filename: "reply.txt", Src: strings.NewReader("assistant document")}}}},
			}
			var req perplexity.ChatRequest
			if err := req.Init(msgs, "test"); err != nil {
				t.Fatal(err)
			}
			want := []perplexity.Message{
				{Role: "user", Content: perplexity.Contents{{Type: "text", Text: "user"}}},
				{Role: "assistant", Content: perplexity.Contents{{Type: "text", Text: "assistant document"}}},
			}
			if diff := cmp.Diff(want, req.Messages); diff != "" {
				t.Fatal(diff)
			}
		})
	})
}
