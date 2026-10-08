// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Tests for shared OpenAI embedding wire unions and validation.

package openaibase_test

import (
	"encoding/json"
	"testing"

	"github.com/maruel/genai/providers/openaibase"
)

func TestEmbeddingInput(t *testing.T) {
	t.Run("MarshalJSON", func(t *testing.T) {
		for _, tc := range []embeddingInputWireCase{
			{"text", openaibase.EmbeddingInput{Text: "hello"}, `"hello"`},
			{"tokens", openaibase.EmbeddingInput{Tokens: []int64{15339, 1917}}, `[15339,1917]`},
			{"token batch", openaibase.EmbeddingInput{TokenSequences: [][]int64{{15339, 1917}, {9906}}}, `[[15339,1917],[9906]]`},
		} {
			t.Run(tc.name, func(t *testing.T) {
				b, err := json.Marshal(&tc.in)
				if err != nil {
					t.Fatal(err)
				}
				if string(b) != tc.want {
					t.Fatalf("got %s, want %s", b, tc.want)
				}
				// Both request values and pointers must use the native input union.
				for _, req := range []any{openaibase.EmbeddingRequest{Model: "model", Input: tc.in}, &openaibase.EmbeddingRequest{Model: "model", Input: tc.in}} {
					b, err := json.Marshal(req)
					if err != nil {
						t.Fatal(err)
					}
					var obj embeddingInputEnvelope
					if err := json.Unmarshal(b, &obj); err != nil {
						t.Fatal(err)
					}
					if string(obj.Input) != tc.want {
						t.Fatalf("request input = %s, want %s", obj.Input, tc.want)
					}
				}
			})
		}
	})
	t.Run("Validate", func(t *testing.T) {
		t.Run("error", func(t *testing.T) {
			for _, tc := range []embeddingInputErrorCase{
				{"missing", openaibase.EmbeddingInput{}},
				{"conflicting", openaibase.EmbeddingInput{Text: "hello", Tokens: []int64{1}}},
				{"empty text batch", openaibase.EmbeddingInput{Texts: []string{}}},
				{"empty text", openaibase.EmbeddingInput{Texts: []string{""}}},
				{"empty tokens", openaibase.EmbeddingInput{Tokens: []int64{}}},
				{"negative token", openaibase.EmbeddingInput{Tokens: []int64{-1}}},
				{"empty token batch", openaibase.EmbeddingInput{TokenSequences: [][]int64{}}},
				{"empty sequence", openaibase.EmbeddingInput{TokenSequences: [][]int64{{}}}},
				{"negative batch token", openaibase.EmbeddingInput{TokenSequences: [][]int64{{-1}}}},
			} {
				t.Run(tc.name, func(t *testing.T) {
					if _, err := json.Marshal(&tc.in); err == nil {
						t.Fatal("expected error")
					}
				})
			}
		})
	})
}

func TestEmbeddingRequest(t *testing.T) {
	t.Run("Validate", func(t *testing.T) {
		t.Run("error", func(t *testing.T) {
			for _, tc := range []embeddingRequestErrorCase{
				{"missing model", openaibase.EmbeddingRequest{Input: openaibase.EmbeddingInput{Text: "hello"}}},
				{"negative dimensions", openaibase.EmbeddingRequest{Model: "model", Dimensions: -1, Input: openaibase.EmbeddingInput{Text: "hello"}}},
			} {
				t.Run(tc.name, func(t *testing.T) {
					if err := tc.in.Validate(); err == nil {
						t.Fatal("expected error")
					}
				})
			}
		})
	})
}

type embeddingInputWireCase struct {
	name string
	in   openaibase.EmbeddingInput
	want string
}

type embeddingInputEnvelope struct {
	Input json.RawMessage `json:"input"`
}

type embeddingInputErrorCase struct {
	name string
	in   openaibase.EmbeddingInput
}

type embeddingRequestErrorCase struct {
	name string
	in   openaibase.EmbeddingRequest
}
