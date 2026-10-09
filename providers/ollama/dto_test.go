// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Tests for native Ollama embedding input shapes and validation.

package ollama_test

import (
	"encoding/json"
	"testing"

	"github.com/maruel/genai/internal"
	"github.com/maruel/genai/providers/ollama"
)

func TestEmbeddingRequest(t *testing.T) {
	t.Run("Validate", func(t *testing.T) {
		t.Run("error", func(t *testing.T) {
			for _, tc := range []embeddingRequestErrorCase{
				{"missing input", ollama.EmbeddingRequest{Model: "model"}},
				{"empty input", ollama.EmbeddingRequest{Model: "model", Input: []string{"hello", ""}}},
				{"negative dimensions", ollama.EmbeddingRequest{Model: "model", Input: []string{"hello"}, Dimensions: -1}},
			} {
				t.Run(tc.name, func(t *testing.T) {
					if err := tc.in.Validate(); err == nil {
						t.Fatal("accepted invalid request")
					}
				})
			}
		})
	})
}

// TestErrorResponse preserves decoder strictness for native and compatible errors.
func TestErrorResponse(t *testing.T) {
	t.Run("UnmarshalJSON", func(t *testing.T) {
		t.Run("error", func(t *testing.T) {
			var out ollama.ErrorResponse
			if err := json.Unmarshal([]byte(`[]`), &out); err == nil {
				t.Fatal("accepted an array error envelope")
			}
		})
	})

	old := internal.BeLenient
	defer func() { internal.BeLenient = old }()
	for _, lenient := range []bool{false, true} {
		internal.BeLenient = lenient
		for _, body := range []string{`{"error":"failed","unknown":true}`, `{"error":{"message":"failed","type":"api_error","param":null,"code":null,"unknown":true}}`} {
			var out ollama.ErrorResponse
			err := json.Unmarshal([]byte(body), &out)
			if (err == nil) != lenient {
				t.Fatalf("lenient %t: error %v", lenient, err)
			}
			if lenient && out.Error() != "failed" {
				t.Fatalf("error %+v", out)
			}
		}
	}
}

type embeddingRequestErrorCase struct {
	name string
	in   ollama.EmbeddingRequest
}
