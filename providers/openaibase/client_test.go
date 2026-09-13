// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Tests for the shared OpenAI-compatible client operations.

package openaibase

import (
	"encoding/json"
	"testing"

	"github.com/maruel/genai"
	"github.com/maruel/genai/base"
)

func TestClient(t *testing.T) {
	t.Run("SelectBestTextModel", func(t *testing.T) {
		models := []genai.Model{
			&Model{ID: "gpt-5.6", Created: base.TimeS(300)},
			&Model{ID: "gpt-5.6-sol", Created: base.TimeS(300)},
			&Model{ID: "gpt-5.6-terra", Created: base.TimeS(300)},
			&Model{ID: "gpt-5.6-luna", Created: base.TimeS(300)},
			&Model{ID: "gpt-5.4-mini", Created: base.TimeS(200)},
			&Model{ID: "gpt-5.4-nano", Created: base.TimeS(200)},
			&Model{ID: "gpt-5.6-sol-2026-07-09", Created: base.TimeS(400)},
			&Model{ID: "gpt-5.6-pro", Created: base.TimeS(400)},
			&Model{ID: "gpt-5.6-codex", Created: base.TimeS(400)},
			&Model{ID: "gpt-transcribe", Created: base.TimeS(500)},
			&Model{ID: "gpt-live-transcribe", Created: base.TimeS(500)},
			&Model{ID: "gpt-live-1", Created: base.TimeS(600)},
		}
		c := &Client{PreloadedModels: models}
		data := []struct {
			name string
			in   genai.ProviderOptionModel
			want string
		}{
			{name: "sota", in: genai.ModelSOTA, want: "gpt-5.6-sol"},
			{name: "good", in: genai.ModelGood, want: "gpt-5.6-terra"},
			{name: "cheap", in: genai.ModelCheap, want: "gpt-5.6-luna"},
			{name: "default", in: "", want: "gpt-5.6-terra"},
		}
		for _, tc := range data {
			t.Run(tc.name, func(t *testing.T) {
				got, err := c.SelectBestTextModel(t.Context(), string(tc.in))
				if err != nil {
					t.Fatal(err)
				}
				if got != tc.want {
					t.Fatalf("got %q, want %q", got, tc.want)
				}
			})
		}
	})
	t.Run("SelectBestImageModel", func(t *testing.T) {
		models := []genai.Model{
			&Model{ID: "gpt-image-1-mini", Created: base.TimeS(200)},
			&Model{ID: "gpt-image-2", Created: base.TimeS(300)},
			&Model{ID: "gpt-image-2-2026-04-21", Created: base.TimeS(400)},
			&Model{ID: "gpt-image-2.5-sunburst", Created: base.TimeS(500)},
		}
		c := &Client{PreloadedModels: models}
		data := []struct {
			name string
			in   genai.ProviderOptionModel
			want string
		}{
			{name: "sota", in: genai.ModelSOTA, want: "gpt-image-2.5-sunburst"},
			{name: "good", in: genai.ModelGood, want: "gpt-image-2.5-sunburst"},
			{name: "cheap", in: genai.ModelCheap, want: "gpt-image-1-mini"},
		}
		for _, tc := range data {
			t.Run(tc.name, func(t *testing.T) {
				got, err := c.SelectBestImageModel(t.Context(), string(tc.in))
				if err != nil {
					t.Fatal(err)
				}
				if got != tc.want {
					t.Fatalf("got %q, want %q", got, tc.want)
				}
			})
		}
	})
}

func TestImageResponse(t *testing.T) {
	var got ImageResponse
	if err := json.Unmarshal([]byte(`{"data":[{"generation_id":"imggen_123"}]}`), &got); err != nil {
		t.Fatal(err)
	}
	if len(got.Data) != 1 {
		t.Fatalf("got %d images, want 1", len(got.Data))
	}
	if got.Data[0].GenerationID != "imggen_123" {
		t.Fatalf("got generation ID %q, want %q", got.Data[0].GenerationID, "imggen_123")
	}
}
