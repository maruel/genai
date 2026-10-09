// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Tests for the llama.cpp System One decision API.

package llamacpp_test

import (
	"slices"
	"strings"
	"testing"

	"github.com/maruel/genai"
	"github.com/maruel/genai/providers/llamacpp"
)

func TestSystemOneRequest(t *testing.T) {
	t.Run("From", func(t *testing.T) {
		t.Run("error", func(t *testing.T) {
			for _, in := range []genai.SystemOneRequest{
				{},
				{State: genai.Text("state"), Docs: []genai.Doc{{Filename: "state.json"}}},
				{State: genai.Text("state"), Docs: []genai.Doc{{URL: "https://example.com/image.png"}}},
				{State: genai.Text("state"), Docs: []genai.Doc{{Filename: "audio.wav", Src: strings.NewReader("audio")}}},
			} {
				in.Questions = genai.Questions{"q": {Type: genai.QuestionNoul, Instructions: genai.Text("yes?")}}
				if err := (&llamacpp.SystemOneRequest{}).From(&in); err == nil {
					t.Errorf("expected error for %+v", in)
				}
			}
		})
	})
	t.Run("Validate", func(t *testing.T) {
		for _, tc := range []struct {
			name string
			q    genai.Question
		}{
			{"missing instructions", genai.Question{Type: genai.QuestionNoul, Noul: &genai.NoulCriteria{True: genai.Text("yes")}}},
		} {
			t.Run(tc.name, func(t *testing.T) {
				r := &llamacpp.SystemOneRequest{State: genai.Text("state"), Questions: genai.Questions{"q": &tc.q}}
				if err := r.Validate(); err == nil {
					t.Fatal("expected error")
				}
			})
		}
	})
}

func TestNew(t *testing.T) {
	for _, model := range []string{"", string(genai.ModelGood), "custom/Decision-Model.gguf"} {
		t.Run(model, func(t *testing.T) {
			opts := []genai.ProviderOption{genai.ProviderOptionRemote("http://localhost:0"), genai.ProviderOptionModalities{genai.ModalityDecision}, genai.ProviderOptionPreloadedModels{&llamacpp.Model{OpenAI: llamacpp.ModelOpenAI{ID: "/models/Decision-Model.gguf"}}}}
			if model != "" {
				opts = append(opts, genai.ProviderOptionModel(model))
			}
			c, err := llamacpp.New(t.Context(), opts...)
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() { _ = c.Close() })
			if !slices.Equal(c.OutputModalities(), genai.Modalities{genai.ModalityDecision}) {
				t.Fatalf("unexpected output modalities: %v", c.OutputModalities())
			}
			if model == string(genai.ModelGood) && c.ModelID() != "Decision-Model.gguf" {
				t.Fatalf("unexpected selected model: %s", c.ModelID())
			}
		})
	}
}
