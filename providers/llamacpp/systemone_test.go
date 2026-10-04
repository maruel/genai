// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Tests for the llama.cpp System One decision API.

package llamacpp_test

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/maruel/genai"
	"github.com/maruel/genai/providers/llamacpp"
)

func TestSystemOneRequest(t *testing.T) {
	t.Run("From", func(t *testing.T) {
		for _, tc := range []struct {
			name   string
			in     genai.SystemOneRequest
			state  string
			images int
		}{
			{"text", genai.SystemOneRequest{State: genai.Text("hello")}, `"hello"`, 0},
			{"structured", genai.SystemOneRequest{State: genai.Object{"ticket": 42}}, `{"ticket":42}`, 0},
			{"array", genai.SystemOneRequest{State: genai.Array{"one", "two"}}, `["one","two"]`, 0},
			{"image", genai.SystemOneRequest{State: genai.Text("look"), Docs: []genai.Doc{{Filename: "image.png", Src: strings.NewReader("image")}}}, `"look"`, 1},
			{"image only", genai.SystemOneRequest{State: genai.Text(""), Docs: []genai.Doc{{Filename: "image.png", Src: strings.NewReader("image")}}}, `""`, 1},
		} {
			t.Run(tc.name, func(t *testing.T) {
				r := &llamacpp.SystemOneRequest{Model: "kev"}
				tc.in.Questions = genai.Questions{"q": {Type: genai.QuestionNoul, Instructions: genai.Text("yes?")}}
				if err := r.From(&tc.in); err != nil {
					t.Fatal(err)
				}
				b, err := json.Marshal(r.State)
				if err != nil {
					t.Fatal(err)
				}
				if string(b) != tc.state || len(r.Images) != tc.images || r.Model != "kev" || len(r.Questions) != 1 {
					t.Errorf("state=%s images=%v", b, r.Images)
				}
				if tc.images > 0 && r.Images[0] != "data:image/png;base64,aW1hZ2U=" {
					t.Errorf("unexpected image %s", r.Images[0])
				}
			})
		}
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
		t.Run("deterministic joined errors", func(t *testing.T) {
			r := llamacpp.SystemOneRequest{State: genai.Text("state"), Questions: genai.Questions{
				"z": {Type: genai.QuestionNoul, Noul: &genai.NoulCriteria{True: genai.Text("yes")}},
				"a": {Type: genai.QuestionScore, Score: []genai.DecisionContent{genai.Text("only")}},
			}}
			want := "question \"a\": field Instructions: is required\nquestion \"a\": field Score: 2 to 10 levels are required\nquestion \"z\": field Instructions: is required"
			for range 20 {
				if err := r.Validate(); err == nil || err.Error() != want {
					t.Fatalf("got %v, want %s", err, want)
				}
			}
		})
		for _, tc := range []struct {
			name string
			q    genai.Question
		}{
			{"missing instructions", genai.Question{Type: genai.QuestionNoul, Noul: &genai.NoulCriteria{True: genai.Text("yes")}}},
			{"one score level", genai.Question{Type: genai.QuestionScore, Instructions: genai.Text("rate"), Score: []genai.DecisionContent{genai.Text("low")}}},
			{"too many score levels", genai.Question{Type: genai.QuestionScore, Instructions: genai.Text("rate"), Score: []genai.DecisionContent{genai.Text("0"), genai.Text("1"), genai.Text("2"), genai.Text("3"), genai.Text("4"), genai.Text("5"), genai.Text("6"), genai.Text("7"), genai.Text("8"), genai.Text("9"), genai.Text("10")}}},
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
