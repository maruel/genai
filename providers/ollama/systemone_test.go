// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Tests for native Ollama System One decisions and inline images.

package ollama_test

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"slices"
	"strings"
	"testing"

	"github.com/maruel/genai"
	"github.com/maruel/genai/providers/ollama"
	"github.com/maruel/genai/scoreboard"
)

func TestSystemOneRequest(t *testing.T) {
	q := genai.Questions{"yes": {Type: genai.QuestionNoul, Instructions: genai.Text("Yes?")}}
	t.Run("From", func(t *testing.T) {
		data, err := scoreboard.TestdataFiles.ReadFile("testdata/image.png")
		if err != nil {
			t.Fatal(err)
		}
		r := ollama.SystemOneRequest{Model: "clef-flash", KeepAlive: "2m"}
		if err := r.From(&genai.SystemOneRequest{State: genai.Object{"ticket": 42}, Questions: q, Docs: []genai.Doc{{Filename: "image.png", Src: bytes.NewReader(data)}}}); err != nil {
			t.Fatal(err)
		}
		if err := r.Validate(); err != nil {
			t.Fatal(err)
		}
		if r.Model != "clef-flash" || r.KeepAlive != "2m" || len(r.Images) != 1 || r.Images[0] != base64.StdEncoding.EncodeToString(data) {
			t.Fatalf("unexpected request: %+v", r)
		}
	})
	t.Run("Validate", func(t *testing.T) {
		for _, tc := range []struct {
			name   string
			model  string
			images []string
		}{{"no model", "", nil}, {"invalid base64", "clef-flash", []string{"data:image/png;base64,aW1n"}}, {"empty image", "clef-flash", []string{""}}} {
			t.Run(tc.name, func(t *testing.T) {
				r := ollama.SystemOneRequest{Model: tc.model, State: genai.Text("state"), Questions: q, Images: tc.images}
				if err := r.Validate(); err == nil {
					t.Fatal("expected error")
				}
			})
		}
	})
	t.Run("From errors", func(t *testing.T) {
		for _, tc := range []struct {
			name string
			doc  genai.Doc
		}{{"URL", genai.Doc{URL: "https://example.com/image.png"}}, {"text", genai.Doc{Filename: "state.txt", Src: strings.NewReader("hello")}}} {
			t.Run(tc.name, func(t *testing.T) {
				var r ollama.SystemOneRequest
				if err := r.From(&genai.SystemOneRequest{State: genai.Text("state"), Questions: q, Docs: []genai.Doc{tc.doc}}); err == nil {
					t.Fatal("expected error")
				}
			})
		}
	})
}

func TestClientSystemOne(t *testing.T) {
	q := genai.Questions{"yes": {Type: genai.QuestionNoul, Instructions: genai.Text("Is this about billing?")}}
	for _, tc := range []struct {
		name    string
		body    string
		wantErr bool
	}{{"valid", `{"model":"clef-flash","answers":{"yes":{"type":"noul","noul":0.9}},"usage":{"input_tokens":42,"output_tokens":0}}`, false}, {"missing answer", `{"model":"clef-flash","answers":{},"usage":{"input_tokens":42,"output_tokens":0}}`, true}, {"invalid probability", `{"model":"clef-flash","answers":{"yes":{"type":"noul","noul":2}},"usage":{"input_tokens":42,"output_tokens":0}}`, true}} {
		t.Run(tc.name, func(t *testing.T) {
			s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.URL.Path != "/v1/systemone" {
					t.Errorf("unexpected path %s", r.URL.Path)
				}
				var req struct {
					Model     string
					State     map[string]int
					Questions json.RawMessage
				}
				if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
					t.Error(err)
				}
				if req.Model != "clef-flash" || req.State["ticket"] != 42 {
					t.Errorf("unexpected request: %+v", req)
				}
				if _, err := w.Write([]byte(tc.body)); err != nil {
					t.Error(err)
				}
			}))
			t.Cleanup(s.Close)
			c, err := ollama.New(t.Context(), genai.ProviderOptionRemote(s.URL), genai.ProviderOptionModel("clef-flash"))
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() { _ = c.Close() })
			res, err := c.SystemOne(t.Context(), &genai.SystemOneRequest{State: genai.Object{"ticket": 42}, Questions: q})
			if (err != nil) != tc.wantErr {
				t.Fatalf("response=%+v err=%v", res, err)
			}
			if !tc.wantErr && (res.Answers["yes"].Noul != 0.9 || res.Usage.InputTokens != 42) {
				t.Fatalf("unexpected response: %+v", res)
			}
		})
	}
}

func TestNew(t *testing.T) {
	for _, tc := range []struct {
		name      string
		model     string
		mod       genai.Modality
		installed []string
		want      string
	}{
		{"capability", "", genai.ModalityDecision, nil, ""},
		{"automatic decision", string(genai.ModelGood), genai.ModalityDecision, []string{"qwen3.5:2b", "clef-flash:latest"}, "clef-flash:latest"},
		{"clef family prefix", "custom/clef-next:latest", genai.ModalityDecision, nil, "custom/clef-next:latest"},
		{"explicit decision", "clef-flash:latest", genai.ModalityDecision, nil, "clef-flash:latest"},
		{"automatic text", string(genai.ModelGood), genai.ModalityText, []string{"clef-flash:latest", "qwen3.5:2b"}, "qwen3.5:2b"},
		{"decision default", string(genai.ModelGood), genai.ModalityDecision, []string{"qwen3.5:2b"}, "clef-flash:latest"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			models := make(genai.ProviderOptionPreloadedModels, 0, len(tc.installed))
			for _, m := range tc.installed {
				models = append(models, &ollama.Model{Name: m})
			}
			opts := []genai.ProviderOption{genai.ProviderOptionRemote("http://localhost:0"), genai.ProviderOptionModalities{tc.mod}}
			if len(models) != 0 {
				opts = append(opts, models)
			}
			if tc.model != "" {
				opts = append(opts, genai.ProviderOptionModel(tc.model))
			}
			c, err := ollama.New(t.Context(), opts...)
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() { _ = c.Close() })
			if !slices.Equal(c.OutputModalities(), genai.Modalities{tc.mod}) {
				t.Fatalf("unexpected output modalities: %v", c.OutputModalities())
			}
			if c.ModelID() != tc.want {
				t.Fatalf("unexpected selected model: %s", c.ModelID())
			}
		})
	}
}
