// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Tests for System One qualification and failed live requests.

package smoke_test

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/maruel/genai"
	"github.com/maruel/genai/providers/ollama"
	"github.com/maruel/genai/smoke"
)

func TestRunSystemOne(t *testing.T) {
	for _, tc := range []struct {
		name    string
		status  int
		wantErr bool
	}{{"valid", 200, false}, {"unauthorized", 401, true}, {"server failure", 500, true}, {"no decision support", 400, true}} {
		t.Run(tc.name, func(t *testing.T) {
			s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if tc.status != 200 {
					w.WriteHeader(tc.status)
					if _, err := w.Write([]byte(`{"error":"failed"}`)); err != nil {
						t.Error(err)
					}
					return
				}
				// Questions carry interface values, so decode their wire discriminator separately.
				var raw struct {
					Questions map[string]struct{ Type string }
					Images    []string
				}
				if err := json.NewDecoder(r.Body).Decode(&raw); err != nil {
					t.Error(err)
				}
				if len(raw.Images) > 0 {
					w.WriteHeader(http.StatusNotImplemented)
					if _, err := w.Write([]byte(`{"error":"images unsupported"}`)); err != nil {
						t.Error(err)
					}
					return
				}
				answers := genai.Answers{}
				for id, q := range raw.Questions {
					a := &genai.Answer{Type: genai.QuestionType(q.Type)}
					switch a.Type {
					case genai.QuestionNoul:
						a.Noul = 0.9
					case genai.QuestionChoice:
						a.Choice = "billing"
						a.Probabilities = map[string]float64{"billing": 0.9, "support": 0.1}
						a.Confidence = 0.8
					case genai.QuestionScore:
						a.Score = 0.8
						a.Probabilities = map[string]float64{"0": 0.2, "1": 0.8}
						a.Legend = genai.ScoreLegend{"0": genai.Text("Can wait"), "1": genai.Text("Needs attention today")}
						a.Confidence = 0.5
					}
					answers[id] = a
				}
				out := ollama.SystemOneResponse{Model: "model", SystemOneResponse: genai.SystemOneResponse{Answers: answers, Usage: genai.DecisionUsage{InputTokens: 10}}}
				if err := json.NewEncoder(w).Encode(&out); err != nil {
					t.Error(err)
				}
			}))
			t.Cleanup(s.Close)
			c, err := ollama.New(t.Context(), genai.ProviderOptionRemote(s.URL), genai.ProviderOptionModel("model"))
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() { _ = c.Close() })
			sc, u, err := smoke.RunSystemOne(t.Context(), func(string) genai.Provider { return c })
			if (err != nil) != tc.wantErr {
				t.Fatalf("scenario=%+v err=%v", sc, err)
			}
			if tc.wantErr {
				return
			}
			if _, ok := sc.Out[genai.ModalityDecision]; !ok {
				t.Fatal("decision output was not qualified")
			}
			if _, ok := sc.Out[genai.ModalityText]; ok {
				t.Fatal("decision model was qualified as text output")
			}
			if sc.SystemOne == nil || !sc.SystemOne.Noul || !sc.SystemOne.Choice || !sc.SystemOne.Score || !sc.SystemOne.Object || !sc.SystemOne.Array {
				t.Fatalf("unexpected functionality: %+v", sc.SystemOne)
			}
			if _, ok := sc.In[genai.ModalityImage]; ok {
				t.Fatal("unsupported image input was qualified")
			}
			if u.InputTokens != 50 || u.TotalTokens != 50 {
				t.Fatalf("unexpected usage: %+v", u)
			}
		})
	}
}
