// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Tests for the llama.cpp System One decision API.

package llamacpp_test

import (
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"

	"github.com/maruel/genai"
	"github.com/maruel/genai/base"
	"github.com/maruel/genai/providers/llamacpp"
)

func TestClientSystemOne(t *testing.T) {
	q := struct {
		Billing llamacpp.Noul   `json:"billing"`
		Route   llamacpp.Choice `json:"route"`
		Urgency llamacpp.Score  `json:"urgency"`
	}{
		Billing: llamacpp.Noul{Instructions: llamacpp.Text("Is this about billing?")},
		Route:   llamacpp.Choice{Instructions: llamacpp.Text("Which team?"), Criteria: map[string]llamacpp.DecisionContent{"billing": nil, "support": nil}},
		Urgency: llamacpp.Score{Instructions: llamacpp.Text("How urgent?"), Criteria: []llamacpp.DecisionContent{llamacpp.Text("low"), llamacpp.Text("high")}},
	}
	const response = `{"model":"kev","answers":{"billing":{"type":"noul","noul":0.9},"route":{"type":"choice","choice":"billing","confidence":0.8,"probabilities":{"billing":0.9,"support":0.1}},"urgency":{"type":"score","score":0.75,"confidence":0.5,"legend":{"0":"low","1":"high"},"probabilities":{"0":0.25,"1":0.75}}},"usage":{"input_tokens":42,"output_tokens":0}}`
	t.Run("GenSystemOneRaw", func(t *testing.T) {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			var req struct {
				Model     string                     `json:"model"`
				State     map[string]int             `json:"state"`
				Images    []string                   `json:"images"`
				Questions map[string]json.RawMessage `json:"questions"`
			}
			if r.URL.Path != "/v1/systemone" {
				t.Errorf("unexpected path %s", r.URL.Path)
			}
			if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
				t.Error(err)
			}
			if req.Model != "kev" || req.State["ticket"] != 42 || len(req.Images) != 1 || len(req.Questions) != 3 {
				t.Errorf("unexpected request: %+v", req)
			}
			w.Header().Set("Content-Type", "application/json")
			if _, err := w.Write([]byte(response)); err != nil {
				t.Error(err)
			}
		}))
		t.Cleanup(srv.Close)
		c, err := llamacpp.New(t.Context(), genai.ProviderOptionRemote(srv.URL))
		if err != nil {
			t.Fatal(err)
		}
		questions, err := llamacpp.QuestionsFrom(&q)
		if err != nil {
			t.Fatal(err)
		}
		out := &llamacpp.SystemOneResponse{}
		if err := c.GenSystemOneRaw(t.Context(), &llamacpp.SystemOneRequest{
			Model: "kev", State: llamacpp.Object{"ticket": 42}, Questions: questions, Images: []string{"data:image/png;base64,aW1hZ2U="},
		}, out); err != nil {
			t.Fatal(err)
		}
		if out.Model != "kev" || out.Answers["billing"].Noul != 0.9 {
			t.Errorf("unexpected response: %+v", out)
		}
	})
	for _, stream := range []bool{false, true} {
		name := "GenSync"
		if stream {
			name = "GenStream"
		}
		t.Run(name, func(t *testing.T) {
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.Method != http.MethodPost || r.URL.Path != "/v1/systemone" {
					t.Errorf("unexpected request: %s %s", r.Method, r.URL.Path)
				}
				var req struct {
					State     string                     `json:"state"`
					Model     string                     `json:"model"`
					Questions map[string]json.RawMessage `json:"questions"`
				}
				if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
					t.Error(err)
				}
				if req.State != "Charged twice" || req.Model != "" || len(req.Questions) != 3 {
					t.Errorf("unexpected request: %+v", req)
				}
				w.Header().Set("Content-Type", "application/json")
				if _, err := w.Write([]byte(response)); err != nil {
					t.Error(err)
				}
			}))
			t.Cleanup(srv.Close)
			c, err := llamacpp.New(t.Context(), genai.ProviderOptionRemote(srv.URL), &llamacpp.ProviderOption{SystemOne: true})
			if err != nil {
				t.Fatal(err)
			}
			if mod := c.OutputModalities(); len(mod) != 1 || mod[0] != genai.ModalityText {
				t.Errorf("unexpected output modalities: %v", mod)
			}
			msgs := genai.Messages{genai.NewTextMessage("Charged twice")}
			opt := &genai.GenOptionText{DecodeAs: &q}
			var res genai.Result
			if stream {
				seq, finish := c.GenStream(t.Context(), msgs, opt)
				n := 0
				for reply := range seq {
					n++
					if reply.Text == "" {
						t.Error("empty streamed answer")
					}
				}
				if n != 1 {
					t.Errorf("got %d fragments", n)
				}
				res, err = finish()
			} else {
				res, err = c.GenSync(t.Context(), msgs, opt)
			}
			if err != nil {
				t.Fatal(err)
			}
			if err := res.Decode(&q); err != nil {
				t.Fatal(err)
			}
			if q.Billing.Probability != 0.9 || q.Route.Label != "billing" || q.Urgency.Value != 0.75 {
				t.Errorf("unexpected answers: %+v", q)
			}
			if res.Usage.InputTokens != 42 || res.Usage.OutputTokens != 0 || res.Usage.TotalTokens != 42 {
				t.Errorf("unexpected usage: %+v", res.Usage)
			}
		})
	}
	t.Run("error", func(t *testing.T) {
		for _, tc := range []struct {
			name, body string
			status     int
		}{
			{"unsupported model", `{"error":{"code":501,"message":"This model is not a decision model","type":"not_supported_error"}}`, 501},
			{"missing answers", `{"model":"kev","answers":{},"usage":{"input_tokens":1,"output_tokens":0}}`, 200},
			{"missing question", `{"model":"kev","answers":{"billing":{"type":"noul","noul":1}},"usage":{"input_tokens":1,"output_tokens":0}}`, 200},
			{"wrong answer type", `{"model":"kev","answers":{"billing":{"type":"choice","choice":"billing"}},"usage":{"input_tokens":1,"output_tokens":0}}`, 200},
			{"malformed response", `{`, 200},
		} {
			t.Run(tc.name, func(t *testing.T) {
				srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
					w.Header().Set("Content-Type", "application/json")
					w.WriteHeader(tc.status)
					if _, err := w.Write([]byte(tc.body)); err != nil {
						t.Error(err)
					}
				}))
				t.Cleanup(srv.Close)
				c, err := llamacpp.New(t.Context(), genai.ProviderOptionRemote(srv.URL), &llamacpp.ProviderOption{SystemOne: true})
				if err != nil {
					t.Fatal(err)
				}
				_, err = c.GenSync(t.Context(), genai.Messages{genai.NewTextMessage("x")}, &genai.GenOptionText{DecodeAs: &q})
				if err == nil {
					t.Fatal("expected an error")
				}
				if tc.status == 501 {
					var api *llamacpp.ErrorResponse
					if !errors.As(err, &api) || api.ErrorVal.Code != 501 {
						t.Errorf("unexpected error: %v", err)
					}
				}
			})
		}
	})
	t.Run("unsupported option", func(t *testing.T) {
		c, err := llamacpp.New(t.Context(), &llamacpp.ProviderOption{SystemOne: true})
		if err != nil {
			t.Fatal(err)
		}
		_, err = c.GenSync(t.Context(), genai.Messages{genai.NewTextMessage("x")}, &genai.GenOptionText{DecodeAs: &q, MaxTokens: 1})
		var unsupported *base.ErrNotSupported
		if !errors.As(err, &unsupported) {
			t.Fatalf("unexpected error: %v", err)
		}
	})
}

func TestSystemOneRequest(t *testing.T) {
	t.Run("From", func(t *testing.T) {
		for _, tc := range []struct {
			name   string
			reqs   []genai.Request
			state  string
			images int
		}{
			{"text", []genai.Request{{Text: "hello"}}, `"hello"`, 0},
			{"structured", []genai.Request{{Doc: genai.Doc{Filename: "state.json", Src: strings.NewReader(" \n{\"ticket\":42}\n")}}}, `{"ticket":42}`, 0},
			{"multiple", []genai.Request{{Text: "one"}, {Text: "two"}}, `["one","two"]`, 0},
			{"image", []genai.Request{{Text: "look", Doc: genai.Doc{Filename: "image.png", Src: strings.NewReader("image")}}}, `"look"`, 1},
			{"image only", []genai.Request{{Doc: genai.Doc{Filename: "image.png", Src: strings.NewReader("image")}}}, `""`, 1},
		} {
			t.Run(tc.name, func(t *testing.T) {
				r := &llamacpp.SystemOneRequest{}
				if err := r.From(&genai.Message{Requests: tc.reqs}); err != nil {
					t.Fatal(err)
				}
				b, err := json.Marshal(r.State)
				if err != nil {
					t.Fatal(err)
				}
				if string(b) != tc.state || len(r.Images) != tc.images {
					t.Errorf("state=%s images=%v", b, r.Images)
				}
				if tc.images > 0 && r.Images[0] != "data:image/png;base64,aW1hZ2U=" {
					t.Errorf("unexpected image %s", r.Images[0])
				}
			})
		}
		t.Run("error", func(t *testing.T) {
			for _, msg := range []genai.Message{
				{},
				{Replies: []genai.Reply{{Text: "answer"}}},
				{Requests: []genai.Request{{Doc: genai.Doc{Filename: "state.json"}}}},
				{Requests: []genai.Request{{Text: "keep this context", Doc: genai.Doc{Filename: "state.json", Src: strings.NewReader(`{"ticket":42}`)}}}},
				{Requests: []genai.Request{{Doc: genai.Doc{URL: "https://example.com/image.png"}}}},
				{Requests: []genai.Request{{Doc: genai.Doc{Filename: "state.json", Src: strings.NewReader(`{`)}}}},
				{Requests: []genai.Request{{Doc: genai.Doc{Filename: "audio.wav", Src: strings.NewReader("audio")}}}},
			} {
				if err := (&llamacpp.SystemOneRequest{}).From(&msg); err == nil {
					t.Errorf("expected error for %+v", msg)
				}
			}
		})
	})
	t.Run("Validate", func(t *testing.T) {
		for _, tc := range []struct {
			name string
			q    llamacpp.Question
		}{
			{"missing instructions", llamacpp.Question{Type: llamacpp.QuestionNoul, Noul: &llamacpp.NoulCriteria{True: llamacpp.Text("yes")}}},
			{"one score level", llamacpp.Question{Type: llamacpp.QuestionScore, Instructions: llamacpp.Text("rate"), Score: []llamacpp.DecisionContent{llamacpp.Text("low")}}},
			{"too many score levels", llamacpp.Question{Type: llamacpp.QuestionScore, Instructions: llamacpp.Text("rate"), Score: []llamacpp.DecisionContent{llamacpp.Text("0"), llamacpp.Text("1"), llamacpp.Text("2"), llamacpp.Text("3"), llamacpp.Text("4"), llamacpp.Text("5"), llamacpp.Text("6"), llamacpp.Text("7"), llamacpp.Text("8"), llamacpp.Text("9"), llamacpp.Text("10")}}},
		} {
			t.Run(tc.name, func(t *testing.T) {
				r := &llamacpp.SystemOneRequest{State: llamacpp.Text("state"), Questions: llamacpp.Questions{"q": tc.q}}
				if err := r.Validate(); err == nil {
					t.Fatal("expected error")
				}
			})
		}
	})
}

func TestQuestionsFrom(t *testing.T) {
	t.Run("valid", func(t *testing.T) {
		q := struct {
			Rating llamacpp.Score `json:"rating"`
			hidden string         `json:"-"`
		}{Rating: llamacpp.Score{Instructions: llamacpp.Text("Rate this"), Criteria: []llamacpp.DecisionContent{nil, llamacpp.Text("high")}}}
		questions, err := llamacpp.QuestionsFrom(&q)
		if err != nil {
			t.Fatal(err)
		}
		b, err := json.Marshal(questions)
		if err != nil {
			t.Fatal(err)
		}
		if !strings.Contains(string(b), `"criteria":[null,"high"]`) || len(questions) != 1 {
			t.Errorf("unexpected questions: %s", b)
		}
		res := &llamacpp.SystemOneResponse{}
		if err := json.Unmarshal([]byte(`{"model":"kev","answers":{"rating":{"type":"score","score":0.75,"confidence":0.5,"legend":{"0":null,"1":"high"},"probabilities":{"0":0.25,"1":0.75}}},"usage":{"input_tokens":42,"output_tokens":0}}`), res); err != nil {
			t.Fatal(err)
		}
		r, err := res.ToResult()
		if err != nil {
			t.Fatal(err)
		}
		if err := r.Decode(&q); err != nil {
			t.Fatal(err)
		}
		if q.Rating.Value != 0.75 || q.Rating.Legend["0"] != nil || q.Rating.Legend["1"] != llamacpp.Text("high") {
			t.Errorf("unexpected rating: %+v", q.Rating)
		}
	})
	t.Run("error", func(t *testing.T) {
		var nilQuestionnaire *struct{ Q llamacpp.Noul }
		for _, in := range []any{
			nil, nilQuestionnaire,
			&struct{ Q string }{},
			&struct{ Q llamacpp.Noul }{},
			reflect.New(reflect.StructOf([]reflect.StructField{
				{Name: "A", Type: reflect.TypeFor[llamacpp.Noul](), Tag: `json:"q"`},
				{Name: "B", Type: reflect.TypeFor[llamacpp.Noul](), Tag: `json:"q"`},
			})).Interface(),
		} {
			if _, err := llamacpp.QuestionsFrom(in); err == nil {
				t.Errorf("expected error for %T", in)
			}
		}
	})
}
