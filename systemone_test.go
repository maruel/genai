// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Tests for shared System One content, questions, requests and answers.

package genai

import (
	"encoding/json"
	"io"
	"strings"
	"testing"
)

func TestParseDecisionContent(t *testing.T) {
	data := []struct {
		name    string
		in      string
		want    DecisionContent
		wantErr string
	}{
		{"text", `"hi"`, Text("hi"), ""},
		{"object", `{"a":1}`, Object{"a": 1}, ""},
		{"array", `["a",1]`, Array{"a", 1}, ""},
		{"empty", ``, nil, "is empty"},
		{"number", `1`, nil, "expected a string, a JSON object or a JSON array, got 1"},
		{"bool", `true`, nil, "expected a string, a JSON object or a JSON array, got true"},
		{"malformed text", `"hi`, nil, "unexpected EOF"},
		{"malformed object", `{"a":}`, nil, "invalid character"},
		{"malformed array", `[1,`, nil, "unexpected EOF"},
	}
	for _, line := range data {
		t.Run(line.name, func(t *testing.T) {
			got, err := ParseDecisionContent([]byte(line.in))
			if line.wantErr == "" {
				if err != nil {
					t.Fatal(err)
				}
				if got == nil {
					t.Fatalf("expected %#v, got nil", line.want)
				}
				switch w := line.want.(type) {
				case Text:
					if v, ok := got.(Text); !ok || v != w {
						t.Fatalf("want %#v, got %#v", line.want, got)
					}
				case Object:
					if v, ok := got.(Object); !ok || len(v) != len(w) {
						t.Fatalf("want %#v, got %#v", line.want, got)
					}
				case Array:
					if v, ok := got.(Array); !ok || len(v) != len(w) {
						t.Fatalf("want %#v, got %#v", line.want, got)
					}
				}
				return
			}
			if err == nil {
				t.Fatalf("expected error, got %#v", got)
			}
			if !strings.Contains(err.Error(), line.wantErr) {
				t.Fatalf("want %q, got %q", line.wantErr, err)
			}
			if got != nil {
				t.Errorf("expected no content, got %#v", got)
			}
		})
	}
}

func TestSystemOneRequest(t *testing.T) {
	t.Run("ReadState", func(t *testing.T) {
		qs := Questions{"q": {Type: QuestionNoul, Instructions: Text("yes?")}}
		t.Run("valid", func(t *testing.T) {
			for _, raw := range []string{`{"z":1,"a":2}`, `[1,{"x":2}]`, `"state"`} {
				r := SystemOneRequest{Questions: qs, Docs: []Doc{{Filename: "state.json", Src: strings.NewReader(raw)}}}
				state, docs, err := r.ReadState(int64(len(raw)))
				if err != nil {
					t.Fatal(err)
				}
				b, err := json.Marshal(state)
				if err != nil || string(b) != raw || len(docs) != 0 {
					t.Fatalf("state=%s, err=%v; want %s", b, err, raw)
				}
				if r.State != nil || len(r.Docs) != 1 {
					t.Fatalf("input was modified: %+v", r)
				}
			}
		})
		t.Run("error", func(t *testing.T) {
			for _, raw := range []string{`{`, `null`, `42`, `true`, `{"x":1} {"y":2}`} {
				r := SystemOneRequest{Questions: qs, Docs: []Doc{{Filename: "state.json", Src: strings.NewReader(raw)}}}
				if state, docs, err := r.ReadState(64); err == nil || state != nil || docs != nil {
					t.Fatalf("accepted invalid state %q", raw)
				}
			}
			for _, limit := range []int64{-1, 0, 1} {
				r := SystemOneRequest{Questions: qs, Docs: []Doc{{Filename: "state.json", Src: strings.NewReader(`{}`)}}}
				if state, docs, err := r.ReadState(limit); err == nil || state != nil || docs != nil {
					t.Fatalf("accepted document limit %d", limit)
				}
			}
		})
		t.Run("attachments stay lazy", func(t *testing.T) {
			src := strings.NewReader("image")
			if _, err := src.Seek(1, io.SeekStart); err != nil {
				t.Fatal(err)
			}
			r := SystemOneRequest{State: Text("look"), Questions: qs, Docs: []Doc{{Filename: "image.png", Src: src}}}
			state, docs, err := r.ReadState(1)
			if err != nil || state != Text("look") || len(docs) != 1 || docs[0].Src != src || &docs[0] != &r.Docs[0] {
				t.Fatalf("state=%v, err=%v", state, err)
			}
			pos, err := src.Seek(0, io.SeekCurrent)
			if err != nil || pos != 1 {
				t.Fatalf("attachment read: position=%d, err=%v", pos, err)
			}
		})
	})
	t.Run("Validate", func(t *testing.T) {
		t.Run("document types are provider-owned", func(t *testing.T) {
			for _, filename := range []string{"image.png", "report.pdf", "clip.wav", "attachment.custom"} {
				r := SystemOneRequest{State: Text("state"), Questions: Questions{"yes": {Type: QuestionNoul, Instructions: Text("yes?")}}, Docs: []Doc{{Filename: filename, Src: strings.NewReader("data")}}}
				if err := r.Validate(); err != nil {
					t.Fatalf("%s rejected by shared validation: %v", filename, err)
				}
			}
		})
		q := Questions{
			"yes":  {Type: QuestionNoul, Noul: &NoulCriteria{}},
			"pick": {Type: QuestionChoice, Choice: map[string]DecisionContent{"only": nil}},
			"rate": {Type: QuestionScore, Score: []DecisionContent{nil}},
		}
		r := SystemOneRequest{State: Object{"ticket": 42}, Questions: q}
		if err := r.Validate(); err != nil {
			t.Fatalf("provider-independent shapes rejected: %v", err)
		}
		r.State = Object{"invalid": make(chan int)}
		if err := r.Validate(); err == nil {
			t.Fatal("expected invalid state error")
		}
	})
}

func TestQuestions(t *testing.T) {
	t.Run("Validate", func(t *testing.T) {
		qs := Questions{"nil": nil}
		if err := qs.Validate(); err == nil {
			t.Fatal("accepted nil question")
		}
	})
	t.Run("MarshalJSON", func(t *testing.T) {
		qs := Questions{"dynamic.name": {Type: QuestionNoul, Instructions: Text("yes?")}}
		b, err := json.Marshal(qs)
		if err != nil || string(b) != `{"dynamic.name":{"type":"noul","instructions":"yes?"}}` {
			t.Fatalf("unexpected question JSON: %s, %v", b, err)
		}
	})
}

func TestAnswers(t *testing.T) {
	t.Run("MarshalJSON", func(t *testing.T) {
		as := Answers{"dynamic.name": {Type: QuestionNoul, Noul: 0}}
		b, err := json.Marshal(as)
		if err != nil || string(b) != `{"dynamic.name":{"type":"noul","noul":0}}` {
			t.Fatalf("unexpected answer JSON: %s, %v", b, err)
		}
	})
}

func TestDecisionUsage(t *testing.T) {
	t.Run("MarshalJSON", func(t *testing.T) {
		for _, tc := range []struct {
			usage DecisionUsage
			want  string
		}{
			{DecisionUsage{InputTokens: 42, OutputTokens: 3}, `{"input_tokens":42,"output_tokens":3}`},
			{DecisionUsage{InputTokens: 42, OutputTokens: 3, ReasoningTokens: 2}, `{"input_tokens":42,"output_tokens":3,"reasoning_tokens":2}`},
		} {
			b, err := json.Marshal(tc.usage)
			if err != nil || string(b) != tc.want {
				t.Fatalf("got %s, %v; want %s", b, err, tc.want)
			}
		}
	})
}

func TestSystemOneResponse(t *testing.T) {
	t.Run("ValidateQuestions", func(t *testing.T) {
		qs := Questions{"yes": {Type: QuestionNoul, Instructions: Text("yes?")}}
		for _, tc := range []struct {
			name    string
			answers Answers
			usage   DecisionUsage
			valid   bool
		}{
			{name: "valid zero", answers: Answers{"yes": {Type: QuestionNoul}}, valid: true},
			{name: "negative reasoning", answers: Answers{"yes": {Type: QuestionNoul}}, usage: DecisionUsage{ReasoningTokens: -1}},
			{name: "missing answers"},
			{name: "nil answer", answers: Answers{"yes": nil}},
			{name: "nil extra answer", answers: Answers{"yes": {Type: QuestionNoul}, "extra": nil}},
			{name: "missing question", answers: Answers{"other": {Type: QuestionNoul}}},
			{name: "wrong type", answers: Answers{"yes": {Type: QuestionChoice}}},
		} {
			t.Run(tc.name, func(t *testing.T) {
				r := SystemOneResponse{Answers: tc.answers, Usage: tc.usage}
				if err := r.ValidateQuestions(&qs); (err == nil) != tc.valid {
					t.Fatalf("valid=%t, got %v", tc.valid, err)
				}
			})
		}
	})
}

func TestQuestion(t *testing.T) {
	t.Run("Validate", func(t *testing.T) {
		t.Run("valid", func(t *testing.T) {
			for name, q := range map[string]Question{
				"noul criteria without instructions": {Type: QuestionNoul, Noul: &NoulCriteria{}},
				"choice without instructions":        {Type: QuestionChoice, Choice: map[string]DecisionContent{"only": nil}},
				"one undescribed score level":        {Type: QuestionScore, Score: []DecisionContent{nil}},
			} {
				t.Run(name, func(t *testing.T) {
					if err := q.Validate(); err != nil {
						t.Fatal(err)
					}
				})
			}
		})
		t.Run("error", func(t *testing.T) {
			q := Question{Type: QuestionChoice, Noul: &NoulCriteria{}, Choice: map[string]DecisionContent{"only": nil}}
			if err := q.Validate(); err == nil {
				t.Fatal("expected inconsistent union error")
			}
		})
	})
}

func TestNoulCriteria(t *testing.T) {
	t.Run("Validate", func(t *testing.T) {
		t.Run("valid", func(t *testing.T) {
			c := NoulCriteria{}
			if err := c.Validate(); err != nil {
				t.Fatal(err)
			}
		})
		t.Run("error", func(t *testing.T) {
			c := NoulCriteria{True: Object{"invalid": make(chan int)}}
			if err := c.Validate(); err == nil {
				t.Fatal("expected invalid content error")
			}
		})
	})
}
