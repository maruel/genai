// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Tests for the TypeSafe provider client.

package typesafe_test

import (
	"encoding/json"
	"errors"
	"math"
	"net/http"
	"net/http/httptest"
	"slices"
	"strings"
	"testing"

	"github.com/maruel/genai"
	"github.com/maruel/genai/base"
	"github.com/maruel/genai/internal"
	"github.com/maruel/genai/internal/internaltest"
	"github.com/maruel/genai/providers/typesafe"
	"github.com/maruel/genai/scoreboard"
)

func apiKey() string {
	if key := internaltest.GetEnv("TYPESAFE_API_KEY"); key != "" {
		return key
	}
	return "<insert_api_key_here>"
}

// questionnaire is the shape genai.GenOptionText.DecodeAs expects.
type questionnaire struct {
	Greeting typesafe.Noul   `json:"greeting"`
	Tone     typesafe.Choice `json:"tone"`
	Urgency  typesafe.Score  `json:"urgency"`
}

func text(s string) genai.Messages {
	return genai.Messages{genai.NewTextMessage(s)}
}

// doc returns messages holding a document, to pass the state as JSON.
func doc(filename, content string) genai.Messages {
	return genai.Messages{genai.Message{Requests: []genai.Request{{
		Doc: genai.Doc{Filename: filename, Src: strings.NewReader(content)},
	}}}}
}

func getClientInner(t *testing.T, opts []genai.ProviderOption, fn func(http.RoundTripper) http.RoundTripper) (*typesafe.Client, error) {
	if !slices.ContainsFunc(opts, func(o genai.ProviderOption) bool { _, ok := o.(genai.ProviderOptionAPIKey); return ok }) {
		opts = append(opts, genai.ProviderOptionAPIKey(apiKey()))
	}
	if fn != nil {
		opts = append(opts, genai.ProviderOptionTransportWrapper(fn))
	}
	return typesafe.New(t.Context(), opts...)
}

func TestClient(t *testing.T) {
	testRecorder := internaltest.NewRecords()
	t.Cleanup(func() {
		if err := testRecorder.Close(); err != nil {
			t.Error(err)
		}
	})
	cl, err := getClientInner(t, nil, func(h http.RoundTripper) http.RoundTripper {
		return testRecorder.RecordWithName(t, t.Name()+"/Warmup", h)
	})
	if err != nil {
		t.Fatal(err)
	}
	cachedModels, err := cl.ListModels(t.Context())
	if err != nil {
		t.Fatal(err)
	}
	if len(cachedModels) == 0 {
		t.Fatal("expected at least one model")
	}
	getClient := func(t *testing.T, m string) *typesafe.Client {
		t.Parallel()
		opts := []genai.ProviderOption{genai.ProviderOptionPreloadedModels(cachedModels)}
		if m != "" {
			opts = append(opts, genai.ProviderOptionModel(m))
		}
		c, err := getClientInner(t, opts, func(h http.RoundTripper) http.RoundTripper {
			return testRecorder.Record(t, h)
		})
		if err != nil {
			t.Fatal(err)
		}
		return c
	}

	t.Run("Capabilities", func(t *testing.T) {
		internaltest.TestCapabilities(t, getClient(t, "jev-latest"))
	})

	t.Run("GenSync", func(t *testing.T) {
		c := getClient(t, "jev-latest")
		var q struct {
			Billing typesafe.Noul   `json:"billing"`
			Tone    typesafe.Choice `json:"tone"`
			Urgency typesafe.Score  `json:"urgency"`
		}
		q.Billing.Instructions = typesafe.Text("Is this request about billing?")
		q.Billing.Criteria = &typesafe.NoulCriteria{
			True:  typesafe.Text("the customer is asking about a charge or an invoice"),
			False: typesafe.Text("the customer is asking about anything else"),
		}
		q.Tone.Instructions = typesafe.Text("What is the tone of the customer?")
		q.Tone.Criteria = map[string]typesafe.Content{
			"calm":       nil,
			"frustrated": typesafe.Text("annoyed but polite"),
			"angry":      nil,
		}
		q.Urgency.Instructions = typesafe.Text("How soon does this need to be handled?")
		q.Urgency.Criteria = []typesafe.Content{
			typesafe.Text("can wait"), typesafe.Text("this week"), typesafe.Text("today"), typesafe.Text("right now"),
		}
		res, err := c.GenSync(t.Context(), text("I was charged twice for order A-104. Please refund the duplicate charge. This is unacceptable."), &genai.GenOptionText{DecodeAs: &q})
		if err != nil {
			t.Fatal(err)
		}
		if res.Usage.InputTokens == 0 {
			t.Error("expected input tokens to be reported")
		}
		if res.Usage.TotalTokens != res.Usage.InputTokens+res.Usage.OutputTokens {
			t.Errorf("inconsistent usage: %s", res.Usage.String())
		}
		if res.Usage.FinishReason != genai.FinishedStop {
			t.Errorf("unexpected finish reason %q", res.Usage.FinishReason)
		}
		var answers typesafe.Answers
		if err = res.Decode(&answers); err != nil {
			t.Fatal(err)
		}
		if len(answers) != 3 {
			t.Fatalf("expected 3 answers, got %d: %s", len(answers), res.String())
		}
		if b := answers["billing"]; b.Type != typesafe.QuestionNoul || b.Noul < 0.5 {
			t.Errorf("unexpected billing answer: %#v", b)
		}
		if tone := answers["tone"]; tone.Type != typesafe.QuestionChoice {
			t.Errorf("unexpected tone answer: %#v", tone)
		} else if !slices.Contains([]string{"calm", "frustrated", "angry"}, tone.Choice) {
			t.Errorf("unexpected choice %q", tone.Choice)
		} else {
			sum := 0.0
			best := ""
			for name, p := range tone.Probabilities {
				sum += p
				if p > tone.Probabilities[best] {
					best = name
				}
			}
			if sum < 0.99 || sum > 1.01 {
				t.Errorf("probabilities don't sum to 1: %#v", tone.Probabilities)
			}
			if best != tone.Choice {
				t.Errorf("choice %q is not the most probable: %#v", tone.Choice, tone.Probabilities)
			}
		}
		if u := answers["urgency"]; u.Type != typesafe.QuestionScore {
			t.Errorf("unexpected urgency answer: %#v", u)
		} else if u.Score < 0 || u.Score > 3 {
			t.Errorf("score out of range: %f", u.Score)
		} else if len(u.Legend) != 4 {
			t.Errorf("unexpected legend: %#v", u.Legend)
		}
	})

	t.Run("GenSync-StructuredState", func(t *testing.T) {
		// The state can be arbitrary JSON, the primary way TypeSafe is meant to be used, passed as a
		// document with a JSON media type.
		c := getClient(t, "jev-latest")
		var q struct {
			Refund typesafe.Noul `json:"refund"`
		}
		q.Refund.Instructions = typesafe.Text("Does the policy say duplicate charges are eligible for a refund?")
		res, err := c.GenSync(
			t.Context(),
			doc("state.json", `{"order":{"id":"A-104","amount_usd":49},"refund_policy":"Duplicate charges are eligible for a refund."}`),
			&genai.GenOptionText{DecodeAs: &q},
		)
		if err != nil {
			t.Fatal(err)
		}
		if err = res.Decode(&q); err != nil {
			t.Fatal(err)
		}
		if q.Refund.Probability < 0.5 {
			t.Errorf("unexpected refund answer: %#v", q.Refund)
		}
	})

	t.Run("DecodeAs-Questions", func(t *testing.T) {
		c := getClient(t, "jev-latest")
		// The questions can also be built directly, as a Questions, instead of being declared by struct
		// fields. The answers are then read from Answers.
		questions := typesafe.Questions{
			"greeting": {Type: typesafe.QuestionNoul, Instructions: typesafe.Text("Is this a greeting?")},
		}
		res, err := c.GenSync(t.Context(), text("Hello there!"), &genai.GenOptionText{DecodeAs: questions})
		if err != nil {
			t.Fatal(err)
		}
		var answers typesafe.Answers
		if err = res.Decode(&answers); err != nil {
			t.Fatal(err)
		}
		if answers["greeting"].Noul < 0.5 {
			t.Errorf("expected a greeting, got %#v", answers["greeting"])
		}
	})

	t.Run("GenStream", func(t *testing.T) {
		c := getClient(t, "jev-latest")
		var q struct {
			Greeting typesafe.Noul `json:"greeting"`
		}
		q.Greeting.Instructions = typesafe.Text("Is this a greeting?")
		fragments, finish := c.GenStream(t.Context(), text("Hello there!"), &genai.GenOptionText{DecodeAs: &q})
		n := 0
		for f := range fragments {
			if f.Text == "" {
				t.Errorf("unexpected empty fragment")
			}
			n++
		}
		if n != 1 {
			t.Errorf("expected a single fragment, got %d", n)
		}
		res, err := finish()
		if err != nil {
			t.Fatal(err)
		}
		var answers typesafe.Answers
		if err = res.Decode(&answers); err != nil {
			t.Fatal(err)
		}
		if answers["greeting"].Noul < 0.5 {
			t.Errorf("expected a greeting, got %#v", answers["greeting"])
		}
	})

	t.Run("GenSyncRaw", func(t *testing.T) {
		c := getClient(t, "jev-latest")
		// Structured criteria: a score rubric of objects and a choice with object and array descriptions.
		// They are echoed back verbatim in the legend. A noul can also be described by its criteria alone.
		resp := &typesafe.SystemOneResponse{}
		err := c.GenSyncRaw(t.Context(), &typesafe.SystemOneRequest{
			State: typesafe.Object{"msg": typesafe.Text("My card was charged twice for order A-104.")},
			Model: "jev-latest",
			Questions: typesafe.Questions{
				"urgency": {
					Type:         typesafe.QuestionScore,
					Instructions: typesafe.Text("How soon does this need to be handled?"),
					Score: []typesafe.Content{
						typesafe.Object{"level": typesafe.Text("can wait"), "sla_days": 30},
						typesafe.Object{"level": typesafe.Text("this week"), "sla_days": 7},
						typesafe.Object{"level": typesafe.Text("right now"), "sla_days": 0},
					},
				},
				"tone": {
					Type:         typesafe.QuestionChoice,
					Instructions: typesafe.Object{"task": typesafe.Text("classify the tone")},
					Choice: map[string]typesafe.Content{
						"calm":  typesafe.Object{"note": typesafe.Text("relaxed")},
						"angry": typesafe.Array{typesafe.Text("loud"), typesafe.Text("rude")},
					},
				},
				"greeting": {
					Type: typesafe.QuestionNoul,
					Noul: &typesafe.NoulCriteria{True: typesafe.Text("it is a greeting"), False: typesafe.Text("it is not")},
				},
			},
		}, resp)
		if err != nil {
			t.Fatal(err)
		}
		if resp.Model == "" {
			t.Error("expected the versioned model that answered")
		}
		if resp.Usage.InputTokens == 0 {
			t.Error("expected input tokens to be reported")
		}
		if len(resp.Answers) != 3 {
			t.Fatalf("expected 3 answers, got %#v", resp.Answers)
		}
		if o, ok := resp.Answers["urgency"].Legend["2"].(typesafe.Object); !ok || o["level"] != "right now" {
			t.Errorf("expected the object level to be echoed back, got %#v", resp.Answers["urgency"].Legend)
		}
		if c := resp.Answers["tone"].Choice; c != "calm" && c != "angry" {
			t.Errorf("unexpected choice %q", c)
		}
		if n := resp.Answers["greeting"].Noul; n < 0 || n > 1 {
			t.Errorf("unexpected noul %f", n)
		}
	})

	t.Run("DecodeAs", func(t *testing.T) {
		c := getClient(t, "jev-latest")
		// The questions are declared as struct fields, with the answer typesafe.Noul, typesafe.Choice and
		// typesafe.Score hold, so the same struct is asked and filled in.
		var q struct {
			Billing typesafe.Noul   `json:"billing"`
			Tone    typesafe.Choice `json:"tone"`
			Urgency typesafe.Score  `json:"urgency"`
		}
		q.Billing.Instructions = typesafe.Text("Is this request about billing?")
		q.Tone.Instructions = typesafe.Text("What is the tone of the customer?")
		q.Tone.Criteria = map[string]typesafe.Content{"calm": nil, "angry": nil}
		q.Urgency.Instructions = typesafe.Text("How soon does this need to be handled?")
		q.Urgency.Criteria = []typesafe.Content{typesafe.Text("can wait"), typesafe.Text("today")}
		// The questions TypeSafe will be asked are derived from the struct.
		questions, err := typesafe.QuestionsFrom(&q)
		if err != nil {
			t.Fatal(err)
		}
		if len(questions) != 3 {
			t.Fatalf("expected 3 questions, got %#v", questions)
		}
		res, err := c.GenSync(t.Context(), text("I was charged twice for order A-104."), &genai.GenOptionText{DecodeAs: &q})
		if err != nil {
			t.Fatal(err)
		}
		if err = res.Decode(&q); err != nil {
			t.Fatal(err)
		}
		if q.Billing.Probability < 0.5 {
			t.Errorf("unexpected billing answer: %#v", q.Billing)
		}
		if q.Tone.Label != "calm" && q.Tone.Label != "angry" {
			t.Errorf("unexpected tone answer: %#v", q.Tone)
		}
		if q.Tone.Probabilities[q.Tone.Label] == 0 {
			t.Errorf("unexpected tone probabilities: %#v", q.Tone.Probabilities)
		}
		if q.Urgency.Value < 0 || q.Urgency.Value > 1 {
			t.Errorf("unexpected urgency answer: %#v", q.Urgency)
		}
		if len(q.Urgency.Legend) != 2 {
			t.Errorf("unexpected urgency legend: %#v", q.Urgency.Legend)
		}
	})

	t.Run("Scoreboard", func(t *testing.T) {
		sb := typesafe.Scoreboard()
		if err := sb.Validate(); err != nil {
			t.Fatal(err)
		}
		// The API lists aliases; versioned model IDs are accepted but not listed.
		listed := map[string]struct{}{}
		for _, m := range cachedModels {
			listed[m.GetID()] = struct{}{}
		}
		for _, sc := range sb.Scenarios {
			if sc.Untested() {
				continue
			}
			if len(sc.Models) != 1 {
				t.Errorf("scenario must have a single model: %v", sc.Models)
			}
			if sc.SOTA && sc.Models[0] == "" {
				t.Error("missing SOTA model")
			}
		}
		if _, ok := listed["jev-latest"]; !ok {
			t.Error("jev-latest is not listed by ListModels")
		}
	})

	t.Run("Preferred", func(t *testing.T) {
		internaltest.TestPreferredModels(t, func(st *testing.T, model string, modality genai.Modality) (genai.Provider, error) {
			opts := []genai.ProviderOption{
				genai.ProviderOptionModalities{modality},
				genai.ProviderOptionPreloadedModels(cachedModels),
			}
			if model != "" {
				opts = append(opts, genai.ProviderOptionModel(model))
			}
			return getClientInner(st, opts, func(h http.RoundTripper) http.RoundTripper {
				return testRecorder.Record(st, h)
			})
		})
	})

	t.Run("errors", func(t *testing.T) {
		var q struct {
			A typesafe.Noul `json:"a"`
		}
		q.A.Instructions = typesafe.Text("Is this a test?")
		questions := &genai.GenOptionText{DecodeAs: &q}
		t.Run("bad apiKey", func(t *testing.T) {
			c, err := getClientInner(t, []genai.ProviderOption{
				genai.ProviderOptionAPIKey("bad apiKey"),
				genai.ProviderOptionModel("jev-latest"),
			}, func(h http.RoundTripper) http.RoundTripper {
				return testRecorder.Record(t, h)
			})
			if err != nil {
				t.Fatal(err)
			}
			_, err = c.GenSync(t.Context(), text("test"), questions)
			if err == nil {
				t.Fatal("expected error")
			}
			if got := err.Error(); !strings.Contains(got, "http 401") || !strings.Contains(got, "authentication_error") {
				t.Fatalf("unexpected error: %q", got)
			}
		})
		t.Run("bad model", func(t *testing.T) {
			c, err := getClientInner(t, []genai.ProviderOption{
				genai.ProviderOptionModel("bad model"),
			}, func(h http.RoundTripper) http.RoundTripper {
				return testRecorder.Record(t, h)
			})
			if err != nil {
				t.Fatal(err)
			}
			_, err = c.GenSync(t.Context(), text("test"), questions)
			if err == nil {
				t.Fatal("expected error")
			}
			if got := err.Error(); !strings.Contains(got, "http 400") || !strings.Contains(got, "Unknown model: bad model") {
				t.Fatalf("unexpected error: %q", got)
			}
		})
		t.Run("invalid raw request", func(t *testing.T) {
			// The API reports the offending fields as a list of validation errors.
			c, err := getClientInner(t, []genai.ProviderOption{
				genai.ProviderOptionModel("jev-latest"),
			}, func(h http.RoundTripper) http.RoundTripper {
				return testRecorder.Record(t, h)
			})
			if err != nil {
				t.Fatal(err)
			}
			err = c.GenSyncRaw(t.Context(), &typesafe.SystemOneRequest{}, &typesafe.SystemOneResponse{})
			if err == nil {
				t.Fatal("expected error")
			}
			if got := err.Error(); !strings.Contains(got, "http 422") || !strings.Contains(got, "body.questions:") {
				t.Fatalf("unexpected error: %q", got)
			}
		})
		t.Run("bad apiKey ListModels", func(t *testing.T) {
			c, err := getClientInner(t, []genai.ProviderOption{
				genai.ProviderOptionAPIKey("bad apiKey"),
			}, func(h http.RoundTripper) http.RoundTripper {
				return testRecorder.Record(t, h)
			})
			if err != nil {
				t.Fatal(err)
			}
			if _, err = c.ListModels(t.Context()); err == nil {
				t.Fatal("expected error")
			}
			if got := err.Error(); !strings.Contains(got, "http 401") {
				t.Fatalf("unexpected error: %q", got)
			}
		})
		t.Run("invalid question", func(t *testing.T) {
			// The API rejects a question that has neither instructions nor criteria.
			c, err := getClientInner(t, []genai.ProviderOption{
				genai.ProviderOptionModel("jev-latest"),
			}, func(h http.RoundTripper) http.RoundTripper {
				return testRecorder.Record(t, h)
			})
			if err != nil {
				t.Fatal(err)
			}
			err = c.GenSyncRaw(t.Context(), &typesafe.SystemOneRequest{
				State: typesafe.Text("test"),
				Model: "jev-latest",
				Questions: typesafe.Questions{
					"a": {Type: typesafe.QuestionNoul, Instructions: typesafe.Text("")},
				},
			}, &typesafe.SystemOneResponse{})
			if err == nil {
				t.Fatal("expected error")
			}
			if got := err.Error(); !strings.Contains(got, "http 400") {
				t.Fatalf("unexpected error: %q", got)
			}
		})
	})

	t.Run("no model", func(t *testing.T) {
		c, err := getClientInner(t, nil, nil)
		if err != nil {
			t.Fatal(err)
		}
		var q struct {
			A typesafe.Noul `json:"a"`
		}
		q.A.Instructions = typesafe.Text("x")
		_, err = c.GenSync(t.Context(), text("test"), &genai.GenOptionText{DecodeAs: &q})
		if got := err.Error(); got != "a model is required" {
			t.Fatalf("unexpected error: %q", got)
		}
	})
}

func TestQuestionsValidate(t *testing.T) {
	data := []struct {
		name      string
		questions typesafe.Questions
		wantErr   string
	}{
		{
			// QuestionsFrom reports this itself, so this is only reachable by calling Validate directly.
			name:    "no questions",
			wantErr: "at least one question is required",
		},
		{
			name: "noul",
			questions: typesafe.Questions{
				"a": {Type: typesafe.QuestionNoul, Instructions: typesafe.Text("Is this a test?")},
			},
		},
		{
			name: "noul criteria only",
			questions: typesafe.Questions{
				"a": {
					Type: typesafe.QuestionNoul,
					Noul: &typesafe.NoulCriteria{True: typesafe.Text("yes"), False: typesafe.Text("no")},
				},
			},
		},
		{
			name: "choice",
			questions: typesafe.Questions{
				"a": {
					Type:         typesafe.QuestionChoice,
					Instructions: typesafe.Object{"task": "classify"},
					Choice:       map[string]typesafe.Content{"a": nil, "b": typesafe.Text("the b option")},
				},
			},
		},
		{
			name: "score",
			questions: typesafe.Questions{
				"a": {Type: typesafe.QuestionScore, Score: []typesafe.Content{typesafe.Text("low"), typesafe.Text("high")}},
			},
		},
		{
			name: "choice without instructions",
			questions: typesafe.Questions{
				"a": {Type: typesafe.QuestionChoice, Choice: map[string]typesafe.Content{"a": nil}},
			},
		},
		{
			name: "score with one level",
			questions: typesafe.Questions{
				"a": {Type: typesafe.QuestionScore, Score: []typesafe.Content{typesafe.Text("only")}},
			},
		},
		{
			name: "invalid noul",
			questions: typesafe.Questions{
				"a": {Type: typesafe.QuestionNoul},
			},
			wantErr: "question \"a\": field Instructions or Noul: one is required",
		},
		{
			name: "noul with a choice field",
			questions: typesafe.Questions{
				"a": {Type: typesafe.QuestionNoul, Choice: map[string]typesafe.Content{"b": nil}},
			},
			wantErr: "fields Choice and Score: can't be set on a noul question",
		},
		{
			name: "score with a noul field",
			questions: typesafe.Questions{
				"a": {
					Type:  typesafe.QuestionScore,
					Noul:  &typesafe.NoulCriteria{True: typesafe.Text("yes")},
					Score: []typesafe.Content{typesafe.Text("x")},
				},
			},
			wantErr: "fields Noul and Choice: can't be set on a score question",
		},
		{
			name: "noul with invalid criteria",
			questions: typesafe.Questions{
				"a": {
					Type: typesafe.QuestionNoul,
					Noul: &typesafe.NoulCriteria{True: typesafe.Object(nil), False: typesafe.Array(nil)},
				},
			},
			wantErr: "field Criteria.True: Object is nil, use nil to leave the field unset",
		},
		{
			name: "choice with an invalid description",
			questions: typesafe.Questions{
				"a": {Type: typesafe.QuestionChoice, Choice: map[string]typesafe.Content{"b": typesafe.Object(nil)}},
			},
			wantErr: "field Choice[b]: Object is nil, use nil to leave the field unset",
		},
		{
			name: "score with a nil level",
			questions: typesafe.Questions{
				"a": {Type: typesafe.QuestionScore, Score: []typesafe.Content{typesafe.Text("x"), nil}},
			},
			wantErr: "field Score[1]: must not be nil",
		},
		{
			name: "score with an invalid level",
			questions: typesafe.Questions{
				"a": {Type: typesafe.QuestionScore, Score: []typesafe.Content{typesafe.Text("x"), typesafe.Array(nil)}},
			},
			wantErr: "field Score[1]: Array is nil, use nil to leave the field unset",
		},
		{
			name: "no type",
			questions: typesafe.Questions{
				"a": {Instructions: typesafe.Text("x")},
			},
			wantErr: `field Type: must be "noul", "choice" or "score", got ""`,
		},
		{
			name: "noul empty criteria",
			questions: typesafe.Questions{
				"a": {Type: typesafe.QuestionNoul, Noul: &typesafe.NoulCriteria{}},
			},
			wantErr: "at least one of True or False is required",
		},
		{
			name: "type mismatch",
			questions: typesafe.Questions{
				"a": {Type: typesafe.QuestionChoice, Score: []typesafe.Content{typesafe.Text("x")}},
			},
			wantErr: "fields Noul and Score: can't be set on a choice question",
		},
		{
			name: "choice without criteria",
			questions: typesafe.Questions{
				"a": {Type: typesafe.QuestionChoice, Instructions: typesafe.Text("x")},
			},
			wantErr: "field Choice: at least one option is required",
		},
		{
			name: "score without criteria",
			questions: typesafe.Questions{
				"a": {Type: typesafe.QuestionScore, Instructions: typesafe.Text("x")},
			},
			wantErr: "field Score: at least one level is required",
		},
		{
			name: "nil object",
			questions: typesafe.Questions{
				"a": {Type: typesafe.QuestionNoul, Instructions: typesafe.Object(nil)},
			},
			wantErr: "field Instructions: Object is nil, use nil to leave the field unset",
		},
	}
	for _, line := range data {
		t.Run(line.name, func(t *testing.T) {
			err := line.questions.Validate()
			if line.wantErr == "" {
				if err != nil {
					t.Fatal(err)
				}
				return
			}
			if err == nil {
				t.Fatal("expected error")
			}
			if !strings.Contains(err.Error(), line.wantErr) {
				t.Fatalf("want %q, got %q", line.wantErr, err)
			}
		})
	}
}

func TestGenSyncErrors(t *testing.T) {
	c, err := typesafe.New(t.Context(), genai.ProviderOptionAPIKey("x"), genai.ProviderOptionModel("jev-latest"))
	if err != nil {
		t.Fatal(err)
	}
	// asked returns a questionnaire whose fields declare valid questions, ready to pass as DecodeAs.
	asked := func() *questionnaire {
		q := &questionnaire{}
		q.Greeting.Instructions = typesafe.Text("Is this a greeting?")
		q.Tone.Criteria = map[string]typesafe.Content{"a": nil}
		q.Urgency.Criteria = []typesafe.Content{typesafe.Text("low"), typesafe.Text("high")}
		return q
	}
	valid := &genai.GenOptionText{DecodeAs: asked()}
	data := []struct {
		name    string
		msgs    genai.Messages
		opts    []genai.GenOption
		wantErr string
	}{
		{
			name:    "missing DecodeAs",
			msgs:    text("test"),
			wantErr: "the questions to ask are required, pass *genai.GenOptionText with DecodeAs",
		},
		{
			name:    "unsupported option",
			msgs:    text("test"),
			opts:    []genai.GenOption{genai.GenOptionSeed(1)},
			wantErr: "not supported: genai.GenOptionSeed",
		},
		{
			name:    "no state",
			msgs:    nil,
			opts:    []genai.GenOption{valid},
			wantErr: "must pass exactly one message",
		},
		{
			name:    "text option without DecodeAs",
			msgs:    text("test"),
			opts:    []genai.GenOption{&genai.GenOptionText{}},
			wantErr: "field DecodeAs: a pointer to a struct of Noul, Choice and Score fields, or a Questions, is required",
		},
		{
			name: "text option with unsupported fields",
			msgs: text("test"),
			opts: []genai.GenOption{&genai.GenOptionText{
				Temperature:  1,
				TopP:         1,
				MaxTokens:    1,
				TopLogprobs:  1,
				TopK:         1,
				SystemPrompt: "be nice",
				Stop:         []string{"stop"},
				ReplyAsJSON:  true,
				DecodeAs:     &questionnaire{},
			}},
			wantErr: "not supported: GenOptionText.Temperature, GenOptionText.TopP, GenOptionText.MaxTokens, GenOptionText.TopLogprobs, GenOptionText.TopK, GenOptionText.SystemPrompt, GenOptionText.Stop, GenOptionText.ReplyAsJSON",
		},
		{
			name:    "DecodeAs is not a questionnaire",
			msgs:    text("test"),
			opts:    []genai.GenOption{&genai.GenOptionText{DecodeAs: &struct{ A bool }{}}},
			wantErr: "field DecodeAs: field A: must be a Noul, a Choice or a Score, got a bool",
		},
		{
			name:    "empty questions",
			msgs:    text("test"),
			opts:    []genai.GenOption{&genai.GenOptionText{DecodeAs: typesafe.Questions{}}},
			wantErr: "field DecodeAs: at least one question is required",
		},
		{
			name: "invalid questions",
			msgs: text("test"),
			opts: []genai.GenOption{&genai.GenOptionText{DecodeAs: typesafe.Questions{
				"a": {Type: typesafe.QuestionNoul},
			}}},
			wantErr: "field DecodeAs: question \"a\": field Instructions or Noul: one is required",
		},
		{
			name: "questions with an unsupported field",
			msgs: text("test"),
			opts: []genai.GenOption{&genai.GenOptionText{
				Temperature: 200,
				DecodeAs: typesafe.Questions{
					"a": {Type: typesafe.QuestionNoul, Instructions: typesafe.Text("x")},
				},
			}},
			wantErr: "field Temperature: must be [0, 100]",
		},
		{
			name:    "invalid text option",
			msgs:    text("test"),
			opts:    []genai.GenOption{&genai.GenOptionText{DecodeAs: "bogus"}},
			wantErr: "field DecodeAs: must be a JSON object or array, or a pointer to one, got string",
		},
		{
			name:    "DecodeAs without questions",
			msgs:    text("test"),
			opts:    []genai.GenOption{&genai.GenOptionText{DecodeAs: &questionnaire{}}},
			wantErr: "field DecodeAs: question \"greeting\": field Instructions or Noul: one is required",
		},
		{
			name:    "document not json",
			msgs:    doc("state.txt", "hi"),
			opts:    []genai.GenOption{valid},
			wantErr: `the state document must be application/json, got "text/plain; charset=utf-8"; name it with a .json extension`,
		},
		{
			name:    "document without name",
			msgs:    doc("", "{}"),
			opts:    []genai.GenOption{valid},
			wantErr: "failed to determine mime-type, pass a filename with an extension",
		},
		{
			name:    "document from url",
			msgs:    genai.Messages{genai.Message{Requests: []genai.Request{{Doc: genai.Doc{URL: "https://example.com/state.json"}}}}},
			opts:    []genai.GenOption{valid},
			wantErr: "the state document must be inline",
		},
		{
			name:    "document not json content",
			msgs:    doc("state.json", "1"),
			opts:    []genai.GenOption{valid},
			wantErr: "the state document must contain a JSON object or array: expected a string, a JSON object or a JSON array, got 1",
		},
		{
			name:    "document truncated",
			msgs:    doc("state.json", "{"),
			opts:    []genai.GenOption{valid},
			wantErr: "the state document must contain a JSON object or array:",
		},
		{
			name:    "document empty",
			msgs:    doc("state.json", ""),
			opts:    []genai.GenOption{valid},
			wantErr: "empty data",
		},
		{
			name:    "two empty requests",
			msgs:    genai.Messages{genai.Message{Requests: []genai.Request{{}, {}}}},
			opts:    []genai.GenOption{valid},
			wantErr: "request #0: must have the state as text or as a JSON document",
		},
		{
			name:    "two messages",
			msgs:    genai.Messages{genai.NewTextMessage("test"), genai.NewTextMessage("test")},
			opts:    []genai.GenOption{valid},
			wantErr: "must pass exactly one message",
		},
		{
			name:    "empty message",
			msgs:    genai.Messages{genai.Message{}},
			opts:    []genai.GenOption{valid},
			wantErr: "the message must have the state as text or as a JSON document",
		},
		{
			name:    "empty request",
			msgs:    genai.Messages{genai.Message{Requests: []genai.Request{{}}}},
			opts:    []genai.GenOption{valid},
			wantErr: "must have the state as text or as a JSON document",
		},
		{
			name:    "reply",
			msgs:    genai.Messages{genai.Message{Replies: []genai.Reply{{Text: "hello"}}}},
			opts:    []genai.GenOption{valid},
			wantErr: "TypeSafe has no conversation support; pass the full state as text or as a JSON document",
		},
		{
			name:    "tool call result",
			msgs:    genai.Messages{genai.Message{ToolCallResults: []genai.ToolCallResult{{}}}},
			opts:    []genai.GenOption{valid},
			wantErr: "TypeSafe has no conversation support; pass the full state as text or as a JSON document",
		},
		{
			name:    "no questions",
			msgs:    text("test"),
			opts:    []genai.GenOption{},
			wantErr: "the questions to ask are required, pass *genai.GenOptionText with DecodeAs",
		},
	}
	for _, line := range data {
		t.Run(line.name, func(t *testing.T) {
			_, err := c.GenSync(t.Context(), line.msgs, line.opts...)
			if err == nil {
				t.Fatal("expected error")
			}
			if !strings.Contains(err.Error(), line.wantErr) {
				t.Fatalf("want %q, got %q", line.wantErr, err)
			}
		})
	}
	t.Run("not supported", func(t *testing.T) {
		_, err := c.GenSync(t.Context(), text("test"), &genai.GenOptionTools{})
		if _, ok := errors.AsType[*base.ErrNotSupported](err); !ok {
			t.Fatalf("expected ErrNotSupported, got %T: %v", err, err)
		}
	})
}

func TestAnswerMarshal(t *testing.T) {
	data := []struct {
		name string
		in   string
		want string
	}{
		{"noul", `{"type":"noul","noul":0}`, `{"type":"noul","noul":0}`},
		{
			"choice", `{"type":"choice","choice":"a","confidence":1,"probabilities":{"a":1,"b":0}}`,
			`{"type":"choice","choice":"a","confidence":1,"probabilities":{"a":1,"b":0}}`,
		},
		{
			"score", `{"type":"score","score":0,"confidence":0.5,"legend":{"0":"low","1":"high"},"probabilities":{"0":0.5,"1":0.5}}`,
			`{"type":"score","score":0,"confidence":0.5,"legend":{"0":"low","1":"high"},"probabilities":{"0":0.5,"1":0.5}}`,
		},
		{
			// The legend holds the Question.Score criteria, which can be JSON structure, and a nil value for
			// a level the caller left undescribed.
			"score structured legend",
			`{"type":"score","score":1,"confidence":0.5,"legend":{"0":"low","1":{"weight":2,"desc":"high"},"2":["a","b"],"3":null},"probabilities":{"0":0.5,"1":0.5}}`,
			`{"type":"score","score":1,"confidence":0.5,"legend":{"0":"low","1":{"desc":"high","weight":2},"2":["a","b"],"3":null},"probabilities":{"0":0.5,"1":0.5}}`,
		},
	}
	for _, line := range data {
		t.Run(line.name, func(t *testing.T) {
			var a typesafe.Answer
			if err := json.Unmarshal([]byte(line.in), &a); err != nil {
				t.Fatal(err)
			}
			raw, err := json.Marshal(a)
			if err != nil {
				t.Fatal(err)
			}
			if string(raw) != line.want {
				t.Fatalf("want %s, got %s", line.want, raw)
			}
		})
	}
	t.Run("unknown", func(t *testing.T) {
		if _, err := json.Marshal(typesafe.Answer{Type: "bogus"}); err == nil {
			t.Fatal("expected error")
		}
	})
	t.Run("not an answer", func(t *testing.T) {
		var a typesafe.Answer
		if err := json.Unmarshal([]byte(`1`), &a); err == nil {
			t.Fatal("expected error")
		}
	})
	t.Run("unknown field", func(t *testing.T) {
		var a typesafe.Answer
		if err := json.Unmarshal([]byte(`{"type":"noul","noul":0.5,"bogus":1}`), &a); err == nil {
			t.Fatal("expected error")
		} else if !strings.Contains(err.Error(), "unknown field") {
			t.Fatalf("unexpected error: %v", err)
		}
	})
	t.Run("unknown from json", func(t *testing.T) {
		// An answer type this client does not know is kept as-is instead of breaking the whole reply.
		var a typesafe.Answer
		in := `{"type":"ranks","order":["a","b"]}`
		if err := json.Unmarshal([]byte(in), &a); err != nil {
			t.Fatal(err)
		}
		raw, err := json.Marshal(a)
		if err != nil {
			t.Fatal(err)
		}
		if string(raw) != in {
			t.Fatalf("want %s, got %s", in, raw)
		}
	})
	t.Run("answers", func(t *testing.T) {
		var a typesafe.Answers
		if err := json.Unmarshal([]byte(`{"a":{"type":"noul","noul":0.25}}`), &a); err != nil {
			t.Fatal(err)
		}
		raw, err := json.Marshal(a)
		if err != nil {
			t.Fatal(err)
		}
		if string(raw) != `{"a":{"type":"noul","noul":0.25}}` {
			t.Fatalf("unexpected: %s", raw)
		}
	})
}

func TestContentValidate(t *testing.T) {
	type payload struct {
		A int `json:"a"`
	}
	data := []struct {
		name    string
		in      typesafe.Content
		wantErr string
	}{
		{"text", typesafe.Text("hi"), ""},
		{"empty text", typesafe.Text(""), ""},
		{"object", typesafe.Object{"a": 1, "b": []any{"c", nil, map[string]any{"d": true}}}, ""},
		{"object with struct", typesafe.Object{"a": payload{A: 1}}, ""},
		{"object with channel", typesafe.Object{"a": make(chan int)}, `key "a": json: unsupported type: chan int`},
		{"object with nested channel", typesafe.Object{"a": []any{[]any{make(chan int)}}}, `key "a": json: unsupported type: chan int`},
		{
			"several bad keys",
			typesafe.Object{"b": make(chan int), "a": make(chan int)},
			"key \"a\": json: unsupported type: chan int\nkey \"b\": json: unsupported type: chan int",
		},
		{
			"several bad items",
			typesafe.Array{"a", make(chan int), nil, make(chan int)},
			"index 1: json: unsupported type: chan int\nindex 3: json: unsupported type: chan int",
		},
		{"nil object", typesafe.Object(nil), "Object is nil, use nil to leave the field unset"},
		{"array", typesafe.Array{"a", 1, true, nil, typesafe.Text("b")}, ""},
		{"array with NaN", typesafe.Array{math.NaN()}, "index 0: json: unsupported value: NaN"},
		{"nil array", typesafe.Array(nil), "Array is nil, use nil to leave the field unset"},
	}
	for _, line := range data {
		t.Run(line.name, func(t *testing.T) {
			err := line.in.Validate()
			if line.wantErr == "" {
				if err != nil {
					t.Fatal(err)
				}
				return
			}
			if err == nil {
				t.Fatal("expected error")
			}
			if got := err.Error(); got != line.wantErr {
				t.Fatalf("want %q, got %q", line.wantErr, got)
			}
		})
	}
}

func TestSystemOneRequestMarshal(t *testing.T) {
	// Content marshals to the value itself, whatever the variant.
	data := []struct {
		name  string
		state typesafe.Content
		want  string
	}{
		{"text", typesafe.Text("hi"), `"state":"hi",`},
		{"empty text", typesafe.Text(""), `"state":"",`},
		{"object", typesafe.Object{"message": "hi", "order": 49}, `"state":{"message":"hi","order":49},`},
		{"empty object", typesafe.Object{}, `"state":{},`},
		{"array", typesafe.Array{typesafe.Text("hi"), typesafe.Text("there")}, `"state":["hi","there"],`},
	}
	for _, line := range data {
		t.Run(line.name, func(t *testing.T) {
			req := &typesafe.SystemOneRequest{
				State:     line.state,
				Model:     "jev-latest",
				Questions: typesafe.Questions{"a": {Type: typesafe.QuestionNoul, Instructions: typesafe.Text("x")}},
			}
			raw, err := json.Marshal(req)
			if err != nil {
				t.Fatal(err)
			}
			if !strings.Contains(string(raw), line.want) {
				t.Fatalf("want %s, got %s", line.want, raw)
			}
		})
	}
}

func TestSystemOneRequestFrom(t *testing.T) {
	// A message holds the state: its request, or the array of them when it has several, since the API
	// takes an array as a sequence of messages or records.
	data := []struct {
		name string
		msg  genai.Message
		want string
	}{
		{"text", genai.Message{Requests: []genai.Request{{Text: "hi"}}}, `"state":"hi",`},
		{"two texts", genai.Message{Requests: []genai.Request{{Text: "hi"}, {Text: "there"}}}, `"state":["hi","there"],`},
		{
			"object document",
			genai.Message{Requests: []genai.Request{{Doc: genai.Doc{
				Filename: "state.json", Src: strings.NewReader(`{"order":"A-104"}`),
			}}}},
			`"state":{"order":"A-104"},`,
		},
		{
			"mixed requests",
			genai.Message{Requests: []genai.Request{
				{Text: "hi"},
				{Doc: genai.Doc{Filename: "state.json", Src: strings.NewReader(`["a","b"]`)}},
			}},
			`"state":["hi",["a","b"]],`,
		},
	}
	for _, line := range data {
		t.Run(line.name, func(t *testing.T) {
			req := &typesafe.SystemOneRequest{}
			if err := req.From(&line.msg); err != nil {
				t.Fatal(err)
			}
			raw, err := json.Marshal(req)
			if err != nil {
				t.Fatal(err)
			}
			if !strings.Contains(string(raw), line.want) {
				t.Fatalf("want %s, got %s", line.want, raw)
			}
		})
	}
}

func TestScoreLegend(t *testing.T) {
	// The legend is the Question.Score criteria echoed back, as Text, Object or Array.
	in := `{"0":"low","1":{"desc":"high","nested":{"x":[1,2]}},"2":["a",{"b":true}],"3":null}`
	var l typesafe.ScoreLegend
	if err := json.Unmarshal([]byte(in), &l); err != nil {
		t.Fatal(err)
	}
	if v, ok := l["0"].(typesafe.Text); !ok || v != "low" {
		t.Errorf("unexpected level 0: %#v", l["0"])
	}
	o, ok := l["1"].(typesafe.Object)
	if !ok || o["desc"] != "high" {
		t.Fatalf("unexpected level 1: %#v", l["1"])
	}
	// Nested values are left as decoded, like Object and Array document.
	if nested, ok := o["nested"].(map[string]any); !ok || len(nested) != 1 {
		t.Errorf("unexpected nested value: %#v", o["nested"])
	}
	a, ok := l["2"].(typesafe.Array)
	if !ok || len(a) != 2 {
		t.Fatalf("unexpected level 2: %#v", l["2"])
	}
	if _, ok := a[1].(map[string]any); !ok {
		t.Errorf("unexpected level 2 item: %#v", a[1])
	}
	if v, ok := l["3"]; !ok || v != nil {
		t.Errorf("unexpected level 3: %#v", v)
	}
	// What was decoded can be encoded back as is.
	raw, err := json.Marshal(l)
	if err != nil {
		t.Fatal(err)
	}
	if string(raw) != in {
		t.Fatalf("want %s, got %s", in, raw)
	}
	// Anything that is not text, object or array is rejected.
	for _, in := range []string{`{"0":1}`, `{"0":true}`} {
		var l typesafe.ScoreLegend
		err := json.Unmarshal([]byte(in), &l)
		if err == nil {
			t.Fatalf("expected error for %s", in)
		}
		if !strings.Contains(err.Error(), "expected a string, a JSON object or a JSON array") {
			t.Fatalf("unexpected error: %v", err)
		}
	}
	// The legend must be an object.
	var l3 typesafe.ScoreLegend
	if err := json.Unmarshal([]byte(`[]`), &l3); err == nil {
		t.Fatal("expected error")
	}
	// A null legend is no legend.
	var l2 typesafe.ScoreLegend
	if err := json.Unmarshal([]byte(`null`), &l2); err != nil {
		t.Fatal(err)
	}
	if l2 != nil {
		t.Errorf("expected nil, got %#v", l2)
	}
}

func TestQuestionMarshal(t *testing.T) {
	data := []struct {
		name string
		in   typesafe.Question
		want string
	}{
		{
			name: "noul",
			in:   typesafe.Question{Type: typesafe.QuestionNoul, Instructions: typesafe.Text("Is this a test?")},
			want: `{"type":"noul","instructions":"Is this a test?"}`,
		},
		{
			name: "noul criteria",
			in: typesafe.Question{
				Type: typesafe.QuestionNoul,
				Noul: &typesafe.NoulCriteria{True: typesafe.Text("yes"), False: typesafe.Text("no")},
			},
			want: `{"type":"noul","criteria":{"true":"yes","false":"no"}}`,
		},
		{
			name: "choice",
			in: typesafe.Question{
				Type:         typesafe.QuestionChoice,
				Instructions: typesafe.Object{"task": "classify"},
				Choice:       map[string]typesafe.Content{"a": nil, "b": typesafe.Text("the b option")},
			},
			want: `{"type":"choice","instructions":{"task":"classify"},"criteria":{"a":null,"b":"the b option"}}`,
		},
		{
			name: "score",
			in:   typesafe.Question{Type: typesafe.QuestionScore, Score: []typesafe.Content{typesafe.Text("low"), typesafe.Text("high")}},
			want: `{"type":"score","criteria":["low","high"]}`,
		},
	}
	for _, line := range data {
		t.Run(line.name, func(t *testing.T) {
			raw, err := json.Marshal(line.in)
			if err != nil {
				t.Fatal(err)
			}
			if string(raw) != line.want {
				t.Fatalf("want %s, got %s", line.want, raw)
			}
		})
	}
	t.Run("unknown type", func(t *testing.T) {
		if _, err := json.Marshal(typesafe.Question{Type: "bogus"}); err == nil {
			t.Fatal("expected error")
		}
	})
	t.Run("criteria that can't be encoded", func(t *testing.T) {
		// MarshalJSON reports the error of the criteria it encodes.
		q := typesafe.Question{
			Type:   typesafe.QuestionChoice,
			Choice: map[string]typesafe.Content{"a": typesafe.Object{"b": make(chan int)}},
		}
		if _, err := json.Marshal(q); err == nil {
			t.Fatal("expected error")
		}
	})
}

func TestErrorResponse(t *testing.T) {
	data := []struct {
		name string
		in   string
		want string
	}{
		{"string", `{"detail":"Noul question must have criteria or instructions: a"}`, "Noul question must have criteria or instructions: a"},
		{"object", `{"detail":{"error_type":"api_usage_error","message":"Unknown model: bad model"}}`, "api_usage_error: Unknown model: bad model"},
		{
			"validation", `{"detail":[{"type":"too_short","loc":["body","questions"],"msg":"Dictionary should have at least 1 item after validation, not 0"}]}`,
			"body.questions: Dictionary should have at least 1 item after validation, not 0",
		},
		{"empty", `{}`, "unknown error"},
		{"object without error_type", `{"detail":{"message":"broken"}}`, "broken"},
		{"validation without loc", `{"detail":[{"type":"t","loc":[],"msg":"broken"}]}`, "body: broken"},
	}
	for _, line := range data {
		t.Run(line.name, func(t *testing.T) {
			var er typesafe.ErrorResponse
			if err := json.Unmarshal([]byte(line.in), &er); err != nil {
				t.Fatal(err)
			}
			if got := er.Error(); got != line.want {
				t.Fatalf("want %q, got %q", line.want, got)
			}
			if !er.IsAPIError() {
				t.Error("expected an API error")
			}
		})
	}
	t.Run("malformed", func(t *testing.T) {
		var er typesafe.ErrorResponse
		if err := json.Unmarshal([]byte(`{"detail":1}`), &er); err == nil {
			t.Fatal("expected error")
		}
	})
}

func TestScoreboard(t *testing.T) {
	t.Parallel()
	sb := typesafe.Scoreboard()
	if err := sb.Validate(); err != nil {
		t.Fatal(err)
	}
	if sb.Country != "US" {
		t.Errorf("unexpected country %q", sb.Country)
	}
	n := 0
	for _, sc := range sb.Scenarios {
		if sc.Untested() {
			continue
		}
		n++
		if sc.GenSync == nil {
			t.Error("expected GenSync functionality")
		}
		if sc.GenStream != nil {
			t.Error("TypeSafe has no streaming, GenStream should not be declared")
		}
		if sc.GenSync != nil && sc.GenSync.JSONSchema.Object != scoreboard.True {
			t.Error("expected structured output with an object at the root to be declared")
		}
		if sc.GenSync != nil && sc.GenSync.JSONSchema.Array == scoreboard.True {
			// The reply is always an object keyed by question name.
			t.Error("expected no array at the root to be declared")
		}
	}
	if n != 1 {
		t.Errorf("expected a single tested scenario, got %d", n)
	}
}

func TestNewErrors(t *testing.T) {
	t.Run("unsupported option", func(t *testing.T) {
		_, err := typesafe.New(t.Context(), bogusOption{})
		if err == nil || !strings.Contains(err.Error(), "unsupported option type") {
			t.Fatalf("unexpected error: %v", err)
		}
	})
	t.Run("duplicate options", func(t *testing.T) {
		_, err := typesafe.New(t.Context(), genai.ProviderOptionAPIKey("a"), genai.ProviderOptionAPIKey("b"))
		if err == nil || !strings.Contains(err.Error(), "duplicate provider option") {
			t.Fatalf("unexpected error: %v", err)
		}
	})
	t.Run("invalid option", func(t *testing.T) {
		_, err := typesafe.New(t.Context(), genai.ProviderOptionAPIKey(""))
		if err == nil || !strings.Contains(err.Error(), "cannot be empty") {
			t.Fatalf("unexpected error: %v", err)
		}
	})
	t.Run("modalities text", func(t *testing.T) {
		if _, err := typesafe.New(t.Context(),
			genai.ProviderOptionAPIKey("x"),
			genai.ProviderOptionModalities{genai.ModalityText},
		); err != nil {
			t.Fatal(err)
		}
	})
	t.Run("modalities", func(t *testing.T) {
		_, err := typesafe.New(t.Context(), genai.ProviderOptionModalities{genai.ModalityImage})
		if err == nil || !strings.Contains(err.Error(), "only text is supported") {
			t.Fatalf("unexpected error: %v", err)
		}
	})
	t.Run("missing api key", func(t *testing.T) {
		t.Setenv("TYPESAFE_API_KEY", "")
		c, err := typesafe.New(t.Context(), genai.ProviderOptionModel("jev-latest"))
		if c == nil {
			t.Fatal("expected a client")
		}
		var want *base.ErrAPIKeyRequired
		if !errors.As(err, &want) {
			t.Fatalf("expected ErrAPIKeyRequired, got %T: %v", err, err)
		}
		if want.EnvVar != "TYPESAFE_API_KEY" {
			t.Errorf("unexpected env var %q", want.EnvVar)
		}
	})
}

func TestGenSyncInvalidResponse(t *testing.T) {
	data := []struct {
		name    string
		body    string
		wantErr string
	}{
		{
			name:    "no answers",
			body:    `{"model":"jev-1.13.0","usage":{"input_tokens":1,"output_tokens":1}}`,
			wantErr: "no answer returned",
		},
		{
			name:    "missing answer",
			body:    `{"model":"jev-1.13.0","answers":{"b":{"type":"noul","noul":0.5}},"usage":{"input_tokens":1,"output_tokens":1}}`,
			wantErr: `no answer returned for question "a"`,
		},
	}
	var q struct {
		A typesafe.Noul `json:"a"`
	}
	q.A.Instructions = typesafe.Text("x")
	for _, line := range data {
		t.Run(line.name, func(t *testing.T) {
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				w.Header().Set("Content-Type", "application/json")
				_, _ = w.Write([]byte(line.body))
			}))
			defer srv.Close()
			c, err := typesafe.New(t.Context(),
				genai.ProviderOptionAPIKey("x"),
				genai.ProviderOptionModel("jev-latest"),
				genai.ProviderOptionRemote(srv.URL),
			)
			if err != nil {
				t.Fatal(err)
			}
			_, err = c.GenSync(t.Context(), text("test"), &genai.GenOptionText{DecodeAs: &q})
			if err == nil {
				t.Fatal("expected error")
			}
			if !strings.Contains(err.Error(), line.wantErr) {
				t.Fatalf("want %q, got %q", line.wantErr, err)
			}
		})
	}
}

// bogusOption is an invalid genai.ProviderOption.
type bogusOption struct{}

func (bogusOption) Validate() error { return nil }

func init() {
	internal.BeLenient = false
}

func TestAccessors(t *testing.T) {
	t.Parallel()
	c, err := typesafe.New(t.Context(),
		genai.ProviderOptionAPIKey("x"),
		genai.ProviderOptionModalities{genai.ModalityText},
		genai.ModelCheap,
	)
	if err != nil {
		t.Fatal(err)
	}
	if got := c.Name(); got != "typesafe" {
		t.Errorf("unexpected name %q", got)
	}
	if got := c.ModelID(); got != "jev-latest" {
		t.Errorf("unexpected model %q", got)
	}
	if got := c.OutputModalities(); !slices.Equal(got, genai.Modalities{genai.ModalityText}) {
		t.Errorf("unexpected output modalities %s", got)
	}
	if c.HTTPClient() == nil {
		t.Error("expected an HTTP client")
	}
	// Preloaded models skip the model list request.
	c, err = typesafe.New(t.Context(),
		genai.ProviderOptionAPIKey("x"),
		genai.ProviderOptionPreloadedModels([]genai.Model{&typesafe.Model{Name: "jev-latest"}}),
	)
	if err != nil {
		t.Fatal(err)
	}
	models, err := c.ListModels(t.Context())
	if err != nil {
		t.Fatal(err)
	}
	if len(models) != 1 || models[0].GetID() != "jev-latest" {
		t.Errorf("unexpected models %#v", models)
	}
}

func TestModel(t *testing.T) {
	t.Parallel()
	m := &typesafe.Model{Name: "jev-latest", Description: "latest", ReleaseDate: "2026-09-10T18:38:01.391457+00:00"}
	if got := m.GetID(); got != "jev-latest" {
		t.Errorf("unexpected ID %q", got)
	}
	if got := m.String(); got != "jev-latest (2026-09-10)" {
		t.Errorf("unexpected string %q", got)
	}
	if got := m.Context(); got != 0 {
		t.Errorf("unexpected context %d", got)
	}
	// A release date that is not RFC3339 is left out.
	m = &typesafe.Model{Name: "jev-latest", ReleaseDate: "yesterday"}
	if got := m.String(); got != "jev-latest" {
		t.Errorf("unexpected string %q", got)
	}
}

func TestNoulCriteriaValidate(t *testing.T) {
	t.Parallel()
	var c *typesafe.NoulCriteria
	if err := c.Validate(); err == nil || !strings.Contains(err.Error(), "field Criteria: is nil") {
		t.Fatalf("unexpected error: %v", err)
	}
	if err := (&typesafe.NoulCriteria{}).Validate(); err == nil ||
		!strings.Contains(err.Error(), "at least one of True or False is required") {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestQuestionsFrom(t *testing.T) {
	type tagged struct {
		Billing typesafe.Noul   `json:"billing"`
		Tone    typesafe.Choice `json:"tone"`
		Urgency typesafe.Score  `json:"urgency"`
		Skipped typesafe.Noul   `json:"-"`
		Plain   string          `json:"plain"`
	}
	q := tagged{}
	q.Billing.Instructions = typesafe.Text("Is this request about billing?")
	q.Billing.Criteria = &typesafe.NoulCriteria{True: typesafe.Text("yes"), False: typesafe.Text("no")}
	q.Tone.Instructions = typesafe.Text("What is the tone?")
	q.Tone.Criteria = map[string]typesafe.Content{"calm": nil, "angry": typesafe.Text("hostile")}
	q.Urgency.Criteria = []typesafe.Content{typesafe.Text("can wait"), typesafe.Text("right now")}
	// The field with a json tag of - and the invalid one are skipped: hmm, the invalid one must error.
	_, err := typesafe.QuestionsFrom(&q)
	if err == nil || !strings.Contains(err.Error(), "field Plain: must be a Noul, a Choice or a Score, got a string") {
		t.Fatalf("unexpected error: %v", err)
	}
	// Without the invalid field, the three questions are derived, in declaration order, skipping json:"-".
	type valid struct {
		Billing typesafe.Noul   `json:"billing"`
		Tone    typesafe.Choice `json:"tone"`
		Urgency typesafe.Score  `json:"urgency"`
		Skipped typesafe.Noul   `json:"-"`
	}
	v := valid{Billing: q.Billing, Tone: q.Tone, Urgency: q.Urgency}
	questions, err := typesafe.QuestionsFrom(&v)
	if err != nil {
		t.Fatal(err)
	}
	want := `{"billing":{"type":"noul","instructions":"Is this request about billing?","criteria":{"true":"yes","false":"no"}},` +
		`"tone":{"type":"choice","instructions":"What is the tone?","criteria":{"angry":"hostile","calm":null}},` +
		`"urgency":{"type":"score","criteria":["can wait","right now"]}}`
	raw, err := json.Marshal(questions)
	if err != nil {
		t.Fatal(err)
	}
	if string(raw) != want {
		t.Fatalf("want %s, got %s", want, raw)
	}
	// The question name is the Go field name when there is no json tag.
	type untagged struct {
		Greeting typesafe.Noul
	}
	u := untagged{}
	u.Greeting.Instructions = typesafe.Text("Is this a greeting?")
	questions, err = typesafe.QuestionsFrom(&u)
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := questions["Greeting"]; !ok {
		t.Fatalf("expected the field name, got %#v", questions)
	}
}

func TestQuestionsFromErrors(t *testing.T) {
	type nested struct {
		Nested struct {
			Billing typesafe.Noul `json:"billing"`
		} `json:"nested"`
	}
	type unexported struct {
		Billing typesafe.Noul `json:"billing"`
		hidden  typesafe.Noul //nolint:unused // only there to be rejected by QuestionsFrom.
	}
	type empty struct {
		Billing typesafe.Noul `json:"-"`
	}
	type invalid struct {
		Billing typesafe.Noul `json:"billing"`
	}
	data := []struct {
		name    string
		in      any
		wantErr string
	}{
		{"nil", nil, "must be a pointer to a struct"},
		{"not a pointer", struct{}{}, "must be a pointer to a struct"},
		{"not a struct", &[]string{}, "must be a pointer to a struct"},
		{"unsupported field", &struct {
			A bool `json:"a"`
		}{}, "field A: must be a Noul, a Choice or a Score, got a bool"},
		{"nested", &nested{}, "field Nested: must be a Noul, a Choice or a Score"},
		{"unexported", &unexported{}, "field hidden: must be exported"},
		{"nothing to ask", &empty{}, "no Noul, Choice or Score field to ask"},
		{"invalid question", &invalid{}, "field Instructions or Noul: one is required"},
	}
	for _, line := range data {
		t.Run(line.name, func(t *testing.T) {
			_, err := typesafe.QuestionsFrom(line.in)
			if err == nil {
				t.Fatal("expected error")
			}
			if !strings.Contains(err.Error(), line.wantErr) {
				t.Fatalf("want %q, got %q", line.wantErr, err)
			}
		})
	}
}

func TestQuestionnaireDecode(t *testing.T) {
	// Decoding fills the answer fields of the same struct that declared the questions.
	var q struct {
		Billing typesafe.Noul   `json:"billing"`
		Tone    typesafe.Choice `json:"tone"`
		Urgency typesafe.Score  `json:"urgency"`
	}
	in := `{"billing":{"type":"noul","noul":0.98},` +
		`"tone":{"type":"choice","choice":"calm","confidence":0.42,"probabilities":{"angry":0.29,"calm":0.71}},` +
		`"urgency":{"type":"score","score":1.59,"confidence":0.38,"legend":{"0":"can wait","1":"today"},"probabilities":{"0":0.41,"1":0.59}}}`
	if err := json.Unmarshal([]byte(in), &q); err != nil {
		t.Fatal(err)
	}
	if q.Billing.Probability != 0.98 {
		t.Errorf("unexpected billing: %#v", q.Billing)
	}
	if q.Tone.Label != "calm" || q.Tone.Confidence != 0.42 || q.Tone.Probabilities["calm"] != 0.71 {
		t.Errorf("unexpected tone: %#v", q.Tone)
	}
	if q.Urgency.Value != 1.59 || q.Urgency.Confidence != 0.38 || q.Urgency.Legend["1"] != typesafe.Text("today") {
		t.Errorf("unexpected urgency: %#v", q.Urgency)
	}
	// Malformed JSON and an answer of the wrong kind are reported instead of being silently ignored.
	for _, line := range []struct {
		name    string
		in      string
		out     any
		wantErr string
	}{
		{"malformed", `{`, &q, "unexpected end of JSON input"},
		{"malformed noul", `{"billing":1}`, &struct {
			Billing typesafe.Noul `json:"billing"`
		}{}, "cannot unmarshal number"},
		{"malformed choice", `{"tone":1}`, &struct {
			Tone typesafe.Choice `json:"tone"`
		}{}, "cannot unmarshal number"},
		{"malformed score", `{"urgency":1}`, &struct {
			Urgency typesafe.Score `json:"urgency"`
		}{}, "cannot unmarshal number"},
		{"wrong kind for a noul", `{"billing":{"type":"choice","choice":"calm"}}`, &struct {
			Billing typesafe.Noul `json:"billing"`
		}{}, `expected a noul answer, got "choice"`},
		{"wrong kind for a choice", `{"tone":{"type":"score","score":1}}`, &struct {
			Tone typesafe.Choice `json:"tone"`
		}{}, `expected a choice answer, got "score"`},
		{"wrong kind for a score", `{"urgency":{"type":"noul","noul":1}}`, &struct {
			Urgency typesafe.Score `json:"urgency"`
		}{}, `expected a score answer, got "noul"`},
	} {
		t.Run(line.name, func(t *testing.T) {
			err := json.Unmarshal([]byte(line.in), line.out)
			if err == nil || !strings.Contains(err.Error(), line.wantErr) {
				t.Fatalf("want %q, got %v", line.wantErr, err)
			}
		})
	}
}
