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
	"github.com/maruel/genai/smoke/smoketest"
)

func apiKey() string {
	if key := internaltest.GetEnv("TYPESAFE_API_KEY"); key != "" {
		return key
	}
	return "<insert_api_key_here>"
}

func getClientInner(t *testing.T, opts []genai.ProviderOption, fn func(http.RoundTripper) http.RoundTripper) (*typesafe.Client, error) {
	if !slices.ContainsFunc(opts, func(o genai.ProviderOption) bool { _, ok := o.(genai.ProviderOptionAPIKey); return ok }) {
		opts = append(opts, genai.ProviderOptionAPIKey(apiKey()))
	}
	if fn != nil {
		opts = append(opts, genai.ProviderOptionTransportWrapper(fn))
	}
	c, err := typesafe.New(t.Context(), opts...)
	if c != nil {
		internaltest.CleanupCloser(t, c)
	}
	return c, err
}

func TestClient(t *testing.T) {
	t.Run("SystemOne invalid response", testClientSystemOneInvalidResponse)
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
	t.Run("Scoreboard", func(t *testing.T) {
		models := make([]scoreboard.Model, 0, len(cachedModels))
		for _, m := range cachedModels {
			models = append(models, scoreboard.Model{Model: m.GetID()})
		}
		smoketest.Run(t, func(t testing.TB, m scoreboard.Model, fn func(http.RoundTripper) http.RoundTripper) genai.Provider {
			opts := []genai.ProviderOption{genai.ProviderOptionAPIKey(apiKey()), genai.ProviderOptionPreloadedModels(cachedModels)}
			if fn != nil {
				opts = append(opts, genai.ProviderOptionTransportWrapper(fn))
			}
			if m.Model != "" {
				opts = append(opts, genai.ProviderOptionModel(m.Model))
			}
			c, err := typesafe.New(t.Context(), opts...)
			if c != nil {
				internaltest.CleanupCloser(t, c)
			}
			if err != nil {
				t.Fatal(err)
			}
			return c
		}, models, testRecorder.Records, nil)
	})

	t.Run("SystemOne", func(t *testing.T) {
		c := getClient(t, "jev-latest")
		q := genai.Questions{
			"billing": {Type: genai.QuestionNoul, Instructions: genai.Text("Is this request about billing?"), Noul: &genai.NoulCriteria{
				True:  genai.Text("the customer is asking about a charge or an invoice"),
				False: genai.Text("the customer is asking about anything else"),
			}},
			"tone": {Type: genai.QuestionChoice, Instructions: genai.Text("What is the tone of the customer?"), Choice: map[string]genai.DecisionContent{
				"calm":       nil,
				"frustrated": genai.Text("annoyed but polite"),
				"angry":      nil,
			}},
			"urgency": {Type: genai.QuestionScore, Instructions: genai.Text("How soon does this need to be handled?"), Score: []genai.DecisionContent{
				genai.Text("can wait"), genai.Text("this week"), genai.Text("today"), genai.Text("right now"),
			}},
		}

		res, err := c.SystemOne(t.Context(), &genai.SystemOneRequest{State: genai.Text("I was charged twice for order A-104. Please refund the duplicate charge. This is unacceptable."), Questions: q})
		if err != nil {
			t.Fatal(err)
		}
		if res.Usage.InputTokens == 0 {
			t.Error("expected input tokens to be reported")
		}

		if len(res.Answers) != 3 {
			t.Fatalf("expected 3 answers, got %d: %+v", len(res.Answers), res)
		}
		if b := res.Answers["billing"]; b.Type != genai.QuestionNoul || b.Noul < 0.5 {
			t.Errorf("unexpected billing answer: %#v", b)
		}
		if tone := res.Answers["tone"]; tone.Type != genai.QuestionChoice {
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
		if u := res.Answers["urgency"]; u.Type != genai.QuestionScore {
			t.Errorf("unexpected urgency answer: %#v", u)
		} else if u.Score < 0 || u.Score > 3 {
			t.Errorf("score out of range: %f", u.Score)
		} else if len(u.Legend) != 4 {
			t.Errorf("unexpected legend: %#v", u.Legend)
		}
	})

	t.Run("SystemOne-StructuredState", func(t *testing.T) {
		// Structured state passes directly to the decision API.
		c := getClient(t, "jev-latest")
		q := genai.Questions{
			"refund": {Type: genai.QuestionNoul, Instructions: genai.Text("Does the policy say duplicate charges are eligible for a refund?")},
		}

		res, err := c.SystemOne(t.Context(), &genai.SystemOneRequest{State: genai.Object{"order": genai.Object{"id": "A-104", "amount_usd": 49}, "refund_policy": "Duplicate charges are eligible for a refund."}, Questions: q})
		if err != nil {
			t.Fatal(err)
		}

		if res.Answers["refund"].Noul < 0.5 {
			t.Errorf("unexpected refund answer: %#v", res.Answers["refund"])
		}
	})

	t.Run("Questions", func(t *testing.T) {
		c := getClient(t, "jev-latest")
		// A single question needs no criteria.
		questions := genai.Questions{
			"greeting": {Type: genai.QuestionNoul, Instructions: genai.Text("Is this a greeting?")},
		}
		res, err := c.SystemOne(t.Context(), &genai.SystemOneRequest{State: genai.Text("Hello there!"), Questions: questions})
		if err != nil {
			t.Fatal(err)
		}

		if res.Answers["greeting"].Noul < 0.5 {
			t.Errorf("expected a greeting, got %#v", res.Answers["greeting"])
		}
	})

	t.Run("SystemOneRaw", func(t *testing.T) {
		t.Run("limits at request boundary", func(t *testing.T) {
			srv := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) { t.Error("unexpected HTTP request") }))
			t.Cleanup(srv.Close)
			c, err := typesafe.New(t.Context(), genai.ProviderOptionAPIKey("fake"), genai.ProviderOptionModel("jev-latest"), genai.ProviderOptionRemote(srv.URL))
			if err != nil {
				t.Fatal(err)
			}
			internaltest.CleanupCloser(t, c)
			req := typesafe.SystemOneRequest{Model: "jev-latest", State: genai.Text("state")}
			qs := genai.Questions{"q": {Type: genai.QuestionScore, Score: []genai.DecisionContent{nil}}}
			req.Questions = qs
			err = c.SystemOneRaw(t.Context(), &req, &typesafe.SystemOneResponse{})
			if err == nil || !strings.Contains(err.Error(), "field Score[0]: must not be nil") {
				t.Fatalf("missing TypeSafe boundary limit: %v", err)
			}
		})
		c := getClient(t, "jev-latest")
		// Structured criteria: a score rubric of objects and a choice with object and array descriptions.
		// They are echoed back verbatim in the legend. A noul can also be described by its criteria alone.
		resp := &typesafe.SystemOneResponse{}
		err := c.SystemOneRaw(t.Context(), &typesafe.SystemOneRequest{Model: "jev-latest", State: genai.Object{"msg": genai.Text("My card was charged twice for order A-104.")},
			Questions: genai.Questions{
				"urgency": {
					Type:         genai.QuestionScore,
					Instructions: genai.Text("How soon does this need to be handled?"),
					Score: []genai.DecisionContent{
						genai.Object{"level": genai.Text("can wait"), "sla_days": 30},
						genai.Object{"level": genai.Text("this week"), "sla_days": 7},
						genai.Object{"level": genai.Text("right now"), "sla_days": 0},
					},
				},
				"tone": {
					Type:         genai.QuestionChoice,
					Instructions: genai.Object{"task": genai.Text("classify the tone")},
					Choice: map[string]genai.DecisionContent{
						"calm":  genai.Object{"note": genai.Text("relaxed")},
						"angry": genai.Array{genai.Text("loud"), genai.Text("rude")},
					},
				},
				"greeting": {
					Type: genai.QuestionNoul,
					Noul: &genai.NoulCriteria{True: genai.Text("it is a greeting"), False: genai.Text("it is not")},
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
		if o, ok := resp.Answers["urgency"].Legend["2"].(genai.Object); !ok || o["level"] != "right now" {
			t.Errorf("expected the object level to be echoed back, got %#v", resp.Answers["urgency"].Legend)
		}
		if c := resp.Answers["tone"].Choice; c != "calm" && c != "angry" {
			t.Errorf("unexpected choice %q", c)
		}
		if n := resp.Answers["greeting"].Noul; n < 0 || n > 1 {
			t.Errorf("unexpected noul %f", n)
		}
	})

	t.Run("Questions-all-types", func(t *testing.T) {
		c := getClient(t, "jev-latest")
		q := genai.Questions{
			"billing": {Type: genai.QuestionNoul, Instructions: genai.Text("Is this request about billing?")},
			"tone":    {Type: genai.QuestionChoice, Instructions: genai.Text("What is the tone of the customer?"), Choice: map[string]genai.DecisionContent{"calm": nil, "angry": nil}},
			"urgency": {Type: genai.QuestionScore, Instructions: genai.Text("How soon does this need to be handled?"), Score: []genai.DecisionContent{genai.Text("can wait"), genai.Text("today")}},
		}

		res, err := c.SystemOne(t.Context(), &genai.SystemOneRequest{State: genai.Text("I was charged twice for order A-104."), Questions: q})
		if err != nil {
			t.Fatal(err)
		}

		if res.Answers["billing"].Noul < 0.5 {
			t.Errorf("unexpected billing answer: %#v", res.Answers["billing"])
		}
		if res.Answers["tone"].Choice != "calm" && res.Answers["tone"].Choice != "angry" {
			t.Errorf("unexpected tone answer: %#v", res.Answers["tone"])
		}
		if res.Answers["tone"].Probabilities[res.Answers["tone"].Choice] == 0 {
			t.Errorf("unexpected tone probabilities: %#v", res.Answers["tone"].Probabilities)
		}
		if res.Answers["urgency"].Score < 0 || res.Answers["urgency"].Score > 1 {
			t.Errorf("unexpected urgency answer: %#v", res.Answers["urgency"])
		}
		if len(res.Answers["urgency"].Legend) != 2 {
			t.Errorf("unexpected urgency legend: %#v", res.Answers["urgency"].Legend)
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
		questions := genai.Questions{
			"a": {Type: genai.QuestionNoul, Instructions: genai.Text("Is this a test?")},
		}
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
			_, err = c.SystemOne(t.Context(), &genai.SystemOneRequest{State: genai.Text("test"), Questions: questions})
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
			_, err = c.SystemOne(t.Context(), &genai.SystemOneRequest{State: genai.Text("test"), Questions: questions})
			if err == nil {
				t.Fatal("expected error")
			}
			if got := err.Error(); !strings.Contains(got, "http 400") || !strings.Contains(got, "Unknown model: bad model") {
				t.Fatalf("unexpected error: %q", got)
			}
		})
		t.Run("invalid raw request", func(t *testing.T) {
			// The request boundary rejects structurally invalid state before HTTP.
			c, err := getClientInner(t, []genai.ProviderOption{
				genai.ProviderOptionModel("jev-latest"),
			}, func(h http.RoundTripper) http.RoundTripper {
				return testRecorder.Record(t, h)
			})
			if err != nil {
				t.Fatal(err)
			}
			err = c.SystemOneRaw(t.Context(), &typesafe.SystemOneRequest{}, &typesafe.SystemOneResponse{})
			if err == nil {
				t.Fatal("expected error")
			}
			if got := err.Error(); got != "state or one application/json document is required" {
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
			err = c.SystemOneRaw(t.Context(), &typesafe.SystemOneRequest{Model: "jev-latest", State: genai.Text("test"),
				Questions: genai.Questions{
					"a": {Type: genai.QuestionNoul, Instructions: genai.Text("")},
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
		q := genai.Questions{
			"a": {Type: genai.QuestionNoul, Instructions: genai.Text("x")},
		}

		_, err = c.SystemOne(t.Context(), &genai.SystemOneRequest{State: genai.Text("test"), Questions: q})
		if got := err.Error(); got != "a model is required" {
			t.Fatalf("unexpected error: %q", got)
		}
	})
}

func TestSystemOneRequest(t *testing.T) {
	t.Run("Validate", func(t *testing.T) {
		data := []struct {
			name      string
			questions genai.Questions
			wantErr   string
		}{
			{
				name:    "no questions",
				wantErr: "at least one question is required",
			},
			{
				name: "noul with a choice field",
				questions: genai.Questions{
					"a": {Type: genai.QuestionNoul, Choice: map[string]genai.DecisionContent{"b": nil}},
				},
				wantErr: "fields Choice and Score: can't be set on a noul question",
			},
			{
				name: "score with a noul field",
				questions: genai.Questions{
					"a": {
						Type:  genai.QuestionScore,
						Noul:  &genai.NoulCriteria{True: genai.Text("yes")},
						Score: []genai.DecisionContent{genai.Text("x")},
					},
				},
				wantErr: "fields Noul and Choice: can't be set on a score question",
			},
			{
				name: "noul with invalid criteria",
				questions: genai.Questions{
					"a": {
						Type: genai.QuestionNoul,
						Noul: &genai.NoulCriteria{True: genai.Object(nil), False: genai.Array(nil)},
					},
				},
				wantErr: "field Criteria.True: Object is nil, use nil to leave the field unset",
			},
			{
				name: "choice with an invalid description",
				questions: genai.Questions{
					"a": {Type: genai.QuestionChoice, Choice: map[string]genai.DecisionContent{"b": genai.Object(nil)}},
				},
				wantErr: "field Choice[b]: Object is nil, use nil to leave the field unset",
			},
			{
				name: "score with an invalid level",
				questions: genai.Questions{
					"a": {Type: genai.QuestionScore, Score: []genai.DecisionContent{genai.Text("x"), genai.Array(nil)}},
				},
				wantErr: "field Score[1]: Array is nil, use nil to leave the field unset",
			},
			{
				name: "no type",
				questions: genai.Questions{
					"a": {Instructions: genai.Text("x")},
				},
				wantErr: `field Type: must be "noul", "choice" or "score", got ""`,
			},
			{
				name: "noul empty criteria",
				questions: genai.Questions{
					"a": {Type: genai.QuestionNoul, Noul: &genai.NoulCriteria{}},
				},
				wantErr: "at least one of True or False is required",
			},
			{
				name: "type mismatch",
				questions: genai.Questions{
					"a": {Type: genai.QuestionChoice, Score: []genai.DecisionContent{genai.Text("x")}},
				},
				wantErr: "fields Noul and Score: can't be set on a choice question",
			},
			{
				name: "score without criteria",
				questions: genai.Questions{
					"a": {Type: genai.QuestionScore, Instructions: genai.Text("x")},
				},
				wantErr: "field Score: at least one level is required",
			},
			{
				name: "nil object",
				questions: genai.Questions{
					"a": {Type: genai.QuestionNoul, Instructions: genai.Object(nil)},
				},
				wantErr: "field Instructions: Object is nil, use nil to leave the field unset",
			},
		}
		for _, line := range data {
			t.Run(line.name, func(t *testing.T) {
				err := (&typesafe.SystemOneRequest{State: genai.Text("state"), Questions: line.questions}).Validate()
				if err == nil {
					t.Fatal("expected error")
				}
				if !strings.Contains(err.Error(), line.wantErr) {
					t.Fatalf("want %q, got %q", line.wantErr, err)
				}
			})
		}
	})
}

func TestAnswerMarshal(t *testing.T) {
	data := []struct {
		name string
		in   string
		want string
	}{
		{
			"choice", `{"type":"choice","choice":"a","confidence":1,"probabilities":{"a":1,"b":0}}`,
			`{"type":"choice","choice":"a","confidence":1,"probabilities":{"a":1,"b":0}}`,
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
			var a genai.Answer
			if err := json.Unmarshal([]byte(line.in), &a); err != nil {
				t.Fatal(err)
			}
			raw, err := json.Marshal(&a)
			if err != nil {
				t.Fatal(err)
			}
			if string(raw) != line.want {
				t.Fatalf("want %s, got %s", line.want, raw)
			}
		})
	}
	t.Run("unknown", func(t *testing.T) {
		if _, err := json.Marshal(&genai.Answer{Type: "bogus"}); err == nil {
			t.Fatal("expected error")
		}
	})
	t.Run("not an answer", func(t *testing.T) {
		var a genai.Answer
		if err := json.Unmarshal([]byte(`1`), &a); err == nil {
			t.Fatal("expected error")
		}
	})
	t.Run("unknown field", func(t *testing.T) {
		var a genai.Answer
		if err := json.Unmarshal([]byte(`{"type":"noul","noul":0.5,"bogus":1}`), &a); err == nil {
			t.Fatal("expected error")
		} else if !strings.Contains(err.Error(), "unknown field") {
			t.Fatalf("unexpected error: %v", err)
		}
	})
	t.Run("unknown from json", func(t *testing.T) {
		// An answer type this client does not know is kept as-is instead of breaking the whole reply.
		var a genai.Answer
		in := `{"type":"ranks","order":["a","b"]}`
		if err := json.Unmarshal([]byte(in), &a); err != nil {
			t.Fatal(err)
		}
		raw, err := json.Marshal(&a)
		if err != nil {
			t.Fatal(err)
		}
		if string(raw) != in {
			t.Fatalf("want %s, got %s", in, raw)
		}
	})
	t.Run("answers", func(t *testing.T) {
		var a genai.Answers
		if err := json.Unmarshal([]byte(`{"a":{"type":"noul","noul":0.25}}`), &a); err != nil {
			t.Fatal(err)
		}
		raw, err := json.Marshal(&a)
		if err != nil {
			t.Fatal(err)
		}
		if string(raw) != `{"a":{"type":"noul","noul":0.25}}` {
			t.Fatalf("unexpected: %s", raw)
		}
	})
}

func TestContentValidate(t *testing.T) {
	data := []struct {
		name    string
		in      genai.DecisionContent
		wantErr string
	}{
		{
			"several bad keys",
			genai.Object{"b": make(chan int), "a": make(chan int)},
			"key \"a\": json: unsupported type: chan int\nkey \"b\": json: unsupported type: chan int",
		},
		{"array with NaN", genai.Array{math.NaN()}, "index 0: json: unsupported value: NaN"},
	}
	for _, line := range data {
		t.Run(line.name, func(t *testing.T) {
			err := line.in.Validate()
			if err == nil {
				t.Fatal("expected error")
			}
			if got := err.Error(); got != line.wantErr {
				t.Fatalf("want %q, got %q", line.wantErr, got)
			}
		})
	}
}

func TestScoreLegend(t *testing.T) {
	// The legend is the Question.Score criteria echoed back, as Text, Object or Array.
	in := `{"0":"low","1":{"desc":"high","nested":{"x":[1,2]}},"2":["a",{"b":true}],"3":null}`
	var l genai.ScoreLegend
	if err := json.Unmarshal([]byte(in), &l); err != nil {
		t.Fatal(err)
	}
	if v, ok := l["0"].(genai.Text); !ok || v != "low" {
		t.Errorf("unexpected level 0: %#v", l["0"])
	}
	o, ok := l["1"].(genai.Object)
	if !ok || o["desc"] != "high" {
		t.Fatalf("unexpected level 1: %#v", l["1"])
	}
	// Nested values are left as decoded, like Object and Array document.
	if nested, ok := o["nested"].(map[string]any); !ok || len(nested) != 1 {
		t.Errorf("unexpected nested value: %#v", o["nested"])
	}
	a, ok := l["2"].(genai.Array)
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
		var l genai.ScoreLegend
		err := json.Unmarshal([]byte(in), &l)
		if err == nil {
			t.Fatalf("expected error for %s", in)
		}
		if !strings.Contains(err.Error(), "expected a string, a JSON object or a JSON array") {
			t.Fatalf("unexpected error: %v", err)
		}
	}
	// The legend must be an object.
	var l3 genai.ScoreLegend
	if err := json.Unmarshal([]byte(`[]`), &l3); err == nil {
		t.Fatal("expected error")
	}
	// A null legend is no legend.
	var l2 genai.ScoreLegend
	if err := json.Unmarshal([]byte(`null`), &l2); err != nil {
		t.Fatal(err)
	}
	if l2 != nil {
		t.Errorf("expected nil, got %#v", l2)
	}
}

func TestQuestionMarshal(t *testing.T) {
	t.Run("unknown type", func(t *testing.T) {
		if _, err := json.Marshal(&genai.Question{Type: "bogus"}); err == nil {
			t.Fatal("expected error")
		}
	})
	t.Run("criteria that can't be encoded", func(t *testing.T) {
		// MarshalJSON reports the error of the criteria it encodes.
		q := genai.Question{
			Type:   genai.QuestionChoice,
			Choice: map[string]genai.DecisionContent{"a": genai.Object{"b": make(chan int)}},
		}
		if _, err := json.Marshal(&q); err == nil {
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

func TestNewErrors(t *testing.T) {
	t.Run("unsupported option", func(t *testing.T) {
		cl, err := typesafe.New(t.Context(), bogusOption{})
		if cl != nil {
			internaltest.CleanupCloser(t, cl)
		}
		if err == nil || !strings.Contains(err.Error(), "unsupported option type") {
			t.Fatalf("unexpected error: %v", err)
		}
	})
	t.Run("duplicate options", func(t *testing.T) {
		cl, err := typesafe.New(t.Context(), genai.ProviderOptionAPIKey("a"), genai.ProviderOptionAPIKey("b"))
		if cl != nil {
			internaltest.CleanupCloser(t, cl)
		}
		if err == nil || !strings.Contains(err.Error(), "duplicate provider option") {
			t.Fatalf("unexpected error: %v", err)
		}
	})
	t.Run("invalid option", func(t *testing.T) {
		cl, err := typesafe.New(t.Context(), genai.ProviderOptionAPIKey(""))
		if cl != nil {
			internaltest.CleanupCloser(t, cl)
		}
		if err == nil || !strings.Contains(err.Error(), "cannot be empty") {
			t.Fatalf("unexpected error: %v", err)
		}
	})
	t.Run("missing api key", func(t *testing.T) {
		t.Setenv("TYPESAFE_API_KEY", "")
		c, err := typesafe.New(t.Context(), genai.ProviderOptionModel("jev-latest"))
		if c != nil {
			internaltest.CleanupCloser(t, c)
		}
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

func testClientSystemOneInvalidResponse(t *testing.T) {
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
	q := genai.Questions{
		"a": {Type: genai.QuestionNoul, Instructions: genai.Text("x")},
	}

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
			if c != nil {
				internaltest.CleanupCloser(t, c)
			}
			if err != nil {
				t.Fatal(err)
			}
			_, err = c.SystemOne(t.Context(), &genai.SystemOneRequest{State: genai.Text("test"), Questions: q})
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
		genai.ProviderOptionModalities{genai.ModalityDecision},
		genai.ModelCheap,
	)
	if c != nil {
		internaltest.CleanupCloser(t, c)
	}
	if err != nil {
		t.Fatal(err)
	}
	if got := c.Name(); got != "typesafe" {
		t.Errorf("unexpected name %q", got)
	}
	if got := c.ModelID(); got != "jev-latest" {
		t.Errorf("unexpected model %q", got)
	}
	if got := c.OutputModalities(); !slices.Equal(got, genai.Modalities{genai.ModalityDecision}) {
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
	if c != nil {
		internaltest.CleanupCloser(t, c)
	}
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
