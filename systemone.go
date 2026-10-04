// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Shared System One decision content, questions, requests and answers.

package genai

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"maps"
	"path/filepath"
	"slices"

	"github.com/maruel/genai/internal"
)

// DecisionContent is text, a JSON object, or a JSON array.
//
// It is implemented by Text, Object and Array, and by JSON state read from a Doc.
// It marshals to the value itself, not to an object with fields. Use nil when the field is optional and is left unset.
//
// It is used for state, instructions and criteria descriptions. It is meant to be sent; the API only
// returns it in Answer.Legend, decoded back by ScoreLegend, since an interface cannot be decoded on its
// own.
type DecisionContent interface {
	// Validate ensures the content is valid.
	Validate() error
	// content restricts DecisionContent to the types of this package.
	content()
}

// Text is DecisionContent that is a JSON string.
type Text string

// content implements DecisionContent.
func (Text) content() {}

// Validate implements internal.Validatable.
//
// Any text is valid, including the empty string, the API is the one that decides whether a specific field
// may be empty.
func (Text) Validate() error {
	return nil
}

// Object is DecisionContent that is a JSON object.
//
// The values are any, like Reply.Opaque: they must be JSON encodable. Go has no recursive JSON
// value type that does not require wrapping every scalar, so a nested map[string]any, []any or a struct
// with JSON tags is passed as is. It is the equivalent of the SDKs' Mapping[str, JSONValue | None].
type Object map[string]any

// content implements DecisionContent.
func (Object) content() {}

// Validate implements internal.Validatable.
//
// Every value is validated so that the error names the offending key.
func (o Object) Validate() error {
	if o == nil {
		return errors.New("Object is nil, use nil to leave the field unset")
	}
	var errs []error
	for _, k := range slices.Sorted(maps.Keys(o)) {
		// The values are any, so encoding/json is the authority on what they may be.
		if _, err := json.Marshal(o[k]); err != nil {
			errs = append(errs, fmt.Errorf("key %q: %w", k, err))
		}
	}
	return errors.Join(errs...)
}

// Array is DecisionContent that is a JSON array.
//
// The values are any, like Object: they must be JSON encodable. They are not restricted to DecisionContent items.
type Array []any

// content implements DecisionContent.
func (Array) content() {}

// Validate implements internal.Validatable.
//
// Every item is validated so that the error names the offending index.
func (a Array) Validate() error {
	if a == nil {
		return errors.New("Array is nil, use nil to leave the field unset")
	}
	var errs []error
	for i := range a {
		// The values are any, so encoding/json is the authority on what they may be.
		if _, err := json.Marshal(a[i]); err != nil {
			errs = append(errs, fmt.Errorf("index %d: %w", i, err))
		}
	}
	return errors.Join(errs...)
}

// QuestionType is the type of a Question.
type QuestionType string

// Question types.
const (
	// QuestionNoul is a yes/no question. The answer is the probability that the answer is yes.
	QuestionNoul QuestionType = "noul"
	// QuestionChoice picks one option from a set defined by the question.
	QuestionChoice QuestionType = "choice"
	// QuestionScore rates the state along an ordered rubric.
	QuestionScore QuestionType = "score"
)

// Question is a typed question about a state.
//
// Exactly one of Noul, Choice or Score can be set, the one matching Type. Instructions is shared by the
// three types.
//
// Type is explicit instead of inferred from the field that is set: a choice or score question whose
// criteria are nil at runtime, which append and conditional map building produce, would otherwise be
// silently asked as a noul question instead of being rejected.
//
// Recommended reading: https://docs.typesafe.ai/primitives/advanced
type Question struct {
	// Type is how the state is evaluated. It selects which of Noul, Choice or Score must be set.
	Type QuestionType
	// Instructions is what the model should decide or rate. Providers may require it.
	Instructions DecisionContent
	// Noul describes what the yes and the no outcomes mean. It can only be set for QuestionNoul, where it
	// is optional.
	//
	// Recommended reading: https://docs.typesafe.ai/primitives/noul
	Noul *NoulCriteria
	// Choice maps the options of a choice question to their descriptions. It can only be set for
	// QuestionChoice, where at least one option is required. Use nil for an option that needs no extra
	// detail.
	//
	// Recommended reading: https://docs.typesafe.ai/primitives/choice
	Choice map[string]DecisionContent
	// Score lists the levels of a score question rubric, in order, starting at level 0. It can only be set
	// for QuestionScore, where at least one level is required. Use nil for an undescribed level.
	// Providers may impose stricter limits.
	//
	// Recommended reading: https://docs.typesafe.ai/primitives/score
	Score []DecisionContent
}

// Validate checks shared content and union consistency, not provider-specific limits.
func (q *Question) Validate() error {
	var errs []error
	if q.Instructions != nil {
		if err := validateContent(q.Instructions); err != nil {
			errs = append(errs, fmt.Errorf("field Instructions: %w", err))
		}
	}
	switch q.Type {
	case QuestionNoul:
		if q.Instructions == nil && q.Noul == nil {
			errs = append(errs, errors.New("field Instructions or Noul: one is required"))
		}
		if q.Choice != nil || q.Score != nil {
			errs = append(errs, errors.New("fields Choice and Score: can't be set on a noul question"))
		}
		if q.Noul != nil {
			if err := q.Noul.Validate(); err != nil {
				errs = append(errs, err)
			}
		}
	case QuestionChoice:
		if q.Noul != nil || q.Score != nil {
			errs = append(errs, errors.New("fields Noul and Score: can't be set on a choice question"))
		}
		if len(q.Choice) == 0 {
			errs = append(errs, errors.New("field Choice: at least one option is required"))
		}
		for _, name := range slices.Sorted(maps.Keys(q.Choice)) {
			if v := q.Choice[name]; v != nil {
				if err := validateContent(v); err != nil {
					errs = append(errs, fmt.Errorf("field Choice[%s]: %w", name, err))
				}
			}
		}
	case QuestionScore:
		if q.Noul != nil || q.Choice != nil {
			errs = append(errs, errors.New("fields Noul and Choice: can't be set on a score question"))
		}
		if len(q.Score) == 0 {
			errs = append(errs, errors.New("field Score: at least one level is required"))
		}
		for i, c := range q.Score {
			if c != nil {
				if err := validateContent(c); err != nil {
					errs = append(errs, fmt.Errorf("field Score[%d]: %w", i, err))
				}
			}
		}
	default:
		errs = append(errs, fmt.Errorf("field Type: must be %q, %q or %q, got %q", QuestionNoul, QuestionChoice, QuestionScore, q.Type))
	}
	return errors.Join(errs...)
}

// MarshalJSON implements json.Marshaler.
func (q *Question) MarshalJSON() ([]byte, error) {
	// Criteria stays nil, and so is omitted, for a noul question without criteria.
	var criteria json.RawMessage
	var err error
	switch q.Type {
	case QuestionNoul:
		if q.Noul != nil {
			criteria, err = json.Marshal(q.Noul)
		}
	case QuestionChoice:
		criteria, err = json.Marshal(q.Choice)
	case QuestionScore:
		criteria, err = json.Marshal(q.Score)
	default:
		return nil, fmt.Errorf("unknown question type %q", q.Type)
	}
	if err != nil {
		return nil, err
	}
	return json.Marshal(questionJSON{Type: q.Type, Instructions: q.Instructions, Criteria: criteria})
}

// Questions is a set of questions to ask about one state, keyed by the name to report each answer under.
//
// The names are chosen by the caller. They are not sent to the model, they are only used to key the
// answers.
//
// Set SystemOneRequest.Questions directly. SystemOneResponse.Answers holds the typed answers
// under the same dynamic names. Entries must not be nil.
type Questions map[string]*Question

// Validate ensures the questions are valid.
func (q Questions) Validate() error {
	if len(q) == 0 {
		return errors.New("at least one question is required")
	}
	var errs []error
	for _, name := range slices.Sorted(maps.Keys(q)) {
		v := q[name]
		if v == nil {
			errs = append(errs, fmt.Errorf("question %q: must not be nil", name))
			continue
		}
		if err := v.Validate(); err != nil {
			errs = append(errs, fmt.Errorf("question %q: %w", name, err))
		}
	}
	return errors.Join(errs...)
}

// NoulCriteria describes what the yes and the no answers of a noul question mean.
type NoulCriteria struct {
	// True describes what a yes (value near 1) means.
	True DecisionContent `json:"true,omitzero"`
	// False describes what a no (value near 0) means.
	False DecisionContent `json:"false,omitzero"`
}

// Validate ensures the criteria are valid.
func (c *NoulCriteria) Validate() error {
	if c == nil {
		return errors.New("field Criteria: is nil")
	}
	var errs []error
	if c.True != nil {
		if err := validateContent(c.True); err != nil {
			errs = append(errs, fmt.Errorf("field Criteria.True: %w", err))
		}
	}
	if c.False != nil {
		if err := validateContent(c.False); err != nil {
			errs = append(errs, fmt.Errorf("field Criteria.False: %w", err))
		}
	}
	return errors.Join(errs...)
}

// Answers holds the answer to each question, keyed by the question name. Entries must not be nil.
type Answers map[string]*Answer

// Answer is the answer to a Question.
//
// It is a union discriminated by Type. Only the fields that match Type are set, and MarshalJSON writes
// only those fields, with the shape the API uses, so a probability or a score of 0 is not dropped.
//
// An answer type the server adds later is kept as is instead of making the whole reply unusable.
type Answer struct {
	// Type is the type of the question this is the answer to.
	Type QuestionType `json:"type"`
	// Noul is the probability that the answer is yes, from 0 to 1. It is set for QuestionNoul.
	Noul float64 `json:"noul,omitzero"`
	// Choice is the highest probability option. It is set for QuestionChoice.
	Choice string `json:"choice,omitzero"`
	// Score is the probability weighted value across the levels. It can land between levels. It is set for
	// QuestionScore.
	Score float64 `json:"score,omitzero"`
	// Confidence is how certain the model is, derived from Probabilities. It is set for QuestionChoice and
	// QuestionScore.
	Confidence float64 `json:"confidence,omitzero"`
	// Probabilities maps each option, or each level as a string key, to its probability. It is set for
	// QuestionChoice and QuestionScore.
	Probabilities map[string]float64 `json:"probabilities,omitzero"`
	// Legend maps each level as a string key to the description passed in Question.Score for that level.
	// It is set for QuestionScore.
	Legend ScoreLegend `json:"legend,omitzero"`

	// raw is the answer as returned by the API. It is only set when Type is not one of the known types, so
	// that a new answer type does not make the whole reply unusable.
	raw json.RawMessage
}

// UnmarshalJSON implements json.Unmarshaler.
func (a *Answer) UnmarshalJSON(b []byte) error {
	t := answerTypeJSON{}
	if err := json.Unmarshal(b, &t); err != nil {
		return err
	}
	switch t.Type {
	case QuestionNoul, QuestionChoice, QuestionScore:
	default:
		// Answer type added by the server after this client was written. Keep it as-is instead of
		// making the whole reply unusable.
		a.Type = t.Type
		a.raw = append(json.RawMessage(nil), b...)
		return nil
	}
	type alias Answer
	v := alias{}
	if err := internal.UnmarshalJSON(b, &v); err != nil {
		return err
	}
	*a = Answer(v)
	return nil
}

// MarshalJSON implements json.Marshaler.
//
// The fields of the union that do not match Type are omitted, even when they are zero, so that a score
// of 0 or a probability of 0 is preserved.
func (a *Answer) MarshalJSON() ([]byte, error) {
	switch a.Type {
	case QuestionNoul:
		return json.Marshal(noulAnswerJSON{Type: a.Type, Noul: a.Noul})
	case QuestionChoice:
		return json.Marshal(choiceAnswerJSON{
			Type: a.Type, Choice: a.Choice, Confidence: a.Confidence, Probabilities: a.Probabilities,
		})
	case QuestionScore:
		return json.Marshal(scoreAnswerJSON{
			Type: a.Type, Score: a.Score, Confidence: a.Confidence, Legend: a.Legend, Probabilities: a.Probabilities,
		})
	default:
		if len(a.raw) != 0 {
			// Answer type added by the server after this client was written.
			return a.raw, nil
		}
		return nil, fmt.Errorf("unknown answer type %q", a.Type)
	}
}

// ScoreLegend is the description of each level of a score answer, keyed by the level.
//
// It is the Question.Score criteria of the question, echoed back by the API. The values are decoded as
// Text, Object or Array; a nil value is a level the caller left undescribed.
type ScoreLegend map[string]DecisionContent

// UnmarshalJSON implements json.Unmarshaler.
//
// DecisionContent is an interface, so encoding/json cannot decode it on its own; each level is decoded into the
// variant matching its JSON token here.
func (l *ScoreLegend) UnmarshalJSON(b []byte) error {
	raw := map[string]json.RawMessage{}
	if err := internal.UnmarshalJSON(b, &raw); err != nil {
		return err
	}
	if raw == nil {
		*l = nil
		return nil
	}
	out := make(ScoreLegend, len(raw))
	for k, v := range raw {
		// A level the caller left undescribed is null.
		if bytes.Equal(v, []byte("null")) {
			out[k] = nil
			continue
		}
		c, err := ParseDecisionContent(v)
		if err != nil {
			return fmt.Errorf("level %q: %w", k, err)
		}
		out[k] = c
	}
	*l = out
	return nil
}

// ParseDecisionContent decodes one JSON value into its DecisionContent variant.
//
// The variant is picked from the JSON token, not from the Go value a decoder would produce for an any, so
// what it accepts is unambiguous.
func ParseDecisionContent(b []byte) (DecisionContent, error) {
	b = bytes.TrimSpace(b)
	if len(b) == 0 {
		return nil, errors.New("is empty")
	}
	switch b[0] {
	case '"':
		t := Text("")
		if err := internal.UnmarshalJSON(b, &t); err != nil {
			return nil, err
		}
		return t, nil
	case '{':
		o := Object{}
		if err := internal.UnmarshalJSON(b, &o); err != nil {
			return nil, err
		}
		return o, nil
	case '[':
		a := Array{}
		if err := internal.UnmarshalJSON(b, &a); err != nil {
			return nil, err
		}
		return a, nil
	default:
		return nil, fmt.Errorf("expected a string, a JSON object or a JSON array, got %s", b)
	}
}

// SystemOneRequest is a System One decision request.
// Providers use their client's configured model and own transport envelopes and stricter protocol limits.
// Docs are inline attachments. Providers select supported document types, enforce read limits,
// and encode document bytes in their native wire format.
type SystemOneRequest struct {
	// State is required unless Docs contains one application/json document.
	State     DecisionContent `json:"state"`
	Questions Questions       `json:"questions"`
	// Docs contains inline document attachments. When State is unset, one application/json document
	// supplies the state instead. Supported attachment types depend on the provider and model.
	// Remote URLs are not supported.
	Docs []Doc `json:"docs,omitzero"`
}

// Validate checks the shared state, questions and inline documents without reading attachment bytes.
// Providers validate document types.
func (r *SystemOneRequest) Validate() error {
	if r.State == nil {
		if len(r.Docs) != 1 || internal.MimeByExt(filepath.Ext(r.Docs[0].GetFilename())) != "application/json" {
			return errors.New("state or one application/json document is required")
		}
	} else if err := r.State.Validate(); err != nil {
		return fmt.Errorf("state: %w", err)
	}
	if err := r.Questions.Validate(); err != nil {
		return err
	}
	for i := range r.Docs {
		d := &r.Docs[i]
		if err := d.Validate(); err != nil {
			return fmt.Errorf("document #%d: %w", i, err)
		}
		if d.URL != "" || d.Src == nil {
			return fmt.Errorf("document #%d: must be an inline document", i)
		}
	}
	return nil
}

// ReadState validates the request and returns the resolved state and remaining attachments.
// It reads a JSON state document with maxDocBytes as the byte limit, but does not read attachments.
// It leaves request fields unchanged. Returned attachments share the request's Docs slice.
func (r *SystemOneRequest) ReadState(maxDocBytes int64) (DecisionContent, []Doc, error) {
	if err := r.Validate(); err != nil {
		return nil, nil, err
	}
	if r.State != nil {
		return r.State, r.Docs, nil
	}
	d := r.Docs[0]
	_, b, err := d.Read(maxDocBytes)
	if err != nil {
		return nil, nil, fmt.Errorf("state document: %w", err)
	}
	state := documentState(b)
	if err := state.Validate(); err != nil {
		return nil, nil, fmt.Errorf("state document: %w", err)
	}
	return state, nil, nil
}

// documentState retains JSON document content without rebuilding its values.
type documentState json.RawMessage

func (documentState) content() {}

// Validate checks the state document's JSON shape.
func (s documentState) Validate() error {
	b := bytes.TrimSpace(s)
	if len(b) == 0 || (b[0] != '{' && b[0] != '[' && b[0] != '"') || !json.Valid(b) {
		return errors.New("must be a JSON string, object or array")
	}
	return nil
}

// MarshalJSON preserves the document's JSON value.
func (s documentState) MarshalJSON() ([]byte, error) {
	return s, nil
}

// SystemOneResponse is the response of a POST /v1/systemone request.
type SystemOneResponse struct {
	// Answers holds one answer per question, keyed by the question name.
	Answers Answers `json:"answers"`
	// Usage is the token usage of the request.
	Usage DecisionUsage `json:"usage"`
}

// ValidateQuestions checks that the response answers each requested question with the right type.
func (r *SystemOneResponse) ValidateQuestions(qs *Questions) error {
	if len(r.Answers) == 0 {
		return errors.New("no answer returned")
	}
	var errs []error
	for _, id := range slices.Sorted(maps.Keys(r.Answers)) {
		if r.Answers[id] == nil {
			errs = append(errs, fmt.Errorf("answer for question %q: must not be nil", id))
		}
	}
	for _, id := range slices.Sorted(maps.Keys(*qs)) {
		a, ok := r.Answers[id]
		switch {
		case !ok:
			errs = append(errs, fmt.Errorf("no answer returned for question %q", id))
		case a == nil:
			continue
		case (*qs)[id] == nil:
			errs = append(errs, fmt.Errorf("question %q: must not be nil", id))
		case a.Type != (*qs)[id].Type:
			errs = append(errs, fmt.Errorf("mismatched answer type for question %q", id))
		}
	}
	if err := r.Usage.Validate(); err != nil {
		errs = append(errs, err)
	}
	return errors.Join(errs...)
}

// DecisionUsage reports the token usage of a request.
type DecisionUsage struct {
	// InputTokens counts the prompt tokens evaluated for all questions.
	InputTokens int64 `json:"input_tokens"`
	// OutputTokens counts output tokens reported by the provider, often zero for decision models.
	OutputTokens int64 `json:"output_tokens"`
	// ReasoningTokens counts reasoning tokens within OutputTokens, when reported by the provider.
	ReasoningTokens int64 `json:"reasoning_tokens,omitzero"`
}

// Validate rejects negative token counts.
func (u *DecisionUsage) Validate() error {
	if u.InputTokens < 0 || u.OutputTokens < 0 || u.ReasoningTokens < 0 {
		return errors.New("decision token usage must not be negative")
	}
	return nil
}

// questionJSON is the wire representation of a Question.
//
// Criteria is the encoded form of whichever of the Noul, Choice and Score fields that Question.Type
// selects. The three do not share a Go type, so Question.MarshalJSON encodes the one that applies. It is
// left nil, and so omitted, for a noul question without criteria.
type questionJSON struct {
	Type         QuestionType    `json:"type"`
	Instructions DecisionContent `json:"instructions,omitzero"`
	Criteria     json.RawMessage `json:"criteria,omitzero"`
}

// answerTypeJSON is the wire representation of an answer read for its type only.
type answerTypeJSON struct {
	Type QuestionType `json:"type"`
}

// noulAnswerJSON is the wire representation of a Noul answer.
type noulAnswerJSON struct {
	Type QuestionType `json:"type"`
	Noul float64      `json:"noul"`
}

// choiceAnswerJSON is the wire representation of a Choice answer.
type choiceAnswerJSON struct {
	Type          QuestionType       `json:"type"`
	Choice        string             `json:"choice"`
	Confidence    float64            `json:"confidence"`
	Probabilities map[string]float64 `json:"probabilities"`
}

// scoreAnswerJSON is the wire representation of a Score answer.
type scoreAnswerJSON struct {
	Type          QuestionType               `json:"type"`
	Score         float64                    `json:"score"`
	Confidence    float64                    `json:"confidence"`
	Legend        map[string]DecisionContent `json:"legend"`
	Probabilities map[string]float64         `json:"probabilities"`
}

func validateContent(c DecisionContent) error {
	if c == nil {
		return errors.New("must not be nil")
	}
	return c.Validate()
}

var (
	_ internal.Validatable = (*NoulCriteria)(nil)
)
