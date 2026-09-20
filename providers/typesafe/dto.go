// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Wire types for the TypeSafe System One API.
//
// Documentation: https://docs.typesafe.ai/api

package typesafe

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"maps"
	"slices"
	"strings"
	"time"

	"github.com/maruel/genai"
	"github.com/maruel/genai/base"
	"github.com/maruel/genai/internal"
)

// Content is text, a JSON object, or a JSON array.
//
// It is implemented by Text, Object and Array. It marshals to the value itself, not to an object with
// fields. Use nil when the field is optional and is left unset.
//
// It is used for state, instructions and criteria descriptions. It is meant to be sent; the API only
// returns it in Answer.Legend, decoded back by ScoreLegend, since an interface cannot be decoded on its
// own.
type Content interface {
	// Validate ensures the content is valid.
	Validate() error
	// content restricts Content to the types of this package.
	content()
}

// Text is Content that is a JSON string.
type Text string

// content implements Content.
func (Text) content() {}

// Validate implements internal.Validatable.
//
// Any text is valid, including the empty string, the API is the one that decides whether a specific field
// may be empty.
func (Text) Validate() error {
	return nil
}

// Object is Content that is a JSON object.
//
// The values are any, like genai.Reply.Opaque: they must be JSON encodable. Go has no recursive JSON
// value type that does not require wrapping every scalar, so a nested map[string]any, []any or a struct
// with JSON tags is passed as is. It is the equivalent of the SDKs' Mapping[str, JSONValue | None].
type Object map[string]any

// content implements Content.
func (Object) content() {}

// Validate implements internal.Validatable.
//
// Every value is validated so that the error names the offending key.
func (o Object) Validate() error {
	if o == nil {
		return errors.New("Object is nil, use nil to leave the field unset")
	}
	// TODO: Validate the values recursively to report the full path, e.g. `key "ticket": index 3: ...`.
	// This needs a depth limit because a map can contain itself, and anything that is not map[string]any,
	// []any, Object or Array must stay delegated to encoding/json.
	var errs []error
	for _, k := range slices.Sorted(maps.Keys(o)) {
		// The values are any, so encoding/json is the authority on what they may be.
		if _, err := json.Marshal(o[k]); err != nil {
			errs = append(errs, fmt.Errorf("key %q: %w", k, err))
		}
	}
	return errors.Join(errs...)
}

// Array is Content that is a JSON array.
//
// The values are any, like Object: they must be JSON encodable. It is the equivalent of the SDKs'
// Sequence[JSONValue | None], so it is not restricted to Content items.
type Array []any

// content implements Content.
func (Array) content() {}

// Validate implements internal.Validatable.
//
// Every item is validated so that the error names the offending index.
func (a Array) Validate() error {
	if a == nil {
		return errors.New("Array is nil, use nil to leave the field unset")
	}
	// TODO: Validate the items recursively, see Object.Validate for the constraints.
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
	// Instructions is what the model should decide or rate. It can be set for the three types, and is
	// required for QuestionNoul unless Noul is set.
	Instructions Content
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
	Choice map[string]Content
	// Score lists the levels of a score question rubric, in order, starting at level 0. It can only be set
	// for QuestionScore, where at least one level is required, although the API documents that a useful
	// rubric has at least two. Every level must be described.
	//
	// Recommended reading: https://docs.typesafe.ai/primitives/score
	Score []Content
}

// Validate implements internal.Validatable.
func (q Question) Validate() error {
	var errs []error
	if q.Instructions != nil {
		if err := validateContent(q.Instructions); err != nil {
			errs = append(errs, fmt.Errorf("field Instructions: %w", err))
		}
	}
	switch q.Type {
	case QuestionNoul:
		if q.Choice != nil || q.Score != nil {
			errs = append(errs, errors.New("fields Choice and Score: can't be set on a noul question"))
		}
		if q.Instructions == nil && q.Noul == nil {
			errs = append(errs, errors.New("field Instructions or Noul: one is required"))
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
			if err := validateContent(c); err != nil {
				errs = append(errs, fmt.Errorf("field Score[%d]: %w", i, err))
			}
		}
	default:
		errs = append(errs, fmt.Errorf("field Type: must be %q, %q or %q, got %q", QuestionNoul, QuestionChoice, QuestionScore, q.Type))
	}
	return errors.Join(errs...)
}

// MarshalJSON implements json.Marshaler.
func (q Question) MarshalJSON() ([]byte, error) {
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
// It is passed as genai.GenOptionText.DecodeAs when the questions are not declared by a struct of Noul,
// Choice and Score fields. The answers then decode into Answers.
type Questions map[string]Question

// Validate ensures the questions are valid.
func (q Questions) Validate() error {
	if len(q) == 0 {
		return errors.New("at least one question is required")
	}
	var errs []error
	for _, name := range slices.Sorted(maps.Keys(q)) {
		if err := q[name].Validate(); err != nil {
			errs = append(errs, fmt.Errorf("question %q: %w", name, err))
		}
	}
	return errors.Join(errs...)
}

// NoulCriteria describes what the yes and the no answers of a noul question mean.
type NoulCriteria struct {
	// True describes what a yes (value near 1) means.
	True Content `json:"true,omitzero"`
	// False describes what a no (value near 0) means.
	False Content `json:"false,omitzero"`
}

// Validate ensures the criteria are valid.
func (c *NoulCriteria) Validate() error {
	if c == nil {
		return errors.New("field Criteria: is nil")
	}
	if c.True == nil && c.False == nil {
		return errors.New("field Criteria: at least one of True or False is required")
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

// questionJSON is the wire representation of a Question.
//
// Criteria is the encoded form of whichever of the Noul, Choice and Score fields that Question.Type
// selects. The three do not share a Go type, so Question.MarshalJSON encodes the one that applies. It is
// left nil, and so omitted, for a noul question without criteria.
type questionJSON struct {
	Type         QuestionType    `json:"type"`
	Instructions Content         `json:"instructions,omitzero"`
	Criteria     json.RawMessage `json:"criteria,omitzero"`
}

// Answers holds the answer to each question, keyed by the question name.
type Answers map[string]Answer

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

// answerTypeJSON is the wire representation of an answer read for its type only.
type answerTypeJSON struct {
	Type QuestionType `json:"type"`
}

// MarshalJSON implements json.Marshaler.
//
// The fields of the union that do not match Type are omitted, even when they are zero, so that a score
// of 0 or a probability of 0 is preserved.
//
//nolint:gocritic // hugeParam: a value receiver is required to marshal the values of an Answers map.
func (a Answer) MarshalJSON() ([]byte, error) {
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
	Type          QuestionType       `json:"type"`
	Score         float64            `json:"score"`
	Confidence    float64            `json:"confidence"`
	Legend        map[string]Content `json:"legend"`
	Probabilities map[string]float64 `json:"probabilities"`
}

// ScoreLegend is the description of each level of a score answer, keyed by the level.
//
// It is the Question.Score criteria of the question, echoed back by the API. The values are decoded as
// Text, Object or Array; a nil value is a level the caller left undescribed.
type ScoreLegend map[string]Content

// UnmarshalJSON implements json.Unmarshaler.
//
// Content is an interface, so encoding/json cannot decode it on its own; each level is decoded into the
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
		c, err := contentFromJSON(v)
		if err != nil {
			return fmt.Errorf("level %q: %w", k, err)
		}
		out[k] = c
	}
	*l = out
	return nil
}

// contentFromJSON decodes one JSON value into its Content variant.
//
// The variant is picked from the JSON token, not from the Go value a decoder would produce for an any, so
// what it accepts is unambiguous.
func contentFromJSON(b []byte) (Content, error) {
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

// SystemOneRequest is the body of a POST /v1/systemone request.
type SystemOneRequest struct {
	// State is the content to evaluate. Use a string for text, or a JSON object or array for structured
	// state like a chat log, records, or the current state of an application.
	State Content `json:"state"`
	// Model is the model to use, e.g. "jev-latest".
	Model string `json:"model"`
	// Questions is the set of questions to ask about the state.
	Questions Questions `json:"questions"`
}

// From sets the state from the message.
//
// The state is the request of the message, or the array of them when the message has several, which the
// API takes as a sequence of messages or records. TypeSafe does not hold a conversation, so replies and
// tool call results are rejected.
//
// The API takes a plain string for a text state, and a JSON object or array for structured state, an
// object being the recommended form. It does not parse JSON given as a string, so structured data must be
// passed as a document.
func (r *SystemOneRequest) From(msg *genai.Message) error {
	if len(msg.Replies) != 0 || len(msg.ToolCallResults) != 0 {
		return errors.New("TypeSafe has no conversation support; pass the full state as text or as a JSON document instead of assistant replies or tool call results")
	}
	if len(msg.Requests) == 0 {
		return errors.New("the message must have the state as text or as a JSON document")
	}
	if len(msg.Requests) == 1 {
		state, err := stateFromRequest(&msg.Requests[0])
		if err != nil {
			return err
		}
		r.State = state
		return nil
	}
	arr := make(Array, 0, len(msg.Requests))
	for i := range msg.Requests {
		state, err := stateFromRequest(&msg.Requests[i])
		if err != nil {
			return fmt.Errorf("request #%d: %w", i, err)
		}
		arr = append(arr, state)
	}
	r.State = arr
	return nil
}

// stateFromRequest returns the state held by a request.
func stateFromRequest(req *genai.Request) (Content, error) {
	if req.Doc.IsZero() {
		if req.Text == "" {
			return nil, errors.New("must have the state as text or as a JSON document")
		}
		return Text(req.Text), nil
	}
	return stateFromDoc(&req.Doc)
}

// stateFromDoc returns the state held by a document.
//
// The state is the API's JSON content: a string, a JSON object or a JSON array.
func stateFromDoc(d *genai.Doc) (Content, error) {
	if d.URL != "" {
		return nil, errors.New("the state document must be inline")
	}
	mimeType, data, err := d.Read(10 * 1024 * 1024)
	if err != nil {
		return nil, err
	}
	if mimeType != "application/json" {
		return nil, fmt.Errorf("the state document must be application/json, got %q; name it with a .json extension", mimeType)
	}
	state, err := contentFromJSON(data)
	if err != nil {
		return nil, fmt.Errorf("the state document must contain a JSON object or array: %w", err)
	}
	return state, nil
}

var (
	_ genai.Provider       = &Client{}
	_ internal.Validatable = Question{}
	_ internal.Validatable = (*NoulCriteria)(nil)
	_ internal.Validatable = Questions(nil)
	_ internal.Validatable = Text("")
	_ internal.Validatable = Object(nil)
	_ internal.Validatable = Array(nil)
)

// FromOptions sets the questions to ask from the options.
//
// They come from genai.GenOptionText.DecodeAs, a pointer to a struct of Noul, Choice and Score fields
// or a Questions.
func (r *SystemOneRequest) FromOptions(opts ...genai.GenOption) error {
	for _, opt := range opts {
		switch v := opt.(type) {
		case *genai.GenOptionText:
			if err := v.Validate(); err != nil {
				return err
			}
			if unsupported := unsupportedTextOptions(v); len(unsupported) != 0 {
				return &base.ErrNotSupported{Options: unsupported}
			}
			switch d := v.DecodeAs.(type) {
			case nil:
				return errors.New("field DecodeAs: a pointer to a struct of Noul, Choice and Score fields, or a Questions, is required to declare the questions")
			case Questions:
				if err := d.Validate(); err != nil {
					return fmt.Errorf("field DecodeAs: %w", err)
				}
				r.Questions = d
			default:
				q, err := QuestionsFrom(v.DecodeAs)
				if err != nil {
					return fmt.Errorf("field DecodeAs: %w", err)
				}
				r.Questions = q
			}
		default:
			return &base.ErrNotSupported{Options: []string{fmt.Sprintf("%T", opt)}}
		}
	}
	if r.Questions == nil {
		return errors.New("the questions to ask are required, pass *genai.GenOptionText with DecodeAs")
	}
	return nil
}

// SystemOneResponse is the response of a POST /v1/systemone request.
type SystemOneResponse struct {
	// Model is the versioned ID of the model that answered, e.g. "jev-1.13.0". It can differ from the
	// model requested when an alias like "jev-latest" was used.
	Model string `json:"model"`
	// Answers holds one answer per question, keyed by the question name.
	Answers Answers `json:"answers"`
	// Usage is the token usage of the request.
	Usage Usage `json:"usage"`
}

// ToResult converts the response to a genai.Result.
//
// The reply is the JSON object of the answers keyed by question name, which Result.Decode decodes into
// the questionnaire struct or into Answers.
func (r *SystemOneResponse) ToResult() (genai.Result, error) {
	out := genai.Result{
		Usage: genai.Usage{
			InputTokens:  r.Usage.InputTokens,
			OutputTokens: r.Usage.OutputTokens,
			TotalTokens:  r.Usage.InputTokens + r.Usage.OutputTokens,
			FinishReason: genai.FinishedStop,
		},
	}
	if len(r.Answers) == 0 {
		return out, errors.New("no answer returned")
	}
	raw, err := marshalAnswers(r.Answers)
	if err != nil {
		return out, err
	}
	out.Replies = []genai.Reply{{Text: string(raw)}}
	return out, nil
}

// Usage reports the token usage of a request.
type Usage struct {
	// InputTokens is the number of input tokens evaluated. TypeSafe charges per input token.
	InputTokens int64 `json:"input_tokens"`
	// OutputTokens is the number of output tokens generated. It is not charged.
	OutputTokens int64 `json:"output_tokens"`
}

// marshalAnswers marshals the answers for genai.Result.
//
// It does not escape HTML so that descriptions are returned verbatim, like the API does.
func marshalAnswers(a Answers) ([]byte, error) {
	buf := &bytes.Buffer{}
	enc := json.NewEncoder(buf)
	enc.SetEscapeHTML(false)
	if err := enc.Encode(a); err != nil {
		return nil, err
	}
	// Encoder.Encode appends a newline.
	return bytes.TrimSuffix(buf.Bytes(), []byte("\n")), nil
}

// unsupportedTextOptions lists the genai.GenOptionText fields that are set but can't be honored, the
// questions being the only thing TypeSafe takes from it.
func unsupportedTextOptions(o *genai.GenOptionText) []string {
	var out []string
	if o.Temperature != 0 {
		out = append(out, "GenOptionText.Temperature")
	}
	if o.TopP != 0 {
		out = append(out, "GenOptionText.TopP")
	}
	if o.MaxTokens != 0 {
		out = append(out, "GenOptionText.MaxTokens")
	}
	if o.TopLogprobs != 0 {
		out = append(out, "GenOptionText.TopLogprobs")
	}
	if o.TopK != 0 {
		out = append(out, "GenOptionText.TopK")
	}
	if o.SystemPrompt != "" {
		out = append(out, "GenOptionText.SystemPrompt")
	}
	if len(o.Stop) != 0 {
		out = append(out, "GenOptionText.Stop")
	}
	if o.ReplyAsJSON {
		out = append(out, "GenOptionText.ReplyAsJSON")
	}
	return out
}

// ListModelsResponse is the response of a GET /v1/models request.
type ListModelsResponse struct {
	Models []Model `json:"models"`
}

// Model is a model or an alias available to the account.
type Model struct {
	// Name is the model ID or alias, as accepted by SystemOneRequest.Model.
	Name string `json:"name"`
	// Description documents what the model is for.
	Description string `json:"description"`
	// ReleaseDate is when the model or alias was released.
	ReleaseDate string `json:"release_date"`
}

// GetID implements genai.Model.
func (m *Model) GetID() string {
	return m.Name
}

// String implements genai.Model.
func (m *Model) String() string {
	if t, err := time.Parse(time.RFC3339Nano, m.ReleaseDate); err == nil {
		return fmt.Sprintf("%s (%s)", m.Name, t.Format("2006-01-02"))
	}
	return m.Name
}

// Context implements genai.Model.
func (m *Model) Context() int64 {
	return 0
}

// ErrorResponse is the error returned by the API on a failed request.
type ErrorResponse struct {
	// Detail describes the failure.
	Detail ErrorDetail `json:"detail"`
}

// Error implements error.
func (er *ErrorResponse) Error() string {
	return er.Detail.Error()
}

// IsAPIError implements base.ErrAPI.
func (er *ErrorResponse) IsAPIError() bool {
	return true
}

// ErrorDetail is the "detail" field of an ErrorResponse.
//
// The API returns a string or an object describing the failure, or a list of validation errors on
// HTTP 422.
type ErrorDetail struct {
	// Message describes the failure. It is set when the API returns a string or an object.
	Message string
	// ValidationErrors lists the fields that failed validation. It is set on HTTP 422.
	ValidationErrors []ValidationError
}

// UnmarshalJSON implements json.Unmarshaler.
func (d *ErrorDetail) UnmarshalJSON(b []byte) error {
	if len(b) != 0 {
		switch b[0] {
		case '"':
			return json.Unmarshal(b, &d.Message)
		case '[':
			return json.Unmarshal(b, &d.ValidationErrors)
		}
	}
	o := errorDetailObjectJSON{}
	if err := json.Unmarshal(b, &o); err != nil {
		return err
	}
	if o.ErrorType != "" {
		d.Message = o.ErrorType + ": " + o.Message
	} else {
		d.Message = o.Message
	}
	return nil
}

// errorDetailObjectJSON is the wire representation of the object form of an ErrorDetail.
type errorDetailObjectJSON struct {
	ErrorType string `json:"error_type"`
	Message   string `json:"message"`
}

// Error implements error.
func (d *ErrorDetail) Error() string {
	if d.Message != "" {
		return d.Message
	}
	if len(d.ValidationErrors) == 0 {
		return "unknown error"
	}
	errs := make([]string, len(d.ValidationErrors))
	for i := range d.ValidationErrors {
		errs[i] = d.ValidationErrors[i].Error()
	}
	return strings.Join(errs, "; ")
}

// ValidationError is one field validation failure, returned on HTTP 422.
type ValidationError struct {
	// Type is the kind of failure, e.g. "too_short".
	Type string `json:"type"`
	// Location is the path to the offending field in the request.
	Location []any `json:"loc"`
	// Message describes the failure.
	Message string `json:"msg"`
	// Input is the offending value.
	Input any `json:"input"`
	// Ctx holds extra details about the failure, e.g. the expected minimum length.
	Ctx map[string]any `json:"ctx"`
}

// Error implements error.
func (v *ValidationError) Error() string {
	loc := make([]string, len(v.Location))
	for i, l := range v.Location {
		loc[i] = fmt.Sprint(l)
	}
	where := strings.Join(loc, ".")
	if where == "" {
		where = "body"
	}
	return where + ": " + v.Message
}

// validateContent ensures c is valid.
//
// A nil Content is not valid, so the caller must skip it for a field that is optional.
func validateContent(c Content) error {
	if c == nil {
		return errors.New("must not be nil")
	}
	return c.Validate()
}
