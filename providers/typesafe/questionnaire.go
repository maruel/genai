// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed by the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Questionnaire types to declare typed questions as Go struct fields, via genai.GenOptionText.DecodeAs.

package typesafe

import (
	"encoding/json"
	"fmt"
	"reflect"
	"strings"

	"github.com/maruel/genai/internal"
)

// Noul is a yes/no question about a state, and holds its answer once asked.
//
// Declare the questions to ask as fields of a struct passed to genai.GenOptionText.DecodeAs; GenSync asks
// them and Decode fills the answers in. The field name, or its `json` tag, is the question name.
type Noul struct {
	// Instructions is the yes/no question to ask. At least one of Instructions or Criteria is required.
	Instructions Content
	// Criteria optionally describes what the yes and the no outcomes mean.
	Criteria *NoulCriteria

	// Probability is the answer, the probability that the answer is yes, from 0 to 1. It is set by
	// Decode.
	Probability float64
}

// UnmarshalJSON implements json.Unmarshaler.
func (n *Noul) UnmarshalJSON(b []byte) error {
	a := Answer{}
	if err := internal.UnmarshalJSON(b, &a); err != nil {
		return err
	}
	if a.Type != QuestionNoul {
		return fmt.Errorf("expected a noul answer, got %q", a.Type)
	}
	n.Probability = a.Noul
	return nil
}

// Choice is a question that picks one option among a set, and holds its answer once asked.
//
// Declare the questions to ask as fields of a struct passed to genai.GenOptionText.DecodeAs; GenSync asks
// them and Decode fills the answers in. The field name, or its `json` tag, is the question name.
type Choice struct {
	// Instructions describes what the model should decide.
	Instructions Content
	// Criteria maps the options to their descriptions. Use nil for an option that needs no extra detail.
	// At least one option is required.
	Criteria map[string]Content

	// Label is the answer, the highest probability option. The other fields are derived from the
	// probabilities the model reported. They are set by Decode.
	Label         string
	Confidence    float64
	Probabilities map[string]float64
}

// UnmarshalJSON implements json.Unmarshaler.
func (c *Choice) UnmarshalJSON(b []byte) error {
	a := Answer{}
	if err := internal.UnmarshalJSON(b, &a); err != nil {
		return err
	}
	if a.Type != QuestionChoice {
		return fmt.Errorf("expected a choice answer, got %q", a.Type)
	}
	c.Label = a.Choice
	c.Confidence = a.Confidence
	c.Probabilities = a.Probabilities
	return nil
}

// Score is a question that rates the state along an ordered rubric, and holds its answer once asked.
//
// Declare the questions to ask as fields of a struct passed to genai.GenOptionText.DecodeAs; GenSync asks
// them and Decode fills the answers in. The field name, or its `json` tag, is the question name.
type Score struct {
	// Instructions describes what the model should rate.
	Instructions Content
	// Criteria lists the levels in order, starting at level 0. At least one level is required, although
	// the API documents that a useful rubric has at least two.
	Criteria []Content

	// Value is the answer, the probability weighted value across the levels, which can land between
	// levels. The other fields are derived from the probabilities the model reported, and legend is the
	// rubric echoed back. They are set by Decode.
	Value         float64
	Confidence    float64
	Legend        ScoreLegend
	Probabilities map[string]float64
}

// UnmarshalJSON implements json.Unmarshaler.
func (s *Score) UnmarshalJSON(b []byte) error {
	a := Answer{}
	if err := internal.UnmarshalJSON(b, &a); err != nil {
		return err
	}
	if a.Type != QuestionScore {
		return fmt.Errorf("expected a score answer, got %q", a.Type)
	}
	s.Value = a.Score
	s.Confidence = a.Confidence
	s.Legend = a.Legend
	s.Probabilities = a.Probabilities
	return nil
}

// QuestionsFrom returns the questions declared by the fields of v.
//
// v must be a pointer to a struct whose fields are Noul, Choice or Score; the field name, or its `json`
// tag, is used as the question name, and a `json:"-"` field is skipped. It is what GenSync does with a
// struct passed as genai.GenOptionText.DecodeAs, exposed so the questions can be printed, reviewed, or
// tuned, and then passed as DecodeAs themselves.
func QuestionsFrom(v any) (Questions, error) {
	t := reflect.TypeOf(v)
	if t == nil || t.Kind() != reflect.Pointer || t.Elem().Kind() != reflect.Struct {
		return nil, fmt.Errorf("%T: must be a pointer to a struct of Noul, Choice or Score fields", v)
	}
	t = t.Elem()
	val := reflect.ValueOf(v).Elem()
	out := make(Questions, t.NumField())
	for i := range t.NumField() {
		f := t.Field(i)
		if name, skip, err := questionName(&f); err != nil {
			return nil, err
		} else if skip {
			continue
		} else {
			switch f.Type {
			case reflect.TypeFor[Noul]():
				n := val.Field(i).Addr().Interface().(*Noul)
				out[name] = Question{Type: QuestionNoul, Instructions: n.Instructions, Noul: n.Criteria}
			case reflect.TypeFor[Choice]():
				c := val.Field(i).Addr().Interface().(*Choice)
				out[name] = Question{Type: QuestionChoice, Instructions: c.Instructions, Choice: c.Criteria}
			case reflect.TypeFor[Score]():
				s := val.Field(i).Addr().Interface().(*Score)
				out[name] = Question{Type: QuestionScore, Instructions: s.Instructions, Score: s.Criteria}
			default:
				return nil, fmt.Errorf("field %s: must be a Noul, a Choice or a Score, got a %s", f.Name, f.Type)
			}
		}
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("%T: no Noul, Choice or Score field to ask", v)
	}
	if err := out.Validate(); err != nil {
		return nil, err
	}
	return out, nil
}

// questionName returns the name to use for a question field.
func questionName(f *reflect.StructField) (string, bool, error) {
	if !f.IsExported() {
		return "", false, fmt.Errorf("field %s: must be exported to receive its answer", f.Name)
	}
	if tag, ok := f.Tag.Lookup("json"); ok {
		name, _, _ := strings.Cut(tag, ",")
		if name == "-" {
			return "", true, nil
		}
		if name != "" {
			return name, false, nil
		}
	}
	return f.Name, false, nil
}

var (
	_ json.Unmarshaler     = (*Noul)(nil)
	_ json.Unmarshaler     = (*Choice)(nil)
	_ json.Unmarshaler     = (*Score)(nil)
	_ internal.Validatable = (*NoulCriteria)(nil)
)
