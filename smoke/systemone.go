// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Smoke qualification for typed System One decision inference.

package smoke

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"net/http"

	"github.com/maruel/httpjson"

	"github.com/maruel/genai"
	"github.com/maruel/genai/base"
	"github.com/maruel/genai/scoreboard"
)

// RunSystemOne qualifies typed decisions and text, JSON and image states.
// ProviderFactory must never fail. Unsupported inputs are omitted from the result;
// transport failures and malformed responses abort qualification.
func RunSystemOne(ctx context.Context, pf ProviderFactory) (scoreboard.Scenario, genai.Usage, error) {
	c := pf("")
	sc := scoreboard.Scenario{Models: []string{c.ModelID()}}
	u := genai.Usage{}
	if c.ModelID() == "" {
		return sc, u, errors.New("provider must have a model")
	}
	if !c.Capabilities().SystemOne {
		return sc, u, errors.New("provider does not support SystemOne")
	}
	f := scoreboard.DecisionFunctionality{}
	q := genai.Questions{
		"billing": {Type: genai.QuestionNoul, Instructions: genai.Text("Is the customer asking about billing?")},
		"route":   {Type: genai.QuestionChoice, Instructions: genai.Text("Which team should handle the request?"), Choice: map[string]genai.DecisionContent{"billing": genai.Text("Charges and refunds"), "support": genai.Text("Technical problems")}},
		"urgency": {Type: genai.QuestionScore, Instructions: genai.Text("How urgent is this request?"), Score: []genai.DecisionContent{genai.Text("Can wait"), genai.Text("Needs attention today")}},
	}
	call := func(name string, req *genai.SystemOneRequest) (bool, error) {
		res, err := pf("SystemOne-"+name).SystemOne(ctx, req)
		if err != nil {
			if _, ok := errors.AsType[*base.ErrNotSupported](err); ok {
				return false, nil
			}
			if e, ok := errors.AsType[*httpjson.Error](err); ok {
				if e.StatusCode == http.StatusBadRequest || e.StatusCode == http.StatusUnprocessableEntity || e.StatusCode == http.StatusNotImplemented {
					return false, nil
				}
				return false, err
			}
			return false, err
		}
		if res == nil {
			return false, errors.New("SystemOne returned a nil response")
		}
		if err := res.ValidateQuestions(&req.Questions); err != nil {
			return false, err
		}
		u.Add(&genai.Usage{InputTokens: res.Usage.InputTokens, OutputTokens: res.Usage.OutputTokens, ReasoningTokens: res.Usage.ReasoningTokens, TotalTokens: res.Usage.InputTokens + res.Usage.OutputTokens})
		reporting := scoreboard.False
		if res.Usage.InputTokens > 0 {
			reporting = scoreboard.True
		}
		if sc.SystemOne == nil {
			f.ReportTokenUsage = reporting
		} else if f.ReportTokenUsage != reporting {
			f.ReportTokenUsage = scoreboard.Flaky
		}
		sc.SystemOne = &f
		return true, nil
	}
	state := genai.Text("I was charged twice for my order. Please refund the duplicate charge today.")
	for _, tc := range []struct {
		name      string
		supported *bool
	}{{"billing", &f.Noul}, {"route", &f.Choice}, {"urgency", &f.Score}} {
		ok, err := call(tc.name, &genai.SystemOneRequest{State: state, Questions: genai.Questions{tc.name: q[tc.name]}})
		if err != nil {
			return sc, u, fmt.Errorf("SystemOne %s: %w", tc.name, err)
		}
		*tc.supported = ok
	}
	if sc.SystemOne == nil {
		return sc, u, errors.New("model does not support any SystemOne question type")
	}
	sc.In = map[genai.Modality]scoreboard.ModalCapability{genai.ModalityText: {Inline: true}}
	sc.Out = map[genai.Modality]scoreboard.ModalCapability{genai.ModalityDecision: {Inline: true}}
	supported := genai.Questions{}
	if f.Noul {
		supported["billing"] = q["billing"]
	}
	if f.Choice {
		supported["route"] = q["route"]
	}
	if f.Score {
		supported["urgency"] = q["urgency"]
	}
	for _, tc := range []struct {
		name  string
		state genai.DecisionContent
	}{{"object", genai.Object{"ticket": state}}, {"array", genai.Array{state}}} {
		ok, err := call(tc.name, &genai.SystemOneRequest{State: tc.state, Questions: supported})
		if err != nil {
			return sc, u, fmt.Errorf("SystemOne %s: %w", tc.name, err)
		}
		if tc.name == "object" {
			f.Object = ok
		} else {
			f.Array = ok
		}
	}
	for _, ext := range []string{"jpg", "png", "webp"} {
		data, err := scoreboard.TestdataFiles.ReadFile("testdata/image." + ext)
		if err != nil {
			return sc, u, err
		}
		ok, err := call(ext, &genai.SystemOneRequest{State: genai.Text("A customer supplied this image with their refund request."), Questions: supported, Docs: []genai.Doc{{Filename: "image." + ext, Src: bytes.NewReader(data)}}})
		if err != nil {
			return sc, u, fmt.Errorf("SystemOne %s: %w", ext, err)
		}
		if ok {
			m := sc.In[genai.ModalityImage]
			m.Inline = true
			m.SupportedFormats = append(m.SupportedFormats, "image/"+map[string]string{"jpg": "jpeg", "png": "png", "webp": "webp"}[ext])
			sc.In[genai.ModalityImage] = m
		}
	}
	return sc, u, nil
}
