// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Wire types for the TypeSafe System One API.
//
// Documentation: https://docs.typesafe.ai/api

package typesafe

import (
	"encoding/json"
	"errors"
	"fmt"
	"maps"
	"slices"
	"strings"
	"time"

	"github.com/maruel/genai"
)

// SystemOneResponse is the native /v1/systemone response.
type SystemOneResponse struct {
	// Model identifies the decision model that answered.
	Model string `json:"model"`
	genai.SystemOneResponse
}

// To converts native decision output to shared decision output.
func (r *SystemOneResponse) To(out *genai.SystemOneResponse) error {
	*out = r.SystemOneResponse
	return nil
}

// SystemOneRequest is a text-only /v1/systemone request.
// TypeSafe requires nonempty noul criteria and described score levels.
type SystemOneRequest struct {
	State genai.DecisionContent `json:"state"`
	// Model selects the native endpoint's decision model.
	Model     string          `json:"model,omitzero"`
	Questions genai.Questions `json:"questions"`
}

// From converts shared decision input, preserving Model.
// One JSON document can supply the state, bounded to 10 MiB. Other attachments are not supported.
func (r *SystemOneRequest) From(in *genai.SystemOneRequest) error {
	state, docs, err := in.ReadState(10 * 1024 * 1024)
	if err != nil {
		return err
	}
	if len(docs) != 0 {
		return errors.New("TypeSafe does not support document attachments")
	}
	r.State, r.Questions = state, in.Questions
	return nil
}

// Validate checks TypeSafe's text-only state and question criteria.
func (r *SystemOneRequest) Validate() error {
	var errs []error
	in := genai.SystemOneRequest{State: r.State, Questions: r.Questions}
	if err := in.Validate(); err != nil {
		errs = append(errs, err)
	}
	for _, id := range slices.Sorted(maps.Keys(r.Questions)) {
		q := r.Questions[id]
		if q == nil {
			continue
		}
		if q.Noul != nil && q.Noul.True == nil && q.Noul.False == nil {
			errs = append(errs, fmt.Errorf("question %q: field Criteria: at least one of True or False is required", id))
		}
		if q.Type == genai.QuestionScore {
			for i, c := range q.Score {
				if c == nil {
					errs = append(errs, fmt.Errorf("question %q: field Score[%d]: must not be nil", id, i))
				}
			}
		}
	}
	return errors.Join(errs...)
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
