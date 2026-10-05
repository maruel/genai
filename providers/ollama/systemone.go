// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Native Ollama System One request conversion and validation.

package ollama

import (
	"encoding/base64"
	"errors"
	"fmt"
	"io"
	"math"
	"strings"

	"github.com/maruel/genai"
	"github.com/maruel/genai/internal"
)

// SystemOneRequest is Ollama's /v1/systemone request.
// Images are base64 encoded, without a data URL prefix. Clef supports structured
// criteria descriptions; other decision models may require strings or null.
type SystemOneRequest struct {
	Model     string                `json:"model"`
	State     genai.DecisionContent `json:"state"`
	Questions genai.Questions       `json:"questions"`
	Images    []string              `json:"images,omitzero"`
	KeepAlive string                `json:"keep_alive,omitzero"`
}

// From converts shared decision input, preserving Model and KeepAlive.
// Each inline document is read with a 10 MiB bound.
func (r *SystemOneRequest) From(in *genai.SystemOneRequest) error {
	state, docs, err := in.ReadState(10 * 1024 * 1024)
	if err != nil {
		return err
	}
	var images []string
	for i := range docs {
		d := docs[i]
		if err := d.Validate(); err != nil {
			return fmt.Errorf("document #%d: %w", i, err)
		}
		if d.URL != "" || d.Src == nil {
			return fmt.Errorf("document #%d: must be an inline image", i)
		}
		mt, data, err := d.Read(10 * 1024 * 1024)
		if err != nil {
			return fmt.Errorf("document #%d: %w", i, err)
		}
		if !strings.HasPrefix(mt, "image/") {
			return fmt.Errorf("document #%d: unsupported document type %q", i, mt)
		}
		images = append(images, base64.StdEncoding.EncodeToString(data))
	}
	r.State, r.Questions, r.Images = state, in.Questions, images
	return nil
}

// Validate checks the shared schema and Ollama's 64-question and 26-option limits.
func (r *SystemOneRequest) Validate() error {
	if strings.TrimSpace(r.Model) == "" {
		return errors.New("model is required")
	}
	in := genai.SystemOneRequest{State: r.State, Questions: r.Questions}
	if err := in.Validate(); err != nil {
		return err
	}
	if len(r.Questions) > 64 {
		return errors.New("at most 64 questions are supported")
	}
	for id, q := range r.Questions {
		if len(q.Choice) > 26 || len(q.Score) > 26 {
			return fmt.Errorf("question %q: at most 26 options are supported", id)
		}
	}
	for i, img := range r.Images {
		if img == "" {
			return fmt.Errorf("image #%d: must not be empty", i)
		}
		if _, err := io.Copy(io.Discard, base64.NewDecoder(base64.StdEncoding, strings.NewReader(img))); err != nil {
			return fmt.Errorf("image #%d: invalid base64: %w", i, err)
		}
	}
	return nil
}

// SystemOneResponse is Ollama's native decision response.
type SystemOneResponse struct {
	Model string `json:"model"`
	genai.SystemOneResponse
}

// UnmarshalJSON checks the required native response fields.
func (r *SystemOneResponse) UnmarshalJSON(b []byte) error {
	var v struct {
		Model   string               `json:"model"`
		Answers *genai.Answers       `json:"answers"`
		Usage   *genai.DecisionUsage `json:"usage"`
	}
	if err := internal.UnmarshalJSON(b, &v); err != nil {
		return err
	}
	if v.Model == "" || v.Answers == nil || v.Usage == nil {
		return errors.New("missing decision model, answers or usage")
	}
	r.Model = v.Model
	r.SystemOneResponse = genai.SystemOneResponse{Answers: *v.Answers, Usage: *v.Usage}
	return nil
}

// To converts the native response to shared decision output.
func (r *SystemOneResponse) To(out *genai.SystemOneResponse) error {
	for id, a := range r.Answers {
		if a == nil {
			return fmt.Errorf("answer %q is nil", id)
		}
		if math.IsNaN(a.Noul) || a.Noul < 0 || a.Noul > 1 || math.IsNaN(a.Confidence) || a.Confidence < 0 || a.Confidence > 1 || math.IsNaN(a.Score) || math.IsInf(a.Score, 0) {
			return fmt.Errorf("answer %q: invalid probability, confidence or score", id)
		}
		for k, p := range a.Probabilities {
			if math.IsNaN(p) || p < 0 || p > 1 {
				return fmt.Errorf("answer %q: invalid probability for %q", id, k)
			}
		}
	}
	*out = r.SystemOneResponse
	return nil
}
