// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Hosted CLEF System One requests, inline images and response validation.

package cloudflare

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"image"
	"image/jpeg"
	"image/png"
	"io"
	"maps"
	"math"
	"path/filepath"
	"slices"
	"strconv"
	"strings"

	"golang.org/x/image/webp"

	"github.com/maruel/genai"
	"github.com/maruel/genai/internal"
)

// DecisionImage is an inline image in one of the hosted schema's two forms.
// Set either DataURL or both ContentType and Base64, never both forms.
type DecisionImage struct {
	DataURL     string `json:"-"`
	ContentType string `json:"content_type"`
	Base64      string `json:"base64"`
}

// MarshalJSON implements json.Marshaler.
func (i *DecisionImage) MarshalJSON() ([]byte, error) {
	if i.DataURL != "" {
		if i.ContentType != "" || i.Base64 != "" {
			return nil, errors.New("image must use only one inline representation")
		}
		return json.Marshal(i.DataURL)
	}
	type plain DecisionImage
	return json.Marshal((*plain)(i))
}

// Validate checks the hosted image format, decoded size and dimensions without decoding pixels.
func (i *DecisionImage) Validate() error {
	_, err := i.decodedSize()
	return err
}

func (i *DecisionImage) decodedSize() (int, error) {
	mt, b := i.ContentType, i.Base64
	if i.DataURL != "" {
		if mt != "" || b != "" {
			return 0, errors.New("image must use only one inline representation")
		}
		hdr, data, ok := strings.Cut(i.DataURL, ",")
		if !ok || len(hdr) < 5 || !strings.EqualFold(hdr[:5], "data:") || !strings.HasSuffix(hdr, ";base64") {
			return 0, errors.New("image must be an inline base64 data URL")
		}
		mt, b = strings.TrimSuffix(hdr[5:], ";base64"), data
	}
	var decode func(io.Reader) (image.Config, error)
	switch mt {
	case "image/png":
		decode = png.DecodeConfig
	case "image/jpeg":
		decode = jpeg.DecodeConfig
	case "image/webp":
		decode = webp.DecodeConfig
	default:
		return 0, fmt.Errorf("unsupported image content type %q", mt)
	}
	const maxSize = 4 * 1024 * 1024
	if b == "" || len(b) > base64.StdEncoding.EncodedLen(maxSize) {
		return 0, errors.New("image must contain at most 4 MiB decoded")
	}
	r := imageSizeReader{Reader: io.LimitReader(base64.NewDecoder(base64.StdEncoding, strings.NewReader(b)), maxSize+1)}
	cfg, err := decode(&r)
	if err != nil {
		return 0, fmt.Errorf("invalid image: %w", err)
	}
	if cfg.Width <= 0 || cfg.Height <= 0 || int64(cfg.Width)*int64(cfg.Height) > 16_000_000 {
		return 0, errors.New("image exceeds 16 megapixels")
	}
	if _, err := io.Copy(io.Discard, &r); err != nil {
		return 0, fmt.Errorf("invalid image base64: %w", err)
	}
	if r.size > maxSize {
		return 0, errors.New("image exceeds 4 MiB decoded")
	}
	return r.size, nil
}

// imageSizeReader counts decoded bytes, including any read-ahead by DecodeConfig.
type imageSizeReader struct {
	io.Reader
	size int
}

func (r *imageSizeReader) Read(b []byte) (int, error) {
	n, err := r.Reader.Read(b)
	r.size += n
	return n, err
}

// SystemOneRequest is a hosted decision request using the CLEF-compatible schema.
// Model is a full Workers AI ID or a short selector under @cf/cloudflare.
// SystemOneRaw sends the final component of a full ID as the body selector, or a short selector
// unchanged. Video is not supported.
// See https://developers.cloudflare.com/workers-ai/models/clef/schema-input.json.
type SystemOneRequest struct {
	Model     string                `json:"model"`
	State     genai.DecisionContent `json:"state"`
	Questions genai.Questions       `json:"questions"`
	Images    []DecisionImage       `json:"images,omitzero"`
}

// From converts shared decision input, preserving Model.
// One JSON document can supply the state, bounded to 13 MiB. Image reads are bounded to 4 MiB each.
// Remote documents and video are rejected.
func (r *SystemOneRequest) From(in *genai.SystemOneRequest) error {
	state, docs, err := in.ReadState(13 * 1024 * 1024)
	if err != nil {
		return err
	}
	images, err := imagesFromDocs(docs)
	if err != nil {
		return err
	}
	r.State = state
	r.Questions = in.Questions
	r.Images = images
	return nil
}

// imagesFromDocs encodes bounded inline documents in the hosted data URL format.
func imagesFromDocs(docs []genai.Doc) ([]DecisionImage, error) {
	if len(docs) == 0 {
		return nil, nil
	}
	if len(docs) > 4 {
		return nil, errors.New("at most 4 images are allowed")
	}
	images := make([]DecisionImage, len(docs))
	for j := range docs {
		d := docs[j]
		if err := d.Validate(); err != nil {
			return nil, fmt.Errorf("document #%d: %w", j, err)
		}
		if d.URL != "" || d.Src == nil {
			return nil, fmt.Errorf("document #%d: must be an inline document", j)
		}
		mt := internal.MimeByExt(filepath.Ext(d.GetFilename()))
		switch mt {
		case "image/jpeg", "image/png", "image/webp":
		default:
			return nil, fmt.Errorf("document #%d: unsupported document type %q", j, mt)
		}
	}
	for j := range docs {
		d := docs[j]
		mt, data, err := d.Read(4 * 1024 * 1024)
		if err != nil {
			return nil, fmt.Errorf("image #%d: %w", j, err)
		}
		images[j] = DecisionImage{DataURL: "data:" + mt + ";base64," + base64.StdEncoding.EncodeToString(data)}
	}
	return images, nil
}

// Validate currently enforces CLEF question and image limits for every model ID:
// 1–64 questions, 2–255 choices, 2–10 described score levels, and at most 4 inline
// PNG/JPEG/WebP images (4 MiB and 16 megapixels each, 8 MiB total decoded).
// Accepting a model selector does not establish that model's capabilities.
// The server enforces the 13 MiB encoded request limit; validation does not encode
// the whole request a second time merely to measure its size.
func (r *SystemOneRequest) Validate() error {
	var errs []error
	if strings.TrimSpace(r.Model) == "" {
		errs = append(errs, errors.New("model selector is required"))
	}
	if r.State == nil {
		errs = append(errs, errors.New("state is required"))
	} else if err := r.State.Validate(); err != nil {
		errs = append(errs, fmt.Errorf("state: %w", err))
	}
	if len(r.Questions) < 1 || len(r.Questions) > 64 {
		errs = append(errs, errors.New("1 to 64 questions are required"))
	}
	if err := r.Questions.Validate(); err != nil {
		errs = append(errs, err)
	}
	for _, id := range slices.Sorted(maps.Keys(r.Questions)) {
		q := r.Questions[id]
		if q == nil {
			continue
		}
		if id == "" || len(id) > 100 || strings.ContainsFunc(id, func(c rune) bool {
			return (c < 'a' || c > 'z') && (c < 'A' || c > 'Z') && (c < '0' || c > '9') && c != '_' && c != '.' && c != '-'
		}) {
			errs = append(errs, fmt.Errorf("invalid question ID %q", id))
		}
		if q.Instructions == nil {
			errs = append(errs, fmt.Errorf("question %q: instructions are required", id))
		}
		if s, ok := q.Instructions.(genai.Text); ok && strings.TrimSpace(string(s)) == "" {
			errs = append(errs, fmt.Errorf("question %q: instructions must not be empty", id))
		}
		if q.Type == genai.QuestionChoice {
			if len(q.Choice) < 2 || len(q.Choice) > 255 {
				errs = append(errs, fmt.Errorf("question %q: 2 to 255 choices are required", id))
			}
			if _, ok := q.Choice[""]; ok {
				errs = append(errs, fmt.Errorf("question %q: choice IDs must not be empty", id))
			}
		}
		if q.Type == genai.QuestionScore {
			if len(q.Score) < 2 || len(q.Score) > 10 {
				errs = append(errs, fmt.Errorf("question %q: 2 to 10 score levels are required", id))
			}
			for i, c := range q.Score {
				if c == nil {
					errs = append(errs, fmt.Errorf("question %q: score level %d must be described", id, i))
				}
			}
		}
	}
	if len(r.Images) > 4 {
		errs = append(errs, errors.New("at most 4 images are allowed"))
	} else {
		total := 0
		for j := range r.Images {
			n, err := r.Images[j].decodedSize()
			if err != nil {
				errs = append(errs, fmt.Errorf("image #%d: %w", j, err))
				continue
			}
			total += n
		}
		if total > 8*1024*1024 {
			errs = append(errs, errors.New("images exceed 8 MiB total decoded"))
		}
	}
	return errors.Join(errs...)
}

// SystemOneResponse is the Workers AI envelope around a hosted decision result.
// See https://developers.cloudflare.com/workers-ai/models/clef/schema-output.json.
type SystemOneResponse struct {
	Result   DecisionResult    `json:"result"`
	Success  bool              `json:"success"`
	Errors   []ErrorDetail     `json:"errors"`
	Messages []json.RawMessage `json:"messages"`
}

// UnmarshalJSON checks required fields and answer shapes before decoding the shared representation.
func (r *SystemOneResponse) UnmarshalJSON(b []byte) error {
	raw := decisionEnvelope{}
	if err := internal.UnmarshalJSON(b, &raw); err != nil {
		return err
	}
	v := SystemOneResponse{Success: raw.Success, Errors: raw.Errors, Messages: raw.Messages}
	if err := v.apiError(); err != nil {
		return err
	}
	if err := internal.UnmarshalJSON(raw.Result, &v.Result); err != nil {
		return err
	}
	required := requiredDecisionResult{}
	if err := json.Unmarshal(raw.Result, &required); err != nil {
		return err
	}
	if required.Model == nil || *required.Model == "" || len(required.Answers) == 0 || required.Usage.Input == nil || required.Usage.Output == nil {
		return errors.New("invalid or missing decision model, answers or usage")
	}
	if err := v.Result.Usage.Validate(); err != nil {
		return err
	}
	var errs []error
	for _, id := range slices.Sorted(maps.Keys(required.Answers)) {
		fields := required.Answers[id]
		a := v.Result.Answers[id]
		if a == nil {
			errs = append(errs, fmt.Errorf("answer %q: must not be nil", id))
			continue
		}
		keys := []string{"type"}
		switch a.Type {
		case genai.QuestionNoul:
			keys = append(keys, "noul")
		case genai.QuestionChoice:
			keys = append(keys, "choice", "probabilities", "confidence")
		case genai.QuestionScore:
			keys = append(keys, "score", "legend", "probabilities", "confidence")
		default:
			errs = append(errs, fmt.Errorf("answer %q: unknown type %q", id, a.Type))
			continue
		}
		missing := false
		for _, k := range keys {
			if data := fields[k]; len(data) == 0 || bytes.Equal(data, []byte("null")) {
				errs = append(errs, fmt.Errorf("answer %q: missing %s", id, k))
				missing = true
			}
		}
		if missing {
			continue
		}
		if !validProbability(a.Noul) || !validProbability(a.Confidence) {
			errs = append(errs, fmt.Errorf("answer %q: probability or confidence out of range", id))
		}
		if a.Type != genai.QuestionNoul {
			var probabilities map[string]*float64
			if err := json.Unmarshal(fields["probabilities"], &probabilities); err != nil {
				errs = append(errs, fmt.Errorf("answer %q: %w", id, err))
				continue
			}
			for _, k := range slices.Sorted(maps.Keys(probabilities)) {
				if p := probabilities[k]; p == nil {
					errs = append(errs, fmt.Errorf("answer %q: null probability for %q", id, k))
				} else if !validProbability(*p) {
					errs = append(errs, fmt.Errorf("answer %q: probability for %q out of range", id, k))
				}
			}
		}
	}
	if err := errors.Join(errs...); err != nil {
		return err
	}
	*r = v
	return nil
}

// To converts a successful hosted response to shared decision output.
func (r *SystemOneResponse) To(out *genai.SystemOneResponse) error {
	if err := r.apiError(); err != nil {
		return err
	}
	*out = r.Result.SystemOneResponse
	return nil
}

// decisionEnvelope preserves API errors without decoding a potentially invalid result.
type decisionEnvelope struct {
	Result   json.RawMessage   `json:"result"`
	Success  bool              `json:"success"`
	Errors   []ErrorDetail     `json:"errors"`
	Messages []json.RawMessage `json:"messages"`
}

// requiredDecisionResult tracks presence separately from the shared representation.
type requiredDecisionResult struct {
	Model   *string                               `json:"model"`
	Answers map[string]map[string]json.RawMessage `json:"answers"`
	Usage   requiredDecisionUsage                 `json:"usage"`
}

type requiredDecisionUsage struct {
	Input  *int64 `json:"input_tokens"`
	Output *int64 `json:"output_tokens"`
}

// apiError owns conversion of an unsuccessful Workers AI decision envelope.
func (r *SystemOneResponse) apiError() error {
	if r.Success && len(r.Errors) == 0 {
		return nil
	}
	return &ErrorResponse{Success: r.Success, Errors: r.Errors}
}

// validateQuestions checks every answer against the requested question and rubric.
func (r *SystemOneResponse) validateQuestions(qs *genai.Questions) error {
	var errs []error
	if len(r.Result.Answers) != len(*qs) {
		errs = append(errs, errors.New("decision answer count does not match questions"))
	}
	for _, id := range slices.Sorted(maps.Keys(*qs)) {
		q := (*qs)[id]
		a, ok := r.Result.Answers[id]
		if !ok || a == nil || a.Type != q.Type {
			errs = append(errs, fmt.Errorf("question %q: missing or mismatched answer", id))
			continue
		}
		switch q.Type {
		case genai.QuestionNoul:
		case genai.QuestionChoice:
			if _, ok := q.Choice[a.Choice]; !ok {
				errs = append(errs, fmt.Errorf("question %q: unknown chosen option", id))
			}
			if len(a.Probabilities) != len(q.Choice) {
				errs = append(errs, fmt.Errorf("question %q: missing choice probabilities", id))
			}
			for _, k := range slices.Sorted(maps.Keys(q.Choice)) {
				if _, ok := a.Probabilities[k]; !ok {
					errs = append(errs, fmt.Errorf("question %q: missing option %q", id, k))
				}
			}
		case genai.QuestionScore:
			if math.IsNaN(a.Score) || math.IsInf(a.Score, 0) || a.Score < 0 || a.Score > float64(len(q.Score)-1) || len(a.Legend) != len(q.Score) || len(a.Probabilities) != len(q.Score) {
				errs = append(errs, fmt.Errorf("question %q: invalid score or rubric", id))
			}
			for j := range q.Score {
				k := strconv.Itoa(j)
				if _, ok := a.Legend[k]; !ok {
					errs = append(errs, fmt.Errorf("question %q: missing legend level %s", id, k))
				}
				if _, ok := a.Probabilities[k]; !ok {
					errs = append(errs, fmt.Errorf("question %q: missing probability level %s", id, k))
				}
			}
		}
	}
	return errors.Join(errs...)
}

// DecisionResult is the native result inside a hosted decision response.
type DecisionResult struct {
	// Model identifies the loaded decision model that answered.
	Model string `json:"model"`
	genai.SystemOneResponse
}

func validProbability(p float64) bool {
	return !math.IsNaN(p) && !math.IsInf(p, 0) && p >= 0 && p <= 1
}
