// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Smoke qualification for text and media embeddings.

package smoke

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"net/http"
	"slices"
	"strings"

	"github.com/maruel/httpjson"

	"github.com/maruel/genai"
	"github.com/maruel/genai/base"
	"github.com/maruel/genai/scoreboard"
)

// EmbeddingProbeDimensions is the requested vector size used by Run and the
// scoreboard test updater. Qualification also measures the model's default size.
const EmbeddingProbeDimensions = 32

// RunEmbeddings qualifies default vectors, input order, semantic retrieval and
// token reporting, plus standalone image and audio inputs. A positive dimensions
// value also probes that requested size;
// zero leaves size selection unmeasured. Only explicit unsupported-option errors
// are negative evidence; authentication, transport and malformed output abort.
// ProviderFactory must never fail. Failed qualification returns no Embed result.
func RunEmbeddings(ctx context.Context, pf ProviderFactory, dimensions int) (scoreboard.Scenario, genai.Usage, error) {
	u := genai.Usage{}
	if dimensions < 0 {
		return scoreboard.Scenario{}, u, errors.New("embedding qualification dimensions must not be negative")
	}
	c := pf("")
	model := c.ModelID()
	if model == "" {
		return scoreboard.Scenario{}, u, errors.New("provider must have a model")
	}
	texts := []genai.Request{
		{Text: "task: search result | query: What causes the northern lights?"},
		{Text: "title: none | text: The northern lights are caused by charged particles from the sun."},
		{Text: "title: none | text: Bananas are a popular tropical fruit."},
	}
	f := scoreboard.EmbeddingFunctionality{}
	calls := 0
	call := func(name string, in *genai.EmbeddingRequest) (*genai.EmbeddingResponse, error) {
		out, err := pf("Embed-"+name).Embed(ctx, in)
		if err != nil {
			return nil, err
		}
		if out == nil {
			return nil, errors.New("embedding returned a nil response")
		}
		if err := out.Validate(); err != nil {
			return nil, err
		}
		if len(out.Embeddings) != len(in.Inputs) {
			return nil, fmt.Errorf("got %d embedding vectors for %d inputs", len(out.Embeddings), len(in.Inputs))
		}
		if in.Dimensions != 0 && len(out.Embeddings[0]) != in.Dimensions {
			return nil, fmt.Errorf("got %d embedding dimensions, want %d", len(out.Embeddings[0]), in.Dimensions)
		}
		if out.Usage.InputTokens < 0 || out.Usage.TotalTokens < out.Usage.InputTokens {
			return nil, errors.New("invalid embedding token usage")
		}
		reporting := scoreboard.False
		if out.Usage.InputTokens > 0 {
			reporting = scoreboard.True
		}
		if calls == 0 {
			f.ReportTokenUsage = reporting
		} else if f.ReportTokenUsage != reporting {
			f.ReportTokenUsage = scoreboard.Flaky
		}
		calls++
		u.Add(&out.Usage)
		return out, nil
	}
	out, err := call("Batch", &genai.EmbeddingRequest{Inputs: texts})
	if err != nil {
		return scoreboard.Scenario{}, u, fmt.Errorf("embedding batch: %w", err)
	}
	if embeddingCosine(out.Embeddings[0], out.Embeddings[1]) <= embeddingCosine(out.Embeddings[0], out.Embeddings[2]) {
		return scoreboard.Scenario{}, u, errors.New("embedding semantic retrieval ranked unrelated text first")
	}
	f.Dimensions = len(out.Embeddings[0])
	for _, probe := range []embeddingOrderProbe{{"Order", []int{2, 1, 0}}, {"Rotate", []int{1, 2, 0}}} {
		batch := make([]genai.Request, len(texts))
		for i, j := range probe.order {
			batch[i] = texts[j]
		}
		permuted, err := call(probe.name, &genai.EmbeddingRequest{Inputs: batch})
		if err != nil {
			return scoreboard.Scenario{}, u, fmt.Errorf("embedding order: %w", err)
		}
		for i, j := range probe.order {
			v, w := out.Embeddings[j], permuted.Embeddings[i]
			if len(v) != len(w) || embeddingCosine(v, w) < 0.99 {
				return scoreboard.Scenario{}, u, errors.New("embedding results did not preserve input order")
			}
		}
	}
	if dimensions > 0 {
		resized, err := call("Dimensions", &genai.EmbeddingRequest{Inputs: texts, Dimensions: dimensions})
		supported := true
		if err != nil {
			if !unsupportedEmbeddingDimensions(err) {
				return scoreboard.Scenario{}, u, fmt.Errorf("embedding dimensions: %w", err)
			}
			supported = false
		} else if embeddingCosine(resized.Embeddings[0], resized.Embeddings[1]) <= embeddingCosine(resized.Embeddings[0], resized.Embeddings[2]) {
			return scoreboard.Scenario{}, u, errors.New("resized embedding semantic retrieval ranked unrelated text first")
		}
		f.RequestedDimensions = &supported
	}
	sc := scoreboard.Scenario{Models: []string{model}, In: map[genai.Modality]scoreboard.ModalCapability{genai.ModalityText: {Inline: true}}, Out: map[genai.Modality]scoreboard.ModalCapability{genai.ModalityEmbedding: {Inline: true}}, Embed: &f}
	for _, probe := range []embeddingMediaProbe{{"Image", "image.jpg", "image/jpeg", genai.ModalityImage}, {"Audio", "audio.wav", "audio/wav", genai.ModalityAudio}} {
		data, err := scoreboard.TestdataFiles.ReadFile("testdata/" + probe.filename)
		if err != nil {
			return scoreboard.Scenario{}, u, err
		}
		media, err := call(probe.name, &genai.EmbeddingRequest{Inputs: []genai.Request{{Doc: genai.Doc{Filename: probe.filename, Src: bytes.NewReader(data)}}}})
		if err != nil {
			if unsupportedEmbeddingMedia(err, probe.modality) {
				continue
			}
			return scoreboard.Scenario{}, u, fmt.Errorf("embedding %s: %w", probe.modality, err)
		}
		if len(media.Embeddings[0]) != f.Dimensions {
			return scoreboard.Scenario{}, u, fmt.Errorf("embedding %s has %d dimensions, want default %d", probe.modality, len(media.Embeddings[0]), f.Dimensions)
		}
		sc.In[probe.modality] = scoreboard.ModalCapability{Inline: true, SupportedFormats: []string{probe.mime}}
	}
	return sc, u, nil
}

func embeddingCosine(a, b []float32) float64 {
	var dot, x, y float64
	for i, v := range a {
		av, bv := float64(v), float64(b[i])
		dot += av * bv
		x += av * av
		y += bv * bv
	}
	return dot / math.Sqrt(x*y)
}

func unsupportedEmbeddingDimensions(err error) bool {
	if e, ok := errors.AsType[*base.ErrNotSupported](err); ok {
		return slices.Contains(e.Options, "EmbeddingRequest.Dimensions")
	}
	e, ok := errors.AsType[*httpjson.Error](err)
	if !ok || e.StatusCode != http.StatusBadRequest {
		return false
	}
	// A generic 400 could represent a broken request or upstream failure. Only
	// the explicit unsupported_parameter discriminator identifies this option.
	var body embeddingOptionErrorResponse
	if json.Unmarshal(e.ResponseBody, &body) != nil {
		return false
	}
	return body.Error.Code == "unsupported_parameter" && strings.EqualFold(body.Error.Param, "dimensions")
}

type embeddingOrderProbe struct {
	name  string
	order []int
}

type embeddingOptionErrorResponse struct {
	Error embeddingOptionError `json:"error"`
}

type embeddingOptionError struct {
	Code  string `json:"code"`
	Param string `json:"param"`
}

// unsupportedEmbeddingMedia recognizes negative evidence only for nonempty media-only probes.
func unsupportedEmbeddingMedia(err error, modality genai.Modality) bool {
	if e, ok := errors.AsType[*base.ErrNotSupported](err); ok {
		return slices.Contains(e.Options, "Request.Doc")
	}
	e, ok := errors.AsType[*httpjson.Error](err)
	if !ok {
		return false
	}
	var body embeddingMediaErrorResponse
	if json.Unmarshal(e.ResponseBody, &body) != nil {
		return false
	}
	if e.StatusCode == http.StatusBadRequest && body.Error.Code == http.StatusBadRequest && body.Error.Status == "INVALID_ARGUMENT" && body.Error.Message == "The text content is empty." {
		// A valid media-only input requiring text identifies an endpoint that cannot embed this modality.
		return true
	}
	return e.StatusCode == http.StatusInternalServerError && body.Error.Code == http.StatusInternalServerError && body.Error.Type == "server_error" && body.Error.Message == fmt.Sprintf("%s input is not supported - hint: if this is unexpected, you may need to provide the mmproj", modality)
}

type embeddingMediaProbe struct {
	name, filename, mime string
	modality             genai.Modality
}
type embeddingMediaErrorResponse struct {
	Error embeddingMediaError `json:"error"`
}
type embeddingMediaError struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
	Status  string `json:"status"`
	Type    string `json:"type"`
}
