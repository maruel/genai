// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Tests for embedding qualification and failed probes.

package smoke_test

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"slices"
	"strings"
	"testing"

	"github.com/maruel/genai"
	"github.com/maruel/genai/base"
	"github.com/maruel/genai/scoreboard"
	"github.com/maruel/genai/smoke"
)

func TestRunEmbeddings(t *testing.T) {
	unsupportedDims := fmt.Errorf("wrapped: %w", &base.ErrNotSupported{Options: []string{"EmbeddingRequest.Dimensions"}})
	unsupportedOther := &base.ErrNotSupported{Options: []string{"GenOptionText.Seed"}}
	failed := errors.New("failed")
	for _, tc := range []embeddingProbeCase{
		{name: "routing modalities", probe: 32, supported: new(true), reporting: scoreboard.True},
		{name: "routing scoreboard", probe: 32, supported: new(true), reporting: scoreboard.True},
		{name: "valid", probe: 32, supported: new(true), reporting: scoreboard.True},
		{name: "no size probe", reporting: scoreboard.True},
		{name: "unreported usage", defect: "usage", probe: 32, supported: new(true)},
		{name: "flaky usage", defect: "flaky usage", reporting: scoreboard.Flaky},
		{name: "unsupported dimensions", probe: 32, dimErr: unsupportedDims, supported: new(false), reporting: scoreboard.True},
		{name: "dimensions failure", probe: 32, dimErr: failed, wantErr: true},
		{name: "wrong unsupported option", probe: 32, dimErr: unsupportedOther, wantErr: true},
		{name: "wrong dimensions", probe: 32, defect: "dimensions", wantErr: true},
		{name: "wrong order", defect: "order", wantErr: true},
		{name: "bad retrieval", defect: "retrieval", wantErr: true},
		{name: "missing vector", defect: "missing", wantErr: true},
		{name: "zero vector", defect: "zero", wantErr: true},
		{name: "negative usage", defect: "negative usage", wantErr: true},
		{name: "bad default batch", defect: "initial", wantErr: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			p := &embeddingProvider{t: t, defect: tc.defect, dimErr: tc.dimErr}
			switch tc.name {
			case "routing modalities":
				p.mods = genai.Modalities{genai.ModalityEmbedding}
			case "routing scoreboard":
				p.score = scoreboard.Score{Scenarios: []scoreboard.Scenario{{Models: []string{"model"}, Embed: &scoreboard.EmbeddingFunctionality{}}}}
			}
			var sc scoreboard.Scenario
			var u genai.Usage
			var err error
			if strings.HasPrefix(tc.name, "routing") {
				sc, u, err = smoke.Run(t.Context(), func(string) genai.Provider { return p })
			} else {
				sc, u, err = smoke.RunEmbeddings(t.Context(), func(string) genai.Provider { return p }, tc.probe)
			}
			if (err != nil) != tc.wantErr {
				t.Fatalf("scenario %+v usage %+v err %v", sc, u, err)
			}
			if tc.wantErr {
				if sc.Embed != nil {
					t.Fatal("failed qualification claimed support")
				}
				return
			}
			if err := sc.Validate(); err != nil {
				t.Fatal(err)
			}
			if sc.Embed == nil || sc.Embed.Dimensions != 2 || sc.Embed.ReportTokenUsage != tc.reporting {
				t.Fatalf("functionality %+v", sc.Embed)
			}
			if (sc.Embed.RequestedDimensions == nil) != (tc.supported == nil) || tc.supported != nil && *sc.Embed.RequestedDimensions != *tc.supported {
				t.Fatalf("requested size support %+v", sc.Embed)
			}
			if tc.probe == 0 && p.calls != 5 || tc.probe > 0 && p.calls != 6 {
				t.Fatalf("requests %d", p.calls)
			}
			if tc.reporting == scoreboard.True && (u.InputTokens < 14 || u.TotalTokens != u.InputTokens) {
				t.Fatalf("usage %+v", u)
			}
		})
	}

	t.Run("Media", func(t *testing.T) {
		unsupportedDoc := fmt.Errorf("wrapped: %w", &base.ErrNotSupported{Options: []string{"Request.Doc"}})
		cases := []embeddingMediaCase{
			{name: "supported"},
			{name: "unsupported", err: unsupportedDoc, unsupported: true},
			{name: "wrong unsupported option", err: unsupportedOther, wantErr: true},
			{name: "failure", err: failed, wantErr: true},
			{name: "audio failure", err: failed, audioOnly: true, wantErr: true},
			{name: "missing", defect: "media missing", wantErr: true},
			{name: "extra", defect: "media extra", wantErr: true},
			{name: "zero", defect: "media zero", wantErr: true},
			{name: "width", defect: "media width", wantErr: true},
			{name: "usage", defect: "media usage", wantErr: true},
		}
		for _, group := range []string{"Valid", "Error"} {
			t.Run(group, func(t *testing.T) {
				for _, tc := range cases {
					if tc.wantErr != (group == "Error") {
						continue
					}
					t.Run(tc.name, func(t *testing.T) {
						p := &embeddingProvider{t: t, defect: tc.defect, mediaErr: tc.err, audioOnly: tc.audioOnly}
						sc, u, err := smoke.RunEmbeddings(t.Context(), func(string) genai.Provider { return p }, 0)
						if (err != nil) != tc.wantErr {
							t.Fatalf("scenario %+v usage %+v error %v", sc, u, err)
						}
						if tc.wantErr {
							if sc.Embed != nil {
								t.Fatal("failed media qualification claimed support")
							}
							if u.InputTokens < 21 {
								t.Fatalf("lost earlier usage %+v", u)
							}
							return
						}
						if p.mediaCalls != 2 {
							t.Fatalf("media calls %d", p.mediaCalls)
						}
						for _, m := range []genai.Modality{genai.ModalityImage, genai.ModalityAudio} {
							mc, ok := sc.In[m]
							if ok == tc.unsupported {
								t.Fatalf("input capability %s %+v", m, mc)
							}
							if ok && (!mc.Inline || len(mc.SupportedFormats) != 1) {
								t.Fatalf("capability %+v", mc)
							}
						}
						wantTokens := int64(35)
						if tc.unsupported {
							wantTokens = 21
						}
						if u.InputTokens != wantTokens || u.TotalTokens != wantTokens {
							t.Fatalf("usage %+v", u)
						}
					})
				}
			})
		}
	})
}

type embeddingProbeCase struct {
	name      string
	probe     int
	dimErr    error
	defect    string
	wantErr   bool
	supported *bool
	reporting scoreboard.TriState
}

type embeddingMediaCase struct {
	name, defect                    string
	err                             error
	unsupported, wantErr, audioOnly bool
}

// embeddingProvider is a fake embedding model with 2-dimension default vectors
// and 7 input tokens per call. defect selects one malformed response.
type embeddingProvider struct {
	base.NotImplemented
	t                *testing.T
	mods             genai.Modalities
	score            scoreboard.Score
	defect           string
	dimErr, mediaErr error
	audioOnly        bool
	calls            int
	mediaCalls       int
}

func (p *embeddingProvider) Close() error                       { return nil }
func (p *embeddingProvider) Name() string                       { return "embedding" }
func (p *embeddingProvider) ModelID() string                    { return "model" }
func (p *embeddingProvider) OutputModalities() genai.Modalities { return p.mods }
func (p *embeddingProvider) HTTPClient() *http.Client           { return nil }
func (p *embeddingProvider) Scoreboard() scoreboard.Score       { return p.score }

func (p *embeddingProvider) Embed(_ context.Context, in *genai.EmbeddingRequest) (*genai.EmbeddingResponse, error) {
	p.calls++
	if p.defect == "initial" {
		return nil, errors.New("failed")
	}
	if doc := in.Inputs[0].Doc; !doc.IsZero() {
		return p.embedMedia(in)
	}
	if in.Dimensions != 0 && p.dimErr != nil {
		return nil, p.dimErr
	}
	n := in.Dimensions
	if n == 0 || p.defect == "dimensions" {
		n = 2
	}
	out := &genai.EmbeddingResponse{Usage: genai.Usage{InputTokens: 7, TotalTokens: 7}}
	for _, r := range in.Inputs {
		v := make([]float32, n)
		switch {
		case strings.Contains(r.Text, "query:"):
			v[0] = 2
		case strings.Contains(r.Text, "charged particles"):
			v[0] = 3
			v[1] = 1
			if p.defect == "retrieval" {
				v[0] = -3
			}
		default:
			v[0] = -1
			v[1] = 2
		}
		if p.defect == "zero" {
			clear(v)
		}
		out.Embeddings = append(out.Embeddings, v)
	}
	switch p.defect {
	case "order":
		slices.Reverse(out.Embeddings)
	case "missing":
		out.Embeddings = out.Embeddings[:2]
	case "negative usage":
		out.Usage = genai.Usage{InputTokens: -1}
	}
	if p.defect == "usage" || p.defect == "flaky usage" && p.calls == 2 {
		out.Usage = genai.Usage{}
	}
	return out, nil
}

func (p *embeddingProvider) embedMedia(in *genai.EmbeddingRequest) (*genai.EmbeddingResponse, error) {
	p.mediaCalls++
	doc := in.Inputs[0].Doc
	want, err := scoreboard.TestdataFiles.ReadFile("testdata/" + doc.Filename)
	if err != nil {
		return nil, err
	}
	got, err := io.ReadAll(doc.Src)
	if err != nil {
		return nil, err
	}
	if len(in.Inputs) != 1 || in.Inputs[0].Text != "" || in.Dimensions != 0 || !bytes.Equal(got, want) {
		p.t.Error("probe must contain the complete standalone media input")
	}
	if p.mediaErr != nil && (!p.audioOnly || doc.Filename == "audio.wav") {
		return nil, p.mediaErr
	}
	out := &genai.EmbeddingResponse{Embeddings: [][]float32{{1, 2}}, Usage: genai.Usage{InputTokens: 7, TotalTokens: 7}}
	switch p.defect {
	case "media missing":
		out.Embeddings = nil
	case "media extra":
		out.Embeddings = append(out.Embeddings, []float32{1, 2})
	case "media zero":
		out.Embeddings[0] = []float32{0, 0}
	case "media width":
		out.Embeddings[0] = []float32{1, 2, 3}
	case "media usage":
		out.Usage.InputTokens = -1
	case "usage":
		out.Usage = genai.Usage{}
	}
	return out, nil
}
