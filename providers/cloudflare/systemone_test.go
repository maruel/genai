// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Tests for hosted Cloudflare CLEF decisions and schema limits.

package cloudflare_test

import (
	"bytes"
	"encoding/base64"
	"encoding/binary"
	"errors"
	"hash/crc32"
	"image"
	"image/jpeg"
	"image/png"
	"reflect"
	"strings"
	"testing"

	"github.com/maruel/genai"
	"github.com/maruel/genai/providers/cloudflare"
)

func clefPNG(t *testing.T) []byte {
	var b bytes.Buffer
	if err := png.Encode(&b, image.NewRGBA(image.Rect(0, 0, 1, 1))); err != nil {
		t.Fatal(err)
	}
	return b.Bytes()
}

func TestDecisionImage(t *testing.T) {
	t.Run("Validate", func(t *testing.T) {
		pngData := clefPNG(t)
		var jpg bytes.Buffer
		if err := jpeg.Encode(&jpg, image.NewRGBA(image.Rect(0, 0, 1, 1)), nil); err != nil {
			t.Fatal(err)
		}
		largeDimensions := bytes.Clone(pngData)
		binary.BigEndian.PutUint32(largeDimensions[16:20], 4001)
		binary.BigEndian.PutUint32(largeDimensions[20:24], 4000)
		binary.BigEndian.PutUint32(largeDimensions[29:33], crc32.ChecksumIEEE(largeDimensions[12:29]))
		t.Run("error", func(t *testing.T) {
			for name, i := range map[string]cloudflare.DecisionImage{
				"remote":          {DataURL: "https://example.com/a.png"},
				"unsupported":     {ContentType: "image/gif", Base64: base64.StdEncoding.EncodeToString(pngData)},
				"bad base64 tail": {ContentType: "image/png", Base64: base64.StdEncoding.EncodeToString(append(bytes.Clone(pngData), make([]byte, 4096)...)) + "!"},
				"bad bytes":       {ContentType: "image/png", Base64: base64.StdEncoding.EncodeToString([]byte("bad"))},
				"empty":           {ContentType: "image/png"},
				"too many pixels": {ContentType: "image/png", Base64: base64.StdEncoding.EncodeToString(largeDimensions)},
				"both forms":      {DataURL: "data:image/png;base64," + base64.StdEncoding.EncodeToString(pngData), ContentType: "image/png"},
			} {
				t.Run(name, func(t *testing.T) {
					if err := i.Validate(); err == nil {
						t.Fatal("expected error")
					}
				})
			}
		})
	})
}

func TestSystemOneRequest(t *testing.T) {
	t.Run("Validate", func(t *testing.T) {
		for name, change := range map[string]func(*cloudflare.SystemOneRequest){
			"blank model":  func(r *cloudflare.SystemOneRequest) { r.Model = " \t\n" },
			"no state":     func(r *cloudflare.SystemOneRequest) { r.State = nil },
			"bad state":    func(r *cloudflare.SystemOneRequest) { r.State = genai.Object{"bad": make(chan int)} },
			"no questions": func(r *cloudflare.SystemOneRequest) { r.Questions = nil },
			"long ID":      func(r *cloudflare.SystemOneRequest) { r.Questions[strings.Repeat("a", 101)] = r.Questions["billing"] },
			"empty instructions": func(r *cloudflare.SystemOneRequest) {
				r.Questions["billing"] = &genai.Question{Type: genai.QuestionNoul, Instructions: genai.Text("")}
			},
			"missing instructions": func(r *cloudflare.SystemOneRequest) {
				r.Questions["billing"] = &genai.Question{Type: genai.QuestionNoul}
			},
			"one choice": func(r *cloudflare.SystemOneRequest) {
				q := r.Questions["route"]
				q.Choice = map[string]genai.DecisionContent{"one": nil}
				r.Questions["route"] = q
			},
			"empty choice ID": func(r *cloudflare.SystemOneRequest) { r.Questions["route"].Choice[""] = nil },
			"too many scores": func(r *cloudflare.SystemOneRequest) {
				q := r.Questions["urgency"]
				for range 11 {
					q.Score = append(q.Score, genai.Text("level"))
				}
				r.Questions["urgency"] = q
			},
			"null score":      func(r *cloudflare.SystemOneRequest) { r.Questions["urgency"].Score[0] = nil },
			"too many images": func(r *cloudflare.SystemOneRequest) { r.Images = make([]cloudflare.DecisionImage, 5) },
		} {
			t.Run(name, func(t *testing.T) {
				r := cloudflare.SystemOneRequest{Model: "clef", State: genai.Text("x"), Questions: clefQuestions()}
				change(&r)
				if err := r.Validate(); err == nil {
					t.Fatal("expected error")
				}
			})
		}
		t.Run("image total and boundary", func(t *testing.T) {
			data := append(clefPNG(t), make([]byte, 4*1024*1024-len(clefPNG(t)))...)
			i := cloudflare.DecisionImage{ContentType: "image/png", Base64: base64.StdEncoding.EncodeToString(data)}
			r := cloudflare.SystemOneRequest{Model: "clef-flash", State: genai.Text(""), Questions: clefQuestions(), Images: []cloudflare.DecisionImage{i, i}}
			if err := r.Validate(); err != nil {
				t.Fatal(err)
			}
			r.Images = append(r.Images, cloudflare.DecisionImage{ContentType: "image/png", Base64: base64.StdEncoding.EncodeToString(clefPNG(t))})
			if err := r.Validate(); err == nil {
				t.Fatal("expected total size error")
			}
		})
	})
}

func TestSystemOneResponse(t *testing.T) {
	t.Run("To", func(t *testing.T) {
		t.Run("valid", func(t *testing.T) {
			r := cloudflare.SystemOneResponse{Success: true, Result: cloudflare.DecisionResult{Model: "versioned", SystemOneResponse: genai.SystemOneResponse{
				Answers: genai.Answers{"q": {Type: genai.QuestionNoul, Noul: 0}},
				Usage:   genai.DecisionUsage{InputTokens: 42, OutputTokens: 3, ReasoningTokens: 2},
			}}}
			var out genai.SystemOneResponse
			if err := r.To(&out); err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(out, r.Result.SystemOneResponse) {
				t.Fatalf("lost decision output: %+v", out)
			}
		})
		r := cloudflare.SystemOneResponse{Success: true, Errors: []cloudflare.ErrorDetail{{Code: 1000, Message: "failed"}}}
		out := genai.SystemOneResponse{}
		err := r.To(&out)
		api, ok := errors.AsType[*cloudflare.ErrorResponse](err)
		if !ok || !api.Success || len(api.Errors) != 1 || api.Errors[0].Code != 1000 || api.Errors[0].Message != "failed" {
			t.Fatalf("lost API details: %v", err)
		}
	})
}
