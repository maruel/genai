// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Tests for hosted Cloudflare CLEF decisions and schema limits.

package cloudflare_test

import (
	"bytes"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
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
		for _, tc := range []struct{ name, mt, b string }{
			{"PNG", "image/png", base64.StdEncoding.EncodeToString(pngData)},
			{"JPEG", "image/jpeg", base64.StdEncoding.EncodeToString(jpg.Bytes())},
			{"WebP", "image/webp", "UklGRiQAAABXRUJQVlA4IBgAAAAwAQCdASoBAAEAAgA0JaQAA3AA/vuUAAA="},
		} {
			t.Run(tc.name, func(t *testing.T) {
				for _, i := range []cloudflare.DecisionImage{{ContentType: tc.mt, Base64: tc.b}, {DataURL: "DaTa:" + tc.mt + ";base64," + tc.b}} {
					if err := i.Validate(); err != nil {
						t.Fatal(err)
					}
				}
			})
		}
		largeDimensions := bytes.Clone(pngData)
		binary.BigEndian.PutUint32(largeDimensions[16:20], 4001)
		binary.BigEndian.PutUint32(largeDimensions[20:24], 4000)
		binary.BigEndian.PutUint32(largeDimensions[29:33], crc32.ChecksumIEEE(largeDimensions[12:29]))
		t.Run("error", func(t *testing.T) {
			for name, i := range map[string]cloudflare.DecisionImage{
				"remote":          {DataURL: "https://example.com/a.png"},
				"unsupported":     {ContentType: "image/gif", Base64: base64.StdEncoding.EncodeToString(pngData)},
				"mismatch":        {ContentType: "image/jpeg", Base64: base64.StdEncoding.EncodeToString(pngData)},
				"bad base64":      {ContentType: "image/png", Base64: "!!"},
				"bad base64 tail": {ContentType: "image/png", Base64: base64.StdEncoding.EncodeToString(append(bytes.Clone(pngData), make([]byte, 4096)...)) + "!"},
				"bad bytes":       {ContentType: "image/png", Base64: base64.StdEncoding.EncodeToString([]byte("bad"))},
				"empty":           {ContentType: "image/png"},
				"too big":         {ContentType: "image/png", Base64: base64.StdEncoding.EncodeToString(make([]byte, 4*1024*1024+1))},
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
	t.Run("From", func(t *testing.T) {
		for _, tc := range []struct {
			name   string
			in     genai.SystemOneRequest
			state  string
			images int
		}{
			{"text", genai.SystemOneRequest{State: genai.Text("hello")}, `"hello"`, 0},
			{"array", genai.SystemOneRequest{State: genai.Array{"one", "two"}}, `["one","two"]`, 0},
			{"image only", genai.SystemOneRequest{State: genai.Text(""), Docs: []genai.Doc{{Filename: "a.png", Src: bytes.NewReader(clefPNG(t))}}}, `""`, 1},
			{"structured array", genai.SystemOneRequest{State: genai.Array{1, genai.Object{"ticket": 42}}}, `[1,{"ticket":42}]`, 0},
		} {
			t.Run(tc.name, func(t *testing.T) {
				r := cloudflare.SystemOneRequest{Model: "clef"}
				tc.in.Questions = clefQuestions()
				if err := r.From(&tc.in); err != nil {
					t.Fatal(err)
				}
				b, err := json.Marshal(r.State)
				if err != nil {
					t.Fatal(err)
				}
				if string(b) != tc.state || len(r.Images) != tc.images || r.Model != "clef" || len(r.Questions) != 3 {
					t.Errorf("state=%s images=%d", b, len(r.Images))
				}
				if err := r.Validate(); err != nil {
					t.Fatal(err)
				}
			})
		}
	})
	t.Run("Validate", func(t *testing.T) {
		t.Run("deterministic joined errors", func(t *testing.T) {
			r := cloudflare.SystemOneRequest{Model: "clef", State: genai.Text("state"), Questions: genai.Questions{
				"z": {Type: genai.QuestionNoul, Instructions: genai.Text("")},
				"a": {Type: genai.QuestionNoul, Instructions: genai.Text("")},
			}}
			want := "question \"a\": instructions must not be empty\nquestion \"z\": instructions must not be empty"
			for range 20 {
				if err := r.Validate(); err == nil || err.Error() != want {
					t.Fatalf("got %v, want %s", err, want)
				}
			}
		})
		t.Run("future model", func(t *testing.T) {
			r := cloudflare.SystemOneRequest{Model: "decision-v2", State: genai.Text("state"), Questions: clefQuestions()}
			if err := r.Validate(); err != nil {
				t.Fatal(err)
			}
		})
		for name, change := range map[string]func(*cloudflare.SystemOneRequest){
			"missing model": func(r *cloudflare.SystemOneRequest) { r.Model = "" },
			"blank model":   func(r *cloudflare.SystemOneRequest) { r.Model = " \t\n" },
			"no state":      func(r *cloudflare.SystemOneRequest) { r.State = nil },
			"bad state":     func(r *cloudflare.SystemOneRequest) { r.State = genai.Object{"bad": make(chan int)} },
			"no questions":  func(r *cloudflare.SystemOneRequest) { r.Questions = nil },
			"too many questions": func(r *cloudflare.SystemOneRequest) {
				for n := range 65 {
					r.Questions[strings.Repeat("a", n+1)] = r.Questions["billing"]
				}
			},
			"invalid ID": func(r *cloudflare.SystemOneRequest) { r.Questions["space id"] = r.Questions["billing"] },
			"long ID":    func(r *cloudflare.SystemOneRequest) { r.Questions[strings.Repeat("a", 101)] = r.Questions["billing"] },
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
			"too many choices": func(r *cloudflare.SystemOneRequest) {
				for n := range 256 {
					r.Questions["route"].Choice[strings.Repeat("a", n+1)] = nil
				}
			},
			"one score": func(r *cloudflare.SystemOneRequest) {
				q := r.Questions["urgency"]
				q.Score = q.Score[:1]
				r.Questions["urgency"] = q
			},
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
		t.Run("maximum questions choices and scores", func(t *testing.T) {
			r := cloudflare.SystemOneRequest{Model: "clef", State: genai.Text("state"), Questions: genai.Questions{}}
			for n := range 64 {
				r.Questions[strings.Repeat("a", n+1)] = &genai.Question{Type: genai.QuestionNoul, Instructions: genai.Text("yes?")}
			}
			q := &genai.Question{Type: genai.QuestionChoice, Instructions: genai.Text("choose"), Choice: map[string]genai.DecisionContent{}}
			for n := range 255 {
				q.Choice[strings.Repeat("a", n+1)] = nil
			}
			r.Questions["a"] = q
			r.Questions["aa"] = &genai.Question{Type: genai.QuestionScore, Instructions: genai.Text("score"), Score: []genai.DecisionContent{genai.Text("0"), genai.Text("1"), genai.Text("2"), genai.Text("3"), genai.Text("4"), genai.Text("5"), genai.Text("6"), genai.Text("7"), genai.Text("8"), genai.Text("9")}}
			if err := r.Validate(); err != nil {
				t.Fatal(err)
			}
		})
		t.Run("valid hosted IDs", func(t *testing.T) {
			r := cloudflare.SystemOneRequest{Model: "clef", State: genai.Object{}, Questions: genai.Questions{"AZaz09_.-": {Type: genai.QuestionNoul, Instructions: genai.Text("yes?")}}}
			if err := r.Validate(); err != nil {
				t.Fatal(err)
			}
		})
	})
}

func TestSystemOneResponse(t *testing.T) {
	t.Run("UnmarshalJSON", func(t *testing.T) {
		r := cloudflare.SystemOneResponse{}
		err := json.Unmarshal([]byte(`{"success":false,"errors":[{"code":1000,"message":"failed"}],"result":{"invalid":"result"}}`), &r)
		api, ok := errors.AsType[*cloudflare.ErrorResponse](err)
		if !ok || len(api.Errors) != 1 || api.Errors[0].Code != 1000 || api.Errors[0].Message != "failed" {
			t.Fatalf("lost API details: %v", err)
		}
	})
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
