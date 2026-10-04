// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Tests for the shared decision API across supported providers.

package providers

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"image"
	"image/jpeg"
	"image/png"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"

	"github.com/maruel/genai"
	"github.com/maruel/genai/providers/cloudflare"
	"github.com/maruel/genai/providers/llamacpp"
	"github.com/maruel/genai/providers/typesafe"
)

func TestSystemOne(t *testing.T) {
	t.Run("docs", testSystemOneDocs)
	for _, name := range []string{"cloudflare", "llamacpp", "typesafe"} {
		t.Run(name, func(t *testing.T) {
			for _, model := range []string{"default", "configured"} {
				t.Run("model="+model, func(t *testing.T) {
					want := model
					id := strings.Join([]string{"billing", name, want}, ".")
					calls := 0
					wantState := `"ticket"`
					answer := `{"type":"noul","noul":0.9}`
					srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
						calls++
						var body struct {
							Model     string                     `json:"model"`
							State     json.RawMessage            `json:"state"`
							Questions map[string]json.RawMessage `json:"questions"`
						}
						if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
							t.Error(err)
						}
						if body.Model != want || string(body.State) != wantState || len(body.Questions) != 1 || body.Questions[id] == nil {
							t.Errorf("unexpected body: %+v", body)
						}
						path := "/v1/systemone"
						response := fmt.Sprintf(`{"model":"versioned","answers":{%q:%s},"usage":{"input_tokens":42,"output_tokens":0}}`, id, answer)
						if name == "cloudflare" {
							path = "/client/v4/accounts/account/ai/run/@cf/cloudflare/" + want
							response = `{"success":true,"errors":[],"messages":[],"result":` + response + `}`
						}
						if r.Method != http.MethodPost || r.URL.Path != path {
							t.Errorf("unexpected route: %s %s", r.Method, r.URL.Path)
						}
						w.Header().Set("Content-Type", "application/json")
						if _, err := w.Write([]byte(response)); err != nil {
							t.Error(err)
						}
					}))
					t.Cleanup(srv.Close)
					var c genai.Provider
					var err error
					switch name {
					case "cloudflare":
						c, err = cloudflare.New(t.Context(), genai.ProviderOptionAPIKey("fake"), cloudflare.AccountID("account"), genai.ProviderOptionModel("@cf/cloudflare/"+model), genai.ProviderOptionTransportWrapper(func(http.RoundTripper) http.RoundTripper {
							return decisionTransport{srv: srv}
						}))
					case "llamacpp":
						c, err = llamacpp.New(t.Context(), genai.ProviderOptionModel(model), genai.ProviderOptionRemote(srv.URL))
					case "typesafe":
						c, err = typesafe.New(t.Context(), genai.ProviderOptionAPIKey("fake"), genai.ProviderOptionModel(model), genai.ProviderOptionRemote(srv.URL))
					}
					if err != nil {
						t.Fatal(err)
					}
					t.Cleanup(func() {
						if err := c.Close(); err != nil {
							t.Error(err)
						}
					})
					if !c.Capabilities().SystemOne {
						t.Fatal("missing decision capability")
					}
					for _, req := range []*genai.SystemOneRequest{{}, {State: genai.Text("ticket")}, {State: genai.Text("ticket"), Questions: genai.Questions{id: nil}}} {
						if out, err := c.SystemOne(t.Context(), req); out != nil || err == nil {
							t.Fatalf("invalid request returned %v, %v", out, err)
						}
					}
					if calls != 0 {
						t.Fatal("invalid request reached transport")
					}
					req := genai.SystemOneRequest{State: genai.Text("ticket"), Questions: genai.Questions{id: {Type: genai.QuestionNoul, Instructions: genai.Text("Is this billing?")}}}
					out, err := c.SystemOne(t.Context(), &req)
					if err != nil {
						t.Fatal(err)
					}
					if calls != 1 || out.Answers[id].Noul != 0.9 || out.Usage.InputTokens != 42 {
						t.Fatalf("unexpected response: %+v; calls=%d", out, calls)
					}
					t.Run("JSON state", func(t *testing.T) {
						for _, raw := range []string{`{"z":1,"a":2}`, `[1,{"x":2}]`, `"state"`} {
							wantState = raw
							in := genai.SystemOneRequest{Questions: req.Questions, Docs: []genai.Doc{{Filename: "state.json", Src: strings.NewReader(raw)}}}
							before := calls
							out, err := c.SystemOne(t.Context(), &in)
							if err != nil || calls != before+1 || out.Answers[id].Noul != 0.9 {
								t.Fatalf("JSON state response: %+v, %v; calls=%d", out, err, calls)
							}
							if in.State != nil || len(in.Docs) != 1 {
								t.Fatal("JSON state input was modified")
							}
						}
					})
					t.Run("invalid JSON state", func(t *testing.T) {
						for _, raw := range []string{`{`, `null`, `42`} {
							in := genai.SystemOneRequest{Questions: req.Questions, Docs: []genai.Doc{{Filename: "state.json", Src: strings.NewReader(raw)}}}
							before := calls
							if _, err := c.SystemOne(t.Context(), &in); err == nil || calls != before {
								t.Fatalf("invalid JSON state reached transport: %v", err)
							}
						}
					})
					t.Run("null answer", func(t *testing.T) {
						wantState = `"ticket"`
						answer = `null`
						if _, err := c.SystemOne(t.Context(), &req); err == nil {
							t.Fatal("accepted a null answer")
						}
					})
				})
			}
		})
	}
}

type decisionDocCase struct {
	name string
	doc  genai.Doc
}

func testSystemOneDocs(t *testing.T) {
	var pngData bytes.Buffer
	if err := png.Encode(&pngData, image.NewRGBA(image.Rect(0, 0, 1, 1))); err != nil {
		t.Fatal(err)
	}
	var jpegData bytes.Buffer
	if err := jpeg.Encode(&jpegData, image.NewRGBA(image.Rect(0, 0, 1, 1)), nil); err != nil {
		t.Fatal(err)
	}
	webpData, err := base64.StdEncoding.DecodeString("UklGRiQAAABXRUJQVlA4IBgAAAAwAQCdASoBAAEAAgA0JaQAA3AA/vuUAAA=")
	if err != nil {
		t.Fatal(err)
	}
	wantImages := []struct {
		mt   string
		data []byte
	}{{"image/png", pngData.Bytes()}, {"image/jpeg", jpegData.Bytes()}, {"image/webp", webpData}}
	for _, name := range []string{"cloudflare", "llamacpp", "typesafe"} {
		t.Run(name, func(t *testing.T) {
			calls := 0
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				calls++
				var body struct {
					Images []string        `json:"images"`
					Docs   json.RawMessage `json:"docs"`
				}
				if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
					t.Error(err)
				}
				if len(body.Images) != 3 || body.Docs != nil {
					t.Errorf("native images field changed: %+v", body)
				}
				for j, img := range body.Images {
					if j >= len(wantImages) {
						break
					}
					prefix, encoded, ok := strings.Cut(img, ",")
					if !ok || prefix != "data:"+wantImages[j].mt+";base64" {
						t.Errorf("invalid wire image %q", img)
					}
					data, err := base64.StdEncoding.DecodeString(encoded)
					if err != nil || !bytes.Equal(data, wantImages[j].data) {
						t.Errorf("image bytes changed: %v", err)
					}
				}
				response := `{"model":"decision","answers":{"yes":{"type":"noul","noul":0.9}},"usage":{"input_tokens":1,"output_tokens":0}}`
				if name == "cloudflare" {
					response = `{"success":true,"errors":[],"messages":[],"result":` + response + `}`
				}
				w.Header().Set("Content-Type", "application/json")
				if _, err := w.Write([]byte(response)); err != nil {
					t.Error(err)
				}
			}))
			t.Cleanup(srv.Close)
			var c genai.Provider
			var err error
			switch name {
			case "cloudflare":
				c, err = cloudflare.New(t.Context(), genai.ProviderOptionAPIKey("fake"), cloudflare.AccountID("account"), genai.ProviderOptionModel("@cf/cloudflare/clef"), genai.ProviderOptionTransportWrapper(func(http.RoundTripper) http.RoundTripper { return decisionTransport{srv: srv} }))
			case "llamacpp":
				c, err = llamacpp.New(t.Context(), genai.ProviderOptionRemote(srv.URL))
			case "typesafe":
				c, err = typesafe.New(t.Context(), genai.ProviderOptionAPIKey("fake"), genai.ProviderOptionModel("jev-latest"), genai.ProviderOptionRemote(srv.URL))
			}
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() {
				if err := c.Close(); err != nil {
					t.Error(err)
				}
			})
			qs := genai.Questions{"yes": {Type: genai.QuestionNoul, Instructions: genai.Text("yes?")}}
			limit := 10 * 1024 * 1024
			if name == "cloudflare" {
				limit = 4 * 1024 * 1024
			}
			invalid := []decisionDocCase{
				{name: "empty"},
				{name: "missing source", doc: genai.Doc{Filename: "image.png"}},
				{name: "remote", doc: genai.Doc{URL: "https://example.com/image.png"}},
				{name: "wrong type", doc: genai.Doc{Filename: "state.json", Src: strings.NewReader(`{}`)}},
				{name: "PDF", doc: genai.Doc{Filename: "report.pdf", Src: strings.NewReader("%PDF-1.7")}},
				{name: "audio", doc: genai.Doc{Filename: "clip.wav", Src: strings.NewReader("RIFF")}},
				{name: "unknown", doc: genai.Doc{Filename: "attachment.custom", Src: strings.NewReader("data")}},
				{name: "extensionless", doc: genai.Doc{Filename: "attachment", Src: strings.NewReader("data")}},
				{name: "empty data", doc: genai.Doc{Filename: "image.png", Src: strings.NewReader("")}},
				{name: "oversized", doc: genai.Doc{Filename: "image.png", Src: bytes.NewReader(make([]byte, limit+1))}},
			}
			if name != "llamacpp" {
				invalid = append(invalid, decisionDocCase{name: "GIF", doc: genai.Doc{Filename: "image.gif", Src: strings.NewReader("GIF87a")}})
			}
			for _, tc := range invalid {
				t.Run(tc.name, func(t *testing.T) {
					in := genai.SystemOneRequest{State: genai.Text("state"), Questions: qs, Docs: []genai.Doc{tc.doc}}
					if out, err := c.SystemOne(t.Context(), &in); out != nil || err == nil {
						t.Fatalf("invalid document returned %v, %v", out, err)
					}
					if calls != 0 {
						t.Fatal("invalid document reached HTTP")
					}
					if tc.name == "PDF" || tc.name == "audio" || tc.name == "unknown" || tc.name == "extensionless" || tc.name == "GIF" || tc.name == "wrong type" {
						pos, err := tc.doc.Src.Seek(0, io.SeekCurrent)
						if err != nil || pos != 0 {
							t.Fatalf("unsupported document was read: position=%d, err=%v", pos, err)
						}
					}
				})
			}
			t.Run("mixed image and unsupported attachment", func(t *testing.T) {
				first := bytes.NewReader(pngData.Bytes())
				in := genai.SystemOneRequest{State: genai.Text("state"), Questions: qs, Docs: []genai.Doc{{Filename: "image.png", Src: first}, {Filename: "report.pdf", Src: strings.NewReader("%PDF-1.7")}}}
				if out, err := c.SystemOne(t.Context(), &in); out != nil || err == nil {
					t.Fatalf("mixed attachments accepted: %v, %v", out, err)
				}
				pos, err := first.Seek(0, io.SeekCurrent)
				if err != nil || pos != 0 || calls != 0 {
					t.Fatalf("earlier image consumed: position=%d, calls=%d, err=%v", pos, calls, err)
				}
			})
			f, err := os.CreateTemp(t.TempDir(), "*.jpg")
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() {
				if err := f.Close(); err != nil {
					t.Error(err)
				}
			})
			if _, err := f.Write(jpegData.Bytes()); err != nil {
				t.Fatal(err)
			}
			r := bytes.NewReader(pngData.Bytes())
			in := genai.SystemOneRequest{State: genai.Text("state"), Questions: qs, Docs: []genai.Doc{{Filename: "image.png", Src: r}, {Src: f}, {Filename: "image.webp", Src: bytes.NewReader(webpData)}}}
			for range 2 {
				out, err := c.SystemOne(t.Context(), &in)
				if name == "typesafe" {
					if out != nil || err == nil || calls != 0 {
						t.Fatalf("TypeSafe accepted images: %v, %v", out, err)
					}
				} else if err != nil {
					t.Fatal(err)
				}
				if in.Docs[0].Src != r || in.Docs[1].Src != f || in.Docs[1].Filename != "" {
					t.Fatal("image document was mutated")
				}
			}
			if name != "typesafe" && calls != 2 {
				t.Fatalf("calls=%d", calls)
			}
		})
	}
}

// decisionTransport redirects the hosted endpoint to a local test server.
type decisionTransport struct{ srv *httptest.Server }

func (d decisionTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	r := req.Clone(req.Context())
	r.URL.Scheme = "http"
	r.URL.Host = d.srv.Listener.Addr().String()
	return d.srv.Client().Transport.RoundTrip(r)
}
