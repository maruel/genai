// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Tests for embedding qualification and failed probes.

package smoke_test

import (
	"bytes"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"net/http"
	"net/http/httptest"
	"net/url"
	"slices"
	"strings"
	"testing"

	"github.com/maruel/genai"
	"github.com/maruel/genai/providers/gemini"
	"github.com/maruel/genai/providers/ollama"
	"github.com/maruel/genai/scoreboard"
	"github.com/maruel/genai/smoke"
)

func TestRunEmbeddings(t *testing.T) {
	for _, tc := range []embeddingProbeCase{
		{name: "routing", probe: 32, supported: new(true), reporting: scoreboard.True},
		{name: "valid", probe: 32, supported: new(true), reporting: scoreboard.True},
		{name: "no size probe", reporting: scoreboard.True},
		{name: "unreported usage", defect: "usage", probe: 32, supported: new(true)},
		{name: "flaky usage", defect: "flaky usage", reporting: scoreboard.Flaky},
		{name: "unsupported dimensions", probe: 32, status: 400, body: `{"error":{"code":"unsupported_parameter","param":"dimensions"}}`, supported: new(false), reporting: scoreboard.True},
		{name: "unauthorized", probe: 32, status: 401, wantErr: true},
		{name: "quota", probe: 32, status: 429, wantErr: true},
		{name: "server failure", probe: 32, status: 500, wantErr: true},
		{name: "generic bad request", probe: 32, status: 400, body: `{"error":{"message":"invalid input","type":"invalid_request_error","param":null,"code":null}}`, wantErr: true},
		{name: "wrong unsupported option", probe: 32, status: 400, body: `{"error":{"code":"unsupported_parameter","param":"model"}}`, wantErr: true},
		{name: "wrong dimensions", probe: 32, defect: "dimensions", wantErr: true},
		{name: "wrong order", defect: "order", wantErr: true},
		{name: "bad retrieval", defect: "retrieval", wantErr: true},
		{name: "missing vector", defect: "missing", wantErr: true},
		{name: "zero vector", defect: "zero", wantErr: true},
		{name: "negative usage", defect: "negative usage", wantErr: true},
		{name: "bad default batch", defect: "initial", wantErr: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			calls := 0
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				calls++
				if r.URL.Path != "/v1/embeddings" {
					t.Errorf("generation was invoked: %s", r.URL.Path)
				}
				var req embeddingProbeRequest
				if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
					t.Error(err)
				}
				if req.EncodingFormat != "base64" {
					t.Errorf("encoding format %q", req.EncodingFormat)
				}
				w.Header().Set("Content-Type", "application/json")
				if tc.defect == "initial" || req.Dimensions != 0 && tc.status != 0 {
					status := tc.status
					if status == 0 {
						status = 500
					}
					w.WriteHeader(status)
					body := tc.body
					if body == "" {
						body = `{"error":{"message":"failed","type":"api_error","param":null,"code":null}}`
					}
					if _, err := w.Write([]byte(body)); err != nil {
						t.Error(err)
					}
					return
				}
				n := req.Dimensions
				if n == 0 || tc.defect == "dimensions" {
					n = 2
				}
				out := genai.EmbeddingResponse{Usage: genai.Usage{InputTokens: 7}}
				for _, s := range req.Input {
					v := make([]float32, n)
					switch {
					case strings.Contains(s, "query:"):
						v[0] = 2
					case strings.Contains(s, "charged particles"):
						v[0] = 3
						v[1] = 1
					default:
						v[0] = -1
						v[1] = 2
					}
					if tc.defect == "retrieval" && strings.Contains(s, "charged particles") {
						v[0] = -3
					}
					if tc.defect == "zero" {
						clear(v)
					}
					out.Embeddings = append(out.Embeddings, v)
				}
				if tc.defect == "order" {
					slices.Reverse(out.Embeddings)
				}
				if tc.defect == "missing" {
					out.Embeddings = out.Embeddings[:2]
				}
				if tc.defect == "usage" || tc.defect == "flaky usage" && calls == 2 {
					out.Usage.InputTokens = 0
				}
				if tc.defect == "negative usage" {
					out.Usage.InputTokens = -1
				}
				wire := embeddingProbeResponse{Object: "list", Model: "model", Usage: embeddingProbeUsage{PromptTokens: out.Usage.InputTokens, TotalTokens: out.Usage.InputTokens}}
				for i, v := range out.Embeddings {
					b := make([]byte, len(v)*4)
					for j, x := range v {
						binary.LittleEndian.PutUint32(b[j*4:], math.Float32bits(x))
					}
					wire.Data = append(wire.Data, embeddingProbeVector{Object: "embedding", Index: i, Embedding: base64.StdEncoding.EncodeToString(b)})
				}
				if err := json.NewEncoder(w).Encode(&wire); err != nil {
					t.Error(err)
				}
			}))
			t.Cleanup(srv.Close)
			model := "model"
			if tc.name == "routing" {
				model = "all-minilm"
			}
			c, err := ollama.New(t.Context(), genai.ProviderOptionRemote(srv.URL), genai.ProviderOptionModel(model), genai.ProviderOptionTransportWrapper(func(http.RoundTripper) http.RoundTripper { return http.DefaultTransport }))
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() {
				if err := c.Close(); err != nil {
					t.Error(err)
				}
			})
			var sc scoreboard.Scenario
			var u genai.Usage
			if tc.name == "routing" {
				sc, u, err = smoke.Run(t.Context(), func(string) genai.Provider { return c })
			} else {
				sc, u, err = smoke.RunEmbeddings(t.Context(), func(string) genai.Provider { return c }, tc.probe)
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
			if tc.probe == 0 && calls != 3 || tc.probe > 0 && calls != 4 {
				t.Fatalf("requests %d", calls)
			}
			if tc.reporting == scoreboard.True && (u.InputTokens < 14 || u.TotalTokens != u.InputTokens) {
				t.Fatalf("usage %+v", u)
			}
		})
	}
	t.Run("Media", func(t *testing.T) {
		cases := []embeddingMediaCase{
			{name: "supported"},
			{name: "text required", status: 400, body: `{"error":{"code":400,"message":"The text content is empty.","status":"INVALID_ARGUMENT"}}`, unsupported: true},
			{name: "projector required", status: 500, projector: true, unsupported: true},
			{name: "unauthorized", status: 401, body: `{"error":{"code":401,"message":"The text content is empty.","status":"INVALID_ARGUMENT"}}`, wantErr: true},
			{name: "quota", status: 429, body: `{"error":{"code":429,"message":"quota exhausted","status":"RESOURCE_EXHAUSTED"}}`, wantErr: true},
			{name: "unrelated empty text", status: 400, body: `{"error":{"code":400,"message":"empty text","status":"INVALID_ARGUMENT"}}`, wantErr: true},
			{name: "wrong discriminator", status: 400, body: `{"error":{"code":400,"message":"The text content is empty.","status":"INTERNAL"}}`, wantErr: true},
			{name: "server failure", status: 500, body: `{"error":{"code":500,"message":"server failed","status":"INTERNAL"}}`, wantErr: true},
			{name: "transport", transport: true, wantErr: true},
			{name: "missing", defect: "missing", wantErr: true},
			{name: "extra", defect: "extra", wantErr: true},
			{name: "zero", defect: "zero", wantErr: true},
			{name: "malformed vector", defect: "malformed", wantErr: true},
			{name: "width", defect: "width", wantErr: true},
			{name: "usage", defect: "usage", wantErr: true},
			{name: "audio failure", status: 500, body: `{"error":{"code":500,"message":"server failed","status":"INTERNAL"}}`, audioOnly: true, wantErr: true},
		}
		for _, group := range []string{"Valid", "Error"} {
			t.Run(group, func(t *testing.T) {
				for _, tc := range cases {
					if tc.wantErr != (group == "Error") {
						continue
					}
					t.Run(tc.name, func(t *testing.T) {
						mediaCalls := 0
						srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
							var in gemini.BatchEmbeddingRequest
							if err := json.NewDecoder(r.Body).Decode(&in); err != nil {
								t.Error(err)
							}
							out := gemini.BatchEmbeddingResponse{UsageMetadata: gemini.EmbeddingUsageMetadata{PromptTokenCount: 7}}
							for _, req := range in.Requests {
								part := req.Content.Parts[0]
								v := []float32{-1, 2}
								switch {
								case part.InlineData.MimeType != "":
									mediaCalls++
									v = []float32{1, 2}
									filename := "image.jpg"
									modality := "image"
									if part.InlineData.MimeType == "audio/wav" {
										filename = "audio.wav"
										modality = "audio"
									}
									data, err := scoreboard.TestdataFiles.ReadFile("testdata/" + filename)
									if err != nil {
										t.Error(err)
										return
									}
									if len(in.Requests) != 1 || len(req.Content.Parts) != 1 || part.Text != "" || !bytes.Equal(part.InlineData.Data, data) {
										t.Error("probe must contain the complete standalone media input")
									}
									if !tc.audioOnly || modality == "audio" {
										if tc.status != 0 {
											body := tc.body
											if tc.projector {
												body = fmt.Sprintf(`{"error":{"code":500,"message":"%s input is not supported - hint: if this is unexpected, you may need to provide the mmproj","type":"server_error"}}`, modality)
											}
											w.Header().Set("Content-Type", "application/json")
											w.WriteHeader(tc.status)
											if _, err := w.Write([]byte(body)); err != nil {
												t.Error(err)
											}
											return
										}
										if tc.defect == "malformed" {
											w.Header().Set("Content-Type", "application/json")
											if _, err := w.Write([]byte(`{"embeddings":[{"values":[1e100,1]}]}`)); err != nil {
												t.Error(err)
											}
											return
										}
										switch tc.defect {
										case "missing":
											out.Embeddings = nil
										case "extra":
											out.Embeddings = []gemini.Embedding{{Values: []float32{1, 2}}}
										case "zero":
											v = []float32{0, 0}
										case "width":
											v = []float32{1, 2, 3}
										case "usage":
											out.UsageMetadata.PromptTokenCount = -1
										}
									}
									if tc.defect == "missing" {
										continue
									}
								case strings.Contains(part.Text, "query:"):
									v = []float32{2, 0}
								case strings.Contains(part.Text, "charged particles"):
									v = []float32{3, 1}
								}
								out.Embeddings = append(out.Embeddings, gemini.Embedding{Values: v})
							}
							w.Header().Set("Content-Type", "application/json")
							if err := json.NewEncoder(w).Encode(&out); err != nil {
								t.Error(err)
							}
						}))
						t.Cleanup(srv.Close)
						tr := &embeddingMediaTransport{url: srv.URL, fail: tc.transport}
						c, err := gemini.New(t.Context(), genai.ProviderOptionModel("gemini-embedding-2"), genai.ProviderOptionTransportWrapper(func(http.RoundTripper) http.RoundTripper { return tr }))
						if err != nil {
							t.Fatal(err)
						}
						t.Cleanup(func() {
							if err := c.Close(); err != nil {
								t.Error(err)
							}
						})
						sc, u, err := smoke.RunEmbeddings(t.Context(), func(string) genai.Provider { return c }, 0)
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
						if mediaCalls != 2 {
							t.Fatalf("media calls %d", mediaCalls)
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
	status    int
	body      string
	defect    string
	wantErr   bool
	supported *bool
	reporting scoreboard.TriState
}

type embeddingProbeRequest struct {
	Input          []string `json:"input"`
	Dimensions     int      `json:"dimensions"`
	EncodingFormat string   `json:"encoding_format"`
}

type embeddingProbeResponse struct {
	Object string                 `json:"object"`
	Model  string                 `json:"model"`
	Data   []embeddingProbeVector `json:"data"`
	Usage  embeddingProbeUsage    `json:"usage"`
}

type embeddingProbeVector struct {
	Object    string `json:"object"`
	Index     int    `json:"index"`
	Embedding string `json:"embedding"`
}

type embeddingProbeUsage struct {
	PromptTokens int64 `json:"prompt_tokens"`
	TotalTokens  int64 `json:"total_tokens"`
}

type embeddingMediaCase struct {
	name, body, defect                                    string
	status                                                int
	unsupported, wantErr, projector, transport, audioOnly bool
}

type embeddingMediaTransport struct {
	url  string
	fail bool
}

func (tr *embeddingMediaTransport) RoundTrip(r *http.Request) (*http.Response, error) {
	b, err := io.ReadAll(r.Body)
	if err != nil {
		return nil, err
	}
	if tr.fail && bytes.Contains(b, []byte("inlineData")) {
		return nil, errors.New("media transport failed")
	}
	r.Body = io.NopCloser(bytes.NewReader(b))
	u, err := url.Parse(tr.url)
	if err != nil {
		return nil, err
	}
	r.URL.Scheme = u.Scheme
	r.URL.Host = u.Host
	return http.DefaultTransport.RoundTrip(r)
}
