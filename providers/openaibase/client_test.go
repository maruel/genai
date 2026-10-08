// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Tests for the shared OpenAI-compatible client operations.

package openaibase

import (
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"reflect"
	"strconv"
	"strings"
	"testing"

	"github.com/maruel/genai"
	"github.com/maruel/genai/base"
	"github.com/maruel/genai/internal"
)

func TestErrorResponse(t *testing.T) {
	var got ErrorResponse
	if err := internal.UnmarshalJSON([]byte(`{"error":{"type":"resource_unavailable","code":"flex_unavailable","headers":{"retry-after":"300","x-retry-metadata":"NO_MORE_RETRY"},"message":"Try standard processing.","param":null}}`), &got); err != nil {
		t.Fatal(err)
	}
	if got.ErrorVal.Headers["retry-after"] != "300" || got.ErrorVal.Headers["x-retry-metadata"] != "NO_MORE_RETRY" {
		t.Fatalf("headers = %+v", got.ErrorVal.Headers)
	}
}

func TestClient(t *testing.T) {
	t.Run("Embed/valid/reordered results", func(t *testing.T) {
		body := `{"object":"list","data":[{"object":"embedding","index":1,"embedding":"AAAAAAAAgEA="},{"object":"embedding","index":0,"embedding":"AAAAQAAAQMA="}],"usage":{"prompt_tokens":7,"total_tokens":7}}`
		c := &Client{Impl: &base.ProviderBase[*ErrorResponse]{Model: "embedding-test", Client: http.Client{Transport: embeddingResponseTransport{body: body}}}, BaseURL: "http://localhost/v1"}
		out, err := c.Embed(t.Context(), &genai.EmbeddingRequest{Inputs: []genai.Request{{Text: "a"}, {Text: "b"}}})
		if err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(out.Embeddings, [][]float32{{2, -3}, {0, 4}}) || out.Usage.InputTokens != 7 || out.Usage.TotalTokens != 7 {
			t.Fatalf("response %+v", out)
		}
	})

	t.Run("EmbedRaw/error", func(t *testing.T) {
		t.Run("input", func(t *testing.T) {
			c := &Client{Impl: &base.ProviderBase[*ErrorResponse]{Model: "embedding-test", Client: http.Client{Transport: embeddingNoTransport{t: t}}}, BaseURL: "http://localhost/v1"}
			var out EmbeddingResponse
			if err := c.EmbedRaw(t.Context(), nil, &out); err == nil {
				t.Fatal("expected nil request error")
			}
			in := EmbeddingRequest{Model: "embedding-test", Input: EmbeddingInput{Texts: []string{"hello"}}}
			if err := c.EmbedRaw(t.Context(), &in, nil); err == nil {
				t.Fatal("expected nil response error")
			}
			if err := c.EmbedRaw(t.Context(), &EmbeddingRequest{}, &out); err == nil {
				t.Fatal("expected invalid request error")
			}
		})
		for _, lenient := range []bool{false, true} {
			t.Run(strconv.FormatBool(lenient), func(t *testing.T) {
				c := &Client{Impl: &base.ProviderBase[*ErrorResponse]{Model: "embedding-test", Lenient: lenient, Client: http.Client{Transport: embeddingResponseTransport{body: `{"data":[{"index":0,"embedding":"AACAPw=="},{"index":1,"embedding":"AA=="}]}`}}}, BaseURL: "http://localhost/v1"}
				var out EmbeddingResponse
				err := c.EmbedRaw(t.Context(), &EmbeddingRequest{Model: "embedding-test", Input: EmbeddingInput{Texts: []string{"a", "b"}}}, &out)
				if _, ok := errors.AsType[*internal.BadError](err); !ok || !reflect.ValueOf(out).IsZero() {
					t.Fatalf("response %+v, error %v", out, err)
				}
			})
		}
	})

	t.Run("Embed/input error", func(t *testing.T) {
		c := &Client{Impl: &base.ProviderBase[*ErrorResponse]{Model: "embedding-test", Client: http.Client{Transport: embeddingNoTransport{t: t}}}, BaseURL: "http://localhost/v1"}
		if out, err := c.Embed(t.Context(), nil); out != nil || err == nil || !strings.Contains(err.Error(), "required") {
			t.Fatalf("response %+v, error %v", out, err)
		}
	})

	t.Run("Embed/error", func(t *testing.T) {
		for _, tc := range []embeddingResponseCase{{"result count", `{"data":[]}`},
			{"duplicate index", `{"data":[{"index":0,"embedding":"AACAPwAAAEA="},{"index":0,"embedding":"AACAPwAAAEA="}]}`},
			{"invalid vector", `{"data":[{"index":0,"embedding":"AAAAAAAAAAA="},{"index":1,"embedding":"AACAPwAAAEA="}]}`},
			{"dimensions", `{"data":[{"index":0,"embedding":"AACAPwAAAEAAAEBA"},{"index":1,"embedding":"AACAPwAAAEAAAEBA"}]}`},
			{"decode", `{"data":[{"index":0,"embedding":"AACAPw=="},{"index":1,"embedding":"AA=="}]}`}} {
			t.Run(tc.name, func(t *testing.T) {
				c := &Client{Impl: &base.ProviderBase[*ErrorResponse]{Model: "embedding-test", Client: http.Client{Transport: embeddingResponseTransport{body: tc.body}}}, BaseURL: "http://localhost/v1"}
				out, err := c.Embed(t.Context(), &genai.EmbeddingRequest{Inputs: []genai.Request{{Text: "a"}, {Text: "b"}}, Dimensions: 2})
				if _, ok := errors.AsType[*internal.BadError](err); !ok || out != nil {
					t.Fatalf("response %+v, error %v", out, err)
				}
			})
		}
	})

	t.Run("SelectBestTextModel", func(t *testing.T) {
		models := []genai.Model{
			&Model{ID: "gpt-6-astra", Created: base.TimeS(600)},
			&Model{ID: "gpt-6.1-sol", Created: base.TimeS(700)},
			&Model{ID: "gpt-5.6", Created: base.TimeS(300)},
			&Model{ID: "gpt-5.6-sol", Created: base.TimeS(300)},
			&Model{ID: "gpt-5.6-terra", Created: base.TimeS(300)},
			&Model{ID: "gpt-5.6-luna", Created: base.TimeS(300)},
			&Model{ID: "gpt-5.4-mini", Created: base.TimeS(200)},
			&Model{ID: "gpt-5.4-nano", Created: base.TimeS(200)},
			&Model{ID: "gpt-5.6-sol-2026-07-09", Created: base.TimeS(400)},
			&Model{ID: "gpt-5.6-pro", Created: base.TimeS(400)},
			&Model{ID: "gpt-5.6-codex", Created: base.TimeS(400)},
			&Model{ID: "gpt-transcribe", Created: base.TimeS(500)},
			&Model{ID: "gpt-live-transcribe", Created: base.TimeS(500)},
			&Model{ID: "gpt-live-1", Created: base.TimeS(600)},
		}
		c := &Client{PreloadedModels: models}
		data := []struct {
			name string
			in   genai.ProviderOptionModel
			want string
		}{
			{name: "sota", in: genai.ModelSOTA, want: "gpt-6-astra"},
			{name: "good", in: genai.ModelGood, want: "gpt-6.1-sol"},
			{name: "cheap", in: genai.ModelCheap, want: "gpt-5.6-luna"},
			{name: "default", in: "", want: "gpt-6.1-sol"},
		}
		for _, tc := range data {
			t.Run(tc.name, func(t *testing.T) {
				got, err := c.SelectBestTextModel(t.Context(), string(tc.in))
				if err != nil {
					t.Fatal(err)
				}
				if got != tc.want {
					t.Fatalf("got %q, want %q", got, tc.want)
				}
			})
		}
	})
	t.Run("SelectBestImageModel", func(t *testing.T) {
		models := []genai.Model{
			&Model{ID: "gpt-image-1-mini", Created: base.TimeS(200)},
			&Model{ID: "gpt-image-2", Created: base.TimeS(300)},
			&Model{ID: "gpt-image-2-2026-04-21", Created: base.TimeS(400)},
			&Model{ID: "gpt-image-2.5-sunburst", Created: base.TimeS(500)},
		}
		c := &Client{PreloadedModels: models}
		data := []struct {
			name string
			in   genai.ProviderOptionModel
			want string
		}{
			{name: "sota", in: genai.ModelSOTA, want: "gpt-image-2.5-sunburst"},
			{name: "good", in: genai.ModelGood, want: "gpt-image-2.5-sunburst"},
			{name: "cheap", in: genai.ModelCheap, want: "gpt-image-1-mini"},
		}
		for _, tc := range data {
			t.Run(tc.name, func(t *testing.T) {
				got, err := c.SelectBestImageModel(t.Context(), string(tc.in))
				if err != nil {
					t.Fatal(err)
				}
				if got != tc.want {
					t.Fatalf("got %q, want %q", got, tc.want)
				}
			})
		}
	})
}

func TestImageResponse(t *testing.T) {
	var got ImageResponse
	if err := json.Unmarshal([]byte(`{"data":[{"generation_id":"imggen_123"}]}`), &got); err != nil {
		t.Fatal(err)
	}
	if len(got.Data) != 1 {
		t.Fatalf("got %d images, want 1", len(got.Data))
	}
	if got.Data[0].GenerationID != "imggen_123" {
		t.Fatalf("got generation ID %q, want %q", got.Data[0].GenerationID, "imggen_123")
	}
}

type embeddingResponseTransport struct{ body string }

func (e embeddingResponseTransport) RoundTrip(r *http.Request) (*http.Response, error) {
	return &http.Response{StatusCode: http.StatusOK, Header: http.Header{"Content-Type": {"application/json"}}, Body: io.NopCloser(strings.NewReader(e.body)), Request: r}, nil
}

type embeddingResponseCase struct{ name, body string }

type embeddingNoTransport struct{ t *testing.T }

func (e embeddingNoTransport) RoundTrip(*http.Request) (*http.Response, error) {
	e.t.Fatal("unexpected HTTP request")
	return nil, errors.New("unexpected HTTP request")
}
