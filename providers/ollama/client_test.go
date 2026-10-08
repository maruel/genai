// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Tests for the Ollama provider client.

package ollama_test

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"iter"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"os"
	"reflect"
	"slices"
	"strconv"
	"strings"
	"sync"
	"testing"

	"github.com/maruel/httpjson"
	"github.com/maruel/roundtrippers"

	"github.com/maruel/genai"
	"github.com/maruel/genai/internal"
	"github.com/maruel/genai/internal/internaltest"
	"github.com/maruel/genai/providers/ollama"
	"github.com/maruel/genai/scoreboard"
	"github.com/maruel/genai/smoke/smoketest"
)

// Not implementing TestClient_AllModels since we need to preload Ollama models. Can be done later.

func TestClient(t *testing.T) {
	t.Run("EmbedRaw/error/partial response", func(t *testing.T) {
		old := internal.BeLenient
		t.Cleanup(func() { internal.BeLenient = old })
		for _, lenient := range []bool{false, true} {
			t.Run(strconv.FormatBool(lenient), func(t *testing.T) {
				internal.BeLenient = lenient
				c, err := ollama.New(t.Context(), genai.ProviderOptionModel("model"), genai.ProviderOptionTransportWrapper(func(http.RoundTripper) http.RoundTripper {
					return embeddingResponseTransport{body: `{"object":"list","data":[{"object":"embedding","index":0,"embedding":"AACAPw=="},{"object":"embedding","index":1,"embedding":"AA=="}]}`}
				}))
				if err != nil {
					t.Fatal(err)
				}
				internaltest.CleanupCloser(t, c)
				var out ollama.EmbeddingResponse
				err = c.EmbedRaw(t.Context(), &ollama.EmbeddingRequest{Model: "model", Input: []string{"a", "b"}}, &out)
				if _, ok := errors.AsType[*internal.BadError](err); !ok || !reflect.ValueOf(out).IsZero() {
					t.Fatalf("response %+v, error %v", out, err)
				}
			})
		}
	})

	t.Run("Embed/valid/reordered results", func(t *testing.T) {
		body := `{"object":"list","data":[{"object":"embedding","index":1,"embedding":"AAAAAAAAgEA="},{"object":"embedding","index":0,"embedding":"AAAAQAAAQMA="}],"usage":{"prompt_tokens":7,"total_tokens":7}}`
		c, err := ollama.New(t.Context(), genai.ProviderOptionModel("embedding-test"), genai.ProviderOptionTransportWrapper(func(http.RoundTripper) http.RoundTripper { return embeddingResponseTransport{body: body} }))
		if err != nil {
			t.Fatal(err)
		}
		internaltest.CleanupCloser(t, c)
		out, err := c.Embed(t.Context(), &genai.EmbeddingRequest{Inputs: []genai.Request{{Text: "a"}, {Text: "b"}}})
		if err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(out.Embeddings, [][]float32{{2, -3}, {0, 4}}) || out.Usage.InputTokens != 7 || out.Usage.TotalTokens != 7 {
			t.Fatalf("response %+v", out)
		}
	})

	t.Run("Embed/input error", func(t *testing.T) {
		c, err := ollama.New(t.Context(), genai.ProviderOptionModel("embedding-test"), genai.ProviderOptionModalities{genai.ModalityEmbedding}, genai.ProviderOptionTransportWrapper(func(http.RoundTripper) http.RoundTripper { return embeddingNoTransport{t: t} }))
		if err != nil {
			t.Fatal(err)
		}
		internaltest.CleanupCloser(t, c)
		if out, err := c.Embed(t.Context(), nil); out != nil || err == nil || !strings.Contains(err.Error(), "required") {
			t.Fatalf("response %+v, error %v", out, err)
		}
	})

	t.Run("Embed/error", func(t *testing.T) {
		for _, tc := range []embeddingResponseCase{{"result count", `{"object":"list","data":[]}`},
			{"duplicate index", `{"object":"list","data":[{"object":"embedding","index":0,"embedding":"AACAPwAAAEA="},{"object":"embedding","index":0,"embedding":"AACAPwAAAEA="}]}`},
			{"invalid vector", `{"object":"list","data":[{"object":"embedding","index":0,"embedding":"AAAAAAAAAAA="},{"object":"embedding","index":1,"embedding":"AACAPwAAAEA="}]}`},
			{"dimensions", `{"object":"list","data":[{"object":"embedding","index":0,"embedding":"AACAPwAAAEAAAEBA"},{"object":"embedding","index":1,"embedding":"AACAPwAAAEAAAEBA"}]}`},
			{"decode", `{"object":"list","data":[{"object":"embedding","index":0,"embedding":"AACAPw=="},{"object":"embedding","index":1,"embedding":"AA=="}]}`}} {
			t.Run(tc.name, func(t *testing.T) {
				c, err := ollama.New(t.Context(), genai.ProviderOptionModel("embedding-test"), genai.ProviderOptionModalities{genai.ModalityEmbedding}, genai.ProviderOptionTransportWrapper(func(http.RoundTripper) http.RoundTripper { return embeddingResponseTransport{body: tc.body} }))
				if err != nil {
					t.Fatal(err)
				}
				internaltest.CleanupCloser(t, c)
				out, err := c.Embed(t.Context(), &genai.EmbeddingRequest{Inputs: []genai.Request{{Text: "a"}, {Text: "b"}}, Dimensions: 2})
				if _, ok := errors.AsType[*internal.BadError](err); !ok || out != nil {
					t.Fatalf("response %+v, error %v", out, err)
				}
			})
		}
	})

	t.Run("New/embedding preference/error", func(t *testing.T) {
		c, err := ollama.New(t.Context(), genai.ModelCheap, genai.ProviderOptionModalities{genai.ModalityEmbedding}, genai.ProviderOptionTransportWrapper(func(http.RoundTripper) http.RoundTripper { return embeddingNoTransport{t: t} }))
		if err == nil || c != nil {
			t.Fatalf("client %v, error %v", c, err)
		}
	})

	testRecorder := internaltest.NewRecords()
	t.Cleanup(func() {
		if err := testRecorder.Close(); err != nil {
			t.Error(err)
		}
	})

	s := lazyServer{t: t}

	t.Run("Embed", func(t *testing.T) {
		transport := testRecorder.Record(t, http.DefaultTransport)
		serverURL := "http://localhost:0"
		if transport.IsNewCassette() || os.Getenv("RECORD") != "" {
			serverURL = s.lazyStart(t)
		}
		c, err := ollama.New(t.Context(), genai.ProviderOptionRemote(serverURL), genai.ProviderOptionModel("all-minilm"), genai.ProviderOptionTransportWrapper(func(http.RoundTripper) http.RoundTripper { return transport }))
		if err != nil {
			t.Fatal(err)
		}
		internaltest.CleanupCloser(t, c)
		if err := c.PullModel(t.Context(), "all-minilm"); err != nil {
			t.Fatal(err)
		}
		in := genai.EmbeddingRequest{Inputs: []genai.Request{{Text: "A kitten plays with yarn."}, {Text: "A cat plays with string."}, {Text: "Quantum field theory predicts particle interactions."}}, Dimensions: 32}
		var embedder genai.Provider = c
		out, err := embedder.Embed(t.Context(), &in)
		if err != nil {
			t.Fatal(err)
		}
		if err := out.Validate(); err != nil {
			t.Fatal(err)
		}
		if out.Usage.InputTokens == 0 || out.Usage.TotalTokens != out.Usage.InputTokens {
			t.Fatalf("unexpected metadata: %+v", out)
		}
		internaltest.AssertEmbeddingRetrieval(t, out.Embeddings[0], out.Embeddings[1], out.Embeddings[2])
	})

	t.Run("Embed-error", func(t *testing.T) {
		transport := testRecorder.Record(t, http.DefaultTransport)
		serverURL := "http://localhost:0"
		if transport.IsNewCassette() || os.Getenv("RECORD") != "" {
			serverURL = s.lazyStart(t)
		}
		c, err := ollama.New(t.Context(), genai.ProviderOptionRemote(serverURL), genai.ProviderOptionModel("genai-nonexistent-embedding-test"), genai.ProviderOptionTransportWrapper(func(http.RoundTripper) http.RoundTripper { return transport }))
		if err != nil {
			t.Fatal(err)
		}
		internaltest.CleanupCloser(t, c)
		out, err := c.Embed(t.Context(), &genai.EmbeddingRequest{Inputs: []genai.Request{{Text: "hello"}}})
		er, ok := errors.AsType[*ollama.ErrorResponse](err)
		he, httpOK := errors.AsType[*httpjson.Error](err)
		if out != nil || !ok || !httpOK || he.StatusCode != http.StatusNotFound || er.Details == nil || er.Details.Type != "not_found_error" || !strings.Contains(er.Error(), "genai-nonexistent-embedding-test") {
			t.Fatalf("response %+v, API error %+v, HTTP error %+v: %v", out, er, he, err)
		}
	})

	t.Run("EmbedRaw", func(t *testing.T) {
		t.Run("valid", func(t *testing.T) {
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.Method != http.MethodPost || r.URL.Path != "/v1/embeddings" {
					t.Errorf("request %s %s", r.Method, r.URL.Path)
				}
				var req embeddingWireRequest
				if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
					t.Error(err)
				}
				if req.Model != "explicit-model" || req.Dimensions != 2 || req.EncodingFormat != "base64" || !slices.Equal(req.Input, []string{"a", "b"}) {
					t.Errorf("request %+v", req)
				}
				w.Header().Set("Content-Type", "application/json")
				if _, err := io.WriteString(w, `{"object":"list","model":"explicit-model","data":[{"object":"embedding","index":0,"embedding":"AACAPwAAAEA="},{"object":"embedding","index":1,"embedding":"AABAQAAAgEA="}],"usage":{"prompt_tokens":7,"total_tokens":7}}`); err != nil {
					t.Error(err)
				}
			}))
			t.Cleanup(srv.Close)
			c, err := ollama.New(t.Context(), genai.ProviderOptionRemote(srv.URL), genai.ProviderOptionModel("unused-generation-model"))
			if err != nil {
				t.Fatal(err)
			}
			internaltest.CleanupCloser(t, c)
			in := ollama.EmbeddingRequest{Model: "explicit-model", Input: []string{"a", "b"}, Dimensions: 2}
			var out ollama.EmbeddingResponse
			if err := c.EmbedRaw(t.Context(), &in, &out); err != nil {
				t.Fatal(err)
			}
			if len(out.Data) != 2 || !slices.Equal(out.Data[0].Embedding, []float32{1, 2}) || !slices.Equal(out.Data[1].Embedding, []float32{3, 4}) || out.Usage.PromptTokens != 7 || out.Usage.TotalTokens != 7 {
				t.Fatalf("response %+v", out)
			}
			if in.Model != "explicit-model" || in.Dimensions != 2 || !slices.Equal(in.Input, []string{"a", "b"}) {
				t.Fatalf("modified request %+v", in)
			}
		})
		t.Run("error", func(t *testing.T) {
			c, err := ollama.New(t.Context(), genai.ProviderOptionModel("unused"), genai.ProviderOptionTransportWrapper(func(http.RoundTripper) http.RoundTripper { return embeddingNoTransport{t: t} }))
			if err != nil {
				t.Fatal(err)
			}
			internaltest.CleanupCloser(t, c)
			var out ollama.EmbeddingResponse
			if err := c.EmbedRaw(t.Context(), nil, &out); err == nil {
				t.Fatal("accepted nil input")
			}
			if err := c.EmbedRaw(t.Context(), &ollama.EmbeddingRequest{Model: "model", Input: []string{"text"}}, nil); err == nil {
				t.Fatal("accepted nil output")
			}
			if err := c.EmbedRaw(t.Context(), &ollama.EmbeddingRequest{}, &out); err == nil {
				t.Fatal("accepted empty input")
			}
		})
	})

	t.Run("Capabilities", func(t *testing.T) {
		c, err := ollama.New(t.Context(), genai.ProviderOptionRemote(s.lazyStart(t)))
		if c != nil {
			internaltest.CleanupCloser(t, c)
		}
		if err != nil {
			t.Fatal(err)
		}
		internaltest.TestCapabilities(t, c)
	})

	t.Run("Scoreboard", func(t *testing.T) {
		scenarios := ollama.Scoreboard().Scenarios
		models := make([]scoreboard.Model, 0, len(scenarios))
		for _, sc := range scenarios {
			for _, m := range sc.Models {
				models = append(models, scoreboard.Model{Model: m, Reason: sc.Reason})
			}
		}
		if !slices.Contains(models, scoreboard.Model{Model: "all-minilm"}) {
			models = append(models, scoreboard.Model{Model: "all-minilm"})
		}
		smoketest.Run(t, func(t testing.TB, model scoreboard.Model, fn func(http.RoundTripper) http.RoundTripper) genai.Provider {
			ctx, l := internaltest.Log(t)
			fnWithLog := func(h http.RoundTripper) http.RoundTripper {
				if fn != nil {
					h = fn(h)
				}
				return &roundtrippers.Log{
					Transport: h,
					Logger:    l,
					Level:     slog.LevelDebug,
				}
			}
			opts := []genai.ProviderOption{genai.ProviderOptionRemote(s.lazyStart(t)), genai.ProviderOptionTransportWrapper(fnWithLog)}
			if model.Model != "" {
				opts = append(opts, genai.ProviderOptionModel(model.Model))
			}
			c, err := ollama.New(ctx, opts...)
			if c != nil {
				internaltest.CleanupCloser(t, c)
			}
			if err != nil {
				t.Fatal(err)
			}
			// Ollama v0.17.4+ auto-enables thinking for thinking-capable
			// models. Use the "think" API parameter to control it.
			if !model.Reason {
				return &ollamaThinkOff{Provider: c}
			}
			return c
		}, models, testRecorder.Records, &smoketest.RunOptions{Qualify: []scoreboard.Model{{Model: "all-minilm"}}})
	})

	// This test doesn't require the server to start.
	t.Run("Preferred", func(t *testing.T) {
		internaltest.TestPreferredModels(t, func(st *testing.T, model string, modality genai.Modality) (genai.Provider, error) {
			opts := []genai.ProviderOption{
				genai.ProviderOptionModalities{modality},
				genai.ProviderOptionRemote("http://localhost:66666"),
				genai.ProviderOptionTransportWrapper(func(h http.RoundTripper) http.RoundTripper {
					return testRecorder.Record(st, h)
				}),
			}
			if model != "" {
				opts = append(opts, genai.ProviderOptionModel(model))
			}
			c, err := ollama.New(st.Context(), opts...)
			if err == nil {
				internaltest.CleanupCloser(st, c)
			}
			return c, err
		})
	})

	t.Run("TextOutputDocInput", func(t *testing.T) {
		internaltest.TestTextOutputDocInput(t, func(t *testing.T) genai.Provider {
			opts := []genai.ProviderOption{
				genai.ProviderOptionRemote(s.lazyStart(t)),
				genai.ModelCheap,
				genai.ProviderOptionTransportWrapper(func(h http.RoundTripper) http.RoundTripper {
					return testRecorder.Record(t, h)
				}),
			}
			c, err := ollama.New(t.Context(), opts...)
			if c != nil {
				internaltest.CleanupCloser(t, c)
			}
			if err != nil {
				t.Fatal(err)
			}
			return c
		})
	})

	t.Run("errors", func(t *testing.T) {
		data := []internaltest.ProviderError{
			{
				Name: "bad model",
				Opts: []genai.ProviderOption{
					genai.ProviderOptionModel("bad_model"),
				},
				ErrGenSync:   "pull failed: http 500\npull model manifest: file does not exist",
				ErrGenStream: "pull failed: http 500\npull model manifest: file does not exist",
			},
		}
		f := func(t *testing.T, opts ...genai.ProviderOption) (genai.Provider, error) {
			serverURL := ""
			transport := testRecorder.Record(t, http.DefaultTransport)
			wrapper := func(h http.RoundTripper) http.RoundTripper { return transport }
			if !transport.IsNewCassette() {
				serverURL = "http://localhost:0"
			} else {
				name := "testdata/" + strings.ReplaceAll(t.Name(), "/", "_") + ".yaml"
				t.Cleanup(func() {
					if t.Failed() {
						t.Log("Removing record")
						_ = os.Remove(name)
					}
				})
				serverURL = s.lazyStart(t)
			}
			opts = append(opts, genai.ProviderOptionRemote(serverURL), genai.ProviderOptionTransportWrapper(wrapper))
			c, err := ollama.New(t.Context(), opts...)
			if c != nil {
				internaltest.CleanupCloser(t, c)
			}
			return c, err
		}
		internaltest.TestClientProviderErrors(t, f, data)
	})
}

type lazyServer struct {
	t   *testing.T
	mu  sync.Mutex
	url string
}

func (l *lazyServer) lazyStart(t testing.TB) string {
	// Skip server startup when not recording. The HTTP cassettes in testdata/
	// are replayed by the recording transport so the URL is never contacted.
	if os.Getenv("RECORD") == "" {
		return "http://localhost:0"
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	if l.url == "" {
		t.Log("Starting server")
		// Use the context of the parent for server lifecycle management.
		srv, err := startServer(l.t.Context())
		if err != nil {
			t.Fatal(err)
		}
		l.url = srv.URL()
		l.t.Cleanup(func() {
			if err := srv.Close(); err != nil && !errors.Is(err, context.Canceled) {
				l.t.Error(err)
			}
		})
	}
	return l.url
}

// ollamaThinkOff wraps a provider to inject ReasoningEffortOff on every call,
// disabling thinking for models that default to it.
type ollamaThinkOff struct {
	genai.Provider
}

func (o *ollamaThinkOff) GenSync(ctx context.Context, msgs genai.Messages, opts ...genai.GenOption) (genai.Result, error) {
	return o.Provider.GenSync(ctx, msgs, append(opts, &ollama.GenOptionText{ReasoningEffort: ollama.ReasoningEffortOff})...)
}

func (o *ollamaThinkOff) GenStream(ctx context.Context, msgs genai.Messages, opts ...genai.GenOption) (iter.Seq[genai.Reply], func() (genai.Result, error)) {
	return o.Provider.GenStream(ctx, msgs, append(opts, &ollama.GenOptionText{ReasoningEffort: ollama.ReasoningEffortOff})...)
}

func (o *ollamaThinkOff) Unwrap() genai.Provider {
	return o.Provider
}

func init() {
	internal.BeLenient = false
}

// TestEmbeddingSelection verifies model metadata selects embedding output for unqualified IDs and aliases.
func TestEmbeddingSelection(t *testing.T) {
	t.Run("mismatched modality", func(t *testing.T) {
		c, err := ollama.New(t.Context(), genai.ProviderOptionModel("new-embedding:latest"), genai.ProviderOptionPreloadedModels{&ollama.Model{Name: "new-embedding:latest", Capabilities: []string{"embedding"}}}, genai.ProviderOptionModalities{genai.ModalityText})
		if err == nil || c != nil {
			t.Fatalf("client %v, error %v", c, err)
		}
	})

	embedding := &ollama.Model{Name: "new-embedding:latest", Capabilities: []string{"embedding"}}
	generation := &ollama.Model{Name: "qwen3.5:2b"}
	for _, preference := range []genai.ProviderOptionModel{genai.ModelCheap, genai.ModelGood, genai.ModelSOTA} {
		t.Run(string(preference), func(t *testing.T) {
			c, err := ollama.New(t.Context(), preference, genai.ProviderOptionPreloadedModels{embedding, generation})
			if err != nil {
				t.Fatal(err)
			}
			internaltest.CleanupCloser(t, c)
			if c.ModelID() != generation.GetID() || !slices.Equal(c.OutputModalities(), genai.Modalities{genai.ModalityText}) {
				t.Fatalf("selected %s with modalities %s", c.ModelID(), c.OutputModalities())
			}
			for _, mod := range []genai.Modality{genai.ModalityText, genai.ModalityDecision} {
				empty, err := ollama.New(t.Context(), preference, genai.ProviderOptionModalities{mod}, genai.ProviderOptionTransportWrapper(func(http.RoundTripper) http.RoundTripper { return &emptyModelsTransport{t: t} }))
				if err != nil {
					t.Fatal(err)
				}
				internaltest.CleanupCloser(t, empty)
				only, err := ollama.New(t.Context(), preference, genai.ProviderOptionModalities{mod}, genai.ProviderOptionPreloadedModels{embedding})
				if err != nil {
					t.Fatal(err)
				}
				internaltest.CleanupCloser(t, only)
				if only.ModelID() != empty.ModelID() || !slices.Equal(only.OutputModalities(), genai.Modalities{mod}) {
					t.Fatalf("embedding-only fallback %s (%s); empty fallback %s", only.ModelID(), only.OutputModalities(), empty.ModelID())
				}
			}
		})
	}
	t.Run("completion and embedding", func(t *testing.T) {
		m := &ollama.Model{Name: "new-multitask:latest", Capabilities: []string{"completion", "embedding"}}
		c, err := ollama.New(t.Context(), genai.ModelGood, genai.ProviderOptionPreloadedModels{m})
		if err != nil {
			t.Fatal(err)
		}
		internaltest.CleanupCloser(t, c)
		if c.ModelID() != m.GetID() || !slices.Equal(c.OutputModalities(), genai.Modalities{genai.ModalityText}) {
			t.Fatalf("selected %s with modalities %s", c.ModelID(), c.OutputModalities())
		}
	})

	for _, tc := range []embeddingSelectionCase{{"new-embedding", true}, {"new-embedding:latest", true}, {"new-embedding:q4", false}} {
		t.Run(tc.id, func(t *testing.T) {
			c, err := ollama.New(t.Context(), genai.ProviderOptionModel(tc.id), genai.ProviderOptionPreloadedModels{embedding})
			if err != nil {
				t.Fatal(err)
			}
			internaltest.CleanupCloser(t, c)
			want := genai.ModalityText
			if tc.embedding {
				want = genai.ModalityEmbedding
			}
			if !slices.Equal(c.OutputModalities(), genai.Modalities{want}) {
				t.Fatalf("modalities %s; want %s", c.OutputModalities(), want)
			}
		})
	}
}

// emptyModelsTransport represents the native /api/tags response without installed models.
type emptyModelsTransport struct{ t *testing.T }

func (e *emptyModelsTransport) RoundTrip(r *http.Request) (*http.Response, error) {
	if r.Method != http.MethodGet || r.URL.Path != "/api/tags" {
		e.t.Fatalf("unexpected request %s %s", r.Method, r.URL.Path)
	}
	return &http.Response{StatusCode: http.StatusOK, Header: http.Header{"Content-Type": {"application/json"}}, Body: io.NopCloser(strings.NewReader(`{"models":[]}`)), Request: r}, nil
}

type embeddingSelectionCase struct {
	id        string
	embedding bool
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

type embeddingWireRequest struct {
	Model          string   `json:"model"`
	Input          []string `json:"input"`
	Dimensions     int      `json:"dimensions"`
	EncodingFormat string   `json:"encoding_format"`
}
