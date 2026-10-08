// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Tests for the llama.cpp provider client.

package llamacpp_test

import (
	"bytes"
	"context"
	"crypto/rand"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"slices"
	"strconv"
	"strings"
	"sync"
	"testing"

	"github.com/maruel/roundtrippers"

	"github.com/maruel/genai"
	"github.com/maruel/genai/base"
	"github.com/maruel/genai/internal"
	"github.com/maruel/genai/internal/internaltest"
	"github.com/maruel/genai/providers/llamacpp"
	"github.com/maruel/genai/providers/llamacpp/llamacppsrv"
	"github.com/maruel/genai/scoreboard"
	"github.com/maruel/genai/smoke/smoketest"
)

func TestClient(t *testing.T) {
	t.Run("New/embedding preference/error", func(t *testing.T) {
		c, err := llamacpp.New(t.Context(), genai.ModelCheap, genai.ProviderOptionModalities{genai.ModalityEmbedding}, genai.ProviderOptionTransportWrapper(func(http.RoundTripper) http.RoundTripper { return compactNoTransport{t: t} }))
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

	b := [12]byte{}
	if _, err := io.ReadFull(rand.Reader, b[:]); err != nil {
		t.Fatal(err)
	}
	apiKey := base64.RawURLEncoding.EncodeToString(b[:])
	s := lazyServer{t: t, apiKey: apiKey}

	t.Run("Embed", func(t *testing.T) {
		url := os.Getenv("LLAMA_EMBEDDING_SERVER")
		if url == "" {
			url = "http://127.0.0.1:0"
		}
		c, err := llamacpp.New(t.Context(), genai.ProviderOptionRemote(url), genai.ProviderOptionTransportWrapper(func(h http.RoundTripper) http.RoundTripper {
			return testRecorder.Record(t, &roundtrippers.Header{Header: http.Header{"Authorization": {"Bearer " + apiKey}}, Transport: &embeddingTransport{RoundTripper: h}})
		}))
		if err != nil {
			t.Fatal(err)
		}
		internaltest.CleanupCloser(t, c)
		in := genai.EmbeddingRequest{Inputs: []genai.Request{{Text: "task: search result | query: What causes the northern lights?"}, {Text: "title: none | text: The northern lights are caused by charged particles from the sun."}, {Text: "title: none | text: Bananas are a popular tropical fruit."}}, Dimensions: 0}
		var embedder genai.Provider = c
		out, err := embedder.Embed(t.Context(), &in)
		if err != nil {
			t.Fatal(err)
		}
		if err := out.Validate(); err != nil {
			t.Fatal(err)
		}
		if out.Usage.InputTokens == 0 || out.Usage.TotalTokens != out.Usage.InputTokens {
			t.Fatalf("unexpected usage: %+v", out.Usage)
		}
		internaltest.AssertEmbeddingRetrieval(t, out.Embeddings[0], out.Embeddings[1], out.Embeddings[2])
	})

	t.Run("Capabilities", func(t *testing.T) {
		c, err := llamacpp.New(t.Context(), genai.ProviderOptionTransportWrapper(func(h http.RoundTripper) http.RoundTripper {
			return testRecorder.Record(t, &roundtrippers.Header{
				Header:    http.Header{"Authorization": {"Bearer " + apiKey}},
				Transport: h,
			})
		}), genai.ProviderOptionRemote(s.lazyStart(t)))
		if c != nil {
			internaltest.CleanupCloser(t, c)
		}
		if err != nil {
			t.Fatal(err)
		}
		internaltest.TestCapabilities(t, c)
	})

	t.Run("Scoreboard", func(t *testing.T) {
		sb := llamacpp.Scoreboard().Scenarios
		models := make([]scoreboard.Model, 0, len(sb))
		for _, sc := range sb {
			for _, id := range sc.Models {
				models = append(models, scoreboard.Model{Model: id, Reason: sc.Reason})
			}
		}
		if !slices.Contains(models, scoreboard.Model{Model: "ggml-org/embeddinggemma-2-GGUF/embeddinggemma-2-Q8_0.gguf"}) {
			models = append(models, scoreboard.Model{Model: "ggml-org/embeddinggemma-2-GGUF/embeddinggemma-2-Q8_0.gguf"})
		}
		ctx := t.Context()
		smoketest.Run(t, func(t testing.TB, model scoreboard.Model, fn func(http.RoundTripper) http.RoundTripper) genai.Provider {
			serverURL := "http://127.0.0.1:0"
			embedding := model.Model == "ggml-org/embeddinggemma-2-GGUF/embeddinggemma-2-Q8_0.gguf"
			if embedding {
				if u := os.Getenv("LLAMA_EMBEDDING_SERVER"); u != "" {
					serverURL = u
				}
			} else {
				serverURL = s.lazyStartModel(t, model) //nolint:contextcheck // uses l.t.Context() for server lifecycle
			}
			opts := []genai.ProviderOption{genai.ProviderOptionRemote(serverURL)}
			if embedding {
				opts = append(opts, genai.ProviderOptionModalities{genai.ModalityEmbedding})
			}
			if model.Model != "" {
				opts = append(opts, genai.ProviderOptionModel(model.Model))
			}
			if fn != nil {
				opts = append([]genai.ProviderOption{genai.ProviderOptionTransportWrapper(func(h http.RoundTripper) http.RoundTripper {
					if embedding {
						h = &embeddingTransport{RoundTripper: h}
					}
					return fn(&roundtrippers.Header{
						Header:    http.Header{"Authorization": {"Bearer " + apiKey}},
						Transport: h,
					})
				})}, opts...)
			}
			c2, err2 := llamacpp.New(ctx, opts...)
			if c2 != nil {
				internaltest.CleanupCloser(t, c2)
			}
			if err2 != nil {
				t.Fatal(err2)
			}
			if model.Reason {
				return &internaltest.InjectOptions{
					Provider: c2,
					Opts:     []genai.GenOption{&llamacpp.GenOption{ReasoningFormat: llamacpp.ReasoningFormatDeepSeek, Thinking: true}},
				}
			}
			// llama-server defaults enable_thinking to true, so explicitly
			// disable it for non-reasoning scenarios.
			return &internaltest.InjectOptions{
				Provider: c2,
				Opts:     []genai.GenOption{&llamacpp.GenOption{}},
			}
		}, models, testRecorder.Records, &smoketest.RunOptions{
			Qualify: []scoreboard.Model{{Model: "ggml-org/embeddinggemma-2-GGUF/embeddinggemma-2-Q8_0.gguf"}},
			// Gemma 4 emits reasoning content even with enable_thinking=false.
			TolerateReasoning: []string{"gemma-4"},
		})
	})

	// Note: Skipping Preferred test as llamacpp scoreboard doesn't define
	// preferred models (SOTA/Good/Cheap). Model selection is handled by
	// querying the running llama-server instance.

	t.Run("TextOutputDocInput", func(t *testing.T) {
		internaltest.TestTextOutputDocInput(t, func(t *testing.T) genai.Provider {
			c, err := llamacpp.New(t.Context(), genai.ProviderOptionTransportWrapper(func(h http.RoundTripper) http.RoundTripper {
				return testRecorder.Record(t, &roundtrippers.Header{
					Header:    http.Header{"Authorization": {"Bearer " + apiKey}},
					Transport: h,
				})
			}), genai.ProviderOptionRemote(s.lazyStart(t)), genai.ModelCheap)
			if c != nil {
				internaltest.CleanupCloser(t, c)
			}
			if err != nil {
				t.Fatal(err)
			}
			return c
		})
	})

	t.Run("ListModels", func(t *testing.T) {
		ctx := t.Context()
		c, err := llamacpp.New(ctx, genai.ProviderOptionTransportWrapper(func(h http.RoundTripper) http.RoundTripper {
			return testRecorder.Record(t, &roundtrippers.Header{
				Header:    http.Header{"Authorization": {"Bearer " + apiKey}},
				Transport: h,
			})
		}), genai.ProviderOptionRemote(s.lazyStart(t)))
		if c != nil {
			internaltest.CleanupCloser(t, c)
		}
		if err != nil {
			t.Fatal(err)
		}
		genaiModels, err := c.ListModels(ctx)
		if err != nil {
			t.Fatal(err)
		}
		if len(genaiModels) != 1 {
			t.Fatalf("unexpected: %#v", genaiModels)
		}
	})

	// Run this at the end so there would be non-zero values.
	t.Run("Metrics", func(t *testing.T) {
		ctx := t.Context()
		c, err := llamacpp.New(ctx, genai.ProviderOptionTransportWrapper(func(h http.RoundTripper) http.RoundTripper {
			return testRecorder.Record(t, &roundtrippers.Header{
				Header:    http.Header{"Authorization": {"Bearer " + apiKey}},
				Transport: h,
			})
		}), genai.ProviderOptionRemote(s.lazyStart(t)))
		if c != nil {
			internaltest.CleanupCloser(t, c)
		}
		if err != nil {
			t.Fatal(err)
		}
		m := llamacpp.Metrics{}
		if err := c.GetMetrics(ctx, &m); err != nil {
			t.Fatal(err)
		}
		t.Logf("Metrics: %+v", m)
	})
}

type lazyServer struct {
	t      testing.TB
	apiKey string

	mu      sync.Mutex
	servers map[string]string // base model path -> URL
}

// lazyStart starts the default server, picking the first model with image
// support since TextOutputDocInput needs it. Scenario ordering may change
// after -update-scoreboard sorts by reasoning first.
func (l *lazyServer) lazyStart(t testing.TB) string {
	sb := llamacpp.Scoreboard()
	for i := range sb.Scenarios {
		sc := &sb.Scenarios[i]
		if _, ok := sc.In[scoreboard.ModalityImage]; ok {
			return l.lazyStartModel(t, scoreboard.Model{Model: sc.Models[0], Reason: sc.Reason})
		}
	}
	sc := llamacpp.Scoreboard().Scenarios[0]
	return l.lazyStartModel(t, scoreboard.Model{Model: sc.Models[0], Reason: sc.Reason})
}

// ensureExe retrieves the cached llama-server binary for the default version.
func (l *lazyServer) ensureExe(ctx context.Context) (string, error) {
	cache, err := filepath.Abs("testdata/tmp")
	if err != nil {
		return "", err
	}
	if err := os.MkdirAll(cache, 0o755); err != nil {
		return "", err
	}
	return llamacppsrv.DownloadVersion(ctx, cache, llamacppsrv.Version)
}

// lazyStartModel starts a server for the given model key, reusing an existing one if already running.
func (l *lazyServer) lazyStartModel(t testing.TB, model scoreboard.Model) string {
	if model.Model == "" {
		// Empty model is used by smoketest.Run to read the static Scoreboard();
		// no server is needed.
		return "http://127.0.0.1:0"
	}
	// Skip server startup when not recording. The HTTP cassettes in testdata/
	// are replayed by the recording transport so the URL is never contacted.
	if os.Getenv("RECORD") == "" {
		return "http://127.0.0.1:0"
	}
	if url := os.Getenv("LLAMA_SERVER"); url != "" {
		return url
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	if l.servers == nil {
		l.servers = make(map[string]string)
	}
	if u, ok := l.servers[model.Model]; ok {
		return u
	}
	t.Logf("Starting server for %s", model.Model)
	exe, err := l.ensureExe(t.Context())
	if err != nil {
		t.Fatal(err)
	}
	parts := strings.Split(model.Model, "/")
	// Use the parent context for server lifecycle management, but report
	// startup failures to the subtest that requested the server.
	srv := startServerTest(t, l.t.Context(), exe, parts[0], parts[1], parts[2], l.apiKey)
	u := srv.URL()
	l.servers[model.Model] = u
	l.t.Cleanup(func() {
		if err := srv.Close(); err != nil && !errors.Is(err, context.Canceled) {
			// llama-server may exit with code 1 on SIGINT; ignore ExitError
			// since we intentionally stopped it.
			if _, ok := errors.AsType[*exec.ExitError](err); !ok {
				l.t.Error(err)
			}
		}
	})
	return u
}

func startServerTest(t testing.TB, ctx context.Context, exe, author, repo, modelfile, apiKey string) *llamacppsrv.Server {
	cache, err := filepath.Abs("testdata/tmp")
	if err != nil {
		t.Fatal(err)
	}
	// Use llama-server's built-in HuggingFace download; mmproj is auto-detected.
	extraArgs := []string{
		"-hf", author + "/" + repo,
		"-hff", modelfile,
		"--jinja",
		"--flash-attn", "on",
		"--ctx-size", "32768", // Thinking models can generate a lot of tokens. Fails on TopLogprob and Citations-text-plain.
		"--cache-type-k", "q8_0",
		"--cache-type-v", "q8_0",
		"--reasoning-preserve",
		"--spec-default",
		// "--spec-type", "draft-mtp",
		"--api-key", apiKey,
		"--cors-origins", "",
		"--no-cors-credentials",
		"--no-ui",
		"--no-slots",
		"--parallel", "4",
		"--kv-unified",
	}
	if repo == "Kev-4B-GGUF" {
		extraArgs = append(extraArgs, "--no-warmup")
	}
	// Allocate an ephemeral port to avoid dual-stack conflicts when running
	// multiple servers (e.g. "localhost:8080" can bind on both IPv4 and IPv6).
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	hostPort := ln.Addr().String()
	_ = ln.Close()
	l := internaltest.LogFile(t, cache, "llama-server.log")
	srv, err := llamacppsrv.New(ctx, exe, "", l, hostPort, 0, extraArgs)
	if err != nil {
		t.Fatal(err)
	}
	return srv
}

func TestGenOption(t *testing.T) {
	msgs := genai.Messages{genai.NewTextMessage("test")}
	t.Run("reasoning_format", func(t *testing.T) {
		var req llamacpp.ChatRequest
		if err := req.Init(msgs, "model", &llamacpp.GenOption{ReasoningFormat: llamacpp.ReasoningFormatDeepSeek}); err != nil {
			t.Fatal(err)
		}
		if req.ReasoningFormat != llamacpp.ReasoningFormatDeepSeek {
			t.Errorf("ReasoningFormat = %q, want %q", req.ReasoningFormat, llamacpp.ReasoningFormatDeepSeek)
		}
	})
	t.Run("enable_thinking", func(t *testing.T) {
		var req llamacpp.ChatRequest
		if err := req.Init(msgs, "model", &llamacpp.GenOption{Thinking: true}); err != nil {
			t.Fatal(err)
		}
		if req.ChatTemplateKWArgs == nil {
			t.Fatal("ChatTemplateKWArgs is nil")
		}
		if v, ok := req.ChatTemplateKWArgs["enable_thinking"]; !ok || string(v) != "true" {
			t.Errorf("ChatTemplateKWArgs[enable_thinking] = %v, want true", v)
		}
	})
	t.Run("disabled", func(t *testing.T) {
		var req llamacpp.ChatRequest
		if err := req.Init(msgs, "model", &llamacpp.GenOption{}); err != nil {
			t.Fatal(err)
		}
		if req.ReasoningFormat != "" {
			t.Errorf("ReasoningFormat = %q, want empty", req.ReasoningFormat)
		}
		if req.ChatTemplateKWArgs == nil {
			t.Fatal("ChatTemplateKWArgs is nil")
		}
		if v, ok := req.ChatTemplateKWArgs["enable_thinking"]; !ok || string(v) != "false" {
			t.Errorf("ChatTemplateKWArgs[enable_thinking] = %v, want false", v)
		}
	})
	t.Run("system_prompt", func(t *testing.T) {
		var req llamacpp.ChatRequest
		opts := &genai.GenOptionText{SystemPrompt: "You are concise."}
		if err := req.Init(msgs, "model", opts); err != nil {
			t.Fatal(err)
		}
		if len(req.Messages) != 2 {
			t.Fatalf("Messages = %d, want 2", len(req.Messages))
		}
		if req.Messages[0].Role != "system" {
			t.Errorf("Messages[0].Role = %q, want system", req.Messages[0].Role)
		}
		if len(req.Messages[0].Content) != 1 || req.Messages[0].Content[0].Text != "You are concise." {
			t.Errorf("Messages[0].Content = %#v, want system prompt", req.Messages[0].Content)
		}
		if req.Messages[1].Role != "user" {
			t.Errorf("Messages[1].Role = %q, want user", req.Messages[1].Role)
		}
	})
}

func TestMessage(t *testing.T) {
	t.Run("To/with_reasoning", func(t *testing.T) {
		m := llamacpp.Message{
			Role:             "assistant",
			Content:          llamacpp.Contents{{Type: "text", Text: "hello"}},
			ReasoningContent: "thinking...",
		}
		var out genai.Message
		if err := m.To(&out); err != nil {
			t.Fatal(err)
		}
		if len(out.Replies) != 2 {
			t.Fatalf("len(Replies) = %d, want 2", len(out.Replies))
		}
		if out.Replies[0].Reasoning != "thinking..." {
			t.Errorf("Replies[0].Reasoning = %q, want %q", out.Replies[0].Reasoning, "thinking...")
		}
		if out.Replies[1].Text != "hello" {
			t.Errorf("Replies[1].Text = %q, want %q", out.Replies[1].Text, "hello")
		}
	})
	t.Run("To/without_reasoning", func(t *testing.T) {
		m := llamacpp.Message{
			Role:    "assistant",
			Content: llamacpp.Contents{{Type: "text", Text: "hello"}},
		}
		var out genai.Message
		if err := m.To(&out); err != nil {
			t.Fatal(err)
		}
		if len(out.Replies) != 1 {
			t.Fatalf("len(Replies) = %d, want 1", len(out.Replies))
		}
		if out.Replies[0].Text != "hello" {
			t.Errorf("Replies[0].Text = %q, want %q", out.Replies[0].Text, "hello")
		}
	})
}

func init() {
	internal.BeLenient = false
}

// embeddingTransport requires an external embedding server only when the
// recorder actually sends a live request. Existing cassettes replay without it.
type embeddingTransport struct {
	http.RoundTripper
}

func (e *embeddingTransport) RoundTrip(r *http.Request) (*http.Response, error) {
	if os.Getenv("LLAMA_EMBEDDING_SERVER") == "" {
		return nil, errors.New("recording EmbeddingGemma 2 requires LLAMA_EMBEDDING_SERVER pointing to a server with --embeddings and embeddinggemma-2-Q8_0.gguf")
	}
	return e.RoundTripper.RoundTrip(r)
}

func TestClientEmbeddings(t *testing.T) {
	t.Run("Embed/error/indices", func(t *testing.T) {
		for _, tc := range []embeddingIndexErrorCase{
			{"duplicate", `{"object":"list","data":[{"object":"embedding","index":0,"embedding":"AACAPw=="},{"object":"embedding","index":0,"embedding":"AACAPw=="}]}`},
			{"negative", `{"object":"list","data":[{"object":"embedding","index":-1,"embedding":"AACAPw=="},{"object":"embedding","index":1,"embedding":"AACAPw=="}]}`},
			{"out of bounds", `{"object":"list","data":[{"object":"embedding","index":2,"embedding":"AACAPw=="},{"object":"embedding","index":1,"embedding":"AACAPw=="}]}`},
		} {
			t.Run(tc.name, func(t *testing.T) {
				c, err := llamacpp.New(t.Context(), genai.ProviderOptionTransportWrapper(func(http.RoundTripper) http.RoundTripper { return embeddingDocumentTransport{body: tc.body} }))
				if err != nil {
					t.Fatal(err)
				}
				internaltest.CleanupCloser(t, c)
				out, err := c.Embed(t.Context(), &genai.EmbeddingRequest{Inputs: []genai.Request{{Text: "a"}, {Text: "b"}}})
				if _, ok := errors.AsType[*internal.BadError](err); !ok || out != nil {
					t.Fatalf("response %+v, error %v", out, err)
				}
			})
		}
	})

	t.Run("EmbedRaw/error/partial response", func(t *testing.T) {
		old := internal.BeLenient
		t.Cleanup(func() { internal.BeLenient = old })
		for _, lenient := range []bool{false, true} {
			t.Run(strconv.FormatBool(lenient), func(t *testing.T) {
				internal.BeLenient = lenient
				c, err := llamacpp.New(t.Context(), genai.ProviderOptionModel("model"), genai.ProviderOptionTransportWrapper(func(http.RoundTripper) http.RoundTripper {
					return embeddingDocumentTransport{body: `{"object":"list","data":[{"object":"embedding","index":0,"embedding":"AACAPw=="},{"object":"embedding","index":1,"embedding":"AA=="}]}`}
				}))
				if err != nil {
					t.Fatal(err)
				}
				internaltest.CleanupCloser(t, c)
				var out llamacpp.EmbeddingResponse
				err = c.EmbedRaw(t.Context(), &llamacpp.EmbeddingRequest{Model: "model", Input: []llamacpp.EmbeddingInput{{Text: "a"}, {Text: "b"}}}, &out)
				if _, ok := errors.AsType[*internal.BadError](err); !ok || !reflect.ValueOf(out).IsZero() {
					t.Fatalf("response %+v, error %v", out, err)
				}
			})
		}
	})

	t.Run("Embed/valid/document ownership", func(t *testing.T) {
		c, err := llamacpp.New(t.Context(), genai.ProviderOptionRemote("http://localhost:8080"), genai.ProviderOptionTransportWrapper(func(http.RoundTripper) http.RoundTripper {
			return embeddingDocumentTransport{body: `{"object":"list","data":[{"object":"embedding","index":1,"embedding":"AABAQAAAgEA="},{"object":"embedding","index":0,"embedding":"AACAPwAAAEA="}]}`}
		}))
		if err != nil {
			t.Fatal(err)
		}
		internaltest.CleanupCloser(t, c)
		src := &embeddingDocumentReader{Reader: *strings.NewReader("image")}
		in := genai.EmbeddingRequest{Inputs: []genai.Request{{Doc: genai.Doc{Filename: "image.png", Src: src}}, {Text: "caption"}}}
		out, err := c.Embed(t.Context(), &in)
		if err != nil {
			t.Fatal(err)
		}
		if in.Inputs[0].Doc.Src != src || !reflect.DeepEqual(out.Embeddings, [][]float32{{1, 2}, {3, 4}}) {
			t.Fatalf("source changed or response invalid: %+v", out)
		}
	})

	t.Run("Embed/input error", func(t *testing.T) {
		c, err := llamacpp.New(t.Context(), genai.ProviderOptionTransportWrapper(func(http.RoundTripper) http.RoundTripper { return compactNoTransport{t: t} }))
		if err != nil {
			t.Fatal(err)
		}
		internaltest.CleanupCloser(t, c)
		if out, err := c.Embed(t.Context(), nil); out != nil || err == nil {
			t.Fatalf("response %+v, error %v", out, err)
		}
		in := &genai.EmbeddingRequest{Inputs: []genai.Request{{Doc: genai.Doc{Filename: "image.png", Src: strings.NewReader("")}}}}
		if out, err := c.Embed(t.Context(), in); out != nil || err == nil || !strings.Contains(err.Error(), "embedding input #0") {
			t.Fatalf("response %+v, error %v", out, err)
		}
	})

	t.Run("EmbedRaw", func(t *testing.T) {
		t.Run("valid native controls", func(t *testing.T) {
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.Method != http.MethodPost || r.URL.Path != "/v1/embeddings" {
					t.Errorf("unexpected request %s %s", r.Method, r.URL.Path)
				}
				var req embeddingCompactWireRequest
				if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
					t.Error(err)
				}
				if req.Model != "alias" || req.EncodingFormat != "base64" || req.Normalize != -1 || len(req.Input) != 2 || string(req.Input[0]) != `[1,2]` || string(req.Input[1]) != `{"content":[{"type":"text","text":"caption"},{"type":"image_url","image_url":{"url":"data:image/png;base64,aW1hZ2U="}}]}` {
					t.Errorf("changed native controls: %+v", req)
				}
				w.Header().Set("Content-Type", "application/json")
				if _, err := io.WriteString(w, `{"model":"server-alias","object":"list","usage":{"prompt_tokens":9,"total_tokens":9},"data":[{"object":"embedding","index":1,"embedding":"AAAAQAAAwEA=","encoding_format":"base64"},{"object":"embedding","index":0,"embedding":"AABAQAAAgEA="}]}`); err != nil {
					t.Error(err)
				}
			}))
			t.Cleanup(srv.Close)
			c, err := llamacpp.New(t.Context(), genai.ProviderOptionRemote(srv.URL))
			if err != nil {
				t.Fatal(err)
			}
			internaltest.CleanupCloser(t, c)
			n := -1
			img := llamacpp.Content{Type: "image_url"}
			img.ImageURL.URL = "data:image/png;base64,aW1hZ2U="
			in := llamacpp.EmbeddingRequest{Model: "alias", Normalize: &n, Input: []llamacpp.EmbeddingInput{{Tokens: []int{1, 2}}, {Content: llamacpp.Contents{{Type: "text", Text: "caption"}, img}}}}
			var out llamacpp.EmbeddingResponse
			err = c.EmbedRaw(t.Context(), &in, &out)
			if err != nil {
				t.Fatal(err)
			}
			if len(out.Data) != 2 || out.Data[0].Index != 1 || !slices.Equal(out.Data[0].Embedding, []float32{2, 6}) || out.Data[1].Index != 0 || !slices.Equal(out.Data[1].Embedding, []float32{3, 4}) || out.Usage.PromptTokens != 9 || out.Usage.TotalTokens != 9 {
				t.Fatalf("changed values/order/usage: %+v", out)
			}
			if err := c.EmbedRaw(t.Context(), &in, nil); err == nil {
				t.Fatal("accepted nil response")
			}
			if len(in.Input[0].Tokens) != 2 || *in.Normalize != -1 {
				t.Fatal("modified request")
			}
		})
		t.Run("error", func(t *testing.T) {
			t.Run("input validation", func(t *testing.T) {
				c, err := llamacpp.New(t.Context(), genai.ProviderOptionTransportWrapper(func(http.RoundTripper) http.RoundTripper { return compactNoTransport{t: t} }))
				if err != nil {
					t.Fatal(err)
				}
				internaltest.CleanupCloser(t, c)
				n := -2
				for _, in := range []*llamacpp.EmbeddingRequest{nil, {}, {Input: []llamacpp.EmbeddingInput{{Tokens: []int{-1}}}}, {Input: []llamacpp.EmbeddingInput{{Text: "text"}}, Normalize: &n}} {
					var out llamacpp.EmbeddingResponse
					if err := c.EmbedRaw(t.Context(), in, &out); err == nil {
						t.Fatalf("accepted invalid request: %+v", in)
					}
				}
			})
			for _, tc := range []embeddingVectorErrorCase{
				{"bad base64", `"!"`}, {"empty", `""`}, {"partial float32", `"AA=="`}, {"NaN", `"AADAfw=="`}, {"infinity", `"AACAfw=="`}, {"null", `null`}, {"object", `{"base64":"AACAPw==","unexpected":true}`},
			} {
				t.Run(tc.name, func(t *testing.T) {
					srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
						w.Header().Set("Content-Type", "application/json")
						if _, err := fmt.Fprintf(w, `{"object":"list","data":[{"object":"embedding","index":0,"embedding":%s}]}`, tc.vector); err != nil {
							t.Error(err)
						}
					}))
					t.Cleanup(srv.Close)
					c, err := llamacpp.New(t.Context(), genai.ProviderOptionRemote(srv.URL))
					if err != nil {
						t.Fatal(err)
					}
					internaltest.CleanupCloser(t, c)
					out, err := c.Embed(t.Context(), &genai.EmbeddingRequest{Inputs: []genai.Request{{Text: "text"}}})
					if _, ok := errors.AsType[*internal.BadError](err); !ok || out != nil {
						t.Fatalf("got %+v, %v", out, err)
					}
				})
			}
			for _, tc := range []embeddingCompactErrorCase{
				{"envelope object", 200, `{"object":"embeddings","data":[{"object":"embedding","index":0,"embedding":"AACAPw=="}]}`, false, true},
				{"result count", 200, `{"object":"list","data":[]}`, false, true},
				{"item object", 200, `{"object":"list","data":[{"object":"vector","index":0,"embedding":"AACAPw=="}]}`, false, true},
				{"unknown field", 200, `{"object":"list","data":[{"object":"embedding","index":0,"embedding":"AACAPw==","unexpected":true}]}`, false, false},
				{"pooling none", 400, `{"error":{"code":400,"message":"Pooling type 'none' is not OAI compatible. Please use a different pooling type","type":"invalid_request_error"}}`, true, false},
				{"unrelated bad request", 400, `{"error":{"code":400,"message":"bad input","type":"invalid_request_error"}}`, false, false},
			} {
				t.Run(tc.name, func(t *testing.T) {
					srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
						w.Header().Set("Content-Type", "application/json")
						w.WriteHeader(tc.code)
						if _, err := io.WriteString(w, tc.body); err != nil {
							t.Error(err)
						}
					}))
					t.Cleanup(srv.Close)
					c, err := llamacpp.New(t.Context(), genai.ProviderOptionRemote(srv.URL))
					if err != nil {
						t.Fatal(err)
					}
					internaltest.CleanupCloser(t, c)
					out, err := c.Embed(t.Context(), &genai.EmbeddingRequest{Inputs: []genai.Request{{Text: "text"}}})
					if err == nil || out != nil {
						t.Fatalf("accepted failed upstream response: %+v", out)
					}
					if _, ok := errors.AsType[*internal.BadError](err); tc.bad && !ok {
						t.Fatalf("expected BadError, got %v", err)
					}
					_, unsupported := errors.AsType[*base.ErrNotSupported](err)
					if err == nil || out != nil || unsupported != tc.unsupported {
						t.Fatalf("misclassified generic error: %+v, %v", out, err)
					}
				})
			}
		})
	})
}

func TestClientSystemOne(t *testing.T) {
	t.Run("disabled uses chat", func(t *testing.T) {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.URL.Path != "/chat/completions" {
				t.Errorf("unexpected route %q", r.URL.Path)
			}
			w.Header().Set("Content-Type", "application/json")
			if _, err := io.WriteString(w, `{"choices":[{"finish_reason":"stop","index":0,"message":{"role":"assistant","content":"chat"}}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`); err != nil {
				t.Error(err)
			}
		}))
		t.Cleanup(srv.Close)
		c, err := llamacpp.New(t.Context(), genai.ProviderOptionRemote(srv.URL))
		if err != nil {
			t.Fatal(err)
		}
		internaltest.CleanupCloser(t, c)
		res, err := c.GenSync(t.Context(), genai.Messages{genai.NewTextMessage("state")})
		if err != nil || len(res.Replies) != 1 || res.Replies[0].Text != "chat" {
			t.Fatalf("chat response = %+v, error = %v", res, err)
		}
	})
	q := genai.Questions{
		"billing": {Type: genai.QuestionNoul, Instructions: genai.Text("Is this about billing?")},
		"route":   {Type: genai.QuestionChoice, Instructions: genai.Text("Which team?"), Choice: map[string]genai.DecisionContent{"billing": nil, "support": nil}},
		"urgency": {Type: genai.QuestionScore, Instructions: genai.Text("How urgent?"), Score: []genai.DecisionContent{genai.Text("low"), genai.Text("high")}},
	}
	const response = `{"model":"kev","answers":{"billing":{"type":"noul","noul":0.9},"route":{"type":"choice","choice":"billing","confidence":0.8,"probabilities":{"billing":0.9,"support":0.1}},"urgency":{"type":"score","score":0.75,"confidence":0.5,"legend":{"0":"low","1":"high"},"probabilities":{"0":0.25,"1":0.75}}},"usage":{"input_tokens":42,"output_tokens":0}}`
	t.Run("SystemOneRaw", func(t *testing.T) {
		t.Run("limits at request boundary", func(t *testing.T) {
			srv := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) { t.Error("unexpected HTTP request") }))
			t.Cleanup(srv.Close)
			c, err := llamacpp.New(t.Context(), genai.ProviderOptionRemote(srv.URL))
			if err != nil {
				t.Fatal(err)
			}
			internaltest.CleanupCloser(t, c)
			req := llamacpp.SystemOneRequest{State: genai.Text("state")}
			qs := genai.Questions{"q": {Type: genai.QuestionScore, Instructions: genai.Text("rate"), Score: []genai.DecisionContent{genai.Text("only")}}}
			req.Questions = qs
			err = c.SystemOneRaw(t.Context(), &req, &llamacpp.SystemOneResponse{})
			if err == nil || !strings.Contains(err.Error(), "2 to 10 levels are required") {
				t.Fatalf("missing native boundary limit: %v", err)
			}
		})
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			var req struct {
				Model     string                     `json:"model"`
				State     map[string]int             `json:"state"`
				Images    []string                   `json:"images"`
				Questions map[string]json.RawMessage `json:"questions"`
			}
			if r.URL.Path != "/v1/systemone" {
				t.Errorf("unexpected path %s", r.URL.Path)
			}
			if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
				t.Error(err)
			}
			if req.Model != "kev" || req.State["ticket"] != 42 || len(req.Images) != 1 || len(req.Questions) != 3 {
				t.Errorf("unexpected request: %+v", req)
			}
			w.Header().Set("Content-Type", "application/json")
			if _, err := w.Write([]byte(response)); err != nil {
				t.Error(err)
			}
		}))
		t.Cleanup(srv.Close)
		c, err := llamacpp.New(t.Context(), genai.ProviderOptionRemote(srv.URL))
		if err != nil {
			t.Fatal(err)
		}
		out := &llamacpp.SystemOneResponse{}
		if err := c.SystemOneRaw(t.Context(), &llamacpp.SystemOneRequest{Model: "kev", State: genai.Object{"ticket": 42}, Questions: q, Images: []string{"data:image/png;base64,aW1hZ2U="}}, out); err != nil {
			t.Fatal(err)
		}
		if out.Model != "kev" || out.Answers["billing"].Noul != 0.9 {
			t.Errorf("unexpected response: %+v", out)
		}
	})
	t.Run("SystemOne", func(t *testing.T) {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.Method != http.MethodPost || r.URL.Path != "/v1/systemone" {
				t.Errorf("unexpected request: %s %s", r.Method, r.URL.Path)
			}
			var req struct {
				State     string                     `json:"state"`
				Model     string                     `json:"model"`
				Questions map[string]json.RawMessage `json:"questions"`
			}
			if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
				t.Error(err)
			}
			if req.State != "Charged twice" || req.Model != "" || len(req.Questions) != 3 {
				t.Errorf("unexpected request: %+v", req)
			}
			w.Header().Set("Content-Type", "application/json")
			if _, err := w.Write([]byte(response)); err != nil {
				t.Error(err)
			}
		}))
		t.Cleanup(srv.Close)
		c, err := llamacpp.New(t.Context(), genai.ProviderOptionRemote(srv.URL))
		if err != nil {
			t.Fatal(err)
		}
		res, err := c.SystemOne(t.Context(), &genai.SystemOneRequest{State: genai.Text("Charged twice"), Questions: q})
		if err != nil {
			t.Fatal(err)
		}

		if res.Answers["billing"].Noul != 0.9 || res.Answers["route"].Choice != "billing" || res.Answers["urgency"].Score != 0.75 {
			t.Errorf("unexpected answers: %+v", res.Answers)
		}
		if res.Usage.InputTokens != 42 || res.Usage.OutputTokens != 0 {
			t.Errorf("unexpected usage: %+v", res.Usage)
		}
	})
	t.Run("error", func(t *testing.T) {
		for _, tc := range []struct {
			name, body string
			status     int
		}{
			{"unsupported model", `{"error":{"code":501,"message":"This model is not a decision model","type":"not_supported_error"}}`, 501},
			{"missing answers", `{"model":"kev","answers":{},"usage":{"input_tokens":1,"output_tokens":0}}`, 200},
			{"missing question", `{"model":"kev","answers":{"billing":{"type":"noul","noul":1}},"usage":{"input_tokens":1,"output_tokens":0}}`, 200},
			{"wrong answer type", `{"model":"kev","answers":{"billing":{"type":"choice","choice":"billing"}},"usage":{"input_tokens":1,"output_tokens":0}}`, 200},
			{"malformed response", `{`, 200},
		} {
			t.Run(tc.name, func(t *testing.T) {
				srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
					w.Header().Set("Content-Type", "application/json")
					w.WriteHeader(tc.status)
					if _, err := w.Write([]byte(tc.body)); err != nil {
						t.Error(err)
					}
				}))
				t.Cleanup(srv.Close)
				c, err := llamacpp.New(t.Context(), genai.ProviderOptionRemote(srv.URL))
				if err != nil {
					t.Fatal(err)
				}
				_, err = c.SystemOne(t.Context(), &genai.SystemOneRequest{State: genai.Text("x"), Questions: q})
				if err == nil {
					t.Fatal("expected an error")
				}
				if tc.status == 501 {
					var api *llamacpp.ErrorResponse
					if !errors.As(err, &api) || api.ErrorVal.Code != 501 {
						t.Errorf("unexpected error: %v", err)
					}
				}
			})
		}
	})
}

func TestClientChronologicalTemplate(t *testing.T) {
	msgs := genai.Messages{
		genai.NewTextMessage("first"), genai.NewTextMessage("correction"),
		{Replies: []genai.Reply{{ToolCall: genai.ToolCall{ID: "A", Name: "status", Arguments: `{}`}}}},
		genai.NewTextMessage("intervening"),
		{Replies: []genai.Reply{{ToolCall: genai.ToolCall{ID: "B", Name: "status", Arguments: `{}`}}}},
		{ToolCallResults: []genai.ToolCallResult{{ID: "B", Name: "status", Result: "waiting"}}},
		{ToolCallResults: []genai.ToolCallResult{{ID: "A", Name: "status", Result: "running"}}},
	}
	before, err := json.Marshal(msgs)
	if err != nil {
		t.Fatal(err)
	}
	var template []llamacpp.Message
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		switch r.URL.Path {
		case "/apply-template":
			var req struct {
				Messages []llamacpp.Message `json:"messages"`
			}
			if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
				t.Error(err)
				w.WriteHeader(http.StatusBadRequest)
				return
			}
			template = req.Messages
			if _, err := io.WriteString(w, `{"prompt":"server-rendered chronological prompt"}`); err != nil {
				t.Error(err)
			}
		case "/completions":
			var req llamacpp.CompletionRequest
			if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
				t.Error(err)
				w.WriteHeader(http.StatusBadRequest)
				return
			}
			if req.Prompt != "server-rendered chronological prompt" {
				t.Errorf("prompt=%q", req.Prompt)
			}
			w.WriteHeader(http.StatusBadRequest)
			if _, err := io.WriteString(w, `{"error":{"code":400,"message":"template chronology rejected","type":"invalid_request_error"}}`); err != nil {
				t.Error(err)
			}
		default:
			t.Errorf("unexpected request: %s", r.URL.Path)
			w.WriteHeader(http.StatusInternalServerError)
		}
	}))
	t.Cleanup(srv.Close)
	c, err := llamacpp.New(t.Context(), genai.ProviderOptionRemote(srv.URL), genai.ProviderOptionModel("test"))
	if err != nil {
		t.Fatal(err)
	}
	internaltest.CleanupCloser(t, c)
	if _, err := c.Completion(t.Context(), msgs); err == nil || !strings.Contains(err.Error(), "template chronology rejected") {
		t.Fatalf("provider error=%v", err)
	}
	if len(template) != 7 || template[0].Role != "user" || template[1].Role != "user" || template[2].ToolCalls[0].ID != "A" || template[3].Content[0].Text != "intervening" || template[4].ToolCalls[0].ID != "B" || template[5].ToolCallID != "B" || template[5].Content[0].Text != "waiting" || template[6].ToolCallID != "A" || template[6].Content[0].Text != "running" {
		t.Fatalf("template history changed: %+v", template)
	}
	after, err := json.Marshal(msgs)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(before, after) {
		t.Fatal("template mutated input")
	}
}

// TestEmbeddingSelection verifies embedding output can be declared for any server model.
func TestEmbeddingSelection(t *testing.T) {
	for _, id := range []string{"ggml-org/embeddinggemma-2-GGUF", "new-embedding.gguf", "server-alias"} {
		t.Run(id, func(t *testing.T) {
			c, err := llamacpp.New(t.Context(), genai.ProviderOptionModel(id), genai.ProviderOptionModalities{genai.ModalityEmbedding}, genai.ProviderOptionTransportWrapper(func(http.RoundTripper) http.RoundTripper { return compactNoTransport{t: t} }))
			if err != nil {
				t.Fatal(err)
			}
			internaltest.CleanupCloser(t, c)
			if c.ModelID() != id || !slices.Equal(c.OutputModalities(), genai.Modalities{genai.ModalityEmbedding}) {
				t.Fatalf("model %s, modalities %v", c.ModelID(), c.OutputModalities())
			}
		})
	}
	t.Run("loaded model selection", func(t *testing.T) {
		mdl := &llamacpp.Model{}
		mdl.OpenAI.ID = "author/new-model"
		for _, preference := range []genai.ProviderOptionModel{genai.ModelCheap, genai.ModelGood, genai.ModelSOTA} {
			t.Run(string(preference), func(t *testing.T) {
				c, err := llamacpp.New(t.Context(), preference, genai.ProviderOptionPreloadedModels{mdl}, genai.ProviderOptionTransportWrapper(func(http.RoundTripper) http.RoundTripper { return compactNoTransport{t: t} }))
				if err != nil {
					t.Fatal(err)
				}
				internaltest.CleanupCloser(t, c)
				if c.ModelID() != mdl.GetID() || !slices.Equal(c.OutputModalities(), genai.Modalities{genai.ModalityText}) {
					t.Fatalf("model %s, modalities %v", c.ModelID(), c.OutputModalities())
				}
			})
		}
	})
}

// compactNoTransport asserts invalid native requests fail before network I/O.
type compactNoTransport struct{ t *testing.T }

func (c compactNoTransport) RoundTrip(*http.Request) (*http.Response, error) {
	c.t.Fatal("invalid compact request reached transport")
	return nil, errors.New("unexpected compact request")
}

type embeddingCompactWireRequest struct {
	Model          string
	Input          []json.RawMessage
	EncodingFormat string `json:"encoding_format"`
	Normalize      int    `json:"embd_normalize"`
}

type embeddingVectorErrorCase struct {
	name   string
	vector string
}

type embeddingCompactErrorCase struct {
	name        string
	code        int
	body        string
	unsupported bool
	bad         bool
}

type embeddingDocumentReader struct{ strings.Reader }

func (*embeddingDocumentReader) Seek(int64, int) (int64, error) { return 0, errors.New("unseekable") }

type embeddingDocumentTransport struct{ body string }

func (e embeddingDocumentTransport) RoundTrip(r *http.Request) (*http.Response, error) {
	return &http.Response{StatusCode: http.StatusOK, Header: http.Header{"Content-Type": {"application/json"}}, Body: io.NopCloser(strings.NewReader(e.body)), Request: r}, nil
}

type embeddingIndexErrorCase struct{ name, body string }
