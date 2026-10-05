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
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	"github.com/maruel/roundtrippers"

	"github.com/maruel/genai"
	"github.com/maruel/genai/internal"
	"github.com/maruel/genai/internal/internaltest"
	"github.com/maruel/genai/providers/llamacpp"
	"github.com/maruel/genai/providers/llamacpp/llamacppsrv"
	"github.com/maruel/genai/scoreboard"
	"github.com/maruel/genai/smoke/smoketest"
)

func TestClient(t *testing.T) {
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
		ctx := t.Context()
		smoketest.Run(t, func(t testing.TB, model scoreboard.Model, fn func(http.RoundTripper) http.RoundTripper) genai.Provider {
			serverURL := s.lazyStartModel(t, model) //nolint:contextcheck // uses l.t.Context() for server lifecycle
			opts := []genai.ProviderOption{genai.ProviderOptionRemote(serverURL)}
			if model.Model != "" {
				opts = append(opts, genai.ProviderOptionModel(model.Model))
			}
			if fn != nil {
				opts = append([]genai.ProviderOption{genai.ProviderOptionTransportWrapper(func(h http.RoundTripper) http.RoundTripper {
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

// ensureExe retrieves the cached llama-server binary for the selected version.
func (l *lazyServer) ensureExe(ctx context.Context, version string) (string, error) {
	cache, err := filepath.Abs("testdata/tmp")
	if err != nil {
		return "", err
	}
	if err := os.MkdirAll(cache, 0o755); err != nil {
		return "", err
	}
	return llamacppsrv.DownloadVersion(ctx, cache, version)
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
	version := llamacppsrv.Version
	if strings.HasPrefix(model.Model, "ggml-org/Kev-4B-GGUF/") {
		// Kev requires decision heads introduced after the v0.5.0 stable release.
		version = "b11361"
	}
	exe, err := l.ensureExe(t.Context(), version)
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
