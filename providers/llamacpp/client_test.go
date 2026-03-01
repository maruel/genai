// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package llamacpp_test

import (
	"context"
	"crypto/rand"
	"encoding/base64"
	"errors"
	"io"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	"github.com/maruel/genai"
	"github.com/maruel/genai/internal"
	"github.com/maruel/genai/internal/internaltest"
	"github.com/maruel/genai/providers/llamacpp"
	"github.com/maruel/genai/providers/llamacpp/llamacppsrv"
	"github.com/maruel/genai/scoreboard"
	"github.com/maruel/genai/smoke/smoketest"
	"github.com/maruel/huggingface"
	"github.com/maruel/roundtrippers"
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
			serverURL := s.lazyStartModel(t, model)
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
			if err2 != nil {
				t.Fatal(err2)
			}
			if model.Reason {
				return &internaltest.InjectOptions{
					Provider: c2,
					Opts:     []genai.GenOption{&llamacpp.GenOption{ReasoningFormat: llamacpp.ReasoningFormatDeepSeek}},
				}
			}
			return c2
		}, models, testRecorder.Records)
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
	exe     string            // cached llama-server path
	servers map[string]string // base model path -> URL
}

// lazyStart starts the default server, picking the first model with image
// support since TextOutputDocInput needs it. Scenario ordering may change
// after -update-scoreboard sorts by reasoning first.
func (l *lazyServer) lazyStart(t testing.TB) string {
	for _, sc := range llamacpp.Scoreboard().Scenarios {
		if _, ok := sc.In[scoreboard.ModalityImage]; ok {
			return l.lazyStartModel(t, scoreboard.Model{Model: sc.Models[0], Reason: sc.Reason})
		}
	}
	sc := llamacpp.Scoreboard().Scenarios[0]
	return l.lazyStartModel(t, scoreboard.Model{Model: sc.Models[0], Reason: sc.Reason})
}

// ensureExe downloads the llama-server binary once. Must be called with l.mu held.
func (l *lazyServer) ensureExe() string {
	if l.exe != "" {
		return l.exe
	}
	cache, err := filepath.Abs("testdata/tmp")
	if err != nil {
		l.t.Fatal(err)
	}
	if err = os.MkdirAll(cache, 0o755); err != nil {
		l.t.Fatal(err)
	}
	exe, err := llamacppsrv.DownloadRelease(l.t.Context(), cache, llamacppsrv.BuildNumber)
	if err != nil {
		l.t.Fatal(err)
	}
	l.exe = exe
	return exe
}

// lazyStartModel starts a server for the given model key, reusing an existing
// one if the same base model (without #mmproj suffix) is already running.
func (l *lazyServer) lazyStartModel(t testing.TB, model scoreboard.Model) string {
	if model.Model == "" {
		return l.lazyStart(t)
	}
	if os.Getenv("RECORD") != "all" && os.Getenv("CI") == "true" {
		return "http://localhost:0"
	}
	if url := os.Getenv("LLAMA_SERVER"); url != "" {
		return url
	}
	// Use the base model path (before #) as the server key since the same
	// server serves the same model regardless of mmproj.
	baseKey := strings.SplitN(model.Model, "#", 2)[0]
	l.mu.Lock()
	defer l.mu.Unlock()
	if l.servers == nil {
		l.servers = make(map[string]string)
	}
	if u, ok := l.servers[baseKey]; ok {
		return u
	}
	t.Logf("Starting server for %s", model.Model)
	exe := l.ensureExe()
	mains := strings.SplitN(model.Model, "#", 2)
	parts := strings.Split(mains[0], "/")
	mmproj := ""
	if len(mains) > 1 {
		mmproj = mains[1]
	}
	// Use the context of the parent for server lifecycle management.
	srv := startServerTest(l.t, exe, parts[0], parts[1], parts[2], mmproj, l.apiKey)
	u := srv.URL()
	l.servers[baseKey] = u
	l.t.Cleanup(func() {
		if err := srv.Close(); err != nil && !errors.Is(err, context.Canceled) {
			// llama-server may exit with code 1 on SIGINT; ignore ExitError
			// since we intentionally stopped it.
			var exitErr *exec.ExitError
			if !errors.As(err, &exitErr) {
				l.t.Error(err)
			}
		}
	})
	return u
}

func startServerTest(t testing.TB, exe, author, repo, modelfile, multimodal, apiKey string) *llamacppsrv.Server {
	cache, err := filepath.Abs("testdata/tmp")
	if err != nil {
		t.Fatal(err)
	}
	ctx := t.Context()
	// llama.cpp now knows how to pull from huggingface but this was not integrated yet, so pull a model
	// manually.
	hf, err := huggingface.New("")
	if err != nil {
		t.Fatal(err)
	}
	modelPath, err := hf.EnsureFile(ctx, huggingface.ModelRef{Author: author, Repo: repo}, "HEAD", modelfile)
	if err != nil {
		t.Fatal(err)
	}
	extraArgs := []string{"--jinja", "--flash-attn", "on", "--ctx-size", "8192", "--cache-type-k", "q8_0", "--cache-type-v", "q8_0", "--api-key", apiKey}
	if multimodal != "" {
		mmPath, err := hf.EnsureFile(ctx, huggingface.ModelRef{Author: author, Repo: repo}, "HEAD", multimodal)
		if err != nil {
			t.Fatal(err)
		}
		extraArgs = append(extraArgs, "--mmproj", mmPath)
	}
	// Allocate an ephemeral port to avoid dual-stack conflicts when running
	// multiple servers (e.g. "localhost:8080" can bind on both IPv4 and IPv6).
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	hostPort := ln.Addr().String()
	ln.Close()
	l := internaltest.LogFile(t, cache, "llama-server.log")
	srv, err := llamacppsrv.New(ctx, exe, modelPath, l, hostPort, 0, extraArgs)
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
		if v, ok := req.ChatTemplateKWArgs["enable_thinking"]; !ok || v != true {
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
		if req.ChatTemplateKWArgs != nil {
			t.Errorf("ChatTemplateKWArgs = %v, want nil", req.ChatTemplateKWArgs)
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
