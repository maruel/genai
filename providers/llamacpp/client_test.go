// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package llamacpp_test

import (
	"context"
	"crypto/rand"
	"encoding/base64"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	"github.com/maruel/genai"
	"github.com/maruel/genai/internal"
	"github.com/maruel/genai/internal/internaltest"
	"github.com/maruel/genai/providers/llamacpp"
	"github.com/maruel/genai/providers/llamacpp/llamacppsrv"
	"github.com/maruel/genai/scoreboard/scoreboardtest"
	"github.com/maruel/huggingface"
)

func TestClient(t *testing.T) {
	b := [12]byte{}
	if _, err := io.ReadFull(rand.Reader, b[:]); err != nil {
		t.Fatal(err)
	}
	apiKey := base64.RawURLEncoding.EncodeToString(b[:])
	s := lazyServer{t: t, apiKey: apiKey}

	t.Run("ListModels", func(t *testing.T) {
		c, err := llamacpp.New(&genai.ProviderOptions{APIKey: apiKey, Remote: s.lazyStart(t), Model: genai.ModelNone}, func(h http.RoundTripper) http.RoundTripper {
			return testRecorder.Record(t, h)
		})
		if err != nil {
			t.Fatal(err)
		}
		genaiModels, err := c.ListModels(t.Context())
		if err != nil {
			t.Fatal(err)
		}
		if len(genaiModels) != 1 {
			t.Fatalf("unexpected: %#v", genaiModels)
		}
	})

	t.Run("Scoreboard", func(t *testing.T) {
		serverURL := s.lazyStart(t)
		c, err := llamacpp.New(&genai.ProviderOptions{APIKey: apiKey, Remote: serverURL, Model: genai.ModelNone}, func(h http.RoundTripper) http.RoundTripper {
			return testRecorder.Record(t, h)
		})
		if err != nil {
			t.Fatal(err)
		}
		var models []scoreboardtest.Model
		for _, sc := range c.Scoreboard().Scenarios {
			for _, id := range sc.Models {
				models = append(models, scoreboardtest.Model{Model: id})
			}
		}
		scoreboardtest.AssertScoreboard(t, func(t testing.TB, model scoreboardtest.Model, fn func(http.RoundTripper) http.RoundTripper) genai.Provider {
			c2, err2 := llamacpp.New(&genai.ProviderOptions{APIKey: apiKey, Remote: serverURL, Model: model.Model}, fn)
			if err2 != nil {
				t.Fatal(err2)
			}
			return c2
		}, models, testRecorder.Records)
	})

	// Run this at the end so there would be non-zero values.
	t.Run("Metrics", func(t *testing.T) {
		serverURL := s.lazyStart(t)
		c, err := llamacpp.New(&genai.ProviderOptions{APIKey: apiKey, Remote: serverURL, Model: genai.ModelNone}, func(h http.RoundTripper) http.RoundTripper {
			return testRecorder.Record(t, h)
		})
		if err != nil {
			t.Fatal(err)
		}
		m := llamacpp.Metrics{}
		if err := c.GetMetrics(t.Context(), &m); err != nil {
			t.Fatal(err)
		}
		t.Logf("Metrics: %+v", m)
	})
}

type lazyServer struct {
	t      testing.TB
	apiKey string

	mu  sync.Mutex
	url string
}

func (l *lazyServer) lazyStart(t testing.TB) string {
	if os.Getenv("RECORD") != "1" && os.Getenv("CI") == "true" {
		return "http://localhost:0"
	}
	if url := os.Getenv("LLAMA_SERVER"); url != "" {
		return url
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	if l.url == "" {
		t.Log("Starting server")
		// Use the context of the parent for server lifecycle management.
		mains := strings.SplitN(llamacpp.Scoreboard.Scenarios[0].Models[0], "#", 2)
		parts := strings.Split(mains[0], "/")
		srv := startServerTest(l.t, parts[0], parts[1], parts[2], mains[1], l.apiKey)
		l.url = srv.URL()
		l.t.Cleanup(func() {
			if err := srv.Close(); err != nil && err != context.Canceled {
				l.t.Error(err)
			}
		})
	}
	return l.url
}

// This test doesn't require the server to start.
func TestClient_Preferred(t *testing.T) {
	data := []struct {
		name string
		want string
	}{
		{genai.ModelCheap, ""},
		{genai.ModelGood, ""},
		{genai.ModelSOTA, ""},
	}
	for _, line := range data {
		t.Run(line.name, func(t *testing.T) {
			c, err := llamacpp.New(&genai.ProviderOptions{Model: line.name}, nil)
			if err != nil {
				t.Fatal(err)
			}
			if got := c.ModelID(); got != line.want {
				t.Fatalf("got model %q, want %q", got, line.want)
			}
		})
	}
}

func startServerTest(t testing.TB, author, repo, modelfile, multimodal, apiKey string) *llamacppsrv.Server {
	cache, err := filepath.Abs("testdata/tmp")
	if err != nil {
		t.Fatal(err)
	}
	if err = os.MkdirAll(cache, 0o755); err != nil {
		t.Fatal(err)
	}
	ctx := t.Context()
	// It's a bit inefficient to download from github every single time.
	exe, err := llamacppsrv.DownloadRelease(ctx, cache, llamacppsrv.BuildNumber)
	if err != nil {
		t.Fatal(err)
	}
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
	extraArgs := []string{"--jinja", "--flash-attn", "--cache-type-k", "q8_0", "--cache-type-v", "q8_0", "--api-key", apiKey}
	mmPath := ""
	if multimodal != "" {
		if mmPath, err = hf.EnsureFile(ctx, huggingface.ModelRef{Author: author, Repo: repo}, "HEAD", multimodal); err != nil {
			t.Fatal(err)
		}
		extraArgs = append(extraArgs, "--mmproj", mmPath)
	}
	l := internaltest.LogFile(t, cache, "llama-server.log")
	srv, err := llamacppsrv.New(ctx, exe, modelPath, l, "", 0, extraArgs)
	if err != nil {
		t.Fatal(err)
	}
	return srv
}

var testRecorder *internaltest.Records

func TestMain(m *testing.M) {
	testRecorder = internaltest.NewRecords()
	code := m.Run()
	os.Exit(max(code, testRecorder.Close()))
}

func init() {
	internal.BeLenient = false
}
