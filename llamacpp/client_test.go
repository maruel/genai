// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package llamacpp_test

import (
	"context"
	"net/http"
	"os"
	"strings"
	"sync"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/maruel/genai"
	"github.com/maruel/genai/internal"
	"github.com/maruel/genai/internal/internaltest"
	"github.com/maruel/genai/llamacpp"
	"gopkg.in/dnaeon/go-vcr.v4/pkg/cassette"
	"gopkg.in/dnaeon/go-vcr.v4/pkg/recorder"
)

// Not implementing TestClient_Chat_allModels since llama-server has no ListModels API.

func TestClient(t *testing.T) {
	s := lazyServer{t: t}

	t.Run("Chat", func(t *testing.T) {
		serverURL, transport := s.shouldStart(t)
		c, err := llamacpp.New(serverURL, nil)
		if err != nil {
			t.Fatal(err)
		}
		c.Client.Client = &http.Client{Transport: transport}
		opts := genai.ChatOptions{
			Seed:        1,
			Temperature: 0.01,
			MaxTokens:   50,
		}
		msgs := genai.Messages{
			genai.NewTextMessage(genai.User, "Say hello. Use only one word."),
		}
		got, err := c.Chat(t.Context(), msgs, &opts)
		if err != nil {
			t.Fatal(err)
		}
		want := genai.NewTextMessage(genai.Assistant, "Hello.")
		if diff := cmp.Diff(want, got.Message); diff != "" {
			t.Fatalf("unexpected response (-want +got):\n%s", diff)
		}
		if got.InputTokens != 3 || got.OutputTokens != 16 {
			t.Logf("Unexpected tokens usage: %v", got.Usage)
		}

		// Second message.
		msgs = append(msgs, got.Message)
		msgs = append(msgs, genai.NewTextMessage(genai.User, "Say banana. Use only one word."))
		got, err = c.Chat(t.Context(), msgs, &opts)
		if err != nil {
			t.Fatal(err)
		}
		want = genai.NewTextMessage(genai.Assistant, "Banana.")
		if diff := cmp.Diff(want, got.Message); diff != "" {
			t.Fatalf("unexpected response (-want +got):\n%s", diff)
		}
		if got.InputTokens != 4 || got.OutputTokens != 36 {
			t.Logf("Unexpected tokens usage: %v", got.Usage)
		}
	})
}

type lazyServer struct {
	t   *testing.T
	mu  sync.Mutex
	url string
}

func (l *lazyServer) shouldStart(t *testing.T) (string, http.RoundTripper) {
	transport := testRecorder.Record(t, http.DefaultTransport,
		recorder.WithHook(func(i *cassette.Interaction) error { return internaltest.SaveIgnorePort(t, i) }, recorder.AfterCaptureHook),
		recorder.WithMatcher(internaltest.MatchIgnorePort),
	)
	if !transport.IsNewCassette() {
		return "http://localhost:0", transport
	}
	name := "testdata/" + strings.ReplaceAll(t.Name(), "/", "_") + ".yaml"
	suffix := " (forced)"
	if os.Getenv("RECORD") != "1" {
		suffix = " (missing " + name + ")"
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	t.Cleanup(func() {
		if t.Failed() {
			t.Log("Removing record")
			_ = os.Remove(name)
		}
	})
	if l.url == "" {
		t.Log("Starting server" + suffix)
		// Use the context of the parent for server lifecycle management.
		srv, err := startServer(l.t.Context())
		if err != nil {
			t.Fatal(err)
		}
		l.url = srv.URL()
		l.t.Cleanup(func() {
			if err := srv.Close(); err != nil && err != context.Canceled {
				l.t.Error(err)
			}
		})
	} else {
		t.Log("Recording " + suffix)
	}
	return l.url, transport
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
