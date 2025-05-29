// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package ollama_test

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
	"github.com/maruel/genai/ollama"
	"gopkg.in/dnaeon/go-vcr.v4/pkg/cassette"
	"gopkg.in/dnaeon/go-vcr.v4/pkg/recorder"
)

// Not implementing TestClient_AllModels since we need to preload Ollama models. Can be done later.

func TestClient(t *testing.T) {
	s := lazyServer{t: t}
	tc := &internaltest.TestCases{
		Default: internaltest.Settings{
			GetClient: func(t *testing.T, m string) genai.ChatProvider { return s.getClient(t, m) },
			Model:     "gemma3:4b",
		},
	}

	t.Run("Chat", func(t *testing.T) {
		c := s.getClient(t, "gemma3:4b")
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
		if got.InputTokens != 17 || got.OutputTokens != 3 {
			t.Logf("Unexpected tokens usage: %v", got.Usage)
		}

		// Second message.
		msgs = append(msgs, got.Message)
		msgs = append(msgs, genai.NewTextMessage(genai.User, "Who created you? Use only one word."))
		got, err = c.Chat(t.Context(), msgs, &opts)
		if err != nil {
			t.Fatal(err)
		}
		want = genai.NewTextMessage(genai.Assistant, "Google.")
		if diff := cmp.Diff(want, got.Message); diff != "" {
			t.Fatalf("unexpected response (-want +got):\n%s", diff)
		}
		if got.InputTokens != 38 || got.OutputTokens != 3 {
			t.Logf("Unexpected tokens usage: %v", got.Usage)
		}
	})

	t.Run("stream", func(t *testing.T) {
		tc.TestChatStream_simple(t, nil)
	})

	t.Run("vision_jpg_inline", func(t *testing.T) {
		tc.TestChatVisionJPGInline(t, nil)
	})

	t.Run("json", func(t *testing.T) {
		tc.TestChatJSON(t, nil)
	})

	t.Run("json_schema", func(t *testing.T) {
		tc.TestChatJSONSchema(t, nil)
	})

	t.Run("tool_use_reply", func(t *testing.T) {
		tc.TestChatToolUseReply(t, &internaltest.Settings{Model: "llama3.1:8b"})
	})

	t.Run("tool_use_position_bias", func(t *testing.T) {
		tc.TestChatToolUsePositionBias(t, &internaltest.Settings{Model: "llama3.1:8b"}, true)
	})

	t.Run("ChatProvider_errors", func(t *testing.T) {
		data := []internaltest.ChatProviderError{
			{
				Name:          "bad model",
				Model:         "bad_model",
				ErrChat:       "pull failed: http 500: error pull model manifest: file does not exist",
				ErrChatStream: "pull failed: http 500: error pull model manifest: file does not exist",
			},
		}
		f := func(t *testing.T, apiKey, model string) genai.ChatProvider {
			return s.getClient(t, model)
		}
		internaltest.TestClient_ChatProvider_errors(t, f, data)
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

func (l *lazyServer) getClient(t *testing.T, model string) genai.ChatProvider {
	serverURL, transport := l.shouldStart(t)
	c, err := ollama.New(serverURL, model, nil)
	if err != nil {
		t.Fatal(err)
	}
	c.ClientJSON.Client = &http.Client{Transport: transport}
	return c
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
