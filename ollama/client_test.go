// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package ollama_test

import (
	"bytes"
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

	t.Run("Chat", func(t *testing.T) {
		serverURL, transport := s.shouldStart(t)
		c, err := ollama.New(serverURL, "gemma3:4b")
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

	t.Run("vision_and_tool", func(t *testing.T) {
		serverURL, transport := s.shouldStart(t)
		c, err := ollama.New(serverURL, "gemma3:4b")
		if err != nil {
			t.Fatal(err)
		}
		c.Client.Client = &http.Client{Transport: transport}
		msgs := genai.Messages{
			{
				Role: genai.User,
				Contents: []genai.Content{
					{Text: "Is it a banana? Reply as JSON."},
					{Filename: "banana.jpg", Document: bytes.NewReader(bananaJpg)},
				},
			},
		}
		var got struct {
			Banana bool `json:"banana"`
		}
		opts := genai.ChatOptions{
			Seed:        1,
			Temperature: 0.01,
			MaxTokens:   50,
			DecodeAs:    &got,
		}
		resp, err := c.Chat(t.Context(), msgs, &opts)
		if err != nil {
			t.Fatal(err)
		}
		t.Logf("Raw response: %#v", resp)
		if resp.InputTokens != 278 || resp.OutputTokens != 11 {
			t.Logf("Unexpected tokens usage: %v", resp.Usage)
		}
		if len(resp.Contents) != 1 {
			t.Fatal("Unexpected response")
		}
		if err := resp.Contents[0].Decode(&got); err != nil {
			t.Fatal(err)
		}
		if !got.Banana {
			t.Fatal("expected a banana")
		}
	})
	t.Run("Tool", func(t *testing.T) {
		serverURL, transport := s.shouldStart(t)
		c, err := ollama.New(serverURL, "llama3.1:8b")
		if err != nil {
			t.Fatal(err)
		}
		c.Client.Client = &http.Client{Transport: transport}
		msgs := genai.Messages{
			genai.NewTextMessage(genai.User, "I wonder if Canada is a better country than the US? Call the tool best_country to tell me which country is the best one."),
		}
		var out struct {
			Country string `json:"country" jsonschema:"enum=Canada,enum=USA"`
		}
		opts := genai.ChatOptions{
			Seed:        1,
			Temperature: 0.01,
			MaxTokens:   50,
			Tools: []genai.ToolDef{
				{
					Name:        "best_country",
					Description: "A tool to determine the best country",
					InputsAs:    &out,
				},
			},
		}
		got, err := c.Chat(t.Context(), msgs, &opts)
		if err != nil {
			t.Fatal(err)
		}
		want := genai.Message{Role: genai.Assistant, ToolCalls: []genai.ToolCall{{Name: "best_country", Arguments: `{"country":"Canada"}`}}}
		if diff := cmp.Diff(want, got.Message); diff != "" {
			t.Fatalf("unexpected response (-want +got):\n%s", diff)
		}
		if got.InputTokens != 188 || got.OutputTokens != 17 {
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
