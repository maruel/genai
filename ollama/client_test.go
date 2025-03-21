// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package ollama_test

import (
	"context"
	"net/http"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/maruel/genai"
	"github.com/maruel/genai/internal/internaltest"
	"github.com/maruel/genai/ollama"
)

func TestNew(t *testing.T) {
	transport := internaltest.Record(t)
	serverURL := "http://localhost:0"
	ctx := context.Background()
	if transport.IsNewCassette() {
		t.Log("Recording")
		srv, err := startServer(ctx)
		if err != nil {
			t.Fatal(err)
		}
		defer srv.Close()
		serverURL = srv.URL()
	}
	c, err := ollama.New(serverURL, "gemma3:1b")
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
	got, err := c.Chat(ctx, msgs, &opts)
	if err != nil {
		t.Fatal(err)
	}
	want := genai.NewTextMessage(genai.Assistant, "Hello.")
	if diff := cmp.Diff(want, got.Message); diff != "" {
		t.Fatalf("unexpected response (-want +got):\n%s", diff)
	}

	msgs = append(msgs, got.Message)
	msgs = append(msgs, genai.NewTextMessage(genai.User, "What is your name?"))
	got, err = c.Chat(ctx, msgs, &opts)
	if err != nil {
		t.Fatal(err)
	}

	want = genai.NewTextMessage(genai.Assistant, "I am a large language model created by Google.")
	if diff := cmp.Diff(want, got.Message); diff != "" {
		t.Fatalf("unexpected response (-want +got):\n%s", diff)
	}
}
