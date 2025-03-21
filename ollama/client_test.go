// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package ollama_test

import (
	"context"
	"net/http"
	"net/url"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/maruel/genai"
	"github.com/maruel/genai/ollama"
	"gopkg.in/dnaeon/go-vcr.v4/pkg/cassette"
	"gopkg.in/dnaeon/go-vcr.v4/pkg/recorder"
)

func TestNew(t *testing.T) {
	ctx := context.Background()
	srv, err := startServer(ctx)
	if err != nil {
		t.Fatal(err)
	}
	defer srv.Close()
	c, err := ollama.New(srv.URL(), "gemma3:1b")
	if err != nil {
		t.Fatal(err)
	}
	c.Client.Client = &http.Client{Transport: ignorePort(t)}

	msgs := genai.Messages{
		genai.NewTextMessage(genai.User, "Say hello. Use only one word."),
	}
	opts := genai.ChatOptions{
		Seed:        1,
		Temperature: 0.01,
		MaxTokens:   50,
	}
	got, err := c.Chat(ctx, msgs, &opts)
	if err != nil {
		t.Fatal(err)
	}
	want := genai.NewTextMessage(genai.Assistant, "Hello.")
	if diff := cmp.Diff(want, got.Message); diff != "" {
		t.Fatalf("unexpected response (-want +got):\n%s", diff)
	}
}

// ignorePort ignores the port number in the URL both for recording and playback.
//
// This is important because we start the ollama server on a random port.
func ignorePort(t *testing.T) *recorder.Recorder {
	m := cassette.NewDefaultMatcher()
	fnMatch := func(r *http.Request, i cassette.Request) bool {
		r = r.Clone(r.Context())
		r.URL.Host = strings.Split(r.URL.Host, ":")[0]
		r.Host = strings.Split(r.Host, ":")[0]
		return m(r, i)
	}
	fnSave := func(i *cassette.Interaction) error {
		i.Request.Host = strings.Split(i.Request.Host, ":")[0]
		u, err := url.Parse(i.Request.URL)
		if err != nil {
			t.Fatal(err)
		}
		u.Host = strings.Split(u.Host, ":")[0]
		i.Request.URL = u.String()
		return nil
	}
	r, err := recorder.New("testdata/"+strings.ReplaceAll(t.Name(), "/", "_"), recorder.WithMatcher(fnMatch), recorder.WithHook(fnSave, recorder.AfterCaptureHook))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := r.Stop(); err != nil {
			t.Error(err)
		}
	})

	return r
}
