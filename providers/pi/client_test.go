// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package pi

import (
	"net/http"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/maruel/genai"
	"github.com/maruel/genai/internal/internaltest"
	"github.com/maruel/genai/internal/myrecorder"
	"github.com/maruel/genai/scoreboard"
	"github.com/maruel/genai/smoke/smoketest"
)

func newTestClient(t *testing.T, name string, opts ...genai.ProviderOption) *Client {
	rec := internaltest.NewSubprocessRecorder(t, name, "pi")
	opts = append(opts, genai.ProviderOptionStarterWrapper(rec.Wrap))
	c, err := New(opts...)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return c
}

func TestClient(t *testing.T) {
	testRecorder := internaltest.NewRecords()
	t.Cleanup(func() {
		if err := testRecorder.Close(); err != nil {
			t.Error(err)
		}
	})

	t.Run("Capabilities", func(t *testing.T) {
		c := newTestClient(t, "TestClient_Capabilities")
		internaltest.TestCapabilities(t, c)
	})

	t.Run("Scoreboard", func(t *testing.T) {
		c := newTestClient(t, "ListModels")
		genaiModels, err := c.ListModels(t.Context())
		if err != nil {
			t.Fatal(err)
		}
		scenarios := c.Scoreboard().Scenarios
		models := make([]scoreboard.Model, 0, len(genaiModels))
		for _, m := range genaiModels {
			id := m.GetID()
			reason := false
			for _, sc := range scenarios {
				if slices.Contains(sc.Models, id) {
					reason = sc.Reason
					break
				}
			}
			models = append(models, scoreboard.Model{Model: id, Reason: reason})
		}
		if err := os.MkdirAll(filepath.Join("testdata", "TestClient", "Scoreboard"), 0o755); err != nil {
			t.Fatal(err)
		}
		getClientRT := func(t testing.TB, model scoreboard.Model, fn func(http.RoundTripper) http.RoundTripper) genai.Provider {
			var opts []genai.ProviderOption
			if model.Model != "" {
				opts = append(opts, genai.ProviderOptionModel(model.Model))
			}
			if fn != nil {
				wrapped := fn(http.DefaultTransport)
				if rec, ok := wrapped.(*myrecorder.Recorder); ok {
					name := strings.TrimSuffix(rec.Name(), ".yaml")
					r := internaltest.NewSubprocessRecorder(t, name, "pi")
					opts = append(opts, genai.ProviderOptionStarterWrapper(r.Wrap))
				}
			}
			c, err := New(opts...)
			if err != nil {
				t.Fatal(err)
			}
			return c
		}
		smoketest.Run(t, getClientRT, models, testRecorder.Records, nil)
	})

	t.Run("gen_sync", func(t *testing.T) {
		t.Run("hello", func(t *testing.T) {
			c := newTestClient(t, "GenSync_hello", genai.ProviderOptionModel("cerebras/gpt-oss-120b"))
			msgs := genai.Messages{genai.NewTextMessage("say hello")}
			res, err := c.GenSync(t.Context(), msgs)
			if err != nil {
				t.Fatalf("GenSync: %v", err)
			}
			if len(res.Replies) == 0 {
				t.Fatal("expected at least one reply")
			}
			var got string
			for _, r := range res.Replies {
				if r.Text != "" {
					got = r.Text
					break
				}
			}
			if got == "" {
				t.Error("expected non-empty reply text")
			}
			if res.Usage.InputTokens == 0 {
				t.Error("InputTokens: got 0, want > 0")
			}
			if res.Usage.OutputTokens == 0 {
				t.Error("OutputTokens: got 0, want > 0")
			}
			if res.Usage.FinishReason != genai.FinishedStop {
				t.Errorf("FinishReason: got %q, want %q", res.Usage.FinishReason, genai.FinishedStop)
			}
		})
	})

	t.Run("gen_stream", func(t *testing.T) {
		t.Run("hello", func(t *testing.T) {
			c := newTestClient(t, "GenStream_hello", genai.ProviderOptionModel("cerebras/gpt-oss-120b"))
			msgs := genai.Messages{genai.NewTextMessage("say hello")}
			seq, finish := c.GenStream(t.Context(), msgs)

			var sb strings.Builder
			for r := range seq {
				sb.WriteString(r.Text)
			}
			res, err := finish()
			if err != nil {
				t.Fatalf("finish: %v", err)
			}

			got := sb.String()
			if !strings.Contains(strings.ToLower(got), "hello") {
				t.Errorf("streamed text: got %q, want something containing hello", got)
			}
			if res.Usage.InputTokens == 0 {
				t.Error("InputTokens: got 0, want > 0")
			}
			if res.Usage.OutputTokens == 0 {
				t.Error("OutputTokens: got 0, want > 0")
			}
			var hasResultText bool
			for _, r := range res.Replies {
				if strings.Contains(strings.ToLower(r.Text), "hello") {
					hasResultText = true
					break
				}
			}
			if !hasResultText {
				t.Error("result missing text reply containing hello")
			}
		})
		t.Run("thinking_delta", func(t *testing.T) {
			c := newTestClient(t, "GenStream_thinking", genai.ProviderOptionModel("cerebras/gpt-oss-120b"))
			msgs := genai.Messages{genai.NewTextMessage("say hello")}
			seq, finish := c.GenStream(t.Context(), msgs)

			var text, reasoning strings.Builder
			for r := range seq {
				text.WriteString(r.Text)
				reasoning.WriteString(r.Reasoning)
			}
			res, err := finish()
			if err != nil {
				t.Fatalf("finish: %v", err)
			}
			if !strings.Contains(strings.ToLower(text.String()), "hello") {
				t.Errorf("streamed text: got %q, want something containing hello", text.String())
			}
			if reasoning.Len() == 0 {
				t.Errorf("streamed reasoning: got empty, want non-empty")
			}
			var hasText, hasReasoning bool
			for _, r := range res.Replies {
				if strings.Contains(strings.ToLower(r.Text), "hello") {
					hasText = true
				}
				if r.Reasoning != "" {
					hasReasoning = true
				}
			}
			if !hasText {
				t.Errorf("result missing text reply")
			}
			if !hasReasoning {
				t.Errorf("result missing reasoning reply")
			}
		})
	})

	t.Run("ListModels", func(t *testing.T) {
		c := newTestClient(t, "ListModels")
		models, err := c.ListModels(t.Context())
		if err != nil {
			t.Fatalf("ListModels: %v", err)
		}
		if len(models) == 0 {
			t.Fatal("expected at least one model")
		}
	})
}

func TestScoreboard(t *testing.T) {
	s := Scoreboard()
	if s.Scenarios == nil {
		t.Fatal("scoreboard scenarios is nil")
	}
}
