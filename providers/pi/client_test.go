// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Tests for the Pi provider client.

package pi

import (
	"io"
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

func TestReadUntilDone(t *testing.T) {
	t.Run("retries", func(t *testing.T) {
		input := strings.Join([]string{
			`{"type":"message_update","assistantMessageEvent":{"type":"text_delta","delta":"failed attempt"}}`,
			`{"type":"agent_end","willRetry":true,"messages":[]}`,
			`{"type":"message_update","assistantMessageEvent":{"type":"text_delta","delta":"retry succeeded"}}`,
			`{"type":"agent_end","willRetry":false,"messages":[{"role":"assistant","usage":{"input":10,"output":2,"totalTokens":12},"stopReason":"stop"}]}`,
			`{"type":"agent_settled"}`,
		}, "\n")
		var deltas strings.Builder
		res, err := readUntilDone(newScanner(strings.NewReader(input)), io.Discard, func(text, reasoning string) bool {
			deltas.WriteString(text)
			return true
		})
		if err != nil {
			t.Fatal(err)
		}
		if got := deltas.String(); got != "failed attemptretry succeeded" {
			t.Errorf("streamed text = %q, want both streamed attempts", got)
		}
		if len(res.Replies) != 1 || res.Replies[0].Text != "retry succeeded" {
			t.Errorf("Replies = %#v, want retried response only", res.Replies)
		}
		if res.Usage.TotalTokens != 12 {
			t.Errorf("TotalTokens = %d, want 12", res.Usage.TotalTokens)
		}
	})

	t.Run("waits for settlement", func(t *testing.T) {
		input := strings.Join([]string{
			`{"type":"message_update","assistantMessageEvent":{"type":"text_delta","delta":"first"}}`,
			`{"type":"agent_end","willRetry":false,"messages":[]}`,
			`{"type":"message_update","assistantMessageEvent":{"type":"text_delta","delta":" follow-up"}}`,
			`{"type":"agent_end","willRetry":false,"messages":[{"role":"assistant","usage":{"input":10,"output":3,"totalTokens":13},"stopReason":"stop"}]}`,
			`{"type":"agent_settled"}`,
		}, "\n")
		res, err := readUntilDone(newScanner(strings.NewReader(input)), io.Discard, func(string, string) bool { return true })
		if err != nil {
			t.Fatal(err)
		}
		if len(res.Replies) != 1 || res.Replies[0].Text != "first follow-up" {
			t.Errorf("Replies = %#v, want all settled output", res.Replies)
		}
	})

	t.Run("terminal retry failure", func(t *testing.T) {
		input := strings.Join([]string{
			`{"type":"agent_end","willRetry":false,"messages":[]}`,
			`{"type":"auto_retry_end","success":false,"attempt":3,"finalError":"502 status code"}`,
			`{"type":"agent_settled"}`,
		}, "\n")
		_, err := readUntilDone(newScanner(strings.NewReader(input)), io.Discard, func(string, string) bool { return true })
		if err == nil || err.Error() != "pi auto retry failed: 502 status code" {
			t.Errorf("error = %v, want terminal retry failure", err)
		}
	})
}

func TestBuildResult(t *testing.T) {
	res, err := buildResult([]byte(`{"type":"agent_end","messages":[{"role":"assistant","content":[{"type":"text","text":"Hello."}],"usage":{"input":10,"output":8,"cacheRead":4,"cacheWrite":2,"cacheWrite1h":1,"reasoning":3,"totalTokens":24,"cost":{"input":0,"output":0,"cacheRead":0,"cacheWrite":0,"total":0}},"stopReason":"stop"}]}`), "", "")
	if err != nil {
		t.Fatal(err)
	}
	if res.Usage.ReasoningTokens != 3 {
		t.Errorf("ReasoningTokens = %d, want 3", res.Usage.ReasoningTokens)
	}
}

func TestScoreboard(t *testing.T) {
	s := Scoreboard()
	if s.Scenarios == nil {
		t.Fatal("scoreboard scenarios is nil")
	}
}
