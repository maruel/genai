// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package codex

import (
	"context"
	"errors"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/maruel/genai"
	"github.com/maruel/genai/base"
	"github.com/maruel/genai/internal/internaltest"
	"github.com/maruel/genai/internal/myrecorder"
	"github.com/maruel/genai/scoreboard"
	"github.com/maruel/genai/smoke/smoketest"
)

// recordingExecutor implements the executor interface for tests.
//
// When RECORD is set, it runs the real codex subprocess and tees its stdout to
// the fixture file. Otherwise it reads the fixture file directly.
type recordingExecutor struct {
	fixture string
	real    *cmdExecutor
}

func newRecordingExecutor(t testing.TB, name string) *recordingExecutor {
	t.Helper()
	fixture := filepath.Join("testdata", name+".ndjson")
	e := &recordingExecutor{fixture: fixture}
	rec := os.Getenv("RECORD")
	if rec == "all" || rec == "failure_only" {
		bin, err := findCodex()
		if err != nil {
			t.Skipf("RECORD=%s but codex not found: %v", rec, err)
		}
		if rec == "all" {
			e.real = &cmdExecutor{bin: bin}
		} else {
			info, err := os.Stat(fixture)
			if err != nil || info.Size() == 0 {
				e.real = &cmdExecutor{bin: bin}
			}
		}
	}
	return e
}

func (e *recordingExecutor) start(ctx context.Context, args []string) (io.WriteCloser, io.ReadCloser, func() error, error) {
	if e.real != nil {
		return e.record(ctx, args)
	}
	return e.replay()
}

func (e *recordingExecutor) replay() (io.WriteCloser, io.ReadCloser, func() error, error) {
	data, err := os.ReadFile(e.fixture)
	if err != nil {
		return nil, nil, nil, err
	}
	pr, pw := io.Pipe()
	go func() { _, _ = io.Copy(io.Discard, pr) }()
	return pw, io.NopCloser(strings.NewReader(string(data))), func() error { return nil }, nil
}

func (e *recordingExecutor) record(ctx context.Context, args []string) (io.WriteCloser, io.ReadCloser, func() error, error) {
	stdin, stdout, wait, err := e.real.start(ctx, args)
	if err != nil {
		return nil, nil, nil, err
	}
	if err := os.MkdirAll(filepath.Dir(e.fixture), 0o755); err != nil {
		return nil, nil, nil, err
	}
	f, err := os.Create(e.fixture)
	if err != nil {
		return nil, nil, nil, err
	}
	tee := io.TeeReader(stdout, f)
	rc := &teeReadCloser{Reader: tee, file: f, orig: stdout}
	return stdin, rc, wait, nil
}

type teeReadCloser struct {
	io.Reader
	file *os.File
	orig io.ReadCloser
}

func (t *teeReadCloser) Close() error {
	return errors.Join(t.orig.Close(), t.file.Close())
}

func findCodex() (string, error) {
	return exec.LookPath("codex")
}

func newTestClient(t *testing.T, name string, opts ...genai.ProviderOption) *Client {
	t.Helper()
	c, err := New(opts...)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	c.exec = newRecordingExecutor(t, name)
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
			c, err := New(opts...)
			if err != nil {
				t.Skipf("codex: %v", err)
			}
			if fn != nil {
				wrapped := fn(http.DefaultTransport)
				if rec, ok := wrapped.(*myrecorder.Recorder); ok {
					name := strings.TrimSuffix(rec.Name(), ".yaml")
					c.exec = newRecordingExecutor(t, name)
				}
			}
			return c
		}
		smoketest.Run(t, getClientRT, models, testRecorder.Records)
	})

	t.Run("model_mapping", func(t *testing.T) {
		cases := []struct {
			opt  genai.ProviderOptionModel
			want string
		}{
			{genai.ModelCheap, "gpt-5.1-codex-mini"},
			{genai.ModelGood, "gpt-5.3-codex"},
			{genai.ModelSOTA, "gpt-5.4"},
			{"gpt-5.3-codex", "gpt-5.3-codex"},
		}
		for _, tc := range cases {
			t.Run(string(tc.opt), func(t *testing.T) {
				c, err := New(tc.opt)
				if err != nil {
					t.Fatalf("New: %v", err)
				}
				if c.model != tc.want {
					t.Errorf("got %q, want %q", c.model, tc.want)
				}
			})
		}
	})

	t.Run("gen_sync", func(t *testing.T) {
		t.Run("hello", func(t *testing.T) {
			c := newTestClient(t, "GenSync_hello", genai.ProviderOptionModel("gpt-5.4"))
			msgs := genai.Messages{genai.NewTextMessage("say hello")}
			res, err := c.GenSync(t.Context(), msgs)
			if err != nil {
				t.Fatalf("GenSync: %v", err)
			}
			if len(res.Replies) == 0 {
				t.Fatal("expected at least one reply")
			}
			got := res.Replies[0].Text
			if !strings.Contains(strings.ToLower(got), "hello") {
				t.Errorf("unexpected reply text %q", got)
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
		t.Run("thread_id_always_in_opaque", func(t *testing.T) {
			c := newTestClient(t, "GenSync_hello")
			msgs := genai.Messages{genai.NewTextMessage("hello")}
			res, err := c.GenSync(t.Context(), msgs)
			if err != nil {
				t.Fatalf("GenSync: %v", err)
			}
			var found string
			for _, r := range res.Replies {
				if id, ok := r.Opaque[threadIDKey].(string); ok {
					found = id
				}
			}
			if found == "" {
				t.Fatal("thread_id not found in Reply.Opaque")
			}
		})
		t.Run("thread_resumed_from_opaque", func(t *testing.T) {
			// Turn 1: establish a session with a unique fact.
			c1 := newTestClient(t, "GenSync_session_turn1", genai.ProviderOptionModel("gpt-5.4"))
			msgs1 := genai.Messages{genai.NewTextMessage("Remember this secret code: blue-fox-42. Just confirm you noted it.")}
			res1, err := c1.GenSync(t.Context(), msgs1)
			if err != nil {
				t.Fatalf("turn 1: %v", err)
			}
			// Extract the thread ID from the first reply.
			var threadID string
			for _, r := range res1.Replies {
				if id, ok := r.Opaque[threadIDKey].(string); ok {
					threadID = id
				}
			}
			if threadID == "" {
				t.Fatal("turn 1 did not return a thread_id")
			}

			// Turn 2: resume the session and ask it to recall the fact.
			c2 := newTestClient(t, "GenSync_session_turn2", genai.ProviderOptionModel("gpt-5.4"))
			msgs2 := genai.Messages{
				genai.NewTextMessage("Remember this secret code: blue-fox-42. Just confirm you noted it."),
				{Replies: res1.Replies},
				genai.NewTextMessage("What was the secret code I told you?"),
			}
			res2, err := c2.GenSync(t.Context(), msgs2)
			if err != nil {
				t.Fatalf("turn 2: %v", err)
			}
			if len(res2.Replies) == 0 {
				t.Fatal("turn 2: expected at least one reply")
			}
			got := strings.ToLower(res2.Replies[0].Text)
			if !strings.Contains(got, "blue-fox-42") {
				t.Errorf("turn 2: expected reply to contain 'blue-fox-42', got %q", res2.Replies[0].Text)
			}
		})
		t.Run("error_result", func(t *testing.T) {
			c := newTestClient(t, "GenSync_error")
			msgs := genai.Messages{genai.NewTextMessage("cause error")}
			_, err := c.GenSync(t.Context(), msgs)
			if err == nil {
				t.Fatal("expected error, got nil")
			}
			if !strings.Contains(err.Error(), "rate limit exceeded") {
				t.Errorf("unexpected error message: %v", err)
			}
		})
	})

	t.Run("gen_stream", func(t *testing.T) {
		t.Run("hello", func(t *testing.T) {
			c := newTestClient(t, "GenStream_hello", genai.ProviderOptionModel("gpt-5.4"))
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
			if len(res.Replies) == 0 || !strings.Contains(strings.ToLower(res.Replies[0].Text), "hello") {
				t.Errorf("Result text: got %v", res.Replies)
			}
		})
		t.Run("thinking_delta", func(t *testing.T) {
			c := newTestClient(t, "GenStream_thinking", genai.ProviderOptionModel("gpt-5.4"))
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
		t.Run("error_event", func(t *testing.T) {
			c := newTestClient(t, "GenStream_error_event")
			msgs := genai.Messages{genai.NewTextMessage("hello")}
			seq, finish := c.GenStream(t.Context(), msgs)
			for range seq {
			}
			_, err := finish()
			if err == nil {
				t.Fatal("expected error from error notification")
			}
			if !strings.Contains(err.Error(), "internal server error") {
				t.Errorf("unexpected error: %v", err)
			}
		})
	})
}

func TestExtractThreadID(t *testing.T) {
	t.Run("found", func(t *testing.T) {
		msgs := genai.Messages{
			genai.NewTextMessage("hi"),
			{Replies: []genai.Reply{
				{Text: "Hello"},
				{Opaque: map[string]any{threadIDKey: "abc-123"}},
			}},
		}
		if got := extractThreadID(msgs); got != "abc-123" {
			t.Errorf("got %q, want %q", got, "abc-123")
		}
	})
	t.Run("not_found", func(t *testing.T) {
		msgs := genai.Messages{genai.NewTextMessage("hi")}
		if got := extractThreadID(msgs); got != "" {
			t.Errorf("got %q, want empty", got)
		}
	})
}

func TestLastUserMsg(t *testing.T) {
	t.Run("valid", func(t *testing.T) {
		msgs := genai.Messages{genai.NewTextMessage("hello world")}
		got, err := lastUserMsg(msgs)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if len(got.Requests) == 0 || got.Requests[0].Text != "hello world" {
			t.Errorf("got %v, want text %q", got.Requests, "hello world")
		}
	})
	t.Run("no_user_message", func(t *testing.T) {
		msgs := genai.Messages{{Replies: []genai.Reply{{Text: "hi"}}}}
		if _, err := lastUserMsg(msgs); err == nil {
			t.Fatal("expected error for no user message")
		}
	})
	t.Run("last_user_wins", func(t *testing.T) {
		msgs := genai.Messages{
			genai.NewTextMessage("first"),
			{Replies: []genai.Reply{{Text: "response"}}},
			genai.NewTextMessage("second"),
		}
		got, err := lastUserMsg(msgs)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if len(got.Requests) == 0 || got.Requests[0].Text != "second" {
			t.Errorf("got %v, want text %q", got.Requests, "second")
		}
	})
}

func TestReasoningEffort(t *testing.T) {
	t.Run("valid", func(t *testing.T) {
		for _, v := range []ReasoningEffort{
			ReasoningEffortNone, ReasoningEffortMinimal, ReasoningEffortLow,
			ReasoningEffortMedium, ReasoningEffortHigh, ReasoningEffortXHigh,
		} {
			c, err := New(v)
			if err != nil {
				t.Fatalf("New(%q): %v", v, err)
			}
			if c.effort != v {
				t.Errorf("effort: got %q, want %q", c.effort, v)
			}
		}
	})
	t.Run("invalid", func(t *testing.T) {
		if _, err := New(ReasoningEffort("turbo")); err == nil {
			t.Fatal("expected error for invalid effort")
		}
	})
	t.Run("default", func(t *testing.T) {
		c, err := New()
		if err != nil {
			t.Fatal(err)
		}
		if c.effort != ReasoningEffortMedium {
			t.Errorf("default effort: got %q, want %q", c.effort, ReasoningEffortMedium)
		}
	})
}

func TestParseOpts(t *testing.T) {
	t.Run("system_prompt", func(t *testing.T) {
		co, err := parseOpts([]genai.GenOption{&genai.GenOptionText{SystemPrompt: "Be helpful"}})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if co.systemPrompt != "Be helpful" {
			t.Errorf("systemPrompt: got %q, want %q", co.systemPrompt, "Be helpful")
		}
	})
	t.Run("unsupported", func(t *testing.T) {
		for _, tc := range []struct {
			name string
			opts []genai.GenOption
			want string
		}{
			{"Temperature", []genai.GenOption{&genai.GenOptionText{Temperature: 0.5}}, "GenOptionText.Temperature"},
			{"Seed", []genai.GenOption{genai.GenOptionSeed(42)}, "GenOptionSeed"},
		} {
			t.Run(tc.name, func(t *testing.T) {
				_, err := parseOpts(tc.opts)
				var uerr *base.ErrNotSupported
				if !errors.As(err, &uerr) {
					t.Fatalf("expected ErrNotSupported, got %v", err)
				}
				if !slices.Contains(uerr.Options, tc.want) {
					t.Errorf("expected %q in unsupported, got %v", tc.want, uerr.Options)
				}
			})
		}
	})
}

func TestListModels(t *testing.T) {
	c := newTestClient(t, "ListModels")
	models, err := c.ListModels(t.Context())
	if err != nil {
		t.Fatalf("ListModels: %v", err)
	}
	if len(models) == 0 {
		t.Fatal("expected at least one model")
	}
	// Verify that the default model is present.
	var found bool
	for _, m := range models {
		if m.GetID() == "gpt-5.3-codex" {
			found = true
		}
	}
	if !found {
		ids := make([]string, len(models))
		for i, m := range models {
			ids[i] = m.GetID()
		}
		t.Errorf("gpt-5.3-codex not found in models: %v", ids)
	}
}

func TestScoreboard(t *testing.T) {
	s := Scoreboard()
	if len(s.Scenarios) == 0 {
		t.Fatal("scoreboard has no scenarios")
	}
}
