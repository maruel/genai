// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package opencode

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
// When RECORD is set, it runs the real opencode subprocess and tees its stdout
// to the fixture file. Otherwise it reads the fixture file directly.
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
		bin, err := findOpenCode()
		if err != nil {
			t.Skipf("RECORD=%s but opencode not found: %v", rec, err)
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

func findOpenCode() (string, error) {
	return exec.LookPath("opencode")
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
				t.Skipf("opencode: %v", err)
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
			{genai.ModelCheap, "opencode/gpt-5-nano"},
			{genai.ModelGood, "opencode/big-pickle"},
			{genai.ModelSOTA, "openai/gpt-5.4/xhigh"},
			{"opencode/big-pickle", "opencode/big-pickle"},
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
			c := newTestClient(t, "GenSync_hello", genai.ProviderOptionModel("opencode/big-pickle"))
			msgs := genai.Messages{genai.NewTextMessage("say hello")}
			res, err := c.GenSync(t.Context(), msgs)
			if err != nil {
				t.Fatalf("GenSync: %v", err)
			}
			if len(res.Replies) == 0 {
				t.Fatal("expected at least one reply")
			}
			found := false
			for _, r := range res.Replies {
				if strings.Contains(strings.ToLower(r.Text), "hello") {
					found = true
					break
				}
			}
			if !found {
				t.Errorf("no reply contains 'hello': %v", res.Replies)
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
		t.Run("session_id_always_in_opaque", func(t *testing.T) {
			c := newTestClient(t, "GenSync_hello_nomodel")
			msgs := genai.Messages{genai.NewTextMessage("hello")}
			res, err := c.GenSync(t.Context(), msgs)
			if err != nil {
				t.Fatalf("GenSync: %v", err)
			}
			var found string
			for _, r := range res.Replies {
				if id, ok := r.Opaque[sessionIDKey].(string); ok {
					found = id
				}
			}
			if found == "" {
				t.Fatal("session_id not found in Reply.Opaque")
			}
		})
		t.Run("session_resumed_from_opaque", func(t *testing.T) {
			c1 := newTestClient(t, "GenSync_session_turn1", genai.ProviderOptionModel("opencode/big-pickle"))
			msgs1 := genai.Messages{genai.NewTextMessage("Remember this secret code: blue-fox-42. Just confirm you noted it.")}
			res1, err := c1.GenSync(t.Context(), msgs1)
			if err != nil {
				t.Fatalf("turn 1: %v", err)
			}
			var sessionID string
			for _, r := range res1.Replies {
				if id, ok := r.Opaque[sessionIDKey].(string); ok {
					sessionID = id
				}
			}
			if sessionID == "" {
				t.Fatal("turn 1 did not return a session_id")
			}

			c2 := newTestClient(t, "GenSync_session_turn2", genai.ProviderOptionModel("opencode/big-pickle"))
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
			found := false
			for _, r := range res2.Replies {
				if strings.Contains(strings.ToLower(r.Text), "blue-fox-42") {
					found = true
					break
				}
			}
			if !found {
				t.Errorf("turn 2: expected reply to contain 'blue-fox-42', got %v", res2.Replies)
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
			c := newTestClient(t, "GenStream_hello", genai.ProviderOptionModel("opencode/big-pickle"))
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
			found := false
			for _, r := range res.Replies {
				if strings.Contains(strings.ToLower(r.Text), "hello") {
					found = true
					break
				}
			}
			if !found {
				t.Errorf("Result text: got %v", res.Replies)
			}
		})
		t.Run("thinking_delta", func(t *testing.T) {
			c := newTestClient(t, "GenStream_thinking", genai.ProviderOptionModel("opencode/big-pickle"))
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
				t.Fatal("expected error from error response")
			}
			if !strings.Contains(err.Error(), "internal server error") {
				t.Errorf("unexpected error: %v", err)
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
		var found bool
		for _, m := range models {
			if m.GetID() == "opencode/big-pickle" {
				found = true
			}
		}
		if !found {
			ids := make([]string, len(models))
			for i, m := range models {
				ids[i] = m.GetID()
			}
			t.Errorf("opencode/big-pickle not found in models: %v", ids)
		}
	})
}

func TestExtractSessionID(t *testing.T) {
	t.Run("found", func(t *testing.T) {
		msgs := genai.Messages{
			genai.NewTextMessage("hi"),
			{Replies: []genai.Reply{
				{Text: "Hello"},
				{Opaque: map[string]any{sessionIDKey: "sess-123"}},
			}},
		}
		if got := extractSessionID(msgs); got != "sess-123" {
			t.Errorf("got %q, want %q", got, "sess-123")
		}
	})
	t.Run("not_found", func(t *testing.T) {
		msgs := genai.Messages{genai.NewTextMessage("hi")}
		if got := extractSessionID(msgs); got != "" {
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

func TestGenSyncUnsupportedOpts(t *testing.T) {
	c := newTestClient(t, "GenSync_hello_nomodel")
	msgs := genai.Messages{genai.NewTextMessage("hello")}
	_, err := c.GenSync(t.Context(), msgs, &genai.GenOptionText{SystemPrompt: "Be helpful"})
	var uerr *base.ErrNotSupported
	if !errors.As(err, &uerr) {
		t.Fatalf("expected ErrNotSupported, got %v", err)
	}
}

func TestScoreboard(t *testing.T) {
	s := Scoreboard()
	if len(s.Scenarios) == 0 {
		t.Fatal("scoreboard has no scenarios")
	}
}
