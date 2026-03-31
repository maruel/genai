// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package claudecode

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
// When RECORD=1, it runs the real claude subprocess and tees its stdout to the
// fixture file so the recording can be replayed later.  Otherwise it reads
// the fixture file directly without running any subprocess.
type recordingExecutor struct {
	fixture string       // path to .ndjson fixture file
	real    *cmdExecutor // non-nil only in record mode
}

// newRecordingExecutor returns an executor whose fixture file is at
// testdata/<name>.ndjson.  If RECORD is "all" or "failure_only" and a real
// claude binary is available, the executor will record a fresh trace;
// otherwise it replays the existing fixture.
//
// In "failure_only" mode the existing fixture is replayed first; recording
// only happens when the fixture is missing or empty.
func newRecordingExecutor(t testing.TB, name string) *recordingExecutor {
	t.Helper()
	fixture := filepath.Join("testdata", name+".ndjson")
	e := &recordingExecutor{fixture: fixture}
	rec := os.Getenv("RECORD")
	if rec == "all" || rec == "failure_only" {
		bin, err := findClaude()
		if err != nil {
			t.Skipf("RECORD=%s but claude not found: %v", rec, err)
		}
		if rec == "all" {
			e.real = &cmdExecutor{bin: bin}
		} else {
			// failure_only: only record when the fixture is missing or empty.
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
	// Tee the subprocess stdout to the fixture file while it streams.
	tee := io.TeeReader(stdout, f)
	rc := &teeReadCloser{Reader: tee, file: f, orig: stdout}
	return stdin, rc, wait, nil
}

// teeReadCloser closes both the fixture file and the original ReadCloser.
type teeReadCloser struct {
	io.Reader
	file *os.File
	orig io.ReadCloser
}

func (t *teeReadCloser) Close() error {
	return errors.Join(t.orig.Close(), t.file.Close())
}

// findClaude returns the path to the claude binary, mirroring the logic in New.
func findClaude() (string, error) {
	return exec.LookPath("claude")
}

// newTestClient creates a Client whose subprocess is backed by a
// recordingExecutor for the given scenario name.
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
		c, err := New()
		if err != nil {
			t.Skipf("claudecode: %v", err)
		}
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
		// Ensure the recordings directory exists for the Stale Recordings check.
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
				t.Skipf("claudecode: %v", err)
			}
			if fn != nil {
				// Let the smoketest framework create HTTP cassettes for bookkeeping
				// (stale recording detection). Claudecode is subprocess-based so
				// the cassettes will be empty; the real data lives in .ndjson fixtures.
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
			{genai.ModelCheap, "haiku"},
			{genai.ModelGood, "sonnet"},
			{genai.ModelSOTA, "opus"},
			{"claude-sonnet-4-6", "claude-sonnet-4-6"},
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
			c := newTestClient(t, "GenSync_hello", genai.ProviderOptionModel("claude-sonnet-4-6"))
			msgs := genai.Messages{genai.NewTextMessage("say hello")}
			res, err := c.GenSync(t.Context(), msgs)
			if err != nil {
				t.Fatalf("GenSync: %v", err)
			}
			if len(res.Replies) == 0 {
				t.Fatal("expected at least one reply")
			}
			got := res.Replies[0].Text
			if !strings.Contains(got, "Hello") {
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
		t.Run("session_id_always_in_opaque", func(t *testing.T) {
			c := newTestClient(t, "GenSync_hello")
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
			c := newTestClient(t, "GenSync_session")
			prevAssistant := genai.Message{
				Replies: []genai.Reply{
					{Text: "Hello!"},
					{Opaque: map[string]any{sessionIDKey: "persist-session-xyz"}},
				},
			}
			msgs := genai.Messages{
				genai.NewTextMessage("hello"),
				prevAssistant,
				genai.NewTextMessage("what did I say?"),
			}
			res, err := c.GenSync(t.Context(), msgs)
			if err != nil {
				t.Fatalf("GenSync: %v", err)
			}
			if len(res.Replies) == 0 {
				t.Fatal("expected at least one reply")
			}
			if !strings.Contains(res.Replies[0].Text, "hello") {
				t.Errorf("unexpected reply: %q", res.Replies[0].Text)
			}
		})
		t.Run("error_result", func(t *testing.T) {
			c := newTestClient(t, "GenSync_error")
			msgs := genai.Messages{genai.NewTextMessage("cause error")}
			_, err := c.GenSync(t.Context(), msgs)
			if err == nil {
				t.Fatal("expected error, got nil")
			}
			if !strings.Contains(err.Error(), "claude error") {
				t.Errorf("unexpected error message: %v", err)
			}
		})
	})
	t.Run("gen_stream", func(t *testing.T) {
		t.Run("hello", func(t *testing.T) {
			c := newTestClient(t, "GenStream_hello", genai.ProviderOptionModel("claude-sonnet-4-6"))
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
			if !strings.Contains(got, "Hello") {
				t.Errorf("streamed text: got %q, want something containing Hello", got)
			}
			if res.Usage.InputTokens == 0 {
				t.Error("InputTokens: got 0, want > 0")
			}
			if res.Usage.OutputTokens == 0 {
				t.Error("OutputTokens: got 0, want > 0")
			}
			if len(res.Replies) == 0 || !strings.Contains(res.Replies[0].Text, "Hello") {
				t.Errorf("Result text: got %v", res.Replies)
			}
		})
		t.Run("thinking_delta", func(t *testing.T) {
			c := newTestClient(t, "GenStream_thinking", genai.ProviderOptionModel("claude-opus-4-6"))
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
			if !strings.Contains(text.String(), "Hello") {
				t.Errorf("streamed text: got %q, want something containing Hello", text.String())
			}
			if !strings.Contains(reasoning.String(), "think") {
				t.Errorf("streamed reasoning: got %q, want something containing think", reasoning.String())
			}
			// Verify result includes both text and reasoning from assistant msg.
			var hasText, hasReasoning bool
			for _, r := range res.Replies {
				if strings.Contains(r.Text, "Hello") {
					hasText = true
				}
				if strings.Contains(r.Reasoning, "think") {
					hasReasoning = true
				}
			}
			if !hasText {
				t.Error("result missing text reply")
			}
			if !hasReasoning {
				t.Error("result missing reasoning reply")
			}
			// Verify duration metadata in opaque.
			for _, r := range res.Replies {
				if r.Opaque != nil {
					if _, ok := r.Opaque["duration_ms"]; ok {
						return
					}
				}
			}
			t.Error("result missing duration_ms in opaque")
		})
		t.Run("error_event", func(t *testing.T) {
			c := newTestClient(t, "GenStream_error_event")
			msgs := genai.Messages{genai.NewTextMessage("hello")}
			seq, finish := c.GenStream(t.Context(), msgs)
			for range seq {
			}
			_, err := finish()
			if err == nil {
				t.Fatal("expected error from stream error event")
			}
			if !strings.Contains(err.Error(), "stream error") {
				t.Errorf("unexpected error: %v", err)
			}
		})
	})

	t.Run("ListModels", func(t *testing.T) {
		c := newTestClient(t, "GenSync_hello")
		models, err := c.ListModels(t.Context())
		if err != nil {
			t.Fatalf("ListModels: %v", err)
		}
		if len(models) == 0 {
			t.Fatal("expected at least one model")
		}
		for _, m := range models {
			if m.GetID() == "" {
				t.Error("model with empty ID")
			}
		}
	})
}

func TestBuildArgs(t *testing.T) {
	t.Run("defaults", func(t *testing.T) {
		c := &Client{model: ""}
		args := c.buildArgs(callOpts{}, "", false)
		want := []string{
			"-p",
			"--verbose",
			"--input-format", "stream-json",
			"--output-format", "stream-json",
			"--strict-mcp-config",
			"--tools", "",
			"--disable-slash-commands",
			"--setting-sources", "project,local",
			"--no-session-persistence",
		}
		if !slices.Equal(args, want) {
			t.Errorf("got  %v\nwant %v", args, want)
		}
	})
	t.Run("with_tools", func(t *testing.T) {
		c := &Client{model: "sonnet"}
		co := callOpts{tools: []string{"Bash", "Read"}, permissionMode: "bypassPermissions"}
		args := c.buildArgs(co, "", false)
		check := func(flag, val string) {
			t.Helper()
			for i, a := range args {
				if a == flag && i+1 < len(args) && args[i+1] == val {
					return
				}
			}
			t.Errorf("flag %q %q not found in args %v", flag, val, args)
		}
		check("--tools", "Bash,Read")
		check("--permission-mode", "bypassPermissions")
		check("--model", "sonnet")
	})
	t.Run("with_session_resume", func(t *testing.T) {
		c := &Client{}
		args := c.buildArgs(callOpts{}, "my-session-id", false)
		for _, a := range args {
			if a == "--no-session-persistence" {
				t.Error("--no-session-persistence must not appear when resuming")
			}
		}
		check := func(flag, val string) {
			t.Helper()
			for i, a := range args {
				if a == flag && i+1 < len(args) && args[i+1] == val {
					return
				}
			}
			t.Errorf("flag %q %q not found in args %v", flag, val, args)
		}
		check("--resume", "my-session-id")
	})
	t.Run("streaming", func(t *testing.T) {
		c := &Client{}
		args := c.buildArgs(callOpts{}, "", true)
		if !slices.Contains(args, "--include-partial-messages") {
			t.Error("--include-partial-messages not found in streaming args")
		}
	})
	t.Run("with_system_prompt", func(t *testing.T) {
		c := &Client{}
		co := callOpts{systemPrompt: "Be helpful"}
		args := c.buildArgs(co, "", false)
		check := func(flag, val string) {
			t.Helper()
			for i, a := range args {
				if a == flag && i+1 < len(args) && args[i+1] == val {
					return
				}
			}
			t.Errorf("flag %q %q not found in args %v", flag, val, args)
		}
		check("--system-prompt", "Be helpful")
	})
}

func TestExtractSessionID(t *testing.T) {
	t.Run("found", func(t *testing.T) {
		msgs := genai.Messages{
			genai.NewTextMessage("hi"),
			{Replies: []genai.Reply{
				{Text: "Hello"},
				{Opaque: map[string]any{sessionIDKey: "abc-123"}},
			}},
		}
		if got := extractSessionID(msgs); got != "abc-123" {
			t.Errorf("got %q, want %q", got, "abc-123")
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

func TestWriteUserMsg(t *testing.T) {
	t.Run("text_only", func(t *testing.T) {
		msg := genai.NewTextMessage("hello")
		var buf strings.Builder
		if err := writeUserMsg(&buf, &msg); err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if !strings.Contains(buf.String(), `"hello"`) {
			t.Errorf("expected plain string content, got %q", buf.String())
		}
	})
	t.Run("image_url", func(t *testing.T) {
		msg := genai.Message{
			Requests: []genai.Request{
				{Text: "describe this"},
				{Doc: genai.Doc{URL: "https://example.com/img.png"}},
			},
		}
		var buf strings.Builder
		if err := writeUserMsg(&buf, &msg); err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		out := buf.String()
		if !strings.Contains(out, `"type":"image"`) {
			t.Errorf("expected image block, got %q", out)
		}
		if !strings.Contains(out, `"url":"https://example.com/img.png"`) {
			t.Errorf("expected url source, got %q", out)
		}
	})
	t.Run("empty_message", func(t *testing.T) {
		msg := genai.NewTextMessage("")
		var buf strings.Builder
		if err := writeUserMsg(&buf, &msg); err == nil {
			t.Fatal("expected error for empty message")
		}
	})
}

func TestGenOption(t *testing.T) {
	t.Run("valid", func(t *testing.T) {
		for _, m := range []string{"acceptEdits", "bypassPermissions", "default", "dontAsk", "plan"} {
			if err := (&GenOption{PermissionMode: m}).Validate(); err != nil {
				t.Errorf("mode %q: unexpected error: %v", m, err)
			}
		}
	})
	t.Run("errors", func(t *testing.T) {
		if err := (&GenOption{Tools: []string{""}}).Validate(); err == nil {
			t.Error("expected error for empty tool name")
		}
		if err := (&GenOption{MaxBudgetUSD: -1}).Validate(); err == nil {
			t.Error("expected error for negative budget")
		}
		if err := (&GenOption{PermissionMode: "hack"}).Validate(); err == nil {
			t.Error("expected error for invalid mode")
		}
	})
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

func TestScoreboard(t *testing.T) {
	// Verify the embedded scoreboard.json parses correctly.
	s := Scoreboard()
	if len(s.Scenarios) == 0 {
		t.Fatal("scoreboard has no scenarios")
	}
}
