// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Tests for the Codex provider client.

package codex

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"io/fs"
	"net/http"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/maruel/genai"
	"github.com/maruel/genai/internal/internaltest"
	"github.com/maruel/genai/internal/msgutil"
	"github.com/maruel/genai/internal/myrecorder"
	"github.com/maruel/genai/scoreboard"
	"github.com/maruel/genai/smoke/smoketest"
)

func newTestClient(t *testing.T, name string, opts ...genai.ProviderOption) *Client {
	rec := internaltest.NewSubprocessRecorder(t, name, "codex", sanitizeCodexFixtureLine)
	opts = append(opts, genai.ProviderOptionStarterWrapper(rec.Wrap))
	c, err := New(opts...)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return c
}

func newOutputClient(t *testing.T, lines ...string) *Client {
	starter := genai.ProviderOptionStarterWrapper(func(genai.Starter) genai.Starter {
		return func(_ context.Context, _ []string) (io.WriteCloser, io.ReadCloser, func() error, error) {
			pr, pw := io.Pipe()
			go func() { _, _ = io.Copy(io.Discard, pr) }()
			return pw, io.NopCloser(strings.NewReader(strings.Join(lines, "\n"))), func() error { return nil }, nil
		}
	})
	c, err := New(starter)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return c
}

func sanitizeCodexFixtureLine(line []byte) ([]byte, error) {
	d := json.NewDecoder(bytes.NewReader(line))
	d.UseNumber()
	var value any
	if err := d.Decode(&value); err != nil {
		return nil, err
	}
	sanitizeCodexFixtureValue(value)
	return json.Marshal(value)
}

func sanitizeCodexFixtureValue(value any) {
	switch value := value.(type) {
	case map[string]any:
		delete(value, "installationId")
		delete(value, "serverName")
		for key, child := range value {
			if path, ok := child.(string); ok {
				value[key] = sanitizeHostHome(path)
				continue
			}
			sanitizeCodexFixtureValue(child)
		}
	case []any:
		for i, child := range value {
			if path, ok := child.(string); ok {
				value[i] = sanitizeHostHome(path)
				continue
			}
			sanitizeCodexFixtureValue(child)
		}
	}
}

func sanitizeHostHome(value string) string {
	for _, prefix := range []string{"/home/", "/Users/"} {
		if !strings.HasPrefix(value, prefix) {
			continue
		}
		rest := strings.TrimPrefix(value, prefix)
		if i := strings.IndexByte(rest, '/'); i >= 0 {
			return "$HOME" + rest[i:]
		}
	}
	return value
}

func setupCodexSmokeHome(t testing.TB) (string, string) {
	src := os.Getenv("CODEX_HOME")
	if src == "" {
		h, err := os.UserHomeDir()
		if err != nil {
			t.Fatalf("find user home: %v", err)
		}
		src = filepath.Join(h, ".codex")
	}
	home := t.TempDir()
	codexHome := filepath.Join(home, ".codex")
	if err := os.MkdirAll(codexHome, 0o700); err != nil {
		t.Fatalf("create temp CODEX_HOME: %v", err)
	}
	for _, name := range []string{"auth.json", ".credentials.json", "config.toml"} {
		copyCodexSmokeFile(t, src, codexHome, name)
	}
	t.Setenv("HOME", home)
	t.Setenv("CODEX_HOME", codexHome)
	return home, codexHome
}

func copyCodexSmokeFile(t testing.TB, src, dst, name string) {
	data, err := os.ReadFile(filepath.Join(src, name))
	if os.IsNotExist(err) {
		return
	}
	if err != nil {
		t.Fatalf("read Codex %s: %v", name, err)
	}
	if err := os.WriteFile(filepath.Join(dst, name), data, 0o600); err != nil {
		t.Fatalf("write temp Codex %s: %v", name, err)
	}
}

func TestSanitizeCodexFixtureLine(t *testing.T) {
	input := []byte(`{"method":"remoteControl/status/changed","params":{"serverName":"host","installationId":"uuid","runtimeWorkspaceRoots":["/tmp/work","/home/maruel/.cache/go-build"]}}`)
	got, err := sanitizeCodexFixtureLine(input)
	if err != nil {
		t.Fatal(err)
	}
	for _, secret := range [][]byte{[]byte("serverName"), []byte("installationId"), []byte("/home/maruel")} {
		if bytes.Contains(got, secret) {
			t.Errorf("sanitized fixture contains %q: %s", secret, got)
		}
	}
	if !bytes.Contains(got, []byte(`"$HOME/.cache/go-build"`)) {
		t.Errorf("sanitized fixture = %s, want normalized home", got)
	}
}

func TestRecordedFixturesSanitized(t *testing.T) {
	var fixtures []string
	err := filepath.WalkDir("testdata", func(path string, entry fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if !entry.IsDir() && filepath.Ext(path) == ".ndjson" {
			fixtures = append(fixtures, path)
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	for _, fixture := range fixtures {
		data, err := os.ReadFile(fixture)
		if err != nil {
			t.Fatal(err)
		}
		for _, secret := range [][]byte{[]byte(`"serverName"`), []byte(`"installationId"`), []byte("/home/"), []byte("/Users/")} {
			if bytes.Contains(data, secret) {
				t.Errorf("%s contains host identifier %q", fixture, secret)
			}
		}
	}
}

func TestGenStreamErrors(t *testing.T) {
	prefix := []string{
		`{"id":1,"result":{"userAgent":"genai-codex/0.154.0"}}`,
		`{"id":2,"result":{"data":[],"nextCursor":null}}`,
		`{"id":3,"result":{"thread":{"id":"thread"}}}`,
	}
	for _, tc := range []struct {
		name string
		line string
		want string
	}{
		{
			name: "error notification",
			line: `{"method":"error","params":{"error":{"message":"internal server error","codexErrorInfo":null,"additionalDetails":null,"misalignment":null},"willRetry":false,"threadId":"thread","turnId":"turn"}}`,
			want: "codex error: internal server error",
		},
		{
			name: "server request",
			line: `{"id":7,"method":"item/tool/requestUserInput","params":{"threadId":"thread","turnId":"turn","itemId":"item","questions":[]}}`,
			want: `unsupported server request "item/tool/requestUserInput" (id 7)`,
		},
		{
			name: "JSON-RPC error response",
			line: `{"id":100,"error":{"code":-32000,"message":"turn failed"}}`,
			want: "JSON-RPC error -32000: turn failed",
		},
		{
			name: "malformed JSON",
			line: `{"method":"turn/completed"`,
			want: "decode app-server message",
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			lines := append(slices.Clone(prefix), tc.line)
			c := newOutputClient(t, lines...)
			seq, finish := c.GenStream(t.Context(), genai.Messages{genai.NewTextMessage("hello")})
			for range seq {
			}
			_, err := finish()
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("error = %v, want %q", err, tc.want)
			}
		})
	}
}

func TestClient(t *testing.T) {
	home, codexHome := setupCodexSmokeHome(t)
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

	t.Run("isolated_smoke_home", func(t *testing.T) {
		if os.Getenv("HOME") != home {
			t.Fatalf("HOME = %q, want %q", os.Getenv("HOME"), home)
		}
		if os.Getenv("CODEX_HOME") != codexHome {
			t.Fatalf("CODEX_HOME = %q, want %q", os.Getenv("CODEX_HOME"), codexHome)
		}
		if filepath.Dir(codexHome) != home {
			t.Fatalf("CODEX_HOME %q is not under HOME %q", codexHome, home)
		}
		for _, name := range []string{"AGENTS.md", "AGENTS.override.md"} {
			if _, err := os.Stat(filepath.Join(codexHome, name)); !os.IsNotExist(err) {
				t.Fatalf("%s copied unexpectedly: %v", name, err)
			}
		}
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
					r := internaltest.NewSubprocessRecorder(t, name, "codex", sanitizeCodexFixtureLine)
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

	t.Run("model_mapping", func(t *testing.T) {
		cases := []struct {
			opt  genai.ProviderOptionModel
			want string
		}{
			{genai.ModelCheap, "gpt-5.6-luna"},
			{genai.ModelGood, "gpt-5.6-terra"},
			{genai.ModelSOTA, "gpt-5.6-sol"},
			{"gpt-5.6-terra", "gpt-5.6-terra"},
		}
		for _, tc := range cases {
			t.Run(string(tc.opt), func(t *testing.T) {
				c, err := New(tc.opt)
				if err != nil {
					t.Fatalf("New: %v", err)
				}
				if got := c.ModelID(); got != tc.want {
					t.Errorf("got %q, want %q", got, tc.want)
				}
			})
		}
	})

	t.Run("gen_sync", func(t *testing.T) {
		t.Run("hello", func(t *testing.T) {
			c := newTestClient(t, "GenSync_hello", genai.ProviderOptionModel("gpt-5.6-terra"))
			msgs := genai.Messages{genai.NewTextMessage("say hello")}
			res, err := c.GenSync(t.Context(), msgs)
			if err != nil {
				t.Fatalf("GenSync: %v", err)
			}
			if len(res.Replies) == 0 {
				t.Fatal("expected at least one reply")
			}
			got := res.Replies[0].Text
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
			c1 := newTestClient(t, "GenSync_session_turn1", genai.ProviderOptionModel("gpt-5.6-terra"))
			msgs1 := genai.Messages{genai.NewTextMessage("Remember this secret code: blue-fox-42. Just confirm you noted it.")}
			res1, err := c1.GenSync(t.Context(), msgs1)
			if err != nil {
				t.Fatalf("turn 1: %v", err)
			}
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
			c2 := newTestClient(t, "GenSync_session_turn2", genai.ProviderOptionModel("gpt-5.6-terra"))
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
	})

	t.Run("gen_stream", func(t *testing.T) {
		t.Run("hello", func(t *testing.T) {
			c := newTestClient(t, "GenStream_hello", genai.ProviderOptionModel("gpt-5.6-terra"))
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
			c := newTestClient(t, "GenStream_thinking", genai.ProviderOptionModel("gpt-5.6-sol"))
			msgs := genai.Messages{genai.NewTextMessage("Think carefully: is 104729 prime? Explain briefly, then say hello.")}
			seq, finish := c.GenStream(t.Context(), msgs)

			var text strings.Builder
			for r := range seq {
				text.WriteString(r.Text)
			}
			res, err := finish()
			if err != nil {
				t.Fatalf("finish: %v", err)
			}
			if !strings.Contains(strings.ToLower(text.String()), "hello") {
				t.Errorf("streamed text: got %q, want something containing hello", text.String())
			}
			var hasText bool
			for _, r := range res.Replies {
				if strings.Contains(strings.ToLower(r.Text), "hello") {
					hasText = true
				}
			}
			if !hasText {
				t.Errorf("result missing text reply")
			}
			if res.Usage.ReasoningTokens == 0 {
				t.Error("ReasoningTokens: got 0, want > 0")
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
			if m.GetID() == "gpt-5.6-sol" {
				found = true
			}
		}
		if !found {
			ids := make([]string, len(models))
			for i, m := range models {
				ids[i] = m.GetID()
			}
			t.Errorf("gpt-5.6-sol not found in models: %v", ids)
		}
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
		if got := msgutil.ExtractOpaqueID(msgs, threadIDKey); got != "abc-123" {
			t.Errorf("got %q, want %q", got, "abc-123")
		}
	})
	t.Run("not_found", func(t *testing.T) {
		msgs := genai.Messages{genai.NewTextMessage("hi")}
		if got := msgutil.ExtractOpaqueID(msgs, threadIDKey); got != "" {
			t.Errorf("got %q, want empty", got)
		}
	})
}

func TestScoreboard(t *testing.T) {
	s := Scoreboard()
	if len(s.Scenarios) == 0 {
		t.Fatal("scoreboard has no scenarios")
	}
}
