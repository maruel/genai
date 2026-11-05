// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Tests for the OpenCode provider client.

package opencode

import (
	"context"
	"errors"
	"io"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/maruel/genai"
	"github.com/maruel/genai/acp"
	"github.com/maruel/genai/base"
	"github.com/maruel/genai/internal"
	"github.com/maruel/genai/internal/internaltest"
	"github.com/maruel/genai/scoreboard"
	"github.com/maruel/genai/smoke"
	"github.com/maruel/genai/smoke/smoketest"
)

func newTestClient(t *testing.T, name string, opts ...genai.ProviderOption) *Client {
	rec := internaltest.NewSubprocessRecorder(t, name, "opencode", nil)
	opts = append(opts, genai.ProviderOptionStarterWrapper(rec.Wrap))
	c, err := New(t.Context(), opts...)
	if c != nil {
		internaltest.CleanupCloser(t, c)
	}
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return c
}

func newOutputClient(t *testing.T, output string, opts ...genai.ProviderOption) *Client {
	opts = append(opts, genai.ProviderOptionStarterWrapper(func(genai.Starter) genai.Starter {
		return func(_ context.Context, _ []string) (io.WriteCloser, io.ReadCloser, func() error, error) {
			pr, pw := io.Pipe()
			go func() { _, _ = io.Copy(io.Discard, pr) }()
			return pw, io.NopCloser(strings.NewReader(output)), func() error { return nil }, nil
		}
	}))
	c, err := New(t.Context(), opts...)
	if c != nil {
		internaltest.CleanupCloser(t, c)
	}
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
		scenarios := c.Scoreboard().Models()
		models := make([]scoreboard.Model, 0, len(genaiModels))
		for _, m := range genaiModels {
			id := m.GetID()
			reason := false
			for _, sc := range scenarios {
				if sc.Model == id {
					reason = sc.Reason
					break
				}
			}
			models = append(models, scoreboard.Model{Model: id, Reason: reason})
		}
		if err := os.MkdirAll(filepath.Join("testdata", "TestClient", "Scoreboard"), 0o755); err != nil {
			t.Fatal(err)
		}
		getClient := func(t testing.TB, model scoreboard.Model, name string) genai.Provider {
			var opts []genai.ProviderOption
			if model.Model != "" {
				opts = append(opts, genai.ProviderOptionModel(model.Model))
			}
			if name != "" {
				r := internaltest.NewSubprocessRecorder(t, name, "opencode", nil)
				opts = append(opts, genai.ProviderOptionStarterWrapper(r.Wrap))
			}
			c, err := New(t.Context(), opts...)
			if c != nil {
				internaltest.CleanupCloser(t, c)
			}
			if err != nil {
				t.Fatal(err)
			}
			return c
		}
		smoketest.Run(t, smoke.Subprocess(getClient), models, nil)
	})

	t.Run("model_markers", func(t *testing.T) {
		for _, m := range []genai.ProviderOptionModel{genai.ModelCheap, genai.ModelGood, genai.ModelSOTA} {
			if cl, err := New(t.Context(), m); err == nil {
				internaltest.CleanupCloser(t, cl)
				t.Errorf("%s: expected error", m)
			}
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
			if res.Usage.TotalTokens != 13256 {
				t.Errorf("TotalTokens: got %d, want recorded ACP total 13256", res.Usage.TotalTokens)
			}
			if res.Usage.FinishReason != genai.FinishedStop {
				t.Errorf("FinishReason: got %q, want %q", res.Usage.FinishReason, genai.FinishedStop)
			}
		})
		t.Run("session_id_always_in_opaque", func(t *testing.T) {
			t.Setenv("OPENCODE_CONFIG_CONTENT", `{"model":"opencode/big-pickle"}`)
			c := newTestClient(t, "GenSync_hello_nomodel")
			msgs := genai.Messages{genai.NewTextMessage("hello")}
			res, err := c.GenSync(t.Context(), msgs)
			if err != nil {
				t.Fatalf("GenSync: %v", err)
			}
			var found string
			for _, r := range res.Replies {
				if id, ok := r.Opaque[acp.SessionIDKey].(string); ok {
					found = id
				}
			}
			if found == "" {
				t.Fatal("session_id not found in Reply.Opaque")
			}
		})
		t.Run("session_resumed_from_opaque", func(t *testing.T) {
			c1 := newTestClient(t, "GenSync_session_turn1", genai.ProviderOptionModel("opencode/big-pickle"))
			msgs1 := genai.Messages{genai.NewTextMessage("Remember this color: cerulean. Just confirm you noted it.")}
			res1, err := c1.GenSync(t.Context(), msgs1)
			if err != nil {
				t.Fatalf("turn 1: %v", err)
			}
			var sessionID string
			for _, r := range res1.Replies {
				if id, ok := r.Opaque[acp.SessionIDKey].(string); ok {
					sessionID = id
				}
			}
			if sessionID == "" {
				t.Fatal("turn 1 did not return a session_id")
			}

			c2 := newTestClient(t, "GenSync_session_turn2", genai.ProviderOptionModel("opencode/big-pickle"))
			msgs2 := genai.Messages{
				genai.NewTextMessage("Remember this color: cerulean. Just confirm you noted it."),
				{Replies: res1.Replies},
				genai.NewTextMessage("What color did I tell you?"),
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
				if strings.Contains(strings.ToLower(r.Text), "cerulean") {
					found = true
					break
				}
			}
			if !found {
				t.Errorf("turn 2: expected reply to contain 'cerulean', got %v", res2.Replies)
			}
		})
		t.Run("error_result", func(t *testing.T) {
			c := newOutputClient(t, strings.Join([]string{
				`{"jsonrpc":"2.0","id":1,"result":{}}`,
				`{"jsonrpc":"2.0","id":2,"result":{"sessionId":"session-1"}}`,
				`{"jsonrpc":"2.0","id":3,"error":{"code":429,"message":"rate limit exceeded","data":{"providerId":"openai"}}}`,
			}, "\n"))
			msgs := genai.Messages{genai.NewTextMessage("cause error")}
			_, err := c.GenSync(t.Context(), msgs)
			if err == nil {
				t.Fatal("expected error, got nil")
			}
			if !strings.HasPrefix(err.Error(), "opencode: ") || !strings.Contains(err.Error(), "rate limit exceeded") || !strings.Contains(err.Error(), `"providerId":"openai"`) {
				t.Errorf("unexpected error message: %v", err)
			}
		})
		t.Run("invalid_initialize_result", func(t *testing.T) {
			c := newOutputClient(t, `{"jsonrpc":"2.0","id":1,"result":{"protocolVersion":1,"unexpected":true}}`)
			_, err := c.GenSync(t.Context(), genai.Messages{genai.NewTextMessage("hello")})
			if err == nil || !strings.HasPrefix(err.Error(), "opencode: ") || !strings.Contains(err.Error(), "parse initialize response") {
				t.Fatalf("expected initialize decode error, got %v", err)
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
			// Use the available free MiMo model to exercise reasoning chunks.
			c := newTestClient(t, "GenStream_thinking", genai.ProviderOptionModel("opencode/mimo-v2.6-flash-free"))
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
			c := newOutputClient(t, strings.Join([]string{
				`{"jsonrpc":"2.0","id":1,"result":{}}`,
				`{"jsonrpc":"2.0","id":2,"result":{"sessionId":"session-1"}}`,
				`{"jsonrpc":"2.0","id":3,"error":{"code":500,"message":"internal server error"}}`,
			}, "\n"))
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

func TestProviderOption(t *testing.T) {
	t.Run("valid", func(t *testing.T) {
		if err := (&ProviderOption{DangerouslySkipPermissions: true, ExtraArgs: []string{"--pure"}}).Validate(); err != nil {
			t.Error(err)
		}
	})
	t.Run("error", func(t *testing.T) {
		for _, p := range []ProviderOption{{ExtraArgs: []string{""}}} {
			if err := p.Validate(); err == nil {
				t.Errorf("%+v: expected error", p)
			}
		}
	})
}

func TestParseOpts(t *testing.T) {
	t.Run("thinking", func(t *testing.T) {
		for _, tc := range []struct {
			level genai.EffortLevel
			want  string
		}{
			{genai.EffortMinimal, "minimal"},
			{genai.EffortOff, "none"},
			{genai.EffortMedium, "medium"},
			{genai.EffortXHigh, "xhigh"},
			{genai.EffortMax, "max"},
		} {
			t.Run(string(tc.level), func(t *testing.T) {
				got, err := parseOpts([]genai.GenOption{tc.level})
				if err != nil {
					t.Fatal(err)
				}
				if string(got.effort) != tc.want {
					t.Errorf("native effort = %q, want %q", string(got.effort), tc.want)
				}
			})
		}
	})

	co, err := parseOpts([]genai.GenOption{genai.EffortLow})
	if err != nil || co.effort != EffortLow {
		t.Errorf("effort = %q, err = %v, want low", co.effort, err)
	}
	co, err = parseOpts([]genai.GenOption{genai.EffortHigh, &GenOption{Mode: "plan"}})
	if err != nil || co.effort != EffortHigh || co.mode != "plan" {
		t.Errorf("co = %+v, err = %v, want high and plan", co, err)
	}
}

func TestGenOption(t *testing.T) {
	t.Run("error", func(t *testing.T) {
		if err := (&GenOption{Mode: "   "}).Validate(); err == nil {
			t.Fatal("expected error for whitespace mode")
		}
	})
	t.Run("unsupported", func(t *testing.T) {
		_, err := parseOpts([]genai.GenOption{
			genai.GenOptionSeed(1),
			&genai.GenOptionText{
				Temperature: 0.5, TopP: 0.5, MaxTokens: 10, TopLogprobs: 2, SystemPrompt: "sys",
				TopK: 2, Stop: []string{"stop"}, ReplyAsJSON: true, DecodeAs: &struct{}{},
			},
			&genai.GenOptionTools{},
			&genai.GenOptionWeb{},
			genai.GenOptionPollInterval(time.Millisecond),
			&genai.GenOptionAudio{},
			&genai.GenOptionImage{},
			&genai.GenOptionVideo{},
		})
		uerr, ok := errors.AsType[*base.ErrNotSupported](err)
		if !ok {
			t.Fatalf("expected ErrNotSupported, got %v", err)
		}
		want := []string{
			"GenOptionSeed",
			"GenOptionText.Temperature", "GenOptionText.TopP", "GenOptionText.MaxTokens", "GenOptionText.TopLogprobs",
			"GenOptionText.SystemPrompt", "GenOptionText.TopK", "GenOptionText.Stop", "GenOptionText.ReplyAsJSON",
			"GenOptionText.DecodeAs",
			"GenOptionTools", "GenOptionWeb", "GenOptionPollInterval", "GenOptionAudio", "GenOptionImage", "GenOptionVideo",
		}
		if !slices.Equal(uerr.Options, want) {
			t.Errorf("options: got %v, want %v", uerr.Options, want)
		}
	})
	t.Run("empty_text", func(t *testing.T) {
		if _, err := parseOpts([]genai.GenOption{&genai.GenOptionText{}}); err != nil {
			t.Fatalf("empty GenOptionText: %v", err)
		}
	})
}

func init() {
	internal.BeLenient = false
}
