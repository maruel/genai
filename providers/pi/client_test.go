// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Tests for the Pi provider client.

package pi

import (
	"bufio"
	"bytes"
	"errors"
	"io"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/maruel/genai"
	"github.com/maruel/genai/base"
	"github.com/maruel/genai/internal/internaltest"
	"github.com/maruel/genai/scoreboard"
	"github.com/maruel/genai/smoke"
	"github.com/maruel/genai/smoke/smoketest"
)

// testProcessOption prevents recordings from including ambient user context or tool output.
func testProcessOption() *ProviderOption {
	return &ProviderOption{ExtraArgs: []string{
		"--no-context-files", "--no-extensions", "--no-tools",
		"--system-prompt", "You are a helpful assistant.",
	}}
}

func newTestClient(t *testing.T, name string, opts ...genai.ProviderOption) *Client {
	rec := internaltest.NewSubprocessRecorder(t, name, "pi", nil)
	opts = append(opts, testProcessOption(), genai.ProviderOptionStarterWrapper(rec.Wrap))
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
			opts := []genai.ProviderOption{testProcessOption()}
			if model.Model != "" {
				opts = append(opts, genai.ProviderOptionModel(model.Model))
			}
			if name != "" {
				r := internaltest.NewSubprocessRecorder(t, name, "pi", nil)
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

	t.Run("gen_sync", func(t *testing.T) {
		t.Run("hello", func(t *testing.T) {
			c := newTestClient(t, "GenSync_hello", genai.ProviderOptionModel("google/gemini-3.1-flash-lite-preview"))
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
			c := newTestClient(t, "GenStream_hello", genai.ProviderOptionModel("google/gemini-3.1-flash-lite-preview"))
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
			t.Skip("A thinking level does not force thought chunks for a trivial greeting")
			c := newTestClient(t, "GenStream_thinking", genai.ProviderOptionModel("google/gemini-3.1-flash-lite-preview"))
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

	t.Run("informational events", func(t *testing.T) {
		input := strings.Join([]string{
			`{"type":"bash_execution_update","id":"req-1","delta":"output"}`,
			`{"type":"session_info_changed","name":"renamed"}`,
			`{"type":"agent_end","willRetry":false,"messages":[]}`,
			`{"type":"agent_settled"}`,
		}, "\n")
		if _, err := readUntilDone(newScanner(strings.NewReader(input)), io.Discard, func(string, string) bool { return true }); err != nil {
			t.Fatal(err)
		}
	})

	t.Run("extension error", func(t *testing.T) {
		input := `{"type":"extension_error","extensionPath":"/tmp/ext.ts","event":"tool_call","error":"failed"}`
		_, err := readUntilDone(newScanner(strings.NewReader(input)), io.Discard, func(string, string) bool { return true })
		if err == nil || !strings.Contains(err.Error(), "/tmp/ext.ts") || !strings.Contains(err.Error(), "failed") {
			t.Fatalf("expected exact extension error, got %v", err)
		}
	})

	t.Run("prompt response decode error", func(t *testing.T) {
		input := `{"type":"response","command":"prompt","success":true,"error":123}`
		_, err := readUntilDone(newScanner(strings.NewReader(input)), io.Discard, func(string, string) bool { return true })
		if err == nil || !strings.Contains(err.Error(), "parse prompt response") {
			t.Fatalf("expected prompt response decode error, got %v", err)
		}
	})
}

func TestReadResponseForCommand(t *testing.T) {
	t.Run("decode error", func(t *testing.T) {
		input := `{"type":"response","command":"get_available_models","success":true,"error":123}`
		_, err := readResponseForCommand(newScanner(strings.NewReader(input)), CmdGetModels)
		if err == nil || !strings.Contains(err.Error(), "parse get_available_models response") {
			t.Fatalf("expected exact response decode error, got %v", err)
		}
	})

	t.Run("extension error", func(t *testing.T) {
		input := `{"type":"extension_error","extensionPath":"/tmp/model.ts","event":"model_select","error":"failed"}`
		_, err := readResponseForCommand(newScanner(strings.NewReader(input)), CmdGetModels)
		if err == nil || !strings.Contains(err.Error(), "/tmp/model.ts") || !strings.Contains(err.Error(), "failed") {
			t.Fatalf("expected exact extension error, got %v", err)
		}
	})
}

func TestBuildResult(t *testing.T) {
	res, err := buildResult([]byte(`{"type":"agent_end","messages":[{"role":"assistant","content":[{"type":"text","text":"Hello."}],"usage":{"input":10,"output":8,"cacheRead":4,"cacheWrite":2,"cacheWrite1h":1,"reasoning":3,"totalTokens":24,"cost":{"input":0,"output":0,"cacheRead":0,"cacheWrite":0,"total":0}},"stopReason":"stop"}]}`), "", "", 0, 0)
	if err != nil {
		t.Fatal(err)
	}
	if res.Usage.ReasoningTokens != 3 {
		t.Errorf("ReasoningTokens = %d, want 3", res.Usage.ReasoningTokens)
	}
}

func TestProviderOption(t *testing.T) {
	t.Run("args", func(t *testing.T) {
		for _, tc := range []struct {
			name string
			p    ProviderOption
			want []string
		}{
			{
				"options",
				ProviderOption{Skills: true, ExtraArgs: []string{"--no-tools"}},
				[]string{"--mode", "rpc", "--no-session", "--no-tools"},
			},
		} {
			t.Run(tc.name, func(t *testing.T) {
				if got := tc.p.args(); !slices.Equal(got, tc.want) {
					t.Errorf("args = %q, want %q", got, tc.want)
				}
			})
		}
	})
	t.Run("Validate", func(t *testing.T) {
		for _, p := range []ProviderOption{{ExtraArgs: []string{"--mode", "json"}}} {
			if err := p.Validate(); err == nil {
				t.Errorf("%+v: expected error", p)
			}
		}
	})
}

func TestSetThinking(t *testing.T) {
	t.Run("error", func(t *testing.T) {
		for _, response := range []string{
			`{"type":"response","command":"set_thinking_level","success":false,"error":"unsupported level"}`,
			"",
		} {
			if err := setThinking(io.Discard, bufio.NewScanner(strings.NewReader(response)), ThinkingHigh); err == nil {
				t.Errorf("response %q: expected error", response)
			}
		}
	})
}

func TestNew(t *testing.T) {
	for _, m := range []genai.ProviderOptionModel{genai.ModelCheap, genai.ModelGood, genai.ModelSOTA} {
		if cl, err := New(t.Context(), m); err == nil {
			internaltest.CleanupCloser(t, cl)
			t.Errorf("%s: expected error", m)
		}
	}
}

func TestParseOpts(t *testing.T) {
	t.Run("thinking", func(t *testing.T) {
		for _, tc := range []struct {
			level genai.EffortLevel
			want  string
		}{
			{genai.EffortMinimal, "minimal"},
			{genai.EffortOff, "off"},
			{genai.EffortMedium, "medium"},
			{genai.EffortHigh, "high"},
			{genai.EffortXHigh, "xhigh"},
			{genai.EffortMax, "max"},
		} {
			t.Run(string(tc.level), func(t *testing.T) {
				got, err := parseOpts([]genai.GenOption{tc.level})
				if err != nil {
					t.Fatal(err)
				}
				if string(got) != tc.want {
					t.Errorf("native effort = %q, want %q", string(got), tc.want)
				}
			})
		}
	})

	t.Run("valid", func(t *testing.T) {
		got, err := parseOpts([]genai.GenOption{genai.EffortLow})
		if err != nil || got != ThinkingLow {
			t.Errorf("effort = %q, err = %v, want low", got, err)
		}
	})
	t.Run("error", func(t *testing.T) {
		if _, err := parseOpts([]genai.GenOption{genai.EffortLevel("turbo")}); err == nil {
			t.Error("expected error")
		}
		if _, err := parseOpts([]genai.GenOption{&genai.GenOptionText{SystemPrompt: "x"}}); err == nil {
			t.Error("expected ErrNotSupported")
		} else if _, ok := errors.AsType[*base.ErrNotSupported](err); !ok {
			t.Errorf("err = %v, want ErrNotSupported", err)
		}
	})
}

func TestReadUntilDoneContextUsage(t *testing.T) {
	input := strings.Join([]string{
		`{"type":"agent_end","messages":[]}`,
		`{"type":"agent_settled"}`,
		`{"type":"response","command":"get_session_stats","success":true,"data":{"sessionId":"s","contextUsage":{"tokens":1234,"contextWindow":10000}}}`,
	}, "\n")
	var stdin bytes.Buffer
	sc := bufio.NewScanner(strings.NewReader(input))
	res, err := readUntilDone(sc, &stdin, func(string, string) bool { return true })
	if err != nil {
		t.Fatalf("readUntilDone: %v", err)
	}
	if res.Usage.ContextTokens != 1234 || res.Usage.ContextWindow != 10000 {
		t.Errorf("context usage: got %d/%d, want 1234/10000", res.Usage.ContextTokens, res.Usage.ContextWindow)
	}
	if !strings.Contains(stdin.String(), string(CmdGetSessionStats)) {
		t.Errorf("get_session_stats command not sent: %q", stdin.String())
	}
}

func TestGenOption(t *testing.T) {
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
