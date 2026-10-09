// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Tests for the Antigravity CLI provider client.

package antigravity

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/maruel/genai"
	"github.com/maruel/genai/base"
	"github.com/maruel/genai/internal/internaltest"
	"github.com/maruel/genai/internal/msgutil"
	"github.com/maruel/genai/internal/myrecorder"
	"github.com/maruel/genai/scoreboard"
	"github.com/maruel/genai/smoke/smoketest"
)

// testModel is the cheapest model, used for the recorded unit tests.
const testModel = "gemini-3.8-flash-low"

func newTestClient(t *testing.T, name string, opts ...genai.ProviderOption) *Client {
	rec := internaltest.NewSubprocessRecorder(t, name, "agy", nil)
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

// newOutputClient returns a client whose subprocess prints output and records
// its arguments into args.
func newOutputClient(t *testing.T, output string, args *[]string, opts ...genai.ProviderOption) *Client {
	opts = append(opts, genai.ProviderOptionModel(testModel), genai.ProviderOptionStarterWrapper(func(genai.Starter) genai.Starter {
		return func(_ context.Context, a []string) (io.WriteCloser, io.ReadCloser, func() error, error) {
			if args != nil {
				*args = slices.Clone(a)
			}
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
		t.Fatal(err)
	}
	return c
}

func replyText(res *genai.Result) string {
	var sb strings.Builder
	for i := range res.Replies {
		sb.WriteString(res.Replies[i].Text)
	}
	return sb.String()
}

func TestClient(t *testing.T) {
	testRecorder := internaltest.NewRecords()
	t.Cleanup(func() {
		if err := testRecorder.Close(); err != nil {
			t.Error(err)
		}
	})

	t.Run("Capabilities", func(t *testing.T) {
		internaltest.TestCapabilities(t, newTestClient(t, "TestClient_Capabilities"))
	})

	t.Run("Scoreboard", func(t *testing.T) {
		all, err := newTestClient(t, "ListModels").listModels(t.Context())
		if err != nil {
			t.Fatal(err)
		}
		scenarios := Scoreboard().Scenarios
		model := func(id string) scoreboard.Model {
			for i := range scenarios {
				if slices.Contains(scenarios[i].Models, id) {
					return scoreboard.Model{Model: id, Reason: scenarios[i].Reason}
				}
			}
			return scoreboard.Model{Model: id}
		}
		models := make([]scoreboard.Model, len(all))
		for i := range all {
			models[i] = model(all[i].ID)
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
				// The smoketest framework creates HTTP cassettes for stale recording
				// bookkeeping; the data lives in the .ndjson fixtures.
				if rec, ok := fn(http.DefaultTransport).(*myrecorder.Recorder); ok {
					r := internaltest.NewSubprocessRecorder(t, strings.TrimSuffix(rec.Name(), ".yaml"), "agy", nil)
					opts = append(opts, genai.ProviderOptionStarterWrapper(r.Wrap))
				}
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
		smoketest.Run(t, getClientRT, models, testRecorder.Records, nil)
	})

	t.Run("ListModels", func(t *testing.T) {
		models, err := newTestClient(t, "ListModels").ListModels(t.Context())
		if err != nil {
			t.Fatal(err)
		}
		if !slices.ContainsFunc(models, func(m genai.Model) bool { return m.GetID() == testModel }) {
			t.Errorf("ListModels() lacks %q", testModel)
		}
	})

	t.Run("GenSync", func(t *testing.T) {
		t.Run("hello", func(t *testing.T) {
			c := newTestClient(t, "GenSync_hello", genai.ProviderOptionModel(testModel))
			res, err := c.GenSync(t.Context(), genai.Messages{genai.NewTextMessage("Say hello. Reply with one word.")})
			if err != nil {
				t.Fatal(err)
			}
			if got := replyText(&res); !strings.Contains(strings.ToLower(got), "hello") {
				t.Errorf("text = %q, want hello", got)
			}
			if res.Usage.InputTokens == 0 || res.Usage.OutputTokens == 0 {
				t.Errorf("Usage = %+v, want input and output tokens", res.Usage)
			}
			if res.Usage.FinishReason != genai.FinishedStop {
				t.Errorf("FinishReason = %q, want %q", res.Usage.FinishReason, genai.FinishedStop)
			}
			if msgutil.ExtractOpaqueID(genai.Messages{res.Message}, conversationIDKey) == "" {
				t.Error("missing conversation ID")
			}
		})
		t.Run("session", func(t *testing.T) {
			c1 := newTestClient(t, "GenSync_session_turn1", genai.ProviderOptionModel(testModel))
			first := genai.NewTextMessage("Remember the number 42. Reply with OK.")
			res, err := c1.GenSync(t.Context(), genai.Messages{first})
			if err != nil {
				t.Fatal(err)
			}
			msgs := genai.Messages{first, res.Message, genai.NewTextMessage("Which number did I ask you to remember? Reply with only the number.")}
			c2 := newTestClient(t, "GenSync_session_turn2", genai.ProviderOptionModel(testModel))
			res, err = c2.GenSync(t.Context(), msgs)
			if err != nil {
				t.Fatal(err)
			}
			if got := replyText(&res); !strings.Contains(got, "42") {
				t.Errorf("text = %q, want 42", got)
			}
		})
		t.Run("DecodeAs", func(t *testing.T) {
			type answer struct {
				Color string `json:"color"`
				N     int    `json:"n"`
			}
			c := newTestClient(t, "GenSync_DecodeAs", genai.ProviderOptionModel(testModel))
			res, err := c.GenSync(t.Context(), genai.Messages{genai.NewTextMessage("Reply in JSON with the color of a clear daytime sky and the number 7.")},
				&genai.GenOptionText{DecodeAs: &answer{}})
			if err != nil {
				t.Fatal(err)
			}
			var got answer
			if err := res.Decode(&got); err != nil {
				t.Fatal(err)
			}
			if got.N != 7 || !strings.EqualFold(got.Color, "blue") {
				t.Errorf("Decode() = %+v, want blue and 7", got)
			}
		})
	})

	t.Run("GenStream", func(t *testing.T) {
		c := newTestClient(t, "GenStream_hello", genai.ProviderOptionModel(testModel))
		seq, finish := c.GenStream(t.Context(), genai.Messages{genai.NewTextMessage("Say hello. Reply with one word.")})
		var sb strings.Builder
		for r := range seq {
			sb.WriteString(r.Text)
		}
		res, err := finish()
		if err != nil {
			t.Fatal(err)
		}
		if got := replyText(&res); got != sb.String() || !strings.Contains(strings.ToLower(got), "hello") {
			t.Errorf("result text = %q, streamed = %q, want equal and containing hello", got, sb.String())
		}
	})
}

func TestNew(t *testing.T) {
	t.Run("error", func(t *testing.T) {
		for _, opt := range []genai.ProviderOption{
			genai.ProviderOptionRemote("http://localhost"),
			genai.ModelCheap,
			genai.ModelGood,
			genai.ModelSOTA,
		} {
			if cl, err := New(t.Context(), opt); err == nil {
				internaltest.CleanupCloser(t, cl)
				t.Errorf("%v: expected error", opt)
			}
		}
	})
}

func TestProviderOption(t *testing.T) {
	t.Run("Validate", func(t *testing.T) {
		t.Run("valid", func(t *testing.T) {
			for _, e := range []Effort{"", EffortLow, EffortMedium, EffortHigh, EffortMax} {
				if err := (&ProviderOption{Effort: e}).Validate(); err != nil {
					t.Errorf("Effort %q: %v", e, err)
				}
			}
		})
		t.Run("error", func(t *testing.T) {
			for _, p := range []ProviderOption{
				{Effort: "xhigh"},
				{Mode: "auto"},
				{ExtraArgs: []string{"--output-format=json"}},
				{ExtraArgs: []string{"-input-format", "text"}},
				{ExtraArgs: []string{"-p"}},
			} {
				if err := p.Validate(); err == nil {
					t.Errorf("%+v: expected error", p)
				}
			}
		})
	})
}

func TestGenSync(t *testing.T) {
	t.Run("valid", func(t *testing.T) {
		t.Run("usage sums steps", func(t *testing.T) {
			// Result.Usage is cumulative over the conversation; the call reports
			// only the usage of its own steps.
			out := strings.Join([]string{
				`{"event":"init","conversation_id":"c1","init":{"model":"m","cwd":"/tmp","tools":["view_file"]}}`,
				`{"event":"step_update","step_update":{"conversation_id":"c1","step_index":4,"state":"DONE","step_type":"user_input"}}`,
				`{"event":"step_update","step_update":{"conversation_id":"c1","step_index":5,"state":"DONE","step_type":"agent_response","usage":{"input_tokens":10,"output_tokens":3,"thinking_tokens":1,"cache_read_tokens":2,"total_tokens":13}}}`,
				`{"event":"step_update","step_update":{"conversation_id":"c1","step_index":6,"state":"DONE","step_type":"tool","tool_name":"view_file","tool_info":{"name":"view_file","parameters":{"AbsolutePath":"/tmp/a"},"output":"1 line"}}}`,
				`{"event":"step_update","step_update":{"conversation_id":"c1","step_index":7,"state":"ACTIVE","step_type":"agent_response","text_delta":"Hel"}}`,
				`{"event":"step_update","step_update":{"conversation_id":"c1","step_index":7,"state":"DONE","step_type":"agent_response","text_delta":"lo","usage":{"input_tokens":20,"output_tokens":4,"thinking_tokens":0,"cache_read_tokens":0,"total_tokens":24}}}`,
				`{"event":"result","result":{"conversation_id":"c1","status":"SUCCESS","response":"Hello","duration_seconds":9,"num_turns":3,"usage":{"input_tokens":999,"output_tokens":999,"thinking_tokens":999,"cache_read_tokens":999,"total_tokens":1998}}}`,
			}, "\n")
			res, err := newOutputClient(t, out, nil).GenSync(t.Context(), genai.Messages{genai.NewTextMessage("hi")})
			if err != nil {
				t.Fatal(err)
			}
			if got := replyText(&res); got != "Hello" {
				t.Errorf("text = %q, want Hello", got)
			}
			if got := msgutil.ExtractOpaqueID(genai.Messages{res.Message}, "conversation_id"); got != "c1" {
				t.Errorf("conversation_id = %q, want c1", got)
			}
			want := genai.Usage{InputTokens: 30, InputCachedTokens: 2, ReasoningTokens: 1, OutputTokens: 7, TotalTokens: 37, FinishReason: genai.FinishedStop}
			if res.Usage.InputTokens != want.InputTokens || res.Usage.InputCachedTokens != want.InputCachedTokens ||
				res.Usage.ReasoningTokens != want.ReasoningTokens || res.Usage.OutputTokens != want.OutputTokens ||
				res.Usage.TotalTokens != want.TotalTokens || res.Usage.FinishReason != want.FinishReason {
				t.Errorf("Usage = %+v, want %+v", res.Usage, want)
			}
		})
		t.Run("args", func(t *testing.T) {
			out := `{"event":"result","result":{"conversation_id":"c1","status":"SUCCESS","response":"","duration_seconds":0,"num_turns":1,"usage":{"input_tokens":0,"output_tokens":0,"thinking_tokens":0,"cache_read_tokens":0,"total_tokens":0}}}`
			resumed := genai.Messages{
				genai.NewTextMessage("first"),
				{Replies: []genai.Reply{{Opaque: map[string]any{"conversation_id": "prev"}}}},
				genai.NewTextMessage("second"),
			}
			for _, tc := range []struct {
				name    string
				msgs    genai.Messages
				opt     ProviderOption
				want    []string
				notWant []string
			}{
				{
					name:    "default",
					msgs:    genai.Messages{genai.NewTextMessage("hi")},
					want:    []string{"--disable-slash-commands", "--model " + testModel},
					notWant: []string{"--conversation", "--dangerously-skip-permissions", "--effort", "--mode", "--sandbox"},
				},
				{
					name: "options",
					msgs: resumed,
					opt: ProviderOption{
						Effort: EffortHigh, Mode: ModePlan, DangerouslySkipPermissions: true, Sandbox: true, Skills: true,
						ExtraArgs: []string{"--add-dir", "/tmp/x"},
					},
					want: []string{
						"--conversation prev", "--dangerously-skip-permissions", "--effort " + string(EffortHigh),
						"--mode " + string(ModePlan), "--sandbox", "--add-dir /tmp/x",
					},
					notWant: []string{"--disable-slash-commands"},
				},
			} {
				t.Run(tc.name, func(t *testing.T) {
					var args []string
					if _, err := newOutputClient(t, out, &args, &tc.opt).GenSync(t.Context(), tc.msgs); err != nil {
						t.Fatal(err)
					}
					joined := " " + strings.Join(args, " ") + " "
					for _, w := range tc.want {
						if !strings.Contains(joined, " "+w+" ") {
							t.Errorf("args = %q, want %q", args, w)
						}
					}
					for _, w := range tc.notWant {
						if slices.Contains(args, w) {
							t.Errorf("args = %q, want no %q", args, w)
						}
					}
				})
			}
		})
	})
	t.Run("error", func(t *testing.T) {
		for _, tc := range []struct {
			name string
			out  string
			want string
		}{
			{
				"status",
				`{"event":"result","result":{"conversation_id":"","status":"ERROR","response":"","error":"invalid model selection","duration_seconds":0,"num_turns":0,"usage":{"input_tokens":0,"output_tokens":0,"thinking_tokens":0,"cache_read_tokens":0,"total_tokens":0}}}`,
				"agy error: invalid model selection",
			},
			{"unknown event", `{"event":"surprise"}`, `unexpected agy event "surprise"`},
			{"no result", "", "agy exited without a result event"},
		} {
			t.Run(tc.name, func(t *testing.T) {
				_, err := newOutputClient(t, tc.out, nil).GenSync(t.Context(), genai.Messages{genai.NewTextMessage("hi")})
				if err == nil || !strings.Contains(err.Error(), tc.want) {
					t.Errorf("err = %v, want %q", err, tc.want)
				}
			})
		}
		t.Run("image", func(t *testing.T) {
			msgs := genai.Messages{{Requests: []genai.Request{{Doc: genai.Doc{Filename: "a.png", Src: strings.NewReader("\x89PNG")}}}}}
			_, err := newOutputClient(t, "", nil).GenSync(t.Context(), msgs)
			if _, ok := errors.AsType[*base.ErrNotSupported](err); !ok {
				t.Errorf("err = %v, want ErrNotSupported", err)
			}
		})
	})
}

func TestGenStream(t *testing.T) {
	const usage = `"usage":{"input_tokens":1,"output_tokens":1,"thinking_tokens":0,"cache_read_tokens":0,"total_tokens":2}`
	t.Run("stop early", func(t *testing.T) {
		out := strings.Join([]string{
			`{"event":"step_update","step_update":{"conversation_id":"c","step_index":1,"state":"ACTIVE","step_type":"agent_response","text_delta":"a"}}`,
			`{"event":"step_update","step_update":{"conversation_id":"c","step_index":1,"state":"DONE","step_type":"agent_response","text_delta":"b",` + usage + `}}`,
			`{"event":"result","result":{"conversation_id":"c","status":"SUCCESS","response":"ab","duration_seconds":1,"num_turns":1,` + usage + `}}`,
		}, "\n")
		seq, finish := newOutputClient(t, out, nil).GenStream(t.Context(), genai.Messages{genai.NewTextMessage("hi")})
		for r := range seq {
			if r.Text != "a" {
				t.Errorf("first reply = %q, want a", r.Text)
			}
			break
		}
		if _, err := finish(); err != nil {
			t.Fatal(err)
		}
	})
	t.Run("DecodeAs", func(t *testing.T) {
		// The streamed text carries extra keys; only structured_output is yielded.
		out := `{"event":"step_update","step_update":{"conversation_id":"c","step_index":1,"state":"DONE","step_type":"agent_response","text_delta":"{\"n\":7,\"toolAction\":\"x\"}",` + usage + `}}` + "\n" + `{"event":"result","result":{"conversation_id":"c","status":"SUCCESS","response":"","duration_seconds":1,"num_turns":1,"structured_output":{"n":7},"json_schema":{"type":"object"},` + usage + `}}`
		var args []string
		seq, finish := newOutputClient(t, out, &args).GenStream(t.Context(), genai.Messages{genai.NewTextMessage("hi")},
			&genai.GenOptionText{DecodeAs: genai.JSONSchema(`{"type":"object"}`)})
		var got []string
		for r := range seq {
			got = append(got, r.Text)
		}
		res, err := finish()
		if err != nil {
			t.Fatal(err)
		}
		if !slices.Equal(got, []string{`{"n":7}`}) || replyText(&res) != `{"n":7}` {
			t.Errorf("streamed = %q, result = %q, want structured output once", got, replyText(&res))
		}
		if i := slices.Index(args, "--json-schema"); i < 0 || args[i+1] != `{"type":"object"}` {
			t.Errorf("args = %q, want --json-schema", args)
		}
	})
}

func TestStreamEvent(t *testing.T) {
	// Every recorded stdout line must decode without unknown fields, so the
	// DTOs stay in sync with real agy output.
	files, err := filepath.Glob(filepath.Join("testdata", "*.ndjson"))
	if err != nil {
		t.Fatal(err)
	}
	more, err := filepath.Glob(filepath.Join("testdata", "TestClient", "Scoreboard", "*", "*.ndjson"))
	if err != nil {
		t.Fatal(err)
	}
	files = append(files, more...)
	if len(files) == 0 {
		t.Fatal("no fixtures")
	}
	for _, f := range files {
		raw, err := os.ReadFile(f)
		if err != nil {
			t.Fatal(err)
		}
		for i, line := range bytes.Split(bytes.TrimSpace(raw), []byte("\n")) {
			d := json.NewDecoder(bytes.NewReader(line))
			d.DisallowUnknownFields()
			var ev StreamEvent
			if err := d.Decode(&ev); err != nil {
				t.Errorf("%s:%d: %v", f, i+1, err)
			}
		}
	}
}

func TestScoreboard(t *testing.T) {
	s := Scoreboard()
	if err := s.Validate(); err != nil {
		t.Fatal(err)
	}
}

func TestStreamInputUserMessage(t *testing.T) {
	t.Run("string content", func(t *testing.T) {
		var m StreamInputUserMessage
		if err := json.Unmarshal([]byte(`{"content":"hello"}`), &m); err != nil {
			t.Fatal(err)
		}
		if len(m.Content) != 1 || m.Content[0].Type != "text" || m.Content[0].Text != "hello" {
			t.Fatalf("unexpected content: %+v", m.Content)
		}
	})
	t.Run("array content", func(t *testing.T) {
		var m StreamInputUserMessage
		if err := json.Unmarshal([]byte(`{"content":[{"type":"text","text":"hi"}]}`), &m); err != nil {
			t.Fatal(err)
		}
		if len(m.Content) != 1 || m.Content[0].Type != "text" || m.Content[0].Text != "hi" {
			t.Fatalf("unexpected content: %+v", m.Content)
		}
	})
	t.Run("null content", func(t *testing.T) {
		var m StreamInputUserMessage
		if err := json.Unmarshal([]byte(`{"content":null}`), &m); err != nil {
			t.Fatal(err)
		}
		if m.Content != nil {
			t.Fatalf("unexpected content: %+v", m.Content)
		}
	})
}

func TestCmdExecutor(t *testing.T) {
	if mode := os.Getenv("GENAI_AGY_ERROR_HELPER"); mode != "" {
		if mode == "hold-stderr" {
			time.Sleep(time.Minute)
			os.Exit(0)
		}
		if _, err := fmt.Fprintln(os.Stderr, "private unrelated stderr"); err != nil {
			t.Fatal(err)
		}
		if strings.HasPrefix(mode, "cancel") || mode == "success-descendant" {
			pid := 0
			if strings.HasSuffix(mode, "-descendant") {
				bin, err := os.Executable()
				if err != nil {
					t.Fatal(err)
				}
				cmd := exec.Command(bin, "-test.run=^TestCmdExecutor$")
				cmd.Env = append(os.Environ(), "GENAI_AGY_ERROR_HELPER=hold-stderr")
				// The descendant outlives wait. On Windows, its working directory
				// would keep the executor from removing its temporary directory.
				cmd.Dir = os.TempDir()
				cmd.Stderr = os.Stderr
				if err := cmd.Start(); err != nil {
					t.Fatal(err)
				}
				pid = cmd.Process.Pid
			}
			if _, err := fmt.Fprintln(os.Stdout, "ready", pid); err != nil {
				t.Fatal(err)
			}
			if mode == "success-descendant" {
				os.Exit(0)
			}
			time.Sleep(time.Minute)
			os.Exit(0)
		}
		if _, err := fmt.Fprint(os.Stderr, `AGY_ERROR: {"short_error":"quota exhausted","status":"RESOURCE_EXHAUSTED","error_code":429,"code_kind":"HTTP","retryable":true,"error_id":"error-123"}`); err != nil {
			t.Fatal(err)
		}
		if mode == "success" {
			os.Exit(0)
		}
		os.Exit(3)
	}
	bin, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	for _, mode := range []string{"error", "success", "success-descendant", "cancel", "cancel-descendant"} {
		t.Run(mode, func(t *testing.T) {
			t.Setenv("GENAI_AGY_ERROR_HELPER", mode)
			ctx, cancel := context.WithCancel(t.Context())
			t.Cleanup(cancel)
			e := cmdExecutor{bin: bin}
			in, out, wait, err := e.start(ctx, []string{"-test.run=^TestCmdExecutor$"})
			if err != nil {
				t.Fatal(err)
			}
			if err := in.Close(); err != nil {
				t.Fatal(err)
			}
			if strings.HasPrefix(mode, "cancel") || mode == "success-descendant" {
				line, err := bufio.NewReader(out).ReadString('\n')
				if err != nil || !strings.HasPrefix(line, "ready ") {
					t.Fatalf("ready = %q, %v", line, err)
				}
				if strings.HasSuffix(mode, "-descendant") {
					pid, err := strconv.Atoi(strings.TrimSpace(strings.TrimPrefix(line, "ready ")))
					if err != nil {
						t.Fatal(err)
					}
					p, err := os.FindProcess(pid)
					if err != nil {
						t.Fatal(err)
					}
					t.Cleanup(func() {
						if err := p.Kill(); err != nil && !errors.Is(err, os.ErrProcessDone) {
							t.Error(err)
						}
						if err := p.Release(); err != nil {
							t.Error(err)
						}
					})
				}
				if strings.HasPrefix(mode, "cancel") {
					cancel()
				}
			}
			if _, err := io.Copy(io.Discard, out); err != nil {
				t.Fatal(err)
			}
			done := make(chan error, 1)
			go func() { done <- wait() }()
			select {
			case err = <-done:
			case <-time.After(5 * time.Second):
				t.Fatal("wait blocked while a descendant held stderr")
			}
			if strings.HasPrefix(mode, "success") {
				if err != nil {
					t.Fatalf("successful run with a prior diagnostic: %v", err)
				}
				return
			}
			if _, ok := errors.AsType[*exec.ExitError](err); !ok {
				t.Fatalf("wait error = %v, want exit error", err)
			}
			if strings.Contains(err.Error(), "private unrelated") {
				t.Fatalf("wait leaked unrelated stderr: %v", err)
			}
			if mode == "error" {
				e, ok := errors.AsType[*AgentError](err)
				if !ok || e.ShortError != "quota exhausted" || e.Status != "RESOURCE_EXHAUSTED" || e.ErrorCode != 429 || e.CodeKind != "HTTP" || !e.Retryable || e.ErrorID != "error-123" {
					t.Fatalf("structured diagnostic = %+v, %v", e, err)
				}
			}
		})
	}
}

func TestAgentErrorWriter(t *testing.T) {
	t.Run("Write", func(t *testing.T) {
		for _, tc := range []struct {
			name string
			text string
			want string
		}{
			{"split prefix and JSON", "AGY_ERROR: {\"short_error\":\"failed\"}\n", "agy error: failed"},
			{"malformed", "AGY_ERROR: {\n", "parse agy error diagnostic"},
			{"oversized diagnostic", "AGY_ERROR: " + strings.Repeat("x", 70<<10) + "\n", "exceeds 64 KiB"},
			{"oversized noise", strings.Repeat("x", 70<<10) + "\nAGY_ERROR: {\"short_error\":\"failed\"}", "agy error: failed"},
			{"first diagnostic", "AGY_ERROR: {\"short_error\":\"first\"}\nAGY_ERROR: {\"short_error\":\"second\"}\n", "agy error: first"},
			{"unrelated stderr", "secret\n" + strings.Repeat("x", 70<<10), ""},
		} {
			t.Run(tc.name, func(t *testing.T) {
				var w agentErrorWriter
				for chunk := range slices.Chunk([]byte(tc.text), 7) {
					if n, err := w.Write(chunk); err != nil || n != len(chunk) {
						t.Fatalf("Write = %d, %v", n, err)
					}
				}
				err := w.finish()
				if tc.want == "" {
					if err != nil {
						t.Fatal(err)
					}
				} else if err == nil || !strings.Contains(err.Error(), tc.want) {
					t.Errorf("error = %v, want %q", err, tc.want)
				}
			})
		}
	})
}
