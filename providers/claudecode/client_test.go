// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Tests for the Claude Code provider client.

package claudecode

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"iter"
	"net/http"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/maruel/genai"
	"github.com/maruel/genai/base"
	"github.com/maruel/genai/internal/internaltest"
	"github.com/maruel/genai/internal/msgutil"
	"github.com/maruel/genai/internal/myrecorder"
	"github.com/maruel/genai/scoreboard"
	"github.com/maruel/genai/smoke/smoketest"
)

func newTestClient(t *testing.T, name string, opts ...genai.ProviderOption) *Client {
	rec := internaltest.NewSubprocessRecorder(t, name, "claude", nil)
	opts = append(opts, genai.ProviderOptionStarterWrapper(rec.Wrap))
	c, err := New(opts...)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return c
}

func newOutputClient(t *testing.T, output string, captureArgs *[]string, opts ...genai.ProviderOption) *Client {
	opts = append(opts, genai.ProviderOptionStarterWrapper(func(genai.Starter) genai.Starter {
		return func(_ context.Context, args []string) (io.WriteCloser, io.ReadCloser, func() error, error) {
			if captureArgs != nil {
				*captureArgs = slices.Clone(args)
			}
			pr, pw := io.Pipe()
			go func() { _, _ = io.Copy(io.Discard, pr) }()
			return pw, io.NopCloser(strings.NewReader(output)), func() error { return nil }, nil
		}
	}))
	c, err := New(opts...)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return c
}

func hasArg(args []string, flag, val string) bool {
	for i, a := range args {
		if a == flag && i+1 < len(args) && args[i+1] == val {
			return true
		}
	}
	return false
}

// skipMediaClient keeps unsupported audio and video smoke cases from reaching Claude Code.
type skipMediaClient struct {
	genai.Provider
}

func (c skipMediaClient) GenSync(ctx context.Context, msgs genai.Messages, opts ...genai.GenOption) (genai.Result, error) {
	if hasUnsupportedMediaClip(msgs) {
		return genai.Result{}, &base.ErrNotSupported{Options: []string{"audio or video input"}}
	}
	return c.Provider.GenSync(ctx, msgs, opts...)
}

func (c skipMediaClient) GenStream(ctx context.Context, msgs genai.Messages, opts ...genai.GenOption) (iter.Seq[genai.Reply], func() (genai.Result, error)) {
	if hasUnsupportedMediaClip(msgs) {
		return func(func(genai.Reply) bool) {}, func() (genai.Result, error) {
			return genai.Result{}, &base.ErrNotSupported{Options: []string{"audio or video input"}}
		}
	}
	return c.Provider.GenStream(ctx, msgs, opts...)
}

func hasUnsupportedMediaClip(msgs genai.Messages) bool {
	for _, msg := range msgs {
		for _, req := range msg.Requests {
			name := req.Doc.Filename
			if name == "" {
				name = req.Doc.URL
			}
			switch strings.ToLower(filepath.Ext(name)) {
			case ".aac", ".flac", ".mp3", ".ogg", ".wav", ".mp4", ".webm":
				return true
			}
		}
	}
	return false
}

func TestSkipMediaClient(t *testing.T) {
	c := skipMediaClient{}
	for _, name := range []string{"audio.aac", "audio.flac", "audio.mp3", "audio.ogg", "audio.wav", "video.mp4", "video.webm"} {
		t.Run(name, func(t *testing.T) {
			msgs := genai.Messages{{Requests: []genai.Request{{Doc: genai.Doc{Filename: name}}}}}
			_, err := c.GenSync(t.Context(), msgs)
			if _, ok := errors.AsType[*base.ErrNotSupported](err); !ok {
				t.Fatalf("GenSync error = %v, want ErrNotSupported", err)
			}
			seq, finish := c.GenStream(t.Context(), msgs)
			for range seq {
				t.Error("GenStream yielded a reply")
			}
			_, err = finish()
			if _, ok := errors.AsType[*base.ErrNotSupported](err); !ok {
				t.Fatalf("GenStream finish error = %v, want ErrNotSupported", err)
			}
		})
	}
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
			t.Fatal(err)
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
			if fn != nil {
				// Let the smoketest framework create HTTP cassettes for bookkeeping
				// (stale recording detection). Claudecode is subprocess-based so
				// the cassettes will be empty; the real data lives in .ndjson fixtures.
				wrapped := fn(http.DefaultTransport)
				if rec, ok := wrapped.(*myrecorder.Recorder); ok {
					name := strings.TrimSuffix(rec.Name(), ".yaml")
					r := internaltest.NewSubprocessRecorder(t, name, "claude", nil)
					opts = append(opts, genai.ProviderOptionStarterWrapper(r.Wrap))
				}
			}
			c, err := New(opts...)
			if err != nil {
				t.Fatal(err)
			}
			// Claude Code does not accept audio or video clips. When asked to process one,
			// it hallucinates Bash commands as text instead of rejecting the input.
			// Keep the unsupported smoke cases out of the CLI while retaining text
			// and image coverage.
			p := skipMediaClient{Provider: c}
			smokeOpts := []genai.GenOption{
				// Scoreboard calls are independent probes, not a conversation.
				&GenOption{SessionPersistence: false},
			}
			if model.Reason {
				smokeOpts = append(smokeOpts, &GenOption{Effort: EffortMedium})
			}
			return &internaltest.InjectOptions{Provider: p, Opts: smokeOpts}
		}
		smoketest.Run(t, getClientRT, models, testRecorder.Records, nil)
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
				if got := c.ModelID(); got != tc.want {
					t.Errorf("got %q, want %q", got, tc.want)
				}
			})
		}
	})
	t.Run("gen_sync", func(t *testing.T) {
		t.Run("hello", func(t *testing.T) {
			c := newTestClient(t, "GenSync_hello", genai.ModelGood)
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
			const output = `{"type":"system","subtype":"init","session_id":"new-session"}
{"type":"assistant","message":{"content":[{"type":"text","text":"You said hello."}],"usage":{"input_tokens":1,"output_tokens":1},"stop_reason":"end_turn"}}
{"type":"result","subtype":"success","is_error":false,"duration_ms":1,"num_turns":1,"result":"You said hello.","stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1},"session_id":"new-session"}
`
			var args []string
			c := newOutputClient(t, output, &args)
			prevAssistant := genai.Message{
				Replies: []genai.Reply{
					{Text: "Hello!"},
					{Opaque: map[string]any{sessionIDKey: "550e8400-e29b-41d4-a716-446655440000"}},
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
			if !hasArg(args, "--resume", "550e8400-e29b-41d4-a716-446655440000") {
				t.Errorf("--resume not found in args %v", args)
			}
		})
		t.Run("live_session_resume", func(t *testing.T) {
			if os.Getenv("CLAUDECODE_LIVE_SESSION_TEST") == "" {
				t.Skip("set CLAUDECODE_LIVE_SESSION_TEST=1 to check CLI session persistence")
			}
			c, err := New(genai.ModelGood)
			if err != nil {
				t.Fatalf("New: %v", err)
			}
			first := make(genai.Messages, 1, 3)
			first[0] = genai.NewTextMessage("Remember the token cedar-ember-47. Reply with exactly that token and nothing else.")
			res, err := c.GenSync(t.Context(), first, &GenOption{MaxBudgetUSD: 0.05, SessionPersistence: true})
			if err != nil {
				t.Fatalf("first GenSync: %v", err)
			}
			if id := msgutil.ExtractOpaqueID(genai.Messages{res.Message}, sessionIDKey); id == "" {
				t.Fatal("first result has no session ID")
			}
			msgs := append(first, res.Message, genai.NewTextMessage("What token did I ask you to remember? Reply with exactly that token and nothing else."))
			res, err = c.GenSync(t.Context(), msgs, &GenOption{MaxBudgetUSD: 0.05})
			if err != nil {
				t.Fatalf("resumed GenSync: %v", err)
			}
			if len(res.Replies) == 0 || !strings.Contains(res.Replies[0].Text, "cedar-ember-47") {
				t.Errorf("resumed reply = %#v, want cedar-ember-47", res.Replies)
			}
		})
		t.Run("live_hello_without_session_persistence", func(t *testing.T) {
			if os.Getenv("CLAUDECODE_LIVE_SESSION_TEST") == "" {
				t.Skip("set CLAUDECODE_LIVE_SESSION_TEST=1 to check a stateless CLI call")
			}
			c, err := New(genai.ModelGood)
			if err != nil {
				t.Fatalf("New: %v", err)
			}
			res, err := c.GenSync(t.Context(), genai.Messages{genai.NewTextMessage("Reply with exactly hello and nothing else.")}, &GenOption{
				MaxBudgetUSD:       0.05,
				SessionPersistence: false,
			})
			if err != nil {
				t.Fatalf("GenSync: %v", err)
			}
			if len(res.Replies) == 0 || !strings.Contains(strings.ToLower(res.Replies[0].Text), "hello") {
				t.Errorf("reply = %#v, want hello", res.Replies)
			}
		})
		t.Run("error_result", func(t *testing.T) {
			const output = `{"type":"result","subtype":"error_during_execution","is_error":true,"result":"boom","usage":{"input_tokens":0,"output_tokens":0},"session_id":"session"}
`
			c := newOutputClient(t, output, nil)
			msgs := genai.Messages{genai.NewTextMessage("cause error")}
			_, err := c.GenSync(t.Context(), msgs)
			if err == nil {
				t.Fatal("expected error, got nil")
			}
			if !strings.Contains(err.Error(), "claude error") {
				t.Errorf("unexpected error message: %v", err)
			}
		})
		t.Run("malformed_output", func(t *testing.T) {
			c := newOutputClient(t, "not-json\n", nil)
			_, err := c.GenSync(t.Context(), genai.Messages{genai.NewTextMessage("hello")})
			if err == nil || !strings.Contains(err.Error(), "parse output envelope") {
				t.Fatalf("GenSync error = %v, want parse output envelope", err)
			}
		})
		t.Run("ask_user_question_haiku", func(t *testing.T) {
			const question = "Which login boundary should Google use in caic?"
			const answer = "Identity only"
			c := newTestClient(t, "GenSync_ask_user_question_haiku", genai.ProviderOptionModel("haiku"))
			msg := genai.NewTextMessage(`Use the AskUserQuestion tool exactly once.
Set the question field exactly to: "` + question + `"
Use exactly two options with labels "Identity only" and "Forge-coupled".
Do not answer the question yourself.`)
			raw, err := c.GenSyncRaw(t.Context(), genai.Messages{msg}, &GenOption{
				Tools:        []string{"AskUserQuestion"},
				MaxBudgetUSD: 0.05,
				ControlHandler: func(_ context.Context, req OutputControlRequestMsg) (InputControlResponseMsg, error) {
					can, err := req.DecodeCanUseTool()
					if err != nil {
						return InputControlResponseMsg{}, err
					}
					if can.Subtype != ControlCanUseTool {
						return InputControlResponseMsg{}, errors.New("unexpected control request subtype " + string(can.Subtype))
					}
					if can.ToolName != "AskUserQuestion" {
						return InputControlResponseMsg{}, errors.New("unexpected tool " + can.ToolName)
					}
					rawInput, err := json.Marshal(can.Input)
					if err != nil {
						return InputControlResponseMsg{}, err
					}
					var input AskUserQuestionInput
					if err := json.Unmarshal(rawInput, &input); err != nil {
						return InputControlResponseMsg{}, err
					}
					if len(input.Questions) != 1 || input.Questions[0].Question != question {
						return InputControlResponseMsg{}, errors.New("unexpected AskUserQuestion input")
					}
					updatedInput, err := json.Marshal(AskUserQuestionUpdatedInput{
						Questions: input.Questions,
						Answers:   map[string]string{question: answer},
					})
					if err != nil {
						return InputControlResponseMsg{}, err
					}
					return InputControlResponseMsg{
						Response: ControlResponse{
							Subtype: ControlResponseSuccess,
							Response: ControlResponsePayload{
								Behavior:     ControlCanUseToolBehaviorAllow,
								UpdatedInput: updatedInput,
							},
						},
					}, nil
				},
			})

			var got, toolUseID string
			var controlSeen bool
			for _, line := range raw {
				var p OutputTypeProbe
				if json.Unmarshal(line, &p) != nil {
					continue
				}
				switch p.Type {
				case OutputControlRequest:
					var m OutputControlRequestMsg
					if err := json.Unmarshal(line, &m); err != nil {
						t.Fatalf("parse control request: %v", err)
					}
					can, err := m.DecodeCanUseTool()
					if err != nil {
						t.Fatalf("parse can_use_tool: %v", err)
					}
					if can.ToolName == "AskUserQuestion" {
						controlSeen = true
					}
				case OutputAssistant:
					var m OutputAssistantMsg
					if err := json.Unmarshal(line, &m); err != nil {
						t.Fatalf("parse assistant: %v", err)
					}
					for i := range m.Message.Content {
						b := &m.Message.Content[i]
						if b.Type != "tool_use" || b.Name != "AskUserQuestion" {
							continue
						}
						rawInput, err := json.Marshal(b.Input)
						if err != nil {
							t.Fatalf("marshal AskUserQuestion input: %v", err)
						}
						var input AskUserQuestionInput
						if err := json.Unmarshal(rawInput, &input); err != nil {
							t.Fatalf("parse AskUserQuestion input: %v", err)
						}
						if len(input.Questions) == 0 {
							t.Fatal("AskUserQuestion input has no questions")
						}
						got = input.Questions[0].Question
						toolUseID = b.ID
						break
					}
				default:
				}
			}
			if toolUseID == "" {
				if err != nil {
					t.Fatalf("GenSyncRaw: %v; AskUserQuestion tool call not found", err)
				}
				t.Fatal("AskUserQuestion tool call not found")
			}
			if err != nil {
				t.Fatalf("GenSyncRaw after AskUserQuestion %s: %v", toolUseID, err)
			}
			if !controlSeen {
				t.Fatal("AskUserQuestion control request not found")
			}

			var result OutputResultMsg
			for _, line := range raw {
				var p OutputTypeProbe
				if json.Unmarshal(line, &p) != nil || p.Type != OutputResult {
					continue
				}
				if err := json.Unmarshal(line, &result); err != nil {
					t.Fatalf("parse result: %v", err)
				}
				break
			}
			if result.Type == "" {
				t.Fatal("result record not found")
			}
			if result.Result == "" {
				t.Fatal("result text is empty")
			}
			if got != question {
				t.Errorf("question = %q, want %q", got, question)
			}
			if len(result.PermissionDenials) != 0 {
				t.Fatalf("PermissionDenials = %d, want 0", len(result.PermissionDenials))
			}

			var toolResultText string
			for _, line := range raw {
				var p OutputTypeProbe
				if json.Unmarshal(line, &p) != nil || p.Type != OutputUser {
					continue
				}
				var m OutputUserMsg
				if err := json.Unmarshal(line, &m); err != nil {
					t.Fatalf("parse user: %v", err)
				}
				msg, err := m.DecodeMessage()
				if err != nil {
					t.Fatalf("decode user message: %v", err)
				}
				if msg.Kind != OutputUserMessageBlock {
					continue
				}
				for i := range msg.Content {
					b := &msg.Content[i]
					if b.Type == "tool_result" && b.ToolUseID == toolUseID {
						if b.IsError {
							t.Fatalf("tool_result for %s is an error: %q", toolUseID, b.Content.Text)
						}
						toolResultText = b.Content.Text
						break
					}
				}
				if toolResultText != "" {
					break
				}
			}
			if toolResultText == "" {
				t.Fatalf("answered tool_result not found for %s", toolUseID)
			}
			if !strings.Contains(toolResultText, answer) {
				t.Errorf("tool_result = %q, want answer %q", toolResultText, answer)
			}
		})
	})
	t.Run("gen_stream", func(t *testing.T) {
		t.Run("hello", func(t *testing.T) {
			c := newTestClient(t, "GenStream_hello", genai.ModelGood)
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
			const output = `{"type":"control_response","response":{"subtype":"success","request_id":"genai-init","response":{}}}
{"type":"system","subtype":"init","session_id":"session"}
{"type":"stream_event","event":{"type":"content_block_delta","delta":{"type":"text_delta","text":"Hello"}}}
{"type":"assistant","message":{"content":[{"type":"text","text":"Hello"}],"usage":{"input_tokens":1,"output_tokens":1},"stop_reason":"end_turn"}}
{"type":"system","subtype":"post_turn_summary","summarizes_uuid":"assistant-1","status_category":"completed","status_detail":"thinking summary: greeted the user.","needs_action":"","uuid":"summary-1","session_id":"session"}
{"type":"result","subtype":"success","is_error":false,"duration_ms":1,"num_turns":1,"result":"Hello","stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1},"session_id":"session"}
`
			c := newOutputClient(t, output, nil, genai.ProviderOptionModel("claude-opus-4-6"))
			msgs := genai.Messages{genai.NewTextMessage("say hello")}
			seq, finish := c.GenStream(t.Context(), msgs, &GenOption{Effort: EffortMedium})

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
			const output = `{"type":"stream_event","event":{"type":"error"}}
`
			c := newOutputClient(t, output, nil)
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

func TestExtractSessionID(t *testing.T) {
	t.Run("found", func(t *testing.T) {
		msgs := genai.Messages{
			genai.NewTextMessage("hi"),
			{Replies: []genai.Reply{
				{Text: "Hello"},
				{Opaque: map[string]any{sessionIDKey: "abc-123"}},
			}},
		}
		if got := msgutil.ExtractOpaqueID(msgs, sessionIDKey); got != "abc-123" {
			t.Errorf("got %q, want %q", got, "abc-123")
		}
	})
	t.Run("not_found", func(t *testing.T) {
		msgs := genai.Messages{genai.NewTextMessage("hi")}
		if got := msgutil.ExtractOpaqueID(msgs, sessionIDKey); got != "" {
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
