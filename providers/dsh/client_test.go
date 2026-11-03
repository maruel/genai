// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Tests for the DeepSeek Harness provider client.

package dsh

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/maruel/genai"
	"github.com/maruel/genai/acp"
	"github.com/maruel/genai/base"
	"github.com/maruel/genai/internal"
	"github.com/maruel/genai/internal/internaltest"
	"github.com/maruel/genai/internal/myrecorder"
	"github.com/maruel/genai/scoreboard"
	"github.com/maruel/genai/smoke/smoketest"
)

func newTestClient(t *testing.T, name string, opts ...genai.ProviderOption) *Client {
	rec := internaltest.NewSubprocessRecorder(t, name, "dsh", nil)
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

func newOutputClient(t *testing.T, output string) *Client {
	c, _ := newCaptureClient(t, output)
	return c
}

func newCaptureClient(t *testing.T, output string, opts ...genai.ProviderOption) (*Client, *bytes.Buffer) {
	var written bytes.Buffer
	opts = append(opts, genai.ProviderOptionStarterWrapper(func(genai.Starter) genai.Starter {
		return func(_ context.Context, _ []string) (io.WriteCloser, io.ReadCloser, func() error, error) {
			return nopWriteCloser{&written}, io.NopCloser(strings.NewReader(output)), func() error { return nil }, nil
		}
	}))
	c, err := New(t.Context(), opts...)
	if c != nil {
		internaltest.CleanupCloser(t, c)
	}
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return c, &written
}

func TestNew(t *testing.T) {
	t.Run("error", func(t *testing.T) {
		t.Run("model_markers", func(t *testing.T) {
			for _, m := range []genai.ProviderOptionModel{genai.ModelCheap, genai.ModelGood, genai.ModelSOTA} {
				if cl, err := New(t.Context(), m); err == nil {
					internaltest.CleanupCloser(t, cl)
					t.Errorf("%s: expected error", m)
				}
			}
		})
		t.Run("invalid_model", func(t *testing.T) {
			for _, id := range []string{"deepseek-v4-flash", "/model", "provider/"} {
				if cl, err := New(t.Context(), genai.ProviderOptionModel(id)); err == nil {
					internaltest.CleanupCloser(t, cl)
					t.Errorf("%q: expected error", id)
				}
			}
		})
		t.Run("unsupported_option", func(t *testing.T) {
			c, err := New(t.Context(), genai.ProviderOptionAPIKey("test"))
			if c != nil {
				internaltest.CleanupCloser(t, c)
			}
			if err == nil || !strings.Contains(err.Error(), "unsupported provider option") {
				t.Fatalf("expected unsupported option error, got %v", err)
			}
		})
	})
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
					r := internaltest.NewSubprocessRecorder(t, name, "dsh", nil)
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

	t.Run("GenSync", func(t *testing.T) {
		t.Run("valid", func(t *testing.T) {
			t.Run("hello", func(t *testing.T) {
				c := newTestClient(t, "GenSync_hello", genai.ProviderOptionModel("deepseek-official/deepseek-v4-flash"))
				msgs := genai.Messages{genai.NewTextMessage("say hello")}
				res, err := c.GenSync(t.Context(), msgs)
				if err != nil {
					t.Fatalf("GenSync: %v", err)
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
				if res.Usage.FinishReason != genai.FinishedStop {
					t.Errorf("FinishReason: got %q, want %q", res.Usage.FinishReason, genai.FinishedStop)
				}
				if res.Usage.ContextTokens != 7496 || res.Usage.ContextWindow != 1000000 {
					t.Errorf("context usage: got %d/%d, want 7496/1000000", res.Usage.ContextTokens, res.Usage.ContextWindow)
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
					if id, ok := r.Opaque[acp.SessionIDKey].(string); ok {
						found = id
					}
				}
				if found == "" {
					t.Fatal("session_id not found in Reply.Opaque")
				}
			})
			t.Run("session_resumed_from_opaque", func(t *testing.T) {
				c1 := newTestClient(t, "GenSync_session_turn1", genai.ProviderOptionModel("deepseek-official/deepseek-v4-flash"))
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

				c2 := newTestClient(t, "GenSync_session_turn2", genai.ProviderOptionModel("deepseek-official/deepseek-v4-flash"))
				msgs2 := genai.Messages{
					genai.NewTextMessage("Remember this color: cerulean. Just confirm you noted it."),
					{Replies: res1.Replies},
					genai.NewTextMessage("What color did I tell you?"),
				}
				res2, err := c2.GenSync(t.Context(), msgs2)
				if err != nil {
					t.Fatalf("turn 2: %v", err)
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
			t.Run("resume", func(t *testing.T) {
				// A resumed session keeps no session ID in the result; the
				// request must use session/resume with the previous ID.
				responses := strings.Join([]string{
					`{"jsonrpc":"2.0","id":1,"result":{}}`,
					`{"jsonrpc":"2.0","id":2,"result":{}}`,
					`{"jsonrpc":"2.0","id":3,"result":{"stopReason":"end_turn"}}`,
				}, "\n")
				c, written := newCaptureClient(t, responses)
				msgs := genai.Messages{
					{Replies: []genai.Reply{{Opaque: map[string]any{acp.SessionIDKey: "prev"}}}},
					genai.NewTextMessage("continue"),
				}
				res, err := c.GenSync(t.Context(), msgs)
				if err != nil {
					t.Fatalf("GenSync: %v", err)
				}
				var got *acp.SessionResumeParams
				for line := range strings.SplitSeq(strings.TrimSpace(written.String()), "\n") {
					var req acp.JSONRPCRequest
					if err := json.Unmarshal([]byte(line), &req); err != nil {
						t.Fatalf("unmarshal request: %v", err)
					}
					if req.Method != acp.MethodSessionResume {
						continue
					}
					var params acp.SessionResumeParams
					if err := json.Unmarshal(req.Params, &params); err != nil {
						t.Fatalf("unmarshal session/resume params: %v", err)
					}
					got = &params
				}
				if got == nil || got.SessionID != "prev" {
					t.Errorf("session/resume params: %#v", got)
				}
				found := false
				for _, r := range res.Replies {
					if id, ok := r.Opaque[acp.SessionIDKey].(string); ok && id == "prev" {
						found = true
					}
				}
				if !found {
					t.Errorf("resumed session ID missing from replies: %#v", res.Replies)
				}
			})
			t.Run("effort", func(t *testing.T) {
				// The recorder keeps only stdout, so assert the emitted
				// reasoning_effort request with a scripted server.
				route := `["deepseek-official","deepseek-v4-flash"]`
				configOptions := func(effort string) []acp.SessionConfigOption {
					return []acp.SessionConfigOption{
						{
							ID:           acp.ConfigOptionModel,
							Name:         "Model",
							Category:     acp.ConfigOptionCategoryModel,
							Type:         acp.ConfigOptionTypeSelect,
							CurrentValue: json.RawMessage(route),
							Options: []acp.ConfigOptionValue{{
								Group:   "deepseek-official",
								Name:    "DeepSeek",
								Options: []acp.ConfigOptionValue{{Value: route, Name: "deepseek-v4-flash"}},
							}},
						},
						{
							ID:           "reasoning_effort",
							Name:         "Reasoning effort",
							Category:     acp.ConfigOptionCategoryThoughtLevel,
							Type:         acp.ConfigOptionTypeSelect,
							CurrentValue: json.RawMessage(`"` + effort + `"`),
							Options:      []acp.ConfigOptionValue{{Value: "low", Name: "Low"}, {Value: "high", Name: "High"}},
						},
					}
				}
				var responses strings.Builder
				write := func(id int64, result any) {
					raw, err := json.Marshal(result)
					if err != nil {
						t.Fatal(err)
					}
					line, err := json.Marshal(acp.JSONRPCResponse{JSONRPC: "2.0", ID: json.RawMessage(strconv.FormatInt(id, 10)), Result: raw})
					if err != nil {
						t.Fatal(err)
					}
					responses.Write(line)
					responses.WriteByte('\n')
				}
				write(1, acp.InitializeResult{})
				write(2, acp.SessionNewResult{SessionID: "session-1", ConfigOptions: configOptions("high")})
				write(3, acp.SetSessionConfigOptionResult{ConfigOptions: configOptions("high")})
				write(4, acp.SetSessionConfigOptionResult{ConfigOptions: configOptions("high")})
				responses.WriteString(`{"jsonrpc":"2.0","method":"session/update","params":{"sessionId":"session-1","update":{"sessionUpdate":"agent_message_chunk","content":{"type":"text","text":"Hello"}}}}`)
				responses.WriteByte('\n')
				write(5, acp.PromptResult{StopReason: acp.StopReasonEndTurn})

				c, written := newCaptureClient(t, responses.String(), genai.ProviderOptionModel("deepseek-official/deepseek-v4-flash"), &ProviderOption{Effort: EffortLow})
				if _, err := c.GenSync(t.Context(), genai.Messages{genai.NewTextMessage("say hello")}, &GenOption{Effort: EffortHigh}); err != nil {
					t.Fatalf("GenSync: %v", err)
				}
				var got []acp.SetSessionConfigOptionParams
				for line := range strings.SplitSeq(strings.TrimSpace(written.String()), "\n") {
					var req acp.JSONRPCRequest
					if err := json.Unmarshal([]byte(line), &req); err != nil {
						t.Fatalf("unmarshal request: %v", err)
					}
					if req.Method != acp.MethodSessionSetConfigOption {
						continue
					}
					var params acp.SetSessionConfigOptionParams
					if err := json.Unmarshal(req.Params, &params); err != nil {
						t.Fatalf("unmarshal session/set_config_option: %v", err)
					}
					got = append(got, params)
				}
				want := []acp.SetSessionConfigOptionParams{
					{SessionID: "session-1", ConfigID: acp.ConfigOptionModel, Value: route},
					{SessionID: "session-1", ConfigID: "reasoning_effort", Value: "high"},
				}
				if !slices.EqualFunc(got, want, func(a, b acp.SetSessionConfigOptionParams) bool {
					return a.SessionID == b.SessionID && a.ConfigID == b.ConfigID && a.Value == b.Value
				}) {
					t.Errorf("config options: got %#v, want %#v", got, want)
				}
			})
			t.Run("permission", func(t *testing.T) {
				responses := strings.Join([]string{
					`{"jsonrpc":"2.0","id":1,"result":{}}`,
					`{"jsonrpc":"2.0","id":2,"result":{"sessionId":"session-1"}}`,
					`{"jsonrpc":"2.0","id":"permission-1","method":"session/request_permission","params":{"sessionId":"session-1","toolCall":{"toolCallId":"tool-1"},"options":[{"optionId":"allow","kind":"allow_once","name":"Allow"},{"optionId":"reject","kind":"reject_once","name":"Reject"}]}}`,
					`{"jsonrpc":"2.0","method":"session/update","params":{"sessionId":"session-1","update":{"sessionUpdate":"agent_message_chunk","content":{"type":"text","text":"done"}}}}`,
					`{"jsonrpc":"2.0","id":3,"result":{"stopReason":"end_turn"}}`,
				}, "\n")
				for _, tc := range []struct {
					name string
					opts []genai.ProviderOption
					want string
				}{
					{"rejected", nil, "reject"},
					{"allowed", []genai.ProviderOption{&ProviderOption{DangerouslySkipPermissions: true}}, "allow"},
				} {
					t.Run(tc.name, func(t *testing.T) {
						c, written := newCaptureClient(t, responses, tc.opts...)
						if _, err := c.GenSync(t.Context(), genai.Messages{genai.NewTextMessage("use a tool")}); err != nil {
							t.Fatalf("GenSync: %v", err)
						}
						if outcome, option := responseOutcome(t, written, `"permission-1"`); outcome != "selected" || option != tc.want {
							t.Errorf("permission outcome: %s %s, want selected %s", outcome, option, tc.want)
						}
					})
				}
			})
		})
		t.Run("error", func(t *testing.T) {
			t.Run("error_result", func(t *testing.T) {
				c := newOutputClient(t, strings.Join([]string{
					`{"jsonrpc":"2.0","id":1,"result":{}}`,
					`{"jsonrpc":"2.0","id":2,"result":{"sessionId":"session-1"}}`,
					`{"jsonrpc":"2.0","id":3,"error":{"code":429,"message":"rate limit exceeded","data":{"providerId":"deepseek-official"}}}`,
				}, "\n"))
				_, err := c.GenSync(t.Context(), genai.Messages{genai.NewTextMessage("cause error")})
				if err == nil || !strings.HasPrefix(err.Error(), "dsh: ") || !strings.Contains(err.Error(), "rate limit exceeded") || !strings.Contains(err.Error(), `"providerId":"deepseek-official"`) {
					t.Fatalf("unexpected error: %v", err)
				}
			})
			t.Run("invalid_initialize_result", func(t *testing.T) {
				c := newOutputClient(t, `{"jsonrpc":"2.0","id":1,"result":{"protocolVersion":1,"unexpected":true}}`)
				_, err := c.GenSync(t.Context(), genai.Messages{genai.NewTextMessage("hello")})
				if err == nil || !strings.HasPrefix(err.Error(), "dsh: ") || !strings.Contains(err.Error(), "parse initialize response") {
					t.Fatalf("expected initialize decode error, got %v", err)
				}
			})
		})
	})

	t.Run("GenStream", func(t *testing.T) {
		t.Run("valid", func(t *testing.T) {
			t.Run("hello", func(t *testing.T) {
				c := newTestClient(t, "GenStream_hello", genai.ProviderOptionModel("deepseek-official/deepseek-v4-flash"))
				seq, finish := c.GenStream(t.Context(), genai.Messages{genai.NewTextMessage("say hello")})

				var sb strings.Builder
				for r := range seq {
					sb.WriteString(r.Text)
				}
				res, err := finish()
				if err != nil {
					t.Fatalf("finish: %v", err)
				}
				if !strings.Contains(strings.ToLower(sb.String()), "hello") {
					t.Errorf("streamed text: got %q, want something containing hello", sb.String())
				}
				if res.Usage.FinishReason != genai.FinishedStop {
					t.Errorf("FinishReason: got %q, want %q", res.Usage.FinishReason, genai.FinishedStop)
				}
				if res.Usage.ContextTokens != 7496 || res.Usage.ContextWindow != 1000000 {
					t.Errorf("context usage: got %d/%d, want 7496/1000000", res.Usage.ContextTokens, res.Usage.ContextWindow)
				}
			})
			t.Run("thinking_delta", func(t *testing.T) {
				c := newTestClient(t, "GenStream_thinking", genai.ProviderOptionModel("deepseek-official/deepseek-v4-pro"))
				seq, finish := c.GenStream(t.Context(), genai.Messages{genai.NewTextMessage("say hello")})

				var text, reasoning strings.Builder
				for r := range seq {
					text.WriteString(r.Text)
					reasoning.WriteString(r.Reasoning)
				}
				if _, err := finish(); err != nil {
					t.Fatalf("finish: %v", err)
				}
				if !strings.Contains(strings.ToLower(text.String()), "hello") {
					t.Errorf("streamed text: got %q, want something containing hello", text.String())
				}
				if reasoning.Len() == 0 {
					t.Errorf("streamed reasoning: got empty, want non-empty")
				}
			})
		})
		t.Run("error", func(t *testing.T) {
			c := newOutputClient(t, strings.Join([]string{
				`{"jsonrpc":"2.0","id":1,"result":{}}`,
				`{"jsonrpc":"2.0","id":2,"result":{"sessionId":"session-1"}}`,
				`{"jsonrpc":"2.0","id":3,"error":{"code":500,"message":"internal server error"}}`,
			}, "\n"))
			seq, finish := c.GenStream(t.Context(), genai.Messages{genai.NewTextMessage("hello")})
			for range seq {
			}
			if _, err := finish(); err == nil || !strings.Contains(err.Error(), "internal server error") {
				t.Fatalf("unexpected error: %v", err)
			}
		})
	})

	t.Run("ListModels", func(t *testing.T) {
		t.Run("valid", func(t *testing.T) {
			c := newTestClient(t, "ListModels")
			models, err := c.ListModels(t.Context())
			if err != nil {
				t.Fatalf("ListModels: %v", err)
			}
			var found bool
			for _, m := range models {
				if m.GetID() == "deepseek-official/deepseek-v4-flash" {
					found = true
				}
			}
			if !found {
				ids := make([]string, len(models))
				for i, m := range models {
					ids[i] = m.GetID()
				}
				t.Errorf("deepseek-official/deepseek-v4-flash not found in models: %v", ids)
			}
		})
		t.Run("error", func(t *testing.T) {
			line := func(id int64, result any) string {
				raw, err := json.Marshal(result)
				if err != nil {
					t.Fatal(err)
				}
				out, err := json.Marshal(acp.JSONRPCResponse{JSONRPC: "2.0", ID: json.RawMessage(strconv.FormatInt(id, 10)), Result: raw})
				if err != nil {
					t.Fatal(err)
				}
				return string(out)
			}
			for _, route := range []string{`["a","b","c"]`, `["","model"]`, `["provider",""]`, `["acme/prod","model"]`, `plain`} {
				current, err := json.Marshal(route)
				if err != nil {
					t.Fatal(err)
				}
				sn := acp.SessionNewResult{SessionID: "session-1", ConfigOptions: []acp.SessionConfigOption{{
					ID:           acp.ConfigOptionModel,
					Name:         "Model",
					Category:     acp.ConfigOptionCategoryModel,
					Type:         acp.ConfigOptionTypeSelect,
					CurrentValue: current,
					Options:      []acp.ConfigOptionValue{{Value: route, Name: "bad"}},
				}}}
				output := line(1, acp.InitializeResult{}) + "\n" + line(2, sn)
				if _, err := newOutputClient(t, output).ListModels(t.Context()); err == nil || !strings.Contains(err.Error(), "invalid model value") {
					t.Errorf("%s: expected invalid model error, got %v", route, err)
				}
			}
		})
	})
}

func TestProviderOption(t *testing.T) {
	t.Run("Validate", func(t *testing.T) {
		t.Run("valid", func(t *testing.T) {
			if err := (&ProviderOption{Effort: EffortLow, DangerouslySkipPermissions: true, ExtraArgs: []string{"--patch", "/tmp/acp.patch.yml"}}).Validate(); err != nil {
				t.Error(err)
			}
		})
		t.Run("error", func(t *testing.T) {
			for _, p := range []ProviderOption{{Effort: "   "}, {ExtraArgs: []string{""}}, {ExtraArgs: []string{"--profile", "other"}}} {
				if err := p.Validate(); err == nil {
					t.Errorf("%+v: expected error", p)
				}
			}
		})
	})
}

func TestGenOption(t *testing.T) {
	t.Run("Validate", func(t *testing.T) {
		t.Run("error", func(t *testing.T) {
			if err := (&GenOption{Effort: "   "}).Validate(); err == nil {
				t.Fatal("expected error for whitespace effort")
			}
		})
	})
	t.Run("unsupported", func(t *testing.T) {
		_, err := parseOpts(&ProviderOption{}, []genai.GenOption{
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
		if _, err := parseOpts(&ProviderOption{}, []genai.GenOption{&genai.GenOptionText{}}); err != nil {
			t.Fatalf("empty GenOptionText: %v", err)
		}
	})
}

func responseOutcome(t *testing.T, b *bytes.Buffer, id string) (string, string) {
	for line := range strings.SplitSeq(strings.TrimSpace(b.String()), "\n") {
		var got struct {
			ID     json.RawMessage              `json:"id"`
			Result acp.PermissionResponseResult `json:"result"`
		}
		if err := json.Unmarshal([]byte(line), &got); err != nil {
			t.Fatalf("unmarshal response: %v", err)
		}
		if string(got.ID) == id {
			return string(got.Result.Outcome.Outcome), got.Result.Outcome.OptionID
		}
	}
	t.Fatalf("response %s not found in %q", id, b.String())
	return "", ""
}

type nopWriteCloser struct {
	io.Writer
}

func (nopWriteCloser) Close() error { return nil }

func init() {
	internal.BeLenient = false
}
