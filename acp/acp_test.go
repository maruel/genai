// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Tests for the shared ACP client.

package acp

import (
	"bytes"
	"encoding/json"
	"fmt"
	"slices"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"

	"github.com/maruel/genai"
	"github.com/maruel/genai/internal"
)

func TestSession(t *testing.T) {
	newOptions := func() *Options {
		return &Options{
			ClientName:         "genai-test",
			ClientVersion:      "1.0.0",
			Cwd:                "/tmp",
			ResumeMethod:       MethodSessionResume,
			ModelID:            ConfigOptionModel,
			EffortID:           "effort",
			ModeID:             "mode",
			AllowUnlistedModel: true,
		}
	}
	sessionResponse := `{"jsonrpc":"2.0","id":2,"result":{"sessionId":"session-1","configOptions":[{"id":"model","name":"Model","category":"model","type":"select","currentValue":"openai/gpt-5.4","options":[{"value":"openai/gpt-5.4","name":"GPT-5.4"}]},{"id":"effort","name":"Effort","category":"thought_level","type":"select","currentValue":"low","options":[{"value":"low","name":"Low"},{"value":"high","name":"High"},{"value":"xhigh","name":"Extra high"}]},{"id":"mode","name":"Mode","category":"mode","type":"select","currentValue":"build","options":[{"value":"build","name":"Build"},{"value":"plan","name":"Plan"}]}]}}`
	newSession := func(t *testing.T, responses string, o *Options, req SessionRequest) (*Session, *bytes.Buffer) {
		var written bytes.Buffer
		s, err := Handshake(&written, strings.NewReader(responses), o, req)
		if err != nil {
			t.Fatalf("Handshake: %v", err)
		}
		return s, &written
	}

	t.Run("Handshake", func(t *testing.T) {
		t.Run("valid", func(t *testing.T) {
			for _, tc := range []struct {
				name string
				req  SessionRequest
				want []SetSessionConfigOptionParams
			}{
				{"current_model", SessionRequest{Effort: "high"}, []SetSessionConfigOptionParams{{SessionID: "session-1", ConfigID: "effort", Value: "high"}}},
				{"requested_model", SessionRequest{Model: "openai/gpt-5.4", Effort: "xhigh", Mode: "plan"}, []SetSessionConfigOptionParams{
					{SessionID: "session-1", ConfigID: ConfigOptionModel, Value: "openai/gpt-5.4"},
					{SessionID: "session-1", ConfigID: "effort", Value: "xhigh"},
					{SessionID: "session-1", ConfigID: "mode", Value: "plan"},
				}},
			} {
				t.Run(tc.name, func(t *testing.T) {
					var sb strings.Builder
					sb.WriteString(`{"jsonrpc":"2.0","id":1,"result":{"agentCapabilities":{"promptCapabilities":{"image":true}}}}`)
					sb.WriteString("\n")
					sb.WriteString(sessionResponse)
					for i := range tc.want {
						sb.WriteString("\n")
						fmt.Fprintf(&sb, `{"jsonrpc":"2.0","id":%d,"result":{"configOptions":[{"id":"model","name":"Model","category":"model","type":"select","currentValue":"openai/gpt-5.4","options":[{"value":"openai/gpt-5.4","name":"GPT-5.4"}]},{"id":"effort","name":"Effort","category":"thought_level","type":"select","currentValue":"low","options":[{"value":"low","name":"Low"},{"value":"high","name":"High"},{"value":"xhigh","name":"Extra high"}]},{"id":"mode","name":"Mode","category":"mode","type":"select","currentValue":"build","options":[{"value":"build","name":"Build"},{"value":"plan","name":"Plan"}]}]}}`, 3+i)
					}
					_, written := newSession(t, sb.String(), newOptions(), tc.req)

					var got []SetSessionConfigOptionParams
					for line := range strings.SplitSeq(strings.TrimSpace(written.String()), "\n") {
						var req JSONRPCRequest
						if err := json.Unmarshal([]byte(line), &req); err != nil {
							t.Fatalf("unmarshal request: %v", err)
						}
						if req.Method == MethodSessionSetConfigOption {
							var params SetSessionConfigOptionParams
							if err := json.Unmarshal(req.Params, &params); err != nil {
								t.Fatalf("unmarshal session/set_config_option params: %v", err)
							}
							got = append(got, params)
						}
					}
					if !slices.EqualFunc(got, tc.want, func(a, b SetSessionConfigOptionParams) bool {
						return a.SessionID == b.SessionID && a.ConfigID == b.ConfigID && a.Value == b.Value && bytes.Equal(a.Meta, b.Meta)
					}) {
						t.Errorf("config options: got %#v, want %#v", got, tc.want)
					}
				})
			}
			t.Run("input_modalities", func(t *testing.T) {
				for _, tc := range []struct {
					name string
					init string
					want []genai.Modality
				}{
					{"text", `{"jsonrpc":"2.0","id":1,"result":{}}`, []genai.Modality{genai.ModalityText}},
					{"image", `{"jsonrpc":"2.0","id":1,"result":{"agentCapabilities":{"promptCapabilities":{"image":true}}}}`, []genai.Modality{genai.ModalityText, genai.ModalityImage}},
				} {
					t.Run(tc.name, func(t *testing.T) {
						s, _ := newSession(t, tc.init+"\n"+sessionResponse, newOptions(), SessionRequest{})
						if !slices.Equal(s.InputModalities, tc.want) {
							t.Errorf("InputModalities: got %v, want %v", s.InputModalities, tc.want)
						}
					})
				}
			})
			t.Run("cwd", func(t *testing.T) {
				o := newOptions()
				o.Cwd = "/work"
				responses := `{"jsonrpc":"2.0","id":1,"result":{}}` + "\n" + sessionResponse
				_, written := newSession(t, responses, o, SessionRequest{})
				var got map[string]json.RawMessage
				for line := range strings.SplitSeq(strings.TrimSpace(written.String()), "\n") {
					var req JSONRPCRequest
					if err := json.Unmarshal([]byte(line), &req); err != nil {
						t.Fatalf("unmarshal request: %v", err)
					}
					if req.Method != MethodSessionNew {
						continue
					}
					if err := json.Unmarshal(req.Params, &got); err != nil {
						t.Fatalf("unmarshal session/new params: %v", err)
					}
				}
				var cwd string
				if err := json.Unmarshal(got["cwd"], &cwd); err != nil || cwd != "/work" {
					t.Errorf("cwd: %q, %v", cwd, err)
				}
			})
			t.Run("load", func(t *testing.T) {
				o := newOptions()
				o.Cwd = "/work"
				o.ResumeMethod = MethodSessionLoad
				responses := `{"jsonrpc":"2.0","id":1,"result":{}}` + "\n" + `{"jsonrpc":"2.0","id":2,"result":{}}`
				_, written := newSession(t, responses, o, SessionRequest{ResumeSessionID: "prev"})
				var got map[string]json.RawMessage
				for line := range strings.SplitSeq(strings.TrimSpace(written.String()), "\n") {
					var req JSONRPCRequest
					if err := json.Unmarshal([]byte(line), &req); err != nil {
						t.Fatalf("unmarshal request: %v", err)
					}
					if req.Method != MethodSessionLoad {
						continue
					}
					if err := json.Unmarshal(req.Params, &got); err != nil {
						t.Fatalf("unmarshal session/load params: %v", err)
					}
				}
				// The method is not observable once decoded, so assert the
				// fields ACP requires: session ID, working directory, and a
				// present mcpServers array.
				if got == nil {
					t.Fatal("session/load request not sent")
				}
				var id string
				if err := json.Unmarshal(got["sessionId"], &id); err != nil || id != "prev" {
					t.Errorf("sessionId: %q, %v", id, err)
				}
				var cwd string
				if err := json.Unmarshal(got["cwd"], &cwd); err != nil || cwd != "/work" {
					t.Errorf("cwd: %q, %v", cwd, err)
				}
				if mcp, ok := got["mcpServers"]; !ok || string(mcp) != "[]" {
					t.Errorf("mcpServers: %s, %v", mcp, ok)
				}
			})
			t.Run("permission_cancelled", func(t *testing.T) {
				responses := strings.Join([]string{
					`{"jsonrpc":"2.0","id":"permission-1","method":"session/request_permission","params":{"sessionId":"session-1","toolCall":{"toolCallId":"tool-1"},"options":[{"optionId":"allow","kind":"allow_once","name":"Allow once"}]}}`,
					`{"jsonrpc":"2.0","id":1,"result":{}}`,
					sessionResponse,
				}, "\n")
				_, written := newSession(t, responses, newOptions(), SessionRequest{})
				if got := permissionOutcome(t, findResponse(t, written, `"permission-1"`)); got.Outcome != PermissionOutcomeCancelled {
					t.Errorf("outcome: %#v", got)
				}
			})
		})

		t.Run("error", func(t *testing.T) {
			init := `{"jsonrpc":"2.0","id":1,"result":{}}` + "\n" + sessionResponse
			t.Run("missing_cwd", func(t *testing.T) {
				o := newOptions()
				o.Cwd = ""
				_, err := Handshake(&bytes.Buffer{}, strings.NewReader(""), o, SessionRequest{})
				if err == nil || !strings.Contains(err.Error(), "cwd is required") {
					t.Fatalf("expected missing cwd error, got %v", err)
				}
			})
			t.Run("unsupported_effort", func(t *testing.T) {
				responses := `{"jsonrpc":"2.0","id":1,"result":{}}` + "\n" +
					`{"jsonrpc":"2.0","id":2,"result":{"sessionId":"session-1","configOptions":[{"id":"model","name":"Model","category":"model","type":"select","currentValue":"openai/gpt-5.4","options":[{"value":"openai/gpt-5.4","name":"GPT-5.4"}]}]}}`
				_, err := Handshake(&bytes.Buffer{}, strings.NewReader(responses), newOptions(), SessionRequest{Effort: "custom"})
				if err == nil || !strings.Contains(err.Error(), `does not expose "effort"`) {
					t.Fatalf("expected missing effort option error, got %v", err)
				}
			})
			t.Run("unlisted_effort", func(t *testing.T) {
				_, err := Handshake(&bytes.Buffer{}, strings.NewReader(init), newOptions(), SessionRequest{Effort: "custom"})
				if err == nil || !strings.Contains(err.Error(), `"effort" "custom" is unavailable`) {
					t.Fatalf("expected unlisted effort error, got %v", err)
				}
			})
			t.Run("unlisted_model", func(t *testing.T) {
				o := newOptions()
				o.AllowUnlistedModel = false
				_, err := Handshake(&bytes.Buffer{}, strings.NewReader(init), o, SessionRequest{Model: "other/model"})
				if err == nil || !strings.Contains(err.Error(), `"model" "other/model" is unavailable`) {
					t.Fatalf("expected unlisted model error, got %v", err)
				}
			})
			t.Run("malformed_response", func(t *testing.T) {
				_, err := Handshake(&bytes.Buffer{}, strings.NewReader(`{"jsonrpc":"2.0","id":1`), newOptions(), SessionRequest{})
				if err == nil || !strings.Contains(err.Error(), "unmarshal JSON-RPC message") {
					t.Fatalf("expected malformed JSON error, got %v", err)
				}
			})
			t.Run("unexpected_id", func(t *testing.T) {
				_, err := Handshake(&bytes.Buffer{}, strings.NewReader(`{"jsonrpc":"2.0","id":2,"result":{}}`), newOptions(), SessionRequest{})
				if err == nil || !strings.Contains(err.Error(), "unexpected JSON-RPC response id 2, want 1") {
					t.Fatalf("expected unexpected ID error, got %v", err)
				}
			})
			t.Run("unknown_envelope_field", func(t *testing.T) {
				_, err := Handshake(&bytes.Buffer{}, strings.NewReader(`{"jsonrpc":"2.0","id":1,"result":{},"unexpected":true}`), newOptions(), SessionRequest{})
				if err == nil || !strings.Contains(err.Error(), "unmarshal JSON-RPC message") {
					t.Fatalf("expected unknown envelope field error, got %v", err)
				}
			})
			t.Run("jsonrpc_error", func(t *testing.T) {
				_, err := Handshake(&bytes.Buffer{}, strings.NewReader(`{"jsonrpc":"2.0","id":1,"error":{"code":-32602,"message":"invalid params","data":{"field":"cwd"}}}`), newOptions(), SessionRequest{})
				if err == nil || !strings.Contains(err.Error(), `JSON-RPC error -32602: invalid params: {"field":"cwd"}`) {
					t.Fatalf("expected complete JSON-RPC error, got %v", err)
				}
			})
			t.Run("exited", func(t *testing.T) {
				_, err := Handshake(&bytes.Buffer{}, strings.NewReader(""), newOptions(), SessionRequest{})
				if err == nil || !strings.Contains(err.Error(), "exited during handshake") {
					t.Fatalf("expected exit error, got %v", err)
				}
			})
		})
	})

	t.Run("ModelChoices", func(t *testing.T) {
		// DeepSeek Harness groups model values by provider.
		responses := `{"jsonrpc":"2.0","id":1,"result":{}}` + "\n" +
			`{"jsonrpc":"2.0","id":2,"result":{"sessionId":"session-1","configOptions":[{"id":"model","name":"Model","category":"model","type":"select","currentValue":"first/a","options":[{"group":"first","name":"First","options":[{"value":"first/a","name":"A"}]},{"group":"second","name":"Second","options":[{"value":"second/b","name":"B"}]}]}]}}`
		s, _ := newSession(t, responses, newOptions(), SessionRequest{})
		got := s.ModelChoices()
		want := []ModelChoice{{Value: "first/a", Name: "A"}, {Value: "second/b", Name: "B"}}
		if !slices.Equal(got, want) {
			t.Errorf("ModelChoices: got %#v, want %#v", got, want)
		}
	})

	t.Run("SendPrompt", func(t *testing.T) {
		t.Run("valid", func(t *testing.T) {
			s, written := newSession(t, `{"jsonrpc":"2.0","id":1,"result":{"agentCapabilities":{"promptCapabilities":{"image":true}}}}`+"\n"+sessionResponse, newOptions(), SessionRequest{})
			msg := genai.Message{Requests: []genai.Request{
				{Text: "hello"},
				{Doc: genai.Doc{Filename: "a.txt", Src: bytes.NewReader([]byte("body"))}},
			}}
			id, err := s.SendPrompt(&msg)
			if err != nil {
				t.Fatalf("SendPrompt: %v", err)
			}
			lines := strings.Split(strings.TrimSpace(written.String()), "\n")
			var req JSONRPCRequest
			if err := json.Unmarshal([]byte(lines[len(lines)-1]), &req); err != nil {
				t.Fatalf("unmarshal request: %v", err)
			}
			var params SessionPromptParams
			if err := json.Unmarshal(req.Params, &params); err != nil {
				t.Fatalf("unmarshal session/prompt params: %v", err)
			}
			if req.Method != MethodSessionPrompt || req.ID != id || params.SessionID != "session-1" {
				t.Errorf("unexpected request: %#v", req)
			}
			want := []PromptContent{{Type: ContentText, Text: "hello"}, {Type: ContentText, Text: "body"}}
			if diff := cmp.Diff(want, params.Prompt); diff != "" {
				t.Errorf("prompt (-want +got):\n%s", diff)
			}
		})
		t.Run("error", func(t *testing.T) {
			s, _ := newSession(t, `{"jsonrpc":"2.0","id":1,"result":{}}`+"\n"+sessionResponse, newOptions(), SessionRequest{})
			t.Run("empty_message", func(t *testing.T) {
				if _, err := s.SendPrompt(&genai.Message{}); err == nil || !strings.Contains(err.Error(), "has no content") {
					t.Fatalf("expected empty message error, got %v", err)
				}
			})
			t.Run("image_not_supported", func(t *testing.T) {
				msg := genai.Message{Requests: []genai.Request{{Doc: genai.Doc{URL: "https://example.com/image.png"}}}}
				if _, err := s.SendPrompt(&msg); err == nil || !strings.Contains(err.Error(), "does not support image input") {
					t.Fatalf("expected image error, got %v", err)
				}
			})
			t.Run("url_scheme", func(t *testing.T) {
				imageSession, _ := newSession(t, `{"jsonrpc":"2.0","id":1,"result":{"agentCapabilities":{"promptCapabilities":{"image":true}}}}`+"\n"+sessionResponse, newOptions(), SessionRequest{})
				msg := genai.Message{Requests: []genai.Request{{Doc: genai.Doc{URL: "file:///tmp/image.png"}}}}
				if _, err := imageSession.SendPrompt(&msg); err == nil || !strings.Contains(err.Error(), "unsupported image URL scheme") {
					t.Fatalf("expected URL scheme error, got %v", err)
				}
			})
		})
	})

	t.Run("ReadTurn", func(t *testing.T) {
		newTurnSession := func(t *testing.T, input string) (*Session, *bytes.Buffer) {
			responses := `{"jsonrpc":"2.0","id":1,"result":{}}` + "\n" + sessionResponse
			if input != "" {
				responses += "\n" + input
			}
			return newSession(t, responses, newOptions(), SessionRequest{})
		}

		t.Run("valid", func(t *testing.T) {
			t.Run("turn", func(t *testing.T) {
				input := strings.Join([]string{
					`{"jsonrpc":"2.0","method":"session/update","params":{"sessionId":"session-1","update":{"sessionUpdate":"agent_thought_chunk","content":{"type":"text","text":"thinking"}}}}`,
					`{"jsonrpc":"2.0","method":"session/update","params":{"sessionId":"session-1","update":{"sessionUpdate":"agent_message_chunk","content":{"type":"text","text":"Hello"}}}}`,
					`{"jsonrpc":"2.0","id":"permission-1","method":"session/request_permission","params":{"sessionId":"session-1","toolCall":{"toolCallId":"tool-1"},"options":[{"optionId":"once","kind":"allow_once","name":"Allow once"}]}}`,
					`{"jsonrpc":"2.0","id":3,"result":{"stopReason":"end_turn"}}`,
				}, "\n")
				s, out := newTurnSession(t, input)
				var text, reasoning strings.Builder
				res, err := s.ReadTurn(3, true, func(t, r string) bool {
					text.WriteString(t)
					reasoning.WriteString(r)
					return true
				})
				if err != nil {
					t.Fatalf("ReadTurn: %v", err)
				}
				if text.String() != "Hello" || reasoning.String() != "thinking" {
					t.Errorf("deltas: text %q, reasoning %q", text.String(), reasoning.String())
				}
				if res.Usage.FinishReason != genai.FinishedStop {
					t.Errorf("FinishReason: %q", res.Usage.FinishReason)
				}
				if len(res.Replies) != 3 || res.Replies[0].Reasoning != "thinking" || res.Replies[1].Text != "Hello" || res.Replies[2].Opaque[SessionIDKey] != "session-1" {
					t.Errorf("replies: %#v", res.Replies)
				}
				got := permissionOutcome(t, findResponse(t, out, `"permission-1"`))
				if got.Outcome != PermissionOutcomeSelected || got.OptionID != "once" {
					t.Errorf("outcome: %#v", got)
				}
			})
			t.Run("permission_rejected", func(t *testing.T) {
				input := `{"jsonrpc":"2.0","id":"permission-1","method":"session/request_permission","params":{"sessionId":"session-1","toolCall":{"toolCallId":"tool-1"},"options":[{"optionId":"allow","kind":"allow_once","name":"Allow once"},{"optionId":"reject","kind":"reject_once","name":"Reject"}]}}` + "\n" + `{"jsonrpc":"2.0","id":3,"result":{"stopReason":"end_turn"}}`
				s, out := newTurnSession(t, input)
				if _, err := s.ReadTurn(3, false, func(string, string) bool { return true }); err != nil {
					t.Fatalf("ReadTurn: %v", err)
				}
				got := permissionOutcome(t, findResponse(t, out, `"permission-1"`))
				if got.Outcome != PermissionOutcomeSelected || got.OptionID != "reject" {
					t.Errorf("outcome: %#v", got)
				}
			})
			t.Run("permission_cancelled", func(t *testing.T) {
				input := `{"jsonrpc":"2.0","id":"permission-1","method":"session/request_permission","params":{"sessionId":"session-1","toolCall":{"toolCallId":"tool-1"},"options":[{"optionId":"allow","kind":"allow_once","name":"Allow once"}]}}` + "\n" + `{"jsonrpc":"2.0","id":3,"result":{"stopReason":"end_turn"}}`
				s, out := newTurnSession(t, input)
				if _, err := s.ReadTurn(3, false, func(string, string) bool { return true }); err != nil {
					t.Fatalf("ReadTurn: %v", err)
				}
				if got := permissionOutcome(t, findResponse(t, out, `"permission-1"`)); got.Outcome != PermissionOutcomeCancelled {
					t.Errorf("outcome: %#v", got)
				}
			})
			t.Run("unsupported_method", func(t *testing.T) {
				input := `{"jsonrpc":"2.0","id":"fs-1","method":"fs/write_text_file","params":{"sessionId":"session-1","path":"a.txt","content":"new"}}` + "\n" + `{"jsonrpc":"2.0","id":3,"result":{"stopReason":"end_turn"}}`
				s, out := newTurnSession(t, input)
				if _, err := s.ReadTurn(3, false, func(string, string) bool { return true }); err != nil {
					t.Fatalf("ReadTurn: %v", err)
				}
				got := findResponse(t, out, `"fs-1"`)
				if got.Error.Code != -32601 || !strings.Contains(got.Error.Message, "fs/write_text_file") {
					t.Errorf("unexpected response: %#v", got.Error)
				}
			})
			t.Run("stream_cancel", func(t *testing.T) {
				input := `{"jsonrpc":"2.0","method":"session/update","params":{"sessionId":"session-1","update":{"sessionUpdate":"agent_message_chunk","content":{"type":"text","text":"partial"}}}}`
				s, out := newTurnSession(t, input)
				if _, err := s.ReadTurn(3, false, func(string, string) bool { return false }); err != nil {
					t.Fatalf("ReadTurn: %v", err)
				}
				var got JSONRPCRequest
				for line := range strings.SplitSeq(strings.TrimSpace(out.String()), "\n") {
					var req JSONRPCRequest
					if err := json.Unmarshal([]byte(line), &req); err != nil {
						t.Fatalf("unmarshal request: %v", err)
					}
					if req.Method == MethodSessionCancel {
						got = req
						break
					}
				}
				var params SessionCancelParams
				if err := json.Unmarshal(got.Params, &params); err != nil {
					t.Fatalf("unmarshal cancellation params: %v", err)
				}
				if got.Method != MethodSessionCancel || got.ID != 0 || params.SessionID != "session-1" {
					t.Errorf("unexpected cancellation: %#v", got)
				}
			})
		})
		t.Run("error", func(t *testing.T) {
			for _, tc := range []struct {
				name  string
				input string
				want  string
			}{
				{"malformed_json", `{"jsonrpc":"2.0","id":3`, "unmarshal JSON-RPC message"},
				{"unexpected_id", `{"jsonrpc":"2.0","id":4,"result":{}}`, "unexpected JSON-RPC response id 4, want 3"},
				{"unknown_envelope_field", `{"jsonrpc":"2.0","method":"session/update","params":{},"unexpected":true}`, "unmarshal JSON-RPC message"},
				{"known_update", `{"jsonrpc":"2.0","method":"session/update","params":{"sessionId":"session-1","update":{"sessionUpdate":"tool_call","toolCallId":7,"title":"read"}}}`, "unmarshal tool_call"},
				{"exited", "", "exited without prompt response"},
			} {
				t.Run(tc.name, func(t *testing.T) {
					s, _ := newTurnSession(t, tc.input)
					_, err := s.ReadTurn(3, false, func(string, string) bool { return true })
					if err == nil || !strings.Contains(err.Error(), tc.want) {
						t.Fatalf("got %v, want %q", err, tc.want)
					}
				})
			}
		})
	})
}

func findResponse(t *testing.T, b *bytes.Buffer, id string) *JSONRPCResponse {
	for line := range strings.SplitSeq(strings.TrimSpace(b.String()), "\n") {
		var got JSONRPCResponse
		if err := json.Unmarshal([]byte(line), &got); err != nil {
			t.Fatalf("unmarshal response: %v", err)
		}
		if string(got.ID) == id {
			return &got
		}
	}
	t.Fatalf("response %s not found in %q", id, b.String())
	return &JSONRPCResponse{}
}

func permissionOutcome(t *testing.T, got *JSONRPCResponse) PermissionOutcome {
	var result PermissionResponseResult
	if err := json.Unmarshal(got.Result, &result); err != nil {
		t.Fatalf("unmarshal permission result: %v", err)
	}
	return result.Outcome
}

func init() {
	internal.BeLenient = false
}
