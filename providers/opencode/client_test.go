// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Tests for the OpenCode provider client.

package opencode

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/maruel/genai"
	"github.com/maruel/genai/internal"
	"github.com/maruel/genai/internal/internaltest"
	"github.com/maruel/genai/internal/msgutil"
	"github.com/maruel/genai/internal/myrecorder"
	"github.com/maruel/genai/scoreboard"
	"github.com/maruel/genai/smoke/smoketest"
)

func newTestClient(t *testing.T, name string, opts ...genai.ProviderOption) *Client {
	rec := internaltest.NewSubprocessRecorder(t, name, "opencode", nil)
	opts = append(opts, genai.ProviderOptionStarterWrapper(rec.Wrap))
	c, err := New(opts...)
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
	c, err := New(opts...)
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
					r := internaltest.NewSubprocessRecorder(t, name, "opencode", nil)
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
				if got := c.ModelID(); got != tc.want {
					t.Errorf("got %q, want %q", got, tc.want)
				}
			})
		}
	})

	t.Run("effort_mapping", func(t *testing.T) {
		for _, tc := range []struct {
			name   string
			model  string
			effort Effort
			mode   Mode
			want   []SetSessionConfigOptionParams
		}{
			{"current_model", "", EffortHigh, "", []SetSessionConfigOptionParams{{SessionID: "session-1", ConfigID: ConfigOptionEffort, Value: "high"}}},
			{"requested_model", "openai/gpt-5.4", EffortXHigh, "plan", []SetSessionConfigOptionParams{
				{SessionID: "session-1", ConfigID: ConfigOptionModel, Value: "openai/gpt-5.4"},
				{SessionID: "session-1", ConfigID: ConfigOptionEffort, Value: "xhigh"},
				{SessionID: "session-1", ConfigID: ConfigOptionMode, Value: "plan"},
			}},
		} {
			t.Run(tc.name, func(t *testing.T) {
				var written bytes.Buffer
				var sb strings.Builder
				sb.WriteString(`{"jsonrpc":"2.0","id":1,"result":{"agentCapabilities":{"promptCapabilities":{"image":true}}}}`)
				sb.WriteString("\n")
				sb.WriteString(`{"jsonrpc":"2.0","id":2,"result":{"sessionId":"session-1","configOptions":[{"id":"model","name":"Model","category":"model","type":"select","currentValue":"openai/gpt-5.4","options":[{"value":"openai/gpt-5.4","name":"GPT-5.4"}]},{"id":"effort","name":"Effort","category":"thought_level","type":"select","currentValue":"low","options":[{"value":"low","name":"Low"},{"value":"high","name":"High"},{"value":"xhigh","name":"Extra high"}]},{"id":"mode","name":"Mode","category":"mode","type":"select","currentValue":"build","options":[{"value":"build","name":"Build"},{"value":"plan","name":"Plan"}]}]}}`)
				for i := range tc.want {
					sb.WriteString("\n")
					fmt.Fprintf(&sb, `{"jsonrpc":"2.0","id":%d,"result":{"configOptions":[{"id":"model","name":"Model","category":"model","type":"select","currentValue":"openai/gpt-5.4","options":[{"value":"openai/gpt-5.4","name":"GPT-5.4"}]},{"id":"effort","name":"Effort","category":"thought_level","type":"select","currentValue":"low","options":[{"value":"low","name":"Low"},{"value":"high","name":"High"},{"value":"xhigh","name":"Extra high"}]},{"id":"mode","name":"Mode","category":"mode","type":"select","currentValue":"build","options":[{"value":"build","name":"Build"},{"value":"plan","name":"Plan"}]}]}}`, 3+i)
				}
				responses := sb.String()
				if _, err := handshake(&written, newScanner(strings.NewReader(responses)), tc.model, tc.effort, tc.mode, ""); err != nil {
					t.Fatalf("handshake: %v", err)
				}

				var got []SetSessionConfigOptionParams
				for _, line := range strings.Split(strings.TrimSpace(written.String()), "\n") {
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
	})

	t.Run("unsupported_effort", func(t *testing.T) {
		responses := `{"jsonrpc":"2.0","id":1,"result":{}}` + "\n" +
			`{"jsonrpc":"2.0","id":2,"result":{"sessionId":"session-1","configOptions":[{"id":"model","name":"Model","category":"model","type":"select","currentValue":"openai/gpt-5.4","options":[{"value":"openai/gpt-5.4","name":"GPT-5.4"}]}]}}`
		_, err := handshake(&bytes.Buffer{}, newScanner(strings.NewReader(responses)), "", "custom", "", "")
		if err == nil {
			t.Fatal("expected unsupported effort error")
		}
		if !strings.Contains(err.Error(), "does not expose \"effort\"") {
			t.Errorf("unexpected error: %v", err)
		}
	})

	t.Run("doc_to_prompt_content", func(t *testing.T) {
		t.Run("url_image", func(t *testing.T) {
			got, err := docToPromptContent(genai.Doc{URL: "https://example.com/image.png"}, true)
			if err != nil {
				t.Fatal(err)
			}
			if got.Type != ContentImage || got.URI != "https://example.com/image.png" || got.MimeType != "image/png" {
				t.Errorf("got %#v", got)
			}
		})
		t.Run("unsupported_url_scheme", func(t *testing.T) {
			_, err := docToPromptContent(genai.Doc{URL: "file:///tmp/image.png"}, true)
			if err == nil {
				t.Fatal("expected unsupported URL scheme error")
			}
		})
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
			msgs1 := genai.Messages{genai.NewTextMessage("Remember this color: cerulean. Just confirm you noted it.")}
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
			if !strings.Contains(err.Error(), "rate limit exceeded") || !strings.Contains(err.Error(), `"providerId":"openai"`) {
				t.Errorf("unexpected error message: %v", err)
			}
		})
		t.Run("invalid_initialize_result", func(t *testing.T) {
			c := newOutputClient(t, `{"jsonrpc":"2.0","id":1,"result":{"protocolVersion":1,"unexpected":true}}`)
			_, err := c.GenSync(t.Context(), genai.Messages{genai.NewTextMessage("hello")})
			if err == nil || !strings.Contains(err.Error(), "parse initialize response") {
				t.Fatalf("expected initialize decode error, got %v", err)
			}
		})
	})

	t.Run("agent_requests", func(t *testing.T) {
		t.Run("permission", func(t *testing.T) {
			line := []byte(`{"jsonrpc":"2.0","id":7,"method":"session/request_permission","params":{"sessionId":"session-1","toolCall":{"toolCallId":"tool-1","title":"Edit file","status":"pending","content":[{"type":"diff","path":"a.txt","oldText":"old","newText":"new"}],"_meta":{"source":"opencode"}},"options":[{"optionId":"once","kind":"allow_once","name":"Allow once"}],"_meta":{}}}`)
			var out bytes.Buffer
			if err := handleAgentRequest(&out, line); err != nil {
				t.Fatalf("handleAgentRequest: %v", err)
			}
			var got struct {
				ID     int `json:"id"`
				Result struct {
					Outcome struct {
						Outcome  string `json:"outcome"`
						OptionID string `json:"optionId"`
					} `json:"outcome"`
				} `json:"result"`
			}
			if err := json.Unmarshal(out.Bytes(), &got); err != nil {
				t.Fatalf("unmarshal response: %v", err)
			}
			if got.ID != 7 || got.Result.Outcome.Outcome != "selected" || got.Result.Outcome.OptionID != "once" {
				t.Errorf("unexpected permission response: %#v", got)
			}
		})

		t.Run("unsupported", func(t *testing.T) {
			line := []byte(`{"jsonrpc":"2.0","id":"request-1","method":"fs/write_text_file","params":{"sessionId":"session-1","path":"a.txt","content":"new"}}`)
			var out bytes.Buffer
			if err := handleAgentRequest(&out, line); err != nil {
				t.Fatalf("handleAgentRequest: %v", err)
			}
			var got struct {
				ID    string       `json:"id"`
				Error JSONRPCError `json:"error"`
			}
			if err := json.Unmarshal(out.Bytes(), &got); err != nil {
				t.Fatalf("unmarshal response: %v", err)
			}
			if got.ID != "request-1" || got.Error.Code != -32601 || !strings.Contains(got.Error.Message, "fs/write_text_file") {
				t.Errorf("unexpected method-not-found response: %#v", got)
			}
		})
	})

	t.Run("stream_cancel", func(t *testing.T) {
		line := `{"jsonrpc":"2.0","method":"session/update","params":{"sessionId":"session-1","update":{"sessionUpdate":"agent_message_chunk","messageId":"message-1","content":{"type":"text","text":"partial"}}}}`
		var out bytes.Buffer
		if _, err := readTurn(newScanner(strings.NewReader(line)), &out, "session-1", 3, func(string, string) bool { return false }); err != nil {
			t.Fatalf("readTurn: %v", err)
		}
		var got JSONRPCRequest
		if err := json.Unmarshal(out.Bytes(), &got); err != nil {
			t.Fatalf("unmarshal cancellation: %v", err)
		}
		if got.Method != MethodSessionCancel || got.ID != 0 {
			t.Errorf("unexpected cancellation: %#v", got)
		}
		var params SessionCancelParams
		if err := json.Unmarshal(got.Params, &params); err != nil {
			t.Fatalf("unmarshal cancellation params: %v", err)
		}
		if params.SessionID != "session-1" {
			t.Errorf("session ID: got %q, want session-1", params.SessionID)
		}
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

func TestReadResponse(t *testing.T) {
	t.Run("malformed JSON", func(t *testing.T) {
		_, err := readResponse(newScanner(strings.NewReader(`{"jsonrpc":"2.0","id":1`)), io.Discard, 1)
		if err == nil || !strings.Contains(err.Error(), "unmarshal JSON-RPC message") {
			t.Fatalf("expected malformed JSON error, got %v", err)
		}
	})

	t.Run("unexpected ID", func(t *testing.T) {
		_, err := readResponse(newScanner(strings.NewReader(`{"jsonrpc":"2.0","id":2,"result":{}}`)), io.Discard, 1)
		if err == nil || !strings.Contains(err.Error(), "unexpected JSON-RPC response id 2, want 1") {
			t.Fatalf("expected unexpected ID error, got %v", err)
		}
	})

	t.Run("JSON-RPC error data", func(t *testing.T) {
		_, err := readResponse(newScanner(strings.NewReader(`{"jsonrpc":"2.0","id":1,"error":{"code":-32602,"message":"invalid params","data":{"field":"cwd"}}}`)), io.Discard, 1)
		if err == nil || !strings.Contains(err.Error(), `JSON-RPC error -32602: invalid params: {"field":"cwd"}`) {
			t.Fatalf("expected complete JSON-RPC error, got %v", err)
		}
	})

	t.Run("agent request", func(t *testing.T) {
		input := `{"jsonrpc":"2.0","id":"permission-1","method":"session/request_permission","params":{"sessionId":"session-1","toolCall":{"toolCallId":"tool-1"},"options":[{"optionId":"once","kind":"allow_once","name":"Allow once"}]}}` + "\n" +
			`{"jsonrpc":"2.0","id":1,"result":{}}`
		var out bytes.Buffer
		if _, err := readResponse(newScanner(strings.NewReader(input)), &out, 1); err != nil {
			t.Fatal(err)
		}
		if !strings.Contains(out.String(), `"id":"permission-1"`) || !strings.Contains(out.String(), `"outcome":"selected"`) {
			t.Fatalf("agent response = %s", out.String())
		}
	})
}

func TestJSONRPCMessageClassification(t *testing.T) {
	for _, tc := range []struct {
		name         string
		input        string
		response     bool
		agentRequest bool
	}{
		{name: "notification", input: `{"method":"session/update"}`},
		{name: "response", input: `{"id":1,"result":{}}`, response: true},
		{name: "agent request", input: `{"id":"permission-1","method":"session/request_permission","params":{}}`, agentRequest: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var msg JSONRPCMessage
			if err := json.Unmarshal([]byte(tc.input), &msg); err != nil {
				t.Fatal(err)
			}
			if got := msg.IsResponse(); got != tc.response {
				t.Errorf("IsResponse() = %t, want %t", got, tc.response)
			}
			if got := msg.IsAgentRequest(); got != tc.agentRequest {
				t.Errorf("IsAgentRequest() = %t, want %t", got, tc.agentRequest)
			}
		})
	}
}

func TestReadTurn(t *testing.T) {
	t.Run("malformed JSON", func(t *testing.T) {
		_, err := readTurn(newScanner(strings.NewReader(`{"jsonrpc":"2.0","id":3`)), io.Discard, "session-1", 3, func(string, string) bool { return true })
		if err == nil || !strings.Contains(err.Error(), "unmarshal JSON-RPC message") {
			t.Fatalf("expected malformed JSON error, got %v", err)
		}
	})

	t.Run("unexpected ID", func(t *testing.T) {
		_, err := readTurn(newScanner(strings.NewReader(`{"jsonrpc":"2.0","id":4,"result":{}}`)), io.Discard, "session-1", 3, func(string, string) bool { return true })
		if err == nil || !strings.Contains(err.Error(), "unexpected JSON-RPC response id 4, want 3") {
			t.Fatalf("expected unexpected ID error, got %v", err)
		}
	})

	t.Run("JSON-RPC error data", func(t *testing.T) {
		_, err := readTurn(newScanner(strings.NewReader(`{"jsonrpc":"2.0","id":3,"error":{"code":-32000,"message":"turn failed","data":{"retry":false}}}`)), io.Discard, "session-1", 3, func(string, string) bool { return true })
		if err == nil || !strings.Contains(err.Error(), `JSON-RPC error -32000: turn failed: {"retry":false}`) {
			t.Fatalf("expected complete JSON-RPC error, got %v", err)
		}
	})

	t.Run("agent request", func(t *testing.T) {
		input := `{"jsonrpc":"2.0","id":"permission-1","method":"session/request_permission","params":{"sessionId":"session-1","toolCall":{"toolCallId":"tool-1"},"options":[{"optionId":"once","kind":"allow_once","name":"Allow once"}]}}` + "\n" +
			`{"jsonrpc":"2.0","id":3,"result":{"stopReason":"end_turn"}}`
		var out bytes.Buffer
		if _, err := readTurn(newScanner(strings.NewReader(input)), &out, "session-1", 3, func(string, string) bool { return true }); err != nil {
			t.Fatal(err)
		}
		if !strings.Contains(out.String(), `"id":"permission-1"`) || !strings.Contains(out.String(), `"outcome":"selected"`) {
			t.Fatalf("agent response = %s", out.String())
		}
	})
}

func TestParseSessionUpdateDelta(t *testing.T) {
	params := json.RawMessage(`{"sessionId":"session-1","update":{"sessionUpdate":"tool_call","toolCallId":7,"title":"read"}}`)
	_, _, err := parseSessionUpdateDelta(params)
	if err == nil || !strings.Contains(err.Error(), "unmarshal tool_call") {
		t.Fatalf("expected known update decode error, got %v", err)
	}
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
		if got := msgutil.ExtractOpaqueID(msgs, sessionIDKey); got != "sess-123" {
			t.Errorf("got %q, want %q", got, "sess-123")
		}
	})
	t.Run("not_found", func(t *testing.T) {
		msgs := genai.Messages{genai.NewTextMessage("hi")}
		if got := msgutil.ExtractOpaqueID(msgs, sessionIDKey); got != "" {
			t.Errorf("got %q, want empty", got)
		}
	})
}

func TestGenOption(t *testing.T) {
	t.Run("valid", func(t *testing.T) {
		for _, effort := range []Effort{"", EffortDefault, EffortNone, EffortMinimal, EffortLow, EffortMedium, EffortHigh, EffortXHigh, EffortMax, "custom"} {
			if err := (&GenOption{Effort: effort}).Validate(); err != nil {
				t.Errorf("Effort %q: unexpected error: %v", effort, err)
			}
		}
		if err := (&GenOption{Mode: "plan"}).Validate(); err != nil {
			t.Errorf("Mode: unexpected error: %v", err)
		}
	})
	t.Run("error", func(t *testing.T) {
		if err := (&GenOption{Effort: "   "}).Validate(); err == nil {
			t.Fatal("expected error for whitespace effort")
		}
		if err := (&GenOption{Mode: "   "}).Validate(); err == nil {
			t.Fatal("expected error for whitespace mode")
		}
	})
}

func TestScoreboard(t *testing.T) {
	s := Scoreboard()
	if len(s.Scenarios) == 0 {
		t.Fatal("scoreboard has no scenarios")
	}
}

func init() {
	internal.BeLenient = false
}
