// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Internal tests for the Codex provider.

package codex

import (
	"bufio"
	"bytes"
	"encoding/json"
	"errors"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/maruel/genai"
	"github.com/maruel/genai/base"
	"github.com/maruel/genai/internal"
)

func TestReasoningEffort(t *testing.T) {
	t.Run("valid", func(t *testing.T) {
		for _, v := range []ReasoningEffort{
			ReasoningEffortNone, ReasoningEffortMinimal, ReasoningEffortLow,
			ReasoningEffortMedium, ReasoningEffortHigh, ReasoningEffortXHigh,
		} {
			c, err := New(v)
			if err != nil {
				t.Fatalf("New(%q): %v", v, err)
			}
			if c.effort != v {
				t.Errorf("effort: got %q, want %q", c.effort, v)
			}
		}
	})
	t.Run("invalid", func(t *testing.T) {
		if _, err := New(ReasoningEffort("turbo")); err == nil {
			t.Fatal("expected error for invalid effort")
		}
	})
	t.Run("default", func(t *testing.T) {
		c, err := New()
		if err != nil {
			t.Fatal(err)
		}
		if c.effort != ReasoningEffortMedium {
			t.Errorf("default effort: got %q, want %q", c.effort, ReasoningEffortMedium)
		}
	})
}

func TestParseOpts(t *testing.T) {
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
				uerr, ok := errors.AsType[*base.ErrNotSupported](err)
				if !ok {
					t.Fatalf("expected ErrNotSupported, got %v", err)
				}
				if !slices.Contains(uerr.Options, tc.want) {
					t.Errorf("expected %q in unsupported, got %v", tc.want, uerr.Options)
				}
			})
		}
	})
}

func TestHandshake(t *testing.T) {
	t.Run("start", func(t *testing.T) {
		responses := strings.Join([]string{
			`{"id":1,"result":{}}`,
			`{"id":2,"result":{"data":[]}}`,
			`{"id":3,"result":{"thread":{"id":"thread"}}}`,
		}, "\n")
		var out bytes.Buffer
		threadID, err := handshake(&out, bufio.NewScanner(strings.NewReader(responses)), "model", "", "write commit messages")
		if err != nil {
			t.Fatal(err)
		}
		if threadID != "thread" {
			t.Errorf("thread ID = %q, want thread", threadID)
		}
		lines := strings.Split(strings.TrimSpace(out.String()), "\n")
		if len(lines) != 4 {
			t.Fatalf("wrote %d messages, want 4", len(lines))
		}
		var req JSONRPCRequest
		if err := json.Unmarshal([]byte(lines[3]), &req); err != nil {
			t.Fatal(err)
		}
		var params ThreadStartParams
		if err := json.Unmarshal(req.Params, &params); err != nil {
			t.Fatal(err)
		}
		if params.DeveloperInstructions != "write commit messages" {
			t.Errorf("developer instructions = %q, want write commit messages", params.DeveloperInstructions)
		}
	})
	t.Run("resume", func(t *testing.T) {
		responses := strings.Join([]string{
			`{"id":1,"result":{}}`,
			`{"id":2,"result":{"data":[]}}`,
			`{"id":3,"result":{"thread":{"id":"thread"},"sandbox":{"type":"workspaceWrite","writableRoots":["/src"],"networkAccess":false,"excludeTmpdirEnvVar":false,"excludeSlashTmp":false},"turnsBackwardsCursor":null,"itemsBackwardsCursor":"items"}}`,
		}, "\n")
		var out bytes.Buffer
		threadID, err := handshake(&out, bufio.NewScanner(strings.NewReader(responses)), "model", "thread", "")
		if err != nil {
			t.Fatal(err)
		}
		if threadID != "thread" {
			t.Fatalf("thread ID = %q, want thread", threadID)
		}
	})
}

func TestInitAndListModels(t *testing.T) {
	t.Run("pagination", func(t *testing.T) {
		responses := strings.Join([]string{
			`{"id":1,"result":{}}`,
			`{"id":2,"result":{"data":[{"id":"one"}],"nextCursor":"next"}}`,
			`{"id":3,"result":{"data":[{"id":"two"}],"nextCursor":null}}`,
		}, "\n")
		var out bytes.Buffer
		models, nextID, err := initAndListModels(&out, bufio.NewScanner(strings.NewReader(responses)))
		if err != nil {
			t.Fatal(err)
		}
		if nextID != 3 || len(models) != 2 || models[0].ID != "one" || models[1].ID != "two" {
			t.Fatalf("models = %#v, nextID = %d", models, nextID)
		}
		lines := strings.Split(strings.TrimSpace(out.String()), "\n")
		if !strings.Contains(lines[0], `"experimentalApi":false`) || !strings.Contains(lines[0], `"requestAttestation":false`) {
			t.Fatalf("initialize request = %s", lines[0])
		}
		var req JSONRPCRequest
		if err := json.Unmarshal([]byte(lines[3]), &req); err != nil {
			t.Fatal(err)
		}
		var params ModelListParams
		if err := json.Unmarshal(req.Params, &params); err != nil {
			t.Fatal(err)
		}
		if req.ID != 3 || params.Cursor != "next" {
			t.Fatalf("request = %#v, params = %#v", req, params)
		}
	})
}

func TestJSONRPCMessage(t *testing.T) {
	t.Run("notification", func(t *testing.T) {
		var m JSONRPCMessage
		if err := json.Unmarshal([]byte(`{"method":"thread/started","params":{},"emittedAtMs":1787231281472}`), &m); err != nil {
			t.Fatal(err)
		}
		if m.EmittedAt != base.TimeMS(1787231281472) {
			t.Errorf("EmittedAt = %v, want 1787231281472", m.EmittedAt)
		}
		if m.IsResponse() {
			t.Error("IsResponse() = true, want false")
		}
	})
	t.Run("response ID", func(t *testing.T) {
		for _, tc := range []struct {
			name string
			data string
			want string
			ok   bool
		}{
			{name: "omitted", data: `{}`, want: "", ok: false},
			{name: "null", data: `{"id":null}`, want: "null", ok: true},
			{name: "value", data: `{"id":1}`, want: "1", ok: true},
		} {
			t.Run(tc.name, func(t *testing.T) {
				var m JSONRPCMessage
				if err := json.Unmarshal([]byte(tc.data), &m); err != nil {
					t.Fatal(err)
				}
				if string(m.ID) != tc.want || m.IsResponse() != tc.ok {
					t.Errorf("message = %#v, want ID %q and IsResponse %t", m, tc.want, tc.ok)
				}
			})
		}
	})
	t.Run("server request", func(t *testing.T) {
		var m JSONRPCMessage
		if err := json.Unmarshal([]byte(`{"id":7,"method":"item/tool/requestUserInput","params":{}}`), &m); err != nil {
			t.Fatal(err)
		}
		if m.IsResponse() || !m.IsServerRequest() {
			t.Fatalf("message = %#v, want server request", m)
		}
	})
}

func TestReadResponse(t *testing.T) {
	t.Run("server_request", func(t *testing.T) {
		_, err := readResponse(bufio.NewScanner(strings.NewReader(`{"id":7,"method":"item/tool/requestUserInput","params":{}}`)), 1)
		if err == nil || !strings.Contains(err.Error(), "unsupported server request") {
			t.Fatalf("error = %v", err)
		}
	})
	t.Run("wrong_id", func(t *testing.T) {
		_, err := readResponse(bufio.NewScanner(strings.NewReader(`{"id":2,"result":{}}`)), 1)
		if err == nil || !strings.Contains(err.Error(), "unexpected JSON-RPC response id") {
			t.Fatalf("error = %v", err)
		}
	})
}

func TestParseCompletedItem(t *testing.T) {
	t.Run("agent message", func(t *testing.T) {
		const input = `{"item":{"type":"agentMessage","id":"a","text":"done","phase":null,"memoryCitation":null,"delivery":"async","questions":[{"title":"Pick","options":["A"]}]},"threadId":"t","turnId":"u","completedAtMs":1}`
		r, ok, err := parseCompletedItem([]byte(input))
		if err != nil {
			t.Fatal(err)
		}
		if !ok || r.Text != "done" {
			t.Fatalf("reply = %#v", r)
		}
	})
	t.Run("reasoning", func(t *testing.T) {
		const input = `{"item":{"type":"reasoning","id":"r","summary":["first","second"],"content":[]},"threadId":"t","turnId":"u","completedAtMs":1}`
		r, ok, err := parseCompletedItem([]byte(input))
		if err != nil {
			t.Fatal(err)
		}
		if !ok || r.Reasoning != "first\nsecond" {
			t.Fatalf("reply = %#v", r)
		}
	})
}

func TestReadTurnSync(t *testing.T) {
	t.Run("server_request", func(t *testing.T) {
		_, err := readTurnSync(bufio.NewScanner(strings.NewReader(`{"id":7,"method":"item/tool/requestUserInput","params":{}}`)), "thread")
		if err == nil || !strings.Contains(err.Error(), "unsupported server request") {
			t.Fatalf("error = %v", err)
		}
	})
	t.Run("notification_decode_error", func(t *testing.T) {
		_, err := readTurnSync(bufio.NewScanner(strings.NewReader(`{"method":"turn/completed","params":{"unknown":true}}`)), "thread")
		if err == nil || !strings.Contains(err.Error(), "decode turn/completed") {
			t.Fatalf("error = %v", err)
		}
	})
	t.Run("response_error", func(t *testing.T) {
		_, err := readTurnSync(bufio.NewScanner(strings.NewReader(`{"id":100,"error":{"code":-32602,"message":"bad turn"}}`)), "thread")
		if err == nil || !strings.Contains(err.Error(), "bad turn") {
			t.Fatalf("error = %v", err)
		}
	})
	t.Run("error_notification", func(t *testing.T) {
		line := `{"method":"error","params":{"error":{"message":"internal server error","codexErrorInfo":null,"additionalDetails":null,"misalignment":null},"willRetry":false,"threadId":"thread","turnId":"turn"}}`
		_, err := readTurnSync(bufio.NewScanner(strings.NewReader(line)), "thread")
		if err == nil || err.Error() != "codex error: internal server error" {
			t.Fatalf("error = %v", err)
		}
	})
	t.Run("failed_turn", func(t *testing.T) {
		line := `{"method":"turn/completed","params":{"threadId":"thread","turn":{"id":"turn","items":[],"itemsView":"full","status":"failed","error":{"message":"rate limit exceeded","codexErrorInfo":null,"additionalDetails":null,"misalignment":null},"startedAt":null,"completedAt":null,"durationMs":null}}}`
		_, err := readTurnSync(bufio.NewScanner(strings.NewReader(line)), "thread")
		if err == nil || err.Error() != "rate limit exceeded" {
			t.Fatalf("error = %v", err)
		}
	})
	t.Run("misalignment", func(t *testing.T) {
		line := `{"method":"turn/completed","params":{"threadId":"thread","turn":{"id":"turn","items":[],"itemsView":"full","status":"failed","error":{"message":"blocked","codexErrorInfo":null,"additionalDetails":null,"misalignment":{"errorType":"policy","detailedExplanation":"details","steer":{"message":"continue"}}},"startedAt":null,"completedAt":null,"durationMs":null}}}`
		_, err := readTurnSync(bufio.NewScanner(strings.NewReader(line)), "thread")
		if err == nil || err.Error() != "blocked" {
			t.Fatalf("error = %v", err)
		}
	})
}

func TestRecordedNotificationFields(t *testing.T) {
	t.Run("thread", func(t *testing.T) {
		var notification ThreadStartedNotification
		input := `{"thread":{"id":"thread","environments":[{"environmentId":"local","cwd":"/src","runtimeWorkspaceRoots":["/src","/cache"]}],"forkedFromId":null,"parentThreadId":"parent","section":null,"sectionEnteredAt":null,"canAcceptDirectInput":true,"model":"gpt-5.6-terra","reasoningEffort":"high","originator":"caic","daybreakEnabled":true}}`
		if err := json.Unmarshal([]byte(input), &notification); err != nil {
			t.Fatal(err)
		}
		if !notification.Thread.CanAcceptDirectInput || notification.Thread.ForkedFromID != "" || notification.Thread.ParentThreadID != "parent" {
			t.Errorf("Thread = %#v, want direct input and value optional IDs", notification.Thread)
		}
		if notification.Thread.Model != "gpt-5.6-terra" {
			t.Errorf("Model = %v, want gpt-5.6-terra", notification.Thread.Model)
		}
		if notification.Thread.ReasoningEffort != ReasoningEffortHigh {
			t.Errorf("ReasoningEffort = %v, want high", notification.Thread.ReasoningEffort)
		}
		if len(notification.Thread.Environments) != 1 || notification.Thread.Environments[0].EnvironmentID != "local" || len(notification.Thread.Environments[0].RuntimeWorkspaceRoots) != 2 {
			t.Errorf("Environments = %#v, want local environment with two workspace roots", notification.Thread.Environments)
		}
		if notification.Thread.Originator != "caic" {
			t.Errorf("Originator = %q, want caic", notification.Thread.Originator)
		}
		if !notification.Thread.DaybreakEnabled {
			t.Error("DaybreakEnabled = false, want true")
		}
	})
	t.Run("thread missing settings", func(t *testing.T) {
		var notification ThreadStartedNotification
		if err := json.Unmarshal([]byte(`{"thread":{"id":"thread","model":null,"reasoningEffort":null}}`), &notification); err != nil {
			t.Fatal(err)
		}
		if notification.Thread.Model != "" || notification.Thread.ReasoningEffort != "" {
			t.Errorf("Thread settings = %#v, want empty settings", notification.Thread)
		}
	})
	t.Run("token usage", func(t *testing.T) {
		var notification ThreadTokenUsageUpdatedNotification
		input := `{"threadId":"thread","turnId":"turn","tokenUsage":{"total":{"cacheWriteInputTokens":1},"last":{"cacheWriteInputTokens":2}}}`
		if err := json.Unmarshal([]byte(input), &notification); err != nil {
			t.Fatal(err)
		}
		if notification.TokenUsage.Total.CacheWriteInputTokens != 1 || notification.TokenUsage.Last.CacheWriteInputTokens != 2 {
			t.Errorf("TokenUsage = %#v, want cache-write token counts", notification.TokenUsage)
		}
	})
	t.Run("MCP startup", func(t *testing.T) {
		var notification McpServerStatusUpdatedNotification
		input := `{"threadId":"thread","name":"node","status":"starting","error":null,"failureReason":null}`
		if err := json.Unmarshal([]byte(input), &notification); err != nil {
			t.Fatal(err)
		}
		if notification.ThreadID != "thread" || notification.Error != "" || notification.FailureReason != "" {
			t.Errorf("notification = %#v, want empty optional errors", notification)
		}
	})
	t.Run("rate limit", func(t *testing.T) {
		for _, tc := range []struct {
			name string
			data string
			want bool
		}{
			{name: "omitted", data: `{}`, want: false},
			{name: "null", data: `{"spendControlReached":null}`, want: false},
			{name: "value", data: `{"spendControlReached":true,"normalModelSlug":"gpt-5.6-sol"}`, want: true},
		} {
			t.Run(tc.name, func(t *testing.T) {
				var snapshot RateLimitSnapshot
				if err := json.Unmarshal([]byte(tc.data), &snapshot); err != nil {
					t.Fatal(err)
				}
				if snapshot.SpendControlReached != tc.want {
					t.Errorf("SpendControlReached = %t, want %t", snapshot.SpendControlReached, tc.want)
				}
				if tc.name == "value" && snapshot.NormalModelSlug != "gpt-5.6-sol" {
					t.Errorf("NormalModelSlug = %q, want gpt-5.6-sol", snapshot.NormalModelSlug)
				}
			})
		}
	})
	t.Run("MCP startup value errors", func(t *testing.T) {
		var notification McpServerStatusUpdatedNotification
		input := `{"threadId":"thread","name":"node","status":"failed","error":"failed","failureReason":"missing binary"}`
		if err := json.Unmarshal([]byte(input), &notification); err != nil {
			t.Fatal(err)
		}
		if notification.Error != "failed" || notification.FailureReason != "missing binary" {
			t.Errorf("notification = %#v, want value optional errors", notification)
		}
	})
	t.Run("skills changed", func(t *testing.T) {
		var notification SkillsChangedNotification
		if err := json.Unmarshal([]byte(`{}`), &notification); err != nil {
			t.Fatal(err)
		}
	})
}

func TestNotificationTimeMS(t *testing.T) {
	t.Run("item_started", func(t *testing.T) {
		const input = `{"item":{"id":"u1","type":"userMessage"},"threadId":"t1","turnId":"turn_1","startedAtMs":1780832660165}`
		var got ItemStartedNotification
		if err := json.Unmarshal([]byte(input), &got); err != nil {
			t.Fatal(err)
		}
		if got.StartedAt != base.TimeMS(1780832660165) {
			t.Errorf("StartedAt = %v, want 1780832660165", got.StartedAt)
		}
		if got.StartedAt.AsTime() != time.Date(2026, 6, 7, 11, 44, 20, 165000000, time.UTC) {
			t.Errorf("StartedAt.AsTime() = %v", got.StartedAt.AsTime())
		}
	})
	t.Run("guardian_review_completed", func(t *testing.T) {
		const input = `{"threadId":"t1","turnId":"turn_1","startedAtMs":1780832660165,"completedAtMs":1780832661123,"reviewId":"r1","targetItemId":null,"decisionSource":"agent_decision","review":{"status":"approved"},"action":{"type":"run_command"}}`
		var got ItemGuardianApprovalReviewCompletedNotification
		if err := json.Unmarshal([]byte(input), &got); err != nil {
			t.Fatal(err)
		}
		if got.StartedAt != base.TimeMS(1780832660165) {
			t.Errorf("StartedAt = %v, want 1780832660165", got.StartedAt)
		}
		if got.CompletedAt != base.TimeMS(1780832661123) {
			t.Errorf("CompletedAt = %v, want 1780832661123", got.CompletedAt)
		}
	})
}

func TestDurationMS(t *testing.T) {
	t.Run("turn", func(t *testing.T) {
		const input = `{"id":"turn_1","status":"completed","startedAt":1780832660.165,"completedAt":1780832661.25,"durationMs":123.5}`
		var got Turn
		if err := json.Unmarshal([]byte(input), &got); err != nil {
			t.Fatal(err)
		}
		if got.StartedAt != base.TimeS(1780832660.165) {
			t.Errorf("StartedAt = %v, want 1780832660.165", got.StartedAt)
		}
		if got.CompletedAt != base.TimeS(1780832661.25) {
			t.Errorf("CompletedAt = %v, want 1780832661.25", got.CompletedAt)
		}
		if got.Duration == nil {
			t.Fatal("Duration = nil")
		}
		if *got.Duration != base.DurationMS(123.5) {
			t.Errorf("Duration = %v, want 123.5", *got.Duration)
		}
		if got.Duration.AsDuration() != 123*time.Millisecond+500*time.Microsecond {
			t.Errorf("Duration.AsDuration() = %v", got.Duration.AsDuration())
		}
	})
	t.Run("command_execution", func(t *testing.T) {
		const input = `{"id":"cmd_1","type":"commandExecution","durationMs":12.25}`
		var got CommandExecutionItem
		if err := json.Unmarshal([]byte(input), &got); err != nil {
			t.Fatal(err)
		}
		if got.Duration == nil {
			t.Fatal("Duration = nil")
		}
		if *got.Duration != base.DurationMS(12.25) {
			t.Errorf("Duration = %v, want 12.25", *got.Duration)
		}
		if got.Duration.AsDuration() != 12*time.Millisecond+250*time.Microsecond {
			t.Errorf("Duration.AsDuration() = %v", got.Duration.AsDuration())
		}
	})
	t.Run("dynamic_tool_call", func(t *testing.T) {
		const input = `{"id":"dyn_1","type":"dynamicToolCall","durationMs":7.75}`
		var got DynamicToolCallItem
		if err := json.Unmarshal([]byte(input), &got); err != nil {
			t.Fatal(err)
		}
		if got.Duration != base.DurationMS(7.75) {
			t.Errorf("Duration = %v, want 7.75", got.Duration)
		}
		if got.Duration.AsDuration() != 7*time.Millisecond+750*time.Microsecond {
			t.Errorf("Duration.AsDuration() = %v", got.Duration.AsDuration())
		}
	})
}

func TestCommandExecutionItem(t *testing.T) {
	t.Run("nullable_plugin_fields", func(t *testing.T) {
		const input = `{"id":"cmd_1","type":"commandExecution","pluginId":null,"scriptPath":null}`
		var got CommandExecutionItem
		if err := internal.UnmarshalJSON([]byte(input), &got); err != nil {
			t.Fatal(err)
		}
		if got.PluginID != "" {
			t.Errorf("PluginID = %q, want empty", got.PluginID)
		}
		if got.ScriptPath != "" {
			t.Errorf("ScriptPath = %q, want empty", got.ScriptPath)
		}
	})
	t.Run("plugin_fields", func(t *testing.T) {
		const input = `{"id":"cmd_1","type":"commandExecution","pluginId":"canva@openai-curated-remote","scriptPath":"scripts/create-design.sh"}`
		var got CommandExecutionItem
		if err := internal.UnmarshalJSON([]byte(input), &got); err != nil {
			t.Fatal(err)
		}
		if got.PluginID != "canva@openai-curated-remote" {
			t.Errorf("PluginID = %v, want canva@openai-curated-remote", got.PluginID)
		}
		if got.ScriptPath != "scripts/create-design.sh" {
			t.Errorf("ScriptPath = %v, want scripts/create-design.sh", got.ScriptPath)
		}
	})
}

func TestThreadItemExtensions(t *testing.T) {
	t.Run("function_call_output", func(t *testing.T) {
		const input = `{"id":"call_1","type":"functionCallOutput","name":"read","namespace":null,"output":[{"type":"inputText","text":"done"}]}`
		var got FunctionCallOutputItem
		if err := internal.UnmarshalJSON([]byte(input), &got); err != nil {
			t.Fatal(err)
		}
		if got.Type != ItemTypeFunctionCallOutput || got.Name != "read" || len(got.Output) == 0 {
			t.Fatalf("FunctionCallOutputItem = %+v", got)
		}
	})
	t.Run("mcp_tool_call_app_context", func(t *testing.T) {
		const input = `{"id":"mcp_1","type":"mcpToolCall","appContext":{"connectorId":"canva","linkId":"link_1","resourceUri":"canva://design/1","appName":"Canva","actionName":"Create design"},"readOnlyHint":true,"durationMs":12.25}`
		var got McpToolCallItem
		if err := internal.UnmarshalJSON([]byte(input), &got); err != nil {
			t.Fatal(err)
		}
		if got.AppContext.ConnectorID != "canva" || got.AppContext.ActionName != "Create design" || !got.ReadOnlyHint || got.Duration != base.DurationMS(12.25) {
			t.Errorf("McpToolCallItem = %+v, want populated app context and read-only hint", got)
		}
	})
	t.Run("sub_agent_activity", func(t *testing.T) {
		const input = `{"id":"activity_1","type":"subAgentActivity","kind":"interacted","agentThreadId":"thread_1","agentPath":"/agents/research"}`
		var got SubAgentActivityItem
		if err := internal.UnmarshalJSON([]byte(input), &got); err != nil {
			t.Fatal(err)
		}
		if got.Kind != SubAgentActivityKindInteracted || got.AgentThreadID != "thread_1" || got.AgentPath != "/agents/research" {
			t.Errorf("SubAgentActivityItem = %+v, want populated activity", got)
		}
	})
	t.Run("sleep", func(t *testing.T) {
		const input = `{"id":"sleep_1","type":"sleep","durationMs":250}`
		var got SleepItem
		if err := internal.UnmarshalJSON([]byte(input), &got); err != nil {
			t.Fatal(err)
		}
		if got.Duration != base.DurationMS(250) {
			t.Errorf("Duration = %v, want 250", got.Duration)
		}
	})
	t.Run("web_search_results", func(t *testing.T) {
		const input = `{"id":"search_1","type":"webSearch","results":[{"title":"Codex"}]}`
		var got WebSearchItem
		if err := internal.UnmarshalJSON([]byte(input), &got); err != nil {
			t.Fatal(err)
		}
		if len(got.Results) != 1 || string(got.Results[0]) != `{"title":"Codex"}` {
			t.Errorf("Results = %s, want one result", got.Results)
		}
	})
}

func TestDurationS(t *testing.T) {
	const input = `{"threadId":"t1","objective":"ship","status":"active","tokensUsed":12,"timeUsedSeconds":3.25,"createdAt":1,"updatedAt":2}`
	var got ThreadGoal
	if err := json.Unmarshal([]byte(input), &got); err != nil {
		t.Fatal(err)
	}
	if got.TimeUsed != base.DurationS(3.25) {
		t.Errorf("TimeUsed = %v, want 3.25", got.TimeUsed)
	}
	if got.TimeUsed.AsDuration() != 3*time.Second+250*time.Millisecond {
		t.Errorf("TimeUsed.AsDuration() = %v", got.TimeUsed.AsDuration())
	}
	if got.CreatedAt != base.TimeS(1) {
		t.Errorf("CreatedAt = %v, want 1", got.CreatedAt)
	}
	if got.UpdatedAt != base.TimeS(2) {
		t.Errorf("UpdatedAt = %v, want 2", got.UpdatedAt)
	}
}

func TestTimeS(t *testing.T) {
	t.Run("thread", func(t *testing.T) {
		const input = `{"id":"t1","createdAt":1780832660.165,"updatedAt":1780832661.25}`
		var got Thread
		if err := json.Unmarshal([]byte(input), &got); err != nil {
			t.Fatal(err)
		}
		if got.CreatedAt != base.TimeS(1780832660.165) {
			t.Errorf("CreatedAt = %v, want 1780832660.165", got.CreatedAt)
		}
		if got.UpdatedAt != base.TimeS(1780832661.25) {
			t.Errorf("UpdatedAt = %v, want 1780832661.25", got.UpdatedAt)
		}
		if got.CreatedAt.AsTime() != time.Date(2026, 6, 7, 11, 44, 20, 165000000, time.UTC) {
			t.Errorf("CreatedAt.AsTime() = %v", got.CreatedAt.AsTime())
		}
	})
	t.Run("rate_limit_window", func(t *testing.T) {
		const input = `{"usedPercent":50,"resetsAt":1780832660.165}`
		var got RateLimitWindow
		if err := json.Unmarshal([]byte(input), &got); err != nil {
			t.Fatal(err)
		}
		if got.ResetsAt != base.TimeS(1780832660.165) {
			t.Errorf("ResetsAt = %v, want 1780832660.165", got.ResetsAt)
		}
	})
	t.Run("spend_control_limit_snapshot", func(t *testing.T) {
		got := SpendControlLimitSnapshot{
			Limit:            "100",
			Used:             "50",
			RemainingPercent: 50,
			ResetsAt:         base.TimeS(1780832660.165),
		}
		b, err := json.Marshal(got)
		if err != nil {
			t.Fatal(err)
		}
		var fields map[string]json.RawMessage
		if err := json.Unmarshal(b, &fields); err != nil {
			t.Fatal(err)
		}
		if _, ok := fields["resetsAt"]; !ok {
			t.Errorf("marshaled fields = %s, want resetsAt", b)
		}
		if _, ok := fields["ResetsAt"]; ok {
			t.Errorf("marshaled fields = %s, did not want ResetsAt", b)
		}
	})
}

func TestContextCompactionThreadItem(t *testing.T) {
	t.Run("valid", func(t *testing.T) {
		const input = `{"id":"cc1","type":"contextCompaction"}`
		var item ContextCompactionThreadItem
		if err := json.Unmarshal([]byte(input), &item); err != nil {
			t.Fatal(err)
		}
		if item.ID != "cc1" {
			t.Errorf("ID = %q, want cc1", item.ID)
		}
		if item.Type != ItemTypeContextCompaction {
			t.Errorf("Type = %q, want %q", item.Type, ItemTypeContextCompaction)
		}
	})
}

func TestUserMessageItem(t *testing.T) {
	t.Run("valid", func(t *testing.T) {
		const input = `{"id":"u1","type":"userMessage","clientId":null,"content":[{"type":"text","text":"hello","text_elements":[]}]}`
		var item UserMessageItem
		if err := json.Unmarshal([]byte(input), &item); err != nil {
			t.Fatal(err)
		}
		if item.ID != "u1" {
			t.Errorf("ID = %q, want u1", item.ID)
		}
		if item.Type != ItemTypeUserMessage {
			t.Errorf("Type = %q, want %q", item.Type, ItemTypeUserMessage)
		}
		if len(item.Content) != 1 {
			t.Fatalf("len(Content) = %d, want 1", len(item.Content))
		}
		if item.Content[0].Type != TurnInputTypeText {
			t.Errorf("Content[0].Type = %q, want %q", item.Content[0].Type, TurnInputTypeText)
		}
		if item.Content[0].Text != "hello" {
			t.Errorf("Content[0].Text = %q, want hello", item.Content[0].Text)
		}
		if len(item.Content[0].TextElements) != 0 {
			t.Errorf("len(Content[0].TextElements) = %d, want 0", len(item.Content[0].TextElements))
		}
	})
}

func init() {
	internal.BeLenient = false
}
