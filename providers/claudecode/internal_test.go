// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Internal tests for the Claude Code provider.

package claudecode

import (
	"encoding/json"
	"errors"
	"slices"
	"strings"
	"testing"

	"github.com/maruel/genai"
	"github.com/maruel/genai/base"
	"github.com/maruel/genai/internal"
)

func TestBuildArgs(t *testing.T) {
	t.Run("defaults", func(t *testing.T) {
		c := &Client{model: ""}
		args := c.buildArgs(&callOpts{}, "", false)
		want := []string{
			"-p",
			"--verbose",
			"--input-format", "stream-json",
			"--output-format", "stream-json",
			"--strict-mcp-config",
			"--no-chrome",
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
		args := c.buildArgs(&co, "", false)
		check := func(flag, val string) {
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
		args := c.buildArgs(&callOpts{}, "my-session-id", false)
		for _, a := range args {
			if a == "--no-session-persistence" {
				t.Error("--no-session-persistence must not appear when resuming")
			}
		}
		check := func(flag, val string) {
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
		args := c.buildArgs(&callOpts{}, "", true)
		if !slices.Contains(args, "--include-partial-messages") {
			t.Error("--include-partial-messages not found in streaming args")
		}
	})
	t.Run("with_system_prompt", func(t *testing.T) {
		c := &Client{}
		co := callOpts{systemPrompt: "Be helpful"}
		args := c.buildArgs(&co, "", false)
		check := func(flag, val string) {
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

func TestBuildResult(t *testing.T) {
	t.Run("thinking_tokens", func(t *testing.T) {
		res := &OutputResultMsg{
			StopReason: "end_turn",
			Usage: MsgUsage{
				InputTokens:  10,
				OutputTokens: 20,
				OutputTokensDetails: OutputTokensDetails{
					ThinkingTokens: 7,
				},
			},
		}
		got := buildResult(res, []OutputContentBlock{{Type: "text", Text: "ok"}}, nil, "session")
		if got.Usage.ReasoningTokens != 7 {
			t.Errorf("ReasoningTokens = %d, want 7", got.Usage.ReasoningTokens)
		}
	})
	t.Run("post_turn_summary", func(t *testing.T) {
		res := &OutputResultMsg{
			StopReason: "end_turn",
			Usage: MsgUsage{
				InputTokens:  10,
				OutputTokens: 20,
			},
		}
		got := buildResult(res, []OutputContentBlock{{Type: "text", Text: "ok"}}, []string{"Checked the request."}, "session")
		if got.Replies[0].Reasoning != "Checked the request." {
			t.Errorf("Reasoning = %q, want summary", got.Replies[0].Reasoning)
		}
	})
}

func TestStreamDelta(t *testing.T) {
	t.Run("estimated_tokens", func(t *testing.T) {
		const data = `{"type":"stream_event","event":{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","estimated_tokens":42,"estimated_tokens_delta":3}}}`
		var got OutputStreamEventMsg
		if err := json.Unmarshal([]byte(data), &got); err != nil {
			t.Fatal(err)
		}
		if got.Event.Delta.EstimatedTokens != 42 {
			t.Errorf("EstimatedTokens = %d, want 42", got.Event.Delta.EstimatedTokens)
		}
		if got.Event.Delta.EstimatedTokensDelta != 3 {
			t.Errorf("EstimatedTokensDelta = %d, want 3", got.Event.Delta.EstimatedTokensDelta)
		}
	})
}

func TestOutputMessages(t *testing.T) {
	t.Run("system_thinking_tokens", func(t *testing.T) {
		const data = `{"type":"system","subtype":"thinking_tokens","estimated_tokens":138,"estimated_tokens_delta":88,"uuid":"u1","session_id":"s1"}`
		var got OutputSystemMsg
		if err := internal.UnmarshalJSON([]byte(data), &got); err != nil {
			t.Fatal(err)
		}
		if got.Subtype != SystemThinkingTokens {
			t.Errorf("Subtype = %q, want %q", got.Subtype, SystemThinkingTokens)
		}
		if got.EstimatedTokens != 138 {
			t.Errorf("EstimatedTokens = %d, want 138", got.EstimatedTokens)
		}
		if got.EstimatedTokensDelta != 88 {
			t.Errorf("EstimatedTokensDelta = %d, want 88", got.EstimatedTokensDelta)
		}
	})
	t.Run("task_updated", func(t *testing.T) {
		const data = `{"type":"system","subtype":"task_updated","task_id":"task-1","patch":{"status":"completed","end_time":1780832660165},"uuid":"u1","session_id":"s1"}`
		var got OutputSystemMsg
		if err := internal.UnmarshalJSON([]byte(data), &got); err != nil {
			t.Fatal(err)
		}
		if got.Subtype != SystemTaskUpdated {
			t.Errorf("Subtype = %q, want %q", got.Subtype, SystemTaskUpdated)
		}
		if got.Patch.Status != "completed" {
			t.Errorf("Patch.Status = %q, want completed", got.Patch.Status)
		}
	})
	t.Run("task_started_subagent_metadata", func(t *testing.T) {
		const data = `{"type":"system","subtype":"task_started","task_id":"task-1","tool_use_id":"toolu_1","description":"Find harness/model selection logic","subagent_type":"Explore","task_type":"local_agent","prompt":"Find harness/model selection logic","uuid":"u1","session_id":"s1"}`
		var got OutputSystemMsg
		if err := internal.UnmarshalJSON([]byte(data), &got); err != nil {
			t.Fatal(err)
		}
		if got.SubagentType != "Explore" {
			t.Errorf("SubagentType = %q, want Explore", got.SubagentType)
		}
	})
	t.Run("init_flags", func(t *testing.T) {
		const data = `{"type":"system","subtype":"init","cwd":"/tmp","session_id":"s1","tools":[],"model":"m","claude_code_version":"1.0","uuid":"u1","analytics_disabled":true,"product_feedback_disabled":true}`
		var got OutputInitMsg
		if err := internal.UnmarshalJSON([]byte(data), &got); err != nil {
			t.Fatal(err)
		}
		if !got.AnalyticsDisabled {
			t.Error("AnalyticsDisabled = false, want true")
		}
		if !got.ProductFeedbackDisabled {
			t.Error("ProductFeedbackDisabled = false, want true")
		}
	})
	t.Run("stream_context_management", func(t *testing.T) {
		const data = `{"type":"stream_event","event":{"type":"message_delta","context_management":{"applied_edits":[{"type":"clear_tool_uses_20250919","cleared_input_tokens":123,"cleared_tool_uses":4},{"type":"clear_thinking_20251015","cleared_input_tokens":456,"cleared_thinking_turns":7}]}},"uuid":"u1","session_id":"s1","parent_tool_use_id":null}`
		var got OutputStreamEventMsg
		if err := internal.UnmarshalJSON([]byte(data), &got); err != nil {
			t.Fatal(err)
		}
		if len(got.Event.ContextManagement.AppliedEdits) != 2 {
			t.Fatalf("len(AppliedEdits) = %d, want 2", len(got.Event.ContextManagement.AppliedEdits))
		}
		edit := got.Event.ContextManagement.AppliedEdits[0]
		if edit.Type != AppliedEditClearToolUses {
			t.Errorf("AppliedEdits[0].Type = %q, want %q", edit.Type, AppliedEditClearToolUses)
		}
		if edit.ClearedInputTokens != 123 {
			t.Errorf("AppliedEdits[0].ClearedInputTokens = %d, want 123", edit.ClearedInputTokens)
		}
		if edit.ClearedToolUses != 4 {
			t.Errorf("AppliedEdits[0].ClearedToolUses = %d, want 4", edit.ClearedToolUses)
		}
		edit = got.Event.ContextManagement.AppliedEdits[1]
		if edit.Type != AppliedEditClearThinking {
			t.Errorf("AppliedEdits[1].Type = %q, want %q", edit.Type, AppliedEditClearThinking)
		}
		if edit.ClearedThinkingTurns != 7 {
			t.Errorf("AppliedEdits[1].ClearedThinkingTurns = %d, want 7", edit.ClearedThinkingTurns)
		}
	})
	t.Run("stream_message_start", func(t *testing.T) {
		const data = `{"type":"stream_event","event":{"type":"message_start","message":{"model":"claude-opus-4-8","id":"msg_01","type":"message","role":"assistant","content":[],"stop_reason":null,"stop_sequence":null,"stop_details":null,"usage":{"input_tokens":268,"cache_creation_input_tokens":398,"cache_read_input_tokens":408224,"cache_creation":{"ephemeral_5m_input_tokens":0,"ephemeral_1h_input_tokens":398},"output_tokens":3,"service_tier":"standard","inference_geo":"not_available"},"diagnostics":null}},"uuid":"u1","session_id":"s1","parent_tool_use_id":null}`
		var got OutputStreamEventMsg
		if err := internal.UnmarshalJSON([]byte(data), &got); err != nil {
			t.Fatal(err)
		}
		if got.Event.Message.ID != "msg_01" {
			t.Errorf("Message.ID = %q, want msg_01", got.Event.Message.ID)
		}
		if got.Event.Message.Usage.InputTokens != 268 {
			t.Errorf("Message.Usage.InputTokens = %d, want 268", got.Event.Message.Usage.InputTokens)
		}
	})
	t.Run("assistant_message_metadata", func(t *testing.T) {
		const data = `{"type":"assistant","message":{"model":"claude-opus-4-8","id":"msg_01","type":"message","role":"assistant","content":[{"type":"tool_use","id":"toolu_1","name":"Bash","input":{"command":"true"},"caller":{"type":"direct"}}],"stop_reason":"refusal","stop_sequence":null,"stop_details":{"type":"refusal","category":"cyber","explanation":"blocked"},"usage":{"input_tokens":1,"output_tokens":2},"container":{"id":"container_1","expires_at":"2026-06-07T12:00:00Z","skills":[{"skill_id":"sk_1","type":"anthropic","version":"latest"}]},"diagnostics":{"cache_miss_reason":{"type":"tools_changed","cache_missed_input_tokens":42}}},"uuid":"u1","session_id":"s1","parent_tool_use_id":null,"subagent_type":"Explore","task_description":"Find harness/model selection logic"}`
		var got OutputAssistantMsg
		if err := internal.UnmarshalJSON([]byte(data), &got); err != nil {
			t.Fatal(err)
		}
		if got.SubagentType != "Explore" {
			t.Errorf("SubagentType = %q, want Explore", got.SubagentType)
		}
		if got.TaskDescription != "Find harness/model selection logic" {
			t.Errorf("TaskDescription = %q, want Find harness/model selection logic", got.TaskDescription)
		}
		if got.Message.Container.ID != "container_1" {
			t.Errorf("Container.ID = %q, want container_1", got.Message.Container.ID)
		}
		if len(got.Message.Container.Skills) != 1 || got.Message.Container.Skills[0].Type != SkillAnthropic {
			t.Fatalf("Container.Skills = %+v, want one anthropic skill", got.Message.Container.Skills)
		}
		if got.Message.StopDetails.Type != "refusal" {
			t.Errorf("StopDetails.Type = %q, want refusal", got.Message.StopDetails.Type)
		}
		if got.Message.Diagnostics.CacheMissReason.Type != CacheMissReasonToolsChanged {
			t.Errorf("CacheMissReason.Type = %q, want %q", got.Message.Diagnostics.CacheMissReason.Type, CacheMissReasonToolsChanged)
		}
		if got.Message.Diagnostics.CacheMissReason.CacheMissedInputTokens != 42 {
			t.Errorf("CacheMissedInputTokens = %d, want 42", got.Message.Diagnostics.CacheMissReason.CacheMissedInputTokens)
		}
		if got.Message.Content[0].Caller.Type != "direct" {
			t.Errorf("Caller.Type = %q, want direct", got.Message.Content[0].Caller.Type)
		}
		inputJSON, err := json.Marshal(got.Message.Content[0].Input)
		if err != nil {
			t.Fatal(err)
		}
		var input BashInput
		if err := json.Unmarshal(inputJSON, &input); err != nil {
			t.Fatal(err)
		}
		if input.Command != "true" {
			t.Errorf("BashInput.Command = %q, want true", input.Command)
		}
	})
	t.Run("user_subagent_metadata", func(t *testing.T) {
		const data = `{"type":"user","message":{"role":"user","content":[{"type":"text","text":"Find harness/model selection logic"}]},"parent_tool_use_id":"toolu_1","session_id":"s1","uuid":"u1","timestamp":"2026-06-13T20:16:11.423Z","subagent_type":"Explore","task_description":"Find harness/model selection logic"}`
		var got OutputUserMsg
		if err := internal.UnmarshalJSON([]byte(data), &got); err != nil {
			t.Fatal(err)
		}
		if got.SubagentType != "Explore" {
			t.Errorf("SubagentType = %q, want Explore", got.SubagentType)
		}
		if got.TaskDescription != "Find harness/model selection logic" {
			t.Errorf("TaskDescription = %q, want Find harness/model selection logic", got.TaskDescription)
		}
	})
	t.Run("permission_suggestions", func(t *testing.T) {
		const data = `{"type":"control_request","request_id":"r1","request":{"subtype":"can_use_tool","tool_name":"Bash","input":{"command":"git status"},"permission_suggestions":[{"type":"addRules","destination":"localSettings","behavior":"allow","rules":[{"toolName":"Bash","ruleContent":"git status"},{"toolName":"Read","ruleContent":null}]},{"type":"setMode","mode":"acceptEdits","destination":"session"},{"type":"addDirectories","directories":["/tmp/a","/tmp/b"],"destination":"userSettings"}],"tool_use_id":"toolu_1"}}`
		var raw struct {
			Type      InputType            `json:"type"`
			RequestID string               `json:"request_id"`
			Request   ControlReqCanUseTool `json:"request"`
		}
		if err := internal.UnmarshalJSON([]byte(data), &raw); err != nil {
			t.Fatal(err)
		}
		if raw.Request.Subtype != ControlCanUseTool {
			t.Errorf("Subtype = %q, want %q", raw.Request.Subtype, ControlCanUseTool)
		}
		if len(raw.Request.PermissionSuggestions) != 3 {
			t.Fatalf("len(PermissionSuggestions) = %d, want 3", len(raw.Request.PermissionSuggestions))
		}
		add := raw.Request.PermissionSuggestions[0]
		if add.Type != PermissionUpdateAddRules {
			t.Errorf("PermissionSuggestions[0].Type = %q, want %q", add.Type, PermissionUpdateAddRules)
		}
		if add.Destination != PermissionUpdateLocalSettings {
			t.Errorf("PermissionSuggestions[0].Destination = %q, want %q", add.Destination, PermissionUpdateLocalSettings)
		}
		if add.Behavior != PermissionBehaviorAllow {
			t.Errorf("PermissionSuggestions[0].Behavior = %q, want %q", add.Behavior, PermissionBehaviorAllow)
		}
		if len(add.Rules) != 2 {
			t.Fatalf("len(PermissionSuggestions[0].Rules) = %d, want 2", len(add.Rules))
		}
		if add.Rules[0].ToolName != "Bash" || add.Rules[0].RuleContent == nil || *add.Rules[0].RuleContent != "git status" {
			t.Errorf("Rules[0] = %+v, want Bash git status", add.Rules[0])
		}
		if add.Rules[1].ToolName != "Read" || add.Rules[1].RuleContent != nil {
			t.Errorf("Rules[1] = %+v, want Read nil content", add.Rules[1])
		}
		mode := raw.Request.PermissionSuggestions[1]
		if mode.Type != PermissionUpdateSetMode || mode.Mode != "acceptEdits" || mode.Destination != PermissionUpdateSession {
			t.Errorf("PermissionSuggestions[1] = %+v, want setMode acceptEdits session", mode)
		}
		dirs := raw.Request.PermissionSuggestions[2]
		if dirs.Type != PermissionUpdateAddDirectories || dirs.Destination != PermissionUpdateUserSettings {
			t.Errorf("PermissionSuggestions[2] = %+v, want addDirectories userSettings", dirs)
		}
		if len(dirs.Directories) != 2 || dirs.Directories[0] != "/tmp/a" || dirs.Directories[1] != "/tmp/b" {
			t.Errorf("PermissionSuggestions[2].Directories = %v, want [/tmp/a /tmp/b]", dirs.Directories)
		}
	})
	t.Run("stream_content_block_start", func(t *testing.T) {
		const data = `{"type":"stream_event","event":{"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_1","name":"Bash","input":{},"caller":{"type":"code_execution_20260120","tool_id":"srv_1"}}},"uuid":"u1","session_id":"s1","parent_tool_use_id":null}`
		var got OutputStreamEventMsg
		if err := internal.UnmarshalJSON([]byte(data), &got); err != nil {
			t.Fatal(err)
		}
		if got.Event.ContentBlock.ID != "toolu_1" {
			t.Errorf("ContentBlock.ID = %q, want toolu_1", got.Event.ContentBlock.ID)
		}
		if got.Event.ContentBlock.Caller.Type != "code_execution_20260120" {
			t.Errorf("ContentBlock.Caller.Type = %q, want code_execution_20260120", got.Event.ContentBlock.Caller.Type)
		}
		if got.Event.ContentBlock.Caller.ToolID != "srv_1" {
			t.Errorf("ContentBlock.Caller.ToolID = %q, want srv_1", got.Event.ContentBlock.Caller.ToolID)
		}
	})
	t.Run("stream_message_delta_usage", func(t *testing.T) {
		const data = `{"type":"stream_event","event":{"type":"message_delta","usage":{"input_tokens":2,"output_tokens":192,"cache_read_input_tokens":409477,"output_tokens_details":{"thinking_tokens":49},"iterations":[{"input_tokens":2,"output_tokens":192,"cache_read_input_tokens":409477,"cache_creation_input_tokens":0,"type":"message"},{"input_tokens":3,"output_tokens":4,"cache_read_input_tokens":5,"cache_creation_input_tokens":6,"model":"claude-opus-4-8","type":"advisor_message"}]}},"uuid":"u1","session_id":"s1","parent_tool_use_id":null}`
		var got OutputStreamEventMsg
		if err := internal.UnmarshalJSON([]byte(data), &got); err != nil {
			t.Fatal(err)
		}
		if got.Event.Usage.OutputTokens != 192 {
			t.Errorf("Usage.OutputTokens = %d, want 192", got.Event.Usage.OutputTokens)
		}
		if got.Event.Usage.OutputTokensDetails.ThinkingTokens != 49 {
			t.Errorf("Usage.OutputTokensDetails.ThinkingTokens = %d, want 49", got.Event.Usage.OutputTokensDetails.ThinkingTokens)
		}
		if len(got.Event.Usage.Iterations) != 2 {
			t.Fatalf("len(Usage.Iterations) = %d, want 2", len(got.Event.Usage.Iterations))
		}
		if got.Event.Usage.Iterations[0].Type != IterationUsageMessage {
			t.Errorf("Usage.Iterations[0].Type = %q, want %q", got.Event.Usage.Iterations[0].Type, IterationUsageMessage)
		}
		if got.Event.Usage.Iterations[1].Type != IterationUsageAdvisorMessage {
			t.Errorf("Usage.Iterations[1].Type = %q, want %q", got.Event.Usage.Iterations[1].Type, IterationUsageAdvisorMessage)
		}
	})
	t.Run("result_latency_fields", func(t *testing.T) {
		const data = `{"type":"result","subtype":"success","is_error":false,"duration_ms":1,"duration_api_ms":2,"ttft_ms":3,"ttft_stream_ms":4,"time_to_request_ms":5,"num_turns":1,"result":"ok","session_id":"s1","total_cost_usd":0,"usage":{},"uuid":"u1"}`
		var got OutputResultMsg
		if err := internal.UnmarshalJSON([]byte(data), &got); err != nil {
			t.Fatal(err)
		}
		if got.TtftStreamMs != 4 {
			t.Errorf("TtftStreamMs = %d, want 4", got.TtftStreamMs)
		}
		if got.TimeToRequestMs != 5 {
			t.Errorf("TimeToRequestMs = %d, want 5", got.TimeToRequestMs)
		}
	})
	t.Run("tool_result_string_content", func(t *testing.T) {
		const data = `{"content":"tool failed","is_error":true}`
		var got OutputToolResult
		if err := internal.UnmarshalJSON([]byte(data), &got); err != nil {
			t.Fatal(err)
		}
		if got.Content.Text != "tool failed" {
			t.Errorf("Content.Text = %q, want tool failed", got.Content.Text)
		}
		blocks := got.Content.TextBlocks()
		if len(blocks) != 1 || blocks[0].Text != "tool failed" {
			t.Fatalf("TextBlocks = %+v, want single text block", blocks)
		}
	})
	t.Run("tool_result_block_content", func(t *testing.T) {
		const data = `{"content":[{"type":"text","text":"ok"}],"is_error":false}`
		var got OutputToolResult
		if err := internal.UnmarshalJSON([]byte(data), &got); err != nil {
			t.Fatal(err)
		}
		if len(got.Content.Blocks) != 1 || got.Content.Blocks[0].Text != "ok" {
			t.Fatalf("Content.Blocks = %+v, want single ok block", got.Content.Blocks)
		}
	})
	t.Run("control_wrappers", func(t *testing.T) {
		const data = `{"subtype":"initialize","hooks":{"PreToolUse":[{"matcher":"Bash","hookCallbackIds":["hook_1"],"timeout":5}]},"jsonSchema":{"type":"object","properties":{"answer":{"type":"string"}}},"agentProgressSummaries":true}`
		var got ControlReqInitialize
		if err := internal.UnmarshalJSON([]byte(data), &got); err != nil {
			t.Fatal(err)
		}
		if got.Hooks[HookPreToolUse][0].HookCallbackIDs[0] != "hook_1" {
			t.Errorf("hook callback id = %q, want hook_1", got.Hooks[HookPreToolUse][0].HookCallbackIDs[0])
		}
		if string(got.JSONSchema["type"]) != `"object"` {
			t.Errorf("json schema type = %v, want object", got.JSONSchema["type"])
		}
	})
	t.Run("hook_callback_input", func(t *testing.T) {
		const data = `{"subtype":"hook_callback","callback_id":"hook_1","input":{"hook_event_name":"PreToolUse","session_id":"s1","transcript_path":"/tmp/t.jsonl","cwd":"/repo","tool_name":"Bash","tool_input":{"command":"true"},"tool_use_id":"toolu_1"},"tool_use_id":"toolu_1"}`
		var got ControlReqHookCallback
		if err := internal.UnmarshalJSON([]byte(data), &got); err != nil {
			t.Fatal(err)
		}
		inputJSON, err := json.Marshal(got.Input.ToolInput)
		if err != nil {
			t.Fatal(err)
		}
		var input BashInput
		if err := json.Unmarshal(inputJSON, &input); err != nil {
			t.Fatal(err)
		}
		if got.Input.HookEventName != string(HookPreToolUse) || input.Command != "true" {
			t.Fatalf("Input = %+v, BashInput = %+v, want PreToolUse true", got.Input, input)
		}
	})
	t.Run("mcp_jsonrpc_message", func(t *testing.T) {
		const data = `{"subtype":"mcp_message","server_name":"srv","message":{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{"cursor":"c1"}}}`
		var got ControlReqMcpMessage
		if err := internal.UnmarshalJSON([]byte(data), &got); err != nil {
			t.Fatal(err)
		}
		if got.Message.JSONRPC != "2.0" || got.Message.Method != "tools/list" {
			t.Fatalf("Message = %+v, want JSON-RPC tools/list", got.Message)
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
	t.Run("web_search", func(t *testing.T) {
		co, err := parseOpts([]genai.GenOption{&genai.GenOptionWeb{Search: true}})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		want := []string{"WebSearch", "WebFetch"}
		if !slices.Equal(co.tools, want) {
			t.Errorf("tools: got %v, want %v", co.tools, want)
		}
	})
	t.Run("web_fetch", func(t *testing.T) {
		co, err := parseOpts([]genai.GenOption{&genai.GenOptionWeb{Fetch: true}})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		want := []string{"WebFetch"}
		if !slices.Equal(co.tools, want) {
			t.Errorf("tools: got %v, want %v", co.tools, want)
		}
	})
	t.Run("web_with_tools", func(t *testing.T) {
		co, err := parseOpts([]genai.GenOption{
			&GenOption{Tools: []string{"Bash"}},
			&genai.GenOptionWeb{Search: true, Fetch: true},
		})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		want := []string{"Bash", "WebSearch", "WebFetch"}
		if !slices.Equal(co.tools, want) {
			t.Errorf("tools: got %v, want %v", co.tools, want)
		}
	})
	t.Run("effort_enables_progress_summaries", func(t *testing.T) {
		co, err := parseOpts([]genai.GenOption{&GenOption{Effort: EffortMedium}})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if !co.progressSummaries {
			t.Fatal("progressSummaries = false, want true")
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

func init() {
	internal.BeLenient = false
}
