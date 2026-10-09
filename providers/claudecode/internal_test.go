// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Internal tests for the Claude Code provider.

package claudecode

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/maruel/genai"
	"github.com/maruel/genai/base"
	"github.com/maruel/genai/internal"
	"github.com/maruel/genai/internal/internaltest"
)

func TestBuildArgs(t *testing.T) {
	t.Run("dangerously_skip_permissions_and_extra_args", func(t *testing.T) {
		co, err := parseOpts(&ProviderOption{DangerouslySkipPermissions: true, ExtraArgs: []string{"--add-dir", "/src"}}, nil)
		if err != nil {
			t.Fatal(err)
		}
		c := &Client{}
		internaltest.CleanupCloser(t, c)
		args := c.buildArgs(&co, "", false)
		if i := slices.Index(args, "--permission-mode"); i < 0 || args[i+1] != "bypassPermissions" {
			t.Errorf("want --permission-mode bypassPermissions: %v", args)
		}
		if !slices.Equal(args[len(args)-2:], []string{"--add-dir", "/src"}) {
			t.Errorf("want ExtraArgs last: %v", args)
		}
	})
	t.Run("with_system_prompt", func(t *testing.T) {
		c := &Client{}
		internaltest.CleanupCloser(t, c)
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

func TestHandleControlRequest(t *testing.T) {
	t.Run("valid", func(t *testing.T) {
		const data = `{"type":"control_request","request_id":"r1","request":{"subtype":"can_use_tool","tool_name":"AskUserQuestion","input":{"questions":[{"question":"Which option?","options":[{"label":"A"},{"label":"B"}]}]},"tool_use_id":"toolu_1"}}`
		h := func(context.Context, OutputControlRequestMsg) (InputControlResponseMsg, error) {
			return InputControlResponseMsg{
				Response: ControlResponse{
					Subtype:  ControlResponseSuccess,
					Response: ControlResponsePayload{Behavior: ControlCanUseToolBehaviorAllow},
				},
			}, nil
		}
		var buf bytes.Buffer
		if err := handleControlRequest(t.Context(), &buf, h, []byte(data)); err != nil {
			t.Fatal(err)
		}
		var got InputControlResponseMsg
		if err := internal.UnmarshalJSON(buf.Bytes(), &got); err != nil {
			t.Fatal(err)
		}
		if got.Type != InputControlResponse {
			t.Errorf("Type = %q, want %q", got.Type, InputControlResponse)
		}
		if got.Response.RequestID != "r1" {
			t.Errorf("RequestID = %q, want r1", got.Response.RequestID)
		}
		if got.Response.Response.Behavior != ControlCanUseToolBehaviorAllow {
			t.Errorf("behavior = %q, want %q", got.Response.Response.Behavior, ControlCanUseToolBehaviorAllow)
		}
		var updated AskUserQuestionInput
		if err := json.Unmarshal(got.Response.Response.UpdatedInput, &updated); err != nil {
			t.Fatal(err)
		}
		if len(updated.Questions) != 1 || updated.Questions[0].Question != "Which option?" {
			t.Errorf("updatedInput = %+v, want original AskUserQuestion input", updated)
		}
	})
	t.Run("missing_handler", func(t *testing.T) {
		const data = `{"type":"control_request","request_id":"r1","request":{"subtype":"can_use_tool","tool_name":"Bash","input":{"command":"git status"},"tool_use_id":"toolu_1"}}`
		var buf bytes.Buffer
		err := handleControlRequest(t.Context(), &buf, nil, []byte(data))
		if err == nil {
			t.Fatal("expected error, got nil")
		}
		if !strings.Contains(err.Error(), "no control handler") {
			t.Errorf("error = %v, want no control handler", err)
		}
	})
	t.Run("malformed_can_use_tool", func(t *testing.T) {
		const data = `{"type":"control_request","request_id":"r1","request":42}`
		h := func(context.Context, OutputControlRequestMsg) (InputControlResponseMsg, error) {
			return InputControlResponseMsg{
				Response: ControlResponse{
					Subtype:  ControlResponseSuccess,
					Response: ControlResponsePayload{Behavior: ControlCanUseToolBehaviorAllow},
				},
			}, nil
		}
		var buf bytes.Buffer
		err := handleControlRequest(t.Context(), &buf, h, []byte(data))
		if err == nil {
			t.Fatal("expected error, got nil")
		}
		if !strings.Contains(err.Error(), "decode can_use_tool control request") {
			t.Errorf("error = %v, want decode can_use_tool control request", err)
		}
	})
}

func TestOutputMessages(t *testing.T) {
	t.Run("result_fallback_credit", func(t *testing.T) {
		for _, tc := range []struct {
			name string
			raw  string
			typ  FallbackCreditType
		}{
			{"notApplied", `{"status":{"type":"not_applied","reason":"variant_fields_present","remove_to_redeem":["temperature"]}}`, FallbackCreditNotApplied},
		} {
			t.Run(tc.name, func(t *testing.T) {
				var got OutputResultMsg
				if err := internal.UnmarshalJSON([]byte(`{"type":"result","usage":{"fallback_credit":`+tc.raw+`},"modelUsage":{"claude-sonnet-5-5":{"canonicalModel":"claude-sonnet-5-5","provider":"firstParty","costBasis":"list","thinkingTokens":0}}}`), &got); err != nil {
					t.Fatal(err)
				}
				if got.Usage.IsZero() != (tc.typ == "") || got.Usage.FallbackCredit.Status.Type != tc.typ {
					t.Fatalf("unexpected usage: %+v", got.Usage)
				}
				if tc.typ == FallbackCreditNotApplied && (got.Usage.FallbackCredit.Status.Reason != FallbackCreditVariantFieldsPresent || !slices.Equal(got.Usage.FallbackCredit.Status.RemoveToRedeem, []string{"temperature"})) {
					t.Fatalf("unexpected fallback credit: %+v", got.Usage.FallbackCredit)
				}
			})
		}
	})
	t.Run("api_retry_fractional_delay", func(t *testing.T) {
		const data = `{"type":"system","subtype":"api_retry","attempt":1,"max_retries":10,"retry_delay_ms":599.3873493672435,"error_status":401,"error":"authentication_failed","session_id":"s1","uuid":"u1"}`
		var got OutputSystemMsg
		if err := internal.UnmarshalJSON([]byte(data), &got); err != nil {
			t.Fatal(err)
		}
		if got.RetryDelay != base.DurationMS(599.3873493672435) {
			t.Errorf("RetryDelay = %.13f, want 599.3873493672435", got.RetryDelay)
		}
		if got.RetryDelay.AsDuration() != 599*time.Millisecond+387*time.Microsecond+349*time.Nanosecond {
			t.Errorf("RetryDelay.AsDuration() = %v", got.RetryDelay.AsDuration())
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
		if got.Patch.Status != TaskPatchStatusCompleted {
			t.Errorf("Patch.Status = %q, want %q", got.Patch.Status, TaskPatchStatusCompleted)
		}
		if got.Patch.EndTime != base.TimeMS(1780832660165) {
			t.Errorf("Patch.EndTime = %v, want 1780832660165", got.Patch.EndTime)
		}
		if got.Patch.EndTime.AsTime() != time.Date(2026, 6, 7, 11, 44, 20, 165000000, time.UTC) {
			t.Errorf("Patch.EndTime.AsTime() = %v", got.Patch.EndTime.AsTime())
		}
	})
	t.Run("synthetic_refusal_assistant", func(t *testing.T) {
		const data = `{"type":"assistant","message":{"id":"m1","container":null,"model":"<synthetic>","role":"assistant","stop_details":{"type":"refusal","category":"bio","explanation":null,"fallback_has_prefill_claim":null,"recommended_model":null},"stop_reason":"refusal","stop_sequence":"","type":"message","usage":{"input_tokens":0,"output_tokens":0,"cache_creation_input_tokens":0,"cache_read_input_tokens":0,"server_tool_use":{"web_search_requests":0},"service_tier":null,"cache_creation":{"ephemeral_1h_input_tokens":0,"ephemeral_5m_input_tokens":0},"inference_geo":null,"iterations":null,"speed":null},"content":[{"type":"text","text":"API Error: blocked"}],"context_management":null},"parent_tool_use_id":null,"session_id":"s1","uuid":"u1","error":"invalid_request","request_id":"req_1"}`
		var got OutputAssistantMsg
		if err := internal.UnmarshalJSON([]byte(data), &got); err != nil {
			t.Fatal(err)
		}
		if got.Message.StopDetails.Category != "bio" || got.Error != "invalid_request" {
			t.Fatalf("Message.StopDetails = %+v, Error = %q, want bio invalid_request", got.Message.StopDetails, got.Error)
		}
	})
	t.Run("user_inline_tool_result_error", func(t *testing.T) {
		const data = `{"type":"user","message":{"role":"user","content":[{"type":"tool_result","content":"Answer questions?","is_error":true,"tool_use_id":"toolu_ask"}]},"parent_tool_use_id":null,"session_id":"s1","uuid":"u1","timestamp":"2026-06-23T18:51:57.326Z","tool_use_result":"Error: Answer questions?"}`
		var got OutputUserMsg
		if err := internal.UnmarshalJSON([]byte(data), &got); err != nil {
			t.Fatal(err)
		}
		msg, err := got.DecodeMessage()
		if err != nil {
			t.Fatal(err)
		}
		if msg.Kind != OutputUserMessageBlock {
			t.Fatalf("Kind = %q, want %q", msg.Kind, OutputUserMessageBlock)
		}
		if len(msg.Content) != 1 {
			t.Fatalf("len(Content) = %d, want 1", len(msg.Content))
		}
		block := msg.Content[0]
		if block.ToolUseID != "toolu_ask" || !block.IsError || block.Content.Text != "Answer questions?" {
			t.Fatalf("tool result block = %+v, want errored AskUserQuestion result", block)
		}
		res, err := got.DecodeToolUseResult()
		if err != nil {
			t.Fatal(err)
		}
		if res.Text != "Error: Answer questions?" {
			t.Errorf("ToolUseResult.Text = %q, want error summary", res.Text)
		}
	})
	t.Run("user_top_level_tool_result", func(t *testing.T) {
		const data = `{"type":"user","message":{"content":[{"type":"text","text":"file not found"}],"is_error":true},"parent_tool_use_id":"toolu_read"}`
		var got OutputUserMsg
		if err := internal.UnmarshalJSON([]byte(data), &got); err != nil {
			t.Fatal(err)
		}
		msg, err := got.DecodeMessage()
		if err != nil {
			t.Fatal(err)
		}
		if msg.Kind != OutputUserMessageToolResult {
			t.Fatalf("Kind = %q, want %q", msg.Kind, OutputUserMessageToolResult)
		}
		if !msg.ToolResult.IsError || len(msg.ToolResult.Content.Blocks) != 1 || msg.ToolResult.Content.Blocks[0].Text != "file not found" {
			t.Fatalf("ToolResult = %+v, want error block content", msg.ToolResult)
		}
	})
	t.Run("tool_progress", func(t *testing.T) {
		const data = `{"type":"tool_progress","tool_use_id":"toolu_1","tool_name":"Bash","parent_tool_use_id":null,"elapsed_time_seconds":1.5,"heartbeat":true,"uuid":"u1","session_id":"s1"}`
		var got OutputToolProgressMsg
		if err := internal.UnmarshalJSON([]byte(data), &got); err != nil {
			t.Fatal(err)
		}
		if got.ElapsedTime != base.DurationS(1.5) {
			t.Errorf("ElapsedTime = %v, want 1.5", got.ElapsedTime)
		}
		if got.ElapsedTime.AsDuration() != 1500*time.Millisecond {
			t.Errorf("ElapsedTime.AsDuration() = %v", got.ElapsedTime.AsDuration())
		}
		if !got.Heartbeat {
			t.Error("Heartbeat = false, want true")
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
}

func TestWriteUserMsg(t *testing.T) {
	t.Run("empty_message", func(t *testing.T) {
		msg := genai.NewTextMessage("")
		var buf strings.Builder
		if err := writeUserMsg(&buf, &msg); err == nil {
			t.Fatal("expected error for empty message")
		}
	})
}

func TestProviderOption(t *testing.T) {
	t.Run("errors", func(t *testing.T) {
		if err := (&ProviderOption{Tools: []string{""}}).Validate(); err == nil {
			t.Error("expected error for empty tool name")
		}
		if err := (&ProviderOption{MaxBudgetUSD: -1}).Validate(); err == nil {
			t.Error("expected error for negative budget")
		}
		if err := (&ProviderOption{PermissionMode: "hack"}).Validate(); err == nil {
			t.Error("expected error for invalid mode")
		}
		if err := (&ProviderOption{Effort: "extreme"}).Validate(); err == nil {
			t.Error("expected error for invalid effort")
		}
		if err := (&ProviderOption{DangerouslySkipPermissions: true, PermissionMode: "plan"}).Validate(); err == nil {
			t.Error("expected error for conflicting permission mode")
		}
		if err := (&ProviderOption{ExtraArgs: []string{"--output-format", "text"}}).Validate(); err == nil {
			t.Error("expected error for reserved ExtraArgs flag")
		}
	})
	t.Run("unsupported", func(t *testing.T) {
		for _, tc := range []struct {
			name string
			opts []genai.GenOption
			want string
		}{
			{"Temperature", []genai.GenOption{&genai.GenOptionText{Temperature: 0.5}}, "GenOptionText.Temperature"},
		} {
			t.Run(tc.name, func(t *testing.T) {
				_, err := parseOpts(&ProviderOption{}, tc.opts)
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

func init() {
	internal.BeLenient = false
}

func TestUserMessageContent(t *testing.T) {
	t.Run("valid", func(t *testing.T) {
		for _, data := range []string{`""`, `"pasted text"`, `[]`, `[{"type":"text","text":"pasted block"}]`} {
			var got UserMessageContent
			if err := json.Unmarshal([]byte(data), &got); err != nil {
				t.Fatal(err)
			}
			out, err := json.Marshal(got)
			if err != nil {
				t.Fatal(err)
			}
			if string(out) != data {
				t.Errorf("round trip = %s, want %s", out, data)
			}
		}
	})
	t.Run("error", func(t *testing.T) {
		for _, data := range []string{`null`, `{}`, `1`} {
			var got UserMessageContent
			if err := json.Unmarshal([]byte(data), &got); err == nil {
				t.Errorf("accepted invalid user content %s", data)
			}
		}
	})
}
