// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Tests for the Pi wire types.

package pi

import (
	"encoding/json"
	"testing"

	"github.com/maruel/genai/internal"
)

func TestModelThinkingLevelMap(t *testing.T) {
	var m Model
	if err := json.Unmarshal([]byte(`{"reasoning":true,"thinkingLevelMap":{"off":null,"high":"high","max":"max"}}`), &m); err != nil {
		t.Fatal(err)
	}

	off, ok := m.ThinkingLevelMap[ThinkingOff]
	if !ok || off != "" {
		t.Fatalf("ThinkingLevelMap[off] = %q, %t; want empty string, true", off, ok)
	}
	for level, want := range map[ThinkingLevel]string{ThinkingHigh: "high", ThinkingMax: "max"} {
		got, ok := m.ThinkingLevelMap[level]
		if !ok || got != want {
			t.Fatalf("ThinkingLevelMap[%q] = %q, %t; want %q, true", level, got, ok, want)
		}
	}
}

func TestV0841DTOs(t *testing.T) {
	t.Run("usage", func(t *testing.T) {
		var m AgentMessage
		if err := json.Unmarshal([]byte(`{"role":"assistant","usage":{"input":8,"output":5,"cacheRead":3,"cacheWrite":2,"cacheWrite1h":1,"reasoning":4,"totalTokens":18},"deferred":{"provider":"test","modelId":"model","api":"api","id":"response","data":{"key":"value"}}}`), &m); err != nil {
			t.Fatal(err)
		}
		if m.Usage.CacheWrite1h != 1 || m.Usage.Reasoning != 4 {
			t.Errorf("usage = %#v, want cacheWrite1h=1 and reasoning=4", m.Usage)
		}
		if got := string(m.Deferred.Data); got != `{"key":"value"}` {
			t.Errorf("deferred data = %s, want object", got)
		}
	})

	t.Run("session tree", func(t *testing.T) {
		var d TreeData
		if err := json.Unmarshal([]byte(`{"tree":[{"entry":{"type":"custom_message","id":"entry","parentId":null,"timestamp":"2026-08-07T00:00:00Z","content":"note"},"children":[]}],"leafId":null}`), &d); err != nil {
			t.Fatal(err)
		}
		if len(d.Tree) != 1 || len(d.Tree[0].Entry.Content) != 1 || d.Tree[0].Entry.Content[0].Text != "note" {
			t.Errorf("tree = %#v, want custom-message text", d.Tree)
		}
	})
}

func TestV0842DTOs(t *testing.T) {
	t.Run("session lifecycle", func(t *testing.T) {
		var end AgentEndEvent
		if err := json.Unmarshal([]byte(`{"type":"agent_end","messages":[],"willRetry":true}`), &end); err != nil {
			t.Fatal(err)
		}
		if !end.WillRetry {
			t.Error("WillRetry = false, want true")
		}

		var retry AutoRetryStartEvent
		if err := json.Unmarshal([]byte(`{"type":"auto_retry_start","attempt":1,"maxAttempts":3,"delayMs":2000,"errorMessage":"502"}`), &retry); err != nil {
			t.Fatal(err)
		}
		if retry.DelayMS != 2000 || retry.MaxAttempts != 3 {
			t.Errorf("retry = %#v, want delay 2000 and max attempts 3", retry)
		}

		var queue QueueUpdateEvent
		if err := json.Unmarshal([]byte(`{"type":"queue_update","steering":["revise"],"followUp":["summarize"]}`), &queue); err != nil {
			t.Fatal(err)
		}
		if len(queue.Steering) != 1 || len(queue.FollowUp) != 1 {
			t.Errorf("queue = %#v, want one item in each queue", queue)
		}
	})

	t.Run("entry appended", func(t *testing.T) {
		var event EntryAppendedEvent
		input := `{"type":"entry_appended","entry":{"type":"custom","customType":"web-search-results","data":{"id":"search"},"id":"entry","parentId":"parent","timestamp":"2026-08-20T19:40:10.328Z"}}`
		if err := json.Unmarshal([]byte(input), &event); err != nil {
			t.Fatal(err)
		}
		if event.Type != EventEntryAppended || event.Entry.CustomType != "web-search-results" || string(event.Entry.Data) != `{"id":"search"}` {
			t.Errorf("entry appended = %#v, want custom web-search entry", event)
		}
	})

	t.Run("compaction", func(t *testing.T) {
		var event CompactionEndEvent
		input := `{"type":"compaction_end","reason":"overflow","result":{"summary":"summary","firstKeptEntryId":"entry","tokensBefore":100,"estimatedTokensAfter":20,"usage":{"input":10,"output":2,"totalTokens":12},"details":{"source":"test"}},"aborted":false,"willRetry":true}`
		if err := json.Unmarshal([]byte(input), &event); err != nil {
			t.Fatal(err)
		}
		if event.Reason != CompactionOverflow || event.Result == nil || event.Result.EstimatedTokensAfter != 20 || event.Result.Usage == nil || event.Result.Usage.TotalTokens != 12 || !event.WillRetry {
			t.Errorf("compaction event = %#v, want decoded compaction result", event)
		}
	})

	t.Run("message update", func(t *testing.T) {
		var event MessageUpdateDeltaEvent
		input := `{"type":"message_update","usage":{"input":10,"output":2,"totalTokens":12},"assistantMessageEvent":{"type":"thinking_end","contentIndex":1,"content":"reasoning"}}`
		if err := json.Unmarshal([]byte(input), &event); err != nil {
			t.Fatal(err)
		}
		if event.Usage.TotalTokens != 12 || event.AssistantMessageEvent.ContentIndex != 1 || event.AssistantMessageEvent.Content != "reasoning" {
			t.Errorf("message update = %#v, want usage and content metadata", event)
		}
	})

	t.Run("custom message", func(t *testing.T) {
		var message AgentMessage
		if err := json.Unmarshal([]byte(`{"role":"custom","customType":"subagent-notify","content":"completed","display":false}`), &message); err != nil {
			t.Fatal(err)
		}
		if message.CustomType != "subagent-notify" || message.Display {
			t.Errorf("message = %#v, want custom type with display=false", message)
		}
	})
}

func TestV0844DTOs(t *testing.T) {
	t.Run("tool call start", func(t *testing.T) {
		var event MessageUpdateEvent
		input := `{"type":"message_update","usage":{"input":0,"output":0,"cacheRead":0,"cacheWrite":0,"totalTokens":0},"assistantMessageEvent":{"type":"toolcall_start","contentIndex":1,"id":"call_1","toolName":"read"}}`
		if err := internal.UnmarshalJSON([]byte(input), &event); err != nil {
			t.Fatal(err)
		}
		if event.AssistantMessageEvent.ID != "call_1" || event.AssistantMessageEvent.ToolName != "read" {
			t.Errorf("message update = %#v, want tool call ID and name", event)
		}
	})
}

func TestToolExecResult(t *testing.T) {
	t.Run("valid", func(t *testing.T) {
		data := []struct {
			name string
			in   string
			want string
		}{
			{
				name: "text_content",
				in:   `{"content":[{"type":"text","text":"hello world"}]}`,
				want: "hello world",
			},
			{
				name: "multiple_blocks",
				in:   `{"content":[{"type":"text","text":"line1\n"},{"type":"text","text":"line2\n"}]}`,
				want: "line1\nline2\n",
			},
			{
				name: "empty_content",
				in:   `{"content":[]}`,
				want: "",
			},
			{
				name: "string_content",
				in:   `{"content":"plain text"}`,
				want: "plain text",
			},
		}
		for _, tc := range data {
			t.Run(tc.name, func(t *testing.T) {
				var r ToolExecResult
				if err := json.Unmarshal([]byte(tc.in), &r); err != nil {
					t.Fatal(err)
				}
				if got := r.Text(); got != tc.want {
					t.Errorf("Text() = %q, want %q", got, tc.want)
				}
			})
		}
	})

	t.Run("zero_value", func(t *testing.T) {
		var r ToolExecResult
		if got := r.Text(); got != "" {
			t.Errorf("zero.Text() = %q, want empty", got)
		}
	})

	t.Run("unmarshal_update_event", func(t *testing.T) {
		raw := `{"type":"tool_execution_update","toolCallId":"call_1","toolName":"bash","args":{"command":"ls"},"partialResult":{"content":[{"type":"text","text":"file1\nfile2\n"}]}}`
		var ev ToolExecUpdateEvent
		if err := json.Unmarshal([]byte(raw), &ev); err != nil {
			t.Fatal(err)
		}
		if got := ev.PartialResult.Text(); got != "file1\nfile2\n" {
			t.Errorf("PartialResult.Text() = %q", got)
		}
	})

	t.Run("unmarshal_end_event", func(t *testing.T) {
		raw := `{"type":"tool_execution_end","toolCallId":"call_1","toolName":"read","result":{"content":[{"type":"text","text":"# README\nHello"}],"isError":true,"details":{"source":"tool"}},"isError":true}`
		var ev ToolExecEndEvent
		if err := json.Unmarshal([]byte(raw), &ev); err != nil {
			t.Fatal(err)
		}
		if got := ev.Result.Text(); got != "# README\nHello" {
			t.Errorf("Result.Text() = %q", got)
		}
		if !ev.IsError || !ev.Result.IsError || string(ev.Result.Details) != `{"source":"tool"}` {
			t.Errorf("result = %#v, want error details", ev.Result)
		}
	})
}
