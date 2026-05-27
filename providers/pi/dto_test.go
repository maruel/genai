// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Tests for the Pi wire types.

package pi

import (
	"encoding/json"
	"testing"
)

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
		raw := `{"type":"tool_execution_end","toolCallId":"call_1","toolName":"read","result":{"content":[{"type":"text","text":"# README\nHello"}]}}`
		var ev ToolExecEndEvent
		if err := json.Unmarshal([]byte(raw), &ev); err != nil {
			t.Fatal(err)
		}
		if got := ev.Result.Text(); got != "# README\nHello" {
			t.Errorf("Result.Text() = %q", got)
		}
	})
}
