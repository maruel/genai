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
