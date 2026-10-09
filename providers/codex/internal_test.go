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

func TestProviderOption(t *testing.T) {
	t.Run("Validate", func(t *testing.T) {
		t.Run("valid", func(t *testing.T) {
			for _, p := range []ProviderOption{
				{},
				{Effort: ReasoningEffortHigh},
				{DangerouslySkipPermissions: true, ExtraArgs: []string{"-c", "model_verbosity=\"low\""}},
			} {
				if err := p.Validate(); err != nil {
					t.Errorf("%+v: %v", p, err)
				}
			}
		})
		t.Run("error", func(t *testing.T) {
			for _, p := range []ProviderOption{
				{Effort: "turbo"},
				{ExtraArgs: []string{"--listen", "ws://127.0.0.1:1"}},
			} {
				if err := p.Validate(); err == nil {
					t.Errorf("%+v: expected error", p)
				}
			}
		})
	})
}

func TestGenOption(t *testing.T) {
	t.Run("Validate", func(t *testing.T) {
		if err := (&GenOption{Effort: ReasoningEffortHigh}).Validate(); err != nil {
			t.Error(err)
		}
		if err := (&GenOption{Effort: "turbo"}).Validate(); err == nil {
			t.Error("expected error")
		}
	})
}

func TestParseOpts(t *testing.T) {
	t.Run("default_effort", func(t *testing.T) {
		for _, tc := range []struct {
			name string
			p    ProviderOption
			opts []genai.GenOption
			want ReasoningEffort
		}{
			{"default", ProviderOption{}, nil, ReasoningEffortMedium},
			{"provider", ProviderOption{Effort: ReasoningEffortLow}, nil, ReasoningEffortLow},
			{"turn", ProviderOption{Effort: ReasoningEffortLow}, []genai.GenOption{&GenOption{Effort: ReasoningEffortHigh}}, ReasoningEffortHigh},
		} {
			co, err := parseOpts(&tc.p, tc.opts)
			if err != nil {
				t.Fatal(err)
			}
			if co.effort != tc.want {
				t.Errorf("%s: effort = %q, want %q", tc.name, co.effort, tc.want)
			}
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

func TestHandshake(t *testing.T) {
	t.Run("dangerously_skip_permissions", func(t *testing.T) {
		responses := strings.Join([]string{
			`{"id":1,"result":{}}`,
			`{"id":2,"result":{"data":[]}}`,
			`{"id":3,"result":{"thread":{"id":"thread"}}}`,
		}, "\n")
		var out bytes.Buffer
		co := callOpts{skipPermissions: true}
		if _, err := handshake(&out, bufio.NewScanner(strings.NewReader(responses)), "model", "", &co); err != nil {
			t.Fatal(err)
		}
		lines := strings.Split(strings.TrimSpace(out.String()), "\n")
		var req JSONRPCRequest
		if err := json.Unmarshal([]byte(lines[3]), &req); err != nil {
			t.Fatal(err)
		}
		var params ThreadStartParams
		if err := json.Unmarshal(req.Params, &params); err != nil {
			t.Fatal(err)
		}
		if string(params.ApprovalPolicy) != `"never"` || params.Sandbox != SandboxModeDangerFullAccess {
			t.Errorf("approval = %s, sandbox = %q, want never and danger-full-access", params.ApprovalPolicy, params.Sandbox)
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
	t.Run("error_notification", func(t *testing.T) {
		line := `{"method":"error","params":{"error":{"message":"internal server error","codexErrorInfo":null,"additionalDetails":null,"misalignment":null},"willRetry":false,"threadId":"thread","turnId":"turn"}}`
		_, err := readTurnSync(bufio.NewScanner(strings.NewReader(line)), "thread")
		if err == nil || err.Error() != "codex error: internal server error" {
			t.Fatalf("error = %v", err)
		}
	})
	t.Run("misalignment", func(t *testing.T) {
		line := `{"method":"turn/completed","params":{"threadId":"thread","turn":{"id":"turn","items":[],"itemsView":"full","status":"failed","error":{"message":"blocked","codexErrorInfo":null,"additionalDetails":null,"misalignment":{"errorType":"policy","detailedExplanation":"details","steer":{"message":"continue"},"reviewTarget":"opaque-block"}},"startedAt":null,"completedAt":null,"durationMs":null}}}`
		_, err := readTurnSync(bufio.NewScanner(strings.NewReader(line)), "thread")
		if err == nil || err.Error() != "blocked" {
			t.Fatalf("error = %v", err)
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
}

func TestDurationMS(t *testing.T) {
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
}

func init() {
	internal.BeLenient = false
}
