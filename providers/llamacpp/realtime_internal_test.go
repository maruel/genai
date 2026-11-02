// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Tests llama.cpp realtime session error paths with scripted peers.

package llamacpp

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"iter"
	"net/http"
	"strings"
	"testing"
	"time"

	"github.com/coder/websocket"

	"github.com/maruel/genai"
	"github.com/maruel/genai/internal/realtime"
	"github.com/maruel/genai/internal/realtimetest"
)

func newRealtimeTestSession(t *testing.T, names []string, script func(context.Context, *websocket.Conn)) *RealtimeSession {
	u := realtimetest.Peer(t, nil, script)
	ws, err := realtime.Dial(t.Context(), "ws"+strings.TrimPrefix(u, "http"), http.DefaultClient, maxMessageBytes, time.Hour, time.Second)
	if err != nil {
		t.Fatal(err)
	}
	info := genai.RealtimeSessionInfo{
		InputFormat:   genai.AudioFormat{SampleRate: 16000},
		OutputFormat:  genai.AudioFormat{SampleRate: 24000},
		TurnDetection: genai.RealtimeTurnManual,
		Capabilities: genai.RealtimeCapabilities{
			AudioCommit:    true,
			CancelResponse: true,
			ToolResults:    len(names) != 0,
			Update: genai.RealtimeUpdateCapabilities{
				Instructions:       true,
				Tools:              true,
				ToolChoice:         true,
				ToolChoiceRequired: true,
				MaxOutputTokens:    true,
				OutputModalities:   true,
			},
		},
	}
	s := newRealtimeSession(ws, &info, names)
	t.Cleanup(func() {
		if err := s.Close(); err != nil {
			t.Error(err)
		}
	})
	return s
}

func expectRealtimeFrame(t *testing.T, ctx context.Context, ws *websocket.Conn, want string) map[string]any {
	m := realtimetest.Read(t, ctx, ws)
	if m["type"] != want {
		t.Errorf("frame %#v, want %s", m, want)
	}
	return m
}

func waitRealtimeSignal(t *testing.T, ch <-chan struct{}, what string) {
	select {
	case <-ch:
	case <-time.After(5 * time.Second):
		t.Fatalf("timeout waiting for %s", what)
	}
}

func sendRealtimeError(t *testing.T, s genai.RealtimeSession, in genai.RealtimeInput, want string) {
	err := s.Send(t.Context(), in)
	if err == nil || !strings.Contains(err.Error(), want) {
		t.Fatalf("Send(%T) = %v, want %q", in, err, want)
	}
}

func TestRealtimeSessionSendErrors(t *testing.T) {
	run := func(t *testing.T, names []string, frames []string, setup []genai.RealtimeInput, invalid genai.RealtimeInput, want string, sentinel genai.RealtimeInput) {
		sent := make(chan struct{})
		s := newRealtimeTestSession(t, names, func(ctx context.Context, ws *websocket.Conn) {
			for _, f := range frames {
				expectRealtimeFrame(t, ctx, ws, f)
			}
			close(sent)
			<-ctx.Done()
		})
		for _, in := range setup {
			if err := s.Send(t.Context(), in); err != nil {
				t.Fatal(err)
			}
		}
		sendRealtimeError(t, s, invalid, want)
		if err := s.Send(t.Context(), sentinel); err != nil {
			t.Fatal(err)
		}
		waitRealtimeSignal(t, sent, "sentinel frame")
	}

	t.Run("text needs audio commit", func(t *testing.T) {
		run(t, nil, []string{"input_audio_buffer.append", "input_audio_buffer.append", "input_audio_buffer.commit", "response.create"}, []genai.RealtimeInput{genai.RealtimeInputContent{Audio: []byte{1, 0}}}, genai.RealtimeInputContent{Text: "turn", Generate: true}, "audio requires commit before generation", genai.RealtimeInputContent{Audio: []byte{2, 0}, Commit: true, Generate: true})
	})
	t.Run("tool results after user turn", func(t *testing.T) {
		run(t, []string{"lookup"}, []string{"conversation.item.create", "conversation.item.create"}, []genai.RealtimeInput{genai.RealtimeInputContent{Text: "turn"}}, genai.RealtimeInputContent{ToolResults: []genai.ToolCallResult{{ID: "c", Name: "lookup", Result: "v"}}}, "tool results cannot follow a new user turn", genai.RealtimeInputContent{Text: "sentinel"})
	})
	t.Run("empty generation without turn", func(t *testing.T) {
		run(t, nil, []string{"conversation.item.create"}, nil, genai.RealtimeInputContent{Generate: true}, "requires a committed user turn or tool results", genai.RealtimeInputContent{Text: "sentinel"})
	})
	t.Run("empty commit", func(t *testing.T) {
		run(t, nil, []string{"conversation.item.create"}, nil, genai.RealtimeInputContent{Commit: true}, "cannot commit an empty audio buffer", genai.RealtimeInputContent{Text: "sentinel"})
	})
	t.Run("empty audio", func(t *testing.T) {
		run(t, nil, []string{"conversation.item.create"}, nil, genai.RealtimeInputContent{Audio: []byte{}}, "audio requires samples unless Commit", genai.RealtimeInputContent{Text: "sentinel"})
	})
	t.Run("unsupported truncate", func(t *testing.T) {
		run(t, nil, []string{"conversation.item.create"}, nil, genai.RealtimeTruncate{ItemID: "item"}, "not supported", genai.RealtimeInputContent{Text: "sentinel"})
	})

	t.Run("idle cancel", func(t *testing.T) {
		sent := make(chan struct{})
		s := newRealtimeTestSession(t, nil, func(ctx context.Context, ws *websocket.Conn) {
			expectRealtimeFrame(t, ctx, ws, "conversation.item.create")
			close(sent)
			<-ctx.Done()
		})
		if err := s.Send(t.Context(), genai.RealtimeCancel{}); !errors.Is(err, ErrNoActiveResponse) {
			t.Fatalf("idle cancel: %v", err)
		}
		if err := s.Send(t.Context(), genai.RealtimeInputContent{Text: "sentinel"}); err != nil {
			t.Fatal(err)
		}
		waitRealtimeSignal(t, sent, "sentinel frame")
	})
}

func TestRealtimeAudioCommitAndEmptyTextGenerate(t *testing.T) {
	sent := make(chan struct{})
	s := newRealtimeTestSession(t, nil, func(ctx context.Context, ws *websocket.Conn) {
		expectRealtimeFrame(t, ctx, ws, "input_audio_buffer.append")
		expectRealtimeFrame(t, ctx, ws, "input_audio_buffer.commit")
		expectRealtimeFrame(t, ctx, ws, "response.create")
		close(sent)
		<-ctx.Done()
	})
	if err := s.Send(t.Context(), genai.RealtimeInputContent{Audio: []byte{1, 0}, Commit: true}); err != nil {
		t.Fatal(err)
	}
	if err := s.Send(t.Context(), genai.RealtimeInputContent{Generate: true}); err != nil {
		t.Fatal(err)
	}
	waitRealtimeSignal(t, sent, "commit and empty text generation")
}

func TestRealtimeAudioGenerateCommits(t *testing.T) {
	sent := make(chan struct{})
	s := newRealtimeTestSession(t, nil, func(ctx context.Context, ws *websocket.Conn) {
		expectRealtimeFrame(t, ctx, ws, "input_audio_buffer.append")
		expectRealtimeFrame(t, ctx, ws, "input_audio_buffer.commit")
		expectRealtimeFrame(t, ctx, ws, "response.create")
		close(sent)
		<-ctx.Done()
	})
	if err := s.Send(t.Context(), genai.RealtimeInputContent{Audio: []byte{1, 0}, Generate: true}); err != nil {
		t.Fatal(err)
	}
	waitRealtimeSignal(t, sent, "audio generation")
}

func TestRealtimeSessionActiveResponseErrors(t *testing.T) {
	for _, tc := range []struct {
		name string
		in   genai.RealtimeInput
		want string
	}{
		{"new input", genai.RealtimeInputContent{Audio: []byte{1, 0}}, "must cancel its active response before new user input"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			sent := make(chan struct{})
			s := newRealtimeTestSession(t, nil, func(ctx context.Context, ws *websocket.Conn) {
				expectRealtimeFrame(t, ctx, ws, "conversation.item.create")
				expectRealtimeFrame(t, ctx, ws, "response.create")
				realtimetest.Write(t, ctx, ws, `{"type":"response.created","response":{"id":"active"}}`)
				realtimetest.Write(t, ctx, ws, `{"type":"response.output_text.delta","response_id":"active","item_id":"text","delta":"x"}`)
				expectRealtimeFrame(t, ctx, ws, "response.cancel")
				close(sent)
				realtimetest.Write(t, ctx, ws, `{"type":"response.done","response":{"id":"active","status":"cancelled"}}`)
				<-ctx.Done()
			})
			if err := s.Send(t.Context(), genai.RealtimeInputContent{Text: "turn", Generate: true}); err != nil {
				t.Fatal(err)
			}
			realtimetest.Receive(t, s, genai.RealtimeEventContentText)
			sendRealtimeError(t, s, tc.in, tc.want)
			if err := s.Send(t.Context(), genai.RealtimeCancel{}); err != nil {
				t.Fatal(err)
			}
			realtimetest.Receive(t, s, genai.RealtimeEventInterrupted)
			realtimetest.Receive(t, s, genai.RealtimeEventGenerationDone)
			realtimetest.Receive(t, s, genai.RealtimeEventTurnDone)
			waitRealtimeSignal(t, sent, "cancel frame")
		})
	}
}

func TestRealtimeSessionToolErrors(t *testing.T) {
	const toolCalls = `{"type":"response.done","response":{"id":"r_tools","status":"completed","output":[{"type":"function_call","call_id":"c","name":"lookup","arguments":"{}"},{"type":"function_call","call_id":"d","name":"lookup","arguments":"{}"}]}}`
	valid := []genai.ToolCallResult{{ID: "d", Name: "lookup", Result: "second"}, {ID: "c", Name: "lookup", Result: "first"}}

	for _, tc := range []struct {
		name string
		in   genai.RealtimeInput
		want string
	}{
		{"unknown id", genai.RealtimeInputContent{ToolResults: []genai.ToolCallResult{{ID: "x", Name: "lookup", Result: "v"}}}, "does not match a pending call"},
		{"wrong name", genai.RealtimeInputContent{ToolResults: []genai.ToolCallResult{{ID: "c", Name: "wrong", Result: "v"}}}, "does not match a pending call"},
		{"incomplete batch", genai.RealtimeInputContent{ToolResults: []genai.ToolCallResult{{ID: "c", Name: "lookup", Result: "v"}}}, "must complete one response batch"},
		{"duplicate result", genai.RealtimeInputContent{ToolResults: []genai.ToolCallResult{{ID: "c", Name: "lookup", Result: "one"}, {ID: "c", Name: "lookup", Result: "two"}}}, "duplicate tool result ID"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			sent := make(chan struct{})
			s := newRealtimeTestSession(t, []string{"lookup"}, func(ctx context.Context, ws *websocket.Conn) {
				realtimetest.Write(t, ctx, ws, `{"type":"response.created","response":{"id":"r_tools"}}`)
				realtimetest.Write(t, ctx, ws, toolCalls)
				for _, want := range []string{"c", "d"} {
					m := expectRealtimeFrame(t, ctx, ws, "conversation.item.create")
					item, _ := m["item"].(map[string]any)
					if item["call_id"] != want {
						t.Errorf("call order %#v, want %s", m, want)
					}
				}
				close(sent)
				<-ctx.Done()
			})
			realtimetest.Receive(t, s, genai.RealtimeEventContentToolCalls)
			sendRealtimeError(t, s, tc.in, tc.want)
			if err := s.Send(t.Context(), genai.RealtimeInputContent{ToolResults: valid}); err != nil {
				t.Fatal(err)
			}
			waitRealtimeSignal(t, sent, "tool results")
		})
	}
}

func TestRealtimeSessionReceiveErrors(t *testing.T) {
	for _, tc := range []struct {
		name  string
		names []string
		in    []string
		want  string
	}{
		{"invalid JSON", nil, []string{`{`}, "invalid realtime event payload"},
		{"invalid audio", nil, []string{`{"type":"response.output_audio.delta","response_id":"r","item_id":"a","delta":"!"}`}, "invalid realtime PCM audio"},
		{"odd audio", nil, []string{`{"type":"response.output_audio.delta","response_id":"r","item_id":"a","delta":"AA=="}`}, "invalid realtime PCM audio"},
		{"audio missing item", nil, []string{`{"type":"response.output_audio.delta","response_id":"r","delta":"AAA="}`}, "invalid realtime PCM audio"},
		{"output text missing response", nil, []string{`{"type":"response.output_text.delta","item_id":"a","delta":"x"}`}, "omitted response identity"},
		{"transcript missing item", nil, []string{`{"type":"response.output_text.delta","response_id":"r","delta":"x"}`}, "omitted item identity"},
		{"response created missing id", nil, []string{`{"type":"response.created","response":{}}`}, "invalid realtime response creation"},
		{"overlapping responses", nil, []string{`{"type":"response.created","response":{"id":"a"}}`, `{"type":"response.created","response":{"id":"b"}}`}, "overlapping realtime responses"},
		{"response done missing id", nil, []string{`{"type":"response.done","response":{}}`}, "invalid realtime response completion"},
		{"response in progress", nil, []string{`{"type":"response.done","response":{"id":"r","status":"in_progress"}}`}, "nonterminal status"},
		{"unknown status", nil, []string{`{"type":"response.done","response":{"id":"r","status":"weird"}}`}, "unknown realtime response completion status"},
		{"negative usage", nil, []string{`{"type":"response.done","response":{"id":"r","status":"completed","usage":{"input_tokens":-1}}}`}, "invalid realtime usage"},
		{"error missing error", nil, []string{`{"type":"error"}`}, "omitted error"},
		{"error correlation", nil, []string{fmt.Sprintf(`{"type":"error","error":{"code":%q}}`, strings.Repeat("x", 129))}, "error correlation exceeds limit"},
		{"tool call missing id", []string{"lookup"}, []string{`{"type":"response.done","response":{"id":"r","status":"completed","output":[{"type":"function_call","name":"lookup","arguments":"{}"}]}}`}, "invalid realtime function call"},
		{"tool call missing name", []string{"lookup"}, []string{`{"type":"response.done","response":{"id":"r","status":"completed","output":[{"type":"function_call","call_id":"c","arguments":"{}"}]}}`}, "invalid realtime function call"},
		{"tool call invalid arguments", []string{"lookup"}, []string{`{"type":"response.done","response":{"id":"r","status":"completed","output":[{"type":"function_call","call_id":"c","name":"lookup","arguments":"{"}]}}`}, "invalid realtime function call"},
		{"tool call undeclared", nil, []string{`{"type":"response.done","response":{"id":"r","status":"completed","output":[{"type":"function_call","call_id":"c","name":"unknown","arguments":"{}"}]}}`}, "names an undeclared tool"},
		{"duplicate tool id", []string{"lookup"}, []string{`{"type":"response.done","response":{"id":"r","status":"completed","output":[{"type":"function_call","call_id":"c","name":"lookup","arguments":"{}"},{"type":"function_call","call_id":"c","name":"lookup","arguments":"{}"}]}}`}, "repeats an ID"},
		{"oversized output", nil, []string{`{"type":"response.done","response":{"id":"r","status":"completed","output":[` + strings.Repeat(`{},`, 1024) + `{}]}}`}, "invalid realtime event payload"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			s := newRealtimeTestSession(t, tc.names, func(ctx context.Context, ws *websocket.Conn) {
				for _, in := range tc.in {
					realtimetest.Write(t, ctx, ws, in)
				}
				<-ctx.Done()
			})
			next, stop := iter.Pull2(s.Receive(t.Context()))
			t.Cleanup(stop)
			var err error
			for range len(tc.in) + 1 {
				var ok bool
				_, err, ok = next()
				if !ok || err != nil {
					break
				}
			}
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("Receive() = %v, want %q", err, tc.want)
			}
			if err := s.Send(t.Context(), genai.RealtimeInputContent{Text: "after"}); err == nil {
				t.Fatal("terminated session accepted input")
			}
		})
	}
}

func TestRealtimeSessionUpdate(t *testing.T) {
	sent := make(chan struct{})
	s := newRealtimeTestSession(t, nil, func(ctx context.Context, ws *websocket.Conn) {
		m := expectRealtimeFrame(t, ctx, ws, "session.update")
		upd, _ := m["session"].(map[string]any)
		if upd["instructions"] != "be terse" {
			t.Errorf("instructions %#v", m)
		}
		tools, _ := upd["tools"].([]any)
		if len(tools) != 1 || tools[0].(map[string]any)["name"] != "lookup" {
			t.Errorf("tools %#v", m)
		}
		expectRealtimeFrame(t, ctx, ws, "conversation.item.create")
		m = expectRealtimeFrame(t, ctx, ws, "session.update")
		upd, _ = m["session"].(map[string]any)
		if upd["tool_choice"] != "none" || upd["max_output_tokens"] != float64(128) {
			t.Errorf("late update %#v", m)
		}
		mods, _ := upd["output_modalities"].([]any)
		if len(mods) != 1 || mods[0] != "text" {
			t.Errorf("output modalities %#v", m)
		}
		close(sent)
	})
	instr := "be terse"
	tools := []genai.ToolDef{{Name: "lookup", Description: "Lookup", InputSchemaOverride: json.RawMessage(`{"type":"object"}`)}}
	if err := s.Send(t.Context(), genai.RealtimeUpdate{Instructions: &instr, Tools: &tools}); err != nil {
		t.Fatal(err)
	}
	if err := s.Send(t.Context(), genai.RealtimeInputContent{Text: "hello"}); err != nil {
		t.Fatal(err)
	}
	if err := s.Send(t.Context(), genai.RealtimeUpdate{Tools: &tools}); err == nil {
		t.Fatal("late tools update accepted")
	}
	if err := s.Send(t.Context(), genai.RealtimeUpdate{Instructions: &instr}); err == nil {
		t.Fatal("late instructions update accepted")
	}
	manual := genai.RealtimeTurnManual
	voice := "marin"
	for _, u := range []genai.RealtimeUpdate{{TurnDetection: &manual}, {Voice: &voice}} {
		if err := s.Send(t.Context(), u); err == nil {
			t.Fatalf("unsupported update accepted: %+v", u)
		}
	}
	none := genai.ToolCallNone
	tokens := int64(128)
	mods := genai.Modalities{genai.ModalityText}
	if err := s.Send(t.Context(), genai.RealtimeUpdate{ToolChoice: &none, MaxOutputTokens: &tokens, OutputModalities: &mods}); err != nil {
		t.Fatal(err)
	}
	select {
	case <-sent:
	case <-time.After(5 * time.Second):
		t.Fatal("timeout waiting for session updates")
	}
}

func TestRealtimeSendRaw(t *testing.T) {
	sent := make(chan struct{})
	s := newRealtimeTestSession(t, nil, func(ctx context.Context, ws *websocket.Conn) {
		m := expectRealtimeFrame(t, ctx, ws, "conversation.item.delete")
		if m["item_id"] != "item_1" {
			t.Errorf("raw message %#v", m)
		}
		close(sent)
	})
	if err := s.SendRaw(t.Context(), nil); err == nil {
		t.Fatal("empty raw message accepted")
	}
	if err := s.SendRaw(t.Context(), []byte(`{"type":"conversation.item.delete","item_id":"item_1"}`)); err != nil {
		t.Fatal(err)
	}
	select {
	case <-sent:
	case <-time.After(5 * time.Second):
		t.Fatal("timeout waiting for raw message")
	}
}
