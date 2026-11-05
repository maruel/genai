// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Tests for the OpenAI Chat provider client.

package openaichat_test

import (
	"bytes"
	"context"
	_ "embed"
	"encoding/json"
	"errors"
	"fmt"
	"iter"
	"net/http"
	"path/filepath"
	"reflect"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/coder/websocket"

	"gopkg.in/dnaeon/go-vcr.v4/pkg/recorder"

	"github.com/maruel/genai"
	"github.com/maruel/genai/base"
	"github.com/maruel/genai/internal"
	"github.com/maruel/genai/internal/internaltest"
	"github.com/maruel/genai/internal/realtimetest"
	"github.com/maruel/genai/providers/openaibase"
	"github.com/maruel/genai/providers/openaichat"
	"github.com/maruel/genai/scoreboard"
	"github.com/maruel/genai/smoke"
	"github.com/maruel/genai/smoke/smoketest"
)

func getClientInner(t *testing.T, fn func(http.RoundTripper) http.RoundTripper, opts ...genai.ProviderOption) (genai.Provider, error) {
	// Check if API key was provided in opts; if not and env var is empty, add a dummy key
	hasAPIKey := false
	for _, opt := range opts {
		if _, ok := opt.(genai.ProviderOptionAPIKey); ok {
			hasAPIKey = true
			break
		}
	}
	if !hasAPIKey {
		key := internaltest.GetEnv("OPENAI_API_KEY")
		if key == "" {
			key = "<insert_api_key_here>"
		}
		opts = append(opts, genai.ProviderOptionAPIKey(key))
	}
	if fn != nil {
		opts = append([]genai.ProviderOption{genai.ProviderOptionTransportWrapper(fn)}, opts...)
	}
	c, err := openaichat.New(t.Context(), opts...)
	if c != nil {
		internaltest.CleanupCloser(t, c)
	}
	return c, err
}

func TestClient(t *testing.T) {
	t.Run("ConnectRealtime", func(t *testing.T) {
		t.Run("unsupported option type", func(t *testing.T) {
			s, err := realtimeClient(t).ConnectRealtime(t.Context(), genai.GenOptionSeed(1))
			if _, ok := errors.AsType[*base.ErrNotSupported](err); s != nil || !ok {
				t.Fatalf("unsupported option: %+v, %v", s, err)
			}
		})
		t.Run("valid", func(t *testing.T) {
			t.Run("rate limits", func(t *testing.T) {
				endpoint := realtimePeer(t, func(ctx context.Context, ws *websocket.Conn) {
					realtimeSetup(t, ctx, ws)
					realtimetest.Write(t, ctx, ws, `{"type":"rate_limits.updated","rate_limits":[{"name":"tokens","limit":200000,"remaining":199388,"reset_seconds":0.183}, {"name":"requests","limit":100,"remaining":99,"reset_seconds":1.5}]}`)
				})
				s, err := realtimeClient(t).ConnectRealtime(t.Context(), &openaichat.RealtimeOption{Endpoint: endpoint})
				if err != nil {
					t.Fatal(err)
				}
				realtimetest.Close(t, s)
				e := realtimetest.Receive(t, s, genai.RealtimeEventUsage)
				if len(e.Usage.Limits) != 2 {
					t.Fatalf("limits=%+v", e.Usage.Limits)
				}
				if l := e.Usage.Limits[0]; l.Type != genai.Tokens || l.Limit != 200000 || l.Remaining != 199388 || l.Period != genai.PerOther || l.Reset.IsZero() {
					t.Fatalf("tokens limit=%+v", l)
				}
				if l := e.Usage.Limits[1]; l.Type != genai.Requests || l.Limit != 100 || l.Remaining != 99 || l.Period != genai.PerOther || l.Reset.IsZero() {
					t.Fatalf("requests limit=%+v", l)
				}
			})

			t.Run("server VAD rejects audio commit", func(t *testing.T) {
				endpoint := realtimePeer(t, func(ctx context.Context, ws *websocket.Conn) {
					realtimeSetup(t, ctx, ws)
				})
				s, err := realtimeClient(t).ConnectRealtime(t.Context(), &openaichat.RealtimeOption{Endpoint: endpoint})
				if err != nil {
					t.Fatal(err)
				}
				realtimetest.Close(t, s)
				if s.Info().Capabilities.AudioCommit {
					t.Fatal("server VAD advertised audio commit")
				}
				if err := s.Send(t.Context(), genai.RealtimeInputContent{Commit: true}); err == nil {
					t.Fatal("server VAD accepted audio commit")
				} else if _, ok := errors.AsType[*base.ErrNotSupported](err); !ok {
					t.Fatalf("audio commit error = %v", err)
				}
			})

			t.Run("manual audio tools and transcripts", func(t *testing.T) {
				commands := make(chan []map[string]any, 1)
				endpoint := realtimePeer(t, func(ctx context.Context, ws *websocket.Conn) {
					setup := realtimeSetup(t, ctx, ws)
					audio := setup["audio"].(map[string]any)
					input := audio["input"].(map[string]any)
					if input["turn_detection"] != nil || !reflect.DeepEqual(setup["output_modalities"], []any{"audio"}) {
						t.Error("incorrect manual audio setup")
					}
					if transcription := input["transcription"].(map[string]any); transcription["language"] != "fr" {
						t.Errorf("transcription language %#v", transcription)
					}
					got := make([]map[string]any, 0, 8)
					for range 5 {
						got = append(got, realtimetest.Read(t, ctx, ws))
					}
					realtimetest.Write(t, ctx, ws, `{"type":"response.created","response":{"id":"resp_tools"}}`)
					realtimetest.Write(t, ctx, ws, `{"type":"response.function_call_arguments.delta","response_id":"resp_tools","item_id":"item_a","output_index":0,"delta":"{\"q\":"}`)
					realtimetest.Write(t, ctx, ws, `{"type":"response.function_call_arguments.done","response_id":"resp_tools","item_id":"item_a","output_index":0,"call_id":"call_a","name":"lookup","arguments":"{}"}`)
					realtimetest.Write(t, ctx, ws, `{"type":"response.done","response":{"id":"resp_tools","status":"completed","output":[{"type":"function_call","call_id":"call_a","name":"lookup","arguments":"{}"},{"type":"function_call","call_id":"call_b","name":"lookup","arguments":"{}"}],"usage":{"input_tokens":5,"output_tokens":3,"total_tokens":8,"input_token_details":{"cached_tokens":2}}}}`)
					for range 3 {
						got = append(got, realtimetest.Read(t, ctx, ws))
					}
					commands <- got
					for _, b := range []string{
						`{"type":"response.created","response":{"id":"resp_audio"}}`,
						`{"type":"input_audio_buffer.speech_started","item_id":"user_audio"}`,
						`{"type":"input_audio_buffer.speech_stopped","item_id":"user_audio"}`,
						`{"type":"conversation.item.input_audio_transcription.delta","item_id":"user_audio","delta":"hi"}`,
						`{"type":"conversation.item.input_audio_transcription.completed","item_id":"user_audio","transcript":"Hi."}`,
						`{"type":"response.output_audio.delta","response_id":"resp_audio","item_id":"assistant_audio","delta":"AQACAA=="}`,
						`{"type":"response.output_audio_transcript.delta","response_id":"resp_audio","item_id":"assistant_audio","delta":"four"}`,
						`{"type":"response.output_audio_transcript.done","response_id":"resp_audio","item_id":"assistant_audio","transcript":"Four."}`,
						`{"type":"response.done","response":{"id":"resp_audio","status":"completed","output":[]}}`,
					} {
						realtimetest.Write(t, ctx, ws, b)
					}
				})
				s, err := realtimeClient(t).ConnectRealtime(t.Context(), genai.RealtimeTurnManual, &genai.RealtimeOptionTools{Tools: []genai.ToolDef{{Name: "lookup", Description: "Lookup a value", InputSchemaOverride: json.RawMessage(`{"type":"object"}`)}}}, &genai.RealtimeOptionAudio{InputTranscription: true, OutputTranscription: true}, &openaichat.RealtimeOption{Endpoint: endpoint, InputTranscriptionModel: "gpt-realtime-whisper", InputLanguage: "fr-CA"})
				if err != nil {
					t.Fatal(err)
				}
				realtimetest.Close(t, s)
				info := s.Info()
				if info.ID != "sess_test" || info.InputFormat != (genai.AudioFormat{SampleRate: 24000}) || info.OutputFormat != info.InputFormat || !info.Capabilities.Truncate || info.TurnDetection != genai.RealtimeTurnManual || !info.Capabilities.AudioCommit || !info.Capabilities.ToolResults {
					t.Fatalf("info %+v", info)
				}
				info.ID = "changed"
				if s.Info().ID != "sess_test" {
					t.Fatal("Info is mutable")
				}
				for _, in := range []genai.RealtimeInput{
					genai.RealtimeInputContent{Text: "passive"},
					genai.RealtimeInputContent{Audio: []byte{1, 0, 2, 0}},
					genai.RealtimeInputContent{Commit: true, Generate: true},
					genai.RealtimeTruncate{ItemID: "old_audio", AudioEndMS: 45},
				} {
					if err := s.Send(t.Context(), in); err != nil {
						t.Fatal(err)
					}
				}
				p := realtimetest.Receive(t, s, genai.RealtimeEventProvider)
				if p.ProviderEvent.RealtimeEventName() != "response.function_call_arguments.delta" {
					t.Fatalf("provider event %+v", p)
				}
				if d, ok := p.ProviderEvent.(*openaibase.FunctionCallArgsDelta); !ok || d.ItemID != "item_a" || d.Delta != `{"q":` {
					t.Fatalf("delta %+v", p)
				}
				p = realtimetest.Receive(t, s, genai.RealtimeEventProvider)
				if d, ok := p.ProviderEvent.(*openaibase.FunctionCallArgsDone); !ok || d.CallID != "call_a" || d.Name != "lookup" || d.Arguments != "{}" {
					t.Fatalf("done %+v", p)
				}
				calls := realtimetest.Receive(t, s, genai.RealtimeEventContentToolCalls)
				if len(calls.Content.ToolCalls) != 2 || calls.ResponseID != "resp_tools" {
					t.Fatalf("calls %+v", calls)
				}
				u := realtimetest.Receive(t, s, genai.RealtimeEventUsage)
				if u.Usage.InputCachedTokens != 2 || u.Usage.TotalTokens != 8 {
					t.Fatalf("usage %+v", u)
				}
				realtimetest.Receive(t, s, genai.RealtimeEventGenerationDone)
				for _, results := range [][]genai.ToolCallResult{
					{{ID: "call_a", Name: "lookup", Result: "one"}},
					{{ID: "call_a", Name: "wrong", Result: "one"}, {ID: "call_b", Name: "lookup", Result: "two"}},
					{{ID: "unknown", Name: "lookup", Result: "one"}},
				} {
					if err := s.Send(t.Context(), genai.RealtimeInputContent{ToolResults: results, Generate: true}); err == nil {
						t.Fatal("accepted invalid tool batch")
					}
				}
				results := []genai.ToolCallResult{{ID: "call_a", Name: "lookup", Result: "one"}, {ID: "call_b", Name: "lookup", Result: "two"}}
				if err := s.Send(t.Context(), genai.RealtimeInputContent{ToolResults: results, Generate: true}); err != nil {
					t.Fatal(err)
				}
				if err := s.Send(t.Context(), genai.RealtimeInputContent{ToolResults: results}); err == nil {
					t.Fatal("replayed tool results")
				}
				got := <-commands
				want := []map[string]any{
					{"type": "conversation.item.create", "item": map[string]any{"type": "message", "role": "user", "content": []any{map[string]any{"type": "input_text", "text": "passive"}}}},
					{"type": "input_audio_buffer.append", "audio": "AQACAA=="},
					{"type": "input_audio_buffer.commit"},
					{"type": "response.create"},
					{"type": "conversation.item.truncate", "item_id": "old_audio", "content_index": float64(0), "audio_end_ms": float64(45)},
					{"type": "conversation.item.create", "item": map[string]any{"type": "function_call_output", "call_id": "call_a", "output": "one"}},
					{"type": "conversation.item.create", "item": map[string]any{"type": "function_call_output", "call_id": "call_b", "output": "two"}},
					{"type": "response.create"},
				}
				if !reflect.DeepEqual(got, want) {
					t.Fatalf("commands %#v, want %#v", got, want)
				}
				d := realtimetest.Receive(t, s, genai.RealtimeEventTranscriptInput)
				f := realtimetest.Receive(t, s, genai.RealtimeEventTranscriptInput)
				if d.Content.Text != "hi" || d.Content.Final || f.Content.Text != "Hi." || !f.Content.Final {
					t.Fatal("input transcript delta/final lost")
				}
				a := realtimetest.Receive(t, s, genai.RealtimeEventContentAudio)
				if !bytes.Equal(a.Content.Audio, []byte{1, 0, 2, 0}) {
					t.Fatal("PCM changed")
				}
				d = realtimetest.Receive(t, s, genai.RealtimeEventTranscriptOutput)
				f = realtimetest.Receive(t, s, genai.RealtimeEventTranscriptOutput)
				if d.Content.Text != "four" || d.Content.Final || f.Content.Text != "Four." || !f.Content.Final {
					t.Fatal("output transcript delta/final lost")
				}
				realtimetest.Receive(t, s, genai.RealtimeEventGenerationDone)
				realtimetest.Receive(t, s, genai.RealtimeEventTurnDone)
			})
			t.Run("server turns text cancel and recoverable errors", func(t *testing.T) {
				endpoint := realtimePeer(t, func(ctx context.Context, ws *websocket.Conn) {
					setup := realtimeSetup(t, ctx, ws)
					if !reflect.DeepEqual(setup["output_modalities"], []any{"text"}) {
						t.Error("text mode changed")
					}
					realtimetest.Write(t, ctx, ws, `{"type":"input_audio_buffer.speech_started","item_id":"user_speech"}`)
					realtimetest.Write(t, ctx, ws, `{"type":"input_audio_buffer.speech_stopped","item_id":"user_speech"}`)
					item := realtimetest.Read(t, ctx, ws)
					create := realtimetest.Read(t, ctx, ws)
					if item["type"] != "conversation.item.create" || create["type"] != "response.create" {
						t.Error("missing text generation")
					}
					realtimetest.Write(t, ctx, ws, `{"type":"response.created","response":{"id":"resp_one"}}`)
					realtimetest.Write(t, ctx, ws, `{"type":"response.output_text.delta","response_id":"resp_one","item_id":"text_one","delta":"start"}`)
					cancel := realtimetest.Read(t, ctx, ws)
					if !reflect.DeepEqual(cancel, map[string]any{"type": "response.cancel", "response_id": "resp_one"}) {
						t.Errorf("cancel %#v", cancel)
					}
					for _, b := range []string{
						`{"type":"response.output_text.delta","response_id":"resp_one","item_id":"text_one","delta":"stale"}`,
						`{"type":"response.output_audio.delta","response_id":"resp_one","item_id":"audio_one","delta":"AQACAA=="}`,
						`{"type":"response.done","response":{"id":"resp_one","status":"cancelled","output":[{"type":"function_call","call_id":"cancelled_call","name":"lookup","arguments":"{}"}]}}`,
						`{"type":"error","error":{"code":"invalid_request","event_id":"event_bad","message":"secret"}}`,
						`{"type":"response.created","response":{"id":"resp_two"}}`,
						`{"type":"response.output_text.done","response_id":"resp_two","item_id":"text_two","text":"done"}`,
						`{"type":"response.done","response":{"id":"resp_two","status":"completed","output":[]}}`,
					} {
						realtimetest.Write(t, ctx, ws, b)
					}
				})
				s, err := realtimeClient(t).ConnectRealtime(t.Context(), genai.Modalities{genai.ModalityText}, &openaichat.RealtimeOption{Endpoint: endpoint})
				if err != nil {
					t.Fatal(err)
				}
				realtimetest.Close(t, s)
				if s.Info().TurnDetection != genai.RealtimeTurnServer || s.Info().Capabilities.Truncate {
					t.Fatalf("capabilities %+v", s.Info())
				}
				if err := s.Send(t.Context(), genai.RealtimeInputContent{Text: "hello", Generate: true}); err != nil {
					t.Fatal(err)
				}
				if e := realtimetest.Receive(t, s, genai.RealtimeEventInputSpeechStart); e.ItemID != "user_speech" {
					t.Fatalf("speech start %+v", e)
				}
				if e := realtimetest.Receive(t, s, genai.RealtimeEventInputSpeechEnd); e.ItemID != "user_speech" {
					t.Fatalf("speech end %+v", e)
				}
				realtimetest.Receive(t, s, genai.RealtimeEventContentText)
				if err := s.Send(t.Context(), genai.RealtimeCancel{ResponseID: "unknown"}); err == nil {
					t.Fatal("unknown target accepted")
				}
				if err := s.Send(t.Context(), genai.RealtimeCancel{ResponseID: "resp_one"}); err != nil {
					t.Fatal(err)
				}
				realtimetest.Receive(t, s, genai.RealtimeEventInterrupted)
				realtimetest.Receive(t, s, genai.RealtimeEventGenerationDone)
				realtimetest.Receive(t, s, genai.RealtimeEventTurnDone)
				e := realtimetest.Receive(t, s, genai.RealtimeEventError)
				if e.Error.Code != "invalid_request" || e.Error.EventID != "event_bad" || e.Error.Message == "secret" {
					t.Fatalf("error %+v", e)
				}
				e = realtimetest.Receive(t, s, genai.RealtimeEventContentText)
				if e.Content.Text != "done" || !e.Content.Final {
					t.Fatalf("text %+v", e)
				}
				realtimetest.Receive(t, s, genai.RealtimeEventGenerationDone)
				realtimetest.Receive(t, s, genai.RealtimeEventTurnDone)
			})
			t.Run("reasoning effort", func(t *testing.T) {
				endpoint := realtimePeer(t, func(ctx context.Context, ws *websocket.Conn) {
					setup := realtimeSetup(t, ctx, ws)
					r, ok := setup["reasoning"].(map[string]any)
					if !ok || r["effort"] != "high" || len(r) != 1 {
						t.Errorf("reasoning: %v", setup["reasoning"])
					}
				})
				s, err := realtimeClient(t).ConnectRealtime(t.Context(), &openaichat.RealtimeOption{Endpoint: endpoint}, genai.EffortHigh)
				if err != nil {
					t.Fatal(err)
				}
				realtimetest.Close(t, s)
			})
			t.Run("session update", func(t *testing.T) {
				sent := make(chan struct{})
				endpoint := realtimePeer(t, func(ctx context.Context, ws *websocket.Conn) {
					realtimeSetup(t, ctx, ws)
					m := realtimetest.Read(t, ctx, ws)
					upd, _ := m["session"].(map[string]any)
					if m["type"] != "session.update" || upd["instructions"] != "be terse" {
						t.Errorf("instructions %#v", m)
					}
					tools, _ := upd["tools"].([]any)
					if len(tools) != 1 || tools[0].(map[string]any)["name"] != "lookup" {
						t.Errorf("tools %#v", m)
					}
					if upd["tool_choice"] != "none" {
						t.Errorf("tool choice %#v", m)
					}
					if upd["max_output_tokens"] != float64(128) {
						t.Errorf("max output tokens %#v", m)
					}
					mods, _ := upd["output_modalities"].([]any)
					if len(mods) != 1 || mods[0] != "text" {
						t.Errorf("output modalities %#v", m)
					}
					audio, _ := upd["audio"].(map[string]any)
					in, _ := audio["input"].(map[string]any)
					if v, ok := in["turn_detection"]; !ok || v != nil {
						t.Errorf("turn detection %#v", m)
					}
					out, _ := audio["output"].(map[string]any)
					if out["voice"] != "marin" {
						t.Errorf("voice %#v", m)
					}
					close(sent)
				})
				s, err := realtimeClient(t).ConnectRealtime(t.Context(), &openaichat.RealtimeOption{Endpoint: endpoint})
				if err != nil {
					t.Fatal(err)
				}
				t.Cleanup(func() { _ = s.Close() })
				if s.Info().Capabilities.AudioCommit {
					t.Fatal("server VAD advertised audio commit")
				}
				instr := "be terse"
				tools := []genai.ToolDef{{Name: "lookup", Description: "Lookup", InputSchemaOverride: json.RawMessage(`{"type":"object"}`)}}
				none := genai.ToolCallNone
				tokens := int64(128)
				mods := genai.Modalities{genai.ModalityText}
				manual := genai.RealtimeTurnManual
				voice := "marin"
				in := genai.RealtimeUpdate{Instructions: &instr, Tools: &tools, ToolChoice: &none, MaxOutputTokens: &tokens, OutputModalities: &mods, TurnDetection: &manual, Voice: &voice}
				if err := s.Send(t.Context(), in); err != nil {
					t.Fatal(err)
				}
				select {
				case <-sent:
				case <-time.After(5 * time.Second):
					t.Fatal("timeout waiting for session update")
				}
				if !s.Info().Capabilities.AudioCommit {
					t.Fatal("manual turn detection did not advertise audio commit")
				}
			})

			t.Run("audio generation commits", func(t *testing.T) {
				sent := make(chan struct{})
				endpoint := realtimePeer(t, func(ctx context.Context, ws *websocket.Conn) {
					realtimeSetup(t, ctx, ws)
					m := realtimetest.Read(t, ctx, ws)
					if m["type"] != "input_audio_buffer.append" {
						t.Errorf("append %#v", m)
					}
					m = realtimetest.Read(t, ctx, ws)
					if m["type"] != "input_audio_buffer.commit" {
						t.Errorf("commit %#v", m)
					}
					m = realtimetest.Read(t, ctx, ws)
					if !reflect.DeepEqual(m, map[string]any{"type": "response.create"}) {
						t.Errorf("generation %#v", m)
					}
					close(sent)
					_, _, _ = ws.Read(ctx)
				})
				s, err := realtimeClient(t).ConnectRealtime(t.Context(), genai.RealtimeTurnManual, &openaichat.RealtimeOption{Endpoint: endpoint})
				if err != nil {
					t.Fatal(err)
				}
				t.Cleanup(func() { _ = s.Close() })
				if err := s.Send(t.Context(), genai.RealtimeInputContent{Audio: []byte{1, 0}, Generate: true}); err != nil {
					t.Fatal(err)
				}
				select {
				case <-sent:
				case <-time.After(5 * time.Second):
					t.Fatal("timeout waiting for audio generation")
				}
			})

			t.Run("empty text generation", func(t *testing.T) {
				sent := make(chan struct{})
				endpoint := realtimePeer(t, func(ctx context.Context, ws *websocket.Conn) {
					realtimeSetup(t, ctx, ws)
					if m := realtimetest.Read(t, ctx, ws); !reflect.DeepEqual(m, map[string]any{"type": "response.create"}) {
						t.Errorf("generation %#v", m)
					}
					close(sent)
					_, _, _ = ws.Read(ctx)
				})
				s, err := realtimeClient(t).ConnectRealtime(t.Context(), &openaichat.RealtimeOption{Endpoint: endpoint})
				if err != nil {
					t.Fatal(err)
				}
				t.Cleanup(func() { _ = s.Close() })
				if err := s.Send(t.Context(), genai.RealtimeInputContent{Generate: true}); err != nil {
					t.Fatal(err)
				}
				select {
				case <-sent:
				case <-time.After(5 * time.Second):
					t.Fatal("timeout waiting for generation")
				}
			})

			t.Run("SendRaw", func(t *testing.T) {
				sent := make(chan struct{})
				endpoint := realtimePeer(t, func(ctx context.Context, ws *websocket.Conn) {
					realtimeSetup(t, ctx, ws)
					m := realtimetest.Read(t, ctx, ws)
					if m["type"] != "conversation.item.delete" || m["item_id"] != "item_1" {
						t.Errorf("raw message %#v", m)
					}
					close(sent)
				})
				s, err := realtimeClient(t).ConnectRealtime(t.Context(), &openaichat.RealtimeOption{Endpoint: endpoint})
				if err != nil {
					t.Fatal(err)
				}
				rs, ok := s.(*openaichat.RealtimeSession)
				if !ok {
					t.Fatalf("session type %T", s)
				}
				t.Cleanup(func() { _ = s.Close() })
				if err := rs.SendRaw(t.Context(), []byte(`{"type":"conversation.item.delete","item_id":"item_1"}`)); err != nil {
					t.Fatal(err)
				}
				select {
				case <-sent:
				case <-time.After(5 * time.Second):
					t.Fatal("timeout waiting for raw message")
				}
			})
		})
		t.Run("error", func(t *testing.T) {
			t.Run("invalid setup before network", func(t *testing.T) {
				c := realtimeClient(t)
				for _, cfg := range [][]genai.RealtimeOption{
					{nil}, {genai.Modalities{genai.ModalityAudio, genai.ModalityText}}, {genai.Modalities{genai.ModalityText}, &genai.RealtimeOptionAudio{Voice: "marin"}}, {&genai.RealtimeOptionAudio{InputTranscription: true}},
				} {
					if _, err := c.ConnectRealtime(t.Context(), append([]genai.RealtimeOption{&openaichat.RealtimeOption{Endpoint: "ws://127.0.0.1:1"}}, cfg...)...); err == nil {
						t.Fatal("accepted invalid setup")
					}
				}
				for _, o := range []*openaichat.RealtimeOption{{Endpoint: "https://example.com"}, {Endpoint: "wss://example.com?key=secret"}, {PingInterval: -time.Second}, {MaxOutputTokens: -1}, {InputLanguage: "en"}, {InputLanguage: "english"}, {InputLanguage: "fr_CA"}, {InputLanguage: "fr--CA"}} {
					if _, err := c.ConnectRealtime(t.Context(), o); err == nil {
						t.Fatal("accepted invalid option")
					}
				}
				var nilOption *openaichat.RealtimeOption
				if _, err := c.ConnectRealtime(t.Context(), nilOption); err == nil || !strings.Contains(err.Error(), "must not be nil") {
					t.Fatal("accepted nil option")
				}
				if _, err := c.ConnectRealtime(t.Context(), &openaichat.RealtimeOption{}, &openaichat.RealtimeOption{}); err == nil || !strings.Contains(err.Error(), "duplicate realtime option type") {
					t.Fatal("accepted duplicate options")
				}
			})
			t.Run("failed incomplete and malformed responses", func(t *testing.T) {
				for _, status := range []string{"failed", "incomplete"} {
					t.Run(status, func(t *testing.T) {
						endpoint := realtimePeer(t, func(ctx context.Context, ws *websocket.Conn) {
							realtimeSetup(t, ctx, ws)
							realtimetest.Write(t, ctx, ws, `{"type":"response.done","response":{"id":"r","status":"`+status+`","output":[{"type":"function_call","name":"lookup","call_id":"never","arguments":"{}"}]}}`)
							realtimetest.Write(t, ctx, ws, `{"type":"response.output_audio.delta","response_id":"r","item_id":"a","delta":"bad"}`)
						})
						s, err := realtimeClient(t).ConnectRealtime(t.Context(), &openaichat.RealtimeOption{Endpoint: endpoint})
						if err != nil {
							t.Fatal(err)
						}
						realtimetest.Close(t, s)
						e := realtimetest.Receive(t, s, genai.RealtimeEventError)
						if e.Error.Code != status {
							t.Fatal(e)
						}
						realtimetest.Receive(t, s, genai.RealtimeEventGenerationDone)
						realtimetest.Receive(t, s, genai.RealtimeEventTurnDone)
						next, stop := iter.Pull2(s.Receive(t.Context()))
						_, err, ok := next()
						stop()
						if !ok {
							t.Fatal("realtime session ended")
						}
						if err == nil {
							t.Fatal("accepted malformed audio")
						}
						if err := s.Send(t.Context(), genai.RealtimeInputContent{Text: "after", Generate: true}); err == nil {
							t.Fatal("send after terminal failure")
						}
					})
				}
			})
			t.Run("unsupported commands", func(t *testing.T) {
				done := make(chan struct{})
				endpoint := realtimePeer(t, func(ctx context.Context, ws *websocket.Conn) {
					realtimeSetup(t, ctx, ws)
					if m := realtimetest.Read(t, ctx, ws); m["type"] != "conversation.item.create" {
						t.Errorf("invalid command leaked %#v", m)
					}
					if m := realtimetest.Read(t, ctx, ws); !reflect.DeepEqual(m, map[string]any{"type": "response.create"}) {
						t.Errorf("invalid command leaked %#v", m)
					}
					close(done)
				})
				s, err := realtimeClient(t).ConnectRealtime(t.Context(), &openaichat.RealtimeOption{Endpoint: endpoint})
				if err != nil {
					t.Fatal(err)
				}
				realtimetest.Close(t, s)
				for _, in := range []genai.RealtimeInput{
					genai.RealtimeInputContent{Audio: []byte{1}},
					genai.RealtimeTruncate{ItemID: "a", ContentIndex: 1},
				} {
					if err := s.Send(t.Context(), in); err == nil {
						t.Fatal("accepted invalid command")
					}
				}
				if err := s.Send(t.Context(), genai.RealtimeInputContent{Text: "after", Generate: true}); err != nil {
					t.Fatal(err)
				}
				<-done
			})
			t.Run("setup acknowledgement and transport termination", func(t *testing.T) {
				for _, data := range []string{`{"type":"session.updated"}`, `{"type":"error","error":{"code":"bad"}}`, `{"type":"session.created","session":{"type":"transcription","id":"bad"}}`} {
					t.Run(data, func(t *testing.T) {
						endpoint := realtimePeer(t, func(ctx context.Context, ws *websocket.Conn) { realtimetest.Write(t, ctx, ws, data) })
						if _, err := realtimeClient(t).ConnectRealtime(t.Context(), &openaichat.RealtimeOption{Endpoint: endpoint}); err == nil {
							t.Fatal("accepted invalid acknowledgement")
						}
					})
				}
				endpoint := realtimePeer(t, func(ctx context.Context, ws *websocket.Conn) { realtimeSetup(t, ctx, ws) })
				ctx, cancel := context.WithCancel(t.Context())
				t.Cleanup(cancel)
				s, err := realtimeClient(t).ConnectRealtime(ctx, &openaichat.RealtimeOption{Endpoint: endpoint})
				if err != nil {
					t.Fatal(err)
				}
				done := make(chan error, 1)
				go func() {
					next, stop := iter.Pull2(s.Receive(t.Context()))
					_, err, _ := next()
					stop()
					done <- err
				}()
				cancel()
				select {
				case err := <-done:
					if !errors.Is(err, context.Canceled) {
						t.Fatal(err)
					}
				case <-time.After(time.Second):
					t.Fatal("receive did not wake")
				}
				if err := s.Close(); err != nil {
					t.Fatal(err)
				}
			})
			t.Run("voice after audio", func(t *testing.T) {
				endpoint := realtimePeer(t, func(ctx context.Context, ws *websocket.Conn) {
					realtimeSetup(t, ctx, ws)
					realtimetest.Write(t, ctx, ws, `{"type":"response.output_audio.delta","response_id":"resp_1","item_id":"item_1","delta":"AAA="}`)
				})
				s, err := realtimeClient(t).ConnectRealtime(t.Context(), &openaichat.RealtimeOption{Endpoint: endpoint})
				if err != nil {
					t.Fatal(err)
				}
				t.Cleanup(func() { _ = s.Close() })
				realtimetest.Receive(t, s, genai.RealtimeEventContentAudio)
				voice := "marin"
				if err := s.Send(t.Context(), genai.RealtimeUpdate{Voice: &voice}); err == nil {
					t.Fatal("voice update accepted after audio output")
				}
			})

			t.Run("unsupported input", func(t *testing.T) {
				sent := make(chan struct{})
				endpoint := realtimePeer(t, func(ctx context.Context, ws *websocket.Conn) {
					realtimeSetup(t, ctx, ws)
					m := realtimetest.Read(t, ctx, ws)
					if m["type"] != "conversation.item.create" {
						t.Errorf("sentinel frame %#v", m)
					}
					close(sent)
				})
				s, err := realtimeClient(t).ConnectRealtime(t.Context(), genai.Modalities{genai.ModalityText}, &openaichat.RealtimeOption{Endpoint: endpoint})
				if err != nil {
					t.Fatal(err)
				}
				t.Cleanup(func() { _ = s.Close() })
				for _, in := range []genai.RealtimeInput{
					genai.RealtimeTruncate{ItemID: "item_1"},
					genai.RealtimeInputContent{ToolResults: []genai.ToolCallResult{{ID: "call_1", Name: "clock", Result: "ok"}}},
				} {
					if err := s.Send(t.Context(), in); err == nil {
						t.Fatalf("unsupported input accepted: %T", in)
					}
				}
				err = s.Send(t.Context(), genai.RealtimeInputContent{Doc: genai.Doc{URL: "https://example.com/image.jpg"}, Text: "what is it", Generate: true})
				if _, ok := errors.AsType[*base.ErrNotSupported](err); !ok {
					t.Fatalf("remote document: %v", err)
				}
				if err := s.Send(t.Context(), genai.RealtimeInputContent{Text: "sentinel"}); err != nil {
					t.Fatal(err)
				}
				select {
				case <-sent:
				case <-time.After(5 * time.Second):
					t.Fatal("timeout waiting for sentinel")
				}
			})

			t.Run("empty SendRaw", func(t *testing.T) {
				endpoint := realtimePeer(t, func(ctx context.Context, ws *websocket.Conn) {
					realtimeSetup(t, ctx, ws)
				})
				s, err := realtimeClient(t).ConnectRealtime(t.Context(), &openaichat.RealtimeOption{Endpoint: endpoint})
				if err != nil {
					t.Fatal(err)
				}
				rs, ok := s.(*openaichat.RealtimeSession)
				if !ok {
					t.Fatalf("session type %T", s)
				}
				t.Cleanup(func() { _ = s.Close() })
				if err := rs.SendRaw(t.Context(), nil); err == nil {
					t.Fatal("empty raw message accepted")
				}
			})
		})
		t.Run("protocol failures", func(t *testing.T) {
			for name, payload := range map[string]string{
				"malformed JSON":                           `{`,
				"missing response":                         `{"type":"response.done"}`,
				"unknown response status":                  `{"type":"response.done","response":{"id":"r","status":"mystery"}}`,
				"nonterminal completion":                   `{"type":"response.done","response":{"id":"r","status":"in_progress"}}`,
				"invalid tool JSON":                        `{"type":"response.done","response":{"id":"r","status":"completed","output":[{"type":"function_call","name":"lookup","call_id":"c","arguments":"bad"}]}}`,
				"unknown tool":                             `{"type":"response.done","response":{"id":"r","status":"completed","output":[{"type":"function_call","name":"unknown","call_id":"c","arguments":"{}"}]}}`,
				"duplicate tool ID":                        `{"type":"response.done","response":{"id":"r","status":"completed","output":[{"type":"function_call","name":"lookup","call_id":"c","arguments":"{}"},{"type":"function_call","name":"lookup","call_id":"c","arguments":"{}"}]}}`,
				"negative usage":                           `{"type":"response.done","response":{"id":"r","status":"completed","usage":{"input_tokens":-1}}}`,
				"missing error":                            `{"type":"error"}`,
				"missing output text delta response":       `{"type":"response.output_text.delta","item_id":"a","delta":"stale"}`,
				"missing output text final response":       `{"type":"response.output_text.done","item_id":"a","text":"stale"}`,
				"missing output transcript delta response": `{"type":"response.output_audio_transcript.delta","item_id":"a","delta":"stale"}`,
				"missing output transcript final response": `{"type":"response.output_audio_transcript.done","item_id":"a","transcript":"stale"}`,
				"missing transcript identity":              `{"type":"response.output_text.done","response_id":"r","text":"hi"}`,
				"overlapping responses":                    `{"type":"response.created","response":{"id":"other"}}`,
			} {
				t.Run(name, func(t *testing.T) {
					endpoint := realtimePeer(t, func(ctx context.Context, ws *websocket.Conn) {
						realtimeSetup(t, ctx, ws)
						realtimetest.Write(t, ctx, ws, `{"type":"response.created","response":{"id":"r"}}`)
						realtimetest.Write(t, ctx, ws, payload)
					})
					s, err := realtimeClient(t).ConnectRealtime(t.Context(), &genai.RealtimeOptionTools{Tools: []genai.ToolDef{{Name: "lookup", Description: "Lookup", InputSchemaOverride: json.RawMessage(`{"type":"object"}`)}}}, &openaichat.RealtimeOption{Endpoint: endpoint})
					if err != nil {
						t.Fatal(err)
					}
					realtimetest.Close(t, s)
					next, stop := iter.Pull2(s.Receive(t.Context()))
					_, err, ok := next()
					stop()
					if !ok {
						t.Fatal("realtime session ended")
					}
					if err == nil {
						t.Fatal("malformed recognized event was accepted")
					}
				})
			}
		})
		t.Run("rejected effective configuration", func(t *testing.T) {
			for name, mutate := range map[string]func(map[string]any){
				"PCM rate": func(v map[string]any) {
					v["audio"].(map[string]any)["input"].(map[string]any)["format"].(map[string]any)["rate"] = 16000
				},
				"output modality":      func(v map[string]any) { v["output_modalities"] = []string{"text"} },
				"oversized modalities": func(v map[string]any) { v["output_modalities"] = []string{"audio", "audio"} },
				"oversized declarations": func(v map[string]any) {
					tools := make([]any, 1025)
					for i := range tools {
						tools[i] = map[string]any{}
					}
					v["tools"] = tools
				},
				"unrequested transcription": func(v map[string]any) {
					v["audio"].(map[string]any)["input"].(map[string]any)["transcription"] = map[string]any{"model": "unexpected"}
				},
				"server VAD":   func(v map[string]any) { v["audio"].(map[string]any)["input"].(map[string]any)["turn_detection"] = nil },
				"instructions": func(v map[string]any) { v["instructions"] = "changed" },
				"tool schema": func(v map[string]any) {
					v["tools"].([]any)[0].(map[string]any)["parameters"] = map[string]any{"type": "string"}
				},
			} {
				t.Run(name, func(t *testing.T) {
					endpoint := realtimePeer(t, func(ctx context.Context, ws *websocket.Conn) {
						realtimetest.Write(t, ctx, ws, `{"type":"session.created","session":{"id":"sess_test","type":"realtime"}}`)
						m := realtimetest.Read(t, ctx, ws)
						v := m["session"].(map[string]any)
						v["id"] = "sess_test"
						mutate(v)
						b, err := json.Marshal(map[string]any{"type": "session.updated", "session": v})
						if err != nil {
							t.Error(err)
							return
						}
						realtimetest.Write(t, ctx, ws, string(b))
					})
					if _, err := realtimeClient(t).ConnectRealtime(t.Context(), &genai.RealtimeOptionTools{Tools: []genai.ToolDef{{Name: "lookup", Description: "Lookup", InputSchemaOverride: json.RawMessage(`{"type":"object"}`)}}}, &openaichat.RealtimeOption{Endpoint: endpoint}); err == nil {
						t.Fatal("provider silently changed configuration")
					}
				})
			}
		})
		t.Run("lossless schema acknowledgement", func(t *testing.T) {
			t.Run("valid", func(t *testing.T) {
				want := json.RawMessage(`{"type":"object","properties":{"n":{"minimum":9007199254740993,"maximum":1e999999999999999999999,"const":-0.00}},"enum":[1.00,2.5,null,true,"1"]}`)
				ack := json.RawMessage(`{"enum":[1e0,25e-1,null,true,"1"],"properties":{"n":{"const":0e-999,"maximum":10e999999999999999999998,"minimum":9007199254740993.000}},"type":"object"}`)
				endpoint := realtimePeer(t, func(ctx context.Context, ws *websocket.Conn) {
					realtimetest.Write(t, ctx, ws, `{"type":"session.created","session":{"id":"sess_test","type":"realtime"}}`)
					_, raw, err := ws.Read(ctx)
					if err != nil {
						t.Error(err)
						return
					}
					var m map[string]any
					d := json.NewDecoder(bytes.NewReader(raw))
					d.UseNumber()
					if err := d.Decode(&m); err != nil {
						t.Error(err)
						return
					}
					v := m["session"].(map[string]any)
					v["id"] = "sess_test"
					v["tools"].([]any)[0].(map[string]any)["parameters"] = ack
					b, err := json.Marshal(map[string]any{"type": "session.updated", "session": v})
					if err != nil {
						t.Error(err)
						return
					}
					realtimetest.Write(t, ctx, ws, string(b))
				})
				s, err := realtimeClient(t).ConnectRealtime(t.Context(), &genai.RealtimeOptionTools{Tools: []genai.ToolDef{{Name: "lookup", Description: "Lookup", InputSchemaOverride: want}}}, &openaichat.RealtimeOption{Endpoint: endpoint})
				if err != nil {
					t.Fatal(err)
				}
				realtimetest.Close(t, s)
			})
			t.Run("error", func(t *testing.T) {
				endpoint := realtimePeer(t, func(ctx context.Context, ws *websocket.Conn) {
					realtimetest.Write(t, ctx, ws, `{"type":"session.created","session":{"id":"sess_test","type":"realtime"}}`)
					_, raw, err := ws.Read(ctx)
					if err != nil {
						t.Error(err)
						return
					}
					var m map[string]any
					d := json.NewDecoder(bytes.NewReader(raw))
					d.UseNumber()
					if err := d.Decode(&m); err != nil {
						t.Error(err)
						return
					}
					v := m["session"].(map[string]any)
					v["id"] = "sess_test"
					v["tools"].([]any)[0].(map[string]any)["parameters"] = json.RawMessage(`{"type":"object","properties":{"n":{"minimum":9007199254740993}}}`)
					b, err := json.Marshal(map[string]any{"type": "session.updated", "session": v})
					if err != nil {
						t.Error(err)
						return
					}
					realtimetest.Write(t, ctx, ws, string(b))
				})
				_, err := realtimeClient(t).ConnectRealtime(t.Context(), &genai.RealtimeOptionTools{Tools: []genai.ToolDef{{Name: "lookup", Description: "Lookup", InputSchemaOverride: json.RawMessage(`{"type":"object","properties":{"n":{"minimum":9007199254740992}}}`)}}}, &openaichat.RealtimeOption{Endpoint: endpoint})
				if err == nil {
					t.Fatal("adjacent integers above 2^53 were treated as equal")
				}
			})
		})

		t.Run("concurrent send and close", func(t *testing.T) {
			ready := make(chan struct{})
			endpoint := realtimePeer(t, func(ctx context.Context, ws *websocket.Conn) {
				realtimeSetup(t, ctx, ws)
				for range 16 {
					m := realtimetest.Read(t, ctx, ws)
					if m["type"] != "conversation.item.create" {
						t.Error("concurrent write corrupted")
					}
				}
				close(ready)
			})
			s, err := realtimeClient(t).ConnectRealtime(t.Context(), &openaichat.RealtimeOption{Endpoint: endpoint})
			if err != nil {
				t.Fatal(err)
			}
			var wg sync.WaitGroup
			for range 16 {
				wg.Go(func() {
					if err := s.Send(t.Context(), genai.RealtimeInputContent{Text: "context"}); err != nil {
						t.Error(err)
					}
				})
			}
			wg.Wait()
			<-ready
			done := make(chan error, 1)
			wg.Go(func() {
				next, stop := iter.Pull2(s.Receive(t.Context()))
				_, err, _ := next()
				stop()
				done <- err
			})
			if err := s.Close(); err != nil {
				t.Fatal(err)
			}
			select {
			case err := <-done:
				if err == nil {
					t.Fatal("Close did not terminate Receive")
				}
			case <-time.After(time.Second):
				t.Fatal("Close did not wake Receive")
			}
			wg.Wait()
		})
		t.Run("cancel current before response creation and recover", func(t *testing.T) {
			endpoint := realtimePeer(t, func(ctx context.Context, ws *websocket.Conn) {
				realtimeSetup(t, ctx, ws)
				m := realtimetest.Read(t, ctx, ws)
				id, ok := m["event_id"].(string)
				if m["type"] != "response.cancel" || !ok || id == "" {
					t.Error("cancel omitted correlation")
				}
				b, err := json.Marshal(map[string]any{"type": "error", "error": map[string]any{"code": "response_cancel_not_active", "event_id": id}})
				if err != nil {
					t.Error(err)
					return
				}
				realtimetest.Write(t, ctx, ws, string(b))
				realtimetest.Write(t, ctx, ws, `{"type":"response.created","response":{"id":"r"}}`)
				realtimetest.Write(t, ctx, ws, `{"type":"response.output_audio.delta","response_id":"r","item_id":"a","delta":"AQACAA=="}`)
				realtimetest.Write(t, ctx, ws, `{"type":"response.done","response":{"id":"r","status":"completed"}}`)
			})
			s, err := realtimeClient(t).ConnectRealtime(t.Context(), &openaichat.RealtimeOption{Endpoint: endpoint})
			if err != nil {
				t.Fatal(err)
			}
			realtimetest.Close(t, s)
			if err := s.Send(t.Context(), genai.RealtimeCancel{}); err != nil {
				t.Fatal(err)
			}
			realtimetest.Receive(t, s, genai.RealtimeEventError)
			realtimetest.Receive(t, s, genai.RealtimeEventContentAudio)
			realtimetest.Receive(t, s, genai.RealtimeEventGenerationDone)
			realtimetest.Receive(t, s, genai.RealtimeEventTurnDone)
		})
		t.Run("tool correlation exhaustion", func(t *testing.T) {
			for _, limit := range []string{"pending calls", "history"} {
				t.Run(limit, func(t *testing.T) {
					batches := 2
					if limit == "history" {
						batches = 5
					}
					endpoint := realtimePeer(t, func(ctx context.Context, ws *websocket.Conn) {
						realtimeSetup(t, ctx, ws)
						for batch := range batches {
							count := 1024
							if batch == batches-1 {
								count = 1
							}
							calls := make([]map[string]any, count)
							for i := range calls {
								calls[i] = map[string]any{"type": "function_call", "call_id": fmt.Sprintf("call_%d_%d", batch, i), "name": "lookup", "arguments": "{}"}
							}
							id := fmt.Sprintf("response_%d", batch)
							b, err := json.Marshal(map[string]any{"type": "response.created", "response": map[string]any{"id": id}})
							if err != nil {
								t.Error(err)
								return
							}
							realtimetest.Write(t, ctx, ws, string(b))
							b, err = json.Marshal(map[string]any{"type": "response.done", "response": map[string]any{"id": id, "status": "completed", "output": calls}})
							if err != nil {
								t.Error(err)
								return
							}
							realtimetest.Write(t, ctx, ws, string(b))
							if limit == "history" && batch < batches-1 {
								for i := range count {
									m := realtimetest.Read(t, ctx, ws)
									want := map[string]any{"type": "conversation.item.create", "item": map[string]any{"type": "function_call_output", "call_id": fmt.Sprintf("call_%d_%d", batch, i), "output": "done"}}
									if !reflect.DeepEqual(m, want) {
										t.Errorf("tool result %#v, want %#v", m, want)
										return
									}
								}
								if m := realtimetest.Read(t, ctx, ws); !reflect.DeepEqual(m, map[string]any{"type": "response.create"}) {
									t.Errorf("tool continuation %#v", m)
									return
								}
							}
						}
					})
					s, err := realtimeClient(t).ConnectRealtime(t.Context(), &genai.RealtimeOptionTools{Tools: []genai.ToolDef{{Name: "lookup", Description: "Lookup", InputSchemaOverride: json.RawMessage(`{"type":"object"}`)}}}, &openaichat.RealtimeOption{Endpoint: endpoint})
					if err != nil {
						t.Fatal(err)
					}
					realtimetest.Close(t, s)
					for range batches - 1 {
						e := realtimetest.Receive(t, s, genai.RealtimeEventContentToolCalls)
						if len(e.Content.ToolCalls) != 1024 {
							t.Fatalf("batch size %d", len(e.Content.ToolCalls))
						}
						realtimetest.Receive(t, s, genai.RealtimeEventGenerationDone)
						if limit == "history" {
							results := make([]genai.ToolCallResult, len(e.Content.ToolCalls))
							for i := range e.Content.ToolCalls {
								v := &e.Content.ToolCalls[i]
								results[i] = genai.ToolCallResult{ID: v.ID, Name: v.Name, Result: "done"}
							}
							if err := s.Send(t.Context(), genai.RealtimeInputContent{ToolResults: results}); err != nil {
								t.Fatal(err)
							}
						}
					}
					next, stop := iter.Pull2(s.Receive(t.Context()))
					_, err, ok := next()
					stop()
					if !ok {
						t.Fatal("realtime session ended")
					}
					if err == nil {
						t.Fatal("exhausted tool state authorized another call")
					}
					if err := s.Send(t.Context(), genai.RealtimeInputContent{Text: "after", Generate: true}); err == nil {
						t.Fatal("exhaustion was not terminal")
					}
				})
			}
		})
		t.Run("oversized decoded response collections", func(t *testing.T) {
			endpoint := realtimePeer(t, func(ctx context.Context, ws *websocket.Conn) {
				realtimeSetup(t, ctx, ws)
				realtimetest.Write(t, ctx, ws, `{"type":"response.done","response":{"id":"r","status":"completed","output":[`+strings.Repeat(`{},`, 1024)+`{}]}}`)
			})
			s, err := realtimeClient(t).ConnectRealtime(t.Context(), &openaichat.RealtimeOption{Endpoint: endpoint})
			if err != nil {
				t.Fatal(err)
			}
			realtimetest.Close(t, s)
			next, stop := iter.Pull2(s.Receive(t.Context()))
			_, err, ok := next()
			stop()
			if !ok {
				t.Fatal("realtime session ended")
			}
			if err == nil {
				t.Fatal("oversized output collection decoded")
			}
		})
		t.Run("requested transcription acknowledgement", func(t *testing.T) {
			for _, got := range []string{`null`, `{"model":"changed","language":"en"}`, `{"model":"transcribe","language":"fr"}`} {
				t.Run(got, func(t *testing.T) {
					endpoint := realtimePeer(t, func(ctx context.Context, ws *websocket.Conn) {
						realtimetest.Write(t, ctx, ws, `{"type":"session.created","session":{"id":"sess_test","type":"realtime"}}`)
						m := realtimetest.Read(t, ctx, ws)
						v := m["session"].(map[string]any)
						v["id"] = "sess_test"
						var transcription any
						if err := json.Unmarshal([]byte(got), &transcription); err != nil {
							t.Error(err)
							return
						}
						v["audio"].(map[string]any)["input"].(map[string]any)["transcription"] = transcription
						b, err := json.Marshal(map[string]any{"type": "session.updated", "session": v})
						if err != nil {
							t.Error(err)
							return
						}
						realtimetest.Write(t, ctx, ws, string(b))
					})
					if _, err := realtimeClient(t).ConnectRealtime(t.Context(), &genai.RealtimeOptionAudio{InputTranscription: true}, &openaichat.RealtimeOption{Endpoint: endpoint, InputTranscriptionModel: "transcribe", InputLanguage: "en"}); err == nil {
						t.Fatal("changed transcription configuration accepted")
					}
				})
			}
		})
		t.Run("successful cancellation before response creation", func(t *testing.T) {
			endpoint := realtimePeer(t, func(ctx context.Context, ws *websocket.Conn) {
				realtimeSetup(t, ctx, ws)
				m := realtimetest.Read(t, ctx, ws)
				if m["type"] != "response.cancel" || m["response_id"] != nil {
					t.Errorf("cancel current %#v", m)
				}
				for _, b := range []string{
					`{"type":"response.created","response":{"id":"cancelled"}}`,
					`{"type":"response.output_audio.delta","response_id":"cancelled","item_id":"audio","delta":"AQACAA=="}`,
					`{"type":"response.output_text.delta","response_id":"cancelled","item_id":"text","delta":"stale"}`,
					`{"type":"response.output_audio_transcript.done","response_id":"cancelled","item_id":"audio","transcript":"stale"}`,
					`{"type":"response.done","response":{"id":"cancelled","status":"cancelled","output":[{"type":"function_call","call_id":"never","name":"lookup","arguments":"{}"}]}}`,
					`{"type":"response.created","response":{"id":"next"}}`,
					`{"type":"response.output_text.done","response_id":"next","item_id":"text_next","text":"fresh"}`,
					`{"type":"response.done","response":{"id":"next","status":"completed"}}`,
				} {
					realtimetest.Write(t, ctx, ws, b)
				}
			})
			s, err := realtimeClient(t).ConnectRealtime(t.Context(), &openaichat.RealtimeOption{Endpoint: endpoint})
			if err != nil {
				t.Fatal(err)
			}
			realtimetest.Close(t, s)
			if err := s.Send(t.Context(), genai.RealtimeCancel{}); err != nil {
				t.Fatal(err)
			}
			realtimetest.Receive(t, s, genai.RealtimeEventInterrupted)
			realtimetest.Receive(t, s, genai.RealtimeEventGenerationDone)
			realtimetest.Receive(t, s, genai.RealtimeEventTurnDone)
			e := realtimetest.Receive(t, s, genai.RealtimeEventContentText)
			if e.Content.Text != "fresh" || e.ResponseID != "next" {
				t.Fatalf("event %+v", e)
			}
			realtimetest.Receive(t, s, genai.RealtimeEventGenerationDone)
			realtimetest.Receive(t, s, genai.RealtimeEventTurnDone)
		})
	})

	t.Run("New", func(t *testing.T) {
		t.Run("valid", func(t *testing.T) {
			t.Run("embedding modality", func(t *testing.T) {
				for _, id := range []string{"text-embedding-3-small", "gpt-6-luna", "unknown-model"} {
					t.Run(id, func(t *testing.T) {
						c, err := openaichat.New(t.Context(), genai.ProviderOptionAPIKey("test"), genai.ProviderOptionModel(id), genai.ProviderOptionTransportWrapper(func(http.RoundTripper) http.RoundTripper { return embeddingNoTransport{t: t} }))
						if err != nil {
							t.Fatal(err)
						}
						internaltest.CleanupCloser(t, c)
						want := genai.Modalities{genai.ModalityText}
						if id == "text-embedding-3-small" {
							want = genai.Modalities{genai.ModalityEmbedding}
						}
						sb := c.Scoreboard()
						if !slices.Equal(c.OutputModalities(), want) || (sb.Embedding(id) != nil) != (id == "text-embedding-3-small") {
							t.Fatalf("modalities %v, qualification %+v", c.OutputModalities(), sb.Embedding(id))
						}
					})
				}
			})
		})
		t.Run("error", func(t *testing.T) {
			t.Run("embedding preference", func(t *testing.T) {
				c, err := openaichat.New(t.Context(), genai.ProviderOptionAPIKey("test"), genai.ModelCheap, genai.ProviderOptionModalities{genai.ModalityEmbedding}, genai.ProviderOptionTransportWrapper(func(http.RoundTripper) http.RoundTripper { return embeddingNoTransport{t: t} }))
				if err == nil || c != nil {
					t.Fatalf("client %v, error %v", c, err)
				}
			})
		})
	})

	testRecorder := internaltest.NewRecords()
	t.Cleanup(func() {
		if err := testRecorder.Close(); err != nil {
			t.Error(err)
		}
	})
	cl, err2 := getClientInner(t, func(h http.RoundTripper) http.RoundTripper {
		return testRecorder.RecordWithName(t, t.Name()+"/Warmup", h)
	})
	if err2 != nil {
		t.Fatal(err2)
	}
	cachedModels, err2 := cl.ListModels(t.Context())
	if err2 != nil {
		t.Fatal(err2)
	}
	getClient := func(t *testing.T, m string, ropts ...recorder.Option) genai.Provider {
		t.Parallel()
		opts := []genai.ProviderOption{genai.ProviderOptionPreloadedModels(cachedModels)}
		if m != "" {
			opts = append(opts, genai.ProviderOptionModel(m))
		}
		ci, err := getClientInner(t, func(h http.RoundTripper) http.RoundTripper {
			return testRecorder.Record(t, h, ropts...)
		}, opts...)
		if err != nil {
			t.Fatal(err)
		}
		return ci
	}

	t.Run("Capabilities", func(t *testing.T) {
		internaltest.TestCapabilities(t, getClient(t, "gpt-6-luna"))
	})

	t.Run("GenAsync-Text", func(t *testing.T) {
		c := getClient(t, "gpt-6.1-terra")
		job := internaltest.TestCapabilitiesGenAsync(t, c)
		if t.Failed() {
			return
		}
		res, err := c.PokeResult(t.Context(), job)
		if err != nil {
			t.Fatal(err)
		}
		if res.Usage.FinishReason != genai.Pending && res.Usage.FinishReason != genai.FinishedStop {
			t.Errorf("finish reason: %s", res.Usage.FinishReason)
		}
	})

	t.Run("Caching-Text", func(t *testing.T) {
		internaltest.TestCapabilitiesCaching(t, getClient(t, string(genai.ModelCheap)))
	})

	t.Run("Scoreboard", func(t *testing.T) {
		c := getClient(t, "")
		genaiModels, err := c.ListModels(t.Context())
		if err != nil {
			t.Fatal(err)
		}
		scenarios := c.Scoreboard().Models()
		models := make([]scoreboard.Model, 0, len(genaiModels))
		for _, m := range genaiModels {
			id := m.GetID()
			if strings.HasPrefix(id, "gpt-5") {
				continue
			}
			found := false
			for _, sc := range scenarios {
				if sc.Model == id {
					models = append(models, scoreboard.Model{Model: id, Reason: sc.Reason})
					found = true
				}
			}
			if !found {
				reason := (strings.HasPrefix(id, "gpt-6") || strings.HasPrefix(id, "o")) && !strings.Contains(id, "moderation")
				models = append(models, scoreboard.Model{Model: id, Reason: reason})
			}
		}
		if !slices.Contains(models, scoreboard.Model{Model: "text-embedding-3-small"}) {
			models = append(models, scoreboard.Model{Model: "text-embedding-3-small"})
		}
		getClientRT := func(t testing.TB, model scoreboard.Model, fn func(http.RoundTripper) http.RoundTripper) genai.Provider {
			provOpts := []genai.ProviderOption{
				genai.ProviderOptionPreloadedModels(cachedModels),
			}
			if model.Model != "" {
				provOpts = append(provOpts, genai.ProviderOptionModel(model.Model))
			}
			key := internaltest.GetEnv("OPENAI_API_KEY")
			if key == "" {
				key = "<insert_api_key_here>"
			}
			provOpts = append(provOpts, genai.ProviderOptionAPIKey(key))
			if fn != nil {
				provOpts = append([]genai.ProviderOption{genai.ProviderOptionTransportWrapper(fn)}, provOpts...)
			}
			c, err := openaichat.New(t.Context(), provOpts...)
			if c != nil {
				internaltest.CleanupCloser(t, c)
			}
			if err != nil {
				t.Fatal(err)
			}
			rec := &internaltest.RealtimeRecorder{
				Provider: c,
				T:        t,
				Target:   "wss://api.openai.com/v1/realtime",
				Dir:      filepath.Join("testdata", filepath.FromSlash(t.Name())),
				Updating: internaltest.UpdatingRecordings(),
				Mutation: func(opts []genai.RealtimeOption, endpoint string) []genai.RealtimeOption {
					out := slices.Clone(opts)
					for i, opt := range out {
						if o, ok := opt.(*openaichat.RealtimeOption); ok {
							clone := *o
							clone.Endpoint = endpoint
							out[i] = &clone
							return out
						}
					}
					return append(out, &openaichat.RealtimeOption{Endpoint: endpoint})
				},
			}
			var p genai.Provider = rec
			if strings.Contains(model.Model, "realtime") {
				effort := genai.EffortOff
				if model.Reason {
					effort = genai.EffortHigh
				}
				p = &internaltest.InjectRealtimeOptions{
					Provider: rec,
					Opts:     []genai.RealtimeOption{effort},
				}
			} else {
				// This will lead to spurious HTTP 500 but it is 25% of the cost.
				tier := openaichat.ServiceTierFlex
				res := genai.EffortLow
				if model.Reason {
					txt := &openaichat.GenOptionText{ServiceTier: tier}
					p = &injectReasoning{
						Provider: &internaltest.InjectOptions{
							Provider: rec,
							Opts:     []genai.GenOption{txt, res},
						},
					}
					if tier == openaichat.ServiceTierFlex {
						p = &internaltest.RetryOnRateLimit{
							Provider: p,
							FallbackOpts: []genai.GenOption{
								&openaichat.GenOptionText{ServiceTier: openaichat.ServiceTierDefault},
							},
						}
					}
				} else if slices.Equal(c.OutputModalities(), []genai.Modality{genai.ModalityText}) {
					// See https://platform.openai.com/docs/guides/flex-processing
					if id := c.ModelID(); id == "o3" || id == "o4-mini" || strings.HasPrefix(id, "gpt-6") {
						txt := &openaichat.GenOptionText{ServiceTier: tier}
						var thinking genai.EffortLevel
						if strings.HasPrefix(id, "gpt-6") {
							thinking = genai.EffortOff
						}
						p = &internaltest.InjectOptions{
							Provider: rec,
							Opts:     []genai.GenOption{txt, thinking},
						}
						if tier == openaichat.ServiceTierFlex {
							p = &internaltest.RetryOnRateLimit{
								Provider: p,
								FallbackOpts: []genai.GenOption{
									&openaichat.GenOptionText{ServiceTier: openaichat.ServiceTierDefault},
								},
							}
						}
					}
				}
			}
			return p
		}
		smoketest.Run(t, smoke.HTTPSession(testRecorder.Records, getClientRT), models, &smoketest.RunOptions{RealtimeVoices: openaibase.RealtimeVoices(), Qualify: []scoreboard.Model{
			{Model: "text-embedding-3-small"},
			// A model that declines a session records why, so the harness does not ask
			// again on every run.
			{Model: "gpt-6-astra", Reason: true},
			{Model: "gpt-6-luna"},
			{Model: "gpt-6-luna", Reason: true},
			{Model: "gpt-6.1-sol", Reason: true},
			{Model: "gpt-6.1-terra", Reason: true},
			{Model: "gpt-audio"},
		}})
	})

	t.Run("Batch", func(t *testing.T) {
		c := getClient(t, "gpt-6-luna")
		// Retrieve a completed GPT-6 Luna batch; submission is covered by GenAsync-Text.
		res, err := c.PokeResult(t.Context(), "batch_6ae531c2038c819086eb55c893c8af0e")
		if err != nil {
			t.Fatal(err)
		}
		if res.Usage.InputTokens == 0 || res.Usage.OutputTokens == 0 {
			t.Error("expected usage")
		}
		if res.Usage.FinishReason != genai.FinishedStop {
			t.Errorf("finish reason: %s", res.Usage.FinishReason)
		}
		if s := res.String(); len(s) < 15 {
			t.Errorf("not enough text: %q", s)
		}
	})

	t.Run("Preferred", func(t *testing.T) {
		internaltest.TestPreferredModels(t, func(st *testing.T, model string, modality genai.Modality) (genai.Provider, error) {
			opts := []genai.ProviderOption{
				genai.ProviderOptionModalities{modality},
				genai.ProviderOptionPreloadedModels(cachedModels),
			}
			if model != "" {
				opts = append(opts, genai.ProviderOptionModel(model))
			}
			return getClientInner(st, func(h http.RoundTripper) http.RoundTripper {
				return testRecorder.Record(st, h)
			}, opts...)
		})
	})

	t.Run("TextOutputDocInput", func(t *testing.T) {
		internaltest.TestTextOutputDocInput(t, func(t *testing.T) genai.Provider {
			return getClient(t, string(genai.ModelCheap))
		})
	})

	t.Run("errors", func(t *testing.T) {
		data := []internaltest.ProviderError{
			{
				Name: "bad apiKey",
				Opts: []genai.ProviderOption{
					genai.ProviderOptionAPIKey("bad apiKey"),
					genai.ProviderOptionModel("gpt-6-luna"),
				},
				ErrGenSync:   "http 401\ninvalid_request_error/invalid_api_key: Incorrect API key provided: bad apiKey. You can find your API key at https://platform.openai.com/account/api-keys.",
				ErrGenStream: "http 401\ninvalid_request_error/invalid_api_key: Incorrect API key provided: bad apiKey. You can find your API key at https://platform.openai.com/account/api-keys.",
				ErrListModel: "http 401\ninvalid_request_error/invalid_api_key: Incorrect API key provided: bad apiKey. You can find your API key at https://platform.openai.com/account/api-keys.",
			},
			{
				Name: "bad apiKey image",
				Opts: []genai.ProviderOption{
					genai.ProviderOptionAPIKey("bad apiKey"),
					genai.ProviderOptionModel("gpt-image-1"),
					genai.ProviderOptionModalities{genai.ModalityImage},
				},
				ErrGenSync:   "http 401\ninvalid_request_error/invalid_api_key: Incorrect API key provided: bad apiKey. You can find your API key at https://platform.openai.com/account/api-keys.",
				ErrGenStream: "http 401\ninvalid_request_error/invalid_api_key: Incorrect API key provided: bad apiKey. You can find your API key at https://platform.openai.com/account/api-keys.",
			},
			{
				Name: "bad model",
				Opts: []genai.ProviderOption{
					genai.ProviderOptionModel("bad model"),
				},
				ErrGenSync:   "http 400\ninvalid_request_error: invalid model ID",
				ErrGenStream: "http 400\ninvalid_request_error: invalid model ID",
			},
			{
				Name: "bad model image",
				Opts: []genai.ProviderOption{
					genai.ProviderOptionModel("bad model"),
					genai.ProviderOptionModalities{genai.ModalityImage},
				},
				ErrGenSync:   "http 400\nimage_generation_user_error/invalid_value for \"model\": The model 'bad model' does not exist.",
				ErrGenStream: "http 400\nimage_generation_user_error/invalid_value for \"model\": The model 'bad model' does not exist.",
			},
		}
		f := func(t *testing.T, opts ...genai.ProviderOption) (genai.Provider, error) {
			return getClientInner(t, func(h http.RoundTripper) http.RoundTripper {
				return testRecorder.Record(t, h)
			}, opts...)
		}
		internaltest.TestClientProviderErrors(t, f, data)
	})
}

// OpenAI returns the count of reasoning tokens but never return them. Duh. This messes up the scoreboard so
// inject fake reasoning whitespace.
type injectReasoning struct {
	genai.Provider
}

func (i *injectReasoning) GenSync(ctx context.Context, msgs genai.Messages, opts ...genai.GenOption) (genai.Result, error) {
	res, err := i.Provider.GenSync(ctx, msgs, opts...)
	if res.Usage.ReasoningTokens > 0 {
		res.Replies = append(res.Replies, genai.Reply{Reasoning: "\n"})
	}
	return res, err
}

func (i *injectReasoning) GenStream(ctx context.Context, msgs genai.Messages, opts ...genai.GenOption) (iter.Seq[genai.Reply], func() (genai.Result, error)) {
	res, err := i.Provider.GenStream(ctx, msgs, opts...)
	return res, err
}

func init() {
	internal.BeLenient = false
}

type embeddingNoTransport struct{ t *testing.T }

func (e embeddingNoTransport) RoundTrip(*http.Request) (*http.Response, error) {
	e.t.Fatal("unexpected HTTP request")
	return nil, errors.New("unexpected HTTP request")
}

func realtimeClient(t *testing.T) *openaichat.Client {
	c, err := openaichat.New(t.Context(), genai.ProviderOptionAPIKey("realtime-test-key"), genai.ProviderOptionModel("gpt-realtime-2.1"))
	if err != nil {
		t.Fatal(err)
	}
	return c
}

func realtimePeer(t *testing.T, script func(context.Context, *websocket.Conn)) string {
	u := realtimetest.Peer(t, func(r *http.Request) {
		if r.Header.Get("Authorization") != "Bearer realtime-test-key" || r.URL.Query().Get("model") != "gpt-realtime-2.1" {
			t.Error("missing authentication or explicit model")
		}
	}, script)
	return "ws" + strings.TrimPrefix(u, "http")
}

func realtimeSetup(t *testing.T, ctx context.Context, ws *websocket.Conn) map[string]any {
	realtimetest.Write(t, ctx, ws, `{"type":"session.created","session":{"id":"sess_test","type":"realtime","max_output_tokens":"inf"}}`)
	m := realtimetest.Read(t, ctx, ws)
	setup, ok := m["session"].(map[string]any)
	if !ok || m["type"] != "session.update" {
		t.Error("missing session update")
		return nil
	}
	if setup["parallel_tool_calls"] != true {
		t.Errorf("parallel tool calls %#v", setup)
	}
	setup["id"] = "sess_test"
	b, err := json.Marshal(map[string]any{"type": "session.updated", "session": setup})
	if err != nil {
		t.Error(err)
		return nil
	}
	realtimetest.Write(t, ctx, ws, string(b))
	return setup
}
