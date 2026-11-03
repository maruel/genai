// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Tests for the llama.cpp provider client.

package llamacpp_test

import (
	"bytes"
	"context"
	"crypto/rand"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"iter"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"slices"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/coder/websocket"

	"github.com/maruel/roundtrippers"

	"github.com/maruel/genai"
	"github.com/maruel/genai/base"
	"github.com/maruel/genai/internal"
	"github.com/maruel/genai/internal/internaltest"
	"github.com/maruel/genai/internal/realtimetest"
	"github.com/maruel/genai/providers/llamacpp"
	"github.com/maruel/genai/providers/llamacpp/llamacppsrv"
	"github.com/maruel/genai/scoreboard"
	"github.com/maruel/genai/smoke/smoketest"
)

// embeddingModel is qualified outside the scoreboard scenarios.
const embeddingModel = "ggml-org/embeddinggemma-2-GGUF/embeddinggemma-2-Q8_0.gguf"

func TestClient(t *testing.T) {
	t.Run("ConnectRealtime", func(t *testing.T) {
		t.Run("unsupported option type", func(t *testing.T) {
			s, err := realtimeClient(t, "http://127.0.0.1:1").ConnectRealtime(t.Context(), genai.GenOptionSeed(1))
			if _, ok := errors.AsType[*base.ErrNotSupported](err); s != nil || !ok {
				t.Fatalf("unsupported option: %+v, %v", s, err)
			}
		})
		t.Run("valid", func(t *testing.T) {
			t.Run("incremental audio tools and native rates", func(t *testing.T) {
				commands := make(chan []map[string]any, 1)
				remote := realtimePeer(t, func(ctx context.Context, ws *websocket.Conn) {
					setup := realtimeSetup(t, ctx, ws)
					if !reflect.DeepEqual(setup["output_modalities"], []any{"text", "audio"}) || setup["max_output_tokens"] != float64(128) {
						t.Errorf("setup %#v", setup)
					}
					got := make([]map[string]any, 0, 8)
					for range 5 {
						got = append(got, realtimetest.Read(t, ctx, ws))
					}
					realtimetest.Write(t, ctx, ws, `{"type":"response.created","response":{"id":"r_tools"}}`)
					realtimetest.Write(t, ctx, ws, `{"type":"response.function_call_arguments.delta","response_id":"r_tools","item_id":"i","output_index":0,"delta":"{}"}`)
					realtimetest.Write(t, ctx, ws, `{"type":"response.function_call_arguments.done","response_id":"r_tools","item_id":"i","output_index":0,"call_id":"c","name":"lookup","arguments":"{}"}`)
					realtimetest.Write(t, ctx, ws, `{"type":"response.done","response":{"id":"r_tools","status":"completed","output":[{"type":"function_call","call_id":"c","name":"lookup","arguments":"{}"},{"type":"function_call","call_id":"d","name":"lookup","arguments":"{}"}],"usage":{"output_tokens":5}}}`)
					for range 3 {
						got = append(got, realtimetest.Read(t, ctx, ws))
					}
					commands <- got
					for _, b := range []string{
						`{"type":"response.created","response":{"id":"r_audio"}}`,
						`{"type":"response.output_text.delta","response_id":"r_audio","item_id":"a","delta":"Hi"}`,
						`{"type":"response.output_audio.delta","response_id":"r_audio","item_id":"a","content_index":1,"delta":"AQACAA=="}`,
						`{"type":"response.output_text.done","response_id":"r_audio","item_id":"a","text":"Hi."}`,
						`{"type":"response.done","response":{"id":"r_audio","status":"completed"}}`,
					} {
						realtimetest.Write(t, ctx, ws, b)
					}
				})
				s, err := realtimeClient(t, remote).ConnectRealtime(t.Context(), genai.Modalities{genai.ModalityAudio, genai.ModalityText}, &genai.RealtimeOptionTools{Tools: []genai.ToolDef{{Name: "lookup", Description: "Lookup", InputSchemaOverride: json.RawMessage(`{"type":"object"}`)}}}, &llamacpp.RealtimeOption{MaxOutputTokens: 128, RequireToolCall: true})
				if err != nil {
					t.Fatal(err)
				}
				realtimetest.Close(t, s)
				info := s.Info()
				want := genai.RealtimeSessionInfo{ID: "local", InputFormat: genai.AudioFormat{SampleRate: 16000}, OutputFormat: genai.AudioFormat{SampleRate: 24000}, InputModalities: []genai.Modality{genai.ModalityAudio, genai.ModalityText}, TurnDetection: genai.RealtimeTurnManual, Capabilities: genai.RealtimeCapabilities{AudioCommit: true, CancelResponse: true, ToolResults: true, Update: genai.RealtimeUpdateCapabilities{Instructions: true, Tools: true, ToolChoice: true, ToolChoiceRequired: true, MaxOutputTokens: true, OutputModalities: true}}}
				if !reflect.DeepEqual(info, want) {
					t.Fatalf("info %+v, want %+v", info, want)
				}
				for _, in := range []genai.RealtimeInput{genai.RealtimeInputContent{Text: "context"}, genai.RealtimeInputContent{Audio: []byte{1, 0}}, genai.RealtimeInputContent{Audio: []byte{2, 0}}, genai.RealtimeInputContent{Commit: true, Generate: true}} {
					if err := s.Send(t.Context(), in); err != nil {
						t.Fatal(err)
					}
				}
				p := realtimetest.Receive(t, s, genai.RealtimeEventProvider)
				if p.ProviderEvent.RealtimeEventName() != "response.function_call_arguments.delta" {
					t.Fatalf("provider event %+v", p)
				}
				if d, ok := p.ProviderEvent.(*llamacpp.FunctionCallArgsDelta); !ok || d.ItemID != "i" || d.Delta != "{}" {
					t.Fatalf("delta %+v", p)
				}
				p = realtimetest.Receive(t, s, genai.RealtimeEventProvider)
				if d, ok := p.ProviderEvent.(*llamacpp.FunctionCallArgsDone); !ok || d.CallID != "c" || d.Name != "lookup" || d.Arguments != "{}" {
					t.Fatalf("done %+v", p)
				}
				e := realtimetest.Receive(t, s, genai.RealtimeEventContentToolCalls)
				if len(e.Content.ToolCalls) != 2 || e.Content.ToolCalls[0].ID != "c" {
					t.Fatal(e)
				}
				u := realtimetest.Receive(t, s, genai.RealtimeEventUsage)
				if u.Usage.OutputTokens != 5 {
					t.Fatal(u)
				}
				realtimetest.Receive(t, s, genai.RealtimeEventGenerationDone)
				if err := s.Send(t.Context(), genai.RealtimeInputContent{Text: "cannot skip pending tool"}); err == nil {
					t.Fatal("accepted new input before tool results")
				}
				if err := s.Send(t.Context(), genai.RealtimeInputContent{ToolResults: []genai.ToolCallResult{{ID: "c", Name: "wrong", Result: "value"}}, Generate: true}); err == nil {
					t.Fatal("mismatched tool name accepted")
				}
				if err := s.Send(t.Context(), genai.RealtimeInputContent{ToolResults: []genai.ToolCallResult{{ID: "d", Name: "lookup", Result: "second"}, {ID: "c", Name: "lookup", Result: "value"}}, Generate: true}); err != nil {
					t.Fatal(err)
				}
				got := <-commands
				wantCommands := []map[string]any{
					{"type": "conversation.item.create", "item": map[string]any{"type": "message", "role": "user", "content": []any{map[string]any{"type": "input_text", "text": "context"}}}},
					{"type": "input_audio_buffer.append", "audio": "AQA="}, {"type": "input_audio_buffer.append", "audio": "AgA="}, {"type": "input_audio_buffer.commit"}, {"type": "response.create", "response": map[string]any{"tool_choice": "required"}},
					{"type": "conversation.item.create", "item": map[string]any{"type": "function_call_output", "call_id": "c", "output": "value"}}, {"type": "conversation.item.create", "item": map[string]any{"type": "function_call_output", "call_id": "d", "output": "second"}}, {"type": "response.create"},
				}
				if !reflect.DeepEqual(got, wantCommands) {
					t.Fatalf("commands %#v", got)
				}
				realtimetest.Receive(t, s, genai.RealtimeEventContentText)
				a := realtimetest.Receive(t, s, genai.RealtimeEventContentAudio)
				if !bytes.Equal(a.Content.Audio, []byte{1, 0, 2, 0}) || a.ContentIndex != 1 {
					t.Fatal(a)
				}
				e = realtimetest.Receive(t, s, genai.RealtimeEventContentText)
				if !e.Content.Final || e.Content.Text != "Hi." {
					t.Fatal(e)
				}
				realtimetest.Receive(t, s, genai.RealtimeEventGenerationDone)
				realtimetest.Receive(t, s, genai.RealtimeEventTurnDone)
			})
			t.Run("discover model-defined native formats", func(t *testing.T) {
				remote := realtimePeer(t, func(ctx context.Context, ws *websocket.Conn) {
					frame := strings.ReplaceAll(strings.ReplaceAll(nativeSession, `"rate":16000`, `"rate":8000`), `"rate":24000`, `"rate":48000`)
					realtimetest.Write(t, ctx, ws, frame)
					m := realtimetest.Read(t, ctx, ws)
					settings := m["session"].(map[string]any)
					settings["id"] = "local"
					b, err := json.Marshal(map[string]any{"type": "session.updated", "session": settings})
					if err != nil {
						t.Error(err)
						return
					}
					realtimetest.Write(t, ctx, ws, string(b))
					_, _, _ = ws.Read(ctx)
				})
				s, err := realtimeClient(t, remote).ConnectRealtime(t.Context())
				if err != nil {
					t.Fatal(err)
				}
				realtimetest.Close(t, s)
				if info := s.Info(); info.InputFormat != (genai.AudioFormat{SampleRate: 8000}) || info.OutputFormat != (genai.AudioFormat{SampleRate: 48000}) {
					t.Fatalf("native model format discovery: %+v", info)
				}
			})
			t.Run("target current response", func(t *testing.T) {
				remote := realtimePeer(t, func(ctx context.Context, ws *websocket.Conn) {
					realtimeSetup(t, ctx, ws)
					for range 2 {
						realtimetest.Read(t, ctx, ws)
					}
					realtimetest.Write(t, ctx, ws, `{"type":"response.created","response":{"id":"active"}}`)
					realtimetest.Write(t, ctx, ws, `{"type":"response.output_text.delta","response_id":"active","item_id":"text","delta":"speaking"}`)
					if m := realtimetest.Read(t, ctx, ws); !reflect.DeepEqual(m, map[string]any{"type": "response.cancel", "response_id": "active"}) {
						t.Errorf("targeted cancellation: %+v", m)
					}
					realtimetest.Write(t, ctx, ws, `{"type":"response.done","response":{"id":"active","status":"cancelled"}}`)
					_, _, _ = ws.Read(ctx)
				})
				s, err := realtimeClient(t, remote).ConnectRealtime(t.Context())
				if err != nil {
					t.Fatal(err)
				}
				realtimetest.Close(t, s)
				if err := s.Send(t.Context(), genai.RealtimeInputContent{Text: "first", Generate: true}); err != nil {
					t.Fatal(err)
				}
				realtimetest.Receive(t, s, genai.RealtimeEventContentText)
				if err := s.Send(t.Context(), genai.RealtimeCancel{ResponseID: "wrong"}); err == nil || errors.Is(err, llamacpp.ErrNoActiveResponse) {
					t.Fatalf("wrong active target: %v", err)
				}
				if err := s.Send(t.Context(), genai.RealtimeCancel{ResponseID: "active"}); err != nil {
					t.Fatal(err)
				}
				realtimetest.Receive(t, s, genai.RealtimeEventInterrupted)
				realtimetest.Receive(t, s, genai.RealtimeEventGenerationDone)
				realtimetest.Receive(t, s, genai.RealtimeEventTurnDone)
			})
			t.Run("cancel pending response and queue next streamed turn", func(t *testing.T) {
				done := make(chan struct{})
				remote := realtimePeer(t, func(ctx context.Context, ws *websocket.Conn) {
					realtimeSetup(t, ctx, ws)
					for range 2 {
						realtimetest.Read(t, ctx, ws)
					}
					m := realtimetest.Read(t, ctx, ws)
					if m["type"] != "response.cancel" {
						t.Fatal("missing cancel")
					}
					for _, b := range []string{
						`{"type":"response.created","response":{"id":"old"}}`,
						`{"type":"response.output_audio.delta","response_id":"old","item_id":"a","delta":"AQACAA=="}`,
						`{"type":"response.output_text.delta","response_id":"old","item_id":"a","delta":"stale"}`,
						`{"type":"response.function_call_arguments.done","response_id":"old","call_id":"never","name":"lookup","arguments":"{}"}`,
						`{"type":"response.done","response":{"id":"old","status":"cancelled","output":[{"type":"function_call","call_id":"never","name":"lookup","arguments":"{}"}]}}`,
					} {
						realtimetest.Write(t, ctx, ws, b)
					}
					for _, kind := range []string{"input_audio_buffer.append", "input_audio_buffer.commit", "response.create"} {
						if m := realtimetest.Read(t, ctx, ws); m["type"] != kind {
							t.Errorf("next command %+v, want %s", m, kind)
						}
					}
					realtimetest.Write(t, ctx, ws, `{"type":"response.created","response":{"id":"new"}}`)
					realtimetest.Write(t, ctx, ws, `{"type":"response.output_text.done","response_id":"new","item_id":"b","text":"fresh"}`)
					realtimetest.Write(t, ctx, ws, `{"type":"response.done","response":{"id":"new","status":"completed"}}`)
					close(done)
				})
				s, err := realtimeClient(t, remote).ConnectRealtime(t.Context())
				if err != nil {
					t.Fatal(err)
				}
				realtimetest.Close(t, s)
				if err := s.Send(t.Context(), genai.RealtimeCancel{}); !errors.Is(err, llamacpp.ErrNoActiveResponse) {
					t.Fatal("idle cancel would poison next response")
				}
				if err := s.Send(t.Context(), genai.RealtimeInputContent{Text: "first", Generate: true}); err != nil {
					t.Fatal(err)
				}
				for _, in := range []genai.RealtimeInput{genai.RealtimeInputContent{Audio: []byte{1, 0}}, genai.RealtimeCancel{ResponseID: "unknown"}} {
					if err := s.Send(t.Context(), in); err == nil {
						t.Fatal("invalid active command accepted")
					}
				}
				if err := s.Send(t.Context(), genai.RealtimeCancel{}); err != nil {
					t.Fatal(err)
				}
				realtimetest.Receive(t, s, genai.RealtimeEventInterrupted)
				realtimetest.Receive(t, s, genai.RealtimeEventGenerationDone)
				realtimetest.Receive(t, s, genai.RealtimeEventTurnDone)
				if err := s.Send(t.Context(), genai.RealtimeInputContent{Audio: []byte{1, 0}, Commit: true, Generate: true}); err != nil {
					t.Fatal(err)
				}
				e := realtimetest.Receive(t, s, genai.RealtimeEventContentText)
				if e.Content.Text != "fresh" || e.ResponseID != "new" {
					t.Fatal(e)
				}
				realtimetest.Receive(t, s, genai.RealtimeEventGenerationDone)
				realtimetest.Receive(t, s, genai.RealtimeEventTurnDone)
				<-done
			})

			t.Run("cancel after decoding terminal response before completion delivery", func(t *testing.T) {
				remote := realtimePeer(t, func(ctx context.Context, ws *websocket.Conn) {
					realtimeSetup(t, ctx, ws)
					for _, kind := range []string{"conversation.item.create", "response.create"} {
						if m := realtimetest.Read(t, ctx, ws); m["type"] != kind {
							t.Errorf("initial command %+v, want %s", m, kind)
						}
					}
					realtimetest.Write(t, ctx, ws, `{"type":"response.created","response":{"id":"old"}}`)
					realtimetest.Write(t, ctx, ws, `{"type":"response.done","response":{"id":"old","status":"completed","usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}}}`)
					// These must be the next commands: idle/targeted/unsupported
					// rejections cannot leak any cancellation to the next response.
					for _, kind := range []string{"input_audio_buffer.append", "input_audio_buffer.commit", "response.create"} {
						if m := realtimetest.Read(t, ctx, ws); m["type"] != kind {
							t.Errorf("next command %+v, want %s", m, kind)
						}
					}
					realtimetest.Write(t, ctx, ws, `{"type":"response.created","response":{"id":"new"}}`)
					realtimetest.Write(t, ctx, ws, `{"type":"response.output_text.done","response_id":"new","item_id":"text","text":"fresh"}`)
					realtimetest.Write(t, ctx, ws, `{"type":"response.done","response":{"id":"new","status":"completed"}}`)
					_, _, _ = ws.Read(ctx)
				})
				s, err := realtimeClient(t, remote).ConnectRealtime(t.Context())
				if err != nil {
					t.Fatal(err)
				}
				realtimetest.Close(t, s)
				if err := s.Send(t.Context(), genai.RealtimeInputContent{Text: "first", Generate: true}); err != nil {
					t.Fatal(err)
				}
				// Usage is the first event from response.done, proving the complete
				// terminal frame was decoded while GenerationDone remains queued.
				realtimetest.Receive(t, s, genai.RealtimeEventUsage)
				if err := s.Send(t.Context(), genai.RealtimeCancel{}); !errors.Is(err, llamacpp.ErrNoActiveResponse) {
					t.Fatalf("idle cancel: %v", err)
				}
				for _, id := range []string{"old", "unknown"} {
					err := s.Send(t.Context(), genai.RealtimeCancel{ResponseID: id})
					if _, unsupported := errors.AsType[*base.ErrNotSupported](err); err == nil || errors.Is(err, llamacpp.ErrNoActiveResponse) || unsupported {
						t.Fatalf("invalid targeted cancellation: %v", err)
					}
				}
				err = s.Send(t.Context(), genai.RealtimeTruncate{ItemID: "a"})
				if _, unsupported := errors.AsType[*base.ErrNotSupported](err); !unsupported || errors.Is(err, llamacpp.ErrNoActiveResponse) {
					t.Fatalf("unsupported command: %v", err)
				}
				if err := s.Send(t.Context(), genai.RealtimeInputContent{Doc: genai.Doc{Src: bytes.NewReader([]byte{0xff}), Filename: "a.png"}, Generate: true}); err == nil {
					t.Fatal("image input accepted")
				} else if _, unsupported := errors.AsType[*base.ErrNotSupported](err); !unsupported {
					t.Fatalf("unsupported image input: %v", err)
				}
				if err := s.Send(t.Context(), genai.RealtimeInputContent{Audio: []byte{1, 0}, Commit: true, Generate: true}); err != nil {
					t.Fatal(err)
				}
				realtimetest.Receive(t, s, genai.RealtimeEventGenerationDone)
				realtimetest.Receive(t, s, genai.RealtimeEventTurnDone)
				if e := realtimetest.Receive(t, s, genai.RealtimeEventContentText); e.Content.Text != "fresh" || e.ResponseID != "new" {
					t.Fatal(e)
				}
				realtimetest.Receive(t, s, genai.RealtimeEventGenerationDone)
				realtimetest.Receive(t, s, genai.RealtimeEventTurnDone)
			})
		})
		t.Run("error", func(t *testing.T) {
			t.Run("unsupported setup and options", func(t *testing.T) {
				c := realtimeClient(t, "http://127.0.0.1:1")
				if _, err := c.ConnectRealtime(t.Context(), nil); err == nil {
					t.Fatal("nil option accepted")
				}
				for _, tc := range []struct {
					name string
					opt  genai.RealtimeOption
				}{
					{"audio", &genai.RealtimeOptionAudio{}},
					{"voice", &genai.RealtimeOptionAudio{Voice: "voice"}},
					{"input transcription", &genai.RealtimeOptionAudio{InputTranscription: true}},
					{"output transcription", &genai.RealtimeOptionAudio{OutputTranscription: true}},
					{"server turns", genai.RealtimeTurnServer},
					{"output modality", genai.Modalities{genai.ModalityImage}},
				} {
					t.Run(tc.name, func(t *testing.T) {
						_, err := c.ConnectRealtime(t.Context(), tc.opt)
						if _, ok := errors.AsType[*base.ErrNotSupported](err); !ok {
							t.Fatalf("error = %v, want ErrNotSupported", err)
						}
					})
				}
				for _, o := range []*llamacpp.RealtimeOption{{MaxOutputTokens: -1}, {MaxOutputTokens: 1 << 32}, {PingInterval: -time.Second}, {Endpoint: "http://127.0.0.1:1"}, {Endpoint: "ws://"}, nil} {
					if _, err := c.ConnectRealtime(t.Context(), o); err == nil {
						t.Fatal("invalid option accepted")
					}
				}
				if _, err := c.ConnectRealtime(t.Context(), &llamacpp.RealtimeOption{}, &llamacpp.RealtimeOption{}); err == nil || !strings.Contains(err.Error(), "duplicate realtime option type") {
					t.Fatal("duplicate options accepted")
				}
			})
			t.Run("malformed native formats before setup write", func(t *testing.T) {
				for name, frame := range map[string]string{
					"input zero rate":     strings.Replace(nativeSession, `"rate":16000`, `"rate":0`, 1),
					"input negative rate": strings.Replace(nativeSession, `"rate":16000`, `"rate":-1`, 1),
					"input encoding":      strings.Replace(nativeSession, `"type":"audio/pcm","rate":16000`, `"type":"audio/pcmu","rate":16000`, 1),
					"output zero rate":    strings.Replace(nativeSession, `"rate":24000`, `"rate":0`, 1),
					"output encoding":     strings.Replace(nativeSession, `"type":"audio/pcm","rate":24000`, `"type":"audio/pcmu","rate":24000`, 1),
				} {
					t.Run(name, func(t *testing.T) {
						done := make(chan struct{})
						remote := realtimePeer(t, func(ctx context.Context, ws *websocket.Conn) {
							realtimetest.Write(t, ctx, ws, frame)
							if _, _, err := ws.Read(ctx); err == nil {
								t.Error("malformed native format wrote setup")
							}
							close(done)
						})
						if _, err := realtimeClient(t, remote).ConnectRealtime(t.Context()); err == nil {
							t.Fatal("malformed native format accepted")
						}
						<-done
					})
				}
			})
			t.Run("command ownership", func(t *testing.T) {
				done := make(chan struct{})
				remote := realtimePeer(t, func(ctx context.Context, ws *websocket.Conn) {
					realtimeSetup(t, ctx, ws)
					for _, want := range []string{"input_audio_buffer.append", "input_audio_buffer.append", "input_audio_buffer.commit", "response.create"} {
						m := realtimetest.Read(t, ctx, ws)
						if m["type"] != want {
							t.Errorf("command %#v", m)
						}
					}
					close(done)
				})
				s, err := realtimeClient(t, remote).ConnectRealtime(t.Context())
				if err != nil {
					t.Fatal(err)
				}
				realtimetest.Close(t, s)
				for _, in := range []genai.RealtimeInput{genai.RealtimeTruncate{ItemID: "a"}} {
					if err := s.Send(t.Context(), in); err == nil {
						t.Fatal("invalid command accepted")
					}
				}
				if err := s.Send(t.Context(), genai.RealtimeInputContent{Audio: []byte{1, 0}}); err != nil {
					t.Fatal(err)
				}
				if err := s.Send(t.Context(), genai.RealtimeInputContent{Text: "turn", Generate: true}); err == nil {
					t.Fatal("generation implicitly committed audio")
				}
				if err := s.Send(t.Context(), genai.RealtimeInputContent{Audio: []byte{2, 0}, Commit: true, Generate: true}); err != nil {
					t.Fatal(err)
				}
				<-done
			})
			t.Run("malformed accepted configuration", func(t *testing.T) {
				for _, field := range []string{"rate", "modality", "tools", "transcription"} {
					t.Run(field, func(t *testing.T) {
						remote := realtimePeer(t, func(ctx context.Context, ws *websocket.Conn) {
							realtimetest.Write(t, ctx, ws, nativeSession)
							m := realtimetest.Read(t, ctx, ws)
							v := m["session"].(map[string]any)
							v["id"] = "local"
							switch field {
							case "rate":
								v["audio"].(map[string]any)["output"].(map[string]any)["format"].(map[string]any)["rate"] = json.Number("16000")
							case "modality":
								v["output_modalities"] = []string{"audio", "text", "audio"}
							case "tools":
								v["tools"] = []any{map[string]any{"type": "function", "name": "extra", "parameters": map[string]any{"type": "object"}}}
							case "transcription":
								v["audio"].(map[string]any)["input"].(map[string]any)["transcription"] = map[string]any{"model": "ignored"}
							}
							b, err := json.Marshal(map[string]any{"type": "session.updated", "session": v})
							if err != nil {
								t.Error(err)
								return
							}
							realtimetest.Write(t, ctx, ws, string(b))
						})
						if _, err := realtimeClient(t, remote).ConnectRealtime(t.Context()); err == nil {
							t.Fatal("changed setup accepted")
						}
					})
				}
			})
			t.Run("recoverable errors release pending generation", func(t *testing.T) {
				remote := realtimePeer(t, func(ctx context.Context, ws *websocket.Conn) {
					realtimeSetup(t, ctx, ws)
					for range 2 {
						realtimetest.Read(t, ctx, ws)
					}
					realtimetest.Write(t, ctx, ws, `{"type":"error","error":{"type":"invalid_request_error","code":null,"message":"bad request","param":null}}`)
					for range 2 {
						realtimetest.Read(t, ctx, ws)
					}
					realtimetest.Write(t, ctx, ws, `{"type":"response.created","response":{"id":"recovered"}}`)
					realtimetest.Write(t, ctx, ws, `{"type":"response.done","response":{"id":"recovered","status":"completed"}}`)
				})
				s, err := realtimeClient(t, remote).ConnectRealtime(t.Context())
				if err != nil {
					t.Fatal(err)
				}
				realtimetest.Close(t, s)
				if err := s.Send(t.Context(), genai.RealtimeInputContent{Text: "first", Generate: true}); err != nil {
					t.Fatal(err)
				}
				realtimetest.Receive(t, s, genai.RealtimeEventError)
				if err := s.Send(t.Context(), genai.RealtimeCancel{}); err == nil {
					t.Fatal("rejected generation remained cancellable")
				}
				if err := s.Send(t.Context(), genai.RealtimeInputContent{Text: "retry", Generate: true}); err != nil {
					t.Fatal(err)
				}
				realtimetest.Receive(t, s, genai.RealtimeEventGenerationDone)
				realtimetest.Receive(t, s, genai.RealtimeEventTurnDone)
			})
			t.Run("lifetime cancellation", func(t *testing.T) {
				remote := realtimePeer(t, func(ctx context.Context, ws *websocket.Conn) { realtimeSetup(t, ctx, ws) })
				ctx, cancel := context.WithCancel(t.Context())
				t.Cleanup(cancel)
				s, err := realtimeClient(t, remote).ConnectRealtime(ctx)
				if err != nil {
					t.Fatal(err)
				}
				realtimetest.Close(t, s)
				cancel()
				next, stop := iter.Pull2(s.Receive(t.Context()))
				_, err, ok := next()
				stop()
				if !ok {
					t.Fatal("realtime session ended")
				}
				if !errors.Is(err, context.Canceled) {
					t.Fatal(err)
				}
				if err := s.Send(t.Context(), genai.RealtimeCancel{}); !errors.Is(err, context.Canceled) || errors.Is(err, llamacpp.ErrNoActiveResponse) {
					t.Fatalf("closed transport cancellation: %v", err)
				}
			})
		})
	})

	t.Run("New", func(t *testing.T) {
		t.Run("error", func(t *testing.T) {
			t.Run("embedding preference", func(t *testing.T) {
				c, err := llamacpp.New(t.Context(), genai.ModelCheap, genai.ProviderOptionModalities{genai.ModalityEmbedding}, genai.ProviderOptionTransportWrapper(func(http.RoundTripper) http.RoundTripper { return compactNoTransport{t: t} }))
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

	b := [12]byte{}
	if _, err := io.ReadFull(rand.Reader, b[:]); err != nil {
		t.Fatal(err)
	}
	apiKey := base64.RawURLEncoding.EncodeToString(b[:])
	s := lazyServer{t: t, apiKey: apiKey}

	t.Run("Embed", func(t *testing.T) {
		t.Run("valid", func(t *testing.T) {
			t.Run("document ownership", func(t *testing.T) {
				c, err := llamacpp.New(t.Context(), genai.ProviderOptionRemote("http://localhost:8080"), genai.ProviderOptionTransportWrapper(func(http.RoundTripper) http.RoundTripper {
					return embeddingDocumentTransport{body: `{"object":"list","data":[{"object":"embedding","index":1,"embedding":"AABAQAAAgEA="},{"object":"embedding","index":0,"embedding":"AACAPwAAAEA="}]}`}
				}))
				if err != nil {
					t.Fatal(err)
				}
				internaltest.CleanupCloser(t, c)
				src := &embeddingDocumentReader{Reader: *strings.NewReader("image")}
				in := genai.EmbeddingRequest{Inputs: []genai.Request{{Doc: genai.Doc{Filename: "image.png", Src: src}}, {Text: "caption"}}}
				out, err := c.Embed(t.Context(), &in)
				if err != nil {
					t.Fatal(err)
				}
				if in.Inputs[0].Doc.Src != src || !reflect.DeepEqual(out.Embeddings, [][]float32{{1, 2}, {3, 4}}) {
					t.Fatalf("source changed or response invalid: %+v", out)
				}
			})
		})
		t.Run("error", func(t *testing.T) {
			t.Run("indices", func(t *testing.T) {
				for _, tc := range []embeddingIndexErrorCase{
					{"duplicate", `{"object":"list","data":[{"object":"embedding","index":0,"embedding":"AACAPw=="},{"object":"embedding","index":0,"embedding":"AACAPw=="}]}`},
					{"negative", `{"object":"list","data":[{"object":"embedding","index":-1,"embedding":"AACAPw=="},{"object":"embedding","index":1,"embedding":"AACAPw=="}]}`},
					{"out of bounds", `{"object":"list","data":[{"object":"embedding","index":2,"embedding":"AACAPw=="},{"object":"embedding","index":1,"embedding":"AACAPw=="}]}`},
				} {
					t.Run(tc.name, func(t *testing.T) {
						c, err := llamacpp.New(t.Context(), genai.ProviderOptionTransportWrapper(func(http.RoundTripper) http.RoundTripper { return embeddingDocumentTransport{body: tc.body} }))
						if err != nil {
							t.Fatal(err)
						}
						internaltest.CleanupCloser(t, c)
						out, err := c.Embed(t.Context(), &genai.EmbeddingRequest{Inputs: []genai.Request{{Text: "a"}, {Text: "b"}}})
						if _, ok := errors.AsType[*internal.BadError](err); !ok || out != nil {
							t.Fatalf("response %+v, error %v", out, err)
						}
					})
				}
			})
			t.Run("input", func(t *testing.T) {
				c, err := llamacpp.New(t.Context(), genai.ProviderOptionTransportWrapper(func(http.RoundTripper) http.RoundTripper { return compactNoTransport{t: t} }))
				if err != nil {
					t.Fatal(err)
				}
				internaltest.CleanupCloser(t, c)
				if out, err := c.Embed(t.Context(), nil); out != nil || err == nil {
					t.Fatalf("response %+v, error %v", out, err)
				}
				in := &genai.EmbeddingRequest{Inputs: []genai.Request{{Doc: genai.Doc{Filename: "image.png", Src: strings.NewReader("")}}}}
				if out, err := c.Embed(t.Context(), in); out != nil || err == nil || !strings.Contains(err.Error(), "embedding input #0") {
					t.Fatalf("response %+v, error %v", out, err)
				}
			})
		})
	})

	t.Run("EmbedRaw", func(t *testing.T) {
		t.Run("valid native controls", func(t *testing.T) {
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.Method != http.MethodPost || r.URL.Path != "/v1/embeddings" {
					t.Errorf("unexpected request %s %s", r.Method, r.URL.Path)
				}
				var req embeddingCompactWireRequest
				if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
					t.Error(err)
				}
				if req.Model != "alias" || req.EncodingFormat != "base64" || req.Normalize != -1 || len(req.Input) != 2 || string(req.Input[0]) != `[1,2]` || string(req.Input[1]) != `{"content":[{"type":"text","text":"caption"},{"type":"image_url","image_url":{"url":"data:image/png;base64,aW1hZ2U="}}]}` {
					t.Errorf("changed native controls: %+v", req)
				}
				w.Header().Set("Content-Type", "application/json")
				if _, err := io.WriteString(w, `{"model":"server-alias","object":"list","usage":{"prompt_tokens":9,"total_tokens":9},"data":[{"object":"embedding","index":1,"embedding":"AAAAQAAAwEA=","encoding_format":"base64"},{"object":"embedding","index":0,"embedding":"AABAQAAAgEA="}]}`); err != nil {
					t.Error(err)
				}
			}))
			t.Cleanup(srv.Close)
			c, err := llamacpp.New(t.Context(), genai.ProviderOptionRemote(srv.URL))
			if err != nil {
				t.Fatal(err)
			}
			internaltest.CleanupCloser(t, c)
			n := -1
			img := llamacpp.Content{Type: "image_url"}
			img.ImageURL.URL = "data:image/png;base64,aW1hZ2U="
			in := llamacpp.EmbeddingRequest{Model: "alias", Normalize: &n, Input: []llamacpp.EmbeddingInput{{Tokens: []int{1, 2}}, {Content: llamacpp.Contents{{Type: "text", Text: "caption"}, img}}}}
			var out llamacpp.EmbeddingResponse
			err = c.EmbedRaw(t.Context(), &in, &out)
			if err != nil {
				t.Fatal(err)
			}
			if len(out.Data) != 2 || out.Data[0].Index != 1 || !slices.Equal(out.Data[0].Embedding, []float32{2, 6}) || out.Data[1].Index != 0 || !slices.Equal(out.Data[1].Embedding, []float32{3, 4}) || out.Usage.PromptTokens != 9 || out.Usage.TotalTokens != 9 {
				t.Fatalf("changed values/order/usage: %+v", out)
			}
			if err := c.EmbedRaw(t.Context(), &in, nil); err == nil {
				t.Fatal("accepted nil response")
			}
			if len(in.Input[0].Tokens) != 2 || *in.Normalize != -1 {
				t.Fatal("modified request")
			}
		})
		t.Run("error", func(t *testing.T) {
			t.Run("input validation", func(t *testing.T) {
				c, err := llamacpp.New(t.Context(), genai.ProviderOptionTransportWrapper(func(http.RoundTripper) http.RoundTripper { return compactNoTransport{t: t} }))
				if err != nil {
					t.Fatal(err)
				}
				internaltest.CleanupCloser(t, c)
				n := -2
				for _, in := range []*llamacpp.EmbeddingRequest{nil, {}, {Input: []llamacpp.EmbeddingInput{{Tokens: []int{-1}}}}, {Input: []llamacpp.EmbeddingInput{{Text: "text"}}, Normalize: &n}} {
					var out llamacpp.EmbeddingResponse
					if err := c.EmbedRaw(t.Context(), in, &out); err == nil {
						t.Fatalf("accepted invalid request: %+v", in)
					}
				}
			})
			for _, tc := range []embeddingVectorErrorCase{
				{"bad base64", `"!"`}, {"empty", `""`}, {"partial float32", `"AA=="`}, {"NaN", `"AADAfw=="`}, {"infinity", `"AACAfw=="`}, {"null", `null`}, {"object", `{"base64":"AACAPw==","unexpected":true}`},
			} {
				t.Run(tc.name, func(t *testing.T) {
					srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
						w.Header().Set("Content-Type", "application/json")
						if _, err := fmt.Fprintf(w, `{"object":"list","data":[{"object":"embedding","index":0,"embedding":%s}]}`, tc.vector); err != nil {
							t.Error(err)
						}
					}))
					t.Cleanup(srv.Close)
					c, err := llamacpp.New(t.Context(), genai.ProviderOptionRemote(srv.URL))
					if err != nil {
						t.Fatal(err)
					}
					internaltest.CleanupCloser(t, c)
					out, err := c.Embed(t.Context(), &genai.EmbeddingRequest{Inputs: []genai.Request{{Text: "text"}}})
					if _, ok := errors.AsType[*internal.BadError](err); !ok || out != nil {
						t.Fatalf("got %+v, %v", out, err)
					}
				})
			}
			for _, tc := range []embeddingCompactErrorCase{
				{"envelope object", 200, `{"object":"embeddings","data":[{"object":"embedding","index":0,"embedding":"AACAPw=="}]}`, false, true, false},
				{"result count", 200, `{"object":"list","data":[]}`, false, true, false},
				{"item object", 200, `{"object":"list","data":[{"object":"vector","index":0,"embedding":"AACAPw=="}]}`, false, true, false},
				{"unknown field", 200, `{"object":"list","data":[{"object":"embedding","index":0,"embedding":"AACAPw==","unexpected":true}]}`, false, false, false},
				{"pooling none", 400, `{"error":{"code":400,"message":"Pooling type 'none' is not OAI compatible. Please use a different pooling type","type":"invalid_request_error"}}`, true, false, false},
				{"unrelated bad request", 400, `{"error":{"code":400,"message":"bad input","type":"invalid_request_error"}}`, false, false, false},
				{"projector required", 500, `{"error":{"code":500,"message":"image input is not supported - hint: if this is unexpected, you may need to provide the mmproj","type":"server_error"}}`, true, false, true},
				{"projector error for text", 500, `{"error":{"code":500,"message":"image input is not supported - hint: if this is unexpected, you may need to provide the mmproj","type":"server_error"}}`, false, false, false},
			} {
				t.Run(tc.name, func(t *testing.T) {
					srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
						w.Header().Set("Content-Type", "application/json")
						w.WriteHeader(tc.code)
						if _, err := io.WriteString(w, tc.body); err != nil {
							t.Error(err)
						}
					}))
					t.Cleanup(srv.Close)
					c, err := llamacpp.New(t.Context(), genai.ProviderOptionRemote(srv.URL))
					if err != nil {
						t.Fatal(err)
					}
					internaltest.CleanupCloser(t, c)
					in := genai.Request{Text: "text"}
					if tc.doc {
						in = genai.Request{Doc: genai.Doc{Filename: "image.png", Src: strings.NewReader("image")}}
					}
					out, err := c.Embed(t.Context(), &genai.EmbeddingRequest{Inputs: []genai.Request{in}})
					if err == nil || out != nil {
						t.Fatalf("accepted failed upstream response: %+v", out)
					}
					if _, ok := errors.AsType[*internal.BadError](err); tc.bad && !ok {
						t.Fatalf("expected BadError, got %v", err)
					}
					_, unsupported := errors.AsType[*base.ErrNotSupported](err)
					if err == nil || out != nil || unsupported != tc.unsupported {
						t.Fatalf("misclassified generic error: %+v, %v", out, err)
					}
				})
			}
			t.Run("partial response", func(t *testing.T) {
				old := internal.BeLenient
				t.Cleanup(func() { internal.BeLenient = old })
				for _, lenient := range []bool{false, true} {
					t.Run(strconv.FormatBool(lenient), func(t *testing.T) {
						internal.BeLenient = lenient
						c, err := llamacpp.New(t.Context(), genai.ProviderOptionModel("model"), genai.ProviderOptionTransportWrapper(func(http.RoundTripper) http.RoundTripper {
							return embeddingDocumentTransport{body: `{"object":"list","data":[{"object":"embedding","index":0,"embedding":"AACAPw=="},{"object":"embedding","index":1,"embedding":"AA=="}]}`}
						}))
						if err != nil {
							t.Fatal(err)
						}
						internaltest.CleanupCloser(t, c)
						var out llamacpp.EmbeddingResponse
						err = c.EmbedRaw(t.Context(), &llamacpp.EmbeddingRequest{Model: "model", Input: []llamacpp.EmbeddingInput{{Text: "a"}, {Text: "b"}}}, &out)
						if _, ok := errors.AsType[*internal.BadError](err); !ok || !reflect.ValueOf(out).IsZero() {
							t.Fatalf("response %+v, error %v", out, err)
						}
					})
				}
			})
		})
	})

	t.Run("Capabilities", func(t *testing.T) {
		c, err := llamacpp.New(t.Context(), genai.ProviderOptionTransportWrapper(func(h http.RoundTripper) http.RoundTripper {
			return testRecorder.Record(t, &roundtrippers.Header{
				Header:    http.Header{"Authorization": {"Bearer " + apiKey}},
				Transport: h,
			})
		}), genai.ProviderOptionRemote(s.lazyStart(t)))
		if c != nil {
			internaltest.CleanupCloser(t, c)
		}
		if err != nil {
			t.Fatal(err)
		}
		internaltest.TestCapabilities(t, c)
	})

	t.Run("Scoreboard", func(t *testing.T) {
		sb := llamacpp.Scoreboard().Scenarios
		models := make([]scoreboard.Model, 0, len(sb))
		for _, sc := range sb {
			for _, id := range sc.Models {
				models = append(models, scoreboard.Model{Model: id, Reason: sc.Reason})
			}
		}
		if !slices.Contains(models, scoreboard.Model{Model: embeddingModel}) {
			models = append(models, scoreboard.Model{Model: embeddingModel})
		}
		ctx := t.Context()
		smoketest.Run(t, func(t testing.TB, model scoreboard.Model, fn func(http.RoundTripper) http.RoundTripper) genai.Provider {
			embedding := model.Model == embeddingModel
			remote := s.lazyStartModel(t, model) //nolint:contextcheck // uses l.t.Context() for server lifecycle
			opts := []genai.ProviderOption{genai.ProviderOptionRemote(remote)}
			if embedding {
				opts = append(opts, genai.ProviderOptionModalities{genai.ModalityEmbedding})
			}
			if model.Model != "" {
				opts = append(opts, genai.ProviderOptionModel(model.Model))
			}
			if fn != nil {
				opts = append([]genai.ProviderOption{genai.ProviderOptionTransportWrapper(func(h http.RoundTripper) http.RoundTripper {
					return fn(&roundtrippers.Header{
						Header:    http.Header{"Authorization": {"Bearer " + apiKey}},
						Transport: h,
					})
				})}, opts...)
			}
			c2, err2 := llamacpp.New(ctx, opts...)
			if c2 != nil {
				internaltest.CleanupCloser(t, c2)
			}
			if err2 != nil {
				t.Fatal(err2)
			}
			var p genai.Provider = &internaltest.InjectOptions{
				Provider: c2,
				Opts:     []genai.GenOption{&llamacpp.GenOption{}},
			}
			if model.Reason {
				p = &internaltest.InjectOptions{
					Provider: c2,
					Opts:     []genai.GenOption{&llamacpp.GenOption{ReasoningFormat: llamacpp.ReasoningFormatDeepSeek, Thinking: true}},
				}
			}
			rec := &internaltest.RealtimeRecorder{
				Provider: p,
				T:        t,
				Target:   "ws" + strings.TrimPrefix(remote, "http") + "/v1/realtime",
				Dir:      filepath.Join("testdata", filepath.FromSlash(t.Name())),
				Updating: internaltest.UpdatingRecordings(),
				Mutation: func(opts []genai.RealtimeOption, endpoint string) []genai.RealtimeOption {
					out := slices.Clone(opts)
					for i, opt := range out {
						if o, ok := opt.(*llamacpp.RealtimeOption); ok {
							clone := *o
							clone.Endpoint = endpoint
							out[i] = &clone
							return out
						}
					}
					return append(out, &llamacpp.RealtimeOption{Endpoint: endpoint})
				},
			}
			if model.Model == "LiquidAI/LFM3-Audio-2B-GGUF/LFM3-Audio-2B-Q4_K_M.gguf" {
				// LFM3-Audio answers the spoken tool probe instead of calling the tool
				// unless the first response requires a tool call.
				return &internaltest.InjectRealtimeOptions{
					Provider: rec,
					Opts:     []genai.RealtimeOption{&llamacpp.RealtimeOption{RequireToolCall: true}},
				}
			}
			return rec
		}, models, testRecorder.Records, &smoketest.RunOptions{
			Qualify: []scoreboard.Model{{Model: embeddingModel}},
			// Gemma 4 emits reasoning content even with enable_thinking=false.
			TolerateReasoning: []string{"gemma-4"},
		})
	})

	// Note: Skipping Preferred test as llamacpp scoreboard doesn't define
	// preferred models (SOTA/Good/Cheap). Model selection is handled by
	// querying the running llama-server instance.

	t.Run("TextOutputDocInput", func(t *testing.T) {
		internaltest.TestTextOutputDocInput(t, func(t *testing.T) genai.Provider {
			c, err := llamacpp.New(t.Context(), genai.ProviderOptionTransportWrapper(func(h http.RoundTripper) http.RoundTripper {
				return testRecorder.Record(t, &roundtrippers.Header{
					Header:    http.Header{"Authorization": {"Bearer " + apiKey}},
					Transport: h,
				})
			}), genai.ProviderOptionRemote(s.lazyStart(t)), genai.ModelCheap)
			if c != nil {
				internaltest.CleanupCloser(t, c)
			}
			if err != nil {
				t.Fatal(err)
			}
			return c
		})
	})

	t.Run("ListModels", func(t *testing.T) {
		ctx := t.Context()
		c, err := llamacpp.New(ctx, genai.ProviderOptionTransportWrapper(func(h http.RoundTripper) http.RoundTripper {
			return testRecorder.Record(t, &roundtrippers.Header{
				Header:    http.Header{"Authorization": {"Bearer " + apiKey}},
				Transport: h,
			})
		}), genai.ProviderOptionRemote(s.lazyStart(t)))
		if c != nil {
			internaltest.CleanupCloser(t, c)
		}
		if err != nil {
			t.Fatal(err)
		}
		genaiModels, err := c.ListModels(ctx)
		if err != nil {
			t.Fatal(err)
		}
		if len(genaiModels) != 1 {
			t.Fatalf("unexpected: %#v", genaiModels)
		}
	})

	// Run this at the end so there would be non-zero values.
	t.Run("Metrics", func(t *testing.T) {
		ctx := t.Context()
		c, err := llamacpp.New(ctx, genai.ProviderOptionTransportWrapper(func(h http.RoundTripper) http.RoundTripper {
			return testRecorder.Record(t, &roundtrippers.Header{
				Header:    http.Header{"Authorization": {"Bearer " + apiKey}},
				Transport: h,
			})
		}), genai.ProviderOptionRemote(s.lazyStart(t)))
		if c != nil {
			internaltest.CleanupCloser(t, c)
		}
		if err != nil {
			t.Fatal(err)
		}
		m := llamacpp.Metrics{}
		if err := c.GetMetrics(ctx, &m); err != nil {
			t.Fatal(err)
		}
		t.Logf("Metrics: %+v", m)
	})
}

type lazyServer struct {
	t      testing.TB
	apiKey string

	mu      sync.Mutex
	servers map[string]string // base model path -> URL
}

// lazyStart starts the default server, picking the first model with image
// support since TextOutputDocInput needs it. Scenario ordering may change
// after -update-scoreboard sorts by reasoning first.
func (l *lazyServer) lazyStart(t testing.TB) string {
	sb := llamacpp.Scoreboard()
	for i := range sb.Scenarios {
		sc := &sb.Scenarios[i]
		if _, ok := sc.In[scoreboard.ModalityImage]; ok {
			return l.lazyStartModel(t, scoreboard.Model{Model: sc.Models[0], Reason: sc.Reason})
		}
	}
	sc := llamacpp.Scoreboard().Scenarios[0]
	return l.lazyStartModel(t, scoreboard.Model{Model: sc.Models[0], Reason: sc.Reason})
}

// ensureExe retrieves the cached llama-server binary for the default version.
func (l *lazyServer) ensureExe(ctx context.Context) (string, error) {
	cache, err := filepath.Abs("testdata/tmp")
	if err != nil {
		return "", err
	}
	if err := os.MkdirAll(cache, 0o755); err != nil {
		return "", err
	}
	return llamacppsrv.DownloadVersion(ctx, cache, llamacppsrv.Version)
}

// lazyStartModel starts a server for the given model key, reusing an existing one if already running.
func (l *lazyServer) lazyStartModel(t testing.TB, model scoreboard.Model) string {
	if model.Model == "" {
		// Empty model is used by smoketest.Run to read the static Scoreboard();
		// no server is needed.
		return "http://127.0.0.1:0"
	}
	// Replay cassettes unless the run records them.
	if os.Getenv("RECORD") == "" {
		return "http://127.0.0.1:0"
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	if l.servers == nil {
		l.servers = make(map[string]string)
	}
	if u, ok := l.servers[model.Model]; ok {
		return u
	}
	t.Logf("Starting server for %s", model.Model)
	exe, err := l.ensureExe(t.Context())
	if err != nil {
		t.Fatal(err)
	}
	parts := strings.Split(model.Model, "/")
	// Use the parent context for server lifecycle management, but report
	// startup failures to the subtest that requested the server.
	srv := startServerTest(t, l.t.Context(), exe, parts[0], parts[1], parts[2], l.apiKey, model.Model == embeddingModel)
	u := srv.URL()
	l.servers[model.Model] = u
	l.t.Cleanup(func() {
		if err := srv.Close(); err != nil && !errors.Is(err, context.Canceled) {
			// llama-server may exit with code 1 on SIGINT; ignore ExitError
			// since we intentionally stopped it.
			if _, ok := errors.AsType[*exec.ExitError](err); !ok {
				l.t.Error(err)
			}
		}
	})
	return u
}

func startServerTest(t testing.TB, ctx context.Context, exe, author, repo, modelfile, apiKey string, embedding bool) *llamacppsrv.Server {
	cache, err := filepath.Abs("testdata/tmp")
	if err != nil {
		t.Fatal(err)
	}
	// Use llama-server's built-in HuggingFace download; mmproj is auto-detected.
	extraArgs := []string{
		"-hf", author + "/" + repo,
		"-hff", modelfile,
		"--api-key", apiKey,
		"--cors-origins", "",
		"--no-cors-credentials",
		"--no-ui",
		"--no-slots",
	}
	if embedding {
		extraArgs = append(extraArgs, "--embeddings", "--pooling", "mean")
	} else {
		extraArgs = append(extraArgs,
			"--jinja",
			"--flash-attn", "on",
			"--ctx-size", "32768", // Thinking models can generate a lot of tokens. Fails on TopLogprob and Citations-text-plain.
			"--cache-type-k", "q8_0",
			"--cache-type-v", "q8_0",
			"--reasoning-preserve",
			"--spec-default",
			"--no-warmup",
			// "--spec-type", "draft-mtp",
			"--parallel", "4",
			"--kv-unified",
		)
	}
	// Allocate an ephemeral port to avoid dual-stack conflicts when running
	// multiple servers (e.g. "localhost:8080" can bind on both IPv4 and IPv6).
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	hostPort := ln.Addr().String()
	_ = ln.Close()
	l := internaltest.LogFile(t, cache, "llama-server.log")
	srv, err := llamacppsrv.New(ctx, exe, &llamacppsrv.Config{HostPort: hostPort, Args: extraArgs, LogOutput: l})
	if err != nil {
		t.Fatal(err)
	}
	return srv
}

func init() {
	internal.BeLenient = false
}

func TestClientSystemOne(t *testing.T) {
	q := genai.Questions{
		"billing": {Type: genai.QuestionNoul, Instructions: genai.Text("Is this about billing?")},
		"route":   {Type: genai.QuestionChoice, Instructions: genai.Text("Which team?"), Choice: map[string]genai.DecisionContent{"billing": nil, "support": nil}},
		"urgency": {Type: genai.QuestionScore, Instructions: genai.Text("How urgent?"), Score: []genai.DecisionContent{genai.Text("low"), genai.Text("high")}},
	}
	const response = `{"model":"kev","answers":{"billing":{"type":"noul","noul":0.9},"route":{"type":"choice","choice":"billing","confidence":0.8,"probabilities":{"billing":0.9,"support":0.1}},"urgency":{"type":"score","score":0.75,"confidence":0.5,"legend":{"0":"low","1":"high"},"probabilities":{"0":0.25,"1":0.75}}},"usage":{"input_tokens":42,"output_tokens":0}}`
	t.Run("SystemOneRaw", func(t *testing.T) {
		t.Run("limits at request boundary", func(t *testing.T) {
			srv := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) { t.Error("unexpected HTTP request") }))
			t.Cleanup(srv.Close)
			c, err := llamacpp.New(t.Context(), genai.ProviderOptionRemote(srv.URL))
			if err != nil {
				t.Fatal(err)
			}
			internaltest.CleanupCloser(t, c)
			req := llamacpp.SystemOneRequest{State: genai.Text("state")}
			qs := genai.Questions{"q": {Type: genai.QuestionScore, Instructions: genai.Text("rate"), Score: []genai.DecisionContent{genai.Text("only")}}}
			req.Questions = qs
			err = c.SystemOneRaw(t.Context(), &req, &llamacpp.SystemOneResponse{})
			if err == nil || !strings.Contains(err.Error(), "2 to 10 levels are required") {
				t.Fatalf("missing native boundary limit: %v", err)
			}
		})
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			var req struct {
				Model     string                     `json:"model"`
				State     map[string]int             `json:"state"`
				Images    []string                   `json:"images"`
				Questions map[string]json.RawMessage `json:"questions"`
			}
			if r.URL.Path != "/v1/systemone" {
				t.Errorf("unexpected path %s", r.URL.Path)
			}
			if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
				t.Error(err)
			}
			if req.Model != "kev" || req.State["ticket"] != 42 || len(req.Images) != 1 || len(req.Questions) != 3 {
				t.Errorf("unexpected request: %+v", req)
			}
			w.Header().Set("Content-Type", "application/json")
			if _, err := w.Write([]byte(response)); err != nil {
				t.Error(err)
			}
		}))
		t.Cleanup(srv.Close)
		c, err := llamacpp.New(t.Context(), genai.ProviderOptionRemote(srv.URL))
		if err != nil {
			t.Fatal(err)
		}
		out := &llamacpp.SystemOneResponse{}
		if err := c.SystemOneRaw(t.Context(), &llamacpp.SystemOneRequest{Model: "kev", State: genai.Object{"ticket": 42}, Questions: q, Images: []string{"data:image/png;base64,aW1hZ2U="}}, out); err != nil {
			t.Fatal(err)
		}
		if out.Model != "kev" || out.Answers["billing"].Noul != 0.9 {
			t.Errorf("unexpected response: %+v", out)
		}
	})
	t.Run("error", func(t *testing.T) {
		for _, tc := range []struct {
			name, body string
			status     int
		}{
			{"missing answers", `{"model":"kev","answers":{},"usage":{"input_tokens":1,"output_tokens":0}}`, 200},
			{"wrong answer type", `{"model":"kev","answers":{"billing":{"type":"choice","choice":"billing"}},"usage":{"input_tokens":1,"output_tokens":0}}`, 200},
		} {
			t.Run(tc.name, func(t *testing.T) {
				srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
					w.Header().Set("Content-Type", "application/json")
					w.WriteHeader(tc.status)
					if _, err := w.Write([]byte(tc.body)); err != nil {
						t.Error(err)
					}
				}))
				t.Cleanup(srv.Close)
				c, err := llamacpp.New(t.Context(), genai.ProviderOptionRemote(srv.URL))
				if err != nil {
					t.Fatal(err)
				}
				_, err = c.SystemOne(t.Context(), &genai.SystemOneRequest{State: genai.Text("x"), Questions: q})
				if err == nil {
					t.Fatal("expected an error")
				}
				if tc.status == 501 {
					var api *llamacpp.ErrorResponse
					if !errors.As(err, &api) || api.ErrorVal.Code != 501 {
						t.Errorf("unexpected error: %v", err)
					}
				}
			})
		}
	})
}

func TestClientChronologicalTemplate(t *testing.T) {
	msgs := genai.Messages{
		genai.NewTextMessage("first"), genai.NewTextMessage("correction"),
		{Replies: []genai.Reply{{ToolCall: genai.ToolCall{ID: "A", Name: "status", Arguments: `{}`}}}},
		genai.NewTextMessage("intervening"),
		{Replies: []genai.Reply{{ToolCall: genai.ToolCall{ID: "B", Name: "status", Arguments: `{}`}}}},
		{ToolCallResults: []genai.ToolCallResult{{ID: "B", Name: "status", Result: "waiting"}}},
		{ToolCallResults: []genai.ToolCallResult{{ID: "A", Name: "status", Result: "running"}}},
	}
	before, err := json.Marshal(msgs)
	if err != nil {
		t.Fatal(err)
	}
	var template []llamacpp.Message
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		switch r.URL.Path {
		case "/apply-template":
			var req struct {
				Messages []llamacpp.Message `json:"messages"`
			}
			if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
				t.Error(err)
				w.WriteHeader(http.StatusBadRequest)
				return
			}
			template = req.Messages
			if _, err := io.WriteString(w, `{"prompt":"server-rendered chronological prompt"}`); err != nil {
				t.Error(err)
			}
		case "/completions":
			var req llamacpp.CompletionRequest
			if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
				t.Error(err)
				w.WriteHeader(http.StatusBadRequest)
				return
			}
			if req.Prompt != "server-rendered chronological prompt" {
				t.Errorf("prompt=%q", req.Prompt)
			}
			w.WriteHeader(http.StatusBadRequest)
			if _, err := io.WriteString(w, `{"error":{"code":400,"message":"template chronology rejected","type":"invalid_request_error"}}`); err != nil {
				t.Error(err)
			}
		default:
			t.Errorf("unexpected request: %s", r.URL.Path)
			w.WriteHeader(http.StatusInternalServerError)
		}
	}))
	t.Cleanup(srv.Close)
	c, err := llamacpp.New(t.Context(), genai.ProviderOptionRemote(srv.URL), genai.ProviderOptionModel("test"))
	if err != nil {
		t.Fatal(err)
	}
	internaltest.CleanupCloser(t, c)
	if _, err := c.Completion(t.Context(), msgs); err == nil || !strings.Contains(err.Error(), "template chronology rejected") {
		t.Fatalf("provider error=%v", err)
	}
	if len(template) != 7 || template[0].Role != "user" || template[1].Role != "user" || template[2].ToolCalls[0].ID != "A" || template[3].Content[0].Text != "intervening" || template[4].ToolCalls[0].ID != "B" || template[5].ToolCallID != "B" || template[5].Content[0].Text != "waiting" || template[6].ToolCallID != "A" || template[6].Content[0].Text != "running" {
		t.Fatalf("template history changed: %+v", template)
	}
	after, err := json.Marshal(msgs)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(before, after) {
		t.Fatal("template mutated input")
	}
}

// compactNoTransport asserts invalid native requests fail before network I/O.
type compactNoTransport struct{ t *testing.T }

func (c compactNoTransport) RoundTrip(*http.Request) (*http.Response, error) {
	c.t.Fatal("invalid compact request reached transport")
	return nil, errors.New("unexpected compact request")
}

type embeddingCompactWireRequest struct {
	Model          string
	Input          []json.RawMessage
	EncodingFormat string `json:"encoding_format"`
	Normalize      int    `json:"embd_normalize"`
}

type embeddingVectorErrorCase struct {
	name   string
	vector string
}

type embeddingCompactErrorCase struct {
	name        string
	code        int
	body        string
	unsupported bool
	bad         bool
	doc         bool
}

type embeddingDocumentReader struct{ strings.Reader }

func (*embeddingDocumentReader) Seek(int64, int) (int64, error) { return 0, errors.New("unseekable") }

type embeddingDocumentTransport struct{ body string }

func (e embeddingDocumentTransport) RoundTrip(r *http.Request) (*http.Response, error) {
	return &http.Response{StatusCode: http.StatusOK, Header: http.Header{"Content-Type": {"application/json"}}, Body: io.NopCloser(strings.NewReader(e.body)), Request: r}, nil
}

type embeddingIndexErrorCase struct{ name, body string }
