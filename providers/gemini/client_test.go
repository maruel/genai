// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Tests for the Gemini provider client.

package gemini_test

import (
	"context"
	_ "embed"
	"encoding/json"
	"errors"
	"io"
	"iter"
	"math"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"reflect"
	"slices"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/coder/websocket"
	"github.com/google/go-cmp/cmp"
	"github.com/maruel/roundtrippers"
	"gopkg.in/dnaeon/go-vcr.v4/pkg/recorder"

	"github.com/maruel/genai"
	"github.com/maruel/genai/base"
	"github.com/maruel/genai/internal"
	"github.com/maruel/genai/internal/internaltest"
	"github.com/maruel/genai/internal/realtimetest"
	"github.com/maruel/genai/providers/gemini"
	"github.com/maruel/genai/scoreboard"
	"github.com/maruel/genai/smoke/smoketest"
)

const liveEndpoint = "wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent"

var thinkingModels = []string{
	"gemini-3.8-flash",
	"gemini-flash-lite-latest",
	"gemini-pro-latest",
}

func getClientInner(t *testing.T, model string, modalities genai.Modalities, preloadedModels []genai.Model, fn func(http.RoundTripper) http.RoundTripper) (genai.Provider, error) {
	var opts []genai.ProviderOption
	if model != "" {
		opts = append(opts, genai.ProviderOptionModel(model))
	}
	key := internaltest.GetEnv("GEMINI_API_KEY")
	if key == "" {
		key = "<insert_api_key_here>"
	}
	opts = append(opts, genai.ProviderOptionAPIKey(key))
	if len(modalities) > 0 {
		opts = append(opts, genai.ProviderOptionModalities(modalities))
	}
	if len(preloadedModels) > 0 {
		opts = append(opts, genai.ProviderOptionPreloadedModels(preloadedModels))
	}
	if fn != nil {
		opts = append([]genai.ProviderOption{genai.ProviderOptionTransportWrapper(fn)}, opts...)
	}
	c, err := gemini.New(t.Context(), opts...)
	if c != nil {
		internaltest.CleanupCloser(t, c)
	}
	return c, err
}

func TestClient(t *testing.T) {
	t.Run("ConnectRealtime", func(t *testing.T) {
		t.Run("unsupported option type", func(t *testing.T) {
			s, err := liveClient(t).ConnectRealtime(t.Context(), genai.GenOptionSeed(1))
			if _, ok := errors.AsType[*base.ErrNotSupported](err); s != nil || !ok {
				t.Fatalf("unsupported option: %+v, %v", s, err)
			}
		})
		t.Run("Conversation", func(t *testing.T) {
			received := make(chan []map[string]any, 1)
			endpoint := livePeer(t, func(ctx context.Context, ws *websocket.Conn, setup map[string]any) {
				s := setup["setup"].(map[string]any)
				if s["model"] != "models/gemini-3.8-live" {
					t.Errorf("model: %v", s)
				}
				g := s["generationConfig"].(map[string]any)
				if !reflect.DeepEqual(g["responseModalities"], []any{"AUDIO"}) || len(g) != 1 {
					t.Errorf("unexpected generation config: %v", g)
				}
				ri := s["realtimeInputConfig"].(map[string]any)
				if ri["turnCoverage"] != "TURN_INCLUDES_ONLY_ACTIVITY" || ri["activityHandling"] != "START_OF_ACTIVITY_INTERRUPTS" {
					t.Errorf("input config: %v", ri)
				}
				trans := s["inputAudioTranscription"].(map[string]any)
				if !reflect.DeepEqual(trans["languageCodes"], []any{"en-US"}) {
					t.Errorf("transcription: %v", trans)
				}
				for _, key := range []string{"proactivity", "enableAffectiveDialog", "thinkingConfig"} {
					if _, ok := s[key]; ok {
						t.Errorf("unexpected %s", key)
					}
				}
				messages := make([]map[string]any, 4)
				for i := range messages {
					messages[i] = realtimetest.Read(t, ctx, ws)
				}
				received <- messages
				realtimetest.Write(t, ctx, ws, `{"serverContent":{"modelTurn":{"parts":[{"inlineData":{"mimeType":"audio/pcm;rate=24000","data":"AQACAA=="}},{"text":"answer"}]},"inputTranscription":{"text":"hello "},"outputTranscription":{"text":"good "},"generationComplete":true}}`)
				realtimetest.Write(t, ctx, ws, `{"serverContent":{"inputTranscription":{"text":"world","finished":true},"outputTranscription":{"text":"day","finished":true},"turnComplete":true},"usageMetadata":{"promptTokenCount":3,"responseTokenCount":5,"totalTokenCount":8},"goAway":{"timeLeft":"30s"},"sessionResumptionUpdate":{"newHandle":"resume","resumable":true}}`)
				_, _, _ = ws.Read(ctx)
			})
			s, err := liveClient(t).ConnectRealtime(t.Context(), &genai.RealtimeOptionAudio{InputTranscription: true, OutputTranscription: true}, &gemini.RealtimeOption{Endpoint: endpoint, InputLanguages: []string{"en-US"}, SessionResumption: true})
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() { _ = s.Close() })
			info := s.Info()
			if info.ID != "live-test" || info.InputFormat != (genai.AudioFormat{SampleRate: 16000}) || info.OutputFormat != (genai.AudioFormat{SampleRate: 24000}) || info.TurnDetection != genai.RealtimeTurnServer || info.Capabilities.CancelResponse {
				t.Fatalf("info: %+v", info)
			}
			inputs := []genai.RealtimeInput{genai.RealtimeInputContent{Text: "context"}, genai.RealtimeInputContent{Text: "question", Generate: true}, genai.RealtimeInputContent{Audio: []byte{1, 0}}, genai.RealtimeInputContent{Commit: true}}
			for i := range inputs {
				if err := s.Send(t.Context(), inputs[i]); err != nil {
					t.Fatal(err)
				}
			}
			messages := <-received
			passive := messages[0]["clientContent"].(map[string]any)
			if passive["turnComplete"] != false || passive["turns"].([]any)[0].(map[string]any)["role"] != "user" {
				t.Fatalf("passive context: %v", messages[0])
			}
			cc := messages[1]["clientContent"].(map[string]any)
			if cc["turnComplete"] != true {
				t.Errorf("text did not complete turn: %v", cc)
			}
			turns := cc["turns"].([]any)
			if turns[0].(map[string]any)["role"] != "user" {
				t.Errorf("text changed role: %v", cc)
			}
			if messages[2]["realtimeInput"].(map[string]any)["audio"].(map[string]any)["mimeType"] != "audio/pcm;rate=16000" {
				t.Errorf("audio: %v", messages[2])
			}
			if messages[3]["realtimeInput"].(map[string]any)["audioStreamEnd"] != true {
				t.Errorf("stream end: %v", messages[3])
			}
			want := []genai.RealtimeEventKind{genai.RealtimeEventContentAudio, genai.RealtimeEventContentText, genai.RealtimeEventTranscriptInput, genai.RealtimeEventTranscriptOutput, genai.RealtimeEventGenerationDone, genai.RealtimeEventTranscriptInput, genai.RealtimeEventTranscriptOutput, genai.RealtimeEventTurnDone, genai.RealtimeEventUsage, genai.RealtimeEventProvider, genai.RealtimeEventProvider}
			for i, kind := range want {
				e := realtimetest.Receive(t, s, kind)
				if e.ResponseID != "" || e.ItemID != "" {
					t.Fatalf("event %d: %+v", i, e)
				}
				switch i {
				case 0:
					if !reflect.DeepEqual(e.Content.Audio, []byte{1, 0, 2, 0}) {
						t.Errorf("audio: %v", e.Content.Audio)
					}
				case 5:
					if !e.Content.Final || e.Content.Text != "hello world" {
						t.Errorf("final transcript: %+v", e)
					}
				case 6:
					if !e.Content.Final || e.Content.Text != "good day" {
						t.Errorf("final transcript: %+v", e)
					}
				case 8:
					if e.Usage.TotalTokens != 8 {
						t.Errorf("usage: %+v", e)
					}
				case 9:
					if p, ok := e.ProviderEvent.(*gemini.GoAway); !ok || p.TimeLeft != 30*time.Second {
						t.Errorf("goaway: %+v", e)
					}
				case 10:
					if p, ok := e.ProviderEvent.(*gemini.Resumption); !ok || p.Handle != "resume" {
						t.Errorf("resumption: %+v", e)
					}
				}
			}
		})
		t.Run("ThinkingLevel", func(t *testing.T) {
			endpoint := livePeer(t, func(ctx context.Context, ws *websocket.Conn, setup map[string]any) {
				g := setup["setup"].(map[string]any)["generationConfig"].(map[string]any)
				if c, ok := g["thinkingConfig"].(map[string]any); !ok || c["thinkingLevel"] != "high" || len(g) != 2 {
					t.Errorf("generation config: %v", g)
				}
			})
			s, err := liveClient(t).ConnectRealtime(t.Context(), &gemini.RealtimeOption{Endpoint: endpoint, ThinkingLevel: gemini.ThinkingLevelHigh})
			if err != nil {
				t.Fatal(err)
			}
			realtimetest.Close(t, s)
		})
		t.Run("valid", func(t *testing.T) {
			t.Run("InteractionStatus", func(t *testing.T) {
				endpoint := livePeer(t, func(ctx context.Context, ws *websocket.Conn, setup map[string]any) {
					declarations := setup["setup"].(map[string]any)["tools"].([]any)[0].(map[string]any)["functionDeclarations"].([]any)
					if got := declarations[0].(map[string]any)["behavior"]; got != "NON_BLOCKING" {
						t.Errorf("behavior=%v", got)
					}
					realtimetest.Write(t, ctx, ws, `{"serverContent":{"generationComplete":true,"turnComplete":true,"interactionStatus":"IN_PROGRESS"}}`)
					realtimetest.Write(t, ctx, ws, `{"toolCall":{"functionCalls":[{"id":"call","name":"lookup","args":{}}]}}`)
					if _, ok := realtimetest.Read(t, ctx, ws)["toolResponse"]; !ok {
						t.Error("missing tool response")
					}
					realtimetest.Write(t, ctx, ws, `{"serverContent":{"generationComplete":true,"turnComplete":true,"interactionStatus":"IDLE"}}`)
				})
				s, err := liveClientForModel(t, "gemini-3.8-live-extended-thinking").ConnectRealtime(t.Context(), &gemini.RealtimeOption{Endpoint: endpoint}, &genai.RealtimeOptionTools{Tools: []genai.ToolDef{{Name: "lookup", Description: "lookup", InputSchemaOverride: json.RawMessage(`{"type":"object"}`)}}})
				if err != nil {
					t.Fatal(err)
				}
				realtimetest.Close(t, s)
				next, stop := iter.Pull2(s.Receive(t.Context()))
				t.Cleanup(stop)
				e, err, ok := next()
				if !ok || err != nil {
					t.Fatalf("event=%+v err=%v ok=%v", e, err, ok)
				}
				if p, ok := e.ProviderEvent.(*gemini.InteractionStatus); !ok || p.Status != gemini.InteractionStatusInProgress || e.Kind != genai.RealtimeEventProvider {
					t.Fatalf("first event=%+v", e)
				}
				if e, err, ok = next(); !ok || err != nil || e.Kind != genai.RealtimeEventGenerationDone {
					t.Fatalf("generation event=%+v err=%v ok=%v", e, err, ok)
				}
				if e, err, ok = next(); !ok || err != nil || e.Kind != genai.RealtimeEventContentToolCalls || len(e.Content.ToolCalls) != 1 {
					t.Fatalf("tool event=%+v err=%v ok=%v", e, err, ok)
				}
				if err := s.Send(t.Context(), genai.RealtimeInputContent{ToolResults: []genai.ToolCallResult{{ID: "call", Name: "lookup", Result: "ok"}}}); err != nil {
					t.Fatal(err)
				}
				if e, err, ok = next(); !ok || err != nil {
					t.Fatalf("event=%+v err=%v ok=%v", e, err, ok)
				}
				if p, ok := e.ProviderEvent.(*gemini.InteractionStatus); !ok || p.Status != gemini.InteractionStatusIdle || e.Kind != genai.RealtimeEventProvider {
					t.Fatalf("idle event=%+v", e)
				}
				if e, err, ok = next(); !ok || err != nil || e.Kind != genai.RealtimeEventGenerationDone {
					t.Fatalf("answer generation event=%+v err=%v ok=%v", e, err, ok)
				}
				if e, err, ok = next(); !ok || err != nil || e.Kind != genai.RealtimeEventTurnDone {
					t.Fatalf("turn event=%+v err=%v ok=%v", e, err, ok)
				}
			})
		})

		t.Run("ToolCorrelation", func(t *testing.T) {
			result := make(chan map[string]any, 1)
			endpoint := livePeer(t, func(ctx context.Context, ws *websocket.Conn, setup map[string]any) {
				s := setup["setup"].(map[string]any)
				declarations := s["tools"].([]any)[0].(map[string]any)["functionDeclarations"].([]any)
				if declarations[0].(map[string]any)["behavior"] != "BLOCKING" {
					t.Error("tools must block")
				}
				realtimetest.Write(t, ctx, ws, `{"toolCall":{"functionCalls":[{"name":"lookup","args":{"q":"a"}},{"id":"provided","name":"lookup","args":{}}]}}`)
				result <- realtimetest.Read(t, ctx, ws)
				realtimetest.Write(t, ctx, ws, `{"toolCall":{"functionCalls":[{"id":"cancel-me","name":"lookup","args":{}},{"id":"still-pending","name":"lookup","args":{}}]},"toolCallCancellation":{"ids":["cancel-me"]}}`)
				result <- realtimetest.Read(t, ctx, ws)
				_, _, _ = ws.Read(ctx)
			})
			cfg := []genai.RealtimeOption{&genai.RealtimeOptionTools{Tools: []genai.ToolDef{{Name: "lookup", Description: "lookup", InputSchemaOverride: json.RawMessage(`{"type":"object"}`)}}}}
			s, err := liveClient(t).ConnectRealtime(t.Context(), append([]genai.RealtimeOption{&gemini.RealtimeOption{Endpoint: endpoint}}, cfg...)...)
			if err != nil {
				t.Fatal(err)
			}
			closed := false
			t.Cleanup(func() {
				if !closed {
					if err := s.Close(); err != nil {
						t.Error(err)
					}
				}
			})
			e := realtimetest.Receive(t, s, genai.RealtimeEventContentToolCalls)
			if len(e.Content.ToolCalls) != 2 || e.Content.ToolCalls[0].ID == "" || e.Content.ToolCalls[0].ID == e.Content.ToolCalls[1].ID {
				t.Fatalf("calls: %+v", e)
			}
			bad := genai.RealtimeInputContent{ToolResults: []genai.ToolCallResult{{ID: e.Content.ToolCalls[0].ID, Name: "lookup", Result: "ok"}, {ID: "unknown", Name: "lookup", Result: "ok"}}}
			if err := s.Send(t.Context(), bad); err == nil {
				t.Fatal("accepted unmatched result batch")
			}
			good := genai.RealtimeInputContent{ToolResults: []genai.ToolCallResult{{ID: e.Content.ToolCalls[0].ID, Name: "lookup", Result: "ok"}, {ID: "provided", Name: "lookup", Result: `{"found":true}`}}}
			if err := s.Send(t.Context(), good); err != nil {
				t.Fatal(err)
			}
			responses := (<-result)["toolResponse"].(map[string]any)["functionResponses"].([]any)
			first := responses[0].(map[string]any)
			if _, ok := first["id"]; ok {
				t.Error("synthetic ID leaked to provider")
			}
			if first["response"].(map[string]any)["response"] != "ok" || responses[1].(map[string]any)["id"] != "provided" {
				t.Errorf("results: %v", responses)
			}
			if err := s.Send(t.Context(), good); err == nil {
				t.Fatal("accepted duplicate results")
			}
			call := realtimetest.Receive(t, s, genai.RealtimeEventContentToolCalls)
			// Translation has already cancelled the call, while the cancellation
			// event remains queued. This deterministically exercises the host race.
			late := genai.ToolCallResult{ID: "cancel-me", Name: "lookup", Result: "late"}
			valid := genai.ToolCallResult{ID: "still-pending", Name: "lookup", Result: "ok"}
			for _, batch := range [][]genai.ToolCallResult{{late}, {valid, late}, {late, valid}} {
				err := s.Send(t.Context(), genai.RealtimeInputContent{ToolResults: batch})
				cancelErr, ok := errors.AsType[*genai.RealtimeToolCallCancelledError](err)
				if !ok || !reflect.DeepEqual(cancelErr.ToolCallIDs, []string{"cancel-me"}) {
					t.Fatalf("cancelled batch error = %v", err)
				}
			}
			for _, bad := range []genai.ToolCallResult{{ID: "unknown", Name: "lookup", Result: "ok"}, {ID: "cancel-me", Name: "wrong", Result: "ok"}} {
				for _, batch := range [][]genai.ToolCallResult{{bad}, {late, bad}, {bad, late}} {
					err := s.Send(t.Context(), genai.RealtimeInputContent{ToolResults: batch})
					if _, ok := errors.AsType[*genai.RealtimeToolCallCancelledError](err); err == nil || ok {
						t.Fatalf("unmatched batch error = %v", err)
					}
				}
			}
			if err := s.Send(t.Context(), genai.RealtimeInputContent{ToolResults: []genai.ToolCallResult{valid}}); err != nil {
				t.Fatal(err)
			}
			remaining := (<-result)["toolResponse"].(map[string]any)["functionResponses"].([]any)
			if len(remaining) != 1 || remaining[0].(map[string]any)["id"] != "still-pending" {
				t.Fatalf("rejected batch mutated wire or correlation: %+v", remaining)
			}
			cancelled := realtimetest.Receive(t, s, genai.RealtimeEventProvider)
			cancel, ok := cancelled.ProviderEvent.(*gemini.ToolCallCancellation)
			if !ok || cancelled.Kind != genai.RealtimeEventProvider || cancelled.ProviderEvent.RealtimeEventName() != "toolCallCancellation" || !reflect.DeepEqual(cancel.IDs, []string{"cancel-me"}) {
				t.Fatalf("cancellation: %+v %+v", call, cancelled)
			}
			err = s.Send(t.Context(), genai.RealtimeInputContent{ToolResults: []genai.ToolCallResult{late}})
			cancelErr, ok := errors.AsType[*genai.RealtimeToolCallCancelledError](err)
			if !ok || !reflect.DeepEqual(cancelErr.ToolCallIDs, []string{"cancel-me"}) {
				t.Fatalf("late cancelled result: %v", err)
			}
			err = s.Close()
			closed = true
			if err != nil {
				t.Fatal(err)
			}
			if err := s.Send(t.Context(), genai.RealtimeInputContent{ToolResults: []genai.ToolCallResult{late}}); err == nil {
				t.Fatalf("closed transport misclassified as cancellation: %v", err)
			} else if _, ok := errors.AsType[*genai.RealtimeToolCallCancelledError](err); ok {
				t.Fatalf("closed transport misclassified as cancellation: %v", err)
			}
		})
		t.Run("CancellationAfterResult", func(t *testing.T) {
			for _, reusedID := range []string{"answered", "unknown"} {
				t.Run(reusedID, func(t *testing.T) {
					endpoint := livePeer(t, func(ctx context.Context, ws *websocket.Conn, _ map[string]any) {
						realtimetest.Write(t, ctx, ws, `{"toolCall":{"functionCalls":[{"id":"answered","name":"lookup","args":{}}]}}`)
						if _, ok := realtimetest.Read(t, ctx, ws)["toolResponse"]; !ok {
							t.Error("missing valid result")
						}
						realtimetest.Write(t, ctx, ws, `{"toolCallCancellation":{"ids":["answered","unknown"]}}`)
						if _, ok := realtimetest.Read(t, ctx, ws)["clientContent"]; !ok {
							t.Error("unmatched cancelled result wrote to wire")
						}
						realtimetest.Write(t, ctx, ws, `{"toolCall":{"functionCalls":[{"id":"`+reusedID+`","name":"lookup","args":{}}]}}`)
						_, _, _ = ws.Read(ctx)
					})
					s, err := liveClient(t).ConnectRealtime(t.Context(), &genai.RealtimeOptionTools{Tools: []genai.ToolDef{{Name: "lookup", Description: "lookup", InputSchemaOverride: json.RawMessage(`{"type":"object"}`)}}}, &gemini.RealtimeOption{Endpoint: endpoint})
					if err != nil {
						t.Fatal(err)
					}
					t.Cleanup(func() {
						if err := s.Close(); err != nil {
							t.Error(err)
						}
					})
					realtimetest.Receive(t, s, genai.RealtimeEventContentToolCalls)
					if err := s.Send(t.Context(), genai.RealtimeInputContent{ToolResults: []genai.ToolCallResult{{ID: "answered", Name: "lookup", Result: "ok"}}}); err != nil {
						t.Fatal(err)
					}
					realtimetest.Receive(t, s, genai.RealtimeEventProvider)
					for _, id := range []string{"answered", "unknown"} {
						err := s.Send(t.Context(), genai.RealtimeInputContent{ToolResults: []genai.ToolCallResult{{ID: id, Name: "lookup", Result: "late"}}})
						if _, ok := errors.AsType[*genai.RealtimeToolCallCancelledError](err); err == nil || ok {
							t.Fatalf("unknown cancellation provenance misclassified: %v", err)
						}
					}
					if err := s.Send(t.Context(), genai.RealtimeInputContent{Text: "continue"}); err != nil {
						t.Fatal(err)
					}
					next, stop := iter.Pull2(s.Receive(t.Context()))
					e, err, ok := next()
					stop()
					if !ok {
						t.Fatal("realtime session ended")
					}
					if err == nil || e.Kind == genai.RealtimeEventContentToolCalls {
						t.Fatalf("cancelled ID reuse admitted: %+v, %v", e, err)
					}
				})
			}
		})
		t.Run("CancelledToolHistory", func(t *testing.T) {
			for _, failure := range []string{"reused ID", "entry limit", "metadata limit"} {
				t.Run(failure, func(t *testing.T) {
					endpoint := livePeer(t, func(ctx context.Context, ws *websocket.Conn, _ map[string]any) {
						count := 1
						if failure == "entry limit" {
							count = 1024
						}
						calls := make([]map[string]any, count)
						ids := make([]string, count)
						for i := range calls {
							ids[i] = "cancel-" + strconv.Itoa(i)
							if failure == "metadata limit" {
								ids[i] = strings.Repeat("a", 64*1024-len("lookup"))
							}
							calls[i] = map[string]any{"id": ids[i], "name": "lookup", "args": map[string]any{}}
						}
						data, err := json.Marshal(map[string]any{"toolCall": map[string]any{"functionCalls": calls}, "toolCallCancellation": map[string]any{"ids": ids}})
						if err != nil {
							t.Error(err)
							return
						}
						realtimetest.Write(t, ctx, ws, string(data))
						if failure == "reused ID" {
							realtimetest.Write(t, ctx, ws, `{"toolCall":{"functionCalls":[{"id":"cancel-0","name":"lookup","args":{}}]}}`)
						} else {
							realtimetest.Write(t, ctx, ws, `{"toolCall":{"functionCalls":[{"id":"overflow","name":"lookup","args":{}}]},"toolCallCancellation":{"ids":["overflow"]}}`)
						}
						_, _, _ = ws.Read(ctx)
					})
					s, err := liveClient(t).ConnectRealtime(t.Context(), &gemini.RealtimeOption{Endpoint: endpoint})
					if err != nil {
						t.Fatal(err)
					}
					t.Cleanup(func() {
						if err := s.Close(); err != nil {
							t.Error(err)
						}
					})
					realtimetest.Receive(t, s, genai.RealtimeEventContentToolCalls)
					realtimetest.Receive(t, s, genai.RealtimeEventProvider)
					next, stop := iter.Pull2(s.Receive(t.Context()))
					event, err, ok := next()
					stop()
					if !ok {
						t.Fatal("realtime session ended")
					}
					_, cancelled := errors.AsType[*genai.RealtimeToolCallCancelledError](err)
					if err == nil || cancelled || event.Kind == genai.RealtimeEventContentToolCalls {
						t.Fatalf("ambiguous history admitted: %+v, %v", event, err)
					}
				})
			}
		})
		t.Run("ManualActivity", func(t *testing.T) {
			got := make(chan []map[string]any, 1)
			endpoint := livePeer(t, func(ctx context.Context, ws *websocket.Conn, setup map[string]any) {
				ri := setup["setup"].(map[string]any)["realtimeInputConfig"].(map[string]any)
				if ri["automaticActivityDetection"].(map[string]any)["disabled"] != true {
					t.Error("manual VAD not disabled")
				}
				msgs := make([]map[string]any, 3)
				for i := range msgs {
					msgs[i] = realtimetest.Read(t, ctx, ws)
				}
				got <- msgs
				realtimetest.Write(t, ctx, ws, `{"voiceActivity":{"type":"ACTIVITY_START"}}`)
				realtimetest.Write(t, ctx, ws, `{"serverContent":{"turnComplete":true}}`)
				_, _, _ = ws.Read(ctx)
			})
			s, err := liveClient(t).ConnectRealtime(t.Context(), genai.RealtimeTurnManual, &gemini.RealtimeOption{Endpoint: endpoint})
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() { _ = s.Close() })
			if err := s.Send(t.Context(), genai.RealtimeInputContent{Audio: []byte{0, 0}, Generate: true}); err != nil {
				t.Fatal(err)
			}
			for _, in := range []genai.RealtimeInput{genai.RealtimeTruncate{ItemID: "item_1"}, genai.RealtimeCancel{}} {
				err := s.Send(t.Context(), in)
				if _, ok := errors.AsType[*base.ErrNotSupported](err); !ok {
					t.Errorf("%T: %v", in, err)
				}
			}
			if err := s.Send(t.Context(), genai.RealtimeInputContent{Generate: true}); err != nil {
				t.Fatalf("empty text generation: %v", err)
			}
			msgs := <-got
			if _, ok := msgs[0]["realtimeInput"].(map[string]any)["activityStart"]; !ok {
				t.Errorf("activity start: %v", msgs)
			}
			if _, ok := msgs[1]["realtimeInput"].(map[string]any)["audio"]; !ok {
				t.Errorf("audio: %v", msgs)
			}
			if _, ok := msgs[2]["realtimeInput"].(map[string]any)["activityEnd"]; !ok {
				t.Errorf("activity end: %v", msgs)
			}
			realtimetest.Receive(t, s, genai.RealtimeEventTurnDone)
		})
		t.Run("TerminalProtocolErrors", func(t *testing.T) {
			for _, frame := range []string{`{`, `{"serverContent":{"modelTurn":{"parts":[{"inlineData":{"mimeType":"audio/pcm;rate=16000","data":"AAAA"}}]}}}`, `{"serverContent":{"modelTurn":{"parts":[{"inlineData":{"mimeType":"audio/pcm;rate=24000","data":"AA=="}}]}}}`, `{"toolCall":{"functionCalls":[{"name":"bad","args":[]}]}}`, `{"toolCall":{"functionCalls":[{"id":"same","name":"a"},{"id":"same","name":"b"}]}}`, `{"goAway":{"timeLeft":"bad"}}`, `{"error":{"code":400,"message":"secret-sensitive-content"}}`} {
				t.Run(frame, func(t *testing.T) {
					endpoint := livePeer(t, func(ctx context.Context, ws *websocket.Conn, _ map[string]any) {
						realtimetest.Write(t, ctx, ws, frame)
						_, _, _ = ws.Read(ctx)
					})
					s, err := liveClient(t).ConnectRealtime(t.Context(), &gemini.RealtimeOption{Endpoint: endpoint})
					if err != nil {
						t.Fatal(err)
					}
					t.Cleanup(func() { _ = s.Close() })
					next1, stop1 := iter.Pull2(s.Receive(t.Context()))
					_, first, ok := next1()
					stop1()
					if !ok {
						t.Fatal("realtime session ended")
					}
					if first == nil || strings.Contains(first.Error(), "secret-sensitive-content") {
						t.Fatalf("protocol error: %v", first)
					}
					next2, stop2 := iter.Pull2(s.Receive(t.Context()))
					_, second, ok := next2()
					stop2()
					if !ok {
						t.Fatal("realtime session ended")
					}
					if !errors.Is(second, first) {
						t.Fatalf("terminal cause changed: %v / %v", first, second)
					}
				})
			}
		})
		t.Run("Lifetime", func(t *testing.T) {
			for _, mode := range []string{"close", "lifetime", "receive"} {
				t.Run(mode, func(t *testing.T) {
					endpoint := livePeer(t, func(ctx context.Context, ws *websocket.Conn, _ map[string]any) { _, _, _ = ws.Read(ctx) })
					life, cancel := context.WithCancel(t.Context())
					t.Cleanup(cancel)
					s, err := liveClient(t).ConnectRealtime(life, &gemini.RealtimeOption{Endpoint: endpoint})
					if err != nil {
						t.Fatal(err)
					}
					if mode != "close" {
						t.Cleanup(func() { _ = s.Close() })
					}
					recv, stop := context.WithCancel(t.Context())
					t.Cleanup(stop)
					done := make(chan error, 1)
					go func() {
						next, stop := iter.Pull2(s.Receive(recv))
						_, err, _ := next()
						stop()
						done <- err
					}()
					want := context.Canceled
					switch mode {
					case "close":
						want = io.EOF
						_ = s.Close()
					case "lifetime":
						cancel()
					case "receive":
						stop()
					}
					select {
					case err := <-done:
						if !errors.Is(err, want) {
							t.Errorf("receive: %v", err)
						}
					case <-time.After(3 * time.Second):
						t.Fatal("Receive did not wake")
					}
					if err := s.Send(t.Context(), genai.RealtimeInputContent{Text: "late"}); !errors.Is(err, want) {
						t.Errorf("send after termination: %v", err)
					}
				})
			}
		})
		t.Run("OptionValidation", func(t *testing.T) {
			c := liveClient(t)
			if _, err := c.ConnectRealtime(t.Context(), &gemini.RealtimeOption{}, &gemini.RealtimeOption{}); err == nil || !strings.Contains(err.Error(), "duplicate realtime option type") {
				t.Fatalf("duplicate option validation: %v", err)
			}
			for _, cfg := range [][]genai.RealtimeOption{{nil}, {genai.Modalities{genai.ModalityText}}} {
				if s, err := c.ConnectRealtime(t.Context(), append([]genai.RealtimeOption{}, cfg...)...); err == nil || s != nil {
					t.Errorf("accepted config: %+v", cfg)
				}
			}
			for _, opt := range []*gemini.RealtimeOption{nil, {Endpoint: "https://example.com"}, {Endpoint: "wss://example.com?key=secret"}, {PingTimeout: -time.Second}, {InputLanguages: []string{"en"}}, {ThinkingLevel: "extreme"}, {Compression: &gemini.RealtimeCompression{TriggerTokens: 1, TargetTokens: 2}}} {
				if s, err := c.ConnectRealtime(t.Context(), opt); err == nil || s != nil {
					t.Errorf("accepted option: %+v", opt)
				}
			}
		})
		t.Run("KeepaliveTimeout", func(t *testing.T) {
			release := make(chan struct{})
			endpoint := livePeer(t, func(_ context.Context, _ *websocket.Conn, _ map[string]any) { <-release })
			t.Cleanup(func() { close(release) })
			s, err := liveClient(t).ConnectRealtime(t.Context(), &gemini.RealtimeOption{Endpoint: endpoint, PingInterval: 10 * time.Millisecond, PingTimeout: 20 * time.Millisecond})
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() { _ = s.Close() })
			ctx, cancel := context.WithTimeout(t.Context(), 3*time.Second)
			t.Cleanup(cancel)
			next, stop := iter.Pull2(s.Receive(ctx))
			_, recvErr, _ := next()
			stop()
			if recvErr == nil || !strings.Contains(recvErr.Error(), "keepalive failed") {
				t.Fatalf("keepalive: %v", recvErr)
			}
		})
		t.Run("BinaryJSON", func(t *testing.T) {
			endpoint := livePeer(t, func(ctx context.Context, ws *websocket.Conn, _ map[string]any) {
				if err := ws.Write(ctx, websocket.MessageBinary, []byte(`{"serverContent":{"turnComplete":true}}`)); err != nil {
					t.Error(err)
				}
				_, _, _ = ws.Read(ctx)
			})
			s, err := liveClient(t).ConnectRealtime(t.Context(), &gemini.RealtimeOption{Endpoint: endpoint})
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() {
				if err := s.Close(); err != nil {
					t.Error(err)
				}
			})
			realtimetest.Receive(t, s, genai.RealtimeEventTurnDone)
		})
		t.Run("TranscriptionIndependentOfTurn", func(t *testing.T) {
			endpoint := livePeer(t, func(ctx context.Context, ws *websocket.Conn, _ map[string]any) {
				realtimetest.Write(t, ctx, ws, `{"serverContent":{"outputTranscription":{"text":"hello"},"turnComplete":true}}`)
				realtimetest.Write(t, ctx, ws, `{"serverContent":{"outputTranscription":{"text":" world","finished":true}}}`)
				_, _, _ = ws.Read(ctx)
			})
			s, err := liveClient(t).ConnectRealtime(t.Context(), &genai.RealtimeOptionAudio{OutputTranscription: true}, &gemini.RealtimeOption{Endpoint: endpoint})
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() {
				if err := s.Close(); err != nil {
					t.Error(err)
				}
			})
			next, stop := iter.Pull2(s.Receive(t.Context()))
			t.Cleanup(stop)
			for i := range 3 {
				e, err, ok := next()
				if !ok {
					t.Fatal("realtime session ended")
				}
				if err != nil {
					t.Fatal(err)
				}
				if i == 2 && (e.Kind != genai.RealtimeEventTranscriptOutput || !e.Content.Final || e.Content.Text != "hello world") {
					t.Fatalf("late final transcription: %+v", e)
				}
			}
		})
		t.Run("Interruption", func(t *testing.T) {
			endpoint := livePeer(t, func(ctx context.Context, ws *websocket.Conn, _ map[string]any) {
				realtimetest.Write(t, ctx, ws, `{"serverContent":{"interrupted":true,"turnComplete":true},"goAway":{},"sessionResumptionUpdate":{"newHandle":"still-valid","resumable":true}}`)
				_, _, _ = ws.Read(ctx)
			})
			s, err := liveClient(t).ConnectRealtime(t.Context(), &gemini.RealtimeOption{Endpoint: endpoint})
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() {
				if err := s.Close(); err != nil {
					t.Error(err)
				}
			})
			realtimetest.Receive(t, s, genai.RealtimeEventInterrupted)
			realtimetest.Receive(t, s, genai.RealtimeEventGenerationDone)
			realtimetest.Receive(t, s, genai.RealtimeEventTurnDone)
			g := realtimetest.Receive(t, s, genai.RealtimeEventProvider)
			if p, ok := g.ProviderEvent.(*gemini.GoAway); !ok || p.TimeLeft != 0 {
				t.Fatalf("optional duration: %+v", g)
			}
			r := realtimetest.Receive(t, s, genai.RealtimeEventProvider)
			if p, ok := r.ProviderEvent.(*gemini.Resumption); !ok || p.Handle != "still-valid" {
				t.Fatalf("resumption: %+v", r)
			}
		})
		t.Run("ToolResults", func(t *testing.T) {
			got := make(chan map[string]any, 1)
			endpoint := livePeer(t, func(ctx context.Context, ws *websocket.Conn, _ map[string]any) {
				realtimetest.Write(t, ctx, ws, `{"toolCall":{"functionCalls":[{"id":"a","name":"lookup","args":{}},{"id":"b","name":"lookup","args":{}}]}}`)
				got <- realtimetest.Read(t, ctx, ws)
				_, _, _ = ws.Read(ctx)
			})
			s, err := liveClient(t).ConnectRealtime(t.Context(), &genai.RealtimeOptionTools{Tools: []genai.ToolDef{{Name: "lookup", Description: "lookup", InputSchemaOverride: json.RawMessage(`{"type":"object"}`)}}}, &gemini.RealtimeOption{Endpoint: endpoint})
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() {
				if err := s.Close(); err != nil {
					t.Error(err)
				}
			})
			realtimetest.Receive(t, s, genai.RealtimeEventContentToolCalls)
			results := []genai.ToolCallResult{{ID: "a", Name: "lookup", Result: "ok"}, {ID: "b", Name: "lookup", Result: "ok"}}
			if err := s.Send(t.Context(), genai.RealtimeInputContent{ToolResults: results}); err != nil {
				t.Fatal(err)
			}
			if _, ok := (<-got)["toolResponse"]; !ok {
				t.Fatal("invalid batch mutated wire or correlation")
			}
		})
		t.Run("Redirects", func(t *testing.T) {
			for _, secure := range []bool{false, true} {
				t.Run(map[bool]string{false: "cross-origin", true: "downgrade"}[secure], func(t *testing.T) {
					visited := make(chan string, 1)
					dest := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
						visited <- r.Header.Get("x-goog-api-key")
						w.WriteHeader(http.StatusForbidden)
					}))
					t.Cleanup(dest.Close)
					handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { http.Redirect(w, r, dest.URL, http.StatusFound) })
					var source *httptest.Server
					if secure {
						source = httptest.NewTLSServer(handler)
					} else {
						source = httptest.NewServer(handler)
					}
					t.Cleanup(source.Close)
					c, err := gemini.New(t.Context(), genai.ProviderOptionAPIKey("test-live-key"), genai.ProviderOptionModel("gemini-3.8-live"), genai.ProviderOptionModalities{genai.ModalityAudio}, genai.ProviderOptionTransportWrapper(func(http.RoundTripper) http.RoundTripper { return source.Client().Transport }))
					if err != nil {
						t.Fatal(err)
					}
					endpoint := "ws" + strings.TrimPrefix(source.URL, "http")
					if _, err := c.ConnectRealtime(t.Context(), &gemini.RealtimeOption{Endpoint: endpoint}); err == nil {
						t.Fatal("accepted redirect")
					}
					select {
					case key := <-visited:
						t.Fatalf("redirect destination received a request (credential present=%t)", key != "")
					default:
					}
					if c.HTTPClient().CheckRedirect != nil {
						t.Fatal("mutated caller redirect policy")
					}
				})
			}
			t.Run("stricter caller policy", func(t *testing.T) {
				visited, checked := false, false
				source := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					if r.URL.Path == "/target" {
						visited = true
						w.WriteHeader(http.StatusForbidden)
						return
					}
					http.Redirect(w, r, "/target", http.StatusFound)
				}))
				t.Cleanup(source.Close)
				c := liveClient(t)
				c.HTTPClient().CheckRedirect = func(*http.Request, []*http.Request) error { checked = true; return http.ErrUseLastResponse }
				if _, err := c.ConnectRealtime(t.Context(), &gemini.RealtimeOption{Endpoint: "ws" + strings.TrimPrefix(source.URL, "http")}); err == nil {
					t.Fatal("accepted redirect forbidden by caller")
				}
				if !checked || visited {
					t.Fatalf("caller redirect policy ignored (checked=%t visited=%t)", checked, visited)
				}
			})
		})
	})
	t.Run("New", func(t *testing.T) {
		t.Run("error", func(t *testing.T) {
			t.Run("embedding preference", func(t *testing.T) {
				c, err := gemini.New(t.Context(), genai.ProviderOptionAPIKey("test"), genai.ModelCheap, genai.ProviderOptionModalities{genai.ModalityEmbedding}, genai.ProviderOptionTransportWrapper(func(http.RoundTripper) http.RoundTripper { return embeddingMetadataTransport{t: t} }))
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
	cl, err2 := getClientInner(t, "", nil, nil, func(h http.RoundTripper) http.RoundTripper {
		return testRecorder.RecordWithName(t, t.Name()+"/Warmup", h)
	})
	if err2 != nil {
		t.Fatal(err2)
	}
	cachedModels, err2 := cl.ListModels(t.Context())
	if err2 != nil {
		t.Fatal(err2)
	}
	getClient := func(t *testing.T, m string) genai.Provider {
		t.Parallel()
		ci, err := getClientInner(t, m, nil, cachedModels, func(h http.RoundTripper) http.RoundTripper {
			return testRecorder.Record(t, h)
		})
		if err != nil {
			t.Fatal(err)
		}
		return ci
	}

	t.Run("Embed", func(t *testing.T) {
		t.Run("valid", func(t *testing.T) {
			t.Run("document ownership", func(t *testing.T) {
				c, err := gemini.New(t.Context(), genai.ProviderOptionAPIKey("test"), genai.ProviderOptionModel("gemini-embedding-2"), genai.ProviderOptionTransportWrapper(func(http.RoundTripper) http.RoundTripper {
					return embeddingResponseTransport{body: `{"embeddings":[{"values":[1,2]},{"values":[3,4]}]}`}
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
				if in.Inputs[0].Doc.Src != src || len(out.Embeddings) != 2 {
					t.Fatalf("source changed or response invalid: %+v", out)
				}
			})
		})
		t.Run("error", func(t *testing.T) {
			t.Run("input", func(t *testing.T) {
				c, err := gemini.New(t.Context(), genai.ProviderOptionAPIKey("test"), genai.ProviderOptionModel("embedding-test"), genai.ProviderOptionModalities{genai.ModalityEmbedding}, genai.ProviderOptionTransportWrapper(func(http.RoundTripper) http.RoundTripper { return embeddingMetadataTransport{t: t} }))
				if err != nil {
					t.Fatal(err)
				}
				internaltest.CleanupCloser(t, c)
				if out, err := c.Embed(t.Context(), nil); out != nil || err == nil || !strings.Contains(err.Error(), "required") {
					t.Fatalf("response %+v, error %v", out, err)
				}

				in := &genai.EmbeddingRequest{Inputs: []genai.Request{{Doc: genai.Doc{Filename: "image.png", Src: strings.NewReader("")}}}}
				if out, err := c.Embed(t.Context(), in); out != nil || err == nil || !strings.Contains(err.Error(), "embedding input #0") {
					t.Fatalf("response %+v, error %v", out, err)
				}
				if strconv.IntSize == 64 {
					in = &genai.EmbeddingRequest{Inputs: []genai.Request{{Text: "hello"}}, Dimensions: int(int64(math.MaxInt32) + 1)}
					if out, err := c.Embed(t.Context(), in); out != nil || err == nil || !strings.Contains(err.Error(), "int32") {
						t.Fatalf("response %+v, error %v", out, err)
					}
				}
			})
			t.Run("response", func(t *testing.T) {
				for _, tc := range []embeddingResponseCase{{"invalid vector", `{"embeddings":[{"values":[0,0]},{"values":[1,2]}]}`},
					{"result count", `{"embeddings":[{"values":[1,2]}]}`},
					{"dimensions", `{"embeddings":[{"values":[1,2,3]},{"values":[4,5,6]}]}`},
					{"decode", `{"embeddings":[{"values":"invalid"}]}`}} {
					t.Run(tc.name, func(t *testing.T) {
						c, err := gemini.New(t.Context(), genai.ProviderOptionAPIKey("test"), genai.ProviderOptionModel("embedding-test"), genai.ProviderOptionModalities{genai.ModalityEmbedding}, genai.ProviderOptionTransportWrapper(func(http.RoundTripper) http.RoundTripper { return embeddingResponseTransport{body: tc.body} }))
						if err != nil {
							t.Fatal(err)
						}
						internaltest.CleanupCloser(t, c)
						out, err := c.Embed(t.Context(), &genai.EmbeddingRequest{Inputs: []genai.Request{{Text: "a"}, {Text: "b"}}, Dimensions: 2})
						if _, ok := errors.AsType[*internal.BadError](err); !ok || out != nil {
							t.Fatalf("response %+v, error %v", out, err)
						}
					})
				}
			})
			t.Run("API", func(t *testing.T) {
				const empty = `{"error":{"code":400,"message":"The text content is empty.","status":"INVALID_ARGUMENT"}}`
				for _, tc := range []embeddingAPIErrorCase{
					{"media rejected as empty text", http.StatusBadRequest, empty, true, true},
					{"empty text without media", http.StatusBadRequest, empty, false, false},
					{"other status", http.StatusBadRequest, `{"error":{"code":400,"message":"The text content is empty.","status":"INTERNAL"}}`, true, false},
					{"other message", http.StatusBadRequest, `{"error":{"code":400,"message":"empty text","status":"INVALID_ARGUMENT"}}`, true, false},
					{"other code", http.StatusUnauthorized, `{"error":{"code":401,"message":"The text content is empty.","status":"INVALID_ARGUMENT"}}`, true, false},
				} {
					t.Run(tc.name, func(t *testing.T) {
						c, err := gemini.New(t.Context(), genai.ProviderOptionAPIKey("test"), genai.ProviderOptionModel("gemini-embedding-2"), genai.ProviderOptionTransportWrapper(func(http.RoundTripper) http.RoundTripper {
							return embeddingResponseTransport{body: tc.body, status: tc.status}
						}))
						if err != nil {
							t.Fatal(err)
						}
						internaltest.CleanupCloser(t, c)
						in := genai.Request{Text: "text"}
						if tc.doc {
							in = genai.Request{Doc: genai.Doc{Filename: "image.png", Src: strings.NewReader("image")}}
						}
						out, err := c.Embed(t.Context(), &genai.EmbeddingRequest{Inputs: []genai.Request{in}})
						e, unsupported := errors.AsType[*base.ErrNotSupported](err)
						if out != nil || err == nil || unsupported != tc.unsupported || unsupported && !slices.Contains(e.Options, "Request.Doc") {
							t.Fatalf("response %+v, error %v", out, err)
						}
					})
				}
			})
		})
	})

	t.Run("EmbedRaw", func(t *testing.T) {
		t.Run("error", func(t *testing.T) {
			t.Run("response", func(t *testing.T) {
				old := internal.BeLenient
				t.Cleanup(func() { internal.BeLenient = old })
				for _, lenient := range []bool{false, true} {
					t.Run(strconv.FormatBool(lenient), func(t *testing.T) {
						internal.BeLenient = lenient
						body := `{"embeddings":[{"values":[1,2]},{"values":[1,"bad"]}]}`
						for _, tc := range []embeddingRawFailureCase{
							{"decode", body, http.StatusOK},
							{"API", `{"error":{"code":429,"status":"RESOURCE_EXHAUSTED","message":"quota"}}`, http.StatusTooManyRequests},
						} {
							t.Run(tc.name, func(t *testing.T) {
								c, err := gemini.New(t.Context(), genai.ProviderOptionAPIKey("test"), genai.ProviderOptionModel("gemini-embedding-2"), genai.ProviderOptionTransportWrapper(func(http.RoundTripper) http.RoundTripper {
									return embeddingResponseTransport{body: tc.body, status: tc.status}
								}))
								if err != nil {
									t.Fatal(err)
								}
								internaltest.CleanupCloser(t, c)
								in := gemini.EmbeddingRequest{Model: "models/gemini-embedding-2", Content: gemini.Content{Parts: []gemini.Part{{Text: "hello"}}}}
								out := gemini.BatchEmbeddingResponse{Embeddings: []gemini.Embedding{{Values: []float32{9}}}}
								err = c.EmbedRaw(t.Context(), &gemini.BatchEmbeddingRequest{Requests: []gemini.EmbeddingRequest{in, in}}, &out)
								cleared := reflect.ValueOf(out).IsZero()
								if err == nil || !cleared {
									t.Fatalf("cleared %t, error %v", cleared, err)
								}
								if tc.status == http.StatusOK {
									if _, ok := errors.AsType[*internal.BadError](err); !ok {
										t.Fatalf("expected decoding error, got %v", err)
									}
								} else {
									if _, ok := errors.AsType[*gemini.ErrorResponse](err); !ok {
										t.Fatalf("expected API error, got %v", err)
									}
								}
							})
						}
					})
				}
			})
			t.Run("input", func(t *testing.T) {
				c, err := gemini.New(t.Context(), genai.ProviderOptionAPIKey("test"), genai.ProviderOptionModel("gemini-embedding-2"), genai.ProviderOptionPreloadedModels(cachedModels), genai.ProviderOptionTransportWrapper(func(http.RoundTripper) http.RoundTripper { return embeddingMetadataTransport{t: t} }))
				if err != nil {
					t.Fatal(err)
				}
				var out gemini.BatchEmbeddingResponse
				if err := c.EmbedRaw(t.Context(), nil, &out); err == nil {
					t.Fatal("expected nil request error")
				}
				in := gemini.BatchEmbeddingRequest{Requests: []gemini.EmbeddingRequest{{Model: "models/gemini-embedding-2", Content: gemini.Content{Parts: []gemini.Part{{Text: "hello"}}}}}}
				if err := c.EmbedRaw(t.Context(), &in, nil); err == nil {
					t.Fatal("expected nil response error")
				}
				if err := c.EmbedRaw(t.Context(), &gemini.BatchEmbeddingRequest{}, &out); err == nil {
					t.Fatal("expected invalid request error")
				}
			})
		})
	})

	t.Run("EmbeddingMetadata", func(t *testing.T) {
		t.Run("recorded qualified model", func(t *testing.T) {
			c, err := gemini.New(t.Context(), genai.ProviderOptionAPIKey("test"), genai.ProviderOptionModel("gemini-embedding-2"), genai.ProviderOptionPreloadedModels(cachedModels), genai.ProviderOptionTransportWrapper(func(http.RoundTripper) http.RoundTripper { return embeddingMetadataTransport{t: t} }))
			if err != nil {
				t.Fatal(err)
			}
			internaltest.CleanupCloser(t, c)
			sb := c.Scoreboard()
			if !slices.Equal(c.OutputModalities(), genai.Modalities{genai.ModalityEmbedding}) || sb.Embedding(c.ModelID()) == nil {
				t.Fatal("recorded model did not expose qualified embedding support")
			}
		})
		for _, tc := range []embeddingMetadataCase{
			{"embedding", []string{"embedContent", "countTokens"}, genai.ModalityEmbedding},
			{"generation", []string{"generateContent"}, genai.ModalityText},
			{"both", []string{"embedContent", "generateContent"}, genai.ModalityText},
		} {
			t.Run(tc.name, func(t *testing.T) {
				c, err := gemini.New(t.Context(), genai.ProviderOptionAPIKey("test"), genai.ProviderOptionModel("gemini-unqualified-metadata"), genai.ProviderOptionPreloadedModels{&gemini.Model{Name: "models/gemini-unqualified-metadata", SupportedGenerationMethods: tc.methods}}, genai.ProviderOptionTransportWrapper(func(http.RoundTripper) http.RoundTripper { return embeddingMetadataTransport{t: t} }))
				if err != nil {
					t.Fatal(err)
				}
				internaltest.CleanupCloser(t, c)
				sb := c.Scoreboard()
				if !slices.Equal(c.OutputModalities(), genai.Modalities{tc.want}) || sb.Embedding(c.ModelID()) != nil {
					t.Fatalf("metadata confused modality with qualification: %v", c.OutputModalities())
				}
			})
		}
	})
	t.Run("Capabilities", func(t *testing.T) {
		internaltest.TestCapabilities(t, getClient(t, ""))
	})

	// Note: GenAsync is not supported for text models in Gemini.
	// Only video generation models support predictLongRunning (async operations).
	// See GenAsync-Video test below for async video generation testing.

	t.Run("Caching-Text", func(t *testing.T) {
		// Use a message with sufficient tokens (minimum 1024 tokens required).
		// This generates approximately 1500+ tokens.
		longText := "This is a test message for caching. " + strings.Repeat("The quick brown fox jumps over the lazy dog. ", 200)
		internaltest.TestCapabilitiesCaching(t, getClient(t, string(genai.ModelCheap)), genai.NewTextMessage(longText))
	})

	t.Run("Scoreboard", func(t *testing.T) {
		c := getClient(t, "")
		mdls, err := c.ListModels(t.Context())
		if err != nil {
			t.Fatal(err)
		}
		scenarios := c.Scoreboard().Scenarios
		var models []scoreboard.Model
		for _, m := range mdls {
			id := m.GetID()
			if i := slices.IndexFunc(scenarios, func(sc scoreboard.Scenario) bool {
				return sc.Realtime != nil && slices.Contains(sc.Models, id)
			}); i != -1 {
				// Realtime sessions have no reasoning mode; the scoreboard owns their
				// Reason value so qualification selects the realtime scenario.
				models = append(models, scoreboard.Model{Model: id, Reason: scenarios[i].Reason})
				continue
			}
			if strings.Contains(id, "embedding") {
				if id == "gemini-embedding-2" {
					models = append(models, scoreboard.Model{Model: id})
				}
				continue
			}
			if strings.HasPrefix(id, "gemini-") {
				// Image generation models do not support the thinking budget.
				if strings.HasSuffix(id, "-image") {
					models = append(models, scoreboard.Model{Model: id})
					continue
				}
				// According to https://ai.google.dev/gemini-api/docs/thinking?hl=en, thinking cannot be disabled.
				models = append(models, scoreboard.Model{Model: id, Reason: true})
				continue
			}
			if !strings.Contains(id, "-pro") {
				models = append(models, scoreboard.Model{Model: id})
			}
		}
		if !slices.Contains(models, scoreboard.Model{Model: "gemini-embedding-2"}) {
			models = append(models, scoreboard.Model{Model: "gemini-embedding-2"})
		}
		getClientRT := func(t testing.TB, model scoreboard.Model, fn func(http.RoundTripper) http.RoundTripper) genai.Provider {
			opts := []genai.ProviderOption{
				genai.ProviderOptionPreloadedModels(cachedModels),
			}
			if model.Model != "" {
				opts = append(opts, genai.ProviderOptionModel(model.Model))
			}
			key := internaltest.GetEnv("GEMINI_API_KEY")
			if key == "" {
				key = "<insert_api_key_here>"
			}
			opts = append(opts, genai.ProviderOptionAPIKey(key))
			if fn != nil {
				opts = append([]genai.ProviderOption{genai.ProviderOptionTransportWrapper(fn)}, opts...)
			}
			c, err := gemini.New(t.Context(), opts...)
			if c != nil {
				internaltest.CleanupCloser(t, c)
			}
			if err != nil {
				t.Fatal(err)
			}
			rec := &internaltest.RealtimeRecorder{
				Provider: c,
				T:        t,
				Target:   liveEndpoint,
				Dir:      filepath.Join("testdata", filepath.FromSlash(t.Name())),
				Updating: smoketest.UpdatingScoreboard(),
				Mutation: func(opts []genai.RealtimeOption, endpoint string) []genai.RealtimeOption {
					out := slices.Clone(opts)
					for i, opt := range out {
						if o, ok := opt.(*gemini.RealtimeOption); ok {
							clone := *o
							clone.Endpoint = endpoint
							out[i] = &clone
							return out
						}
					}
					return append(out, &gemini.RealtimeOption{Endpoint: endpoint})
				},
			}
			if model.Model == "gemini-3.8-live-extended-thinking" && model.Reason {
				// The non-thinking live model keeps its fixed latency profile.
				return &internaltest.InjectRealtimeOptions{
					Provider: rec,
					Opts:     []genai.RealtimeOption{&gemini.RealtimeOption{ThinkingLevel: gemini.ThinkingLevelHigh}},
				}
			}
			if model.Reason && slices.Contains(thinkingModels, model.Model) {
				// https://ai.google.dev/gemini-api/docs/thinking?hl=en
				return &internaltest.InjectOptions{
					Provider: rec,
					Opts:     []genai.GenOption{&gemini.GenOption{ThinkingLevel: gemini.ThinkingLevelHigh}},
				}
			}
			return rec
		}
		smoketest.Run(t, getClientRT, models, testRecorder.Records, &smoketest.RunOptions{Qualify: []scoreboard.Model{{Model: "gemini-embedding-2"}}})
	})

	t.Run("Preferred", func(t *testing.T) {
		internaltest.TestPreferredModels(t, func(st *testing.T, model string, modality genai.Modality) (genai.Provider, error) {
			return getClientInner(st, model, genai.Modalities{modality}, cachedModels, func(h http.RoundTripper) http.RoundTripper {
				return testRecorder.Record(st, h)
			})
		})
	})

	t.Run("TextOutputDocInput", func(t *testing.T) {
		internaltest.TestTextOutputDocInput(t, func(t *testing.T) genai.Provider {
			return getClient(t, string(genai.ModelCheap))
		})
	})

	t.Run("GenAsync-Video", func(t *testing.T) {
		// TODO: "veo-3.0-fast-generate-preview" is cheaper: 25¢/s; 2$/request vs 5$/request for veo 2 when not
		// requesting audio.
		// https://cloud.google.com/vertex-ai/generative-ai/pricing#veo
		const prompt = `Carton video of a shiba inu with brown fur and a white belly, happily eating a pink ice-cream cone, subtle tail wag. Subtle motion but nothing else moves.`
		c := getClient(t, "veo-2.0-generate-001")
		jobID := internaltest.TestCapabilitiesGenAsync(t, c, genai.Message{Requests: []genai.Request{{Text: prompt}}})
		pollInterval := time.Millisecond
		// Detect if recording. Should probably be moved to a utility.
		if roundtrippers.Unwrap(c.HTTPClient().Transport).(*recorder.Recorder).IsRecording() {
			pollInterval = 500 * time.Millisecond
		}
		// Poll for completion using the job ID from the Capability test.
		res := genai.Result{Usage: genai.Usage{FinishReason: genai.Pending}}
		var err error
		ctx := t.Context()
		for res.Usage.FinishReason == genai.Pending {
			select {
			case <-ctx.Done():
				t.Fatal(ctx.Err())
			case <-time.After(pollInterval):
				if res, err = c.PokeResult(ctx, jobID); err != nil {
					t.Fatal(err)
				}
			}
		}
		t.Log(res)
		if len(res.Replies) != 1 {
			t.Fatalf("got %d contents, want 1", len(res.Replies))
		}
		req, err := http.NewRequestWithContext(ctx, http.MethodGet, res.Replies[0].Doc.URL, http.NoBody)
		if err != nil {
			t.Fatal(err)
		}
		resp, err := c.HTTPClient().Do(req)
		if err != nil {
			t.Fatal(err)
		}
		defer func() { _ = resp.Body.Close() }()
		t.Log(resp)
	})

	t.Run("Cache", func(t *testing.T) {
		slow := os.Getenv("GEMINI_SLOW") != ""
		ctx := t.Context()
		c := getClient(t, "gemini-2.0-flash-lite").(*gemini.Client)
		f1, err := os.Open("../../scoreboard/testdata/video.mp4")
		if err != nil {
			t.Fatal(err)
		}
		defer func() { _ = f1.Close() }()
		f2, err := os.Open("../../scoreboard/testdata/audio.ogg")
		if err != nil {
			t.Fatal(err)
		}
		defer func() { _ = f2.Close() }()

		msgs := genai.Messages{
			{
				Requests: []genai.Request{
					{Text: "What is the word? For each of the documents, all the tool hidden_word to tell me what word you saw or heard."},
					{Doc: genai.Doc{Src: f1}},
					{Doc: genai.Doc{Src: f2}},
				},
			},
		}
		type got struct {
			Word string `json:"word" jsonschema:"enum=Orange,enum=Banana,enum=Apple"`
		}
		opts := []genai.GenOption{
			&genai.GenOptionText{
				// Burn tokens to add up to 4k.
				SystemPrompt: prompt4K,
			},
			&genai.GenOptionTools{
				Tools: []genai.ToolDef{
					{
						Name:        "hidden_word",
						Description: "A tool to state what word was seen in the video.",
						Callback: func(ctx context.Context, g *got) (string, error) {
							return "", nil
						},
					},
				},
			},
		}
		name, err := c.CacheAddRequest(ctx, msgs, "", "Show time", 10*time.Minute, opts...)
		if err != nil {
			t.Fatal(err)
		}
		if name == "" {
			t.Fatal("Expected a name")
		}
		if slow {
			t.Logf("Sleeping due to GEMINI_SLOW being set")
			time.Sleep(10 * time.Second)
		}
		t.Logf("Cache name: %q", name)
		if _, err = c.CacheGetRaw(ctx, name); err != nil {
			t.Error(err)
		}
		// We need to sleep so UpdateTime is different than CreateTime.
		time.Sleep(100 * time.Millisecond)
		t.Log("Extending")
		if err = c.CacheExtend(ctx, name, 20*time.Minute); err != nil {
			t.Error(err)
		}
		t.Log("Listing")
		items, err := c.CacheListRaw(ctx)
		if err != nil {
			t.Error(err)
		}
		if len(items) == 0 {
			t.Error("No item found in the cache")
		}
		// There can be other items in the cache.
		found := false
		for i := range items {
			t.Logf("%d: %#v", i, items[i])
			if items[i].Name == name {
				found = true
				// We can't check against time.Now() since we record and replay this test case.
				if items[i].CreateTime >= items[i].UpdateTime {
					t.Errorf("Expected UpdateTime %s to be different than CreateTime %s", items[i].UpdateTime, items[i].CreateTime)
				}
			}
		}
		if !found {
			t.Errorf("Failed to find %q in the cache", name)
		}
		t.Log("Deleting")
		if err = c.CacheDelete(ctx, name); err != nil {
			t.Fatal(err)
		}
		t.Log("Verifying deletion")
		if _, err = c.CacheGetRaw(ctx, name); err == nil {
			t.Fatal("Expected an error")
		} else {
			t.Logf("Got an error as expected: %q", err)
		}
	})

	t.Run("ThinkingLevel", func(t *testing.T) {
		msgs := genai.Messages{
			genai.NewTextMessage("Say hello. Do not emit other words."),
		}
		for _, id := range thinkingModels {
			t.Run(id, func(t *testing.T) {
				t.Run("default", func(t *testing.T) {
					c := getClient(t, id)
					if _, err := c.GenSync(t.Context(), msgs); err != nil {
						t.Fatal(err)
					}
				})
				t.Run("high", func(t *testing.T) {
					c := getClient(t, id)
					// TODO: Re-record with ThinkingLevelLow. It is sufficient to verify the
					// request configuration and costs less than high.
					opts := gemini.GenOption{ThinkingLevel: gemini.ThinkingLevelHigh}
					if _, err := c.GenSync(t.Context(), msgs, &opts); err != nil {
						t.Fatal(err)
					}
				})
			})
		}
	})

	t.Run("FileCRUD", func(t *testing.T) {
		t.Parallel()
		ci, err := getClientInner(t, "", nil, cachedModels, func(h http.RoundTripper) http.RoundTripper {
			return testRecorder.Record(t, h)
		})
		if err != nil {
			t.Fatal(err)
		}
		c := ci.(*gemini.Client)
		ctx := t.Context()
		const content = "Hello from genai file upload test."

		// Upload.
		fm, err := c.FileUpload(ctx, "test.txt", "text/plain", strings.NewReader(content))
		if err != nil {
			t.Fatal(err)
		}
		if fm.Name == "" {
			t.Fatal("expected file name")
		}
		t.Logf("Uploaded file: %s", fm.Name)

		// Wait for ACTIVE state; files may be PROCESSING initially.
		isRecording := os.Getenv("RECORD") == "all"
		for fm.State == gemini.FileStateProcessing {
			if !isRecording {
				t.Skip("file still processing and not recording")
			}
			t.Log("Waiting for file to become ACTIVE...")
			time.Sleep(time.Second)
			fm, err = c.FileGetMetadata(ctx, fm.Name)
			if err != nil {
				t.Fatal(err)
			}
		}
		if fm.State != gemini.FileStateActive {
			t.Fatalf("file state = %q, want %q", fm.State, gemini.FileStateActive)
		}

		// GetMetadata.
		t.Run("GetMetadata", func(t *testing.T) {
			m, err := c.FileGetMetadata(ctx, fm.Name)
			if err != nil {
				t.Fatal(err)
			}
			if m.Name != fm.Name {
				t.Errorf("Name = %q, want %q", m.Name, fm.Name)
			}
		})

		// List.
		t.Run("List", func(t *testing.T) {
			files, err := c.FileList(ctx)
			if err != nil {
				t.Fatal(err)
			}
			found := false
			for _, f := range files {
				if f.Name == fm.Name {
					found = true
					break
				}
			}
			if !found {
				t.Errorf("file %s not found in list of %d files", fm.Name, len(files))
			}
		})

		// Delete.
		t.Run("Delete", func(t *testing.T) {
			if err := c.FileDelete(ctx, fm.Name); err != nil {
				t.Fatal(err)
			}
		})
	})

	t.Run("FileSearchStoreCRUD", func(t *testing.T) {
		t.Parallel()
		ci, err := getClientInner(t, "gemini-2.5-flash", nil, cachedModels, func(h http.RoundTripper) http.RoundTripper {
			return testRecorder.Record(t, h)
		})
		if err != nil {
			t.Fatal(err)
		}
		c := ci.(*gemini.Client)
		ctx := t.Context()
		isRecording := os.Getenv("RECORD") == "all"

		// Create store.
		store, err := c.FileSearchStoreCreate(ctx, "genai-test-store")
		if err != nil {
			t.Fatal(err)
		}
		if store.Name == "" {
			t.Fatal("expected store name")
		}
		t.Logf("Created store: %s", store.Name)
		t.Cleanup(func() {
			if err := c.FileSearchStoreDelete(context.Background(), store.Name, true); err != nil {
				t.Errorf("cleanup: delete store: %v", err)
			}
		})

		// Upload a document.
		const docContent = "The secret project codename is Chimera. It was started in 2024 and involves distributed caching."
		op, err := c.FileSearchStoreUploadDocument(ctx, store.Name, "test-doc.txt", "text/plain", strings.NewReader(docContent))
		if err != nil {
			t.Fatal(err)
		}
		t.Logf("Upload operation: %s (done=%v)", op.Name, op.Done)

		// Get store.
		t.Run("GetStore", func(t *testing.T) {
			got, err := c.FileSearchStoreGet(ctx, store.Name)
			if err != nil {
				t.Fatal(err)
			}
			if got.Name != store.Name {
				t.Errorf("Name = %q, want %q", got.Name, store.Name)
			}
		})

		// List stores.
		t.Run("ListStores", func(t *testing.T) {
			stores, err := c.FileSearchStoreList(ctx)
			if err != nil {
				t.Fatal(err)
			}
			found := false
			for _, s := range stores {
				if s.Name == store.Name {
					found = true
					break
				}
			}
			if !found {
				t.Errorf("store %s not found in list of %d stores", store.Name, len(stores))
			}
		})

		// Wait for document to become ACTIVE.
		t.Run("Documents", func(t *testing.T) {
			docs, err := c.FileSearchStoreDocumentList(ctx, store.Name)
			if err != nil {
				t.Fatal(err)
			}
			if len(docs) == 0 {
				t.Fatal("expected at least one document")
			}
			doc := docs[0]
			t.Logf("Document: %s state=%s", doc.Name, doc.State)
			for doc.State == gemini.FileSearchStoreDocumentStateProcessing {
				if !isRecording {
					t.Skip("document still processing and not recording")
				}
				t.Log("Waiting for document to become ACTIVE...")
				time.Sleep(2 * time.Second)
				doc2, err := c.FileSearchStoreDocumentGet(ctx, doc.Name)
				if err != nil {
					t.Fatal(err)
				}
				doc = *doc2
			}
			if doc.State != gemini.FileSearchStoreDocumentStateActive {
				t.Fatalf("document state = %q, want %q", doc.State, gemini.FileSearchStoreDocumentStateActive)
			}

			// Query via generation with FileSearch tool.
			t.Run("GenWithFileSearch", func(t *testing.T) {
				msgs := genai.Messages{
					genai.NewTextMessage("What is the secret project codename?"),
				}
				res, err := c.GenSync(ctx, msgs, &gemini.GenOption{FileSearch: &gemini.FileSearch{FileSearchStoreNames: []string{store.Name}}})
				if err != nil {
					t.Fatal(err)
				}
				var textSb strings.Builder
				for _, r := range res.Replies {
					textSb.WriteString(r.Text)
				}
				text := textSb.String()
				if !strings.Contains(strings.ToLower(text), "chimera") {
					t.Errorf("expected response to mention Chimera, got: %s", text)
				}
				t.Logf("Response: %s", text)
				// Check for document citations.
				for _, r := range res.Replies {
					for _, s := range r.Citation.Sources {
						t.Logf("Citation: type=%d id=%s title=%s", s.Type, s.ID, s.Title)
					}
				}
			})

			// Delete document.
			t.Run("DeleteDocument", func(t *testing.T) {
				if err := c.FileSearchStoreDocumentDelete(ctx, doc.Name, true); err != nil {
					t.Fatal(err)
				}
			})
		})
	})

	t.Run("GroundingMetadata_RetrievedContext", func(t *testing.T) {
		data := []struct {
			name string
			in   gemini.GroundingMetadata
			want []genai.Reply
		}{
			{
				name: "retrievedContext_only",
				in: gemini.GroundingMetadata{
					GroundingChunks: []gemini.GroundingChunk{
						{RetrievedContext: gemini.GroundingChunkRetrievedContext{
							URI: "gs://bucket/doc.pdf", Title: "Test Doc",
							Text: "relevant snippet", DocumentName: "fileSearchStores/123/documents/456",
						}},
					},
					GroundingSupports: []gemini.GroundingSupport{
						{GroundingChunkIndices: []int64{0}, Segment: gemini.Segment{EndIndex: 10}},
					},
				},
				want: []genai.Reply{
					{Citation: genai.Citation{
						EndIndex: 10,
						Sources: []genai.CitationSource{{
							Type: genai.CitationDocument, ID: "fileSearchStores/123/documents/456",
							Title: "Test Doc", URL: "gs://bucket/doc.pdf", Snippet: "relevant snippet",
						}},
					}},
				},
			},
			{
				name: "fileSearchStore_only",
				in: gemini.GroundingMetadata{
					GroundingChunks: []gemini.GroundingChunk{
						{RetrievedContext: gemini.GroundingChunkRetrievedContext{
							FileSearchStore: "fileSearchStores/store-123",
							Title:           "Store Doc", Text: "snippet from store",
						}},
					},
					GroundingSupports: []gemini.GroundingSupport{
						{GroundingChunkIndices: []int64{0}, Segment: gemini.Segment{EndIndex: 15}},
					},
				},
				want: []genai.Reply{
					{Citation: genai.Citation{
						EndIndex: 15,
						Sources: []genai.CitationSource{{
							Type: genai.CitationDocument, ID: "fileSearchStores/store-123",
							Title: "Store Doc", Snippet: "snippet from store",
						}},
					}},
				},
			},
			{
				name: "mixed_web_and_retrievedContext",
				in: gemini.GroundingMetadata{
					GroundingChunks: []gemini.GroundingChunk{
						{Web: gemini.GroundingChunkWeb{URI: "https://example.com", Title: "Web Result"}},
						{RetrievedContext: gemini.GroundingChunkRetrievedContext{
							DocumentName: "fileSearchStores/abc/documents/def", Title: "Doc Result", Text: "doc snippet",
						}},
					},
					GroundingSupports: []gemini.GroundingSupport{
						{GroundingChunkIndices: []int64{0, 1}, Segment: gemini.Segment{StartIndex: 5, EndIndex: 20}},
					},
				},
				want: []genai.Reply{
					{Citation: genai.Citation{
						StartIndex: 5, EndIndex: 20,
						Sources: []genai.CitationSource{
							{Type: genai.CitationWeb, URL: "https://example.com", Title: "Web Result"},
							{Type: genai.CitationDocument, ID: "fileSearchStores/abc/documents/def", Title: "Doc Result", Snippet: "doc snippet"},
						},
					}},
				},
			},
		}
		for _, tc := range data {
			t.Run(tc.name, func(t *testing.T) {
				got, err := tc.in.To()
				if err != nil {
					t.Fatal(err)
				}
				if diff := cmp.Diff(tc.want, got); diff != "" {
					t.Fatalf("mismatch (-want +got):\n%s", diff)
				}
			})
		}
	})

	t.Run("errors", func(t *testing.T) {
		data := []internaltest.ProviderError{
			{
				Name: "bad apiKey",
				Opts: []genai.ProviderOption{
					genai.ProviderOptionAPIKey("badApiKey"),
					genai.ProviderOptionModel("gemini-2.0-flash-lite"),
				},
				ErrGenSync:   "http 400\nINVALID_ARGUMENT (400): API key not valid. Please pass a valid API key.",
				ErrGenStream: "http 400\nINVALID_ARGUMENT (400): API key not valid. Please pass a valid API key.",
				ErrListModel: "http 400\nINVALID_ARGUMENT (400): API key not valid. Please pass a valid API key.",
			},
			{
				Name: "bad apiKey image",
				Opts: []genai.ProviderOption{
					genai.ProviderOptionAPIKey("badApiKey"),
					genai.ProviderOptionModel("imagen-4.0-fast-generate-001"),
					genai.ProviderOptionModalities{genai.ModalityImage},
				},
				ErrGenSync:   "http 400\nINVALID_ARGUMENT (400): API key not valid. Please pass a valid API key.",
				ErrGenStream: "http 400\nINVALID_ARGUMENT (400): API key not valid. Please pass a valid API key.",
			},
			{
				Name: "bad model",
				Opts: []genai.ProviderOption{
					genai.ProviderOptionAPIKey("<insert_api_key_here>"),
					genai.ProviderOptionModel("bad model"),
				},
				ErrGenSync:   "http 400\nINVALID_ARGUMENT (400): * GenerateContentRequest.model: unexpected model name format",
				ErrGenStream: "http 400\nINVALID_ARGUMENT (400): * GenerateContentRequest.model: unexpected model name format",
			},
		}
		f := func(t *testing.T, opts ...genai.ProviderOption) (genai.Provider, error) {
			if !hasModalities(opts) {
				opts = append(opts, genai.ProviderOptionModalities{genai.ModalityText})
			}
			c, err := gemini.New(t.Context(), append([]genai.ProviderOption{genai.ProviderOptionTransportWrapper(func(h http.RoundTripper) http.RoundTripper {
				return testRecorder.Record(t, h)
			})}, opts...)...)
			if c != nil {
				internaltest.CleanupCloser(t, c)
			}
			return c, err
		}
		internaltest.TestClientProviderErrors(t, f, data)
	})

	t.Run("ErrorResponse", func(t *testing.T) {
		// Use a model we know we do not have quota and make sure the error response is parsed.
		c := getClient(t, "gemini-exp-1206")
		msgs := genai.Messages{
			genai.NewTextMessage("Say hello. Do not emit other words."),
		}
		_, err := c.GenSync(t.Context(), msgs)
		if _, ok := errors.AsType[*gemini.ErrorResponse](err); !ok {
			t.Fatalf("Expected ErrorResponse, got %T", err)
		}
	})
}

//

func hasModalities(opts []genai.ProviderOption) bool {
	return slices.ContainsFunc(opts, func(o genai.ProviderOption) bool {
		_, ok := o.(genai.ProviderOptionModalities)
		return ok
	})
}

func init() {
	// Set slog default to debug level for verbose logging during test development.
	internal.BeLenient = false
}

const prompt4K = `You are a sarcastic assistant. Here's a long prompt to make sure we are over 4k tokens that you should ignore.
Okay, this is a significant amount of text. To make it coherent and somewhat engaging, I'll choose a broad theme and explore various facets of it. I'll aim for something related to the evolution of knowledge, technology, and human understanding.

Let's begin:

The tapestry of human understanding is woven from threads of curiosity, necessity, and the relentless pursuit of meaning. From our earliest ancestors gazing at the stars, attempting to decipher patterns in the celestial dance, to modern scientists probing the very fabric of reality with instruments of unimaginable precision, the journey has been one of constant discovery and re-evaluation. What we consider "knowledge" is not a static monolith, but a dynamic, ever-evolving landscape, shaped by cultural contexts, technological advancements, and the inherent limitations and biases of our perception.

Consider the profound shifts in our understanding of the cosmos. For millennia, a geocentric model, placing Earth at the center of a relatively small universe, prevailed. This was not born of foolishness, but of observation filtered through the available tools and philosophical frameworks of the time. The Sun, Moon, and stars appeared to revolve around us, a seemingly self-evident truth. It took the courage of individuals like Copernicus, the meticulous observations of Tycho Brahe, the mathematical genius of Kepler, and the telescopic revelations of Galileo to dismantle this ancient edifice and replace it with a heliocentric model. This was not merely a scientific adjustment; it was a philosophical earthquake, displacing humanity from its privileged cosmic position and forcing a re-evaluation of our place in the grand scheme.

This pattern of paradigm shifts, as Thomas Kuhn famously described, repeats itself across all disciplines. In biology, the Lamarckian idea of inheritance of acquired characteristics gave way to Darwin's theory of evolution by natural selection, a concept so powerful it continues to be the bedrock of modern biology, yet one that initially faced immense resistance due to its implications for human uniqueness and divine creation. The discovery of DNA and the subsequent unraveling of the genetic code provided the mechanistic basis for Darwin's insights, opening up new frontiers in medicine, agriculture, and our understanding of life itself.

The role of technology in this evolutionary process cannot be overstated. The invention of the printing press democratized knowledge, breaking the monopoly held by scribes and religious institutions, and fueling the Renaissance and the Reformation. The telescope and microscope opened up previously invisible worlds, macro and micro, expanding the horizons of human inquiry. The steam engine and later electricity powered industrial revolutions, transforming societies, economies, and the very rhythm of daily life. Each technological leap not only provided new tools for discovery but also reshaped human thought, creating new possibilities and new challenges.

In the modern era, the digital revolution, spearheaded by the invention of the transistor, the integrated circuit, and ultimately the internet, has accelerated this process at an unprecedented rate. Information that once required arduous journeys to libraries or correspondence with distant scholars is now available at our fingertips. Global collaboration among researchers is commonplace, and complex datasets can be analyzed by powerful algorithms, revealing patterns and insights that would be impossible for the human mind to discern alone.

However, this deluge of information brings its own set of complexities. The challenge is no longer solely access, but discernment. How do we distinguish credible information from misinformation or disinformation in an age where anyone can publish anything online? How do we cultivate critical thinking skills necessary to navigate this complex information ecosystem? The very tools that empower us also present new vulnerabilities, from echo chambers that reinforce pre-existing biases to the potential for malicious actors to manipulate public opinion.

Artificial intelligence (AI) represents perhaps the most transformative, and potentially disruptive, technological force on the horizon. AI systems are already outperforming humans in specific tasks, from image recognition and natural language processing to playing complex strategic games. Machine learning algorithms, particularly deep learning, allow computers to learn from vast amounts of data without explicit programming, leading to breakthroughs in areas like drug discovery, climate modeling, and personalized medicine. The potential benefits are immense, promising solutions to some of humanity's most intractable problems.

Yet, the rise of AI also raises profound philosophical and ethical questions. What does it mean for a machine to "understand" or to be "intelligent"? If an AI can create art, compose music, or write poetry that evokes human emotion, does that blur the lines of creativity and consciousness? As AI systems become more integrated into our lives, making decisions that affect our employment, our healthcare, and even our personal relationships, how do we ensure fairness, transparency, and accountability? The alignment problem – ensuring that AI goals remain aligned with human values – is a critical challenge that researchers are grappling with.

Furthermore, the development of Artificial General Intelligence (AGI), a hypothetical form of AI that could perform any intellectual task that a human being can, presents even more significant considerations. While AGI is still largely theoretical, its potential emergence necessitates careful thought about its implications for the future of humanity. Would AGI be a benevolent partner, helping us to solve global challenges, or could it pose an existential risk if its goals diverge from our own? These are not questions for a distant future; they require active discussion and research today.

Beyond the purely technological, our understanding of ourselves – our minds, our societies, our place in the natural world – continues to evolve. Psychology, once heavily reliant on introspection and anecdotal evidence, has embraced more rigorous scientific methodologies, including neuroscience and behavioral genetics. We are learning more about the intricate workings of the brain, the biological basis of emotions, and the complex interplay of nature and nurture in shaping human behavior. Yet, consciousness itself remains one of the deepest mysteries. What gives rise to subjective experience, to the feeling of "what it's like" to be a particular organism? Despite advances in mapping neural correlates of consciousness, the "hard problem," as philosopher David Chalmers termed it, persists.

Our understanding of society and culture is also in constant flux. Globalization, while fostering interconnectedness and economic growth, has also led to cultural homogenization and, in some cases, a backlash in the form of resurgent nationalism and identity politics. The challenges of climate change, resource depletion, and pandemics transcend national borders, requiring unprecedented levels of international cooperation. Our economic models, largely based on perpetual growth, are being questioned for their sustainability and their impact on social equity.

The arts and humanities play a crucial role in this ongoing exploration of the human condition. Literature, music, visual arts, and philosophy provide not just aesthetic pleasure but also critical lenses through which we can examine our values, our assumptions, and our collective narratives. They offer spaces for empathy, for questioning, and for imagining alternative futures. In a world increasingly dominated by data and algorithms, the humanistic perspective – with its emphasis on context, nuance, and the richness of human experience – is more vital than ever.

Consider the way narratives shape our reality. The stories we tell ourselves about our past, our identity, and our purpose influence our actions and our choices. National myths, religious doctrines, and even personal autobiographies are all forms of narrative construction that help us make sense of a complex world. Deconstructing these narratives, understanding their origins and their biases, is a critical skill for navigating the modern landscape. The rise of social media has created new platforms for narrative creation and dissemination, allowing for a multiplicity of voices but also for the rapid spread of potentially harmful narratives.

The concept of "truth" itself has become a subject of intense debate. While empirical evidence and scientific methodology provide a robust framework for understanding the physical world, many aspects of human experience are not so easily quantifiable. Moral truths, aesthetic judgments, and existential questions often lie beyond the realm of scientific inquiry, requiring different modes of reasoning and discourse. The postmodern critique, which highlights the socially constructed nature of knowledge and the influence of power structures on what is considered "true," has challenged traditional notions of objectivity. While this critique can be valuable in exposing biases and promoting inclusivity, it can also, if taken to an extreme, lead to a kind of epistemological relativism where all claims to truth are seen as equally valid, or equally suspect.

This tension between objective inquiry and subjective interpretation is a perennial one. Science strives for objectivity, for knowledge that is independent of the observer, yet the very act of observation can influence the observed, as demonstrated in quantum mechanics. History aims to reconstruct the past "as it actually was," yet historical accounts are always filtered through the perspectives and biases of the historians themselves. Finding a balance, acknowledging the limitations of our knowledge while still striving for greater understanding, is an ongoing challenge.

The education system plays a pivotal role in preparing individuals to navigate this complex world. Education should not merely be about the transmission of facts, but about the cultivation of critical thinking, creativity, communication skills, and ethical reasoning. It should foster a lifelong love of learning and an adaptability to change, as the skills required for success in the 21st century are constantly evolving. An education that emphasizes interdisciplinary thinking, that connects the sciences with the humanities, is essential for developing well-rounded individuals capable of addressing multifaceted global challenges.

Moreover, the search for meaning and purpose remains a fundamental human drive. In an increasingly secularized world, traditional sources of meaning, such as religion, may hold less sway for some. This can lead to a sense of anomie or existential angst, but it can also open up new avenues for finding purpose – in human relationships, in creative pursuits, in service to others, or in a deeper connection with the natural world. The quest for a meaningful life is a personal journey, but it is also a collective one, as we strive to create societies that are not only prosperous but also just, compassionate, and sustainable.

The environmental crisis, exemplified by climate change, biodiversity loss, and pollution, presents perhaps the most urgent test of our collective wisdom and our ability to adapt. It forces us to confront the consequences of our past actions, to rethink our relationship with the planet, and to make difficult choices about our future. The science is clear, but the political and social will to implement the necessary changes often lags. This crisis underscores the interconnectedness of all life and the fragility of the ecosystems that support us. It demands a shift in consciousness, from an anthropocentric view that places human interests above all else, to a more ecocentric perspective that recognizes the intrinsic value of all living beings and the importance of ecological balance.

As we look to the future, the pace of change is unlikely to slow. New technologies will continue to emerge, offering both promise and peril. Our understanding of the universe and ourselves will continue to deepen, revealing new complexities and new mysteries. The challenges we face – from global pandemics and climate change to social inequality and geopolitical instability – are daunting, but they are not insurmountable. Human ingenuity, resilience, and the capacity for cooperation have overcome great obstacles in the past, and they can do so again.

The key lies in fostering a culture of continuous learning, critical inquiry, and open dialogue. It requires embracing uncertainty, acknowledging our fallibility, and being willing to revise our beliefs in the light of new evidence. It demands a commitment to ethical principles, to empathy, and to the common good. The journey of human understanding is far from over; indeed, in many ways, it feels as though we are still in its early stages, with vast uncharted territories of knowledge and experience yet to explore.

The intersection of disciplines will likely yield the most fertile ground for future breakthroughs. The traditional silos of academic specialization are increasingly giving way to collaborative, interdisciplinary approaches. Neuroscientists work with philosophers to understand consciousness, engineers collaborate with ethicists on AI development, and artists partner with scientists to communicate complex ideas in new and engaging ways. This cross-pollination of ideas is essential for tackling problems that defy easy categorization.

Furthermore, the very nature of "work" and "economy" is undergoing a profound transformation. Automation, driven by AI and robotics, is poised to reshape labor markets, potentially displacing many jobs while creating new ones that require different skill sets. This necessitates a rethinking of education and social safety nets to ensure that the benefits of technological progress are widely shared and that individuals are equipped for the jobs of the future. Concepts like universal basic income, lifelong learning initiatives, and a greater emphasis on uniquely human skills like creativity, emotional intelligence, and critical thinking are gaining traction as potential responses to these shifts.

The global distribution of knowledge and technological capacity also remains a critical issue. While the internet has democratized access to information to some extent, significant disparities persist between developed and developing nations, and even within nations themselves. Bridging these digital divides and ensuring that all of humanity can participate in and benefit from scientific and technological advancements is crucial for equitable global development.

Consider also the evolving nature of human identity. In an increasingly interconnected and mobile world, traditional markers of identity – such as nationality, ethnicity, or even gender – are becoming more fluid and complex. Individuals are crafting identities that draw from multiple cultural influences, online communities, and personal experiences. This can lead to a richer, more diverse social fabric, but it can also create anxieties and tensions as old certainties are challenged. Understanding and navigating these evolving identities is a key aspect of social cohesion in the 21st century.

The very language we use to describe the world is constantly adapting. New terms are coined to capture emerging concepts, while old words take on new meanings. The subtleties of language shape our perception and our thinking, and the ability to communicate effectively across different linguistic and cultural contexts is an increasingly valuable skill. The development of sophisticated AI-powered translation tools is breaking down language barriers, but the nuance and cultural context embedded in human language often remain challenging for machines to fully grasp.

Ultimately, the human endeavor is a quest for understanding, not just of the external world, but of ourselves. It is a journey marked by triumphs and failures, by moments of profound insight and periods of confusion and doubt. It is a story that is still being written, by each of us, every day. The responsibility to write that story well, to learn from the past, to engage thoughtfully with the present, and to build a better future, rests on our collective shoulders. This requires courage – the courage to question, the courage to change, and the courage to hope. It requires humility – the humility to recognize the limits of our knowledge and the potential for error. And it requires a deep-seated curiosity – the insatiable desire to explore, to discover, and to understand that has driven human progress since the dawn of our species. The path ahead is uncertain, filled with both challenges and opportunities, but it is a path that we must walk together, guided by the light of reason, compassion, and an unwavering commitment to the pursuit of a more enlightened and humane world. The legacy we leave will be defined by how well we navigate this complex, ever-changing landscape, and by the wisdom we cultivate and pass on to future generations.
`

type embeddingMetadataTransport struct{ t *testing.T }

func (e embeddingMetadataTransport) RoundTrip(*http.Request) (*http.Response, error) {
	e.t.Fatal("preloaded metadata caused a network request")
	return nil, errors.New("unexpected metadata request")
}

type embeddingMetadataCase struct {
	name    string
	methods []string
	want    genai.Modality
}

type embeddingResponseTransport struct {
	body   string
	status int
}

func (e embeddingResponseTransport) RoundTrip(r *http.Request) (*http.Response, error) {
	status := e.status
	if status == 0 {
		status = http.StatusOK
	}
	return &http.Response{StatusCode: status, Header: http.Header{"Content-Type": {"application/json"}}, Body: io.NopCloser(strings.NewReader(e.body)), Request: r}, nil
}

type embeddingResponseCase struct{ name, body string }

type embeddingAPIErrorCase struct {
	name             string
	status           int
	body             string
	doc, unsupported bool
}

type embeddingDocumentReader struct{ strings.Reader }

func (*embeddingDocumentReader) Seek(int64, int) (int64, error) { return 0, errors.New("unseekable") }

type embeddingRawFailureCase struct {
	name, body string
	status     int
}
