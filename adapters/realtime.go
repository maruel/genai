// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Realtime tool-call handling.

package adapters

import (
	"context"
	"fmt"
	"iter"
	"slices"
	"sync"

	"github.com/maruel/genai"
)

type realtimeToolSession struct {
	genai.RealtimeSession

	mu    sync.RWMutex
	tools []genai.ToolDef
}

// Send implements genai.RealtimeSession.
func (s *realtimeToolSession) Send(ctx context.Context, in genai.RealtimeInput) error {
	if err := s.RealtimeSession.Send(ctx, in); err != nil {
		return err
	}
	if update, ok := in.(genai.RealtimeUpdate); ok && update.Tools != nil {
		s.mu.Lock()
		s.tools = cloneToolDefs(*update.Tools)
		s.mu.Unlock()
	}
	return nil
}

// Receive implements genai.RealtimeSession.
func (s *realtimeToolSession) Receive(ctx context.Context) iter.Seq2[genai.RealtimeEvent, error] {
	return func(yield func(genai.RealtimeEvent, error) bool) {
		for e, err := range s.RealtimeSession.Receive(ctx) {
			if err != nil {
				yield(e, err)
				return
			}
			if !yield(e, nil) {
				return
			}
			if e.Kind == genai.RealtimeEventContentToolCalls {
				if err := s.runTools(ctx, e.Content.ToolCalls); err != nil {
					yield(genai.RealtimeEvent{}, err)
					return
				}
			}
		}
	}
}

func (s *realtimeToolSession) runTools(ctx context.Context, calls []genai.ToolCall) error {
	if len(calls) == 0 {
		return nil
	}
	s.mu.RLock()
	tools := cloneToolDefs(s.tools)
	s.mu.RUnlock()
	results := make([]genai.ToolCallResult, len(calls))
	for i := range calls {
		call := &calls[i]
		found := false
		for j := range tools {
			if tools[j].Name == call.Name {
				found = true
				if tools[j].Callback == nil {
					return fmt.Errorf("realtime tool %q has no callback", call.Name)
				}
				break
			}
		}
		if !found {
			return fmt.Errorf("realtime tool %q is not declared", call.Name)
		}
		result, err := call.Call(ctx, tools)
		if err != nil {
			return fmt.Errorf("realtime tool %q: %w", call.Name, err)
		}
		results[i] = genai.ToolCallResult{ID: call.ID, Name: call.Name, Result: result}
	}
	if err := s.Send(ctx, genai.RealtimeInputContent{ToolResults: results, Generate: true}); err != nil {
		return fmt.Errorf("realtime tool results: %w", err)
	}
	return nil
}

// RealtimeWithToolCallLoop wraps a realtime session and executes tool calls in a
// loop. The caller still owns the event loop and decides when to stop.
//
// It updates its tool declarations after a successful RealtimeUpdate, so
// later tool calls use the currently configured tools. Tools without a callback
// are rejected when called.
func RealtimeWithToolCallLoop(s genai.RealtimeSession, tools []genai.ToolDef) genai.RealtimeSession {
	return &realtimeToolSession{RealtimeSession: s, tools: cloneToolDefs(tools)}
}

func cloneToolDefs(in []genai.ToolDef) []genai.ToolDef {
	out := slices.Clone(in)
	for i := range out {
		out[i].InputSchemaOverride = slices.Clone(out[i].InputSchemaOverride)
	}
	return out
}

var _ genai.RealtimeSession = (*realtimeToolSession)(nil)
