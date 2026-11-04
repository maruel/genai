// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Package smoke defines probe families and the recorder they measure through.
//
// A provider test implements one Recorder, which hides where each wire attaches. A
// family owns the probes of one scoreboard section and the rule that decides whether a
// model can serve it. The harness asks the recorder which wire to use and fills each
// section, and the scoreboard merges what was measured; none of them knows the others'
// business.
package smoke

import (
	"context"
	"errors"
	"fmt"
	"slices"

	"github.com/maruel/genai"
	"github.com/maruel/genai/scoreboard"
)

// Family measures one probe suite for one model, and fills one scoreboard section.
//
// A family owns what only it knows: which wires can carry it, when a model can serve
// it, and how to measure one row. The scoreboard section carries the rest, so a family
// never has to decide which suite a row belongs to.
//
// Chat fills [scoreboard.Score.Chat], over HTTP, WebSocket or a subprocess. Embed
// fills [scoreboard.Score.Embed], over HTTP or a subprocess. SystemOne fills
// [scoreboard.Score.SystemOne] and Realtime fills [scoreboard.Score.Realtime], each
// over its own wire alone.
type Family[T any] interface {
	// Wires reports the wires that can carry this family.
	Wires() []scoreboard.Wire
	// Fits reports whether the model can serve this family. spec tells what the model
	// offers; the section it fills tells whether the model was already measured, and
	// the rest of the scoreboard tells what it supports.
	Fits(spec Spec, sb *scoreboard.Score, model string) bool
	// Run measures one row over wire. An empty want asks the family to discover
	// whether the model serves it.
	Run(ctx context.Context, rec Recorder, m scoreboard.Model, wire scoreboard.Wire, want *T) (T, genai.Usage, error)
}

// ErrUnsupported reports that a model turns a family down, so no row is recorded.
var ErrUnsupported = errors.New("model does not support the probes")

// Chat measures GenSync, GenStream and GenAsync.
//
// It is the only family more than one wire carries, so one model can hold an HTTP row
// and a WebSocket row: the wire on the row tells them apart.
type Chat struct{}

// Wires implements [Family].
func (Chat) Wires() []scoreboard.Wire {
	return []scoreboard.Wire{scoreboard.WireHTTP, scoreboard.WireWebSocket, scoreboard.WireSubprocess}
}

// Fits implements [Family].
//
// A model whose section holds a row has an HTTP surface. A model without one is
// asked: an embedding or decision model answers no text prompt.
func (Chat) Fits(spec Spec, sb *scoreboard.Score, model string) bool {
	for i := range sb.Chat {
		if slices.Contains(sb.Chat[i].Models, model) {
			return true
		}
	}
	return !slices.Contains(spec.Modalities, genai.ModalityEmbedding) && !slices.Contains(spec.Modalities, genai.ModalityDecision)
}

// Run implements [Family].
func (Chat) Run(ctx context.Context, rec Recorder, _ scoreboard.Model, wire scoreboard.Wire, _ *scoreboard.Chat) (scoreboard.Chat, genai.Usage, error) {
	return runSync(ctx, func(name string) genai.Provider { return rec.Provider(wire, name) })
}

// Embed measures text and media embeddings.
type Embed struct{}

// Wires implements [Family].
func (Embed) Wires() []scoreboard.Wire {
	return []scoreboard.Wire{scoreboard.WireHTTP, scoreboard.WireSubprocess}
}

// Fits implements [Family].
//
// A model reports embeddings alone, or its section holds a row that measured
// them.
func (Embed) Fits(spec Spec, sb *scoreboard.Score, model string) bool {
	for i := range sb.Embed {
		if slices.Contains(sb.Embed[i].Models, model) {
			return true
		}
	}
	return slices.Equal(spec.Modalities, genai.Modalities{genai.ModalityEmbedding})
}

// Run implements [Family].
func (Embed) Run(ctx context.Context, rec Recorder, m scoreboard.Model, wire scoreboard.Wire, _ *scoreboard.Embed) (scoreboard.Embed, genai.Usage, error) {
	sc, usage, err := RunEmbeddings(ctx, func(name string) genai.Provider { return rec.Provider(wire, name) }, EmbeddingProbeDimensions)
	if err != nil {
		return sc, usage, err
	}
	sc.Models = []string{m.Model}
	return sc, usage, nil
}

// SystemOne measures typed decisions.
type SystemOne struct{}

// Wires implements [Family].
func (SystemOne) Wires() []scoreboard.Wire {
	return []scoreboard.Wire{scoreboard.WireHTTP}
}

// Fits implements [Family].
func (SystemOne) Fits(spec Spec, sb *scoreboard.Score, model string) bool {
	for i := range sb.SystemOne {
		if slices.Contains(sb.SystemOne[i].Models, model) {
			return true
		}
	}
	return spec.Caps.SystemOne && (len(spec.Modalities) == 0 || slices.Contains(spec.Modalities, genai.ModalityDecision))
}

// Run implements [Family].
func (SystemOne) Run(ctx context.Context, rec Recorder, m scoreboard.Model, wire scoreboard.Wire, _ *scoreboard.SystemOne) (scoreboard.SystemOne, genai.Usage, error) {
	sc, usage, err := RunSystemOne(ctx, func(name string) genai.Provider { return rec.Provider(wire, name) })
	if err != nil {
		return sc, usage, err
	}
	sc.Models = []string{m.Model}
	return sc, usage, nil
}

// Realtime measures a realtime session.
type Realtime struct{}

// Wires implements [Family].
func (Realtime) Wires() []scoreboard.Wire {
	return []scoreboard.Wire{scoreboard.WireSession}
}

// Fits implements [Family].
//
// A realtime session returns speech or text, so an image or embedding model is
// out of scope even when the provider serves sessions. A model that already has
// a realtime row is measured, not discovered.
func (Realtime) Fits(spec Spec, sb *scoreboard.Score, model string) bool {
	if !spec.Caps.Realtime {
		// A provider that serves no session is not asked; asking would record a
		// refusal for every chat model it has.
		return false
	}
	speech := false
	for i := range sb.Realtime {
		r := &sb.Realtime[i]
		if !slices.Contains(r.Models, model) {
			continue
		}
		if r.Realtime != nil {
			return false
		}
		for mod := range r.Out {
			if mod == genai.ModalityText || mod == genai.ModalityAudio {
				speech = true
			}
		}
	}
	for i := range sb.Chat {
		r := &sb.Chat[i]
		if !slices.Contains(r.Models, model) {
			continue
		}
		for mod := range r.Out {
			if mod == genai.ModalityText || mod == genai.ModalityAudio {
				speech = true
			}
		}
	}
	return speech
}

// Run implements [Family].
//
// A row the scoreboard already holds is measured. A model without one is asked
// first: only a session tells whether it accepts one.
func (Realtime) Run(ctx context.Context, rec Recorder, _ scoreboard.Model, wire scoreboard.Wire, want *scoreboard.Realtime) (scoreboard.Realtime, genai.Usage, error) {
	c := rec.Provider(wire, "")
	if want != nil && want.Realtime != nil {
		return RunRealtimeScenario(ctx, c, want.Reason)
	}
	ok, reason, err := RealtimeSupported(ctx, c)
	if err != nil {
		return scoreboard.Realtime{}, genai.Usage{}, err
	}
	if !ok {
		return scoreboard.Realtime{}, genai.Usage{}, fmt.Errorf("%w: %s", ErrUnsupported, reason)
	}
	return RunRealtimeScenario(ctx, c, false)
}
