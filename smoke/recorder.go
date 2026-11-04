// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Package smoke defines probe families and the recorder they measure through.
package smoke

import (
	"net/http"
	"testing"

	"github.com/maruel/genai"
	"github.com/maruel/genai/internal/myrecorder"
	"github.com/maruel/genai/scoreboard"
)

// ClientFactory creates a client for one model, wrapping its HTTP transport when
// fn is not nil.
//
// The factory must register client cleanup with t.
type ClientFactory func(t testing.TB, m scoreboard.Model, fn func(http.RoundTripper) http.RoundTripper) genai.Provider

// Recorder replays or records the provider a probe talks to.
//
// It hides where a wire attaches, so the harness never learns which wire it is
// dealing with beyond asking. [HTTP] wraps the transport, [HTTPSession] does the same
// and records sessions inside the provider, [Subprocess] wraps the process, and a
// WebSocket wire would proxy the frames. A provider test implements one, which is why
// a wire the recorder does not declare fails loudly instead of replaying nothing.
type Recorder interface {
	// Wires reports the wires this recorder serves.
	Wires() []scoreboard.Wire
	// Provider returns a provider whose exchanges replay or record under name.
	//
	// name is a fixture name, not a label: it comes from the test and the model, never
	// from [scoreboard.Key.String]. A wire that records per session numbers its own
	// fixtures and ignores it.
	Provider(w scoreboard.Wire, name string) genai.Provider
}

// WebSocket records JSON frames over a WebSocket, one fixture per scenario name.
//
// The factory builds the client the proxy serves: it starts a proxy, points the client
// at it, and returns the connection.
func WebSocket(pf SubprocessFactory) RecorderFactory {
	return func(t testing.TB, m scoreboard.Model) Recorder {
		return &webSocketRecorder{t: t, model: m, pf: pf}
	}
}

type webSocketRecorder struct {
	t     testing.TB
	model scoreboard.Model
	pf    SubprocessFactory
}

// Wires implements [Recorder].
func (r *webSocketRecorder) Wires() []scoreboard.Wire {
	return []scoreboard.Wire{scoreboard.WireWebSocket}
}

// Provider implements [Recorder].
func (r *webSocketRecorder) Provider(w scoreboard.Wire, name string) genai.Provider {
	if w != scoreboard.WireWebSocket {
		r.t.Fatalf("WebSocket recorder does not serve wire %q", w)
	}
	if name == "" {
		return r.pf(r.t, r.model, "")
	}
	return r.pf(r.t, r.model, r.t.Name()+"/"+name)
}

// SubprocessFactory creates a client whose recordings use name.
//
// An empty name asks for the provider without a recording, which a test uses to
// ask what a model supports without replaying a probe.
type SubprocessFactory func(t testing.TB, m scoreboard.Model, name string) genai.Provider

// RecorderFactory returns the recorder for one model.
type RecorderFactory func(t testing.TB, m scoreboard.Model) Recorder

// HTTP records HTTP exchanges, one cassette per scenario name.
func HTTP(rec *myrecorder.Records, pf ClientFactory) RecorderFactory {
	return func(t testing.TB, m scoreboard.Model) Recorder {
		return &httpRecorder{t: t, model: m, rec: rec, pf: pf}
	}
}

// HTTPSession records HTTP exchanges per scenario name, and sessions inside the
// provider, which the factory builds with its own session recorder.
func HTTPSession(rec *myrecorder.Records, pf ClientFactory) RecorderFactory {
	return func(t testing.TB, m scoreboard.Model) Recorder {
		return &httpRecorder{t: t, model: m, rec: rec, pf: pf, session: true}
	}
}

// Subprocess records command line exchanges, one trace per scenario name.
func Subprocess(pf SubprocessFactory) RecorderFactory {
	return func(t testing.TB, m scoreboard.Model) Recorder {
		return &subprocessRecorder{t: t, model: m, pf: pf}
	}
}

type subprocessRecorder struct {
	t     testing.TB
	model scoreboard.Model
	pf    SubprocessFactory
}

// Wires implements [Recorder].
func (r *subprocessRecorder) Wires() []scoreboard.Wire {
	return []scoreboard.Wire{scoreboard.WireSubprocess}
}

// Provider implements [Recorder].
func (r *subprocessRecorder) Provider(w scoreboard.Wire, name string) genai.Provider {
	if w != scoreboard.WireSubprocess {
		r.t.Fatalf("subprocess recorder does not serve wire %q", w)
	}
	if name == "" {
		return r.pf(r.t, r.model, "")
	}
	return r.pf(r.t, r.model, r.t.Name()+"/"+name)
}

type httpRecorder struct {
	t       testing.TB
	model   scoreboard.Model
	rec     *myrecorder.Records
	pf      ClientFactory
	session bool
}

func (r *httpRecorder) Wires() []scoreboard.Wire {
	if r.session {
		return []scoreboard.Wire{scoreboard.WireHTTP, scoreboard.WireSession}
	}
	return []scoreboard.Wire{scoreboard.WireHTTP}
}

func (r *httpRecorder) Provider(w scoreboard.Wire, name string) genai.Provider {
	switch w {
	case scoreboard.WireHTTP:
		return r.pf(r.t, r.model, func(h http.RoundTripper) http.RoundTripper {
			if name == "" {
				return h
			}
			cassette, err := r.rec.Record(r.t.Name()+"/"+name, h)
			if err != nil {
				r.t.Fatal(err)
			}
			r.t.Cleanup(func() {
				if err := cassette.Stop(); err != nil {
					r.t.Error(err)
				}
			})
			return cassette
		})
	case scoreboard.WireSession:
		if r.session {
			return r.pf(r.t, r.model, nil)
		}
	case scoreboard.WireWebSocket, scoreboard.WireSubprocess:
	}
	r.t.Fatalf("recorder does not serve wire %q", w)
	return nil
}

// Direct gives the probes the providers of pf without recording anything.
//
// It is what a caller outside a test uses: cmd/scoreboard measures a live or
// already wrapped provider, and nothing replays it.
func Direct(pf ProviderFactory) Recorder {
	return directRecorder{pf: pf}
}

type directRecorder struct {
	pf ProviderFactory
}

func (directRecorder) Wires() []scoreboard.Wire {
	return []scoreboard.Wire{scoreboard.WireHTTP, scoreboard.WireSession}
}

func (r directRecorder) Provider(w scoreboard.Wire, name string) genai.Provider {
	return r.pf(name)
}
