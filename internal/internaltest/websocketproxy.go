// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// WebSocket wire recording proxy helpers.

package internaltest

import (
	"errors"
	"io/fs"
	"os"
	"path/filepath"
	"testing"

	"github.com/maruel/genai/websocketrecord"
)

// NewWebSocketProxy records or replays the frames of one WebSocket wire.
//
// The fixture lives at testdata/<name>.ndjson. When RECORD is "all", the fixture is
// recorded again; when it is "failure_only", only a missing or empty fixture is
// recorded. Otherwise the fixture is replayed, and a missing fixture fails the test:
// an empty frame stream would read as a refusal instead of a gap.
//
// The returned recorder serves the proxy the client dials; target is the base URL the
// client is configured with, and the proxy forwards each dial under it.
func NewWebSocketProxy(t testing.TB, name, target string) *websocketrecord.Recorder {
	fixture := filepath.Join("testdata", name)
	markRecording(t, fixture+".ndjson")
	_, err := os.Stat(fixture + ".ndjson")
	exists := err == nil
	switch rec := os.Getenv("RECORD"); rec {
	case "all", "failure_only":
		if rec == "all" || !exists {
			_ = os.Remove(fixture + ".ndjson")
		}
	default:
		if !exists {
			if err != nil && !errors.Is(err, fs.ErrNotExist) {
				t.Fatalf("stat WebSocket recording: %v", err)
			}
			t.Fatalf("missing WebSocket recording %s.ndjson; record it with RECORD=failure_only scoped to this model", fixture)
		}
	}
	rec, err := websocketrecord.New(fixture)
	if err != nil {
		t.Fatal(err)
	}
	if err := rec.Start(target); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		// Keep what was recorded even when a probe failed: the fixture replays the
		// same frames, and a missing one is fatal. Only RECORD=all replaces it.
		if err := rec.Close(); err != nil {
			t.Error(err)
		}
	})
	return rec
}
