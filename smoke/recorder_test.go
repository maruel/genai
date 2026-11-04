// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed by the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Tests for the recorder seam.

package smoke

import (
	"net/http"
	"testing"

	"github.com/maruel/genai"
	"github.com/maruel/genai/internal/myrecorder"
	"github.com/maruel/genai/scoreboard"
)

// stubProvider answers the questions the recorder asks; anything else panics.
type stubProvider struct {
	genai.Provider
}

func (stubProvider) Scoreboard() scoreboard.Score             { return scoreboard.Score{} }
func (stubProvider) OutputModalities() genai.Modalities       { return nil }
func (stubProvider) Capabilities() genai.ProviderCapabilities { return genai.ProviderCapabilities{} }

func TestHTTPSpecRecordsNothing(t *testing.T) {
	// Score and Spec answer what the harness asks before it measures, so they must not
	// open a cassette; a probe must.
	var wrappers []func(http.RoundTripper) http.RoundTripper
	rec, err := myrecorder.NewRecords(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	factory := ClientFactory(func(_ testing.TB, _ scoreboard.Model, fn func(http.RoundTripper) http.RoundTripper) genai.Provider {
		wrappers = append(wrappers, fn)
		return stubProvider{}
	})
	r := HTTP(rec, factory)(t, scoreboard.Model{Model: "m"})
	_ = r.Score()
	_ = r.Spec(scoreboard.Model{Model: "m"})
	_ = r.Provider(scoreboard.WireHTTP, "GenSync")
	if len(wrappers) != 3 {
		t.Fatalf("factory calls: %d", len(wrappers))
	}
	if wrappers[0] != nil || wrappers[1] != nil {
		t.Fatal("Score or Spec opened a recording")
	}
	if wrappers[2] == nil {
		t.Fatal("a probe did not record")
	}
}
