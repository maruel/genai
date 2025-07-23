// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Package scoreboardtest runs a scoreboard in test mode.
package scoreboardtest

import (
	"net/http"
	"slices"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/maruel/genai"
	"github.com/maruel/genai/internal/internaltest"
	"github.com/maruel/genai/scoreboard"
)

// GetClient returns a provider client for a specific model.
type GetClient func(t testing.TB, scenarioName string) (genai.Provider, http.RoundTripper)

// RunOneModel runs the scoreboard on one model.
//
// It must implement genai.ProviderScoreboard. If it is wrapped, the wrappers must implement
// genai.ProviderUnwrap.
func RunOneModel(t testing.TB, gc GetClient) genai.Usage {
	// Find the reference.
	var want genai.Scenario
	cc, _ := gc(t, "")
	id := cc.ModelID()
	og := cc
	for {
		if u, ok := og.(genai.ProviderUnwrap); ok {
			og = u.Unwrap()
		} else {
			break
		}
	}
	for _, sc := range og.(genai.ProviderScoreboard).Scoreboard().Scenarios {
		if slices.Contains(sc.Models, id) {
			want = sc
			want.Models = []string{id}
			break
		}
	}
	if len(want.Models) == 0 {
		t.Fatalf("no scenario for model %q", id)
	}
	if want.In == nil && want.Out == nil {
		t.Skip("Explicitly unsupported model")
	}

	// Calculate the scenario.
	providerFactory := func(name string) (genai.Provider, http.RoundTripper) {
		if name == "" {
			p, rt := gc(t, name)
			return p, rt
		}
		p, rt := gc(t, t.Name()+"/"+name)
		return p, rt
	}
	ctx, _ := internaltest.Log(t)
	got, usage, err := scoreboard.CreateScenario(ctx, providerFactory)
	t.Logf("Usage: %#v", usage)
	if err != nil {
		t.Fatalf("CreateScenario failed: %v", err)
	}

	// Check if valid.
	if diff := cmp.Diff(want, got, opt); diff != "" {
		t.Errorf("mismatch (-want +got):\n%s", diff)
	}
	return usage
}

//

var opt = cmp.Comparer(func(x, y genai.TriState) bool {
	// TODO: Make this more solid. This requires a better assessment of what "Flaky" is.
	if x == genai.Flaky || y == genai.Flaky {
		return true
	}
	return x == y
})
