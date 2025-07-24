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
	"github.com/maruel/genai/internal"
	"github.com/maruel/genai/internal/internaltest"
	"github.com/maruel/genai/scoreboard"
)

// GetClient is the client to assert the scoreboard. It will have the HTTP requests recorded.
type GetClient func(t testing.TB, model string, fn func(http.RoundTripper) http.RoundTripper) genai.Provider

// AssertScoreboard regenerates the scoreboard and asserts it is up to date.
func AssertScoreboard(t *testing.T, gc GetClient, models []genai.Model, rec *internal.Records) {
	if len(models) == 0 {
		t.Fatal("no models")
	}
	usage := genai.Usage{}

	// Find the reference.
	cc := gc(t, "", nil)
	og := cc
	for {
		if u, ok := og.(genai.ProviderUnwrap); ok {
			og = u.Unwrap()
		} else {
			break
		}
	}
	sb := og.(genai.ProviderScoreboard).Scoreboard()
	// Check for duplicates
	sbModels := map[string]struct{}{}
	for _, sc := range sb.Scenarios {
		for _, model := range sc.Models {
			if _, ok := sbModels[model]; ok {
				t.Fatalf("duplicate model in scoreboard: %q", model)
			}
			sbModels[model] = struct{}{}
		}
	}

	seen := map[string]struct{}{}
	for _, m := range models {
		model := m.GetID()
		t.Run(model, func(t *testing.T) {
			if _, ok := seen[model]; ok {
				t.Fatalf("duplicate model in ListModel: %q", model)
			}
			seen[model] = struct{}{}

			// Find the reference.
			var want genai.Scenario
			for _, sc := range sb.Scenarios {
				if slices.Contains(sc.Models, model) {
					want = sc
					want.Models = []string{model}
					break
				}
			}
			if len(want.Models) == 0 {
				t.Fatalf("no scenario for model %q", model)
			}
			if want.In == nil && want.Out == nil {
				t.Skip("Explicitly unsupported model")
			}

			// Run one model at a time otherwise we can't collect the total usage.
			u := runOneModel(t, func(t testing.TB, sn string) (genai.Provider, http.RoundTripper) {
				var rt http.RoundTripper
				fn := func(h http.RoundTripper) http.RoundTripper {
					if sn == "" {
						rt = h
						return h
					}
					r, err2 := rec.Record(sn, h)
					if err2 != nil {
						t.Fatal(err2)
					}
					t.Cleanup(func() {
						if err3 := r.Stop(); err3 != nil {
							t.Error(err3)
						}
					})
					rt = r
					return r
				}
				return gc(t, model, fn), rt
			}, model, want)
			usage.Add(u)
		})
	}
	t.Logf("Usage: %#v", usage)

	for model := range sbModels {
		if _, ok := seen[model]; !ok {
			t.Errorf("stale model in scoreboard: %q", model)
		}
	}
}

// getClientOneModel returns a provider client for a specific model.
type getClientOneModel func(t testing.TB, scenarioName string) (genai.Provider, http.RoundTripper)

// runOneModel runs the scoreboard on one model.
//
// It must implement genai.ProviderScoreboard. If it is wrapped, the wrappers must implement
// genai.ProviderUnwrap.
func runOneModel(t testing.TB, gc getClientOneModel, model string, want genai.Scenario) genai.Usage {
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
