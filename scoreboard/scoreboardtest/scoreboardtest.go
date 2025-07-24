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

// GetClientOneModel returns a provider client for a specific model.
type GetClientOneModel func(t testing.TB, scenarioName string) (genai.Provider, http.RoundTripper)

type GetClient func(t testing.TB, model string, fn func(http.RoundTripper) http.RoundTripper) genai.Provider

func TestClient_Scoreboard(t *testing.T, gc GetClient, models []genai.Model, rec *internal.Records) {
	if len(models) == 0 {
		t.Fatal("no models")
	}
	usage := genai.Usage{}
	for _, m := range models {
		id := m.GetID()
		t.Run(id, func(t *testing.T) {
			// Run one model at a time otherwise we can't collect the total usage.
			usage.Add(RunOneModel(t, func(t testing.TB, sn string) (genai.Provider, http.RoundTripper) {
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
				return gc(t, id, fn), rt
			}))
		})
	}
	t.Logf("Usage: %#v", usage)
}

// RunOneModel runs the scoreboard on one model.
//
// It must implement genai.ProviderScoreboard. If it is wrapped, the wrappers must implement
// genai.ProviderUnwrap.
func RunOneModel(t testing.TB, gc GetClientOneModel) genai.Usage {
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
