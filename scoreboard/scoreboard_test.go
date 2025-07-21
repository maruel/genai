// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package scoreboard_test

import (
	"flag"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"slices"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/maruel/genai"
	"github.com/maruel/genai/adapters"
	"github.com/maruel/genai/internal"
	"github.com/maruel/genai/providers/cerebras"
	"github.com/maruel/genai/providers/groq"
	"github.com/maruel/genai/scoreboard"
)

func TestCreateScenario(t *testing.T) {
	t.Parallel()
	// Eventually use providers.All?
	for _, provider := range []string{"cerebras"} {
		t.Run(provider, func(t *testing.T) {
			t.Parallel()
			cc := getClient(t, provider, t.Name()+"/ListModels", "")
			models, err2 := cc.(genai.ProviderModel).ListModels(t.Context())
			if err2 != nil {
				t.Fatal(err2)
			}
			refs := cc.(genai.ProviderScoreboard).Scoreboard()
			for _, m := range models {
				id := m.GetID()
				t.Run(id, func(t *testing.T) {
					t.Parallel()
					// Find the reference.
					var want genai.Scenario
					for i := range refs.Scenarios {
						if slices.Contains(refs.Scenarios[i].Models, id) {
							want = refs.Scenarios[i]
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
					providerFactory := func(name string) genai.Provider {
						return getClient(t, provider, t.Name()+"/"+name, id)
					}
					logger := slog.New(slog.NewTextHandler(&testWriter{t: t}, &slog.HandlerOptions{
						Level: programLevel,
						ReplaceAttr: func(groups []string, a slog.Attr) slog.Attr {
							if a.Key == "time" {
								return slog.Attr{}
							}
							return a
						},
					}))
					ctx := internal.WithLogger(t.Context(), logger)
					got, err := scoreboard.CreateScenario(ctx, providerFactory)
					if err != nil {
						t.Fatalf("CreateScenario failed: %v", err)
					}

					// Check if valid.
					if diff := cmp.Diff(want, got, opt); diff != "" {
						t.Errorf("mismatch (-want +got):\n%s", diff)
					}
				})
			}
		})
	}
}

var opt = cmp.Comparer(func(x, y genai.TriState) bool {
	// TODO: Make this more solid. This requires a better assessment of what "Flaky" is.
	if x == genai.Flaky || y == genai.Flaky {
		return true
	}
	return x == y
})

func getClient(t *testing.T, provider, name, m string) genai.Provider {
	fn := func(h http.RoundTripper) http.RoundTripper {
		r, err2 := recorder.Record(name, h)
		if err2 != nil {
			t.Fatal(err2)
		}
		t.Cleanup(func() {
			if err3 := r.Stop(); err3 != nil {
				t.Error(err3)
			}
		})
		return r
	}
	switch provider {
	case "cerebras":
		apiKey := ""
		if os.Getenv("CEREBRAS_API_KEY") == "" {
			apiKey = "<insert_api_key_here>"
		}
		c, err := cerebras.New(apiKey, m, fn)
		if err != nil {
			t.Fatal(err)
		}
		if strings.HasPrefix(m, "qwen") {
			return &adapters.ProviderGenThinking{
				ProviderGen: &adapters.ProviderGenAppend{
					ProviderGen: c,
					Append:      genai.NewTextMessage(genai.User, "/think"),
				},
				TagName: "think",
			}
		}
		return c
	case "groq":
		apiKey := ""
		if os.Getenv("GROQ_API_KEY") == "" {
			apiKey = "<insert_api_key_here>"
		}
		c, err := groq.New(apiKey, m, fn)
		if err != nil {
			t.Fatal(err)
		}
		if strings.HasPrefix(m, "qwen/") {
			return &adapters.ProviderGenThinking{
				ProviderGen: &adapters.ProviderGenAppend{
					ProviderGen: c,
					Append:      genai.NewTextMessage(genai.User, "/think"),
				},
				TagName: "think",
			}
		}
		/*
			if m == "deepseek-r1-distill-llama-70b" {
				return &adapters.ProviderGenThinking{
					ProviderGen: c,
					TagName:     "think",
				}
			}
		*/
		return c
	default:
		t.Fatal("implement me")
		return nil
	}
}

var (
	recorder     *internal.Records
	programLevel = new(slog.LevelVar)
)

func TestMain(m *testing.M) {
	var err error
	recorder, err = internal.NewRecords("testdata")
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	flag.Parse()
	filtered := false
	flag.Visit(func(f *flag.Flag) {
		if f.Name == "test.run" {
			filtered = true
		}
	})
	flag.Visit(func(f *flag.Flag) {
		if f.Name == "test.v" {
			programLevel.Set(slog.LevelDebug)
		}
	})
	code := m.Run()
	if err = recorder.Close(); err != nil {
		if !filtered {
			fmt.Fprintf(os.Stderr, "%v\n", err)
			code = max(code, 1)
		}
	}
	os.Exit(code)
}

func init() {
	internal.BeLenient = false
}

// testWriter wraps t.Log() to implement io.Writer
type testWriter struct {
	t testing.TB
}

func (tw *testWriter) Write(p []byte) (n int, err error) {
	tw.t.Log(string(p))
	return len(p), nil
}
