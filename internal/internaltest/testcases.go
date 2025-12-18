// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package internaltest

import (
	"errors"
	"slices"
	"testing"
	"time"

	"github.com/maruel/genai"
	"github.com/maruel/genai/base"
	"github.com/maruel/genai/smoke"
)

// ProviderError defines a test case for provider errors.
type ProviderError struct {
	Name         string
	Opts         genai.ProviderOptions
	ErrGenSync   string
	ErrGenStream string
	ErrListModel string
}

// TestClientProviderErrors tests that the provider returns the expected error.
func TestClientProviderErrors(t *testing.T, getClient func(t *testing.T, opts genai.ProviderOptions) (genai.Provider, error), lines []ProviderError) {
	for _, line := range lines {
		t.Run(line.Name, func(t *testing.T) {
			if _, err := getClient(t, line.Opts); line.ErrGenSync != "" || line.ErrGenStream != "" || line.ErrListModel != "" {
				if err != nil {
					// It failed but it was not expected.
					if line.ErrGenSync != "" {
						t.Fatalf("ErrGenSync: want %q, got %q", line.ErrGenSync, err)
					} else if line.ErrGenStream != "" {
						t.Fatalf("ErrGenStream: want %q, got %q", line.ErrGenStream, err)
					} else if line.ErrListModel != "" {
						t.Fatalf("ErrListModel: want %q, got %q", line.ErrListModel, err)
					}
				}
			} else if err != nil {
				t.Fatal(err)
			}

			msgs := genai.Messages{genai.NewTextMessage("Tell a short joke.")}
			t.Run("GenSync", func(t *testing.T) {
				c, err := getClient(t, line.Opts)
				if err != nil {
					if err.Error() == line.ErrGenSync {
						return
					}
					t.Fatal(err)
				}
				var unsupported *base.ErrNotSupported
				if _, err = c.GenSync(t.Context(), msgs); line.ErrGenSync == "" {
					if !errors.As(err, &unsupported) {
						t.Fatal("expected unsupported")
					}
				} else {
					if err == nil {
						t.Fatal("expected error")
					} else if errors.As(err, &unsupported) {
						t.Fatal("should not be structured")
					} else if got := err.Error(); got != line.ErrGenSync {
						t.Fatalf("Unexpected error.\nwant: %q\ngot : %q", line.ErrGenSync, got)
					}
				}
			})
			t.Run("GenStream", func(t *testing.T) {
				c, err := getClient(t, line.Opts)
				if err != nil {
					if err.Error() == line.ErrGenStream {
						return
					}
					t.Fatal(err)
				}

				fragments, finish := c.GenStream(t.Context(), msgs)
				for range fragments {
				}
				var unsupported *base.ErrNotSupported
				if _, err = finish(); line.ErrGenStream == "" {
					if !errors.As(err, &unsupported) {
						t.Fatal("expected unsupported")
					}
				} else {
					if err == nil {
						t.Fatal("expected error")
					} else if errors.As(err, &unsupported) {
						t.Fatal("should not be structured")
					} else if got := err.Error(); got != line.ErrGenStream {
						t.Fatalf("Unexpected error.\nwant: %q\ngot : %q", line.ErrGenStream, got)
					}
				}
			})
			if line.ErrListModel != "" {
				t.Run("ListModels", func(t *testing.T) {
					c, err := getClient(t, line.Opts)
					if err != nil {
						if err.Error() == line.ErrListModel {
							return
						}
						t.Fatal(err)
					}
					var unsupported *base.ErrNotSupported
					if _, err = c.ListModels(t.Context()); errors.As(err, &unsupported) {
						if line.ErrListModel != "" {
							t.Fatal("expected error")
						}
					} else if got := err.Error(); got != line.ErrListModel {
						t.Fatalf("Unexpected error.\nwant: %q\ngot : %q", line.ErrListModel, got)
					}
				})
			}
		})
	}
}

// TestCapabilities tests that declared provider capabilities match actual behavior.
//
// For each capability (GenAsync, Caching):
// - If declared as true, calling the method should not return ErrNotSupported
// - If declared as false, calling the method should return ErrNotSupported
func TestCapabilities(t *testing.T, c genai.Provider) {
	caps := c.Capabilities()
	msgs := genai.Messages{genai.NewTextMessage("test")}
	t.Run("GenAsync", func(t *testing.T) {
		var notSupported *base.ErrNotSupported
		if _, err := c.GenAsync(t.Context(), msgs); caps.GenAsync {
			if errors.As(err, &notSupported) {
				t.Error("GenAsync capability declared but returned ErrNotSupported")
			}
		} else {
			if !errors.As(err, &notSupported) {
				t.Errorf("GenAsync should return ErrNotSupported, got %T: %v", err, err)
			}
		}
	})

	t.Run("Caching", func(t *testing.T) {
		var notSupported *base.ErrNotSupported
		if _, err := c.CacheAddRequest(t.Context(), msgs, "test", "test", time.Hour); caps.Caching {
			if errors.As(err, &notSupported) {
				t.Error("Caching capability declared but CacheAddRequest returned ErrNotSupported")
			}
		} else {
			if !errors.As(err, &notSupported) {
				t.Errorf("CacheAddRequest should return ErrNotSupported, got %T: %v", err, err)
			}
		}
	})
}

// TestCapabilitiesGenAsync tests GenAsync capability.
//
// Returns the job ID from GenAsync, which can be used to poll for the result.
func TestCapabilitiesGenAsync(t *testing.T, c genai.Provider, msgs ...genai.Message) genai.Job {
	if caps := c.Capabilities(); !caps.GenAsync {
		t.Fatal("GenAsync capability not declared")
	}
	if len(msgs) == 0 {
		if slices.Contains(c.OutputModalities(), genai.ModalityImage) {
			msgs = genai.Messages{genai.NewTextMessage(smoke.ContentsImage)}
		} else {
			msgs = genai.Messages{genai.NewTextMessage("test")}
		}
	}
	id, err := c.GenAsync(t.Context(), msgs)
	var notSupported *base.ErrNotSupported
	if errors.As(err, &notSupported) {
		t.Error("GenAsync capability declared but returned ErrNotSupported")
	} else if err != nil {
		t.Errorf("GenAsync returned error: %v", err)
	}
	return id
}

// TestCapabilitiesCaching tests Caching capability.
func TestCapabilitiesCaching(t *testing.T, c genai.Provider, msgs ...genai.Message) {
	if caps := c.Capabilities(); !caps.Caching {
		t.Fatal("Caching capability not declared")
	}
	if len(msgs) == 0 {
		msgs = genai.Messages{genai.NewTextMessage("test")}
	}
	var notSupported *base.ErrNotSupported
	if _, err := c.CacheAddRequest(t.Context(), msgs, "", "test", time.Hour); errors.As(err, &notSupported) {
		t.Error("Caching capability declared but CacheAddRequest returned ErrNotSupported")
	} else if err != nil {
		t.Errorf("CacheAddRequest returned error: %v", err)
	}
}

// TestPreferredModels tests that the provider returns the expected model ID
// for each tier (SOTA/Good/Cheap) per output modality.
//
// The test data is automatically extracted from the provider's scoreboard.
// The newProvider function should create a provider instance with the given
// model tier and output modality.
func TestPreferredModels(t *testing.T, newProvider func(t *testing.T, model string, modality genai.Modality) (genai.Provider, error)) {
	data := loadPreferredModelsFromScoreboard(t, newProvider)
	for _, tc := range data {
		t.Run(tc.Tier, func(t *testing.T) {
			t.Run(string(tc.Modality)+"-"+tc.Tier, func(t *testing.T) {
				c, err := newProvider(t, tc.Tier, tc.Modality)
				if err != nil {
					t.Fatal(err)
				}
				if got := c.ModelID(); got != tc.Want {
					t.Fatalf("got model %q, want %q", got, tc.Want)
				}
			})
		})
	}
}

// preferredModelTest defines a test case for preferred model selection.
type preferredModelTest struct {
	Modality genai.Modality
	Tier     string // genai.ModelSOTA, genai.ModelGood, or genai.ModelCheap
	Want     string // Expected model ID
}

// loadPreferredModelsFromScoreboard extracts the preferred model test data
// (SOTA/Good/Cheap per output modality) from the provider's scoreboard.
func loadPreferredModelsFromScoreboard(t *testing.T, newProvider func(t *testing.T, model string, modality genai.Modality) (genai.Provider, error)) []preferredModelTest {
	// Create a provider instance with ModelNone just to read the scoreboard.
	// Try text modality first, if that fails try image modality.
	provider, err := newProvider(t, genai.ModelNone, genai.ModalityText)
	if err != nil {
		provider, err = newProvider(t, genai.ModelNone, genai.ModalityImage)
		if err != nil {
			t.Fatalf("failed to create provider for scoreboard reading: %v", err)
		}
	}
	score := provider.Scoreboard()
	var tests []preferredModelTest
	for _, sc := range score.Scenarios {
		if sc.SOTA {
			for modality := range sc.Out {
				tests = append(tests, preferredModelTest{
					Modality: genai.Modality(modality),
					Tier:     genai.ModelSOTA,
					Want:     sc.Models[0],
				})
			}
		}
		if sc.Good {
			for modality := range sc.Out {
				tests = append(tests, preferredModelTest{
					Modality: genai.Modality(modality),
					Tier:     genai.ModelGood,
					Want:     sc.Models[0],
				})
			}
		}
		if sc.Cheap {
			for modality := range sc.Out {
				tests = append(tests, preferredModelTest{
					Modality: genai.Modality(modality),
					Tier:     genai.ModelCheap,
					Want:     sc.Models[0],
				})
			}
		}
	}
	if len(tests) == 0 {
		t.Fatal("no preferred models found in scoreboard")
	}
	return tests
}
