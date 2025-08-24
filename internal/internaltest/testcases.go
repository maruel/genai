// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package internaltest

import (
	"testing"

	"github.com/maruel/genai"
	"github.com/maruel/genai/base"
)

type ProviderError struct {
	Name         string
	Opts         genai.ProviderOptions
	ErrGenSync   string
	ErrGenStream string
	ErrListModel string
}

func TestClient_Provider_errors(t *testing.T, getClient func(t *testing.T, opts genai.ProviderOptions) (genai.Provider, error), lines []ProviderError) {
	for _, line := range lines {
		t.Run(line.Name, func(t *testing.T) {
			if _, err := getClient(t, line.Opts); line.ErrGenSync != "" || line.ErrGenStream != "" || line.ErrListModel != "" {
				if err != nil {
					// It failed but it was not expected.
					if line.ErrGenSync != "" {
						t.Fatalf("want %q, got %q", line.ErrGenSync, err)
					} else if line.ErrGenStream != "" {
						t.Fatalf("want %q, got %q", line.ErrGenStream, err)
					} else if line.ErrListModel != "" {
						t.Fatalf("want %q, got %q", line.ErrListModel, err)
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
				_, err = c.GenSync(t.Context(), msgs)
				if line.ErrGenSync == "" {
					if err != base.ErrNotSupported {
						t.Fatal("expected unsupported")
					}
				} else {
					if err == nil {
						t.Fatal("expected error")
					} else if _, ok := err.(*genai.UnsupportedContinuableError); ok {
						t.Fatal("should not be continuable")
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
				_, err = finish()
				if line.ErrGenStream == "" {
					if err != base.ErrNotSupported {
						t.Fatal("expected unsupported")
					}
				} else {
					if err == nil {
						t.Fatal("expected error")
					} else if _, ok := err.(*genai.UnsupportedContinuableError); ok {
						t.Fatal("should not be continuable")
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
					_, err = c.ListModels(t.Context())
					if err == base.ErrNotSupported {
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
