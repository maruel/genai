// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package internaltest

import (
	"errors"
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
