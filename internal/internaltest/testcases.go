// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package internaltest

import (
	"testing"

	"github.com/maruel/genai"
)

type ProviderGenError struct {
	Name         string
	ApiKey       string
	Model        string
	ErrGenSync   string
	ErrGenStream string
}

func TestClient_ProviderGen_errors(t *testing.T, getClient func(t *testing.T, apiKey, model string) genai.ProviderGen, lines []ProviderGenError) {
	for _, line := range lines {
		t.Run(line.Name, func(t *testing.T) {
			msgs := genai.Messages{genai.NewTextMessage(genai.User, "Tell a short joke.")}
			if line.ErrGenSync != "" {
				t.Run("GenSync", func(t *testing.T) {
					c := getClient(t, line.ApiKey, line.Model)
					_, err := c.GenSync(t.Context(), msgs, &genai.OptionsText{})
					if err == nil {
						t.Fatal("expected error")
					} else if _, ok := err.(*genai.UnsupportedContinuableError); ok {
						t.Fatal("should not be continuable")
					} else if got := err.Error(); got != line.ErrGenSync {
						t.Fatalf("Unexpected error.\nwant: %q\ngot : %q", line.ErrGenSync, got)
					}
				})
			}
			if line.ErrGenStream != "" {
				t.Run("GenStream", func(t *testing.T) {
					c := getClient(t, line.ApiKey, line.Model)
					ch := make(chan genai.ContentFragment, 1)
					_, err := c.GenStream(t.Context(), msgs, ch, &genai.OptionsText{})
					if err == nil {
						t.Fatal("expected error")
					} else if _, ok := err.(*genai.UnsupportedContinuableError); ok {
						t.Fatal("should not be continuable")
					} else if got := err.Error(); got != line.ErrGenStream {
						t.Fatalf("Unexpected error.\nwant: %q\ngot : %q", line.ErrGenStream, got)
					}
					select {
					case pkt := <-ch:
						t.Fatal(pkt)
					default:
					}
				})
			}
		})
	}
}

type ProviderModelError struct {
	Name   string
	ApiKey string
	Err    string
}

func TestClient_ProviderModel_errors(t *testing.T, getClient func(t *testing.T, apiKey string) genai.ProviderModel, lines []ProviderModelError) {
	for _, line := range lines {
		t.Run(line.Name, func(t *testing.T) {
			t.Run("ListModels", func(t *testing.T) {
				c := getClient(t, line.ApiKey)
				_, err := c.ListModels(t.Context())
				if err == nil {
					t.Fatal("expected error")
				} else if got := err.Error(); got != line.Err {
					t.Fatalf("Unexpected error.\nwant: %q\ngot : %q", line.Err, got)
				}
			})
		})
	}
}
