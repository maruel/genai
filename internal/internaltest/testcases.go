// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package internaltest

import (
	"testing"

	"github.com/maruel/genai"
)

type ProviderError struct {
	Name         string
	APIKey       string
	Model        string
	ErrGenSync   string
	ErrGenStream string
	ErrGenDoc    string
	ErrListModel string
}

func TestClient_Provider_errors(t *testing.T, getClient func(t *testing.T, apiKey, model string) genai.Provider, lines []ProviderError) {
	for _, line := range lines {
		t.Run(line.Name, func(t *testing.T) {
			tester := getClient(t, line.APIKey, line.Model)
			if _, ok := tester.(genai.ProviderGen); ok {
				msgs := genai.Messages{genai.NewTextMessage(genai.User, "Tell a short joke.")}
				if line.ErrGenSync != "" {
					t.Run("GenSync", func(t *testing.T) {
						c := getClient(t, line.APIKey, line.Model).(genai.ProviderGen)
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
						c := getClient(t, line.APIKey, line.Model).(genai.ProviderGen)
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
			} else {
				if line.ErrGenSync != "" {
					t.Fatal("ErrGenSync is set but client does not support ProviderGen")
				}
				if line.ErrGenStream != "" {
					t.Fatal("ErrGenStream is set but client does not support ProviderGen")
				}
			}
			if _, ok := tester.(genai.ProviderGenDoc); ok {
				msg := genai.NewTextMessage(genai.User, "Generate a short joke.")
				t.Run("GenDoc", func(t *testing.T) {
					c := getClient(t, line.APIKey, line.Model).(genai.ProviderGenDoc)
					_, err := c.GenDoc(t.Context(), msg, nil)
					if err == nil {
						t.Fatal("expected error")
					} else if _, ok := err.(*genai.UnsupportedContinuableError); ok {
						t.Fatal("should not be continuable")
					} else if got := err.Error(); got != line.ErrGenDoc {
						t.Fatalf("Unexpected error.\nwant: %q\ngot : %q", line.ErrGenDoc, got)
					}
				})
			} else {
				if line.ErrGenDoc != "" {
					t.Fatal("ErrGenDoc is set but client does not support ProviderGenDoc")
				}
			}
			if _, ok := tester.(genai.ProviderModel); ok {
				t.Run("ListModels", func(t *testing.T) {
					c := getClient(t, line.APIKey, "").(genai.ProviderModel)
					_, err := c.ListModels(t.Context())
					if err == nil {
						if line.ErrListModel != "" {
							t.Fatal("expected error")
						}
					} else if got := err.Error(); got != line.ErrListModel {
						t.Fatalf("Unexpected error.\nwant: %q\ngot : %q", line.ErrListModel, got)
					}
				})
			} else {
				if line.ErrListModel != "" {
					t.Fatal("ErrListModel is set but client does not support ProviderModel")
				}
			}
		})
	}
}
