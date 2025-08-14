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
	APIKey       string
	Model        string
	ErrGenSync   string
	ErrGenStream string
	ErrGenDoc    string
	ErrListModel string
}

func TestClient_Provider_errors(t *testing.T, getClient func(t *testing.T, apiKey, model string) (genai.Provider, error), lines []ProviderError) {
	for _, line := range lines {
		t.Run(line.Name, func(t *testing.T) {
			tester, err := getClient(t, line.APIKey, line.Model)
			if line.ErrGenSync != "" || line.ErrGenStream != "" || line.ErrGenDoc != "" || line.ErrListModel != "" {
				if err != nil {
					// It failed but it was not expected.
					if line.ErrGenSync != "" {
						t.Fatalf("want %q, got %q", line.ErrGenSync, err)
					} else if line.ErrGenStream != "" {
						t.Fatalf("want %q, got %q", line.ErrGenStream, err)
					} else if line.ErrGenDoc != "" {
						t.Fatalf("want %q, got %q", line.ErrGenDoc, err)
					} else if line.ErrListModel != "" {
						t.Fatalf("want %q, got %q", line.ErrListModel, err)
					}
				}
			} else if err != nil {
				t.Fatal(err)
			}

			if _, ok := tester.(genai.ProviderGen); ok {
				msgs := genai.Messages{genai.NewTextMessage("Tell a short joke.")}
				if line.ErrGenSync != "" {
					t.Run("GenSync", func(t *testing.T) {
						c, err := getClient(t, line.APIKey, line.Model)
						if err != nil {
							if err.Error() == line.ErrGenSync {
								return
							}
							t.Fatal(err)
						}
						_, err = c.(genai.ProviderGen).GenSync(t.Context(), msgs, &genai.OptionsText{})
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
						c, err := getClient(t, line.APIKey, line.Model)
						if err != nil {
							if err.Error() == line.ErrGenStream {
								return
							}
							t.Fatal(err)
						}
						ch := make(chan genai.ReplyFragment, 1)
						_, err = c.(genai.ProviderGen).GenStream(t.Context(), msgs, ch, &genai.OptionsText{})
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
			} else if tester != nil {
				if line.ErrGenSync != "" {
					t.Fatal("ErrGenSync is set but client does not support ProviderGen")
				}
				if line.ErrGenStream != "" {
					t.Fatal("ErrGenStream is set but client does not support ProviderGen")
				}
			}
			if _, ok := tester.(genai.ProviderGenDoc); ok {
				msg := genai.NewTextMessage("Generate a short joke.")
				if line.ErrGenDoc != "" {
					t.Run("GenDoc", func(t *testing.T) {
						c, err := getClient(t, line.APIKey, line.Model)
						if err != nil {
							if err.Error() == line.ErrGenDoc {
								return
							}
							t.Fatal(err)
						}
						_, err = c.(genai.ProviderGenDoc).GenDoc(t.Context(), msg, nil)
						if err == nil {
							t.Fatal("expected error")
						} else if _, ok := err.(*genai.UnsupportedContinuableError); ok {
							t.Fatal("should not be continuable")
						} else if got := err.Error(); got != line.ErrGenDoc {
							t.Fatalf("Unexpected error.\nwant: %q\ngot : %q", line.ErrGenDoc, got)
						}
					})
				}
			} else if tester != nil {
				if line.ErrGenDoc != "" {
					t.Fatal("ErrGenDoc is set but client does not support ProviderGenDoc")
				}
			}
			if _, ok := tester.(genai.ProviderModel); ok {
				if line.ErrListModel != "" {
					t.Run("ListModels", func(t *testing.T) {
						c, err := getClient(t, line.APIKey, base.NoModel)
						if err != nil {
							if err.Error() == line.ErrListModel {
								return
							}
							t.Fatal(err)
						}
						_, err = c.(genai.ProviderModel).ListModels(t.Context())
						if err == nil {
							if line.ErrListModel != "" {
								t.Fatal("expected error")
							}
						} else if got := err.Error(); got != line.ErrListModel {
							t.Fatalf("Unexpected error.\nwant: %q\ngot : %q", line.ErrListModel, got)
						}
					})
				}
			} else if tester != nil {
				if line.ErrListModel != "" {
					t.Fatal("ErrListModel is set but client does not support ProviderModel")
				}
			}
		})
	}
}
