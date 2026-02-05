// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package genai

import (
	"net/http"
	"testing"
)

type mockModel struct {
	id string
}

func (m mockModel) GetID() string  { return m.id }
func (m mockModel) String() string { return m.id }
func (m mockModel) Context() int64 { return 4096 }

func TestProviderOptionAPIKey(t *testing.T) {
	t.Run("valid", func(t *testing.T) {
		if err := ProviderOptionAPIKey("sk-key").Validate(); err != nil {
			t.Fatal(err)
		}
	})
	t.Run("error", func(t *testing.T) {
		if err := ProviderOptionAPIKey("").Validate(); err == nil || err.Error() != "ProviderOptionAPIKey cannot be empty" {
			t.Fatalf("want %q, got %q", "ProviderOptionAPIKey cannot be empty", err)
		}
	})
}

func TestProviderOptionRemote(t *testing.T) {
	t.Run("valid", func(t *testing.T) {
		if err := ProviderOptionRemote("http://localhost:8080").Validate(); err != nil {
			t.Fatal(err)
		}
	})
	t.Run("error", func(t *testing.T) {
		if err := ProviderOptionRemote("").Validate(); err == nil || err.Error() != "ProviderOptionRemote cannot be empty" {
			t.Fatalf("want %q, got %q", "ProviderOptionRemote cannot be empty", err)
		}
	})
}

func TestProviderOptionModel(t *testing.T) {
	t.Run("valid", func(t *testing.T) {
		for _, v := range []ProviderOptionModel{"gpt-4", ModelCheap, ModelGood, ModelSOTA} {
			if err := v.Validate(); err != nil {
				t.Fatalf("%q: %v", v, err)
			}
		}
	})
	t.Run("error", func(t *testing.T) {
		if err := ProviderOptionModel("").Validate(); err == nil || err.Error() != "ProviderOptionModel cannot be empty" {
			t.Fatalf("want %q, got %q", "ProviderOptionModel cannot be empty", err)
		}
	})
}

func TestProviderOptionModalities(t *testing.T) {
	t.Run("valid", func(t *testing.T) {
		if err := (ProviderOptionModalities{ModalityText}).Validate(); err != nil {
			t.Fatal(err)
		}
	})
	t.Run("error", func(t *testing.T) {
		tests := []struct {
			name   string
			in     ProviderOptionModalities
			errMsg string
		}{
			{
				name:   "empty",
				in:     ProviderOptionModalities{},
				errMsg: "ProviderOptionModalities cannot be empty",
			},
			{
				name:   "invalid modality",
				in:     ProviderOptionModalities{"invalid"},
				errMsg: "invalid Modality: \"invalid\"",
			},
		}
		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				if err := tt.in.Validate(); err == nil || err.Error() != tt.errMsg {
					t.Fatalf("want %q, got %q", tt.errMsg, err)
				}
			})
		}
	})
}

func TestProviderOptionPreloadedModels(t *testing.T) {
	t.Run("valid", func(t *testing.T) {
		if err := ProviderOptionPreloadedModels([]Model{mockModel{id: "m1"}}).Validate(); err != nil {
			t.Fatal(err)
		}
	})
	t.Run("error", func(t *testing.T) {
		if err := ProviderOptionPreloadedModels(nil).Validate(); err == nil || err.Error() != "ProviderOptionPreloadedModels cannot be empty" {
			t.Fatalf("want %q, got %q", "ProviderOptionPreloadedModels cannot be empty", err)
		}
	})
}

func TestProviderOptionTransportWrapper(t *testing.T) {
	t.Run("valid", func(t *testing.T) {
		fn := ProviderOptionTransportWrapper(func(rt http.RoundTripper) http.RoundTripper { return rt })
		if err := fn.Validate(); err != nil {
			t.Fatal(err)
		}
	})
	t.Run("error", func(t *testing.T) {
		if err := ProviderOptionTransportWrapper(nil).Validate(); err == nil || err.Error() != "ProviderOptionTransportWrapper cannot be nil" {
			t.Fatalf("want %q, got %q", "ProviderOptionTransportWrapper cannot be nil", err)
		}
	})
}

func TestProviderOptionInterface(t *testing.T) {
	// Verify all types implement ProviderOption.
	opts := []ProviderOption{
		ProviderOptionAPIKey("key"),
		ProviderOptionRemote("http://localhost"),
		ProviderOptionModel("model"),
		ProviderOptionModalities{ModalityText},
		ProviderOptionPreloadedModels{mockModel{id: "m"}},
		ProviderOptionTransportWrapper(func(rt http.RoundTripper) http.RoundTripper { return rt }),
	}
	for _, o := range opts {
		if err := o.Validate(); err != nil {
			t.Fatal(err)
		}
	}
}
