// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Tests for list-models

package main

import (
	"slices"
	"testing"
)

func TestProviderNames(t *testing.T) {
	t.Run("valid", func(t *testing.T) {
		t.Setenv("ANTHROPIC_API_KEY", "")
		names := providerNames()
		if !slices.Contains(names, "anthropic") {
			t.Fatalf("providerNames() = %q, want anthropic even when ANTHROPIC_API_KEY is unset", names)
		}
		if !slices.IsSorted(names) {
			t.Fatalf("providerNames() = %q, want sorted", names)
		}
	})
}
