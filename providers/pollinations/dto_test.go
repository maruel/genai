// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Tests for the Pollinations wire types.

package pollinations_test

import (
	"testing"

	"github.com/maruel/genai/providers/pollinations"
)

func TestTextModel(t *testing.T) {
	t.Run("String omits empty metadata", func(t *testing.T) {
		m := pollinations.TextModel{
			Name:            "test-model",
			Category:        "text",
			InputModalities: []string{"text"},
		}
		want := "test-model in:text; out:text"
		if got := m.String(); got != want {
			t.Fatalf("String() = %q, want %q", got, want)
		}
	})
	t.Run("String includes metadata", func(t *testing.T) {
		m := pollinations.TextModel{
			Name:            "test-model",
			Category:        "text",
			Description:     "Test description",
			InputModalities: []string{"text"},
			Provider:        "test-provider",
		}
		want := "test-model in:text; out:text; provider:test-provider; Test description"
		if got := m.String(); got != want {
			t.Fatalf("String() = %q, want %q", got, want)
		}
	})
}
