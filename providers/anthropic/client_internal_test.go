// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package anthropic

import "testing"

func TestModelsMaxTokens(t *testing.T) {
	data := []struct {
		model string
		want  int64
	}{
		// Legacy models.
		{"claude-3-opus-20240229", 4096},
		{"claude-3-haiku-20240307", 4096},
		{"claude-3-5-sonnet-20241022", 8192},
		{"claude-3-5-haiku-20241022", 8192},
		// Claude 3.7.
		{"claude-3-7-sonnet-20250219", 64000},
		// Claude 4 family (32K output models).
		{"claude-opus-4-20250514", 32000},
		{"claude-opus-4-0", 32000},
		{"claude-4-opus-20250514", 32000},
		{"claude-opus-4-1-20250805", 32000},
		// Claude 4 family (64K output models).
		{"claude-sonnet-4-20250514", 64000},
		{"claude-sonnet-4-0", 64000},
		{"claude-4-sonnet-20250514", 64000},
		{"claude-sonnet-4-5-20250929", 64000},
		{"claude-opus-4-5-20251101", 64000},
		{"claude-haiku-4-5-20251001", 64000},
		// Claude 4.6.
		{"claude-opus-4-6", 128000},
		// Unknown future model.
		{"claude-future-99", 64000},
	}
	for _, tc := range data {
		if got := modelsMaxTokens(tc.model); got != tc.want {
			t.Errorf("modelsMaxTokens(%q) = %d, want %d", tc.model, got, tc.want)
		}
	}
}
