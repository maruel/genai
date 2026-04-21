// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Internal tests for the Codex provider.

package codex

import (
	"errors"
	"slices"
	"testing"

	"github.com/maruel/genai"
	"github.com/maruel/genai/base"
	"github.com/maruel/genai/internal"
)

func TestReasoningEffort(t *testing.T) {
	t.Run("valid", func(t *testing.T) {
		for _, v := range []ReasoningEffort{
			ReasoningEffortNone, ReasoningEffortMinimal, ReasoningEffortLow,
			ReasoningEffortMedium, ReasoningEffortHigh, ReasoningEffortXHigh,
		} {
			c, err := New(v)
			if err != nil {
				t.Fatalf("New(%q): %v", v, err)
			}
			if c.effort != v {
				t.Errorf("effort: got %q, want %q", c.effort, v)
			}
		}
	})
	t.Run("invalid", func(t *testing.T) {
		if _, err := New(ReasoningEffort("turbo")); err == nil {
			t.Fatal("expected error for invalid effort")
		}
	})
	t.Run("default", func(t *testing.T) {
		c, err := New()
		if err != nil {
			t.Fatal(err)
		}
		if c.effort != ReasoningEffortMedium {
			t.Errorf("default effort: got %q, want %q", c.effort, ReasoningEffortMedium)
		}
	})
}

func TestParseOpts(t *testing.T) {
	t.Run("system_prompt", func(t *testing.T) {
		co, err := parseOpts([]genai.GenOption{&genai.GenOptionText{SystemPrompt: "Be helpful"}})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if co.systemPrompt != "Be helpful" {
			t.Errorf("systemPrompt: got %q, want %q", co.systemPrompt, "Be helpful")
		}
	})
	t.Run("unsupported", func(t *testing.T) {
		for _, tc := range []struct {
			name string
			opts []genai.GenOption
			want string
		}{
			{"Temperature", []genai.GenOption{&genai.GenOptionText{Temperature: 0.5}}, "GenOptionText.Temperature"},
			{"Seed", []genai.GenOption{genai.GenOptionSeed(42)}, "GenOptionSeed"},
		} {
			t.Run(tc.name, func(t *testing.T) {
				_, err := parseOpts(tc.opts)
				var uerr *base.ErrNotSupported
				if !errors.As(err, &uerr) {
					t.Fatalf("expected ErrNotSupported, got %v", err)
				}
				if !slices.Contains(uerr.Options, tc.want) {
					t.Errorf("expected %q in unsupported, got %v", tc.want, uerr.Options)
				}
			})
		}
	})
}

func init() {
	internal.BeLenient = false
}
