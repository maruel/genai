// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package scoreboard

import "testing"

func TestTriState(t *testing.T) {
	t.Run("String", func(t *testing.T) {
		tests := []struct {
			name string
			in   TriState
			want string
		}{
			{
				name: "False",
				in:   False,
				want: "false",
			},
			{
				name: "True",
				in:   True,
				want: "true",
			},
			{
				name: "Flaky",
				in:   Flaky,
				want: "flaky",
			},
			{
				name: "Unknown value",
				in:   TriState(99),
				want: "TriState(99)",
			},
		}

		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				if got := tt.in.String(); got != tt.want {
					t.Fatalf("TriState.String() got = %q, want %q", got, tt.want)
				}
			})
		}
	})

	t.Run("GoString", func(t *testing.T) {
		tests := []struct {
			name string
			in   TriState
			want string
		}{
			{
				name: "False",
				in:   False,
				want: "false",
			},
			{
				name: "True",
				in:   True,
				want: "true",
			},
			{
				name: "Flaky",
				in:   Flaky,
				want: "flaky",
			},
		}

		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				if got := tt.in.GoString(); got != tt.want {
					t.Fatalf("TriState.GoString() got = %q, want %q", got, tt.want)
				}
			})
		}
	})
}
