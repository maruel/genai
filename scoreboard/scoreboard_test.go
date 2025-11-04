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

func TestScenarioValidate(t *testing.T) {
	t.Run("Valid", func(t *testing.T) {
		sc := Scenario{
			In: map[Modality]ModalCapability{
				ModalityText: {},
			},
			Out: map[Modality]ModalCapability{
				ModalityText: {},
			},
			GenSync: &Functionality{},
		}
		if err := sc.Validate(); err != nil {
			t.Fatalf("Scenario.Validate() returned error: %v", err)
		}
	})

	t.Run("InvalidModality", func(t *testing.T) {
		sc := Scenario{
			In: map[Modality]ModalCapability{
				Modality("unsupported"): {},
			},
		}
		if err := sc.Validate(); err == nil {
			t.Fatal("Scenario.Validate() returned nil error")
		}
	})

	t.Run("InvalidFunctionality", func(t *testing.T) {
		sc := Scenario{
			GenStream: &Functionality{Tools: TriState(42)},
		}
		if err := sc.Validate(); err == nil {
			t.Fatal("Scenario.Validate() returned nil error")
		}
	})
}
