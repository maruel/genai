// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package scoreboard

import (
	"encoding/json"
	"testing"
)

func TestModel(t *testing.T) {
	t.Run("String without reasoning", func(t *testing.T) {
		m := Model{Model: "gpt-4", Reason: false}
		if got := m.String(); got != "gpt-4" {
			t.Fatalf("got %q, want %q", got, "gpt-4")
		}
	})

	t.Run("String with reasoning", func(t *testing.T) {
		m := Model{Model: "gpt-4", Reason: true}
		if got := m.String(); got != "gpt-4_thinking" {
			t.Fatalf("got %q, want %q", got, "gpt-4_thinking")
		}
	})

	t.Run("String with colons in model name", func(t *testing.T) {
		m := Model{Model: "claude-3:opus", Reason: false}
		if got := m.String(); got != "claude-3-opus" {
			t.Fatalf("got %q, want %q", got, "claude-3-opus")
		}
	})

	t.Run("String with colons and reasoning", func(t *testing.T) {
		m := Model{Model: "claude-3:opus", Reason: true}
		if got := m.String(); got != "claude-3-opus_thinking" {
			t.Fatalf("got %q, want %q", got, "claude-3-opus_thinking")
		}
	})
}

func TestModality(t *testing.T) {
	t.Run("Valid modalities", func(t *testing.T) {
		for _, m := range []Modality{ModalityAudio, ModalityDocument, ModalityImage, ModalityText, ModalityVideo} {
			if err := m.Validate(); err != nil {
				t.Fatalf("valid modality %q failed validation: %v", m, err)
			}
		}
	})

	t.Run("Invalid modality", func(t *testing.T) {
		m := Modality("invalid")
		if err := m.Validate(); err == nil {
			t.Fatal("invalid modality should fail validation")
		}
	})
}

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

	t.Run("Validate", func(t *testing.T) {
		t.Run("Valid values", func(t *testing.T) {
			for _, v := range []TriState{False, True, Flaky} {
				if err := v.Validate(); err != nil {
					t.Fatalf("valid value %v failed validation: %v", v, err)
				}
			}
		})

		t.Run("Invalid value", func(t *testing.T) {
			if err := TriState(99).Validate(); err == nil {
				t.Fatal("invalid TriState should fail validation")
			}
		})
	})

	t.Run("MarshalJSON", func(t *testing.T) {
		tests := []struct {
			name string
			in   TriState
			want []byte
		}{
			{name: "False", in: False, want: []byte(`"false"`)},
			{name: "True", in: True, want: []byte(`"true"`)},
			{name: "Flaky", in: Flaky, want: []byte(`"flaky"`)},
		}

		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				got, err := tt.in.MarshalJSON()
				if err != nil {
					t.Fatalf("MarshalJSON failed: %v", err)
				}
				if string(got) != string(tt.want) {
					t.Fatalf("got %s, want %s", got, tt.want)
				}
			})
		}

		t.Run("Invalid value", func(t *testing.T) {
			_, err := TriState(99).MarshalJSON()
			if err == nil {
				t.Fatal("invalid TriState should fail")
			}
		})
	})

	t.Run("UnmarshalJSON", func(t *testing.T) {
		tests := []struct {
			name string
			in   []byte
			want TriState
		}{
			{name: "false", in: []byte(`"false"`), want: False},
			{name: "true", in: []byte(`"true"`), want: True},
			{name: "flaky", in: []byte(`"flaky"`), want: Flaky},
		}

		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				var got TriState
				err := got.UnmarshalJSON(tt.in)
				if err != nil {
					t.Fatalf("UnmarshalJSON failed: %v", err)
				}
				if got != tt.want {
					t.Fatalf("got %v, want %v", got, tt.want)
				}
			})
		}

		t.Run("Invalid value", func(t *testing.T) {
			var got TriState
			if err := got.UnmarshalJSON([]byte(`"invalid"`)); err == nil {
				t.Fatal("should fail on invalid value")
			}
		})

		t.Run("Invalid JSON", func(t *testing.T) {
			var got TriState
			if err := got.UnmarshalJSON([]byte(`not valid json`)); err == nil {
				t.Fatal("should fail on invalid JSON")
			}
		})
	})
}

func TestFunctionality(t *testing.T) {
	t.Run("Validate valid functionality", func(t *testing.T) {
		f := &Functionality{
			ReportTokenUsage:  True,
			ReportFinishReason: True,
			Tools:             True,
			ToolsBiased:       False,
			ToolsIndecisive:   False,
		}
		if err := f.Validate(); err != nil {
			t.Fatalf("valid functionality failed validation: %v", err)
		}
	})

	t.Run("Validate invalid ReportTokenUsage", func(t *testing.T) {
		f := &Functionality{ReportTokenUsage: TriState(99)}
		if err := f.Validate(); err == nil {
			t.Fatal("should fail on invalid ReportTokenUsage")
		}
	})

	t.Run("Validate ToolsBiased when Tools is false", func(t *testing.T) {
		f := &Functionality{Tools: False, ToolsBiased: True}
		if err := f.Validate(); err == nil {
			t.Fatal("should fail when ToolsBiased is set but Tools is false")
		}
	})

	t.Run("Validate ToolsIndecisive when Tools is false", func(t *testing.T) {
		f := &Functionality{Tools: False, ToolsIndecisive: True}
		if err := f.Validate(); err == nil {
			t.Fatal("should fail when ToolsIndecisive is set but Tools is false")
		}
	})

	t.Run("Validate ToolCallRequired when Tools is false", func(t *testing.T) {
		f := &Functionality{Tools: False, ToolCallRequired: true}
		if err := f.Validate(); err == nil {
			t.Fatal("should fail when ToolCallRequired is set but Tools is false")
		}
	})

	t.Run("Less comparison - ReportRateLimits", func(t *testing.T) {
		f1 := &Functionality{ReportRateLimits: false}
		f2 := &Functionality{ReportRateLimits: true}
		if !f1.Less(f2) {
			t.Fatal("f1 should be less than f2")
		}
		if f2.Less(f1) {
			t.Fatal("f2 should not be less than f1")
		}
	})

	t.Run("Less comparison - ReportTokenUsage", func(t *testing.T) {
		f1 := &Functionality{ReportTokenUsage: False}
		f2 := &Functionality{ReportTokenUsage: True}
		if !f1.Less(f2) {
			t.Fatal("f1 should be less than f2")
		}
	})

	t.Run("Less comparison - ReportFinishReason", func(t *testing.T) {
		f1 := &Functionality{ReportFinishReason: False}
		f2 := &Functionality{ReportFinishReason: True}
		if !f1.Less(f2) {
			t.Fatal("f1 should be less than f2")
		}
	})

	t.Run("Less comparison - Seed", func(t *testing.T) {
		f1 := &Functionality{Seed: false}
		f2 := &Functionality{Seed: true}
		if !f1.Less(f2) {
			t.Fatal("f1 should be less than f2")
		}
	})

	t.Run("Less comparison - Tools", func(t *testing.T) {
		f1 := &Functionality{Tools: False}
		f2 := &Functionality{Tools: True}
		if !f1.Less(f2) {
			t.Fatal("f1 should be less than f2")
		}
	})

	t.Run("Less comparison - ToolCallRequired", func(t *testing.T) {
		f1 := &Functionality{ToolCallRequired: false}
		f2 := &Functionality{ToolCallRequired: true}
		if !f1.Less(f2) {
			t.Fatal("f1 should be less than f2")
		}
	})

	t.Run("Less comparison - JSON", func(t *testing.T) {
		f1 := &Functionality{JSON: false}
		f2 := &Functionality{JSON: true}
		if !f1.Less(f2) {
			t.Fatal("f1 should be less than f2")
		}
	})

	t.Run("Less comparison - JSONSchema", func(t *testing.T) {
		f1 := &Functionality{JSONSchema: false}
		f2 := &Functionality{JSONSchema: true}
		if !f1.Less(f2) {
			t.Fatal("f1 should be less than f2")
		}
	})

	t.Run("Less comparison - Citations", func(t *testing.T) {
		f1 := &Functionality{Citations: false}
		f2 := &Functionality{Citations: true}
		if !f1.Less(f2) {
			t.Fatal("f1 should be less than f2")
		}
	})

	t.Run("Less comparison - MaxTokens", func(t *testing.T) {
		f1 := &Functionality{MaxTokens: false}
		f2 := &Functionality{MaxTokens: true}
		if !f1.Less(f2) {
			t.Fatal("f1 should be less than f2")
		}
	})

	t.Run("Less comparison - StopSequence", func(t *testing.T) {
		f1 := &Functionality{StopSequence: false}
		f2 := &Functionality{StopSequence: true}
		if !f1.Less(f2) {
			t.Fatal("f1 should be less than f2")
		}
	})

	t.Run("Less comparison - not less", func(t *testing.T) {
		f1 := &Functionality{ReportRateLimits: true}
		f2 := &Functionality{ReportRateLimits: true}
		if f1.Less(f2) {
			t.Fatal("f1 should not be less than f2 when equal")
		}
	})
}

func TestScenario(t *testing.T) {
	t.Run("Untested scenario", func(t *testing.T) {
		s := &Scenario{Models: []string{"gpt-4"}}
		if !s.Untested() {
			t.Fatal("empty scenario should be untested")
		}
	})

	t.Run("Tested scenario with GenSync", func(t *testing.T) {
		s := &Scenario{
			Models: []string{"gpt-4"},
			GenSync: &Functionality{},
		}
		if s.Untested() {
			t.Fatal("scenario with GenSync should not be untested")
		}
	})

	t.Run("Tested scenario with GenStream", func(t *testing.T) {
		s := &Scenario{
			Models: []string{"gpt-4"},
			GenStream: &Functionality{},
		}
		if s.Untested() {
			t.Fatal("scenario with GenStream should not be untested")
		}
	})

	t.Run("Tested scenario with In/Out", func(t *testing.T) {
		s := &Scenario{
			Models: []string{"gpt-4"},
			In:     map[Modality]ModalCapability{ModalityText: {}},
			Out:    map[Modality]ModalCapability{ModalityText: {}},
		}
		if s.Untested() {
			t.Fatal("scenario with In/Out should not be untested")
		}
	})

	t.Run("Validate no models", func(t *testing.T) {
		s := &Scenario{Models: []string{}}
		if err := s.Validate(); err == nil {
			t.Fatal("should fail with no models")
		}
	})

	t.Run("Validate invalid modality in In", func(t *testing.T) {
		s := &Scenario{
			Models: []string{"gpt-4"},
			In: map[Modality]ModalCapability{Modality("invalid"): {}},
		}
		if err := s.Validate(); err == nil {
			t.Fatal("should fail with invalid modality in In")
		}
	})

	t.Run("Validate invalid modality in Out", func(t *testing.T) {
		s := &Scenario{
			Models: []string{"gpt-4"},
			Out: map[Modality]ModalCapability{Modality("invalid"): {}},
		}
		if err := s.Validate(); err == nil {
			t.Fatal("should fail with invalid modality in Out")
		}
	})

	t.Run("Validate In without Out", func(t *testing.T) {
		s := &Scenario{
			Models: []string{"gpt-4"},
			In: map[Modality]ModalCapability{ModalityText: {}},
		}
		if err := s.Validate(); err == nil {
			t.Fatal("should fail with In but no Out")
		}
	})

	t.Run("Validate Out without In", func(t *testing.T) {
		s := &Scenario{
			Models: []string{"gpt-4"},
			Out: map[Modality]ModalCapability{ModalityText: {}},
		}
		if err := s.Validate(); err == nil {
			t.Fatal("should fail with Out but no In")
		}
	})

	t.Run("Validate GenSync without In/Out", func(t *testing.T) {
		s := &Scenario{
			Models: []string{"gpt-4"},
			GenSync: &Functionality{},
		}
		if err := s.Validate(); err == nil {
			t.Fatal("should fail with GenSync but no In/Out")
		}
	})

	t.Run("Validate valid with In/Out and GenSync", func(t *testing.T) {
		s := &Scenario{
			Models: []string{"gpt-4"},
			In: map[Modality]ModalCapability{ModalityText: {}},
			Out: map[Modality]ModalCapability{ModalityText: {}},
			GenSync: &Functionality{},
		}
		if err := s.Validate(); err != nil {
			t.Fatalf("valid scenario failed: %v", err)
		}
	})
}

func TestScore(t *testing.T) {
	t.Run("Validate valid score", func(t *testing.T) {
		s := &Score{
			Country: "US",
			Scenarios: []Scenario{
				{
					Models: []string{"gpt-4"},
					SOTA:   true,
					In:     map[Modality]ModalCapability{ModalityText: {}},
					Out:    map[Modality]ModalCapability{ModalityText: {}},
					GenSync: &Functionality{},
				},
				{
					Models: []string{"gpt-3.5"},
					Good:   true,
					In:     map[Modality]ModalCapability{ModalityText: {}},
					Out:    map[Modality]ModalCapability{ModalityText: {}},
					GenSync: &Functionality{},
				},
				{
					Models: []string{"gpt-2"},
					Cheap:  true,
					In:     map[Modality]ModalCapability{ModalityText: {}},
					Out:    map[Modality]ModalCapability{ModalityText: {}},
					GenSync: &Functionality{},
				},
			},
		}
		if err := s.Validate(); err != nil {
			t.Fatalf("valid score failed: %v", err)
		}
	})

	t.Run("Validate duplicate model", func(t *testing.T) {
		s := &Score{
			Scenarios: []Scenario{
				{Models: []string{"gpt-4"}, Reason: false, GenSync: &Functionality{}},
				{Models: []string{"gpt-4"}, Reason: false, GenSync: &Functionality{}},
			},
		}
		if err := s.Validate(); err == nil {
			t.Fatal("should fail with duplicate model")
		}
	})

	t.Run("Validate SOTA not first", func(t *testing.T) {
		s := &Score{
			Scenarios: []Scenario{
				{
					Models: []string{"gpt-3.5"},
					SOTA:   false,
					GenSync: &Functionality{},
				},
				{
					Models: []string{"gpt-4"},
					SOTA:   true,
					GenSync: &Functionality{},
				},
				{
					Models: []string{"gpt-2"},
					Good:   true,
					Cheap:  true,
					GenSync: &Functionality{},
				},
			},
		}
		if err := s.Validate(); err == nil {
			t.Fatal("should fail when SOTA is not first")
		}
	})

	t.Run("Validate no SOTA", func(t *testing.T) {
		s := &Score{
			Scenarios: []Scenario{
				{Models: []string{"gpt-4"}, GenSync: &Functionality{}},
				{Models: []string{"gpt-3.5"}, GenSync: &Functionality{}},
			},
		}
		if err := s.Validate(); err == nil {
			t.Fatal("should fail with no SOTA")
		}
	})

	t.Run("Validate multiple SOTA", func(t *testing.T) {
		s := &Score{
			Scenarios: []Scenario{
				{Models: []string{"gpt-4"}, SOTA: true, GenSync: &Functionality{}},
				{Models: []string{"gpt-3.5"}, SOTA: true, GenSync: &Functionality{}},
			},
		}
		if err := s.Validate(); err == nil {
			t.Fatal("should fail with multiple SOTA")
		}
	})

	t.Run("Validate no Good", func(t *testing.T) {
		s := &Score{
			Scenarios: []Scenario{
				{Models: []string{"gpt-4"}, SOTA: true, GenSync: &Functionality{}},
				{Models: []string{"gpt-3.5"}, GenSync: &Functionality{}},
				{Models: []string{"gpt-2"}, Cheap: true, GenSync: &Functionality{}},
			},
		}
		if err := s.Validate(); err == nil {
			t.Fatal("should fail with no Good")
		}
	})

	t.Run("Validate Good after Cheap", func(t *testing.T) {
		s := &Score{
			Scenarios: []Scenario{
				{Models: []string{"gpt-4"}, SOTA: true, GenSync: &Functionality{}},
				{Models: []string{"gpt-2"}, Cheap: true, GenSync: &Functionality{}},
				{Models: []string{"gpt-3.5"}, Good: true, GenSync: &Functionality{}},
			},
		}
		if err := s.Validate(); err == nil {
			t.Fatal("should fail when Good comes after Cheap")
		}
	})

	t.Run("Validate single scenario", func(t *testing.T) {
		s := &Score{
			Scenarios: []Scenario{
				{Models: []string{"gpt-4"}, SOTA: true, GenSync: &Functionality{}},
			},
		}
		if err := s.Validate(); err == nil {
			t.Fatal("should fail with single scenario missing Good/Cheap")
		}
	})

	t.Run("Validate multiple Cheap models", func(t *testing.T) {
		s := &Score{
			Scenarios: []Scenario{
				{Models: []string{"gpt-4"}, SOTA: true, GenSync: &Functionality{}},
				{Models: []string{"gpt-3.5"}, Good: true, GenSync: &Functionality{}},
				{Models: []string{"gpt-2"}, Cheap: true, GenSync: &Functionality{}},
				{Models: []string{"gpt-1"}, Cheap: true, GenSync: &Functionality{}},
			},
		}
		if err := s.Validate(); err == nil {
			t.Fatal("should fail with multiple Cheap models")
		}
	})

	t.Run("Validate SOTA with empty models list", func(t *testing.T) {
		s := &Score{
			Scenarios: []Scenario{
				{Models: []string{}, SOTA: true, GenSync: &Functionality{}},
				{Models: []string{"gpt-3.5"}, Good: true, GenSync: &Functionality{}},
				{Models: []string{"gpt-2"}, Cheap: true, GenSync: &Functionality{}},
			},
		}
		if err := s.Validate(); err == nil {
			t.Fatal("should fail with empty models list")
		}
	})

	t.Run("Validate no Cheap", func(t *testing.T) {
		s := &Score{
			Scenarios: []Scenario{
				{Models: []string{"gpt-4"}, SOTA: true, GenSync: &Functionality{}},
				{Models: []string{"gpt-3.5"}, Good: true, GenSync: &Functionality{}},
				{Models: []string{"gpt-2"}, GenSync: &Functionality{}},
			},
		}
		if err := s.Validate(); err == nil {
			t.Fatal("should fail with no Cheap")
		}
	})

	t.Run("Validate valid scenario with reason", func(t *testing.T) {
		s := &Score{
			Country: "US",
			Scenarios: []Scenario{
				{
					Models: []string{"gpt-4"},
					SOTA:   true,
					Reason: true,
					In:     map[Modality]ModalCapability{ModalityText: {}},
					Out:    map[Modality]ModalCapability{ModalityText: {}},
					GenSync: &Functionality{},
				},
				{
					Models: []string{"gpt-3.5"},
					Good:   true,
					Reason: false,
					In:     map[Modality]ModalCapability{ModalityText: {}},
					Out:    map[Modality]ModalCapability{ModalityText: {}},
					GenSync: &Functionality{},
				},
				{
					Models: []string{"gpt-2"},
					Cheap:  true,
					Reason: false,
					In:     map[Modality]ModalCapability{ModalityText: {}},
					Out:    map[Modality]ModalCapability{ModalityText: {}},
					GenSync: &Functionality{},
				},
			},
		}
		if err := s.Validate(); err != nil {
			t.Fatalf("valid score with Reason should pass: %v", err)
		}
	})
}

func TestReason(t *testing.T) {
	t.Run("Validate valid values", func(t *testing.T) {
		for _, r := range []Reason{ReasonNone, ReasonInline, ReasonAuto} {
			if err := r.Validate(); err != nil {
				t.Fatalf("valid Reason %v failed: %v", r, err)
			}
		}
	})

	t.Run("Validate invalid value", func(t *testing.T) {
		if err := Reason(99).Validate(); err == nil {
			t.Fatal("invalid Reason should fail")
		}
	})

	t.Run("MarshalJSON", func(t *testing.T) {
		tests := []struct {
			name string
			in   Reason
			want []byte
		}{
			{name: "none", in: ReasonNone, want: []byte(`"none"`)},
			{name: "inline", in: ReasonInline, want: []byte(`"inline"`)},
			{name: "auto", in: ReasonAuto, want: []byte(`"auto"`)},
		}

		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				got, err := tt.in.MarshalJSON()
				if err != nil {
					t.Fatalf("MarshalJSON failed: %v", err)
				}
				if string(got) != string(tt.want) {
					t.Fatalf("got %s, want %s", got, tt.want)
				}
			})
		}

		t.Run("Invalid value", func(t *testing.T) {
			_, err := Reason(99).MarshalJSON()
			if err == nil {
				t.Fatal("should fail on invalid Reason")
			}
		})
	})

	t.Run("UnmarshalJSON", func(t *testing.T) {
		tests := []struct {
			name string
			in   []byte
			want Reason
		}{
			{name: "none", in: []byte(`"none"`), want: ReasonNone},
			{name: "inline", in: []byte(`"inline"`), want: ReasonInline},
			{name: "auto", in: []byte(`"auto"`), want: ReasonAuto},
		}

		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				var got Reason
				if err := got.UnmarshalJSON(tt.in); err != nil {
					t.Fatalf("UnmarshalJSON failed: %v", err)
				}
				if got != tt.want {
					t.Fatalf("got %v, want %v", got, tt.want)
				}
			})
		}

		t.Run("Invalid value", func(t *testing.T) {
			var got Reason
			if err := got.UnmarshalJSON([]byte(`"invalid"`)); err == nil {
				t.Fatal("should fail on invalid Reason")
			}
		})

		t.Run("Invalid JSON", func(t *testing.T) {
			var got Reason
			if err := got.UnmarshalJSON([]byte(`not json`)); err == nil {
				t.Fatal("should fail on invalid JSON")
			}
		})
	})
}

func TestCompareScenarios(t *testing.T) {
	t.Run("SOTA comes first", func(t *testing.T) {
		a := Scenario{Models: []string{"gpt-4"}, SOTA: true, Reason: false}
		b := Scenario{Models: []string{"gpt-3"}, Good: true, Reason: false}
		if CompareScenarios(a, b) >= 0 {
			t.Fatal("SOTA should come before Good")
		}
	})

	t.Run("Good comes after SOTA", func(t *testing.T) {
		a := Scenario{Models: []string{"gpt-3"}, Good: true, Reason: false}
		b := Scenario{Models: []string{"gpt-2"}, Cheap: true, Reason: false}
		if CompareScenarios(a, b) >= 0 {
			t.Fatal("Good should come before Cheap")
		}
	})

	t.Run("Cheap comes after Good", func(t *testing.T) {
		a := Scenario{Models: []string{"gpt-4"}, SOTA: true, Reason: false}
		b := Scenario{Models: []string{"gpt-2"}, Cheap: true, Reason: false}
		if CompareScenarios(a, b) >= 0 {
			t.Fatal("SOTA should come before Cheap")
		}
	})

	t.Run("Reasoning comes before non-reasoning with same priority", func(t *testing.T) {
		a := Scenario{Models: []string{"gpt-4"}, SOTA: true, Reason: true}
		b := Scenario{Models: []string{"gpt-4"}, SOTA: true, Reason: false}
		if CompareScenarios(a, b) >= 0 {
			t.Fatal("reasoning should come before non-reasoning")
		}
	})

	t.Run("Unflagged scenarios get low priority", func(t *testing.T) {
		a := Scenario{Models: []string{"gpt-4"}, SOTA: true, Reason: false}
		b := Scenario{Models: []string{"gpt-3"}, Reason: false}
		if CompareScenarios(a, b) >= 0 {
			t.Fatal("SOTA should come before unflagged")
		}
	})

	t.Run("Alphabetical order for same priority and reasoning", func(t *testing.T) {
		a := Scenario{Models: []string{"gpt-unknown-b"}, Reason: false}
		b := Scenario{Models: []string{"gpt-unknown-a"}, Reason: false}
		if CompareScenarios(a, b) <= 0 {
			t.Fatal("gpt-unknown-a should come before gpt-unknown-b alphabetically")
		}
	})

	t.Run("Non-reasoning before reasoning when different priority", func(t *testing.T) {
		a := Scenario{Models: []string{"gpt-3"}, Good: true, Reason: true}
		b := Scenario{Models: []string{"gpt-2"}, Cheap: true, Reason: false}
		if CompareScenarios(a, b) >= 0 {
			t.Fatal("Good should come before Cheap regardless of reasoning")
		}
	})

	t.Run("Untested scenarios come last", func(t *testing.T) {
		// Create a properly tested scenario
		a_tested := Scenario{
			Models: []string{"gpt-4"},
			GenSync: &Functionality{},
			In:     map[Modality]ModalCapability{ModalityText: {}},
			Out:    map[Modality]ModalCapability{ModalityText: {}},
		}
		// Create an untested scenario
		b_untested := Scenario{Models: []string{"zzz-model"}}

		if CompareScenarios(a_tested, b_untested) >= 0 {
			t.Fatal("tested scenario should come before untested")
		}
	})

	t.Run("Empty models sort alphabetically (empty comes first)", func(t *testing.T) {
		a := Scenario{Models: []string{}, Reason: false}
		b := Scenario{Models: []string{"gpt-4"}, Reason: false}
		if CompareScenarios(a, b) >= 0 {
			t.Fatal("empty model name should come before non-empty alphabetically")
		}
	})
}

func TestTriStateJSON(t *testing.T) {
	t.Run("Round trip", func(t *testing.T) {
		tests := []TriState{False, True, Flaky}
		for _, v := range tests {
			data, err := json.Marshal(v)
			if err != nil {
				t.Fatalf("Marshal failed: %v", err)
			}
			var got TriState
			if err := json.Unmarshal(data, &got); err != nil {
				t.Fatalf("Unmarshal failed: %v", err)
			}
			if got != v {
				t.Fatalf("round trip failed: %v -> %s -> %v", v, string(data), got)
			}
		}
	})
}

func TestReasonJSON(t *testing.T) {
	t.Run("Round trip", func(t *testing.T) {
		tests := []Reason{ReasonNone, ReasonInline, ReasonAuto}
		for _, v := range tests {
			data, err := json.Marshal(v)
			if err != nil {
				t.Fatalf("Marshal failed: %v", err)
			}
			var got Reason
			if err := json.Unmarshal(data, &got); err != nil {
				t.Fatalf("Unmarshal failed: %v", err)
			}
			if got != v {
				t.Fatalf("round trip failed: %v -> %s -> %v", v, string(data), got)
			}
		}
	})
}
