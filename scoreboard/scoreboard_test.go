// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package scoreboard

import (
	"encoding/json"
	"testing"
)

func TestModel(t *testing.T) {
	tests := []struct {
		m    Model
		want string
	}{
		{
			m:    Model{Model: "gpt-4", Reason: false},
			want: "gpt-4",
		},
		{
			m:    Model{Model: "gpt-4", Reason: true},
			want: "gpt-4_thinking",
		},
		{
			m:    Model{Model: "claude-3:opus", Reason: false},
			want: "claude-3-opus",
		},
		{
			m:    Model{Model: "claude-3:opus", Reason: true},
			want: "claude-3-opus_thinking",
		},
	}

	for _, tt := range tests {
		if got := tt.m.String(); got != tt.want {
			t.Fatalf("got %q, want %q", got, tt.want)
		}
	}
}

func TestModality(t *testing.T) {
	t.Run("Validate", func(t *testing.T) {
		tests := []Modality{ModalityAudio, ModalityDocument, ModalityImage, ModalityText, ModalityVideo}
		for _, m := range tests {
			if err := m.Validate(); err != nil {
				t.Fatalf("Modality %q: got err=%v", m, err)
			}
		}
	})

	t.Run("Error", func(t *testing.T) {
		tests := []Modality{Modality("invalid")}
		for _, m := range tests {
			if err := m.Validate(); err == nil {
				t.Fatalf("Modality %q: want error", m)
			}
		}
	})
}

func TestTriState(t *testing.T) {
	t.Run("String", func(t *testing.T) {
		tests := []struct {
			in   TriState
			want string
		}{
			{False, "false"},
			{True, "true"},
			{Flaky, "flaky"},
			{TriState(99), "TriState(99)"},
		}

		for _, tt := range tests {
			if got := tt.in.String(); got != tt.want {
				t.Fatalf("got %q, want %q", got, tt.want)
			}
		}
	})

	t.Run("GoString", func(t *testing.T) {
		tests := []struct {
			in   TriState
			want string
		}{
			{False, "false"},
			{True, "true"},
			{Flaky, "flaky"},
		}

		for _, tt := range tests {
			if got := tt.in.GoString(); got != tt.want {
				t.Fatalf("got %q, want %q", got, tt.want)
			}
		}
	})

	t.Run("Validate", func(t *testing.T) {
		tests := []TriState{False, True, Flaky}
		for _, ts := range tests {
			if err := ts.Validate(); err != nil {
				t.Fatalf("TriState %v: got err=%v", ts, err)
			}
		}
	})

	t.Run("Validate Error", func(t *testing.T) {
		tests := []TriState{TriState(99)}
		for _, ts := range tests {
			if err := ts.Validate(); err == nil {
				t.Fatalf("TriState %v: want error", ts)
			}
		}
	})

	t.Run("MarshalJSON", func(t *testing.T) {
		tests := []struct {
			ts   TriState
			want []byte
		}{
			{False, []byte(`"false"`)},
			{True, []byte(`"true"`)},
			{Flaky, []byte(`"flaky"`)},
		}

		for _, tt := range tests {
			got, err := tt.ts.MarshalJSON()
			if err != nil {
				t.Fatalf("TriState %v: got err=%v", tt.ts, err)
			}
			if string(got) != string(tt.want) {
				t.Fatalf("got %s, want %s", got, tt.want)
			}
		}
	})

	t.Run("MarshalJSON Error", func(t *testing.T) {
		tests := []TriState{TriState(99)}
		for _, ts := range tests {
			_, err := ts.MarshalJSON()
			if err == nil {
				t.Fatalf("TriState %v: want error", ts)
			}
		}
	})

	t.Run("UnmarshalJSON", func(t *testing.T) {
		tests := []struct {
			in   []byte
			want TriState
		}{
			{[]byte(`"false"`), False},
			{[]byte(`"true"`), True},
			{[]byte(`"flaky"`), Flaky},
		}

		for _, tt := range tests {
			var got TriState
			err := got.UnmarshalJSON(tt.in)
			if err != nil {
				t.Fatalf("input %s: got err=%v", tt.in, err)
			}
			if got != tt.want {
				t.Fatalf("got %v, want %v", got, tt.want)
			}
		}
	})

	t.Run("UnmarshalJSON Error", func(t *testing.T) {
		tests := [][]byte{
			[]byte(`"invalid"`),
			[]byte(`not valid json`),
		}

		for _, in := range tests {
			var got TriState
			if err := got.UnmarshalJSON(in); err == nil {
				t.Fatalf("input %s: want error", in)
			}
		}
	})
}

func TestFunctionality(t *testing.T) {
	t.Run("Validate", func(t *testing.T) {
		tests := []*Functionality{
			{
				ReportTokenUsage:   True,
				ReportFinishReason: True,
				Tools:              True,
				ToolsBiased:        False,
				ToolsIndecisive:    False,
			},
		}

		for _, f := range tests {
			if err := f.Validate(); err != nil {
				t.Fatalf("got err=%v", err)
			}
		}
	})

	t.Run("Validate Error", func(t *testing.T) {
		tests := []*Functionality{
			{ReportTokenUsage: TriState(99)},
			{Tools: False, ToolsBiased: True},
			{Tools: False, ToolsIndecisive: True},
			{Tools: False, ToolCallRequired: true},
		}

		for _, f := range tests {
			if err := f.Validate(); err == nil {
				t.Fatalf("got err=nil, want error")
			}
		}
	})

	t.Run("Less", func(t *testing.T) {
		tests := []struct {
			f1, f2 *Functionality
			want   bool
		}{
			{&Functionality{ReportRateLimits: false}, &Functionality{ReportRateLimits: true}, true},
			{&Functionality{ReportRateLimits: true}, &Functionality{ReportRateLimits: false}, false},
			{&Functionality{ReportTokenUsage: False}, &Functionality{ReportTokenUsage: True}, true},
			{&Functionality{ReportFinishReason: False}, &Functionality{ReportFinishReason: True}, true},
			{&Functionality{Seed: false}, &Functionality{Seed: true}, true},
			{&Functionality{Tools: False}, &Functionality{Tools: True}, true},
			{&Functionality{ToolCallRequired: false}, &Functionality{ToolCallRequired: true}, true},
			{&Functionality{JSON: false}, &Functionality{JSON: true}, true},
			{&Functionality{JSONSchema: false}, &Functionality{JSONSchema: true}, true},
			{&Functionality{Citations: false}, &Functionality{Citations: true}, true},
			{&Functionality{MaxTokens: false}, &Functionality{MaxTokens: true}, true},
			{&Functionality{StopSequence: false}, &Functionality{StopSequence: true}, true},
			{&Functionality{ReportRateLimits: true}, &Functionality{ReportRateLimits: true}, false},
		}

		for _, tt := range tests {
			if got := tt.f1.Less(tt.f2); got != tt.want {
				t.Fatalf("f1.Less(f2) = %v, want %v", got, tt.want)
			}
		}
	})
}

func TestScenario(t *testing.T) {
	t.Run("Untested", func(t *testing.T) {
		tests := []struct {
			s       *Scenario
			untested bool
		}{
			{&Scenario{Models: []string{"gpt-4"}}, true},
			{&Scenario{Models: []string{"gpt-4"}, GenSync: &Functionality{}}, false},
			{&Scenario{Models: []string{"gpt-4"}, GenStream: &Functionality{}}, false},
			{&Scenario{Models: []string{"gpt-4"}, In: map[Modality]ModalCapability{ModalityText: {}}, Out: map[Modality]ModalCapability{ModalityText: {}}}, false},
		}

		for _, tt := range tests {
			if got := tt.s.Untested(); got != tt.untested {
				t.Fatalf("Untested() = %v, want %v", got, tt.untested)
			}
		}
	})

	t.Run("Validate", func(t *testing.T) {
		tests := []*Scenario{
			{
				Models:  []string{"gpt-4"},
				In:      map[Modality]ModalCapability{ModalityText: {}},
				Out:     map[Modality]ModalCapability{ModalityText: {}},
				GenSync: &Functionality{},
			},
		}

		for _, s := range tests {
			if err := s.Validate(); err != nil {
				t.Fatalf("got err=%v", err)
			}
		}
	})

	t.Run("Validate Error", func(t *testing.T) {
		tests := []*Scenario{
			{Models: []string{}},
			{Models: []string{"gpt-4"}, In: map[Modality]ModalCapability{Modality("invalid"): {}}},
			{Models: []string{"gpt-4"}, Out: map[Modality]ModalCapability{Modality("invalid"): {}}},
			{Models: []string{"gpt-4"}, In: map[Modality]ModalCapability{ModalityText: {}}},
			{Models: []string{"gpt-4"}, Out: map[Modality]ModalCapability{ModalityText: {}}},
			{Models: []string{"gpt-4"}, GenSync: &Functionality{}},
		}

		for _, s := range tests {
			if err := s.Validate(); err == nil {
				t.Fatalf("got err=nil, want error")
			}
		}
	})
}

func TestScore(t *testing.T) {
	t.Run("Validate", func(t *testing.T) {
		tests := []*Score{
			{
				Country: "US",
				Scenarios: []Scenario{
					{
						Models:  []string{"gpt-4"},
						SOTA:    true,
						In:      map[Modality]ModalCapability{ModalityText: {}},
						Out:     map[Modality]ModalCapability{ModalityText: {}},
						GenSync: &Functionality{},
					},
					{
						Models:  []string{"gpt-3.5"},
						Good:    true,
						In:      map[Modality]ModalCapability{ModalityText: {}},
						Out:     map[Modality]ModalCapability{ModalityText: {}},
						GenSync: &Functionality{},
					},
					{
						Models:  []string{"gpt-2"},
						Cheap:   true,
						In:      map[Modality]ModalCapability{ModalityText: {}},
						Out:     map[Modality]ModalCapability{ModalityText: {}},
						GenSync: &Functionality{},
					},
				},
			},
		}

		for _, s := range tests {
			if err := s.Validate(); err != nil {
				t.Fatalf("got err=%v", err)
			}
		}
	})

	t.Run("Validate Error", func(t *testing.T) {
		tests := []*Score{
			{
				Scenarios: []Scenario{
					{Models: []string{"gpt-4"}, Reason: false, GenSync: &Functionality{}},
					{Models: []string{"gpt-4"}, Reason: false, GenSync: &Functionality{}},
				},
			},
			{
				Scenarios: []Scenario{
					{Models: []string{"gpt-3.5"}, SOTA: false, GenSync: &Functionality{}},
					{Models: []string{"gpt-4"}, SOTA: true, GenSync: &Functionality{}},
					{Models: []string{"gpt-2"}, Good: true, Cheap: true, GenSync: &Functionality{}},
				},
			},
			{
				Scenarios: []Scenario{
					{Models: []string{"gpt-4"}, GenSync: &Functionality{}},
					{Models: []string{"gpt-3.5"}, GenSync: &Functionality{}},
				},
			},
			{
				Scenarios: []Scenario{
					{Models: []string{"gpt-4"}, SOTA: true, GenSync: &Functionality{}},
					{Models: []string{"gpt-3.5"}, SOTA: true, GenSync: &Functionality{}},
				},
			},
			{
				Scenarios: []Scenario{
					{Models: []string{"gpt-4"}, SOTA: true, GenSync: &Functionality{}},
					{Models: []string{"gpt-3.5"}, GenSync: &Functionality{}},
					{Models: []string{"gpt-2"}, Cheap: true, GenSync: &Functionality{}},
				},
			},
			{
				Scenarios: []Scenario{
					{Models: []string{"gpt-4"}, SOTA: true, GenSync: &Functionality{}},
					{Models: []string{"gpt-2"}, Cheap: true, GenSync: &Functionality{}},
					{Models: []string{"gpt-3.5"}, Good: true, GenSync: &Functionality{}},
				},
			},
			{
				Scenarios: []Scenario{
					{Models: []string{"gpt-4"}, SOTA: true, GenSync: &Functionality{}},
				},
			},
			{
				Scenarios: []Scenario{
					{Models: []string{"gpt-4"}, SOTA: true, GenSync: &Functionality{}},
					{Models: []string{"gpt-3.5"}, Good: true, GenSync: &Functionality{}},
					{Models: []string{"gpt-2"}, Cheap: true, GenSync: &Functionality{}},
					{Models: []string{"gpt-1"}, Cheap: true, GenSync: &Functionality{}},
				},
			},
			{
				Scenarios: []Scenario{
					{Models: []string{}, SOTA: true, GenSync: &Functionality{}},
					{Models: []string{"gpt-3.5"}, Good: true, GenSync: &Functionality{}},
					{Models: []string{"gpt-2"}, Cheap: true, GenSync: &Functionality{}},
				},
			},
		}

		for _, s := range tests {
			if err := s.Validate(); err == nil {
				t.Fatalf("got err=nil, want error")
			}
		}
	})

	t.Run("Validate no Cheap Error", func(t *testing.T) {
		tests := []*Score{
			{
				Scenarios: []Scenario{
					{Models: []string{"gpt-4"}, SOTA: true, GenSync: &Functionality{}},
					{Models: []string{"gpt-3.5"}, Good: true, GenSync: &Functionality{}},
					{Models: []string{"gpt-2"}, GenSync: &Functionality{}},
				},
			},
		}

		for _, s := range tests {
			if err := s.Validate(); err == nil {
				t.Fatalf("got err=nil, want error")
			}
		}
	})

	t.Run("Validate with reason", func(t *testing.T) {
		tests := []*Score{
			{
				Country: "US",
				Scenarios: []Scenario{
					{
						Models:  []string{"gpt-4"},
						SOTA:    true,
						Reason:  true,
						In:      map[Modality]ModalCapability{ModalityText: {}},
						Out:     map[Modality]ModalCapability{ModalityText: {}},
						GenSync: &Functionality{},
					},
					{
						Models:  []string{"gpt-3.5"},
						Good:    true,
						Reason:  false,
						In:      map[Modality]ModalCapability{ModalityText: {}},
						Out:     map[Modality]ModalCapability{ModalityText: {}},
						GenSync: &Functionality{},
					},
					{
						Models:  []string{"gpt-2"},
						Cheap:   true,
						Reason:  false,
						In:      map[Modality]ModalCapability{ModalityText: {}},
						Out:     map[Modality]ModalCapability{ModalityText: {}},
						GenSync: &Functionality{},
					},
				},
			},
			{
				Country: "US",
				Scenarios: []Scenario{
					{
						Models:  []string{"gpt-4"},
						SOTA:    true,
						Reason:  true,
						In:      map[Modality]ModalCapability{ModalityText: {}},
						Out:     map[Modality]ModalCapability{ModalityText: {}},
						GenSync: &Functionality{},
					},
					{
						Models:  []string{"gpt-4"},
						Good:    true,
						Reason:  false,
						In:      map[Modality]ModalCapability{ModalityText: {}},
						Out:     map[Modality]ModalCapability{ModalityText: {}},
						GenSync: &Functionality{},
					},
					{
						Models:  []string{"gpt-2"},
						Cheap:   true,
						Reason:  false,
						In:      map[Modality]ModalCapability{ModalityText: {}},
						Out:     map[Modality]ModalCapability{ModalityText: {}},
						GenSync: &Functionality{},
					},
				},
			},
		}

		for _, s := range tests {
			if err := s.Validate(); err != nil {
				t.Fatalf("got err=%v", err)
			}
		}
	})

	t.Run("Validate per-modality SOTA/Good/Cheap", func(t *testing.T) {
		tests := []*Score{
			{
				Country: "US",
				Scenarios: []Scenario{
					{
						Models:  []string{"gpt-4"},
						SOTA:    true,
						In:      map[Modality]ModalCapability{ModalityText: {}},
						Out:     map[Modality]ModalCapability{ModalityText: {}},
						GenSync: &Functionality{},
					},
					{
						Models:  []string{"gpt-4-vision"},
						SOTA:    true,
						In:      map[Modality]ModalCapability{ModalityImage: {}},
						Out:     map[Modality]ModalCapability{ModalityImage: {}},
						GenSync: &Functionality{},
					},
					{
						Models:  []string{"gpt-3.5"},
						Good:    true,
						In:      map[Modality]ModalCapability{ModalityText: {}},
						Out:     map[Modality]ModalCapability{ModalityText: {}},
						GenSync: &Functionality{},
					},
					{
						Models:  []string{"gpt-3.5-vision"},
						Good:    true,
						In:      map[Modality]ModalCapability{ModalityImage: {}},
						Out:     map[Modality]ModalCapability{ModalityImage: {}},
						GenSync: &Functionality{},
					},
					{
						Models:  []string{"gpt-2"},
						Cheap:   true,
						In:      map[Modality]ModalCapability{ModalityText: {}},
						Out:     map[Modality]ModalCapability{ModalityText: {}},
						GenSync: &Functionality{},
					},
					{
						Models:  []string{"gpt-2-vision"},
						Cheap:   true,
						In:      map[Modality]ModalCapability{ModalityImage: {}},
						Out:     map[Modality]ModalCapability{ModalityImage: {}},
						GenSync: &Functionality{},
					},
				},
			},
		}

		for _, s := range tests {
			if err := s.Validate(); err != nil {
				t.Fatalf("got err=%v", err)
			}
		}
	})

	t.Run("Validate per-modality Error", func(t *testing.T) {
		tests := []*Score{
			{
				Country: "US",
				Scenarios: []Scenario{
					{
						Models:  []string{"gpt-4"},
						SOTA:    true,
						In:      map[Modality]ModalCapability{ModalityText: {}},
						Out:     map[Modality]ModalCapability{ModalityText: {}},
						GenSync: &Functionality{},
					},
					{
						Models:  []string{"claude-3-opus"},
						SOTA:    true,
						In:      map[Modality]ModalCapability{ModalityText: {}},
						Out:     map[Modality]ModalCapability{ModalityText: {}},
						GenSync: &Functionality{},
					},
					{
						Models:  []string{"gpt-3.5"},
						Good:    true,
						In:      map[Modality]ModalCapability{ModalityText: {}},
						Out:     map[Modality]ModalCapability{ModalityText: {}},
						GenSync: &Functionality{},
					},
					{
						Models:  []string{"gpt-2"},
						Cheap:   true,
						In:      map[Modality]ModalCapability{ModalityText: {}},
						Out:     map[Modality]ModalCapability{ModalityText: {}},
						GenSync: &Functionality{},
					},
				},
			},
			{
				Country: "US",
				Scenarios: []Scenario{
					{
						Models:  []string{"gpt-4"},
						SOTA:    true,
						In:      map[Modality]ModalCapability{ModalityText: {}},
						Out:     map[Modality]ModalCapability{ModalityText: {}},
						GenSync: &Functionality{},
					},
					{
						Models:  []string{"gpt-3.5"},
						Good:    true,
						In:      map[Modality]ModalCapability{ModalityText: {}},
						Out:     map[Modality]ModalCapability{ModalityText: {}},
						GenSync: &Functionality{},
					},
					{
						Models:  []string{"claude-3-sonnet"},
						Good:    true,
						In:      map[Modality]ModalCapability{ModalityText: {}},
						Out:     map[Modality]ModalCapability{ModalityText: {}},
						GenSync: &Functionality{},
					},
					{
						Models:  []string{"gpt-2"},
						Cheap:   true,
						In:      map[Modality]ModalCapability{ModalityText: {}},
						Out:     map[Modality]ModalCapability{ModalityText: {}},
						GenSync: &Functionality{},
					},
				},
			},
			{
				Country: "US",
				Scenarios: []Scenario{
					{
						Models:  []string{"gpt-4"},
						SOTA:    true,
						In:      map[Modality]ModalCapability{ModalityText: {}},
						Out:     map[Modality]ModalCapability{ModalityText: {}},
						GenSync: &Functionality{},
					},
					{
						Models:  []string{"gpt-3.5"},
						Good:    true,
						In:      map[Modality]ModalCapability{ModalityText: {}},
						Out:     map[Modality]ModalCapability{ModalityText: {}},
						GenSync: &Functionality{},
					},
					{
						Models:  []string{"gpt-2"},
						Cheap:   true,
						In:      map[Modality]ModalCapability{ModalityText: {}},
						Out:     map[Modality]ModalCapability{ModalityText: {}},
						GenSync: &Functionality{},
					},
					{
						Models:  []string{"gemini-1.5-flash"},
						Cheap:   true,
						In:      map[Modality]ModalCapability{ModalityText: {}},
						Out:     map[Modality]ModalCapability{ModalityText: {}},
						GenSync: &Functionality{},
					},
				},
			},
		}

		for _, s := range tests {
			if err := s.Validate(); err == nil {
				t.Fatalf("got err=nil, want error")
			}
		}
	})

	t.Run("Validate modality scenarios success", func(t *testing.T) {
		tests := []*Score{
			{
				Country: "US",
				Scenarios: []Scenario{
					{
						Models:  []string{"gpt-4-text"},
						SOTA:    true,
						In:      map[Modality]ModalCapability{ModalityText: {}},
						Out:     map[Modality]ModalCapability{ModalityText: {}},
						GenSync: &Functionality{},
					},
					{
						Models:  []string{"gpt-3.5-text"},
						Good:    true,
						In:      map[Modality]ModalCapability{ModalityText: {}},
						Out:     map[Modality]ModalCapability{ModalityText: {}},
						GenSync: &Functionality{},
					},
					{
						Models:  []string{"gpt-2-text"},
						Cheap:   true,
						In:      map[Modality]ModalCapability{ModalityText: {}},
						Out:     map[Modality]ModalCapability{ModalityText: {}},
						GenSync: &Functionality{},
					},
					{
						Models:  []string{"gpt-4-vision"},
						SOTA:    true,
						In:      map[Modality]ModalCapability{ModalityImage: {}},
						Out:     map[Modality]ModalCapability{ModalityImage: {}},
						GenSync: &Functionality{},
					},
					{
						Models:  []string{"claude-vision"},
						Good:    true,
						In:      map[Modality]ModalCapability{ModalityImage: {}},
						Out:     map[Modality]ModalCapability{ModalityImage: {}},
						GenSync: &Functionality{},
					},
					{
						Models:  []string{"llava"},
						Cheap:   true,
						In:      map[Modality]ModalCapability{ModalityImage: {}},
						Out:     map[Modality]ModalCapability{ModalityImage: {}},
						GenSync: &Functionality{},
					},
				},
			},
			{
				Country: "US",
				Scenarios: []Scenario{
					{
						Models:  []string{"gpt-4"},
						SOTA:    true,
						In:      map[Modality]ModalCapability{ModalityText: {}},
						Out:     map[Modality]ModalCapability{ModalityText: {}},
						GenSync: &Functionality{},
					},
					{
						Models:  []string{"gpt-3.5"},
						Good:    true,
						In:      map[Modality]ModalCapability{ModalityText: {}},
						Out:     map[Modality]ModalCapability{ModalityText: {}},
						GenSync: &Functionality{},
					},
					{
						Models:  []string{"gpt-2"},
						Cheap:   true,
						In:      map[Modality]ModalCapability{ModalityText: {}},
						Out:     map[Modality]ModalCapability{ModalityText: {}},
						GenSync: &Functionality{},
					},
					{
						Models:  []string{"gpt-4-vision"},
						SOTA:    true,
						In:      map[Modality]ModalCapability{ModalityImage: {}},
						Out:     map[Modality]ModalCapability{ModalityImage: {}},
						GenSync: &Functionality{},
					},
					{
						Models:  []string{"gpt-3.5-vision"},
						Good:    true,
						In:      map[Modality]ModalCapability{ModalityImage: {}},
						Out:     map[Modality]ModalCapability{ModalityImage: {}},
						GenSync: &Functionality{},
					},
					{
						Models:  []string{"gpt-2-vision"},
						Cheap:   true,
						In:      map[Modality]ModalCapability{ModalityImage: {}},
						Out:     map[Modality]ModalCapability{ModalityImage: {}},
						GenSync: &Functionality{},
					},
				},
			},
		}

		for _, s := range tests {
			if err := s.Validate(); err != nil {
				t.Fatalf("got err=%v", err)
			}
		}
	})

	t.Run("Validate modality scenarios error", func(t *testing.T) {
		tests := []*Score{
			{
				Country: "US",
				Scenarios: []Scenario{
					{
						Models:  []string{"gpt-4-vision"},
						SOTA:    true,
						In:      map[Modality]ModalCapability{ModalityImage: {}},
						Out:     map[Modality]ModalCapability{ModalityImage: {}},
						GenSync: &Functionality{},
					},
					{
						Models:  []string{"claude-3-5-sonnet"},
						SOTA:    true,
						In:      map[Modality]ModalCapability{ModalityImage: {}},
						Out:     map[Modality]ModalCapability{ModalityImage: {}},
						GenSync: &Functionality{},
					},
					{
						Models:  []string{"gpt-4-standard"},
						Good:    true,
						In:      map[Modality]ModalCapability{ModalityImage: {}},
						Out:     map[Modality]ModalCapability{ModalityImage: {}},
						GenSync: &Functionality{},
					},
					{
						Models:  []string{"gpt-3.5-vision"},
						Cheap:   true,
						In:      map[Modality]ModalCapability{ModalityImage: {}},
						Out:     map[Modality]ModalCapability{ModalityImage: {}},
						GenSync: &Functionality{},
					},
				},
			},
			{
				Country: "US",
				Scenarios: []Scenario{
					{
						Models:  []string{"gpt-4"},
						SOTA:    true,
						GenSync: &Functionality{},
					},
					{
						Models:  []string{"claude-3-opus"},
						SOTA:    true,
						GenSync: &Functionality{},
					},
					{
						Models:  []string{"gpt-3.5"},
						Good:    true,
						GenSync: &Functionality{},
					},
					{
						Models:  []string{"gpt-2"},
						Cheap:   true,
						GenSync: &Functionality{},
					},
				},
			},
			{
				Country: "US",
				Scenarios: []Scenario{
					{
						Models:  []string{"gpt-4-vision"},
						SOTA:    true,
						In:      map[Modality]ModalCapability{ModalityImage: {}},
						Out:     map[Modality]ModalCapability{ModalityImage: {}},
						GenSync: &Functionality{},
					},
					{
						Models:  []string{"gpt-2-vision"},
						Cheap:   true,
						In:      map[Modality]ModalCapability{ModalityImage: {}},
						Out:     map[Modality]ModalCapability{ModalityImage: {}},
						GenSync: &Functionality{},
					},
					{
						Models:  []string{"gpt-3.5-vision"},
						Good:    true,
						In:      map[Modality]ModalCapability{ModalityImage: {}},
						Out:     map[Modality]ModalCapability{ModalityImage: {}},
						GenSync: &Functionality{},
					},
					{
						Models:  []string{"gpt-3.5"},
						Good:    true,
						In:      map[Modality]ModalCapability{ModalityText: {}},
						Out:     map[Modality]ModalCapability{ModalityText: {}},
						GenSync: &Functionality{},
					},
					{
						Models:  []string{"gpt-2"},
						Cheap:   true,
						In:      map[Modality]ModalCapability{ModalityText: {}},
						Out:     map[Modality]ModalCapability{ModalityText: {}},
						GenSync: &Functionality{},
					},
					{
						Models:  []string{"gpt-4"},
						SOTA:    true,
						In:      map[Modality]ModalCapability{ModalityText: {}},
						Out:     map[Modality]ModalCapability{ModalityText: {}},
						GenSync: &Functionality{},
					},
				},
			},
		}

		for _, s := range tests {
			if err := s.Validate(); err == nil {
				t.Fatalf("got err=nil, want error")
			}
		}
	})

	t.Run("Validate SOTA not first in modality group", func(t *testing.T) {
		// SOTA should be first within its modality group
		s := &Score{
			Country: "US",
			Scenarios: []Scenario{
				{
					Models:  []string{"gpt-3.5-vision"},
					Good:    true,
					In:      map[Modality]ModalCapability{ModalityImage: {}},
					Out:     map[Modality]ModalCapability{ModalityImage: {}},
					GenSync: &Functionality{},
				},
				{
					Models:  []string{"gpt-4-vision"},
					SOTA:    true,
					In:      map[Modality]ModalCapability{ModalityImage: {}},
					Out:     map[Modality]ModalCapability{ModalityImage: {}},
					GenSync: &Functionality{},
				},
				{
					Models:  []string{"gpt-2-vision"},
					Cheap:   true,
					In:      map[Modality]ModalCapability{ModalityImage: {}},
					Out:     map[Modality]ModalCapability{ModalityImage: {}},
					GenSync: &Functionality{},
				},
			},
		}
		if err := s.Validate(); err == nil {
			t.Fatal("should fail when SOTA is not first in image modality group")
		}
	})

	t.Run("Validate Good before SOTA per modality", func(t *testing.T) {
		// Good should not come before SOTA even across different modalities
		s := &Score{
			Country: "US",
			Scenarios: []Scenario{
				{
					Models:  []string{"gpt-3.5-text"},
					Good:    true,
					In:      map[Modality]ModalCapability{ModalityText: {}},
					Out:     map[Modality]ModalCapability{ModalityText: {}},
					GenSync: &Functionality{},
				},
				{
					Models:  []string{"gpt-4-text"},
					SOTA:    true,
					In:      map[Modality]ModalCapability{ModalityText: {}},
					Out:     map[Modality]ModalCapability{ModalityText: {}},
					GenSync: &Functionality{},
				},
				{
					Models:  []string{"gpt-2-text"},
					Cheap:   true,
					In:      map[Modality]ModalCapability{ModalityText: {}},
					Out:     map[Modality]ModalCapability{ModalityText: {}},
					GenSync: &Functionality{},
				},
			},
		}
		if err := s.Validate(); err == nil {
			t.Fatal("should fail when Good comes before SOTA in text modality")
		}
	})
}

func TestReason(t *testing.T) {
	t.Run("Validate", func(t *testing.T) {
		tests := []Reason{ReasonNone, ReasonInline, ReasonAuto}
		for _, r := range tests {
			if err := r.Validate(); err != nil {
				t.Fatalf("Reason %v: got err=%v", r, err)
			}
		}
	})

	t.Run("Validate Error", func(t *testing.T) {
		tests := []Reason{Reason(99)}
		for _, r := range tests {
			if err := r.Validate(); err == nil {
				t.Fatalf("Reason %v: want error", r)
			}
		}
	})

	t.Run("MarshalJSON", func(t *testing.T) {
		tests := []struct {
			in   Reason
			want []byte
		}{
			{ReasonNone, []byte(`"none"`)},
			{ReasonInline, []byte(`"inline"`)},
			{ReasonAuto, []byte(`"auto"`)},
		}

		for _, tt := range tests {
			got, err := tt.in.MarshalJSON()
			if err != nil {
				t.Fatalf("Reason %v: got err=%v", tt.in, err)
			}
			if string(got) != string(tt.want) {
				t.Fatalf("got %s, want %s", got, tt.want)
			}
		}
	})

	t.Run("MarshalJSON Error", func(t *testing.T) {
		tests := []Reason{Reason(99)}
		for _, r := range tests {
			_, err := r.MarshalJSON()
			if err == nil {
				t.Fatalf("Reason %v: want error", r)
			}
		}
	})

	t.Run("UnmarshalJSON", func(t *testing.T) {
		tests := []struct {
			in   []byte
			want Reason
		}{
			{[]byte(`"none"`), ReasonNone},
			{[]byte(`"inline"`), ReasonInline},
			{[]byte(`"auto"`), ReasonAuto},
		}

		for _, tt := range tests {
			var got Reason
			if err := got.UnmarshalJSON(tt.in); err != nil {
				t.Fatalf("input %s: got err=%v", tt.in, err)
			}
			if got != tt.want {
				t.Fatalf("got %v, want %v", got, tt.want)
			}
		}
	})

	t.Run("UnmarshalJSON Error", func(t *testing.T) {
		tests := [][]byte{
			[]byte(`"invalid"`),
			[]byte(`not json`),
		}

		for _, in := range tests {
			var got Reason
			if err := got.UnmarshalJSON(in); err == nil {
				t.Fatalf("input %s: want error", in)
			}
		}
	})
}

func TestCompareScenarios(t *testing.T) {
	tests := []struct {
		a, b Scenario
		want int
	}{
		{Scenario{Models: []string{"gpt-4"}, SOTA: true, Reason: false}, Scenario{Models: []string{"gpt-3"}, Good: true, Reason: false}, -1},
		{Scenario{Models: []string{"gpt-3"}, Good: true, Reason: false}, Scenario{Models: []string{"gpt-2"}, Cheap: true, Reason: false}, -1},
		{Scenario{Models: []string{"gpt-4"}, SOTA: true, Reason: false}, Scenario{Models: []string{"gpt-2"}, Cheap: true, Reason: false}, -1},
		{Scenario{Models: []string{"gpt-4"}, SOTA: true, Reason: true}, Scenario{Models: []string{"gpt-4"}, SOTA: true, Reason: false}, -1},
		{Scenario{Models: []string{"gpt-4"}, SOTA: true, Reason: false}, Scenario{Models: []string{"gpt-3"}, Reason: false}, -1},
		{Scenario{Models: []string{"gpt-unknown-b"}, Reason: false}, Scenario{Models: []string{"gpt-unknown-a"}, Reason: false}, 1},
		{Scenario{Models: []string{"gpt-3"}, Good: true, Reason: true}, Scenario{Models: []string{"gpt-2"}, Cheap: true, Reason: false}, -1},
	}

	for _, tt := range tests {
		cmp := CompareScenarios(tt.a, tt.b)
		// Normalize comparison result to -1, 0, or 1
		var got int
		if cmp < 0 {
			got = -1
		} else if cmp > 0 {
			got = 1
		}

		if got != tt.want {
			t.Fatalf("CompareScenarios got %v, want %v", got, tt.want)
		}
	}

	t.Run("Untested scenarios come last", func(t *testing.T) {
		// Create a properly tested scenario
		a_tested := Scenario{
			Models:  []string{"gpt-4"},
			GenSync: &Functionality{},
			In:      map[Modality]ModalCapability{ModalityText: {}},
			Out:     map[Modality]ModalCapability{ModalityText: {}},
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
