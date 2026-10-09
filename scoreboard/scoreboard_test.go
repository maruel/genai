// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Tests for the scoreboard package.

package scoreboard

import (
	"bytes"
	"encoding/json"
	"reflect"
	"slices"
	"strings"
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
		tests := []Modality{ModalityAudio, ModalityDecision, ModalityDocument, ModalityEmbedding, ModalityImage, ModalityText, ModalityVideo}
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
			if !bytes.Equal(got, tt.want) {
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
	t.Run("JSON", func(t *testing.T) {
		t.Run("valid", func(t *testing.T) {
			for _, tc := range []struct {
				name, raw string
				measured  bool
				value     TriState
			}{
				{"unmeasured", `{}`, false, False},
				{"false", `{"outOfOrder":"false"}`, true, False},
				{"true", `{"outOfOrder":"true"}`, true, True},
				{"flaky", `{"outOfOrder":"flaky"}`, true, Flaky},
			} {
				t.Run(tc.name, func(t *testing.T) {
					f := Functionality{}
					if err := json.Unmarshal([]byte(tc.raw), &f); err != nil {
						t.Fatal(err)
					}
					value := f.OutOfOrder
					if (value != nil) != tc.measured {
						t.Fatal("lost presence")
					}
					if tc.measured && *value != tc.value {
						t.Fatal("wrong value")
					}
					if err := f.Validate(); err != nil {
						t.Fatal(err)
					}
					raw, err := json.Marshal(f)
					if err != nil {
						t.Fatal(err)
					}
					if string(raw) != tc.raw {
						t.Fatalf("round trip=%s want=%s", raw, tc.raw)
					}
				})
			}
		})
		t.Run("error", func(t *testing.T) {
			f := Functionality{}
			if err := json.Unmarshal([]byte(`{"outOfOrder":"unknown"}`), &f); err == nil {
				t.Fatal("accepted invalid score")
			}
		})
	})
	t.Run("Validate", func(t *testing.T) {
		t.Run("valid", func(t *testing.T) {
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

		t.Run("error", func(t *testing.T) {
			tests := []*Functionality{
				{OutOfOrder: new(TriState(3))},
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
	})

	t.Run("Less", func(t *testing.T) {
		tests := []struct {
			f1, f2 *Functionality
			want   bool
		}{
			{&Functionality{}, &Functionality{OutOfOrder: new(True)}, false},
			{&Functionality{}, &Functionality{OutOfOrder: new(False)}, false},
			{&Functionality{OutOfOrder: new(True)}, &Functionality{}, false},
			{&Functionality{OutOfOrder: new(False)}, &Functionality{}, false},
			{&Functionality{OutOfOrder: new(False)}, &Functionality{OutOfOrder: new(True)}, true},
			{&Functionality{OutOfOrder: new(True)}, &Functionality{OutOfOrder: new(False)}, false},
			{&Functionality{OutOfOrder: new(Flaky)}, &Functionality{OutOfOrder: new(True)}, true},
			{&Functionality{ReportRateLimits: false}, &Functionality{ReportRateLimits: true}, true},
			{&Functionality{ReportRateLimits: true}, &Functionality{ReportRateLimits: false}, false},
			{&Functionality{ReportTokenUsage: False}, &Functionality{ReportTokenUsage: True}, true},
			{&Functionality{ReportFinishReason: False}, &Functionality{ReportFinishReason: True}, true},
			{&Functionality{Seed: false}, &Functionality{Seed: true}, true},
			{&Functionality{Tools: False}, &Functionality{Tools: True}, true},
			{&Functionality{ToolCallRequired: false}, &Functionality{ToolCallRequired: true}, true},
			{&Functionality{JSON: false}, &Functionality{JSON: true}, true},
			{&Functionality{}, &Functionality{JSONSchema: JSONSchema{Object: True}}, true},
			{&Functionality{JSONSchema: JSONSchema{Object: True}}, &Functionality{JSONSchema: JSONSchema{Object: True, Array: True}}, true},
			{&Functionality{JSONSchema: JSONSchema{Object: True}}, &Functionality{JSONSchema: JSONSchema{Object: True}}, false},
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
			s        *Scenario
			untested bool
		}{
			{&Scenario{Models: []string{"gpt-4"}}, true},
			{&Scenario{Models: []string{"gpt-4"}, GenSync: &Functionality{}}, false},
			{&Scenario{Models: []string{"gpt-4"}, GenStream: &Functionality{}}, false},
			{&Scenario{Models: []string{"clef"}, SystemOne: &DecisionFunctionality{}}, false},
			{&Scenario{Models: []string{"embed"}, Embed: &EmbeddingFunctionality{}}, false},
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
			{Models: []string{"new-model"}, GenSync: &Functionality{}},
			{Models: []string{"new-model"}, GenStream: &Functionality{}},
			{Models: []string{"clef"}, SystemOne: &DecisionFunctionality{}},
			{Models: []string{"embed"}, Embed: &EmbeddingFunctionality{}},
			{Models: []string{"embed"}, In: map[Modality]ModalCapability{ModalityText: {Inline: true}}, Out: map[Modality]ModalCapability{ModalityEmbedding: {Inline: true}}, Embed: &EmbeddingFunctionality{Dimensions: 768}},
			{Models: []string{"new-model"}, GenSync: &Functionality{}, GenStream: &Functionality{}},
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
			{Models: []string{"gpt-4"}, GenSync: &Functionality{JSON: true}},
			{Models: []string{"gpt-4"}, GenStream: &Functionality{JSON: true}},
			{Models: []string{"clef"}, SystemOne: &DecisionFunctionality{Noul: true}},
			{Models: []string{"embed"}, Embed: &EmbeddingFunctionality{Dimensions: 768}},
			{Models: []string{"embed"}, Embed: &EmbeddingFunctionality{RequestedDimensions: new(false)}},
			{Models: []string{"embed"}, In: map[Modality]ModalCapability{ModalityText: {Inline: true}}, Out: map[Modality]ModalCapability{ModalityEmbedding: {Inline: true}}, Embed: &EmbeddingFunctionality{Dimensions: -1}},
			{Models: []string{"embed"}, In: map[Modality]ModalCapability{ModalityText: {Inline: true}}, Out: map[Modality]ModalCapability{ModalityText: {Inline: true}}, Embed: &EmbeddingFunctionality{Dimensions: 768}},
			{Models: []string{"clef"}, SystemOne: &DecisionFunctionality{ReportTokenUsage: TriState(2)}},
			{Models: []string{"gpt-4"}, GenSync: &Functionality{}, GenStream: &Functionality{JSON: true}},
		}

		for _, s := range tests {
			if err := s.Validate(); err == nil {
				t.Fatalf("got err=nil, want error")
			}
		}
	})
}

func TestScore(t *testing.T) {
	t.Run("Embedding", func(t *testing.T) {
		measured := &EmbeddingFunctionality{Dimensions: 768}
		s := Score{Scenarios: []Scenario{{Models: []string{"qualified"}, Embed: measured}, {Models: []string{"pending"}, Embed: &EmbeddingFunctionality{}}, {Models: []string{"untested"}}, {Models: []string{"generation"}, GenSync: &Functionality{}}}}
		if s.Embedding("qualified") != measured {
			t.Fatal("qualified embedding missing")
		}
		for _, model := range []string{"missing", "pending", "untested", "generation", ""} {
			if s.Embedding(model) != nil {
				t.Fatalf("claimed support for %s", model)
			}
		}
	})
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
					{Models: []string{"gpt-4"}, SOTA: true, GenSync: &Functionality{}},
					{Models: []string{"gpt-3.5"}, SOTA: true, GenSync: &Functionality{}},
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

	t.Run("Validate qualification markers", func(t *testing.T) {
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
			if err := s.Validate(); err != nil {
				t.Fatalf("got err=%v", err)
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

	t.Run("Validate per-modality tiers", func(t *testing.T) {
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
			if !bytes.Equal(got, tt.want) {
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

func TestConsolidateUntestedScenarios(t *testing.T) {
	tests := []struct {
		name      string
		scenarios []Scenario
		wantCount int
		wantCheck func(*testing.T, []Scenario)
	}{
		{
			name: "Basic consolidation",
			scenarios: []Scenario{
				{Models: []string{"model-a"}, Comments: "reason"},
				{Models: []string{"model-b"}, Comments: "reason"},
			},
			wantCount: 1,
			wantCheck: func(t *testing.T, result []Scenario) {
				if len(result[0].Models) != 2 {
					t.Fatalf("expected 2 models, got %d", len(result[0].Models))
				}
				if result[0].Models[0] != "model-a" || result[0].Models[1] != "model-b" {
					t.Fatalf("expected [model-a, model-b], got %v", result[0].Models)
				}
			},
		},
		{
			name: "Do not merge with different reasons",
			scenarios: []Scenario{
				{Models: []string{"model-a"}, Comments: "reason1"},
				{Models: []string{"model-b"}, Comments: "reason2"},
			},
			wantCount: 2,
		},
		{
			name: "Do not merge scenarios with SOTA flag",
			scenarios: []Scenario{
				{Models: []string{"model-a"}, Comments: "reason", SOTA: true},
				{Models: []string{"model-b"}, Comments: "reason"},
			},
			wantCount: 2,
		},
		{
			name: "Do not merge scenarios with Good flag",
			scenarios: []Scenario{
				{Models: []string{"model-a"}, Comments: "reason", Good: true},
				{Models: []string{"model-b"}, Comments: "reason"},
			},
			wantCount: 2,
		},
		{
			name: "Do not merge scenarios with Cheap flag",
			scenarios: []Scenario{
				{Models: []string{"model-a"}, Comments: "reason", Cheap: true},
				{Models: []string{"model-b"}, Comments: "reason"},
			},
			wantCount: 2,
		},
		{
			name: "Avoid duplicate models",
			scenarios: []Scenario{
				{Models: []string{"model-a", "model-b"}, Comments: "reason"},
				{Models: []string{"model-b", "model-c"}, Comments: "reason"},
			},
			wantCount: 1,
			wantCheck: func(t *testing.T, result []Scenario) {
				if len(result[0].Models) != 3 {
					t.Fatalf("expected 3 models, got %d", len(result[0].Models))
				}
				expected := []string{"model-a", "model-b", "model-c"}
				for i, m := range result[0].Models {
					if m != expected[i] {
						t.Fatalf("expected %q at position %d, got %q", expected[i], i, m)
					}
				}
			},
		},
		{
			name: "Skip tested scenarios",
			scenarios: []Scenario{
				{Models: []string{"model-a"}, Comments: "reason", GenSync: &Functionality{}},
				{Models: []string{"model-b"}, Comments: "reason"},
			},
			wantCount: 1,
			wantCheck: func(t *testing.T, result []Scenario) {
				if result[0].Models[0] != "model-b" {
					t.Fatalf("expected model-b, got %v", result[0].Models)
				}
			},
		},
		{
			name: "Multiple consolidation groups",
			scenarios: []Scenario{
				{Models: []string{"model-a"}, Comments: "reason1"},
				{Models: []string{"model-b"}, Comments: "reason1"},
				{Models: []string{"model-c"}, Comments: "reason2"},
				{Models: []string{"model-d"}, Comments: "reason2"},
			},
			wantCount: 2,
			wantCheck: func(t *testing.T, result []Scenario) {
				if len(result[0].Models) != 2 || len(result[1].Models) != 2 {
					t.Fatalf("expected 2 models in each group")
				}
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := ConsolidateUntestedScenarios(tt.scenarios)
			if len(result) != tt.wantCount {
				t.Fatalf("expected %d scenarios, got %d", tt.wantCount, len(result))
			}
			if tt.wantCheck != nil {
				tt.wantCheck(t, result)
			}
		})
	}
}

func TestBFLScoreboardValidation(t *testing.T) {
	// This test reproduces the bug where consolidation + sorting creates invalid scenarios
	// The issue is that when consolidating untested scenarios with preference flags,
	// the resulting score fails validation due to duplicate models.
	s := &Score{
		Country: "DE",
		Scenarios: []Scenario{
			{
				Comments: "Untested",
				Models:   []string{"flux-pro-1.1-ultra"},
				SOTA:     true,
				Reason:   false,
			},
			{
				Comments: "Untested",
				Models:   []string{"flux-pro-1.1"},
				Good:     true,
				Reason:   false,
			},
			{
				Comments: "Has In/Out",
				Models:   []string{"flux-dev"},
				Cheap:    true,
				Reason:   false,
				In: map[Modality]ModalCapability{
					ModalityText: {Inline: true},
				},
				Out: map[Modality]ModalCapability{
					ModalityImage: {URL: true},
				},
				GenSync: &Functionality{
					ReportRateLimits: true,
					Seed:             true,
				},
			},
			{
				Comments: "Multiple models",
				Models:   []string{"flux-tools", "flux-pro-1.0-depth"},
				Reason:   false,
			},
		},
	}

	// First, validate the original score
	if err := s.Validate(); err != nil {
		t.Fatalf("original score validation failed: %v", err)
	}

	// Now simulate what happens during consolidation
	// Separate tested and untested scenarios
	testedScenarios := []Scenario{}
	untestedScenarios := []Scenario{}

	for _, sc := range s.Scenarios {
		if sc.Untested() {
			untestedScenarios = append(untestedScenarios, sc)
		} else {
			testedScenarios = append(testedScenarios, sc)
		}
	}

	t.Logf("Tested scenarios: %d, Untested scenarios: %d", len(testedScenarios), len(untestedScenarios))
	for i, sc := range untestedScenarios {
		t.Logf("  Untested[%d]: Comments=%q, Models=%v, SOTA=%v, Good=%v, Cheap=%v", i, sc.Comments, sc.Models, sc.SOTA, sc.Good, sc.Cheap)
	}

	// Consolidate untested scenarios
	consolidated := ConsolidateUntestedScenarios(untestedScenarios)
	t.Logf("After consolidation: %d scenarios", len(consolidated))
	for i, sc := range consolidated {
		t.Logf("  Consolidated[%d]: Comments=%q, Models=%v", i, sc.Comments, sc.Models)
	}

	// Rebuild the score with consolidated scenarios
	s.Scenarios = testedScenarios
	s.Scenarios = append(s.Scenarios, consolidated...)
	s.SortScenarios()

	t.Logf("After sorting: %d total scenarios", len(s.Scenarios))
	for i, sc := range s.Scenarios {
		t.Logf("  Sorted[%d]: Comments=%q, Models=%v, SOTA=%v, Good=%v, Cheap=%v", i, sc.Comments, sc.Models, sc.SOTA, sc.Good, sc.Cheap)
	}

	// This should not fail
	if err := s.Validate(); err != nil {
		t.Fatalf("validation failed after consolidation and sorting: %v", err)
	}
}

func TestUntestedScenariosWithPreferenceFlagsMustNotMerge(t *testing.T) {
	// Test case that reproduces the actual bug: when consolidating,
	// scenarios with the same Comments but different preference flags (SOTA, Good, Cheap)
	// should NOT be merged together, even if they're untested.
	scenarios := []Scenario{
		{
			Comments: "Untested",
			Models:   []string{"model-a"},
			SOTA:     true,
			Reason:   false,
		},
		{
			Comments: "Untested",
			Models:   []string{"model-b"},
			Good:     true,
			Reason:   false,
		},
		{
			Comments: "Untested",
			Models:   []string{"model-c"},
			Reason:   false,
		},
	}

	result := ConsolidateUntestedScenarios(scenarios)

	// Should have 3 scenarios: model-a (SOTA), model-b (Good), and model-c (no preference)
	if len(result) != 3 {
		t.Fatalf("expected 3 scenarios after consolidation, got %d", len(result))
	}

	// Verify each scenario has only its model
	if len(result[0].Models) != 1 || result[0].Models[0] != "model-a" {
		t.Fatalf("expected scenario 0 to have [model-a], got %v", result[0].Models)
	}
	if len(result[1].Models) != 1 || result[1].Models[0] != "model-b" {
		t.Fatalf("expected scenario 1 to have [model-b], got %v", result[1].Models)
	}
	if len(result[2].Models) != 1 || result[2].Models[0] != "model-c" {
		t.Fatalf("expected scenario 2 to have [model-c], got %v", result[2].Models)
	}

	// Now validate a score with these scenarios
	s := &Score{
		Country:   "US",
		Scenarios: result,
	}

	if err := s.Validate(); err != nil {
		t.Fatalf("validation failed: %v", err)
	}
}

func TestDuplicateModelWithDifferentModalitiesAllowed(t *testing.T) {
	// The bug seems to be that models with the same name but in different output modality groups
	// are being treated as duplicates. They should NOT be - each modality group is independent.
	// For example, a model can be Good for text output and a different model can be Good for image output.
	s := &Score{
		Country: "US",
		Scenarios: []Scenario{
			{
				Comments: "Text SOTA",
				Models:   []string{"model-text-sota"},
				SOTA:     true,
				Reason:   false,
				In: map[Modality]ModalCapability{
					ModalityText: {Inline: true},
				},
				Out: map[Modality]ModalCapability{
					ModalityText: {Inline: true},
				},
				GenSync: &Functionality{},
			},
			{
				Comments: "Image SOTA",
				Models:   []string{"model-image-sota"},
				SOTA:     true,
				Reason:   false,
				In: map[Modality]ModalCapability{
					ModalityImage: {Inline: true},
				},
				Out: map[Modality]ModalCapability{
					ModalityImage: {Inline: true},
				},
				GenSync: &Functionality{},
			},
			{
				Comments: "Text Good",
				Models:   []string{"model-text-good"},
				Good:     true,
				Reason:   false,
				In: map[Modality]ModalCapability{
					ModalityText: {Inline: true},
				},
				Out: map[Modality]ModalCapability{
					ModalityText: {Inline: true},
				},
				GenSync: &Functionality{},
			},
			{
				Comments: "Image Good",
				Models:   []string{"model-image-good"},
				Good:     true,
				Reason:   false,
				In: map[Modality]ModalCapability{
					ModalityImage: {Inline: true},
				},
				Out: map[Modality]ModalCapability{
					ModalityImage: {Inline: true},
				},
				GenSync: &Functionality{},
			},
			{
				Comments: "Text Cheap",
				Models:   []string{"model-text-cheap"},
				Cheap:    true,
				Reason:   false,
				In: map[Modality]ModalCapability{
					ModalityText: {Inline: true},
				},
				Out: map[Modality]ModalCapability{
					ModalityText: {Inline: true},
				},
				GenSync: &Functionality{},
			},
			{
				Comments: "Image Cheap",
				Models:   []string{"model-image-cheap"},
				Cheap:    true,
				Reason:   false,
				In: map[Modality]ModalCapability{
					ModalityImage: {Inline: true},
				},
				Out: map[Modality]ModalCapability{
					ModalityImage: {Inline: true},
				},
				GenSync: &Functionality{},
			},
		},
	}

	if err := s.Validate(); err != nil {
		t.Fatalf("validation failed: %v", err)
	}
}

func TestMergeUntestedScenariosWithSameCommentsAndNoPreferenceFlags(t *testing.T) {
	// When consolidating, scenarios with the same Comments and Reason but WITHOUT preference flags
	// should BE merged (models combined and sorted).
	scenarios := []Scenario{
		{
			Comments: "Expensive to test",
			Models:   []string{"model-c", "model-a"},
			Reason:   false,
		},
		{
			Comments: "Expensive to test",
			Models:   []string{"model-b", "model-d"},
			Reason:   false,
		},
	}

	result := ConsolidateUntestedScenarios(scenarios)

	// Should have 1 scenario with all models merged
	if len(result) != 1 {
		t.Fatalf("expected 1 scenario after consolidation, got %d", len(result))
	}

	// Verify models are merged and sorted
	expected := []string{"model-a", "model-b", "model-c", "model-d"}
	if !slices.Equal(result[0].Models, expected) {
		t.Fatalf("expected models %v, got %v", expected, result[0].Models)
	}
}

func TestDuplicateUntestedScenariosWithPreferenceFlagsDoNotValidate(t *testing.T) {
	// This test reproduces the bug that was happening in smoketest:
	// When duplicate untested scenarios with preference flags are passed to Score.Validate(),
	// it would fail with a "duplicate model" error because the same model appears in two different scenarios.
	//
	// The scenario happened when:
	// 1. The smoketest collected untested scenarios from both new results and old scoreboard
	// 2. Without deduplication, the same scenario (e.g. flux-pro-1.1-ultra) appeared twice
	// 3. Each had sota=true (untested scenario with preference flag)
	// 4. ConsolidateUntestedScenarios didn't merge them (correctly, since they have preference flags)
	// 5. Score.Validate() then failed on seeing flux-pro-1.1-ultra twice
	//
	// The fix is to deduplicate in smoketest before calling ConsolidateUntestedScenarios.
	duplicateScenarios := []Scenario{
		{
			Comments: "Untested",
			Models:   []string{"flux-pro-1.1-ultra"},
			SOTA:     true,
			Reason:   false,
		},
		{
			Comments: "Untested",
			Models:   []string{"flux-pro-1.1-ultra"}, // Same model!
			SOTA:     true,
			Reason:   false,
		},
	}

	// ConsolidateUntestedScenarios should NOT merge these (they have preference flags)
	result := ConsolidateUntestedScenarios(duplicateScenarios)
	if len(result) != 2 {
		t.Fatalf("expected 2 scenarios (not merged due to preference flag), got %d", len(result))
	}

	// But Score.Validate() should fail because flux-pro-1.1-ultra appears twice
	s := &Score{
		Country:   "US",
		Scenarios: result,
	}

	validationErr := s.Validate()
	if validationErr == nil {
		t.Fatal("expected validation to fail on duplicate models")
	}
	if !strings.Contains(validationErr.Error(), "duplicate model") {
		t.Fatalf("expected 'duplicate model' error, got: %v", validationErr)
	}
}

func TestEmbeddingFunctionality(t *testing.T) {
	t.Run("Validate", func(t *testing.T) {
		if err := (&EmbeddingFunctionality{Dimensions: 768, ReportTokenUsage: True}).Validate(); err != nil {
			t.Fatal(err)
		}
		for _, f := range []EmbeddingFunctionality{{Dimensions: -1}, {ReportTokenUsage: TriState(2)}} {
			if err := f.Validate(); err == nil {
				t.Fatalf("accepted %+v", f)
			}
		}
	})
	t.Run("JSON", func(t *testing.T) {
		// Historical scoreboards omit Embed; current markers and measurements round trip.
		for _, raw := range []string{`{"models":["model"]}`, `{"models":["model"],"Embed":{}}`, `{"models":["model"],"in":{"text":{"inline":true}},"out":{"embedding":{"inline":true}},"Embed":{"dimensions":768,"requestedDimensions":false,"reportTokenUsage":"true"}}`} {
			var sc Scenario
			if err := json.Unmarshal([]byte(raw), &sc); err != nil {
				t.Fatal(err)
			}
			if err := sc.Validate(); err != nil {
				t.Fatal(err)
			}
			b, err := json.Marshal(&sc)
			if err != nil {
				t.Fatal(err)
			}
			var got Scenario
			if err := json.Unmarshal(b, &got); err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(sc, got) {
				t.Fatalf("lost state: %s", b)
			}
		}
	})
}
