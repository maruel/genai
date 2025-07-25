// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package gemini

import (
	"sort"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
)

func TestSchema_FromGoType(t *testing.T) {
	data := []struct {
		name string
		in   any
		want Schema
	}{
		{
			name: "empty struct",
			in:   &struct{}{},
			want: Schema{
				Type:       "OBJECT",
				Properties: map[string]Schema{},
			},
		},
		{
			name: "struct with simple string",
			in: &struct {
				A string
			}{},
			want: Schema{
				Type:       "OBJECT",
				Properties: map[string]Schema{"A": {Type: "STRING"}},
				Required:   []string{"A"},
			},
		},
		{
			name: "struct with pointer field",
			in: &struct {
				A *string
			}{},
			want: Schema{
				Type: "OBJECT",
				Properties: map[string]Schema{
					"A": {
						Type:     "STRING",
						Nullable: true,
					},
				},
			},
		},
		{
			name: "struct with enum",
			in: &struct {
				E string `jsonschema:"enum=A,enum=B"`
			}{},
			want: Schema{
				Type: "OBJECT",
				Properties: map[string]Schema{
					"E": {
						Type: "STRING",
						Enum: []string{"A", "B"},
					},
				},
				Required: []string{"E"},
			},
		},
		{
			name: "struct with array",
			in: &struct {
				A []int
			}{},
			want: Schema{
				Type: "OBJECT",
				Properties: map[string]Schema{
					"A": {
						Type:  "ARRAY",
						Items: &Schema{Type: "INTEGER"},
					},
				},
				Required: []string{"A"},
			},
		},
		{
			name: "struct with ordered fields",
			in: &struct {
				S string `json:"s"`
				B bool   `json:"b"`
				// Use a pointer to make it nullable.
				Opt *string `json:"opt,omitempty"`
			}{},
			want: Schema{
				Type: "OBJECT",
				Properties: map[string]Schema{
					"s": {Type: "STRING"},
					"b": {Type: "BOOLEAN"},
					"opt": {
						Type:     "STRING",
						Nullable: true,
					},
				},
				PropertyOrdering: []string{"s", "b", "opt"},
				Required:         []string{"b", "s"},
			},
		},
		{
			name: "struct with nested struct",
			in: &struct {
				O struct {
					P string `json:"p"`
				} `json:"o"`
			}{},
			want: Schema{
				Type: "OBJECT",
				Properties: map[string]Schema{
					"o": {
						Type: "OBJECT",
						Properties: map[string]Schema{
							"p": {Type: "STRING"},
						},
						Required: []string{"p"},
					},
				},
				Required: []string{"o"},
			},
		},
		{
			name: "struct with field description",
			in: &struct {
				A string `jsonschema_description:"This is a test"`
			}{},
			want: Schema{
				Type: "OBJECT",
				Properties: map[string]Schema{
					"A": {
						Type:        "STRING",
						Description: "This is a test",
					},
				},
				Required: []string{"A"},
			},
		},
		{
			name: "struct with int types",
			in: &struct {
				I   int    `json:"i"`
				I32 int32  `json:"i32"`
				I64 int64  `json:"i64"`
				U   uint   `json:"u"`
				U32 uint32 `json:"u32"`
				U64 uint64 `json:"u64"`
			}{},
			want: Schema{
				Type: "OBJECT",
				Properties: map[string]Schema{
					"i":   {Type: "INTEGER"},
					"i32": {Type: "INTEGER", Format: "int32"},
					"i64": {Type: "INTEGER", Format: "int64"},
					"u":   {Type: "INTEGER"},
					"u32": {Type: "INTEGER"},
					"u64": {Type: "INTEGER"},
				},
				Required:         []string{"i", "i32", "i64", "u", "u32", "u64"},
				PropertyOrdering: []string{"i", "i32", "i64", "u", "u32", "u64"},
			},
		},
		{
			name: "struct with float types",
			in: &struct {
				F32 float32 `json:"f32"`
				F64 float64 `json:"f64"`
			}{},
			want: Schema{
				Type: "OBJECT",
				Properties: map[string]Schema{
					"f32": {Type: "NUMBER", Format: "float"},
					"f64": {Type: "NUMBER", Format: "double"},
				},
				Required:         []string{"f32", "f64"},
				PropertyOrdering: []string{"f32", "f64"},
			},
		},
		{
			name: "struct with omitempty",
			in: &struct {
				A string  `json:"a,omitempty"`
				B *string `json:"b,omitempty"`
			}{},
			want: Schema{
				Type: "OBJECT",
				Properties: map[string]Schema{
					"a": {Type: "STRING"},
					"b": {Type: "STRING", Nullable: true},
				},
				PropertyOrdering: []string{"a", "b"},
			},
		},
		{
			name: "struct with omitzero",
			in: &struct {
				A int `json:"a,omitzero"`
			}{},
			want: Schema{
				Type: "OBJECT",
				Properties: map[string]Schema{
					"a": {Type: "INTEGER"},
				},
			},
		},
		{
			name: "struct with time.Time",
			in: &struct {
				T time.Time `json:"t"`
			}{},
			want: Schema{
				Type: "OBJECT",
				Properties: map[string]Schema{
					"t": {Type: "STRING", Format: "date-time"},
				},
				Required: []string{"t"},
			},
		},
		{
			name: "struct with ignored field",
			in: &struct {
				A string `json:"-"`
				B string
			}{},
			want: Schema{
				Type: "OBJECT",
				Properties: map[string]Schema{
					"B": {Type: "STRING"},
				},
				Required: []string{"B"},
				// TODO: We could fix that in the future but it's just a nice to have to skip this.
				PropertyOrdering: []string{"B"},
			},
		},
		{
			name: "struct with default int value",
			in: &struct {
				A int `jsonschema:"default=123"`
			}{},
			want: Schema{
				Type: "OBJECT",
				Properties: map[string]Schema{
					"A": {
						Type:    "INTEGER",
						Default: int64(123),
					},
				},
				Required: []string{"A"},
			},
		},
		{
			name: "struct with default float value",
			in: &struct {
				A float64 `jsonschema:"default=123.45"`
			}{},
			want: Schema{
				Type: "OBJECT",
				Properties: map[string]Schema{
					"A": {
						Type:    "NUMBER",
						Format:  "double",
						Default: 123.45,
					},
				},
				Required: []string{"A"},
			},
		},
		{
			name: "struct with default bool value",
			in: &struct {
				A bool `jsonschema:"default=true"`
			}{},
			want: Schema{
				Type: "OBJECT",
				Properties: map[string]Schema{
					"A": {
						Type:    "BOOLEAN",
						Default: true,
					},
				},
				Required: []string{"A"},
			},
		},
		{
			name: "struct with example int value",
			in: &struct {
				A int `jsonschema:"example=456"`
			}{},
			want: Schema{
				Type: "OBJECT",
				Properties: map[string]Schema{
					"A": {
						Type:    "INTEGER",
						Example: int64(456),
					},
				},
				Required: []string{"A"},
			},
		},
		{
			name: "struct with example float value",
			in: &struct {
				A float64 `jsonschema:"example=67.89"`
			}{},
			want: Schema{
				Type: "OBJECT",
				Properties: map[string]Schema{
					"A": {
						Type:    "NUMBER",
						Format:  "double",
						Example: 67.89,
					},
				},
				Required: []string{"A"},
			},
		},
		{
			name: "struct with example bool value",
			in: &struct {
				A bool `jsonschema:"example=false"`
			}{},
			want: Schema{
				Type: "OBJECT",
				Properties: map[string]Schema{
					"A": {
						Type:    "BOOLEAN",
						Example: false,
					},
				},
				Required: []string{"A"},
			},
		},
	}
	for _, line := range data {
		t.Run(line.name, func(t *testing.T) {
			s := Schema{}
			if err := s.FromGoObj(line.in); err != nil {
				t.Fatal(err)
			}
			sort.Strings(s.Required)
			if diff := cmp.Diff(line.want, s); diff != "" {
				t.Errorf("Schema mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestSchema_FromGoType_errors(t *testing.T) {
	data := []struct {
		name string
		in   any
		want string
	}{
		{
			name: "unsupported type",
			in:   map[string]any{},
			want: "unsupported type: map",
		},
	}
	for _, line := range data {
		t.Run(line.name, func(t *testing.T) {
			s := Schema{}
			if err := s.FromGoObj(line.in); err == nil {
				t.Fatal("expected error")
			} else if got := err.Error(); got != line.want {
				t.Fatalf("got error %q, want %q", got, line.want)
			}
		})
	}
}
