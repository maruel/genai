// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package genai

import (
	"context"
	"testing"

	"github.com/invopop/jsonschema"
)

func TestUnsupportedContinuableError(t *testing.T) {
	tests := []struct {
		name        string
		unsupported []string
		want        string
	}{
		{
			name:        "no unsupported options",
			unsupported: []string{},
			want:        "no unsupported options",
		},
		{
			name:        "single unsupported option",
			unsupported: []string{"foo"},
			want:        "unsupported options: foo",
		},
		{
			name:        "multiple unsupported options",
			unsupported: []string{"foo", "bar"},
			want:        "unsupported options: foo, bar",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			u := &UnsupportedContinuableError{Unsupported: tt.unsupported}
			if got := u.Error(); got != tt.want {
				t.Errorf("Error() = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestModalities(t *testing.T) {
	t.Run("String", func(t *testing.T) {
		tests := []struct {
			name string
			in   Modalities
			want string
		}{
			{
				name: "empty",
				in:   Modalities{},
				want: "",
			},
			{
				name: "single modality",
				in:   Modalities{ModalityText},
				want: "text",
			},
			{
				name: "multiple modalities",
				in:   Modalities{ModalityText, ModalityImage, ModalityVideo},
				want: "text,image,video",
			},
		}
		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				if got := tt.in.String(); got != tt.want {
					t.Errorf("String() = %q, want %q", got, tt.want)
				}
			})
		}
	})
}

func TestOptionsText(t *testing.T) {
	t.Run("Validate", func(t *testing.T) {
		t.Run("valid", func(t *testing.T) {
			tests := []struct {
				name string
				in   OptionsText
			}{
				{
					name: "Valid options with all fields set",
					in: OptionsText{
						Seed:        1,
						Temperature: 0.5,
						TopP:        0.5,
						TopK:        10,
						MaxTokens:   100,
						Stop:        []string{"stop"},
						ReplyAsJSON: true,
						DecodeAs:    struct{}{},
						Tools: []ToolDef{
							{
								Name:        "tool",
								Description: "do stuff",
							},
						},
					},
				},
				{
					name: "Valid options with only DecodeAs pointer",
					in:   OptionsText{DecodeAs: &struct{}{}},
				},
			}
			for _, tt := range tests {
				t.Run(tt.name, func(t *testing.T) {
					if err := tt.in.Validate(); err != nil {
						t.Fatalf("unexpected error: %q", err)
					}
				})
			}
		})
		t.Run("error", func(t *testing.T) {
			tests := []struct {
				name   string
				in     OptionsText
				errMsg string
			}{
				{
					name:   "Invalid Seed",
					in:     OptionsText{Seed: -1},
					errMsg: "field Seed: must be non-negative",
				},
				{
					name:   "Invalid Temperature",
					in:     OptionsText{Temperature: -1},
					errMsg: "field Temperature: must be [0, 100]",
				},
				{
					name:   "Invalid MaxTokens",
					in:     OptionsText{MaxTokens: 1024*1024*1024 + 1},
					errMsg: "field MaxTokens: must be [0, 1 GiB]",
				},
				{
					name:   "Invalid TopP",
					in:     OptionsText{TopP: -1},
					errMsg: "field TopP: must be [0, 1]",
				},
				{
					name:   "Invalid TopK",
					in:     OptionsText{TopK: 1025},
					errMsg: "field TopK: must be [0, 1024]",
				},
				{
					name:   "Invalid DecodeAs jsonschema.Schema",
					in:     OptionsText{DecodeAs: &jsonschema.Schema{}},
					errMsg: "field DecodeAs: must be an actual struct serializable as JSON, not a *jsonschema.Schema",
				},
				{
					name:   "Invalid DecodeAs string",
					in:     OptionsText{DecodeAs: "string"},
					errMsg: "field DecodeAs: must be a struct, not string",
				},
				{
					name: "ToolCallRequired without Tools",
					in: OptionsText{
						ToolCallRequest: ToolCallRequired,
					},
					errMsg: "field ToolCallRequest is ToolCallRequired: Tools are required",
				},
				{
					name: "Duplicate tool names",
					in: OptionsText{
						Tools: []ToolDef{
							{Name: "tool1", Description: "desc1"},
							{Name: "tool1", Description: "desc2"},
						},
					},
					errMsg: "tool 1: has name \"tool1\" which is the same as tool 0",
				},
				{
					name: "Tool validation error",
					in: OptionsText{
						Tools: []ToolDef{
							{Name: "tool1"},
						},
					},
					errMsg: "tool 0: field Description: required",
				},
			}
			for _, tt := range tests {
				t.Run(tt.name, func(t *testing.T) {
					if err := tt.in.Validate(); err == nil || err.Error() != tt.errMsg {
						t.Fatalf("error mismatch\nwant %q\ngot  %q", tt.errMsg, err)
					}
				})
			}
		})
	})
}

func TestToolDef(t *testing.T) {
	type inputStruct struct {
		Name string
	}
	t.Run("Validate", func(t *testing.T) {
		t.Run("valid", func(t *testing.T) {
			tests := []struct {
				name string
				in   ToolDef
			}{
				{
					name: "Valid ToolDef with function and pointer InputsAs",
					in: ToolDef{
						Name:        "tool",
						Description: "do stuff",
						Callback:    func(ctx context.Context, input *inputStruct) (string, error) { return "", nil },
					},
				},
				{
					name: "Valid ToolDef with InputSchemaOverride",
					in: ToolDef{
						Name:                "tool",
						Description:         "do stuff",
						InputSchemaOverride: &jsonschema.Schema{},
					},
				},
			}

			for _, tt := range tests {
				t.Run(tt.name, func(t *testing.T) {
					if err := tt.in.Validate(); err != nil {
						t.Fatalf("unexpected error: %q", err)
					}
				})
			}
		})
		t.Run("error", func(t *testing.T) {
			tests := []struct {
				name   string
				in     ToolDef
				errMsg string
			}{
				{
					name:   "Missing Name",
					in:     ToolDef{Description: "do stuff"},
					errMsg: "field Name: required",
				},
				{
					name:   "Missing Description",
					in:     ToolDef{Name: "tool"},
					errMsg: "field Description: required",
				},
				{
					name: "Callback not a function",
					in: ToolDef{
						Name:        "tool",
						Description: "do stuff",
						Callback:    "not a function",
					},
					errMsg: "field Callback: must be a function",
				},
				{
					name: "Callback returns wrong type first",
					in: ToolDef{
						Name:        "tool",
						Description: "do stuff",
						Callback:    func(ctx context.Context, b *inputStruct) (int, error) { return 1, nil },
					},
					errMsg: "field Callback: must return a string first, not \"int\"",
				},
				{
					name: "Callback returns wrong type second",
					in: ToolDef{
						Name:        "tool",
						Description: "do stuff",
						Callback:    func(ctx context.Context, b *inputStruct) (string, string) { return "", "" },
					},
					errMsg: "field Callback: must return an error second, not \"string\"",
				},
				{
					name: "Callback with wrong parameter count",
					in: ToolDef{
						Name:        "tool",
						Description: "do stuff",
						Callback:    func(a, b, c *inputStruct) string { return "" },
					},
					errMsg: "field Callback: must accept exactly two parameters: (context.Context, input *struct{})",
				},
				{
					name: "Callback first parameter not context.Context",
					in: ToolDef{
						Name:        "tool",
						Description: "do stuff",
						Callback:    func(a string, b *inputStruct) (string, error) { return "", nil },
					},
					errMsg: "field Callback: must accept exactly two parameters, first that is a context.Context, not a \"string\"",
				},
				{
					name: "Callback second parameter not pointer",
					in: ToolDef{
						Name:        "tool",
						Description: "do stuff",
						Callback:    func(ctx context.Context, input inputStruct) (string, error) { return "", nil },
					},
					errMsg: "field Callback: must accept exactly two parameters, second that is a pointer to a struct, not a \"inputStruct\"",
				},
				{
					name: "Callback second parameter not struct",
					in: ToolDef{
						Name:        "tool",
						Description: "do stuff",
						Callback:    func(ctx context.Context, input *string) (string, error) { return "", nil },
					},
					errMsg: "field Callback: must accept exactly two parameters, second that is a pointer to a struct, not a \"string\"",
				},
			}
			for _, tt := range tests {
				t.Run(tt.name, func(t *testing.T) {
					if err := tt.in.Validate(); err == nil || err.Error() != tt.errMsg {
						t.Fatalf("error mismatch\nwant %q\ngot  %q", tt.errMsg, err)
					}
				})
			}
		})
	})

	t.Run("GetInputSchema", func(t *testing.T) {
		type testInput struct {
			Value string `json:"value"`
		}
		tool := ToolDef{
			Name:        "testTool",
			Description: "A test tool",
			Callback:    func(ctx context.Context, input *testInput) (string, error) { return "", nil },
		}
		schema := tool.GetInputSchema()
		if schema == nil {
			t.Fatal("GetInputSchema returned nil")
		}
		if _, ok := schema.Properties.Get("value"); !ok {
			t.Errorf("Expected schema to have 'value' property, got %+v", schema.Properties)
		}
	})
}

func TestOptionsAudio(t *testing.T) {
	t.Run("Validate", func(t *testing.T) {
		o := &OptionsAudio{}
		if err := o.Validate(); err != nil {
			t.Errorf("Validate() got unexpected error: %v", err)
		}
	})
}

func TestOptionsImage(t *testing.T) {
	t.Run("Validate", func(t *testing.T) {
		t.Run("valid", func(t *testing.T) {
			o := &OptionsImage{Seed: 1, Width: 100, Height: 200}
			if err := o.Validate(); err != nil {
				t.Errorf("Validate() got unexpected error: %v", err)
			}
		})
		t.Run("error", func(t *testing.T) {
			tests := []struct {
				name   string
				in     OptionsImage
				errMsg string
			}{
				{
					name:   "Invalid Seed",
					in:     OptionsImage{Seed: -1},
					errMsg: "field Seed: must be non-negative",
				},
				{
					name:   "Invalid Height",
					in:     OptionsImage{Height: -1},
					errMsg: "field Height: must be non-negative",
				},
				{
					name:   "Invalid Width",
					in:     OptionsImage{Width: -1},
					errMsg: "field Width: must be non-negative",
				},
			}
			for _, tt := range tests {
				t.Run(tt.name, func(t *testing.T) {
					if err := tt.in.Validate(); err == nil || err.Error() != tt.errMsg {
						t.Fatalf("error mismatch\nwant %q\ngot  %q", tt.errMsg, err)
					}
				})
			}
		})
	})
}

func TestOptionsVideo(t *testing.T) {
	t.Run("Validate", func(t *testing.T) {
		o := &OptionsVideo{}
		if err := o.Validate(); err != nil {
			t.Errorf("Validate() got unexpected error: %v", err)
		}
	})
}

func TestValidateReflectedToJSON(t *testing.T) {
	type testStruct struct{}
	t.Run("valid", func(t *testing.T) {
		tests := []struct {
			name string
			in   any
		}{
			{
				name: "valid struct",
				in:   testStruct{},
			},
			{
				name: "valid pointer to struct",
				in:   &testStruct{},
			},
		}
		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				if err := validateReflectedToJSON(tt.in); err != nil {
					t.Fatalf("unexpected error: %v", err)
				}
			})
		}
	})
	t.Run("error", func(t *testing.T) {
		tests := []struct {
			name   string
			in     any
			errMsg string
		}{
			{
				name:   "jsonschema.Schema pointer",
				in:     &jsonschema.Schema{},
				errMsg: "must be an actual struct serializable as JSON, not a *jsonschema.Schema",
			},
			{
				name:   "string type",
				in:     "hello",
				errMsg: "must be a struct, not string",
			},
			{
				name:   "int type",
				in:     123,
				errMsg: "must be a struct, not int",
			},
		}
		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				if err := validateReflectedToJSON(tt.in); err == nil || err.Error() != tt.errMsg {
					t.Fatalf("error mismatch\nwant %q\ngot  %q", tt.errMsg, err)
				}
			})
		}
	})
}
