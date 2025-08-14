// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package genai

import (
	"context"
	"testing"

	"github.com/invopop/jsonschema"
)

func TestOptionsText(t *testing.T) {
	t.Run("Validate", func(t *testing.T) {
		t.Run("valid", func(t *testing.T) {
			tests := []struct {
				name    string
				options OptionsText
			}{
				{
					name: "Valid options with all fields set",
					options: OptionsText{
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
					name:    "Valid options with only DecodeAs pointer",
					options: OptionsText{DecodeAs: &struct{}{}},
				},
			}
			for _, tt := range tests {
				t.Run(tt.name, func(t *testing.T) {
					if err := tt.options.Validate(); err != nil {
						t.Fatalf("unexpected error: %q", err)
					}
				})
			}
		})
		t.Run("error", func(t *testing.T) {
			tests := []struct {
				name    string
				options OptionsText
				errMsg  string
			}{
				{
					name:    "Invalid Seed",
					options: OptionsText{Seed: -1},
					errMsg:  "field Seed: must be non-negative",
				},
				{
					name:    "Invalid Temperature",
					options: OptionsText{Temperature: -1},
					errMsg:  "field Temperature: must be [0, 100]",
				},
				{
					name:    "Invalid MaxTokens",
					options: OptionsText{MaxTokens: 1024*1024*1024 + 1},
					errMsg:  "field MaxTokens: must be [0, 1 GiB]",
				},
				{
					name:    "Invalid TopP",
					options: OptionsText{TopP: -1},
					errMsg:  "field TopP: must be [0, 1]",
				},
				{
					name:    "Invalid TopK",
					options: OptionsText{TopK: 1025},
					errMsg:  "field TopK: must be [0, 1024]",
				},
				{
					name:    "Invalid DecodeAs jsonschema.Schema",
					options: OptionsText{DecodeAs: &jsonschema.Schema{}},
					errMsg:  "field DecodeAs: must be an actual struct serializable as JSON, not a *jsonschema.Schema",
				},
				{
					name:    "Invalid DecodeAs string",
					options: OptionsText{DecodeAs: "string"},
					errMsg:  "field DecodeAs: must be a struct, not string",
				},
			}
			for _, tt := range tests {
				t.Run(tt.name, func(t *testing.T) {
					if err := tt.options.Validate(); err == nil || err.Error() != tt.errMsg {
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
				name    string
				toolDef ToolDef
			}{
				{
					name: "Valid ToolDef with function and pointer InputsAs",
					toolDef: ToolDef{
						Name:        "tool",
						Description: "do stuff",
						Callback:    func(ctx context.Context, input *inputStruct) (string, error) { return "", nil },
					},
				},
			}

			for _, tt := range tests {
				t.Run(tt.name, func(t *testing.T) {
					if err := tt.toolDef.Validate(); err != nil {
						t.Fatalf("unexpected error: %q", err)
					}
				})
			}
		})
		t.Run("error", func(t *testing.T) {
			tests := []struct {
				name    string
				toolDef ToolDef
				errMsg  string
			}{
				{
					name:    "Missing Name",
					toolDef: ToolDef{Description: "do stuff"},
					errMsg:  "field Name: required",
				},
				{
					name:    "Missing Description",
					toolDef: ToolDef{Name: "tool"},
					errMsg:  "field Description: required",
				},
				{
					name: "Callback not a function",
					toolDef: ToolDef{
						Name:        "tool",
						Description: "do stuff",
						Callback:    "not a function",
					},
					errMsg: "field Callback: must be a function",
				},
				{
					name: "Callback returns wrong type first",
					toolDef: ToolDef{
						Name:        "tool",
						Description: "do stuff",
						Callback:    func(ctx context.Context, b *inputStruct) (int, error) { return 1, nil },
					},
					errMsg: "field Callback: must return a string first, not \"int\"",
				},
				{
					name: "Callback returns wrong type second",
					toolDef: ToolDef{
						Name:        "tool",
						Description: "do stuff",
						Callback:    func(ctx context.Context, b *inputStruct) (string, string) { return "", "" },
					},
					errMsg: "field Callback: must return an error second, not \"string\"",
				},
				{
					name: "Callback with wrong parameter count",
					toolDef: ToolDef{
						Name:        "tool",
						Description: "do stuff",
						Callback:    func(a, b, c *inputStruct) string { return "" },
					},
					errMsg: "field Callback: must accept exactly two parameters: (context.Context, input *struct{})",
				},
				{
					name: "parameter not pointer",
					toolDef: ToolDef{
						Name:        "tool",
						Description: "do stuff",
						Callback:    func(ctx context.Context, input inputStruct) string { return "" },
					},
					errMsg: "field Callback: must accept exactly two parameters, second that is a pointer to a struct, not a \"inputStruct\"",
				},
			}
			for _, tt := range tests {
				t.Run(tt.name, func(t *testing.T) {
					if err := tt.toolDef.Validate(); err == nil || err.Error() != tt.errMsg {
						t.Fatalf("error mismatch\nwant %q\ngot  %q", tt.errMsg, err)
					}
				})
			}
		})
	})
}
