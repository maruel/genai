// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package internal

import "testing"

type testStruct struct{}

func TestTypeName(t *testing.T) {
	tests := []struct {
		name string
		in   any
		want string
	}{
		{"value type", testStruct{}, "testStruct"},
		{"pointer type", &testStruct{}, "testStruct"},
		{"double pointer", func() any { v := &testStruct{}; return &v }(), "testStruct"},
		{"int", 42, "int"},
		{"pointer to int", func() any { v := 42; return &v }(), "int"},
		{"string", "hello", "string"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := TypeName(tt.in); got != tt.want {
				t.Errorf("TypeName() = %q, want %q", got, tt.want)
			}
		})
	}
}
