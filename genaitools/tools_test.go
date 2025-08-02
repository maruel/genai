// Copyright 2024 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package genaitools

import (
	"context"
	"encoding/json"
	"regexp"
	"strings"
	"testing"
	"time"
)

func TestArithmetic(t *testing.T) {
	// We can't use table-driven tests directly with the callback since it requires
	// a specific struct type, so we'll test each case individually
	tests := []struct {
		name      string
		operation string
		first     json.Number
		second    json.Number
		expected  string
		expectErr bool
		errSubstr string
	}{
		// Integer operations
		{"addition_int", "addition", "5", "3", "8", false, ""},
		{"subtraction_int", "subtraction", "10", "4", "6", false, ""},
		{"multiplication_int", "multiplication", "6", "7", "42", false, ""},
		{"division_int_exact", "division", "10", "2", "5", false, ""},
		{"division_int_to_float", "division", "10", "3", "3.333333", false, ""},
		{"large_int", "addition", "922337203685477580", "1", "922337203685477581", false, ""},

		// Float operations
		{"addition_float", "addition", "3.5", "2.1", "5.600000", false, ""},
		{"subtraction_float", "subtraction", "7.5", "2.5", "5.000000", false, ""},
		{"multiplication_float", "multiplication", "2.5", "4.0", "10", false, ""},
		{"division_float", "division", "10.5", "2.1", "5.000000", false, ""},

		// Mixed integer and float
		{"mixed_types", "addition", "5", "3.5", "8.500000", false, ""},

		// Error cases
		{"invalid_operation", "unknown", "5", "3", "", true, "unknown operation"},
		{"invalid_first_number", "addition", "not_a_number", "3", "", true, "couldn't understand the first number"},
		{"invalid_second_number", "addition", "5", "not_a_number", "", true, "couldn't understand the second number"},
	}

	for _, tt := range tests {
		testName := tt.name
		expected := strings.TrimRight(tt.expected, "0")
		expected = strings.TrimRight(expected, ".")

		// Run each test in a subtest
		t.Run(testName, func(t *testing.T) {
			ctx := t.Context()
			// Create the args for the callback
			args := &calculateArgs{
				Operation:    tt.operation,
				FirstNumber:  tt.first,
				SecondNumber: tt.second,
			}

			// Call the callback with the args
			callback := Arithmetic.Callback.(func(context.Context, *calculateArgs) (string, error))
			result, err := callback(ctx, args)

			// Check error expectation
			if tt.expectErr {
				if err == nil {
					t.Fatalf("Expected error containing %q but got nil", tt.errSubstr)
				} else if !strings.Contains(err.Error(), tt.errSubstr) {
					t.Fatalf("Expected error containing %q but got %q", tt.errSubstr, err.Error())
				}
				return
			}

			// Check for unexpected error
			if err != nil {
				t.Fatalf("Unexpected error: %v", err)
			}

			// For floating point results, format for comparison
			result = strings.TrimRight(result, "0")
			result = strings.TrimRight(result, ".")

			// For division cases with floating point, check approximate equality
			if tt.operation == "division" && strings.Contains(result, ".") {
				// For floating point division, just check the prefix matches
				if !strings.HasPrefix(result, expected) {
					t.Fatalf("Expected result starting with %q but got %q", expected, result)
				}
			} else if result != expected {
				t.Fatalf("Expected %q but got %q", expected, result)
			}
		})
	}
}

func TestGetTodayClockTime(t *testing.T) {
	ctx := t.Context()
	before := time.Now()

	// Call the callback directly with an empty struct
	callback := GetTodayClockTime.Callback.(func(context.Context, *empty) (string, error))
	result, err := callback(ctx, &empty{})
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	// Verify the format follows "Monday 2006-01-02 15:04:05"
	expectedPattern := `^(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday) [0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}$`
	matched, err := regexp.MatchString(expectedPattern, result)
	if err != nil {
		t.Fatalf("Regex error: %v", err)
	}
	if !matched {
		t.Fatalf("Time format doesn't match expected pattern. Got: %q", result)
	}

	// Verify the time is within a reasonable range (last minute)
	parsedTime, err := time.ParseInLocation("Monday 2006-01-02 15:04:05", result, time.Local)
	if err != nil {
		t.Fatalf("Failed to parse time %q: %v", result, err)
	}
	if diff := parsedTime.Sub(before); diff > time.Minute {
		t.Fatalf("Time is too old: %v - %v = %s", parsedTime, before, diff)
	}
	if parsedTime.After(before.Add(time.Minute)) {
		t.Fatalf("Time is in the future: %v", parsedTime)
	}
}
