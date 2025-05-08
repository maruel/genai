// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package internaltest

import (
	"testing"
)

func TestNewRecords(t *testing.T) {
	r := NewRecords()

	// Check that files in testdata/ are found
	if _, exists := r.preexisting["test.yaml"]; !exists {
		t.Errorf("Failed to find test.yaml in testdata/")
	}

	// Check that files in subdirectories are found
	if _, exists := r.preexisting["subdir/nested.yaml"]; !exists {
		t.Errorf("Failed to find nested.yaml in testdata/subdir/")
	}
}
