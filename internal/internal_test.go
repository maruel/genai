// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package internal

import (
	"path/filepath"
	"testing"
)

func TestNewRecords(t *testing.T) {
	r, err := NewRecords("testdata")
	if err != nil {
		t.Fatal(err)
	}
	// Check that files in testdata/ are found
	if _, exists := r.preexisting["test.yaml"]; !exists {
		t.Errorf("Failed to find test.yaml in testdata/")
	}
	// Check that files in subdirectories are found
	if _, exists := r.preexisting[filepath.Join("subdir", "nested.yaml")]; !exists {
		t.Errorf("Failed to find nested.yaml in testdata/subdir/")
	}
}
