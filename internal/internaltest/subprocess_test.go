// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Tests for subprocess recording helpers.

package internaltest

import (
	"os"
	"path/filepath"
	"testing"
)

func TestTakeSubprocessRecordings(t *testing.T) {
	t.Chdir(t.TempDir())
	path := filepath.Join("testdata", "TestTakeSubprocessRecordings", "probe.ndjson")
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte("{}\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	NewSubprocessRecorder(t, "TestTakeSubprocessRecordings/probe", "true", nil)
	abs, err := filepath.Abs(path)
	if err != nil {
		t.Fatal(err)
	}
	used := TakeSubprocessRecordings(t.Name())
	if _, ok := used[abs]; !ok {
		t.Fatalf("fixture not recorded: %v", used)
	}
	if used := TakeSubprocessRecordings(t.Name()); len(used) != 0 {
		t.Fatalf("fixture not cleared: %v", used)
	}
}
