// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Tests for recording fixture tracking.

package internaltest

import (
	"os"
	"path/filepath"
	"testing"
)

func TestTakeRecordings(t *testing.T) {
	t.Chdir(t.TempDir())
	path := filepath.Join("testdata", "TestTakeRecordings", "probe.ndjson")
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte("{}\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	NewSubprocessRecorder(t, "TestTakeRecordings/probe", "true", nil)
	abs, err := filepath.Abs(path)
	if err != nil {
		t.Fatal(err)
	}
	usage := TakeRecordings(t.Name())
	if _, ok := usage.Used[abs]; !ok {
		t.Fatalf("fixture not recorded: %v", usage.Used)
	}
	if usage := TakeRecordings(t.Name()); len(usage.Used) != 0 || len(usage.Missing) != 0 {
		t.Fatalf("fixture not cleared: %+v", usage)
	}
	NewSubprocessRecorder(t, "TestTakeRecordings/missing", "true", nil)
	missingPath, err := filepath.Abs(filepath.Join("testdata", "TestTakeRecordings", "missing.ndjson"))
	if err != nil {
		t.Fatal(err)
	}
	if usage := TakeRecordings(t.Name()); len(usage.Missing) != 1 {
		t.Fatalf("missing fixture not recorded: %+v", usage)
	} else if _, ok := usage.Missing[missingPath]; !ok {
		t.Fatalf("missing fixture path: %+v", usage)
	}
	NewSubprocessRecorder(t, "TestTakeRecordings/failed", "true", nil)
	failedPath, err := filepath.Abs(filepath.Join("testdata", "TestTakeRecordings", "failed.ndjson"))
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(failedRecordingPath(failedPath), []byte("{}\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	if usage := TakeRecordings(t.Name()); len(usage.Used) != 1 {
		t.Fatalf("failed twin not used: %+v", usage)
	} else if _, ok := usage.Used[failedPath]; !ok {
		t.Fatalf("failed twin path: %+v", usage)
	}
}
