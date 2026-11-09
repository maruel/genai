// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Recording fixture tracking for tests.

package internaltest

import (
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
)

var recordings = struct {
	sync.Mutex

	used map[string]map[string]struct{}
}{used: map[string]map[string]struct{}{}}

// RecordingUsage reports which fixtures tests attempted to use.
type RecordingUsage struct {
	// Used contains fixtures present on disk, including failed twins.
	Used map[string]struct{}
	// Missing contains fixtures absent from disk.
	Missing map[string]struct{}
}

// TakeRecordings returns the fixture paths used by tests named prefix or one
// of its subtests, then clears those entries. It classifies each path as Used
// or Missing at call time; a Realtime-N.failed.ndjson twin counts as Used.
func TakeRecordings(prefix string) RecordingUsage {
	recordings.Lock()
	defer recordings.Unlock()
	out := RecordingUsage{Used: map[string]struct{}{}, Missing: map[string]struct{}{}}
	for name, paths := range recordings.used {
		if name == prefix || strings.HasPrefix(name, prefix+"/") {
			for path := range paths {
				if _, err := os.Stat(path); err == nil {
					out.Used[path] = struct{}{}
					continue
				}
				if _, err := os.Stat(failedRecordingPath(path)); err == nil {
					out.Used[path] = struct{}{}
					continue
				}
				out.Missing[path] = struct{}{}
			}
			delete(recordings.used, name)
		}
	}
	return out
}

func failedRecordingPath(path string) string {
	return strings.TrimSuffix(path, ".ndjson") + ".failed.ndjson"
}

func markRecording(t testing.TB, fixture string) {
	abs, err := filepath.Abs(fixture)
	if err != nil {
		t.Fatalf("resolve fixture path: %v", err)
	}
	name := t.Name()
	recordings.Lock()
	defer recordings.Unlock()
	if recordings.used[name] == nil {
		recordings.used[name] = map[string]struct{}{}
	}
	recordings.used[name][abs] = struct{}{}
}
