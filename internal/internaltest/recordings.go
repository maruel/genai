// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Recording fixture tracking for tests.

package internaltest

import (
	"path/filepath"
	"strings"
	"sync"
	"testing"
)

var recordings = struct {
	sync.Mutex
	used map[string]map[string]struct{}
}{used: map[string]map[string]struct{}{}}

// TakeRecordings returns the fixture paths used by tests named prefix or one
// of its subtests, then clears those entries.
func TakeRecordings(prefix string) map[string]struct{} {
	recordings.Lock()
	defer recordings.Unlock()
	out := map[string]struct{}{}
	for name, paths := range recordings.used {
		if name == prefix || strings.HasPrefix(name, prefix+"/") {
			for path := range paths {
				out[path] = struct{}{}
			}
			delete(recordings.used, name)
		}
	}
	return out
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
