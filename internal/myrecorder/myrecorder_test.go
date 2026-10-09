// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Tests for the myrecorder package.

package myrecorder

import (
	"io"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"sync/atomic"
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

func TestTrimPolls(t *testing.T) {
	t.Setenv("RECORD", "")
	var polls atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		state := "pending"
		if polls.Add(1) == 6 {
			state = "done"
		}
		_, _ = io.WriteString(w, state)
	}))
	t.Cleanup(srv.Close)
	root := t.TempDir()
	// poll returns the number of requests sent until the server reports done.
	poll := func(t *testing.T) int {
		r, err := NewRecords(root)
		if err != nil {
			t.Fatal(err)
		}
		rec, err := r.Record("poll", http.DefaultTransport, TrimPolls())
		if err != nil {
			t.Fatal(err)
		}
		c := http.Client{Transport: rec}
		n := 0
		for state := ""; state != "done"; {
			n++
			req, err := http.NewRequestWithContext(t.Context(), http.MethodGet, srv.URL, http.NoBody)
			if err != nil {
				t.Fatal(err)
			}
			resp, err := c.Do(req)
			if err != nil {
				t.Fatal(err)
			}
			b, err := io.ReadAll(resp.Body)
			if err2 := resp.Body.Close(); err == nil {
				err = err2
			}
			if err != nil {
				t.Fatal(err)
			}
			state = string(b)
		}
		if err := rec.Stop(); err != nil {
			t.Fatal(err)
		}
		return n
	}
	if n := poll(t); n != 6 {
		t.Fatalf("recorded %d polls, want 6", n)
	}
	if n := poll(t); n != 3 {
		t.Fatalf("replayed %d polls, want 3: two pending and the final state", n)
	}
	if n := polls.Load(); n != 6 {
		t.Fatalf("server saw %d polls, want 6: replay must not reach it", n)
	}
}
