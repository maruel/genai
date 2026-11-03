// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Tests for subprocess recording helpers.

package internaltest

import (
	"slices"
	"testing"
)

func TestSanitizeSubprocessLine(t *testing.T) {
	in := []byte(`{"error":"API rate limit exceeded for 69.156.8.173 from user@example.com"}`)
	want := []byte(`{"error":"API rate limit exceeded for <ip> from <email>"}`)
	got, err := sanitizeSubprocessLine(in)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(got, want) {
		t.Fatalf("got %s, want %s", got, want)
	}
	composed := composeSanitizers(sanitizeSubprocessLine, func(line []byte) ([]byte, error) {
		return append(line, " provider"...), nil
	})
	got, err = composed(in)
	if err != nil {
		t.Fatal(err)
	}
	want = []byte(`{"error":"API rate limit exceeded for <ip> from <email>"} provider`)
	if !slices.Equal(got, want) {
		t.Fatalf("composed: got %s, want %s", got, want)
	}
}
