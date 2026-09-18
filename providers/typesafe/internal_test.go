// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed by the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Tests for the unexported helpers of the typesafe package.

package typesafe

import (
	"strings"
	"testing"
)

func TestContentSealed(t *testing.T) {
	// content() is a marker; it is only there to seal the Content interface to this package. Call it so
	// that these stay the only three implementations.
	Text("hi").content()
	Object(nil).content()
	Array(nil).content()
}

func TestContentFromJSON(t *testing.T) {
	data := []struct {
		name    string
		in      string
		want    Content
		wantErr string
	}{
		{"text", `"hi"`, Text("hi"), ""},
		{"object", `{"a":1}`, Object{"a": 1}, ""},
		{"array", `["a",1]`, Array{"a", 1}, ""},
		{"empty", ``, nil, "is empty"},
		{"number", `1`, nil, "expected a string, a JSON object or a JSON array, got 1"},
		{"bool", `true`, nil, "expected a string, a JSON object or a JSON array, got true"},
		{"malformed text", `"hi`, nil, "unexpected EOF"},
		{"malformed object", `{"a":}`, nil, "invalid character"},
		{"malformed array", `[1,`, nil, "unexpected EOF"},
	}
	for _, line := range data {
		t.Run(line.name, func(t *testing.T) {
			got, err := contentFromJSON([]byte(line.in))
			if line.wantErr == "" {
				if err != nil {
					t.Fatal(err)
				}
				if got == nil {
					t.Fatalf("expected %#v, got nil", line.want)
				}
				switch w := line.want.(type) {
				case Text:
					if v, ok := got.(Text); !ok || v != w {
						t.Fatalf("want %#v, got %#v", line.want, got)
					}
				case Object:
					if v, ok := got.(Object); !ok || len(v) != len(w) {
						t.Fatalf("want %#v, got %#v", line.want, got)
					}
				case Array:
					if v, ok := got.(Array); !ok || len(v) != len(w) {
						t.Fatalf("want %#v, got %#v", line.want, got)
					}
				}
				return
			}
			if err == nil {
				t.Fatalf("expected error, got %#v", got)
			}
			if !strings.Contains(err.Error(), line.wantErr) {
				t.Fatalf("want %q, got %q", line.wantErr, err)
			}
			if got != nil {
				t.Errorf("expected no content, got %#v", got)
			}
		})
	}
}
