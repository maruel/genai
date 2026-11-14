// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Tests for the ISO-639-1 input language mapping.

package realtime_test

import (
	"errors"
	"testing"

	"github.com/maruel/genai/base"
	"github.com/maruel/genai/internal/realtime"
)

func TestPrimaryLanguage(t *testing.T) {
	for _, tc := range []struct {
		in   string
		want string
	}{
		{in: "", want: ""},
		{in: "FR", want: "fr"},
		{in: "zh-Hant-TW", want: "zh"},
	} {
		got, err := realtime.PrimaryLanguage("test", tc.in)
		if err != nil || got != tc.want {
			t.Errorf("PrimaryLanguage(%q) = %q, %v; want %q", tc.in, got, err, tc.want)
		}
	}
	if _, err := realtime.PrimaryLanguage("test", "fil"); !errors.As(err, new(*base.ErrNotSupported)) {
		t.Errorf("language without an ISO-639-1 code: %v", err)
	}
}
