// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Input language mapping for providers that take an ISO-639-1 code.

package realtime

import (
	"fmt"

	"golang.org/x/text/language"

	"github.com/maruel/genai/base"
)

// PrimaryLanguage returns the ISO-639-1 primary language of a BCP 47 tag that
// genai.RealtimeOptionAudio.Validate accepted, or "" for an empty tag. A
// language without a two-letter code is base.ErrNotSupported; provider names
// the caller in that error.
func PrimaryLanguage(provider, tag string) (string, error) {
	if tag == "" {
		return "", nil
	}
	t, err := language.Parse(tag)
	if err != nil {
		return "", err
	}
	b, _ := t.Base()
	if s := b.String(); len(s) == 2 {
		return s, nil
	}
	return "", fmt.Errorf("%s realtime input language %q has no ISO-639-1 code: %w", provider, tag, &base.ErrNotSupported{Options: []string{"input language"}})
}
