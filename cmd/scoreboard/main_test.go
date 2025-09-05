// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package main

import (
	"testing"

	"github.com/maruel/genai/internal/internaltest"
)

func TestPrintList(t *testing.T) {
	t.Parallel()
	printList(t.Context(), &internaltest.WriterToLog{T: t})
}

func TestPrintTable(t *testing.T) {
	t.Parallel()
	ctx := t.Context()
	printTable(ctx, &internaltest.WriterToLog{T: t}, "")
	printTable(ctx, &internaltest.WriterToLog{T: t}, "openaicompatible")
}
