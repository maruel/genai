// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package main

import "testing"

func TestPrintList(t *testing.T) {
	t.Parallel()
	printList()
}

func TestPrintTable(t *testing.T) {
	t.Parallel()
	printTable("")
	printTable("openaicompatible")
}
