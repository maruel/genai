///usr/bin/true; exec /usr/bin/env go run "$0" "$@"
// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

//go:build ignore

package main

import (
	"fmt"
	"os"
	"path/filepath"

	"github.com/maruel/genai/base"
)

func mainImpl() error {
	if len(os.Args) != 2 {
		return fmt.Errorf("usage: %s filename", os.Args[0])
	}
	fmt.Println(base.MimeByExt(filepath.Ext(os.Args[1])))
	return nil
}

func main() {
	if err := mainImpl(); err != nil {
		fmt.Fprintf(os.Stderr, "Failed: %v\n", err)
		os.Exit(1)
	}
}
