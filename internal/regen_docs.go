///usr/bin/true; exec /usr/bin/env go run "$0" "$@"
// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

//go:build ignore

package main

import (
	"bytes"
	"fmt"
	"maps"
	"os"
	"os/exec"
	"path/filepath"
	"slices"

	"github.com/maruel/genai/providers"
)

func run(root, name string) error {
	dst := filepath.Join(root, "docs", name+".md")
	c := exec.Command("go", "run", filepath.Join(root, "cmd", "scoreboard"), "-table", "-provider", name)
	rawNew, err := c.Output()
	if err != nil {
		return fmt.Errorf("go run failed: %w: %s", err, string(rawNew))
	}
	rawNew = append([]byte("# Scoreboard\n\n"), rawNew...)
	rawOld, _ := os.ReadFile(dst)
	if bytes.Equal(rawNew, rawOld) {
		return nil
	}
	fmt.Printf("- Updating %s\n", dst)
	return os.WriteFile(dst, rawNew, 0o644)
}

func mainImpl() error {
	if len(os.Args) != 2 {
		return fmt.Errorf("usage: %s <root>", os.Args[0])
	}
	root := os.Args[1]
	for _, name := range slices.Sorted(maps.Keys(providers.All)) {
		if name == "openaicompatible" {
			continue
		}
		if err := run(root, name); err != nil {
			return fmt.Errorf("failed processing %s: %w", name, err)
		}
	}
	return nil
}

func main() {
	if err := mainImpl(); err != nil {
		fmt.Fprintf(os.Stderr, "Failed: %v\n", err)
		os.Exit(1)
	}
}
