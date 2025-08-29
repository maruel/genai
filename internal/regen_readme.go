///usr/bin/true; exec /usr/bin/env go run "$0" "$@"
// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

//go:build ignore

package main

import (
	"bytes"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
)

func mainImpl() error {
	if len(os.Args) != 2 {
		return fmt.Errorf("usage: %s <root>", os.Args[0])
	}
	root := os.Args[1]
	c := exec.Command("go", "run", filepath.Join(root, "cmd", "scoreboard"), "-table")
	rawNewTable, err := c.CombinedOutput()
	if err != nil {
		return fmt.Errorf("go run failed: %w: %s", err, string(rawNewTable))
	}
	p := filepath.Join(root, "README.md")
	rawReadmeOld, err := os.ReadFile(p)
	if err != nil {
		return fmt.Errorf("failed to read %s: %w", p, err)
	}
	readme := string(rawReadmeOld)
	start := strings.Index(readme, "| Provider")
	if start < 0 {
		return errors.New("could not find table in README.md")
	}
	end := strings.Index(readme[start:], "\n\n")
	if end < 0 {
		return errors.New("could not find end of table in README.md")
	}
	n := readme[:start] + string(rawNewTable) + readme[start+end+1:]
	if n == readme {
		return nil
	}
	rawReadmeNew := []byte(n)
	if bytes.Equal(rawReadmeNew, rawReadmeOld) {
		return nil
	}
	fmt.Printf("- Updating %s\n", p)
	if err := os.WriteFile(p, rawReadmeNew, 0o644); err != nil {
		return fmt.Errorf("failed to write %s: %w", p, err)
	}
	return nil
}

func main() {
	if err := mainImpl(); err != nil {
		fmt.Fprintf(os.Stderr, "Failed: %v\n", err)
		os.Exit(1)
	}
}
