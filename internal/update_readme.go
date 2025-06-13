///usr/bin/true; exec /usr/bin/env go run "$0" "$@"
// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

//go:build ignore

package main

import (
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
)

func mainImpl() error {
	root := ".."

	c := exec.Command("go", "run", filepath.Join(root, "cmd", "scoreboard"), "-table")
	rawNew, err := c.Output()
	if err != nil {
		return fmt.Errorf("go run failed: %w: %s", err, string(rawNew))
	}
	rawReadme, err := os.ReadFile(filepath.Join(root, "README.md"))
	if err != nil {
		return err
	}
	readme := string(rawReadme)
	start := strings.Index(readme, "| Provider")
	if start < 0 {
		return errors.New("could not find table in README.md")
	}
	end := strings.Index(readme[start:], "\n\n")
	if end < 0 {
		return errors.New("could not find end of table in README.md")
	}
	n := readme[:start] + string(rawNew) + readme[start+end+1:]
	if n == readme {
		return nil
	}
	if err := os.WriteFile(filepath.Join(root, "README.md"), []byte(n), 0644); err != nil {
		return err
	}
	return nil
}

func main() {
	if err := mainImpl(); err != nil {
		fmt.Fprintf(os.Stderr, "Failed: %v\n", err)
		os.Exit(1)
	}
}
