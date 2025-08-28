///usr/bin/true; exec /usr/bin/env go run "$0" "$@"
// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

//go:build ignore

package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"

	"github.com/maruel/genai/scoreboard"
)

func regenScoreboard(jsonFile string) error {
	rawOld, err := os.ReadFile(jsonFile)
	if err != nil {
		return fmt.Errorf("failed to read %s: %w", jsonFile, err)
	}
	d := json.NewDecoder(bytes.NewReader(rawOld))
	d.DisallowUnknownFields()
	s := scoreboard.Score{}
	if err := d.Decode(&s); err != nil {
		return fmt.Errorf("failed to decode %s: %w", jsonFile, err)
	}
	rawNew, err := json.MarshalIndent(s, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to encode %s: %w", jsonFile, err)
	}
	rawNew = append(rawNew, '\n')
	if bytes.Equal(rawNew, rawOld) {
		return nil
	}
	fmt.Printf("- Updating %s\n", jsonFile)
	if err = os.WriteFile(jsonFile, rawNew, 0o644); err != nil {
		return fmt.Errorf("failed to write %s: %w", jsonFile, err)
	}
	return nil
}

func mainImpl() error {
	if len(os.Args) != 2 {
		return fmt.Errorf("usage: %s <root>", os.Args[0])
	}
	root := os.Args[1]
	var jsonFiles []string
	err := filepath.Walk(root, func(path string, info os.FileInfo, err2 error) error {
		if err2 != nil {
			return err2
		}
		if !info.IsDir() && filepath.Base(info.Name()) == "scoreboard.json" {
			jsonFiles = append(jsonFiles, path)
		}
		return nil
	})
	if err != nil {
		return err
	}
	for _, jsonFile := range jsonFiles {
		if err = regenScoreboard(jsonFile); err != nil {
			return err
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
