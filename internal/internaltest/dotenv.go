// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// GetEnv and its .env fallback for provider credentials in tests.

package internaltest

import (
	"bufio"
	"os"
	"path/filepath"
	"strings"
	"sync"
)

// GetEnv returns the environment variable named by key.
//
// If unset, it falls back to a KEY=VALUE entry in a .env file at the git
// repository root, so provider API keys can be kept in one untracked file
// instead of being exported in every shell.
func GetEnv(key string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return dotEnv()[key]
}

var (
	dotEnvOnce sync.Once
	dotEnvVars map[string]string
)

func dotEnv() map[string]string {
	dotEnvOnce.Do(func() {
		dotEnvVars = map[string]string{}
		root, err := os.Getwd()
		if err != nil {
			return
		}
		for {
			if _, err := os.Stat(filepath.Join(root, ".git")); err == nil {
				break
			}
			parent := filepath.Dir(root)
			if parent == root {
				return
			}
			root = parent
		}
		f, err := os.Open(filepath.Join(root, ".env"))
		if err != nil {
			return
		}
		defer func() { _ = f.Close() }()
		s := bufio.NewScanner(f)
		for s.Scan() {
			line := strings.TrimSpace(s.Text())
			if line == "" || strings.HasPrefix(line, "#") {
				continue
			}
			line = strings.TrimPrefix(line, "export ")
			k, v, ok := strings.Cut(line, "=")
			if !ok {
				continue
			}
			k = strings.TrimSpace(k)
			v = strings.Trim(strings.TrimSpace(v), `"'`)
			dotEnvVars[k] = v
		}
	})
	return dotEnvVars
}
