// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Command update-servers checks for newer releases of llama.cpp and ollama
// servers used by llamacppsrv and ollamasrv, updates the version constants,
// and regenerates test recordings.
//
// Usage:
//
//	go run ./internal/cmd/update-servers           # dry-run
//	go run ./internal/cmd/update-servers --apply    # update + re-record
//	go run ./internal/cmd/update-servers --apply llamacpp ollama
//
// Set GITHUB_TOKEN to avoid GitHub API rate limiting.
package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"

	"github.com/maruel/genai/internal/ghrelease"
)

type update struct {
	name    string                       // human-readable
	file    string                       // path to Go source file
	ghOwner string                       // GitHub owner
	ghRepo  string                       // GitHub repo
	parse   func(string) (string, error) // tag -> version string
	current func(string) (string, error) // file content -> current version
	apply   func(orig, newVersion string) (string, error)
	testPkg string // go test package pattern
}

const (
	colorRed    = "\033[0;31m"
	colorGreen  = "\033[0;32m"
	colorYellow = "\033[1;33m"
	colorReset  = "\033[0m"
)

var updates = []update{
	{
		name:    "llamacpp",
		file:    filepath.Join("providers", "llamacpp", "llamacppsrv", "llamacppsrv.go"),
		ghOwner: "ggml-org",
		ghRepo:  "llama.cpp",
		parse: func(tag string) (string, error) {
			// Tags are like "b9020".
			if !strings.HasPrefix(tag, "b") {
				return "", fmt.Errorf("unexpected llama.cpp tag %q", tag)
			}
			return tag[1:], nil
		},
		current: func(content string) (string, error) {
			re := regexp.MustCompile(`const BuildNumber\s*=\s*(\d+)`)
			m := re.FindStringSubmatch(content)
			if len(m) < 2 {
				return "", fmt.Errorf("BuildNumber not found")
			}
			return m[1], nil
		},
		apply: func(orig, newVersion string) (string, error) {
			re := regexp.MustCompile(`(const BuildNumber\s*=\s*)\d+`)
			result := re.ReplaceAllString(orig, `${1}`+newVersion)
			if result == orig {
				return "", fmt.Errorf("BuildNumber replacement had no effect")
			}
			return result, nil
		},
		testPkg: "./providers/llamacpp/...",
	},
	{
		name:    "ollama",
		file:    filepath.Join("providers", "ollama", "ollamasrv", "ollamasrv.go"),
		ghOwner: "ollama",
		ghRepo:  "ollama",
		parse: func(tag string) (string, error) {
			if !strings.HasPrefix(tag, "v") {
				return "", fmt.Errorf("unexpected ollama tag %q", tag)
			}
			return tag, nil
		},
		current: func(content string) (string, error) {
			re := regexp.MustCompile(`const Version\s*=\s*"([^"]+)"`)
			m := re.FindStringSubmatch(content)
			if len(m) < 2 {
				return "", fmt.Errorf("Version not found")
			}
			return m[1], nil
		},
		apply: func(orig, newVersion string) (string, error) {
			re := regexp.MustCompile(`(const Version\s*=\s*)"[^"]*"`)
			result := re.ReplaceAllString(orig, `${1}"`+newVersion+`"`)
			if result == orig {
				return "", fmt.Errorf("Version replacement had no effect")
			}
			return result, nil
		},
		testPkg: "./providers/ollama/...",
	},
}

func main() {
	apply := flag.Bool("apply", false, "apply updates and re-record tests")
	flag.Parse()

	targets := flag.Args()
	if len(targets) == 0 {
		for _, u := range updates {
			targets = append(targets, u.name)
		}
	}

	root, err := findRepoRoot()
	if err != nil {
		log.Fatal(err)
	}
	ctx := context.Background()

	outdated := false
	needRecord := map[string]struct{}{}

	for _, target := range targets {
		u, err := findUpdate(target)
		if err != nil {
			log.Fatal(err)
		}
		cur, latest, err := check(ctx, root, u)
		if err != nil {
			log.Fatal(err)
		}
		if cur == latest {
			fmt.Printf("%s %sup-to-date%s (%s)\n", u.name, colorGreen, colorReset, formatVersion(u.name, cur))
			continue
		}
		outdated = true
		fmt.Printf("%s %s%s -> %s%s\n", u.name, colorYellow, formatVersion(u.name, cur), formatVersion(u.name, latest), colorReset)
		if !*apply {
			continue
		}
		if err := applyUpdate(root, u, cur, latest); err != nil {
			log.Fatal(err)
		}
		fmt.Printf("%s %supdated to %s%s\n", u.name, colorGreen, formatVersion(u.name, latest), colorReset)
		needRecord[u.name] = struct{}{}
	}

	if !*apply && outdated {
		fmt.Println("\nRun with --apply to apply the updates and re-record tests.")
		return
	}

	for _, u := range updates {
		if _, ok := needRecord[u.name]; !ok {
			continue
		}
		fmt.Printf("%sRe-recording %s tests...%s\n", colorYellow, u.name, colorReset)
		if err := runTests(root, u.testPkg); err != nil {
			log.Fatal(err)
		}
		fmt.Printf("%s%s recordings updated%s\n", colorGreen, u.name, colorReset)
	}
}

func findRepoRoot() (string, error) {
	dir, err := os.Getwd()
	if err != nil {
		return "", err
	}
	for {
		if _, err := os.Stat(filepath.Join(dir, "go.mod")); err == nil {
			return dir, nil
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			return "", fmt.Errorf("not inside a Go module")
		}
		dir = parent
	}
}

func findUpdate(name string) (update, error) {
	for _, u := range updates {
		if u.name == name {
			return u, nil
		}
	}
	return update{}, fmt.Errorf("unknown target %q; valid: llamacpp, ollama", name)
}

func check(ctx context.Context, root string, u update) (cur, latest string, err error) {
	content, err := os.ReadFile(filepath.Join(root, u.file))
	if err != nil {
		return "", "", err
	}
	cur, err = u.current(string(content))
	if err != nil {
		return "", "", err
	}
	rel, err := ghrelease.GetLatestRelease(ctx, u.ghOwner, u.ghRepo)
	if err != nil {
		return "", "", fmt.Errorf("fetching latest %s/%s release: %w", u.ghOwner, u.ghRepo, err)
	}
	latest, err = u.parse(rel.TagName)
	if err != nil {
		return "", "", err
	}
	return cur, latest, nil
}

func applyUpdate(root string, u update, cur, latest string) error {
	path := filepath.Join(root, u.file)
	orig, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	newContent, err := u.apply(string(orig), latest)
	if err != nil {
		return fmt.Errorf("updating %s: %w", path, err)
	}
	// Verify the new content parses.
	cur2, err := u.current(newContent)
	if err != nil {
		return fmt.Errorf("verifying update of %s: %w", path, err)
	}
	if cur2 != latest {
		return fmt.Errorf("verification failed: expected %q, got %q in updated %s", latest, cur2, path)
	}
	if err := os.WriteFile(path, []byte(newContent), 0o644); err != nil {
		return err
	}
	return nil
}

func runTests(root, pkg string) error {
	cmd := exec.Command("go", "test", pkg, "-update-scoreboard")
	cmd.Dir = root
	cmd.Env = append(os.Environ(), "RECORD=all")
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	return cmd.Run()
}

func formatVersion(name, v string) string {
	switch name {
	case "llamacpp":
		n, _ := strconv.Atoi(v)
		return fmt.Sprintf("b%d", n)
	default:
		return v
	}
}
