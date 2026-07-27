// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Tests the weekly regeneration pull request workflow.
package main

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

const (
	draftEnv  = "GENAI_AUTOFIX_WEEKLY_REGEN_DRAFT"
	helperEnv = "GENAI_AUTOFIX_WEEKLY_REGEN_HELPER"
	logEnv    = "GENAI_AUTOFIX_WEEKLY_REGEN_COMMAND_LOG"
)

func TestMain(m *testing.M) {
	if os.Getenv(helperEnv) != "" {
		runHelper()
		return
	}
	os.Exit(m.Run())
}

func TestPushPR(t *testing.T) {
	for _, tc := range []struct {
		name  string
		draft string
		want  [][]string
	}{
		{
			name: "create ready for review",
			want: [][]string{
				{"push"},
				{"pr", "view", "--json", "isDraft", "--jq", ".isDraft"},
				{"pr", "create", "--title", title, "--body", prBody},
			},
		},
		{
			name:  "promote existing draft",
			draft: "true",
			want: [][]string{
				{"push"},
				{"pr", "view", "--json", "isDraft", "--jq", ".isDraft"},
				{"pr", "ready"},
				{"pr", "edit", "--title", title, "--body", prBody},
			},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			bin := t.TempDir()
			log := filepath.Join(t.TempDir(), "commands.jsonl")
			exe, err := os.Executable()
			if err != nil {
				t.Fatal(err)
			}
			for _, name := range []string{"git", "gh"} {
				ext := ""
				if runtime.GOOS == "windows" {
					ext = ".exe"
				}
				data, err := os.ReadFile(exe)
				if err != nil {
					t.Fatal(err)
				}
				if err := os.WriteFile(filepath.Join(bin, name+ext), data, 0o755); err != nil {
					t.Fatal(err)
				}
			}
			t.Setenv(draftEnv, tc.draft)
			t.Setenv(helperEnv, "1")
			t.Setenv(logEnv, log)
			t.Setenv("PATH", bin+string(os.PathListSeparator)+os.Getenv("PATH"))

			if err := pushPR(t.Context()); err != nil {
				t.Fatal(err)
			}

			data, err := os.ReadFile(log)
			if err != nil {
				t.Fatal(err)
			}
			var got [][]string
			for line := range strings.SplitSeq(strings.TrimSpace(string(data)), "\n") {
				var args []string
				if err := json.Unmarshal([]byte(line), &args); err != nil {
					t.Fatal(err)
				}
				got = append(got, args)
			}
			if diff := diffArgs(tc.want, got); diff != "" {
				t.Fatalf("pushPR() commands differ (-want +got):\n%s", diff)
			}
		})
	}
}

func runHelper() {
	data, err := json.Marshal(os.Args[1:])
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	f, err := os.OpenFile(os.Getenv(logEnv), os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o600)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	if _, err := fmt.Fprintln(f, string(data)); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	if err := f.Close(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	if len(os.Args) >= 3 && os.Args[1] == "pr" && os.Args[2] == "view" {
		if os.Getenv(draftEnv) == "true" {
			fmt.Println("true")
			return
		}
		os.Exit(1)
	}
}

func diffArgs(want, got [][]string) string {
	if len(want) != len(got) {
		return fmt.Sprintf("command count: want %d, got %d", len(want), len(got))
	}
	for i := range want {
		if len(want[i]) != len(got[i]) {
			return fmt.Sprintf("command %d argument count: want %d, got %d", i, len(want[i]), len(got[i]))
		}
		for j := range want[i] {
			if want[i][j] != got[i][j] {
				return fmt.Sprintf("command %d argument %d: want %q, got %q", i, j, want[i][j], got[i][j])
			}
		}
	}
	return ""
}
