// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package claudecode

import (
	"errors"
	"slices"
	"strings"
	"testing"

	"github.com/maruel/genai"
	"github.com/maruel/genai/base"
	"github.com/maruel/genai/internal"
)

func TestBuildArgs(t *testing.T) {
	t.Run("defaults", func(t *testing.T) {
		c := &Client{model: ""}
		args := c.buildArgs(callOpts{}, "", false)
		want := []string{
			"-p",
			"--verbose",
			"--input-format", "stream-json",
			"--output-format", "stream-json",
			"--strict-mcp-config",
			"--no-chrome",
			"--tools", "",
			"--disable-slash-commands",
			"--setting-sources", "project,local",
			"--no-session-persistence",
		}
		if !slices.Equal(args, want) {
			t.Errorf("got  %v\nwant %v", args, want)
		}
	})
	t.Run("with_tools", func(t *testing.T) {
		c := &Client{model: "sonnet"}
		co := callOpts{tools: []string{"Bash", "Read"}, permissionMode: "bypassPermissions"}
		args := c.buildArgs(co, "", false)
		check := func(flag, val string) {
			t.Helper()
			for i, a := range args {
				if a == flag && i+1 < len(args) && args[i+1] == val {
					return
				}
			}
			t.Errorf("flag %q %q not found in args %v", flag, val, args)
		}
		check("--tools", "Bash,Read")
		check("--permission-mode", "bypassPermissions")
		check("--model", "sonnet")
	})
	t.Run("with_session_resume", func(t *testing.T) {
		c := &Client{}
		args := c.buildArgs(callOpts{}, "my-session-id", false)
		for _, a := range args {
			if a == "--no-session-persistence" {
				t.Error("--no-session-persistence must not appear when resuming")
			}
		}
		check := func(flag, val string) {
			t.Helper()
			for i, a := range args {
				if a == flag && i+1 < len(args) && args[i+1] == val {
					return
				}
			}
			t.Errorf("flag %q %q not found in args %v", flag, val, args)
		}
		check("--resume", "my-session-id")
	})
	t.Run("streaming", func(t *testing.T) {
		c := &Client{}
		args := c.buildArgs(callOpts{}, "", true)
		if !slices.Contains(args, "--include-partial-messages") {
			t.Error("--include-partial-messages not found in streaming args")
		}
	})
	t.Run("with_system_prompt", func(t *testing.T) {
		c := &Client{}
		co := callOpts{systemPrompt: "Be helpful"}
		args := c.buildArgs(co, "", false)
		check := func(flag, val string) {
			t.Helper()
			for i, a := range args {
				if a == flag && i+1 < len(args) && args[i+1] == val {
					return
				}
			}
			t.Errorf("flag %q %q not found in args %v", flag, val, args)
		}
		check("--system-prompt", "Be helpful")
	})
}

func TestWriteUserMsg(t *testing.T) {
	t.Run("text_only", func(t *testing.T) {
		msg := genai.NewTextMessage("hello")
		var buf strings.Builder
		if err := writeUserMsg(&buf, &msg); err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if !strings.Contains(buf.String(), `"hello"`) {
			t.Errorf("expected plain string content, got %q", buf.String())
		}
	})
	t.Run("image_url", func(t *testing.T) {
		msg := genai.Message{
			Requests: []genai.Request{
				{Text: "describe this"},
				{Doc: genai.Doc{URL: "https://example.com/img.png"}},
			},
		}
		var buf strings.Builder
		if err := writeUserMsg(&buf, &msg); err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		out := buf.String()
		if !strings.Contains(out, `"type":"image"`) {
			t.Errorf("expected image block, got %q", out)
		}
		if !strings.Contains(out, `"url":"https://example.com/img.png"`) {
			t.Errorf("expected url source, got %q", out)
		}
	})
	t.Run("empty_message", func(t *testing.T) {
		msg := genai.NewTextMessage("")
		var buf strings.Builder
		if err := writeUserMsg(&buf, &msg); err == nil {
			t.Fatal("expected error for empty message")
		}
	})
}

func TestGenOption(t *testing.T) {
	t.Run("valid", func(t *testing.T) {
		for _, m := range []string{"acceptEdits", "bypassPermissions", "default", "dontAsk", "plan"} {
			if err := (&GenOption{PermissionMode: m}).Validate(); err != nil {
				t.Errorf("mode %q: unexpected error: %v", m, err)
			}
		}
	})
	t.Run("errors", func(t *testing.T) {
		if err := (&GenOption{Tools: []string{""}}).Validate(); err == nil {
			t.Error("expected error for empty tool name")
		}
		if err := (&GenOption{MaxBudgetUSD: -1}).Validate(); err == nil {
			t.Error("expected error for negative budget")
		}
		if err := (&GenOption{PermissionMode: "hack"}).Validate(); err == nil {
			t.Error("expected error for invalid mode")
		}
	})
	t.Run("system_prompt", func(t *testing.T) {
		co, err := parseOpts([]genai.GenOption{&genai.GenOptionText{SystemPrompt: "Be helpful"}})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if co.systemPrompt != "Be helpful" {
			t.Errorf("systemPrompt: got %q, want %q", co.systemPrompt, "Be helpful")
		}
	})
	t.Run("web_search", func(t *testing.T) {
		co, err := parseOpts([]genai.GenOption{&genai.GenOptionWeb{Search: true}})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		want := []string{"WebSearch", "WebFetch"}
		if !slices.Equal(co.tools, want) {
			t.Errorf("tools: got %v, want %v", co.tools, want)
		}
	})
	t.Run("web_fetch", func(t *testing.T) {
		co, err := parseOpts([]genai.GenOption{&genai.GenOptionWeb{Fetch: true}})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		want := []string{"WebFetch"}
		if !slices.Equal(co.tools, want) {
			t.Errorf("tools: got %v, want %v", co.tools, want)
		}
	})
	t.Run("web_with_tools", func(t *testing.T) {
		co, err := parseOpts([]genai.GenOption{
			&GenOption{Tools: []string{"Bash"}},
			&genai.GenOptionWeb{Search: true, Fetch: true},
		})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		want := []string{"Bash", "WebSearch", "WebFetch"}
		if !slices.Equal(co.tools, want) {
			t.Errorf("tools: got %v, want %v", co.tools, want)
		}
	})
	t.Run("unsupported", func(t *testing.T) {
		for _, tc := range []struct {
			name string
			opts []genai.GenOption
			want string
		}{
			{"Temperature", []genai.GenOption{&genai.GenOptionText{Temperature: 0.5}}, "GenOptionText.Temperature"},
			{"Seed", []genai.GenOption{genai.GenOptionSeed(42)}, "GenOptionSeed"},
		} {
			t.Run(tc.name, func(t *testing.T) {
				_, err := parseOpts(tc.opts)
				var uerr *base.ErrNotSupported
				if !errors.As(err, &uerr) {
					t.Fatalf("expected ErrNotSupported, got %v", err)
				}
				if !slices.Contains(uerr.Options, tc.want) {
					t.Errorf("expected %q in unsupported, got %v", tc.want, uerr.Options)
				}
			})
		}
	})
}

func init() {
	internal.BeLenient = false
}
