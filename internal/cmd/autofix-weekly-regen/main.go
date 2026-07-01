// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Command autofix-weekly-regen asks pi.dev to repair weekly model regeneration failures.
package main

import (
	"bytes"
	"context"
	"errors"
	"flag"
	"fmt"
	"log"
	"os"
	"os/exec"
	"os/signal"
	"strings"
	"syscall"
)

const (
	title  = "ci: auto-fix weekly model regeneration"
	prBody = "Generated automatically after weekly model regeneration failed.\n"
	prompt = `The weekly model regeneration workflow failed for this repository.

Fix the repository so this command can complete:

    ./internal/regen_weekly.sh

Constraints:
- Follow AGENTS.md.
- Keep the fix scoped to the weekly regeneration failure.
- Prefer deterministic schema, DTO, test, fixture, or regeneration fixes over broad refactors.
- If the failure is an unknown provider JSON field, update the typed DTO and tests.
- Run the failing command or the smallest relevant provider test after changes.
- Run gofmt when changing Go files.
- Do not commit, push, create a PR, inspect secrets, or print secret environment variables.`
)

var forwardedEnvNames = [...]string{
	"ANTHROPIC_API_KEY",
	"BASETEN_API_KEY",
	"BFL_API_KEY",
	"CEREBRAS_API_KEY",
	"CLOUDFLARE_ACCOUNT_ID",
	"CLOUDFLARE_API_KEY",
	"COHERE_API_KEY",
	"DASHSCOPE_API_KEY",
	"DASHSCOPE_API_KEY_CN",
	"DASHSCOPE_API_KEY_INTL",
	"DASHSCOPE_API_KEY_US",
	"DEEPSEEK_API_KEY",
	"GEMINI_API_KEY",
	"GROK_API_KEY",
	"GROQ_API_KEY",
	"HUGGINGFACE_API_KEY",
	"MIMO_API_KEY",
	"MISTRAL_API_KEY",
	"OPENAI_API_KEY",
	"OPENROUTER_API_KEY",
	"PERPLEXITY_API_KEY",
	"POLLINATIONS_API_KEY",
	"TOGETHER_API_KEY",
}

func main() {
	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM, os.Interrupt)
	defer stop()
	if err := mainImpl(ctx); err != nil {
		log.Fatal(err)
	}
}

func mainImpl(ctx context.Context) error {
	push := flag.Bool("push", false, "push the current branch and open or update a draft PR")
	provider := flag.String("provider", "deepseek", "pi provider")
	model := flag.String("model", "deepseek-v4-flash", "pi model")
	thinking := flag.String("thinking", "high", "pi thinking level")
	flag.Parse()
	if flag.NArg() != 0 {
		return errors.New("unexpected arguments")
	}

	if _, err := exec.LookPath("md"); err != nil {
		return fmt.Errorf("md CLI not found on PATH: %w", err)
	}
	if *push {
		if _, err := exec.LookPath("gh"); err != nil {
			return fmt.Errorf("gh CLI not found on PATH: %w", err)
		}
	}
	if err := runPiInMD(ctx, *provider, *model, *thinking); err != nil {
		return err
	}
	if !*push {
		return nil
	}
	return pushPR(ctx)
}

func runPiInMD(ctx context.Context, provider, model, thinking string) error {
	args := []string{"run", "-apply-patch"}
	args = appendEnvFlags(args, forwardedEnvNames[:]...)
	args = append(args,
		"--env", "PI_TELEMETRY=0",
		"pi", "--approve", "--no-session",
		"--provider", provider,
		"--model", model,
		"--thinking", thinking,
		"-p", prompt,
	)
	if err := run(ctx, "md", args...); err != nil {
		return err
	}
	return nil
}

func appendEnvFlags(args []string, names ...string) []string {
	for _, name := range names {
		if _, ok := os.LookupEnv(name); ok {
			args = append(args, "--env", name)
		}
	}
	return args
}

func pushPR(ctx context.Context) error {
	if err := run(ctx, "git", "push"); err != nil {
		return err
	}
	if _, err := output(ctx, "gh", "pr", "view", "--json", "number", "--jq", ".number"); err == nil {
		return run(ctx, "gh", "pr", "edit", "--title", title, "--body", prBody)
	} else if createErr := run(ctx, "gh", "pr", "create", "--draft", "--title", title, "--body", prBody); createErr != nil {
		return errors.Join(fmt.Errorf("checking current branch PR: %w", err), createErr)
	}
	return nil
}

func run(ctx context.Context, name string, args ...string) error {
	cmd := exec.CommandContext(ctx, name, args...)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	cmd.Stdin = os.Stdin
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("%s %s: %w", name, strings.Join(args, " "), err)
	}
	return nil
}

func output(ctx context.Context, name string, args ...string) (string, error) {
	cmd := exec.CommandContext(ctx, name, args...)
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		msg := strings.TrimSpace(stderr.String())
		if msg != "" {
			return "", fmt.Errorf("%s %s: %w: %s", name, strings.Join(args, " "), err, msg)
		}
		return "", fmt.Errorf("%s %s: %w", name, strings.Join(args, " "), err)
	}
	return stdout.String(), nil
}
