// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Authenticate with Codex's saved ChatGPT subscription and call GPT-6 Luna
// directly over HTTP, without launching Codex. Run `codex login` first with
// cli_auth_credentials_store = "file" in ~/.codex/config.toml. CODEX_HOME selects
// a different credential directory. Access tokens are renewed automatically.
package main

import (
	"context"
	"errors"
	"fmt"
	"log"
	"os"
	"os/signal"
	"syscall"

	"github.com/maruel/genai"
	"github.com/maruel/genai/providers/openairesponses"
)

func mainImpl() (err error) {
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	c, err := openairesponses.New(ctx, &openairesponses.ProviderOption{Auth: openairesponses.AuthCodex, AppName: "genai-example"}, genai.ProviderOptionModel("gpt-6-luna"))
	if c != nil {
		defer func() { err = errors.Join(err, c.Close()) }()
	}
	if err != nil {
		return err
	}
	chunks, finish := c.GenStream(ctx, genai.Messages{genai.NewTextMessage("Give me life advice that sounds good but is a bad idea in practice. Answer succinctly.")}, &genai.GenOptionText{SystemPrompt: "You are a helpful assistant."}, genai.EffortLow)
	for chunk := range chunks {
		fmt.Print(chunk.Text)
	}
	fmt.Println()
	_, err = finish()
	return err
}

func main() {
	if err := mainImpl(); err != nil {
		log.Fatal(err)
	}
}
