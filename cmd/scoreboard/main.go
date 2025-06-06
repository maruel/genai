// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Command scoreboard generates a scoreboard for every providers supported.
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"os"

	"github.com/maruel/genai"
	"github.com/maruel/genai/anthropic"
	"github.com/maruel/genai/bfl"
	"github.com/maruel/genai/cerebras"
	"github.com/maruel/genai/cloudflare"
	"github.com/maruel/genai/cohere"
	"github.com/maruel/genai/deepseek"
	"github.com/maruel/genai/gemini"
	"github.com/maruel/genai/groq"
	"github.com/maruel/genai/huggingface"
	"github.com/maruel/genai/mistral"
	"github.com/maruel/genai/openai"
	"github.com/maruel/genai/perplexity"
	"github.com/maruel/genai/togetherai"
)

// providers is the list of known providers. We only look at their scoreboard, so no need for an API key or
// model name.
var providers = map[string]func() (genai.Provider, error){
	"anthropic": func() (genai.Provider, error) {
		return anthropic.New("FAKE_API_KEY", "FAKE_MODEL", nil)
	},
	"bfl": func() (genai.Provider, error) {
		return bfl.New("FAKE_API_KEY", "FAKE_MODEL", nil)
	},
	"cerebras": func() (genai.Provider, error) {
		return cerebras.New("FAKE_API_KEY", "FAKE_MODEL", nil)
	},
	"cloudflare": func() (genai.Provider, error) {
		return cloudflare.New("FAKE_API_KEY", "", "FAKE_MODEL", nil)
	},
	"cohere": func() (genai.Provider, error) {
		return cohere.New("FAKE_API_KEY", "FAKE_MODEL", nil)
	},
	"deepseek": func() (genai.Provider, error) {
		return deepseek.New("FAKE_API_KEY", "FAKE_MODEL", nil)
	},
	"gemini": func() (genai.Provider, error) {
		return gemini.New("FAKE_API_KEY", "FAKE_MODEL", nil)
	},
	"groq": func() (genai.Provider, error) {
		return groq.New("FAKE_API_KEY", "FAKE_MODEL", nil)
	},
	"huggingface": func() (genai.Provider, error) {
		return huggingface.New("FAKE_API_KEY", "FAKE_MODEL", nil)
	},
	"mistral": func() (genai.Provider, error) {
		return mistral.New("FAKE_API_KEY", "FAKE_MODEL", nil)
	},
	"openai": func() (genai.Provider, error) {
		return openai.New("FAKE_API_KEY", "FAKE_MODEL", nil)
	},
	"perplexity": func() (genai.Provider, error) {
		return perplexity.New("FAKE_API_KEY", "FAKE_MODEL", nil)
	},
	"togetherai": func() (genai.Provider, error) {
		return togetherai.New("FAKE_API_KEY", "FAKE_MODEL", nil)
	},
}

func mainImpl() error {
	table := flag.Bool("table", false, "output a markdown table")
	flag.Parse()
	if flag.NArg() != 0 {
		return errors.New("unexpected arguments")
	}
	if *table {
		return printTable()
	}
	return printList()
}

func main() {
	if err := mainImpl(); err != nil {
		if err != context.Canceled {
			fmt.Fprintf(os.Stderr, "scoreboard: %s\n", err)
		}
		os.Exit(1)
	}
}
