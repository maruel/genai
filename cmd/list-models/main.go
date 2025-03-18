// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"os"
	"os/signal"
	"sort"
	"strings"
	"syscall"

	"github.com/maruel/genai/anthropic"
	"github.com/maruel/genai/cerebras"
	"github.com/maruel/genai/cloudflare"
	"github.com/maruel/genai/cohere"
	"github.com/maruel/genai/deepseek"
	"github.com/maruel/genai/gemini"
	"github.com/maruel/genai/genaiapi"
	"github.com/maruel/genai/groq"
	"github.com/maruel/genai/huggingface"
	"github.com/maruel/genai/mistral"
	"github.com/maruel/genai/openai"
	"github.com/maruel/genai/togetherai"
)

var providers = map[string]func() (genaiapi.ModelProvider, error){
	"anthropic": func() (genaiapi.ModelProvider, error) {
		return anthropic.New("", "")
	},
	"cerebras": func() (genaiapi.ModelProvider, error) {
		return cerebras.New("", "")
	},
	"cloudflare": func() (genaiapi.ModelProvider, error) {
		return cloudflare.New("", "", "")
	},
	"cohere": func() (genaiapi.ModelProvider, error) {
		return cohere.New("", "")
	},
	"deepseek": func() (genaiapi.ModelProvider, error) {
		return deepseek.New("", "")
	},
	"gemini": func() (genaiapi.ModelProvider, error) {
		return gemini.New("", "")
	},
	"groq": func() (genaiapi.ModelProvider, error) {
		return groq.New("", "")
	},
	"huggingface": func() (genaiapi.ModelProvider, error) {
		return huggingface.New("", "")
	},
	"mistral": func() (genaiapi.ModelProvider, error) {
		return mistral.New("", "")
	},
	"openai": func() (genaiapi.ModelProvider, error) {
		return openai.New("", "")
	},
	"togetherai": func() (genaiapi.ModelProvider, error) {
		return togetherai.New("", "")
	},
}

func mainImpl() error {
	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM, os.Interrupt)
	defer stop()

	names := make([]string, 0, len(providers))
	for name := range providers {
		names = append(names, name)
	}
	sort.Strings(names)
	provider := flag.String("provider", "", "backend to use: "+strings.Join(names, ", "))
	flag.Parse()
	if flag.NArg() != 0 {
		return errors.New("unexpected arguments")
	}
	fn := providers[*provider]
	if fn == nil {
		return fmt.Errorf("unknown backend %q", *provider)
	}
	b, err := fn()
	if err != nil {
		return err
	}
	models, err := b.ListModels(ctx)
	if err != nil {
		return err
	}
	s := make([]string, 0, len(models))
	for _, m := range models {
		if t, ok := m.(*huggingface.Model); ok {
			if t.TrendingScore < 1 {
				continue
			}
		}
		s = append(s, m.String())
	}
	sort.Slice(s, func(i, j int) bool {
		return strings.ToLower(s[i]) < strings.ToLower(s[j])
	})
	for _, m := range s {
		fmt.Printf("%s\n", m)
	}
	return nil
}

func main() {
	if err := mainImpl(); err != nil {
		if err != context.Canceled {
			fmt.Fprintf(os.Stderr, "list-models: %s\n", err)
		}
		os.Exit(1)
	}
}
